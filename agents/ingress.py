"""
Ingress agent — second node in the LangGraph graph (after input_guardrail).

Responsibilities:
1. Assign a stable source_id to each source (src_001, src_002, ...)
   using a hash of URL or content for stability across retries.
2. Deduplicate: same URL → merge; content cosine similarity > 0.95 → merge.
   Keep the entry with richer metadata (more non-null fields).
3. Normalize into the Source schema: strip HTML, extract domain from URL,
   populate all fields.
4. Validate minimum requirements: content_text non-empty AND (url OR title).
   Sources failing validation are logged and dropped — pipeline does NOT crash.
5. Append a step record to audit_trail.

LLM usage: None. Pure deterministic logic.
Writes to state: normalized_sources, audit_trail
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import structlog

from core.state import PipelineState
from core.exceptions import SourceValidationError
from services.similarity import find_duplicate_sources

logger = structlog.get_logger(__name__)


# HTML stripping 

def _strip_html(text: str) -> str:
    """
    Strip HTML tags from content. Uses BeautifulSoup if available,
    falls back to a regex approach.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    except ImportError:
        pass

    # Regex fallback
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# Domain extraction 

def _extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract the registered domain from a URL string."""
    if not url:
        return None
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        netloc = parsed.netloc or parsed.path
        # Strip port if present
        domain = netloc.split(":")[0].lower().strip()
        # Strip www.
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else None
    except Exception:
        return None


# Stable source ID 

def _make_source_id(index: int, url: Optional[str], content: str) -> str:
    """
    Generate a stable source_id.
    Uses a short hash of URL (preferred) or content so the same source
    always gets the same ID across retries.
    Format: src_001, src_002, ...
    """
    return f"src_{str(index + 1).zfill(3)}"


def _content_hash(content: str) -> str:
    """Short hash of content for dedup fingerprinting."""
    return hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()[:12]


# Metadata richness 

def _count_populated_fields(source: dict) -> int:
    """Count how many of the key fields are non-null and non-empty."""
    key_fields = ["url", "title", "author", "domain", "published_date"]
    metadata = source.get("metadata", {})
    base_count = sum(
        1 for f in key_fields
        if source.get(f) or metadata.get(f)
    )
    # Extra credit for rich metadata
    meta_keys = len([k for k, v in metadata.items() if v])
    return base_count + meta_keys


# Normalization 

def _normalize_source(raw: dict, source_id: str) -> dict:
    """
    Normalize a raw source dict into the standard Source schema shape.

    raw has the shape from AnalyzeRequest.SourceInput:
    {url, content, metadata: {title, author, published_date, doi, ...}}
    """
    metadata = raw.get("metadata", {}) or {}
    url = raw.get("url") or metadata.get("url")
    domain = _extract_domain(url)

    # Strip HTML from content
    raw_content = raw.get("content", "")
    content_text = _strip_html(raw_content)

    # Normalize whitespace
    content_text = re.sub(r"\s{3,}", "\n\n", content_text).strip()

    title = metadata.get("title") or raw.get("title")
    author = metadata.get("author") or raw.get("author")
    published_date = (
        metadata.get("published_date")
        or metadata.get("date")
        or raw.get("published_date")
    )

    return {
        "source_id": source_id,
        "url": url,
        "title": title,
        "author": author,
        "domain": domain,
        "published_date": published_date,
        "content_text": content_text,
        "metadata": metadata,
        # Internal field used during dedup — not exposed downstream
        "_content_hash": _content_hash(content_text),
    }


def _validate_normalized(source: dict) -> Optional[str]:
    """
    Validate minimum requirements for a normalized source.
    Returns an error string if invalid, None if valid.
    """
    content = source.get("content_text", "")
    if not content or len(content.strip()) < 20:
        return "content_text is empty or too short after HTML stripping"

    url = source.get("url")
    title = source.get("title") or source.get("metadata", {}).get("title")
    if not url and not title:
        return "source must have at least one of: url, title"

    return None


# Merge duplicate pair 

def _merge_pair(a: dict, b: dict) -> dict:
    """
    Merge two near-duplicate sources. Keep the one with richer metadata
    as the base, and fill in any missing fields from the other.
    """
    if _count_populated_fields(a) >= _count_populated_fields(b):
        primary, secondary = a, b
    else:
        primary, secondary = b, a

    merged = dict(primary)
    # Fill missing fields from secondary
    for field in ["url", "title", "author", "published_date"]:
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]

    # Merge metadata dicts
    merged_meta = dict(secondary.get("metadata", {}))
    merged_meta.update(primary.get("metadata", {}))
    merged["metadata"] = merged_meta

    # Keep primary source_id
    return merged


# Audit helper 

def _make_audit_record(request_id: str, data: dict) -> dict:
    return {
        "event": "ingress_complete",
        "stage": "ingress",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_ingress(state: PipelineState) -> dict:
    """
    LangGraph node function for the Ingress agent.

    Reads:  state["raw_sources"], state["request_id"]
    Writes: normalized_sources, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    raw_sources = state.get("raw_sources", [])

    log = logger.bind(stage="ingress", request_id=request_id)
    log.info("ingress_start", raw_count=len(raw_sources))

    validation_errors: list[dict] = []
    candidates: list[dict] = []

    # Step 1: Normalize and assign IDs 
    for i, raw in enumerate(raw_sources):
        source_id = _make_source_id(i, raw.get("url"), raw.get("content", ""))
        try:
            normalized = _normalize_source(raw, source_id)
        except Exception as exc:
            log.warning("ingress_normalize_error", index=i, error=str(exc))
            validation_errors.append({
                "index": i,
                "source_id": source_id,
                "reason": f"normalization_error: {exc}",
            })
            continue

        error = _validate_normalized(normalized)
        if error:
            err = SourceValidationError(source_index=i, reason=error, request_id=request_id)
            log.warning("ingress_validation_failed", index=i, source_id=source_id, reason=error)
            validation_errors.append({
                "index": i,
                "source_id": source_id,
                "reason": error,
            })
            continue

        candidates.append(normalized)

    log.info("ingress_after_validation", valid=len(candidates), dropped=len(validation_errors))

    # Step 2: URL-based deduplication 
    # Group by URL; within each group keep richest metadata
    url_groups: dict[str, list[dict]] = {}
    no_url: list[dict] = []

    for src in candidates:
        url = src.get("url")
        if url:
            url_groups.setdefault(url, []).append(src)
        else:
            no_url.append(src)

    url_deduped: list[dict] = []
    url_dup_count = 0
    for url, group in url_groups.items():
        if len(group) == 1:
            url_deduped.append(group[0])
        else:
            url_dup_count += len(group) - 1
            merged = group[0]
            for other in group[1:]:
                merged = _merge_pair(merged, other)
            url_deduped.append(merged)
            log.info("ingress_url_dedup", url=url, merged_count=len(group))

    after_url_dedup = url_deduped + no_url

    # Step 3: Content similarity deduplication 
    content_dup_count = 0
    if len(after_url_dedup) >= 2:
        texts = [s["content_text"] for s in after_url_dedup]
        ids = [s["source_id"] for s in after_url_dedup]

        try:
            similar_pairs = find_duplicate_sources(
                source_texts=texts,
                source_ids=ids,
                similarity_threshold=settings.dedup_similarity_threshold,
            )
        except Exception as exc:
            # Embedding may fail in test environments — log and skip content dedup
            log.warning("ingress_content_dedup_skipped", error=str(exc))
            similar_pairs = []

        # Build a mapping from source_id to index for O(1) lookup
        id_to_idx: dict[str, int] = {s["source_id"]: i for i, s in enumerate(after_url_dedup)}
        merged_into: dict[str, str] = {}  # source_id → absorbed into which id

        for id_a, id_b, score in similar_pairs:
            # Skip if either was already absorbed
            if id_a in merged_into or id_b in merged_into:
                continue

            idx_a = id_to_idx[id_a]
            idx_b = id_to_idx[id_b]
            src_a = after_url_dedup[idx_a]
            src_b = after_url_dedup[idx_b]

            merged = _merge_pair(src_a, src_b)
            # Keep id_a (lower numbered) as canonical
            merged["source_id"] = id_a
            after_url_dedup[idx_a] = merged
            merged_into[id_b] = id_a
            content_dup_count += 1
            log.info(
                "ingress_content_dedup",
                kept=id_a,
                absorbed=id_b,
                similarity=round(score, 3),
            )

        # Remove absorbed sources
        after_url_dedup = [
            s for s in after_url_dedup
            if s["source_id"] not in merged_into
        ]

    # Step 4: Re-assign sequential IDs after dedup 
    # IDs may have gaps after dedup; re-assign cleanly src_001, src_002, ...
    final_sources: list[dict] = []
    for i, src in enumerate(after_url_dedup):
        src = dict(src)
        src["source_id"] = f"src_{str(i + 1).zfill(3)}"
        # Remove internal dedup field
        src.pop("_content_hash", None)
        final_sources.append(src)

    log.info(
        "ingress_complete",
        normalized=len(final_sources),
        url_dups=url_dup_count,
        content_dups=content_dup_count,
        validation_dropped=len(validation_errors),
    )

    audit = _make_audit_record(request_id, {
        "raw_source_count": len(raw_sources),
        "normalized_count": len(final_sources),
        "url_duplicates_merged": url_dup_count,
        "content_duplicates_merged": content_dup_count,
        "validation_failures": validation_errors,
    })

    return {
        "normalized_sources": final_sources,
        "audit_trail": state["audit_trail"] + [audit],
    }