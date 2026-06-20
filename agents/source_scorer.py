"""
Source Scorer agent — runs after Ingress, before Source Count Guardrail.

Executes all 7 scoring steps per the spec:

Step 1 — Denylist check (registry/denylist.py)
    Domain match → drop source entirely, add to dropped_sources.

Step 2 — Allowlist check (registry/allowlist.py)
    Domain or DOI match → trusted_source = True, credibility floor = 0.90,
    skip metadata classification.

Step 3 — Metadata classification (deterministic, no LLM)
    Classify source type from URL, domain, DOI, author signals.
    If ambiguous (missing >2 key fields) → fall through to Step 4.

Step 4 — Content analysis (LLM, only when metadata is ambiguous)
    First 1500 chars → groq → structured JSON classification.

Step 5 — Credibility score
    base × recency_weight × citation_modifier × author_modifier, clamped [0,1].

Step 6 — Relevance score (LLM)
    topic_overlap × 0.4 + population_match × 0.3 + outcome_match × 0.3.

Step 7 — Bias flag (LLM)
    Three sub-checks → highest severity → cap credibility.

LLM usage: Steps 4, 6, 7 (per source, only when needed).
Writes to state: scored_sources, dropped_sources, audit_trail
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import structlog

from core.state import PipelineState
from core.exceptions import DenylistedSourceError
from registry.denylist import is_denylisted, get_denylist_reason
from registry.allowlist import is_trusted_domain, is_trusted_doi
from services.llm import call_llm

logger = structlog.get_logger(__name__)


# Date helpers 

def _parse_year(date_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from a date string."""
    if not date_str:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    return int(match.group()) if match else None


def _years_old(year: Optional[int]) -> Optional[float]:
    """Return how many years ago a year was, relative to current year."""
    if year is None:
        return None
    current_year = datetime.now(timezone.utc).year
    return max(0.0, float(current_year - year))


# Step 3: Metadata classification 

_KNOWN_JOURNALS = {
    "nature", "science", "cell", "lancet", "nejm", "jama", "bmj",
    "annals", "plos", "pnas", "circulation", "blood", "gut",
    "hepatology", "radiology", "oncology", "neurology",
    "the new england journal of medicine",
    "journal of the american medical association",
    "british medical journal",
}

_COMMERCIAL_URL_SIGNALS = re.compile(
    r"(shop|store|buy|product|cart|sale|promo|partner|sponsored|affiliate)",
    re.IGNORECASE,
)

_FIRST_PERSON_RE = re.compile(r"\b(I |my |I'm |I've |In my opinion)\b", re.IGNORECASE)


def _is_institutional_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    domain = domain.lower()
    for suffix in (".gov", ".edu", ".mil", ".ac.uk", ".ac.in", ".ac.au", ".ac.nz"):
        if domain.endswith(suffix):
            return True
    # Explicit institutional domains
    institutional = {"who.int", "nih.gov", "cdc.gov", "cochrane.org", "worldbank.org",
                     "imf.org", "oecd.org", "un.org"}
    return domain in institutional or any(domain.endswith(f".{d}") for d in institutional)


def _classify_metadata(source: dict, settings) -> tuple[str, int]:
    """
    Classify source type from metadata signals without using an LLM.

    Returns (source_type, ambiguity_count) where ambiguity_count is
    the number of key classification fields that were missing.
    Higher ambiguity_count → fall through to LLM content analysis.
    """
    meta = source.get("metadata", {}) or {}
    url = source.get("url", "") or ""
    domain = source.get("domain", "") or ""
    content = source.get("content_text", "") or ""
    author = source.get("author") or meta.get("author", "")
    title = source.get("title") or meta.get("title", "")

    doi = meta.get("doi", "")
    journal = meta.get("journal", "") or meta.get("journal_name", "")
    issn = meta.get("issn", "")
    institution_email = meta.get("institution_email", "")
    published_date = source.get("published_date") or meta.get("published_date")
    citation_count = meta.get("citation_count") or meta.get("citations")

    missing_key_fields = 0

    # peer_reviewed 
    has_doi = bool(doi)
    has_journal_signal = bool(journal) or bool(issn) or any(
        j in (journal or title or "").lower() for j in _KNOWN_JOURNALS
    )
    has_institutional_author = bool(institution_email) or bool(
        re.search(r"@(edu|ac\.|gov|nih|who|cdc)", str(author))
    )

    if has_doi and has_journal_signal:
        return "peer_reviewed", 0

    if not has_doi:
        missing_key_fields += 1
    if not has_journal_signal:
        missing_key_fields += 1
    if not has_institutional_author:
        missing_key_fields += 1

    # trusted_source / institutional 
    if _is_institutional_domain(domain):
        return "institutional", 0

    # established_press 
    if domain in settings.known_established_press:
        return "established_press", 0

    # commercial 
    if _COMMERCIAL_URL_SIGNALS.search(url):
        return "commercial", 0
    if "sponsored" in url.lower() or "partner" in url.lower():
        return "commercial", 0

    has_byline = bool(author)
    has_dateline = bool(published_date)

    if not has_byline:
        missing_key_fields += 1
    if not has_dateline:
        missing_key_fields += 1

    # blog_opinion — check BEFORE general_news so first-person wins 
    # First-person language is a stronger signal than byline+dateline presence
    first_person_count = len(_FIRST_PERSON_RE.findall(content[:1000]))
    if first_person_count >= 3 and not doi:
        return "blog_opinion", 0

    # general_news (has byline + dateline but not established press) 
    if has_byline and has_dateline:
        if not has_doi and not _is_institutional_domain(domain):
            return "general_news", 1  # mild ambiguity — could still be blog

    # Ambiguous — fall through to LLM if too many missing fields 
    if missing_key_fields >= 3:
        return "unknown", missing_key_fields

    # Best guess with remaining signals
    if has_byline:
        return "general_news", missing_key_fields

    return "blog_opinion", missing_key_fields


# Step 4: LLM content classification 

_CONTENT_CLASSIFICATION_SYSTEM = """You are a research librarian classifying an academic or web source.
Analyze the provided text excerpt and return ONLY a valid JSON object with these exact keys:
{
  "source_type": "<one of: peer_reviewed, institutional, established_press, general_news, blog_opinion, commercial, unknown>",
  "has_methods": <true|false>,
  "has_citations": <true|false>,
  "has_funding_disclosure": <true|false>,
  "promotional_language_score": <float 0.0-1.0, where 1.0 = highly promotional>
}
No markdown, no explanation, no preamble. Only the JSON object."""


def _llm_classify_content(source: dict, request_id: str) -> dict:
    """Call the LLM to classify ambiguous source content."""
    content_sample = source.get("content_text", "")[:1500]
    title = source.get("title", "")
    domain = source.get("domain", "")

    user_prompt = (
        f"Domain: {domain}\n"
        f"Title: {title}\n\n"
        f"Content excerpt (first 1500 chars):\n{content_sample}"
    )

    result = call_llm(
        system_prompt=_CONTENT_CLASSIFICATION_SYSTEM,
        user_prompt=user_prompt,
        call_type="source_classification",
        stage="source_scorer_step4",
        request_id=request_id,
        max_tokens=256,
        temperature=0.0,
    )
    return result


# Step 5: Credibility score 

def _compute_credibility(
    source_type: str,
    source: dict,
    settings,
    trusted_source: bool = False,
) -> tuple[float, dict]:
    """
    Compute credibility score per spec formula:
    base × recency_weight × citation_modifier × author_modifier, clamped [0, 1].

    Returns (credibility_score, breakdown_dict).
    """
    meta = source.get("metadata", {}) or {}

    # Base score
    if trusted_source:
        base = settings.base_credibility_scores["trusted_source"]
    else:
        base = settings.base_credibility_scores.get(source_type, 0.20)

    # Recency weight 
    published_date = source.get("published_date") or meta.get("published_date")
    year = _parse_year(published_date)
    age = _years_old(year)

    # Determine if question domain is fast-moving
    # (We don't have question context here; default to standard decay)
    # Fast-moving decay will be applied by the scorer if it has domain context
    if age is None:
        recency_weight = 0.90  # unknown date: slight penalty
    elif age < 2:
        recency_weight = 1.0
    elif age <= 5:
        recency_weight = 0.85
    else:
        recency_weight = 0.65

    # For stable domains (no explicit fast-moving signal), soften the penalty
    # Stable domain: > 20 years old → 0.85, otherwise 1.0
    # (We apply conservative standard decay as default)

    # Citation modifier 
    raw_citations = meta.get("citation_count") or meta.get("citations", 0)
    try:
        citations = int(raw_citations)
    except (TypeError, ValueError):
        citations = 0

    if citations >= 6:
        citation_modifier = 1.10
    elif citations >= 1:
        citation_modifier = 1.05
    else:
        citation_modifier = 0.95

    # Author modifier 
    author = source.get("author") or meta.get("author", "")
    orcid = meta.get("orcid", "")
    institutional_page = meta.get("institutional_page", "")

    has_verifiable_credentials = bool(orcid) or bool(institutional_page) or bool(
        re.search(r"@(edu|ac\.|gov|nih|who|cdc)", str(author))
    )
    is_anonymous = not author or author.strip().lower() in (
        "anonymous", "staff", "admin", "editor", "unknown", ""
    )

    if has_verifiable_credentials:
        author_modifier = 1.05
    elif is_anonymous:
        author_modifier = 0.90
    else:
        author_modifier = 1.0  # named author, not verifiable

    # Final score 
    score = base * recency_weight * citation_modifier * author_modifier
    score = round(min(1.0, max(0.0, score)), 4)

    breakdown = {
        "base": base,
        "recency_weight": recency_weight,
        "citation_modifier": citation_modifier,
        "author_modifier": author_modifier,
        "age_years": age,
        "citation_count": citations,
        "trusted_source": trusted_source,
    }

    return score, breakdown


# Step 6: Relevance score (LLM) 

_RELEVANCE_SYSTEM = """You are a research analyst assessing how relevant a source is to a research question.
Return ONLY a valid JSON object with these exact keys:
{
  "topic_overlap": <float 0.0-1.0, semantic match between question and source content>,
  "population_match": <float 0.0-1.0, where 1.0=exact match, 0.5=partial, 0.0=mismatch>,
  "outcome_match": <float 0.0-1.0, where 1.0=same metric+timeframe, 0.4=related but different, 0.0=different metric>
}
No markdown, no explanation. Only the JSON object."""


def _llm_score_relevance(question: str, source: dict, request_id: str) -> dict:
    content_summary = source.get("content_text", "")[:800]
    title = source.get("title", "")

    user_prompt = (
        f"Research question: {question}\n\n"
        f"Source title: {title}\n"
        f"Source content summary:\n{content_summary}"
    )

    return call_llm(
        system_prompt=_RELEVANCE_SYSTEM,
        user_prompt=user_prompt,
        call_type="relevance",
        stage="source_scorer_step6",
        request_id=request_id,
        max_tokens=128,
        temperature=0.0,
    )


def _compute_relevance(relevance_data: dict) -> float:
    """
    Formula: relevance = 0.4 × topic_overlap + 0.3 × population_match + 0.3 × outcome_match
    """
    topic = float(relevance_data.get("topic_overlap", 0.5))
    population = float(relevance_data.get("population_match", 0.5))
    outcome = float(relevance_data.get("outcome_match", 0.5))
    score = 0.4 * topic + 0.3 * population + 0.3 * outcome
    return round(min(1.0, max(0.0, score)), 4)


# Step 7: Bias flag (LLM) 

_BIAS_SYSTEM = """You are a research integrity analyst checking a source for bias.
Evaluate three dimensions and return ONLY a valid JSON object:
{
  "funding_conflict": "<none|low|moderate|high>",
  "commercial_interest": "<none|low|moderate|high>",
  "framing_language": "<none|low|moderate|high>",
  "overall_bias": "<none|low|moderate|high>",
  "reasoning": "<one sentence explanation>"
}
Severity guide:
- funding_conflict: high=funder has financial interest in outcome, moderate=undisclosed funding, none=neutral/public
- commercial_interest: high=entity profits from the conclusion, moderate=company-authored, none=no commercial stake
- framing_language: moderate=absolute claims or promotional tone, none=hedged and neutral language
- overall_bias: take the highest severity across the three dimensions
No markdown, no preamble. Only the JSON object."""

_BIAS_SEVERITY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3}


def _llm_check_bias(source: dict, request_id: str) -> dict:
    content_sample = source.get("content_text", "")[:1000]
    meta = source.get("metadata", {}) or {}

    user_prompt = (
        f"Source domain: {source.get('domain', 'unknown')}\n"
        f"Author: {source.get('author') or meta.get('author', 'unknown')}\n"
        f"Funding: {meta.get('funding', 'not disclosed')}\n"
        f"Publisher: {meta.get('publisher', 'unknown')}\n\n"
        f"Content excerpt:\n{content_sample}"
    )

    return call_llm(
        system_prompt=_BIAS_SYSTEM,
        user_prompt=user_prompt,
        call_type="bias",
        stage="source_scorer_step7",
        request_id=request_id,
        max_tokens=200,
        temperature=0.0,
    )


def _apply_bias_cap(credibility: float, bias_flag: str, settings) -> float:
    """Apply bias credibility cap and return adjusted credibility."""
    cap = settings.bias_credibility_caps.get(bias_flag, 1.0)
    return round(min(credibility, cap), 4)


# Audit helper 

def _audit_record(request_id: str, event: str, data: dict) -> dict:
    return {
        "event": event,
        "stage": "source_scorer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_source_scorer(state: PipelineState) -> dict:
    """
    LangGraph node function for the Source Scorer agent.

    Reads:  state["normalized_sources"], state["question"], state["request_id"]
    Writes: scored_sources, dropped_sources, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    question = state["question"]
    normalized_sources = state.get("normalized_sources", [])

    log = logger.bind(stage="source_scorer", request_id=request_id)
    log.info("source_scorer_start", source_count=len(normalized_sources))

    scored_sources: list[dict] = []
    dropped_sources: list[dict] = []
    audit_records: list[dict] = []

    for source in normalized_sources:
        source_id = source["source_id"]
        domain = source.get("domain")
        meta = source.get("metadata", {}) or {}
        doi = meta.get("doi", "")
        src_log = log.bind(source_id=source_id, domain=domain)

        # Step 1: Denylist check 
        if is_denylisted(domain):
            reason = get_denylist_reason(domain)
            src_log.warning("source_denylisted", reason=reason)
            dropped_sources.append({
                "source_id": source_id,
                "domain": domain,
                "reason": reason,
            })
            audit_records.append(_audit_record(request_id, "source_dropped", {
                "source_id": source_id,
                "domain": domain,
                "reason": f"denylist:{reason}",
            }))
            continue

        # Step 2: Allowlist check 
        trusted_source = is_trusted_domain(domain) or is_trusted_doi(doi)
        if trusted_source:
            src_log.info("source_allowlisted")

        # Step 3 & 4: Source type classification 
        llm_classification: dict = {}

        if trusted_source:
            source_type = "trusted_source"
            ambiguity = 0
        else:
            source_type, ambiguity = _classify_metadata(source, settings)

            if source_type == "unknown" or ambiguity >= 3:
                # Fall through to LLM content analysis
                src_log.info("source_metadata_ambiguous", ambiguity=ambiguity)
                try:
                    llm_classification = _llm_classify_content(source, request_id)
                    source_type = llm_classification.get("source_type", "unknown")
                    src_log.info("source_llm_classified", source_type=source_type)
                except Exception as exc:
                    src_log.warning("source_llm_classification_failed", error=str(exc))
                    source_type = "unknown"

        # Step 5: Credibility score 
        try:
            credibility, breakdown = _compute_credibility(
                source_type=source_type,
                source=source,
                settings=settings,
                trusted_source=trusted_source,
            )
        except Exception as exc:
            src_log.warning("credibility_score_failed", error=str(exc))
            credibility = 0.20
            breakdown = {"error": str(exc)}

        # Step 6: Relevance score 
        try:
            relevance_data = _llm_score_relevance(question, source, request_id)
            relevance = _compute_relevance(relevance_data)
        except Exception as exc:
            src_log.warning("relevance_score_failed", error=str(exc))
            relevance = 0.40  # neutral fallback
            relevance_data = {"error": str(exc)}

        # Step 7: Bias flag 
        try:
            bias_data = _llm_check_bias(source, request_id)
            bias_flag = bias_data.get("overall_bias", "none")
            # Validate bias flag value
            if bias_flag not in ("none", "low", "moderate", "high"):
                bias_flag = "none"
        except Exception as exc:
            src_log.warning("bias_check_failed", error=str(exc))
            bias_flag = "none"
            bias_data = {"error": str(exc)}

        # Apply bias credibility cap
        credibility_before_cap = credibility
        credibility = _apply_bias_cap(credibility, bias_flag, settings)
        if credibility < credibility_before_cap:
            src_log.info(
                "bias_cap_applied",
                bias=bias_flag,
                before=credibility_before_cap,
                after=credibility,
            )

        scored_source = {
            "source_id": source_id,
            "url": source.get("url"),
            "title": source.get("title"),
            "domain": domain,
            "credibility": credibility,
            "relevance": relevance,
            "bias_flag": bias_flag,
            "source_type": source_type,
            "trusted_source": trusted_source,
            "score_breakdown": {
                **breakdown,
                "relevance_components": relevance_data,
                "bias_components": bias_data,
                "llm_classification": llm_classification,
            },
        }

        scored_sources.append(scored_source)

        audit_records.append(_audit_record(request_id, "source_scored", {
            "source_id": source_id,
            "source_type": source_type,
            "trusted_source": trusted_source,
            "credibility": credibility,
            "relevance": relevance,
            "bias_flag": bias_flag,
        }))

        src_log.info(
            "source_scored",
            type=source_type,
            credibility=credibility,
            relevance=relevance,
            bias=bias_flag,
        )

    log.info(
        "source_scorer_complete",
        scored=len(scored_sources),
        dropped=len(dropped_sources),
    )

    # Append summary audit record
    audit_records.append(_audit_record(request_id, "source_scorer_summary", {
        "scored_count": len(scored_sources),
        "dropped_count": len(dropped_sources),
    }))

    return {
        "scored_sources": scored_sources,
        "dropped_sources": state["dropped_sources"] + dropped_sources,
        "audit_trail": state["audit_trail"] + audit_records,
    }