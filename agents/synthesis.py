"""
Synthesis agent — first node in the adversarial loop.

Responsibilities:
1. Filter scored sources to those meeting quality thresholds (credibility >= 0.50
   AND relevance >= 0.40). If no sources meet these, use top-3 by combined score.
2. Build a preliminary answer using ONLY the filtered sources.
3. Every sentence in the answer cites at least one source_id.
4. Note scope limitations inline (population, timeframe, outcome mismatches).
5. Produce an initial confidence estimate using the spec's heuristics.
6. Build the claim_source_map for the current answer state.

Confidence heuristics (conservative — take lower tier when borderline):
  High:        >= 2 peer-reviewed sources, credibility > 0.80, relevance > 0.80,
               no irreconcilable conflicts
  Moderate:    1 peer-reviewed OR 2+ institutional sources, no irreconcilable conflicts
  Low:         Only blog/news sources, OR irreconcilable conflicts present,
               OR relevance < 0.50
  Insufficient: < 2 usable sources OR all credibility < 0.30

The Synthesis agent runs on EVERY loop iteration (round 0, 1, 2).
On rounds > 0, it receives the Arbiter's revision instructions and adjusts.

LLM usage: One call per synthesis pass (call_llm_text for prose output).
Writes to state: preliminary_answer, confidence, claim_source_map, audit_trail
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from core.state import PipelineState, ConfidenceTier
from services.llm import call_llm_text

logger = structlog.get_logger(__name__)

# Quality thresholds for primary filtering 
SYNTHESIS_CREDIBILITY_FLOOR = 0.50
SYNTHESIS_RELEVANCE_FLOOR = 0.40
FALLBACK_TOP_N = 3


# Prompts 

_SYNTHESIS_SYSTEM = """You are a rigorous research analyst synthesizing evidence to answer a research question.

Rules you MUST follow:
1. Answer ONLY from the sources provided. Do not add outside knowledge.
2. Every factual sentence must cite the source using [src_XXX] notation inline.
3. Note scope limitations explicitly: if a study covers a specific population or timeframe that may not match the question, say so in parentheses.
4. Be precise about effect sizes, percentages, and p-values when provided by the sources.
5. Do not overstate certainty. Use hedging language ("suggests", "found that", "one study shows") rather than absolutes.
6. If sources conflict, acknowledge the conflict explicitly rather than ignoring one side.
7. Keep the answer focused and under 400 words.

Format: Plain prose paragraphs only. No headers, no bullet points, no markdown."""

_REVISION_SYSTEM = """You are a rigorous research analyst revising a preliminary answer based on specific feedback from a peer reviewer.

Rules you MUST follow:
1. Address EVERY revision instruction listed. Do not skip any.
2. Answer ONLY from the sources provided. Do not add outside knowledge.
3. Every factual sentence must cite the source using [src_XXX] notation inline.
4. If a revision instruction says to remove or downweight a source, do so.
5. If a revision instruction says to add a scope limitation, add it explicitly.
6. Be more conservative about conclusions than the previous draft.
7. Keep the answer focused and under 400 words.

Format: Plain prose paragraphs only. No headers, no bullet points, no markdown."""


def _build_synthesis_prompt(
    question: str,
    filtered_sources: list[dict],
    normalized_lookup: dict[str, dict],
    claims: list[dict],
    contradictions: list[dict],
    revision_instructions: list[str],
    loop_round: int,
) -> str:
    lines = [f"Research question: {question}\n"]

    if revision_instructions:
        lines.append("REVISION INSTRUCTIONS FROM PEER REVIEWER:")
        for i, instr in enumerate(revision_instructions, 1):
            lines.append(f"  {i}. {instr}")
        lines.append("")

    lines.append("SOURCES TO USE (ordered by credibility):\n")

    # Sort filtered sources by credibility desc
    sorted_sources = sorted(
        filtered_sources, key=lambda s: s.get("credibility", 0), reverse=True
    )

    for src in sorted_sources:
        sid = src["source_id"]
        norm = normalized_lookup.get(sid, {})
        title = norm.get("title") or src.get("title") or "Untitled"
        content = norm.get("content_text", "")[:800]
        cred = src.get("credibility", 0)
        rel = src.get("relevance", 0)
        src_type = src.get("source_type", "unknown")
        bias = src.get("bias_flag", "none")

        lines.append(
            f"[{sid}] {title}\n"
            f"  Type: {src_type} | Credibility: {cred:.2f} | Relevance: {rel:.2f} | Bias: {bias}\n"
            f"  Content: {content}\n"
        )

    # Claims relevant to filtered sources
    relevant_claims = [
        c for c in claims
        if c.get("source_id") in {s["source_id"] for s in filtered_sources}
        and not c.get("low_credibility", False)
    ]
    if relevant_claims:
        lines.append("\nKEY CLAIMS EXTRACTED FROM SOURCES:")
        for c in relevant_claims[:12]:  # cap to avoid token bloat
            scope_parts = []
            if c.get("population"):
                scope_parts.append(f"population={c['population']}")
            if c.get("timeframe"):
                scope_parts.append(f"timeframe={c['timeframe']}")
            if c.get("outcome_metric"):
                scope_parts.append(f"outcome={c['outcome_metric']}")
            scope = " | ".join(scope_parts) if scope_parts else "scope unspecified"
            lines.append(f"  [{c['source_id']}] {c['text']} ({scope})")

    # Active contradictions
    active_contradictions = [
        con for con in contradictions
        if con.get("conflict_type") in ("direct_conflict", "irreconcilable")
    ]
    if active_contradictions:
        lines.append("\nACTIVE CONTRADICTIONS (you must acknowledge these):")
        for con in active_contradictions:
            lines.append(
                f"  {con.get('conflict_type', '').upper()}: "
                f"{con.get('claim_a_id')} vs {con.get('claim_b_id')} — "
                f"{con.get('explanation', '')}"
            )

    if loop_round > 0:
        lines.append(f"\n[This is revision round {loop_round}. Be more conservative than the previous draft.]")

    return "\n".join(lines)


# Confidence heuristics 

def _estimate_confidence(
    filtered_sources: list[dict],
    contradictions: list[dict],
    source_count_ceiling: Optional[ConfidenceTier],
) -> ConfidenceTier:
    """
    Apply spec confidence heuristics. Always take the LOWER tier when borderline.
    The source_count_ceiling from the guardrail is enforced here as a hard cap.
    """
    _TIER_ORDER: dict[ConfidenceTier, int] = {
        "High": 3, "Moderate": 2, "Low": 1, "Insufficient Evidence": 0
    }

    has_irreconcilable = any(
        c.get("conflict_type") == "irreconcilable" for c in contradictions
    )

    usable = filtered_sources  # already filtered to pass thresholds

    # Insufficient Evidence 
    if len(usable) < 2:
        return "Insufficient Evidence"
    if all(s.get("credibility", 0) < 0.30 for s in usable):
        return "Insufficient Evidence"

    # High 
    peer_reviewed_high = [
        s for s in usable
        if s.get("source_type") in ("peer_reviewed", "trusted_source")
        and s.get("credibility", 0) > 0.80
        and s.get("relevance", 0) > 0.80
    ]
    if len(peer_reviewed_high) >= 2 and not has_irreconcilable:
        estimated: ConfidenceTier = "High"
    # Moderate 
    elif not has_irreconcilable and (
        any(s.get("source_type") in ("peer_reviewed", "trusted_source") for s in usable)
        or len([s for s in usable if s.get("source_type") == "institutional"]) >= 2
    ):
        estimated = "Moderate"
    # Low 
    elif (
        has_irreconcilable
        or all(s.get("source_type") in ("blog_opinion", "general_news", "commercial") for s in usable)
        or all(s.get("relevance", 0) < 0.50 for s in usable)
    ):
        estimated = "Low"
    else:
        estimated = "Low"  # conservative default

    # Apply source_count_ceiling guardrail 
    if source_count_ceiling is not None:
        ceiling_order = _TIER_ORDER.get(source_count_ceiling, 0)
        estimated_order = _TIER_ORDER.get(estimated, 0)
        if estimated_order > ceiling_order:
            estimated = source_count_ceiling

    return estimated


# Claim-source map builder 

def _build_claim_source_map(
    filtered_sources: list[dict],
    claims: list[dict],
) -> list[dict]:
    """Build the claim→source mapping for the current synthesis pass."""
    source_ids = {s["source_id"] for s in filtered_sources}
    scored_lookup = {s["source_id"]: s for s in filtered_sources}

    result = []
    for claim in claims:
        sid = claim.get("source_id", "")
        if sid not in source_ids:
            continue
        if claim.get("low_credibility"):
            continue
        src = scored_lookup[sid]
        result.append({
            "claim_text": claim["text"],
            "source_id": sid,
            "credibility": src.get("credibility", 0.0),
            "relevance": src.get("relevance", 0.0),
        })
    return result


# Audit helper 

def _audit(request_id: str, event: str, data: dict) -> dict:
    return {
        "event": event,
        "stage": "synthesis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_synthesis(state: PipelineState) -> dict:
    """
    LangGraph node function for the Synthesis agent.

    Reads:  state["scored_sources"], state["normalized_sources"],
            state["claims"], state["contradictions"],
            state["arbiter_rulings"], state["loop_round"],
            state["source_count_ceiling"], state["request_id"]
    Writes: preliminary_answer, confidence, claim_source_map, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    question = state["question"]
    scored_sources = state.get("scored_sources", [])
    normalized_sources = state.get("normalized_sources", [])
    claims = state.get("claims", [])
    contradictions = state.get("contradictions", [])
    arbiter_rulings = state.get("arbiter_rulings", [])
    loop_round = state.get("loop_round", 0)
    source_count_ceiling = state.get("source_count_ceiling")

    log = logger.bind(stage="synthesis", request_id=request_id, round=loop_round)
    log.info("synthesis_start", source_count=len(scored_sources), round=loop_round)

    normalized_lookup = {s["source_id"]: s for s in normalized_sources}

    # Filter sources to synthesis quality thresholds 
    filtered = [
        s for s in scored_sources
        if s.get("credibility", 0) >= SYNTHESIS_CREDIBILITY_FLOOR
        and s.get("relevance", 0) >= SYNTHESIS_RELEVANCE_FLOOR
    ]

    # Fallback: if nothing clears the threshold, use top-N by combined score
    if not filtered and scored_sources:
        log.warning("synthesis_threshold_fallback", total_sources=len(scored_sources))
        sorted_all = sorted(
            scored_sources,
            key=lambda s: (s.get("credibility", 0) + s.get("relevance", 0)) / 2,
            reverse=True,
        )
        filtered = sorted_all[:FALLBACK_TOP_N]

    log.info(
        "synthesis_sources_selected",
        filtered_count=len(filtered),
        total_scored=len(scored_sources),
    )

    # Collect revision instructions from valid Arbiter rulings 
    revision_instructions: list[str] = []
    if loop_round > 0:
        # Extract suggested revisions from critical/valid challenges this round
        current_round = loop_round - 1  # rulings reference the previous round
        valid_rulings = [
            r for r in arbiter_rulings
            if r.get("ruling") == "valid_critical" and r.get("round") == current_round
        ]
        # Cross-reference with skeptic challenges to get suggested_revision
        challenges_lookup = {
            c["challenge_id"]: c for c in state.get("skeptic_challenges", [])
        }
        for ruling in valid_rulings:
            challenge = challenges_lookup.get(ruling.get("challenge_id", ""), {})
            revision = challenge.get("suggested_revision", "")
            if revision:
                revision_instructions.append(revision)
            reason = ruling.get("reason", "")
            if reason and reason not in revision_instructions:
                revision_instructions.append(reason)

    # LLM synthesis call 
    system_prompt = _REVISION_SYSTEM if loop_round > 0 else _SYNTHESIS_SYSTEM
    user_prompt = _build_synthesis_prompt(
        question=question,
        filtered_sources=filtered,
        normalized_lookup=normalized_lookup,
        claims=claims,
        contradictions=contradictions,
        revision_instructions=revision_instructions,
        loop_round=loop_round,
    )

    try:
        answer = call_llm_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            call_type="synthesis",
            stage="synthesis",
            request_id=request_id,
            max_tokens=1000,
            temperature=0.2,
        )
    except Exception as exc:
        log.error("synthesis_llm_failed", error=str(exc))
        answer = (
            f"Synthesis failed at round {loop_round} due to LLM error: {exc}. "
            "Unable to produce a preliminary answer."
        )

    # Confidence estimate 
    confidence = _estimate_confidence(
        filtered_sources=filtered,
        contradictions=contradictions,
        source_count_ceiling=source_count_ceiling,
    )

    # Claim-source map 
    claim_source_map = _build_claim_source_map(filtered, claims)

    log.info(
        "synthesis_complete",
        confidence=confidence,
        answer_len=len(answer),
        claim_map_entries=len(claim_source_map),
        revision_instructions=len(revision_instructions),
    )

    audit = _audit(request_id, "synthesis_complete", {
        "loop_round": loop_round,
        "sources_used": [s["source_id"] for s in filtered],
        "confidence_estimate": confidence,
        "revision_instructions": revision_instructions,
        "answer_length": len(answer),
    })

    return {
        "preliminary_answer": answer,
        "confidence": confidence,
        "claim_source_map": claim_source_map,
        "audit_trail": state["audit_trail"] + [audit],
    }