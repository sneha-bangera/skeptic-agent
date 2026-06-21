"""
Egress agent — formats the final output and applies the output disclaimer guardrail.

Responsibilities:
1. Take final_answer, confidence, claim_source_map, unresolved_contradictions from state.
2. Apply output disclaimer if input_domain_flag is set (medical/legal/financial).
3. Format response matching the APIResponse Pydantic schema.
4. Structure claim_source_map into CitationEntry objects, enriched with URL/title
   from normalized_sources.
5. Summarise skeptic_challenges + arbiter_rulings into SkepticChallengeSummary objects.
6. Determine HITL status from state["hitl_required"] and state["hitl_approved"].
7. Count sources_used and sources_dropped for the response metadata.

LLM usage: None. Pure formatting logic.
Writes to state: disclaimer, audit_trail
The formatted APIResponse is stored in state["_api_response"] for the API layer to return.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from core.state import PipelineState
from guardrails.output_disclaimer import get_disclaimer

logger = structlog.get_logger(__name__)


# Citation builder 

def _build_citations(
    claim_source_map: list[dict],
    scored_sources: list[dict],
    normalized_sources: list[dict],
) -> list[dict]:
    """
    Enrich the raw claim_source_map with URL and title from normalized_sources.
    Returns a list of CitationEntry-compatible dicts.
    """
    scored_lookup = {s["source_id"]: s for s in scored_sources}
    norm_lookup = {s["source_id"]: s for s in normalized_sources}

    citations = []
    seen_claims: set[str] = set()

    for entry in claim_source_map:
        claim_text = entry.get("claim_text", "").strip()
        source_id = entry.get("source_id", "")

        # Deduplicate identical claim texts
        if claim_text in seen_claims:
            continue
        if not claim_text or not source_id:
            continue
        seen_claims.add(claim_text)

        scored = scored_lookup.get(source_id, {})
        norm = norm_lookup.get(source_id, {})

        citations.append({
            "claim_text": claim_text,
            "source_id": source_id,
            "credibility": entry.get("credibility", scored.get("credibility", 0.0)),
            "relevance": entry.get("relevance", scored.get("relevance", 0.0)),
            "source_url": norm.get("url") or scored.get("url"),
            "source_title": norm.get("title") or scored.get("title"),
        })

    # Sort by credibility descending so highest-quality citations appear first
    citations.sort(key=lambda c: c["credibility"], reverse=True)
    return citations


# Challenge summary builder 

def _build_challenge_summaries(
    skeptic_challenges: list[dict],
    arbiter_rulings: list[dict],
) -> list[dict]:
    """
    Cross-reference skeptic challenges with arbiter rulings to produce
    SkepticChallengeSummary-compatible dicts.
    """
    rulings_lookup = {r["challenge_id"]: r for r in arbiter_rulings}
    summaries = []

    for ch in skeptic_challenges:
        cid = ch["challenge_id"]
        ruling = rulings_lookup.get(cid)
        was_resolved = ruling is not None and ruling.get("ruling") in (
            "valid_critical", "valid_minor", "invalid", "dismissed"
        )
        summaries.append({
            "challenge_id": cid,
            "type": ch.get("type", "unknown"),
            "description": ch.get("description", ""),
            "severity": ch.get("severity", "minor"),
            "was_resolved": was_resolved,
            "arbiter_ruling": ruling.get("ruling") if ruling else None,
        })

    return summaries


# HITL status resolver 

def _resolve_hitl_status(state: PipelineState) -> str:
    hitl_required = state.get("hitl_required", False)
    hitl_approved = state.get("hitl_approved")

    if not hitl_required:
        return "not_required"
    if hitl_approved is True:
        return "approved"
    if hitl_approved is False:
        return "returned_for_revision"
    return "pending"


# Audit helper 

def _audit(request_id: str, data: dict) -> dict:
    return {
        "event": "egress_complete",
        "stage": "egress",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_egress(state: PipelineState) -> dict:
    """
    LangGraph node function for the Egress agent.

    Reads:  state["final_answer"], state["confidence"], state["claim_source_map"],
            state["unresolved_contradictions"], state["input_domain_flag"],
            state["scored_sources"], state["dropped_sources"],
            state["normalized_sources"], state["skeptic_challenges"],
            state["arbiter_rulings"], state["loop_round"],
            state["hitl_required"], state["hitl_approved"]
    Writes: disclaimer, _api_response, audit_trail
    """
    request_id = state["request_id"]
    log = logger.bind(stage="egress", request_id=request_id)
    log.info("egress_start")

    final_answer = state.get("final_answer") or ""
    confidence = state.get("confidence")
    claim_source_map = state.get("claim_source_map", [])
    unresolved_contradictions = state.get("unresolved_contradictions", [])
    input_domain_flag = state.get("input_domain_flag")
    scored_sources = state.get("scored_sources", [])
    dropped_sources = state.get("dropped_sources", [])
    normalized_sources = state.get("normalized_sources", [])
    skeptic_challenges = state.get("skeptic_challenges", [])
    arbiter_rulings = state.get("arbiter_rulings", [])
    loop_round = state.get("loop_round", 0)
    input_rejected = state.get("input_rejected", False)

    # 1. Disclaimer 
    disclaimer = get_disclaimer(input_domain_flag, confidence)

    # 2. Citations 
    citations = _build_citations(claim_source_map, scored_sources, normalized_sources)

    # 3. Challenge summaries 
    challenge_summaries = _build_challenge_summaries(skeptic_challenges, arbiter_rulings)

    # 4. HITL status 
    hitl_status = _resolve_hitl_status(state)

    # 5. Pipeline status 
    if input_rejected:
        pipeline_status = "failed"
    elif hitl_status == "pending":
        pipeline_status = "hitl_pending"
    else:
        pipeline_status = "complete"

    # 6. Assemble APIResponse dict 
    api_response = {
        "request_id": request_id,
        "question": state.get("question", ""),
        "answer": final_answer if final_answer else None,
        "confidence": confidence,
        "citations": citations,
        "unresolved_contradictions": unresolved_contradictions,
        "disclaimer": disclaimer,
        "sources_used": len(scored_sources),
        "sources_dropped": len(dropped_sources),
        "loop_rounds": loop_round,
        "skeptic_challenges": challenge_summaries,
        "hitl_status": hitl_status,
        "status": pipeline_status,
        "error_message": (
            state.get("input_rejection_reason")
            if input_rejected else None
        ),
    }

    log.info(
        "egress_complete",
        confidence=confidence,
        citations=len(citations),
        challenges=len(challenge_summaries),
        has_disclaimer=disclaimer is not None,
        hitl_status=hitl_status,
        pipeline_status=pipeline_status,
    )

    audit = _audit(request_id, {
        "confidence": confidence,
        "citations_count": len(citations),
        "challenges_count": len(challenge_summaries),
        "has_disclaimer": disclaimer is not None,
        "domain_flag": input_domain_flag,
        "pipeline_status": pipeline_status,
    })

    return {
        "disclaimer": disclaimer,
        "api_response": api_response,
        "audit_trail": state["audit_trail"] + [audit],
    }