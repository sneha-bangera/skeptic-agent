"""
Confidence floor guardrail — runs after the Arbiter finalizes the loop.

Responsibility:
- If loop_terminated=True AND unresolved critical challenges remain AND
  loop_round >= max_loop_rounds → force confidence to "Insufficient Evidence"
- If any irreconcilable contradiction exists and confidence is "High" → downgrade to "Moderate"
- If source_count_ceiling is set, enforce it as a hard cap one final time
  (belt-and-suspenders after Arbiter)
- Append specific reasons for the floor enforcement to unresolved_contradictions

This is the last line of defense before Egress. It is pure code — no LLM.

LLM usage: None.
Writes to state: confidence, unresolved_contradictions, audit_trail
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from core.state import PipelineState, ConfidenceTier

logger = structlog.get_logger(__name__)

_TIER_ORDER: dict[ConfidenceTier, int] = {
    "High": 3, "Moderate": 2, "Low": 1, "Insufficient Evidence": 0
}


def _audit(request_id: str, data: dict) -> dict:
    return {
        "event": "confidence_floor_guardrail",
        "stage": "confidence_floor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


def run_confidence_floor(state: PipelineState) -> dict:
    """
    LangGraph node function for the confidence floor guardrail.

    Reads:  state["confidence"], state["loop_round"], state["loop_terminated"],
            state["arbiter_rulings"], state["skeptic_challenges"],
            state["contradictions"], state["source_count_ceiling"]
    Writes: confidence, unresolved_contradictions, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    confidence: ConfidenceTier = state.get("confidence") or "Low"
    loop_round = state.get("loop_round", 0)
    loop_terminated = state.get("loop_terminated", True)
    arbiter_rulings = state.get("arbiter_rulings", [])
    skeptic_challenges = state.get("skeptic_challenges", [])
    contradictions = state.get("contradictions", [])
    source_count_ceiling = state.get("source_count_ceiling")
    existing_unresolved = state.get("unresolved_contradictions", [])

    log = logger.bind(stage="confidence_floor", request_id=request_id)
    log.info(
        "confidence_floor_start",
        current_confidence=confidence,
        loop_round=loop_round,
        loop_terminated=loop_terminated,
    )

    adjustments: list[str] = []
    adjusted_confidence = confidence

    # Rule 1: Unresolved critical challenges after max rounds 
    if loop_terminated and loop_round >= settings.max_loop_rounds:
        ruled_ids = {r["challenge_id"] for r in arbiter_rulings}
        unresolved_critical = [
            c for c in skeptic_challenges
            if c.get("severity") == "critical"
            and c["challenge_id"] not in ruled_ids
        ]
        if unresolved_critical:
            reason = (
                f"Forced to Insufficient Evidence: {len(unresolved_critical)} critical "
                f"challenge(s) remained unresolved after {loop_round} adversarial rounds."
            )
            adjustments.append(reason)
            adjusted_confidence = "Insufficient Evidence"
            log.warning(
                "confidence_floor_forced_insufficient",
                unresolved_critical=len(unresolved_critical),
                loop_round=loop_round,
            )

    # Rule 2: Irreconcilable contradiction → cannot be High 
    has_irreconcilable = any(
        c.get("conflict_type") == "irreconcilable" for c in contradictions
    )
    if (
        has_irreconcilable
        and _TIER_ORDER.get(adjusted_confidence, 0) >= _TIER_ORDER["High"]
    ):
        reason = (
            "Confidence downgraded from High to Moderate: "
            "irreconcilable contradiction exists between high-credibility sources."
        )
        adjustments.append(reason)
        adjusted_confidence = "Moderate"
        log.info("confidence_floor_irreconcilable_cap", from_tier="High", to_tier="Moderate")

    # Rule 3: source_count_ceiling hard cap 
    if source_count_ceiling is not None:
        ceiling_order = _TIER_ORDER.get(source_count_ceiling, 0)
        current_order = _TIER_ORDER.get(adjusted_confidence, 0)
        if current_order > ceiling_order:
            reason = (
                f"Confidence capped at {source_count_ceiling}: "
                f"fewer than {settings.source_count_floor} usable sources were available."
            )
            adjustments.append(reason)
            adjusted_confidence = source_count_ceiling
            log.info(
                "confidence_floor_ceiling_cap",
                ceiling=source_count_ceiling,
                from_tier=confidence,
            )

    # Append adjustment reasons to unresolved_contradictions for transparency
    new_unresolved = list(existing_unresolved) + adjustments

    log.info(
        "confidence_floor_complete",
        original=confidence,
        adjusted=adjusted_confidence,
        adjustments=len(adjustments),
    )

    return {
        "confidence": adjusted_confidence,
        "unresolved_contradictions": new_unresolved,
        "audit_trail": state["audit_trail"] + [_audit(request_id, {
            "original_confidence": confidence,
            "adjusted_confidence": adjusted_confidence,
            "adjustments": adjustments,
            "loop_round": loop_round,
            "has_irreconcilable": has_irreconcilable,
            "source_count_ceiling": source_count_ceiling,
        })],
    }