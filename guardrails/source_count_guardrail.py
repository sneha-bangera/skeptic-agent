"""
Source count guardrail node — runs after Source Scorer, before Evidence Extractor.

Responsibility:
- Count how many scored sources are "usable" (credibility >= threshold AND relevance >= threshold)
- If usable count < source_count_floor (default 3), set source_count_ceiling = "Low"
- If usable count == 0, set source_count_ceiling = "Insufficient Evidence"
- The Arbiter enforces this ceiling when setting final confidence — it can never
  exceed the ceiling regardless of what Synthesis estimates.

LLM usage: None. Pure arithmetic.
Writes to state: source_count_ceiling, audit_trail
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from core.state import PipelineState, ConfidenceTier

logger = structlog.get_logger(__name__)


def _make_audit_record(request_id: str, data: dict) -> dict:
    return {
        "event": "source_count_guardrail",
        "stage": "source_count_guardrail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


def run_source_count_guardrail(state: PipelineState) -> dict:
    """
    LangGraph node function for the source count guardrail.

    Reads:  state["scored_sources"], config thresholds
    Writes: source_count_ceiling, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    scored_sources = state.get("scored_sources", [])

    log = logger.bind(stage="source_count_guardrail", request_id=request_id)

    # Count usable sources
    usable = [
        s for s in scored_sources
        if (
            s.get("credibility", 0.0) >= settings.min_credibility_threshold
            and s.get("relevance", 0.0) >= settings.min_relevance_threshold
        )
    ]
    usable_count = len(usable)
    total_scored = len(scored_sources)

    ceiling: ConfidenceTier | None = None

    if usable_count == 0:
        ceiling = "Insufficient Evidence"
        log.warning(
            "source_count_guardrail_triggered",
            usable=0,
            total=total_scored,
            ceiling=ceiling,
        )
    elif usable_count < settings.source_count_floor:
        ceiling = "Low"
        log.warning(
            "source_count_guardrail_triggered",
            usable=usable_count,
            floor=settings.source_count_floor,
            ceiling=ceiling,
        )
    else:
        log.info(
            "source_count_guardrail_passed",
            usable=usable_count,
            floor=settings.source_count_floor,
        )

    audit = _make_audit_record(request_id, {
        "total_scored_sources": total_scored,
        "usable_source_count": usable_count,
        "source_count_floor": settings.source_count_floor,
        "ceiling_applied": ceiling,
    })

    return {
        "source_count_ceiling": ceiling,
        "audit_trail": state["audit_trail"] + [audit],
    }