"""
FastAPI route definitions for The Skeptic's Brief.

Endpoints:
  POST /analyze                  — Submit a question + sources, run the pipeline
  GET  /status/{request_id}      — Poll current pipeline state
  POST /feedback/{request_id}    — Submit rating + source corrections
  POST /approve/{request_id}     — HITL: approve or return for revision
  GET  /replay/{request_id}      — Retrieve full audit trail for a past decision

Design notes:
  - /analyze uses an in-memory idempotency cache keyed by MD5(question + sources).
    Identical requests within the same process return the cached response instantly.
  - Pipeline execution is synchronous inside an async route via asyncio.get_event_loop()
    + run_in_executor so the FastAPI event loop is not blocked.
  - All pipeline state is held in-memory during the request; only the audit trail
    and replay record are persisted to SQLite.
  - HITL: responses with confidence below the threshold set hitl_required=True in state.
    The /approve endpoint writes the approval decision back to an in-memory HITL registry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from schemas.request import AnalyzeRequest
from schemas.response import (
    APIResponse,
    StatusResponse,
    FeedbackResponse,
    ApprovalResponse,
    ReplayResponse,
    CitationEntry,
    SkepticChallengeSummary,
)
from api.dependencies import get_compiled_graph, get_db_path
from storage.audit_db import (
    read_replay_record,
    read_audit_trail,
    write_feedback,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# In-memory stores (per-process)    

# Idempotency cache: request_hash → APIResponse dict
_IDEMPOTENCY_CACHE: dict[str, dict] = {}

# Pipeline state cache: request_id → final PipelineState
_STATE_CACHE: dict[str, dict] = {}

# HITL pending registry: request_id → {"approved": bool|None, "notes": str|None}
_HITL_REGISTRY: dict[str, dict] = {}

# Thread pool for running the synchronous LangGraph pipeline in async context
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")


# Helpers   

def _make_request_hash(question: str, sources: list) -> str:
    """Stable MD5 hash of question + source URLs/content (first 200 chars each)."""
    fingerprint = json.dumps({
        "question": question.strip().lower(),
        "sources": [
            {"url": s.url or "", "content": (s.content or "")[:200]}
            for s in sources
        ],
    }, sort_keys=True)
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _should_trigger_hitl(confidence: Optional[str], settings) -> bool:
    """Return True if confidence is below the HITL threshold."""
    tier_order = {"High": 3, "Moderate": 2, "Low": 1, "Insufficient Evidence": 0}
    threshold = settings.hitl_confidence_threshold
    current = confidence or "Insufficient Evidence"
    return tier_order.get(current, 0) < tier_order.get(threshold, 2)


def _state_to_api_response(state: dict) -> dict:
    """
    Extract the formatted api_response from pipeline state.
    Falls back to constructing a minimal response if egress didn't populate it.
    """
    if state.get("api_response"):
        return state["api_response"]

    # Fallback: construct minimal response from raw state fields
    return {
        "request_id": state.get("request_id", "unknown"),
        "question": state.get("question", ""),
        "answer": state.get("final_answer"),
        "confidence": state.get("confidence"),
        "citations": state.get("claim_source_map", []),
        "unresolved_contradictions": state.get("unresolved_contradictions", []),
        "disclaimer": state.get("disclaimer"),
        "sources_used": len(state.get("scored_sources", [])),
        "sources_dropped": len(state.get("dropped_sources", [])),
        "loop_rounds": state.get("loop_round", 0),
        "skeptic_challenges": [],
        "hitl_status": "not_required",
        "status": "complete",
        "error_message": state.get("input_rejection_reason"),
    }


def _run_pipeline_sync(graph, request_id: str, question: str, raw_sources: list) -> dict:
    """
    Run the LangGraph pipeline synchronously.
    Called from a thread pool to avoid blocking the event loop.
    """
    from core.state import make_initial_state
    initial_state = make_initial_state(request_id, question, raw_sources)
    return graph.invoke(initial_state)


# POST /analyze    

@router.post(
    "/analyze",
    response_model=APIResponse,
    summary="Analyze a research question against a corpus of sources",
    description=(
        "Runs the full Skeptic's Brief pipeline: ingest → score → extract → "
        "contradict → synthesize → challenge → adjudicate → format. "
        "Returns a confidence-tiered answer with citations and adversarial audit trail."
    ),
)
async def analyze(
    body: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    graph=Depends(get_compiled_graph),
    db_path: str = Depends(get_db_path),
) -> APIResponse:
    from config.settings import get_settings
    settings = get_settings()

    # Idempotency check  ─
    request_hash = _make_request_hash(body.question, body.sources)
    if request_hash in _IDEMPOTENCY_CACHE:
        logger.info("analyze_cache_hit", hash=request_hash[:8])
        cached = _IDEMPOTENCY_CACHE[request_hash]
        return APIResponse(**cached)

    # Generate request ID  
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.info(
        "analyze_start",
        request_id=request_id,
        question_len=len(body.question),
        source_count=len(body.sources),
    )

    # Convert request to raw_sources format   
    raw_sources = [
        {
            "url": s.url,
            "content": s.content,
            "metadata": s.metadata,
        }
        for s in body.sources
    ]

    # Run pipeline in thread pool (non-blocking)  
    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            _EXECUTOR,
            _run_pipeline_sync,
            graph,
            request_id,
            body.question,
            raw_sources,
        )
    except Exception as exc:
        logger.error("analyze_pipeline_error", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")

    # Cache state for /status polling  
    _STATE_CACHE[request_id] = final_state

    # Check HITL threshold  ─
    confidence = final_state.get("confidence")
    hitl_required = _should_trigger_hitl(confidence, settings)

    if hitl_required:
        _HITL_REGISTRY[request_id] = {"approved": None, "notes": None}
        logger.info(
            "hitl_triggered",
            request_id=request_id,
            confidence=confidence,
        )

    # Build API response  
    raw_response = _state_to_api_response(final_state)

    # Override hitl_status from HITL registry
    if hitl_required:
        raw_response["hitl_status"] = "pending"
        raw_response["status"] = "hitl_pending"

    # Ensure request_id matches (pipeline may have generated its own)
    raw_response["request_id"] = request_id

    # Validate and return  
    try:
        api_response = APIResponse(**raw_response)
    except Exception as exc:
        logger.error("response_validation_error", request_id=request_id, error=str(exc))
        # Return a safe fallback response
        api_response = APIResponse(
            request_id=request_id,
            question=body.question,
            answer=final_state.get("final_answer"),
            confidence=final_state.get("confidence"),
            status="complete",
        )

    # Cache for idempotency  
    _IDEMPOTENCY_CACHE[request_hash] = api_response.model_dump()

    logger.info(
        "analyze_complete",
        request_id=request_id,
        confidence=api_response.confidence,
        status=api_response.status,
        citations=len(api_response.citations),
    )

    return api_response


# GET /status/{request_id}  

@router.get(
    "/status/{request_id}",
    response_model=StatusResponse,
    summary="Poll the current status of a pipeline run",
)
async def get_status(request_id: str) -> StatusResponse:
    state = _STATE_CACHE.get(request_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Request '{request_id}' not found. "
                   "It may have expired or never existed.",
        )

    hitl_info = _HITL_REGISTRY.get(request_id, {})
    hitl_approved = hitl_info.get("approved")

    if hitl_info and hitl_approved is None:
        hitl_status = "pending"
        pipeline_status = "hitl_pending"
    elif hitl_info and hitl_approved is True:
        hitl_status = "approved"
        pipeline_status = "complete"
    elif hitl_info and hitl_approved is False:
        hitl_status = "returned_for_revision"
        pipeline_status = "complete"
    else:
        hitl_status = "not_required"
        pipeline_status = "complete"

    return StatusResponse(
        request_id=request_id,
        status=pipeline_status,
        confidence=state.get("confidence"),
        loop_round=state.get("loop_round", 0),
        hitl_status=hitl_status,
        message=f"Completed {state.get('loop_round', 0)} adversarial loop round(s).",
    )


# POST /feedback/{request_id}  

@router.post(
    "/feedback/{request_id}",
    response_model=FeedbackResponse,
    summary="Submit user rating and source corrections",
)
async def submit_feedback(
    request_id: str,
    body: dict,
    db_path: str = Depends(get_db_path),
) -> FeedbackResponse:
    """
    Body: {
        "rating": int (1-5, optional),
        "source_corrections": [{"source_id": str, "flag": str}] (optional)
    }
    Flag values: commercial_bias | low_credibility | outdated | trusted | misinformation
    """
    rating = body.get("rating")
    source_corrections = body.get("source_corrections", [])

    # Validate rating
    if rating is not None:
        if not isinstance(rating, int) or not (1 <= rating <= 5):
            raise HTTPException(
                status_code=422,
                detail="rating must be an integer between 1 and 5",
            )

    # Validate source_corrections
    if not isinstance(source_corrections, list):
        raise HTTPException(
            status_code=422,
            detail="source_corrections must be a list",
        )

    adjustments_applied = await write_feedback(
        db_path=db_path,
        request_id=request_id,
        rating=rating,
        source_corrections=source_corrections,
    )

    logger.info(
        "feedback_received",
        request_id=request_id,
        rating=rating,
        corrections=len(source_corrections),
        adjustments=adjustments_applied,
    )

    return FeedbackResponse(
        acknowledged=True,
        request_id=request_id,
        adjustments_applied=adjustments_applied,
    )


# POST /approve/{request_id}  

@router.post(
    "/approve/{request_id}",
    response_model=ApprovalResponse,
    summary="HITL: approve or return a response for revision",
)
async def approve_response(
    request_id: str,
    body: dict,
) -> ApprovalResponse:
    """
    Body: {
        "approved": bool,
        "reviewer_notes": str (optional)
    }
    Only applicable when the response is in 'hitl_pending' status.
    """
    if request_id not in _HITL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"No HITL review pending for request '{request_id}'. "
                   "Either it does not exist or HITL was not triggered.",
        )

    approved = body.get("approved")
    if not isinstance(approved, bool):
        raise HTTPException(
            status_code=422,
            detail="'approved' field must be a boolean",
        )

    reviewer_notes = body.get("reviewer_notes", "")

    _HITL_REGISTRY[request_id]["approved"] = approved
    _HITL_REGISTRY[request_id]["notes"] = reviewer_notes

    # Update state cache if it exists
    if request_id in _STATE_CACHE:
        _STATE_CACHE[request_id]["hitl_approved"] = approved
        _STATE_CACHE[request_id]["hitl_reviewer_notes"] = reviewer_notes

    status = "approved" if approved else "returned_for_revision"

    logger.info(
        "hitl_decision",
        request_id=request_id,
        approved=approved,
        has_notes=bool(reviewer_notes),
    )

    return ApprovalResponse(
        request_id=request_id,
        status=status,
        reviewer_notes=reviewer_notes or None,
    )


# GET /replay/{request_id}  

@router.get(
    "/replay/{request_id}",
    response_model=ReplayResponse,
    summary="Retrieve the full audit trail for a past decision",
)
async def get_replay(
    request_id: str,
    db_path: str = Depends(get_db_path),
) -> ReplayResponse:
    """
    Returns the complete audit trail and pipeline metadata for a past request.
    Data is read from the SQLite audit database.
    """
    replay = await read_replay_record(db_path, request_id)

    if not replay:
        raise HTTPException(
            status_code=404,
            detail=f"No replay record found for request '{request_id}'.",
        )

    audit_trail = await read_audit_trail(db_path, request_id)

    return ReplayResponse(
        request_id=replay["request_id"],
        question=replay["question"],
        final_answer=replay.get("final_answer"),
        confidence=replay.get("confidence"),
        timestamp=replay["timestamp"],
        audit_trail=audit_trail,
        skeptic_challenges=replay.get("skeptic_challenges", []),
        arbiter_rulings=replay.get("arbiter_rulings", []),
    )


# Health check  

@router.get("/health", include_in_schema=False)
async def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cached_requests": len(_STATE_CACHE),
        "hitl_pending": sum(
            1 for v in _HITL_REGISTRY.values() if v.get("approved") is None
        ),
    }