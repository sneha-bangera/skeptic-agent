"""
Audit agent — async post-decision logging. Last node before END.

Responsibilities:
1. Write the full audit_trail to the SQLite audit_log table.
2. Write a replay record: {request_id, question, sources_hash, final_answer,
   confidence, timestamp, skeptic_challenges, arbiter_rulings}.
3. Load any persisted credibility adjustments from the DB into memory
   (so scorer adjustments from previous sessions are applied).

The audit agent never crashes the pipeline — all DB errors are caught and
logged. If the DB is unavailable, the response is still returned to the user.

LLM usage: None. Pure I/O.
Writes to state: nothing (audit is fire-and-forget; state is already final).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone

import structlog

from core.state import PipelineState
from storage.audit_db import (
    init_db,
    write_audit_trail,
    write_replay_record,
    load_adjustments_from_db,
)

logger = structlog.get_logger(__name__)


def _hash_sources(raw_sources: list[dict]) -> str:
    """Stable hash of source URLs+content for replay deduplication."""
    fingerprint = json.dumps(
        [{"url": s.get("url", ""), "content": s.get("content", "")[:200]}
         for s in raw_sources],
        sort_keys=True,
    )
    return hashlib.md5(fingerprint.encode()).hexdigest()


async def _run_audit_async(state: PipelineState, db_path: str) -> None:
    """Async audit writes — all errors are caught and logged, never raised."""
    request_id = state["request_id"]
    log = logger.bind(stage="audit", request_id=request_id)

    try:
        await init_db(db_path)
    except Exception as exc:
        log.error("audit_db_init_failed", error=str(exc))
        return

    audit_trail = state.get("audit_trail", [])
    try:
        await write_audit_trail(db_path, request_id, audit_trail)
    except Exception as exc:
        log.error("audit_trail_write_failed", error=str(exc))

    try:
        sources_hash = _hash_sources(state.get("raw_sources", []))
        await write_replay_record(
            db_path=db_path,
            request_id=request_id,
            question=state.get("question", ""),
            sources_hash=sources_hash,
            final_answer=state.get("final_answer"),
            confidence=state.get("confidence"),
            loop_rounds=state.get("loop_round", 0),
            skeptic_challenges=state.get("skeptic_challenges", []),
            arbiter_rulings=state.get("arbiter_rulings", []),
        )
    except Exception as exc:
        log.error("replay_record_write_failed", error=str(exc))

    try:
        await load_adjustments_from_db(db_path)
    except Exception as exc:
        log.error("adjustments_load_failed", error=str(exc))

    log.info(
        "audit_complete",
        audit_records=len(audit_trail),
        confidence=state.get("confidence"),
    )


def run_audit(state: PipelineState) -> dict:
    """
    LangGraph node function for the Audit agent.
    Runs async DB writes synchronously via asyncio.run() or get_event_loop().
    Returns an empty dict — audit does not modify pipeline state.
    """
    from config.settings import get_settings
    settings = get_settings()
    db_path = settings.audit_db_path

    try:
        # Try to get a running event loop (e.g. inside FastAPI)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule as a background task — don't await here
            loop.create_task(_run_audit_async(state, db_path))
        else:
            loop.run_until_complete(_run_audit_async(state, db_path))
    except RuntimeError:
        # No event loop at all — create one
        asyncio.run(_run_audit_async(state, db_path))
    except Exception as exc:
        logger.error("audit_node_failed", error=str(exc))

    return {}