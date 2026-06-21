"""
SQLite audit database via aiosqlite.

Schema (3 tables):
  audit_log      — every agent step record from state["audit_trail"]
  replay_records — one row per completed request (for GET /replay)
  feedback       — user ratings and source corrections

All writes are async (aiosqlite). Reads are also async.
The DB file is created automatically on first use.

Credibility adjustments from source corrections are applied to an in-memory
registry that the Source Scorer checks on subsequent calls. Corrections are
also persisted to the feedback table so they survive restarts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# In-memory credibility adjustment registry 
# Maps domain → adjustment_delta (float, -1.0 to 1.0)
# Used by the Source Scorer on the next request to adjust base scores.
_CREDIBILITY_ADJUSTMENTS: dict[str, float] = {}

# Schema SQL 

_CREATE_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    event           TEXT NOT NULL,
    stage           TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_log_request_id ON audit_log(request_id);
"""

_CREATE_REPLAY_RECORDS = """
CREATE TABLE IF NOT EXISTS replay_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT UNIQUE NOT NULL,
    question        TEXT NOT NULL,
    sources_hash    TEXT,
    final_answer    TEXT,
    confidence      TEXT,
    loop_rounds     INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL,
    skeptic_json    TEXT NOT NULL DEFAULT '[]',
    arbiter_json    TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    rating          INTEGER,
    source_id       TEXT,
    flag            TEXT,
    domain          TEXT,
    adjustment      REAL DEFAULT 0.0,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id);
"""


# DB initializer 

async def init_db(db_path: str) -> None:
    """
    Create the database file and all tables if they don't exist.
    Safe to call multiple times (idempotent).
    """
    try:
        import aiosqlite
    except ImportError:
        logger.warning("aiosqlite_not_installed", hint="pip install aiosqlite")
        return

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_CREATE_AUDIT_LOG)
        await db.executescript(_CREATE_REPLAY_RECORDS)
        await db.executescript(_CREATE_FEEDBACK)
        await db.commit()

    logger.info("audit_db_initialized", path=db_path)


# Audit trail writer 

async def write_audit_trail(
    db_path: str,
    request_id: str,
    audit_trail: list[dict],
) -> int:
    """
    Write all audit_trail records for a request to the audit_log table.
    Returns the number of records written.
    """
    try:
        import aiosqlite
    except ImportError:
        logger.warning("aiosqlite_not_installed")
        return 0

    if not audit_trail:
        return 0

    rows = [
        (
            request_id,
            record.get("event", "unknown"),
            record.get("stage", "unknown"),
            record.get("timestamp", datetime.now(timezone.utc).isoformat()),
            json.dumps(record.get("data", {}), default=str),
        )
        for record in audit_trail
    ]

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.executemany(
                "INSERT INTO audit_log (request_id, event, stage, timestamp, data_json) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()
        logger.info("audit_trail_written", request_id=request_id, count=len(rows))
        return len(rows)
    except Exception as exc:
        logger.error("audit_trail_write_failed", request_id=request_id, error=str(exc))
        return 0


# Replay record writer 

async def write_replay_record(
    db_path: str,
    request_id: str,
    question: str,
    sources_hash: Optional[str],
    final_answer: Optional[str],
    confidence: Optional[str],
    loop_rounds: int,
    skeptic_challenges: list[dict],
    arbiter_rulings: list[dict],
) -> bool:
    """
    Write or replace a replay record for a completed request.
    Returns True on success.
    """
    try:
        import aiosqlite
    except ImportError:
        logger.warning("aiosqlite_not_installed")
        return False

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO replay_records
                (request_id, question, sources_hash, final_answer, confidence,
                 loop_rounds, timestamp, skeptic_json, arbiter_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    question,
                    sources_hash,
                    final_answer,
                    confidence,
                    loop_rounds,
                    timestamp,
                    json.dumps(skeptic_challenges, default=str),
                    json.dumps(arbiter_rulings, default=str),
                ),
            )
            await db.commit()
        logger.info("replay_record_written", request_id=request_id)
        return True
    except Exception as exc:
        logger.error("replay_record_write_failed", request_id=request_id, error=str(exc))
        return False


# Replay record reader 

async def read_replay_record(db_path: str, request_id: str) -> Optional[dict]:
    """
    Read a replay record for a given request_id.
    Returns None if not found.
    """
    try:
        import aiosqlite
    except ImportError:
        return None

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM replay_records WHERE request_id = ?",
                (request_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        return {
            "request_id": row["request_id"],
            "question": row["question"],
            "final_answer": row["final_answer"],
            "confidence": row["confidence"],
            "loop_rounds": row["loop_rounds"],
            "timestamp": row["timestamp"],
            "skeptic_challenges": json.loads(row["skeptic_json"] or "[]"),
            "arbiter_rulings": json.loads(row["arbiter_json"] or "[]"),
        }
    except Exception as exc:
        logger.error("replay_record_read_failed", request_id=request_id, error=str(exc))
        return None


# Audit trail reader 

async def read_audit_trail(db_path: str, request_id: str) -> list[dict]:
    """
    Read all audit_log records for a request_id, ordered by id (insertion order).
    """
    try:
        import aiosqlite
    except ImportError:
        return []

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT event, stage, timestamp, data_json FROM audit_log "
                "WHERE request_id = ? ORDER BY id ASC",
                (request_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "event": row["event"],
                "stage": row["stage"],
                "timestamp": row["timestamp"],
                "data": json.loads(row["data_json"] or "{}"),
            }
            for row in rows
        ]
    except Exception as exc:
        logger.error("audit_trail_read_failed", request_id=request_id, error=str(exc))
        return []


# Feedback writer 

async def write_feedback(
    db_path: str,
    request_id: str,
    rating: Optional[int],
    source_corrections: list[dict],
) -> int:
    """
    Write user feedback (rating + source corrections) to the feedback table.
    Also applies credibility adjustments to the in-memory registry.

    source_corrections: list of {"source_id": str, "flag": str}
      flag values: "commercial_bias" | "low_credibility" | "trusted" | "outdated"

    Returns the number of corrections applied to the in-memory registry.
    """
    try:
        import aiosqlite
    except ImportError:
        logger.warning("aiosqlite_not_installed")
        return 0

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    adjustments_applied = 0

    # Rating row (no source)
    if rating is not None:
        rows.append((request_id, rating, None, None, None, 0.0, timestamp))

    # Source correction rows
    _FLAG_ADJUSTMENTS: dict[str, float] = {
        "commercial_bias": -0.20,
        "low_credibility": -0.15,
        "outdated": -0.10,
        "trusted": +0.10,
        "misinformation": -0.40,
    }

    for correction in source_corrections:
        source_id = correction.get("source_id", "")
        flag = correction.get("flag", "")
        delta = _FLAG_ADJUSTMENTS.get(flag, 0.0)

        # Apply to in-memory registry (keyed by source_id for this session)
        if delta != 0.0 and source_id:
            current = _CREDIBILITY_ADJUSTMENTS.get(source_id, 0.0)
            # Clamp the cumulative adjustment to [-0.50, +0.20]
            _CREDIBILITY_ADJUSTMENTS[source_id] = max(-0.50, min(0.20, current + delta))
            adjustments_applied += 1
            logger.info(
                "credibility_adjustment_applied",
                source_id=source_id,
                flag=flag,
                delta=delta,
                new_total=_CREDIBILITY_ADJUSTMENTS[source_id],
            )

        rows.append((request_id, rating, source_id, flag, None, delta, timestamp))

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.executemany(
                "INSERT INTO feedback (request_id, rating, source_id, flag, domain, adjustment, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()
        logger.info(
            "feedback_written",
            request_id=request_id,
            rating=rating,
            corrections=len(source_corrections),
            adjustments_applied=adjustments_applied,
        )
    except Exception as exc:
        logger.error("feedback_write_failed", request_id=request_id, error=str(exc))

    return adjustments_applied


# Credibility adjustment reader (for Source Scorer) 

def get_credibility_adjustment(source_id: str) -> float:
    """
    Return the current in-memory credibility adjustment for a source_id.
    Returns 0.0 if no adjustment has been applied.
    """
    return _CREDIBILITY_ADJUSTMENTS.get(source_id, 0.0)


async def load_adjustments_from_db(db_path: str) -> int:
    """
    Load persisted credibility adjustments from the feedback table into memory.
    Call this at app startup so adjustments survive restarts.
    Returns the number of adjustments loaded.
    """
    try:
        import aiosqlite
    except ImportError:
        return 0

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT source_id, SUM(adjustment) as total_adj "
                "FROM feedback WHERE source_id IS NOT NULL AND source_id != '' "
                "GROUP BY source_id"
            ) as cursor:
                rows = await cursor.fetchall()

        loaded = 0
        for row in rows:
            source_id = row["source_id"]
            total = float(row["total_adj"] or 0.0)
            if source_id and total != 0.0:
                _CREDIBILITY_ADJUSTMENTS[source_id] = max(-0.50, min(0.20, total))
                loaded += 1

        logger.info("adjustments_loaded_from_db", count=loaded)
        return loaded
    except Exception as exc:
        logger.error("adjustments_load_failed", error=str(exc))
        return 0