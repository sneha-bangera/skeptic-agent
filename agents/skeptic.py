"""
Skeptic agent — the adversarial core of the system.

Actively argues against the Synthesis agent's preliminary answer by running a
mandatory 7-item checklist. Every item on the checklist must be evaluated.
A challenge is produced only when a genuine issue is found.

Checklist (per spec):
  1. wrong_population         — cited source studies population not matching the question
  2. commercial_bias          — high/moderate bias sources driving the answer
  3. outcome_mismatch         — proxy metric or different timeframe cited for main question
  4. sample_size              — studies too small to support the conclusion
  5. recency_gap              — answer built on old sources when newer conflicting evidence exists
  6. missing_counter_evidence — high-credibility sources in corpus contradicting conclusion but not cited
  7. ignored_conflict         — Synthesis glossed over irreconcilable contradictions

The Skeptic MUST be willing to produce zero challenges if the answer is genuinely
well-supported. It must NOT manufacture challenges to extend the loop.

LLM prompt strategy:
- The Skeptic is told it is a rigorous peer reviewer at a top journal.
- The checklist is given EXPLICITLY in the prompt.
- It is instructed to cite source IDs by name — vague challenges are invalid.
- Response is structured JSON.

LLM usage: One call per loop round.
Writes to state: skeptic_challenges, audit_trail
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from core.state import PipelineState
from services.llm import call_llm

logger = structlog.get_logger(__name__)

# Valid challenge types 
_VALID_CHALLENGE_TYPES = frozenset({
    "wrong_population",
    "commercial_bias",
    "outcome_mismatch",
    "sample_size",
    "recency_gap",
    "missing_counter_evidence",
    "ignored_conflict",
})

_VALID_SEVERITIES = frozenset({"critical", "moderate", "minor"})


# Prompts 

_SKEPTIC_SYSTEM = """You are a rigorous peer reviewer at a top academic journal. Your job is to find specific, evidence-based problems with a preliminary research answer.

You MUST evaluate ALL seven items on the checklist below, even if you find no issue with some. For each item where you find a genuine problem, produce a structured challenge.

CHECKLIST — evaluate every item:
1. wrong_population: Does any cited source study a population that doesn't match the question? (e.g., Type 2 diabetics cited for a general-population question)
2. commercial_bias: Are any cited sources flagged with bias=high or bias=moderate? Are their conclusions driving the answer?
3. outcome_mismatch: Does any cited source measure a proxy metric or different timeframe than the question asks about?
4. sample_size: Are any cited studies too small (e.g., n<100 for a clinical claim) to support the conclusion?
5. recency_gap: Is the answer built on old sources when newer conflicting evidence exists in the corpus?
6. missing_counter_evidence: Are there sources in the corpus with high credibility (>0.60) that contradict the conclusion but weren't cited?
7. ignored_conflict: Did the synthesis gloss over irreconcilable or direct contradictions between sources?

RULES:
- Be SPECIFIC. Cite exact source IDs (e.g., src_001). A vague challenge like "are you sure?" is NOT acceptable and will be automatically rejected.
- Only produce a challenge if you have concrete evidence of the problem from the provided data.
- If the answer is genuinely well-supported for a checklist item, produce NO challenge for that item.
- Severity: "critical" = would materially change the conclusion; "moderate" = important caveat; "minor" = footnote-level issue.

Return ONLY a valid JSON object — no markdown, no preamble:
{
  "challenges": [
    {
      "type": "<one of the 7 types above>",
      "description": "<specific description citing source IDs and exact problem>",
      "source_ids_implicated": ["src_001", ...],
      "severity": "<critical|moderate|minor>",
      "suggested_revision": "<what Synthesis should do differently, specific>"
    }
  ]
}

If you find NO problems, return: {"challenges": []}"""


def _build_skeptic_prompt(
    question: str,
    preliminary_answer: str,
    scored_sources: list[dict],
    normalized_lookup: dict[str, dict],
    claims: list[dict],
    contradictions: list[dict],
    loop_round: int,
) -> str:
    lines = [
        f"Research question: {question}\n",
        f"Loop round: {loop_round}\n",
        "=" * 60,
        "PRELIMINARY ANSWER TO CHALLENGE:",
        preliminary_answer,
        "=" * 60,
        "\nALL AVAILABLE SOURCES (scored — including ones NOT cited in answer):\n",
    ]

    for src in scored_sources:
        sid = src["source_id"]
        norm = normalized_lookup.get(sid, {})
        title = norm.get("title") or "Untitled"
        pub_date = norm.get("published_date") or src.get("metadata", {}).get("published_date", "unknown")
        lines.append(
            f"[{sid}] {title}\n"
            f"  type={src.get('source_type','?')} | "
            f"credibility={src.get('credibility',0):.2f} | "
            f"relevance={src.get('relevance',0):.2f} | "
            f"bias={src.get('bias_flag','none')} | "
            f"published={pub_date}"
        )

    if claims:
        lines.append("\nALL EXTRACTED CLAIMS (including from uncited sources):\n")
        for c in claims:
            lines.append(
                f"  [{c['claim_id']} | {c['source_id']}] {c['text']}"
                + (f" (population={c['population']})" if c.get("population") else "")
                + (f" (outcome={c['outcome_metric']})" if c.get("outcome_metric") else "")
                + (" [LOW CREDIBILITY]" if c.get("low_credibility") else "")
            )

    if contradictions:
        lines.append("\nDETECTED CONTRADICTIONS:\n")
        for con in contradictions:
            lines.append(
                f"  [{con.get('conflict_id')}] {con.get('conflict_type','?').upper()}: "
                f"{con.get('claim_a_id')} vs {con.get('claim_b_id')} — "
                f"{con.get('explanation','')}"
            )

    return "\n".join(lines)


# Parse and validate challenges 

def _parse_challenges(
    raw: dict,
    loop_round: int,
    existing_challenge_count: int,
) -> list[dict]:
    """
    Parse and validate the Skeptic's challenge list.

    - Validates type and severity fields.
    - Rejects vague challenges (description < 30 chars or no source_ids_implicated).
    - Assigns sequential challenge IDs continuing from existing count.
    - Tags each challenge with the current loop round.
    """
    raw_challenges = raw.get("challenges", [])
    if not isinstance(raw_challenges, list):
        return []

    result: list[dict] = []
    counter = existing_challenge_count + 1

    for item in raw_challenges:
        if not isinstance(item, dict):
            continue

        challenge_type = item.get("type", "")
        description = (item.get("description", "") or "").strip()
        source_ids = item.get("source_ids_implicated", [])
        severity = item.get("severity", "minor")
        suggested_revision = (item.get("suggested_revision", "") or "").strip()

        # Validate type
        if challenge_type not in _VALID_CHALLENGE_TYPES:
            logger.warning("skeptic_invalid_challenge_type", type=challenge_type)
            continue

        # Validate severity
        if severity not in _VALID_SEVERITIES:
            severity = "minor"

        # Reject vague challenges — enforce specificity
        if len(description) < 30:
            logger.warning(
                "skeptic_vague_challenge_rejected",
                type=challenge_type,
                description=description,
            )
            continue

        if not source_ids or not isinstance(source_ids, list):
            logger.warning(
                "skeptic_no_source_ids_rejected",
                type=challenge_type,
                description=description[:60],
            )
            continue

        result.append({
            "challenge_id": f"c_{str(counter).zfill(3)}",
            "type": challenge_type,
            "description": description,
            "source_ids_implicated": [str(s) for s in source_ids],
            "severity": severity,
            "suggested_revision": suggested_revision,
            "round": loop_round,
        })
        counter += 1

    return result


# Audit helper 

def _audit(request_id: str, event: str, data: dict) -> dict:
    return {
        "event": event,
        "stage": "skeptic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_skeptic(state: PipelineState) -> dict:
    """
    LangGraph node function for the Skeptic agent.

    Reads:  state["preliminary_answer"], state["scored_sources"],
            state["normalized_sources"], state["claims"],
            state["contradictions"], state["skeptic_challenges"],
            state["loop_round"], state["request_id"]
    Writes: skeptic_challenges (appended), audit_trail
    """
    request_id = state["request_id"]
    question = state["question"]
    preliminary_answer = state.get("preliminary_answer") or ""
    scored_sources = state.get("scored_sources", [])
    normalized_sources = state.get("normalized_sources", [])
    claims = state.get("claims", [])
    contradictions = state.get("contradictions", [])
    existing_challenges = state.get("skeptic_challenges", [])
    loop_round = state.get("loop_round", 0)

    log = logger.bind(stage="skeptic", request_id=request_id, round=loop_round)
    log.info("skeptic_start", round=loop_round, answer_len=len(preliminary_answer))

    normalized_lookup = {s["source_id"]: s for s in normalized_sources}

    # Build prompt 
    user_prompt = _build_skeptic_prompt(
        question=question,
        preliminary_answer=preliminary_answer,
        scored_sources=scored_sources,
        normalized_lookup=normalized_lookup,
        claims=claims,
        contradictions=contradictions,
        loop_round=loop_round,
    )

    # LLM call 
    try:
        raw = call_llm(
            system_prompt=_SKEPTIC_SYSTEM,
            user_prompt=user_prompt,
            call_type="skeptic",
            stage="skeptic",
            request_id=request_id,
            max_tokens=1500,
            temperature=0.1,
        )
    except Exception as exc:
        log.error("skeptic_llm_failed", error=str(exc))
        audit = _audit(request_id, "skeptic_llm_failed", {
            "round": loop_round,
            "error": str(exc),
        })
        return {
            "skeptic_challenges": existing_challenges,
            "audit_trail": state["audit_trail"] + [audit],
        }

    # Parse challenges 
    new_challenges = _parse_challenges(
        raw=raw,
        loop_round=loop_round,
        existing_challenge_count=len(existing_challenges),
    )

    type_counts: dict[str, int] = {}
    for c in new_challenges:
        t = c["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    severity_counts: dict[str, int] = {}
    for c in new_challenges:
        s = c["severity"]
        severity_counts[s] = severity_counts.get(s, 0) + 1

    log.info(
        "skeptic_complete",
        round=loop_round,
        new_challenges=len(new_challenges),
        critical=severity_counts.get("critical", 0),
        type_counts=type_counts,
    )

    audit = _audit(request_id, "skeptic_complete", {
        "round": loop_round,
        "new_challenges": len(new_challenges),
        "challenge_ids": [c["challenge_id"] for c in new_challenges],
        "severity_counts": severity_counts,
        "type_counts": type_counts,
    })

    # State lists are append-only — prepend existing challenges
    return {
        "skeptic_challenges": existing_challenges + new_challenges,
        "audit_trail": state["audit_trail"] + [audit],
    }