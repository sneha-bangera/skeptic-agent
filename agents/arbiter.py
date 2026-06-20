"""
Arbiter / Orchestrator agent — evaluates Skeptic challenges, issues rulings,
sets final confidence, and controls loop termination.

CRITICAL DESIGN RULE: Loop termination (loop_round >= max_loop_rounds) is a
Python `if` statement in the graph's conditional edge function, NOT an LLM
instruction. LLMs cannot reliably count to 3.

Routing decisions (returned as the node's output, consumed by the graph edge):
  "revise"   → route back to Synthesis (loop_round incremented here)
  "finalize" → route forward to Confidence Floor guardrail then Egress

Hard termination rules (code-level — never delegated to LLM):
  1. loop_round >= max_loop_rounds         → force "finalize"
  2. source_count_ceiling is set           → enforce confidence ceiling
  3. irreconcilable contradiction exists   → cannot output "High" confidence
  4. loop_round >= max_loop_rounds AND critical challenges remain → "Insufficient Evidence"

Per-challenge rulings (LLM-evaluated):
  valid_critical  → instruct Synthesis to revise, route back
  valid_minor     → note in unresolved_contradictions, do NOT force revision
  invalid         → dismiss with reasoning
  dismissed       → challenge not applicable given available evidence

LLM usage: One call per loop round covering all unresolved challenges from that round.
Writes to state: arbiter_rulings, loop_terminated, loop_round (incremented on revise),
                 final_answer, confidence, unresolved_contradictions, audit_trail
Routing signal: state["_arbiter_route"] = "revise" | "finalize"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from core.state import PipelineState, ConfidenceTier
from services.llm import call_llm

logger = structlog.get_logger(__name__)

# Valid ruling types 
_VALID_RULINGS = frozenset({"valid_critical", "valid_minor", "invalid", "dismissed"})

_TIER_ORDER: dict[ConfidenceTier, int] = {
    "High": 3, "Moderate": 2, "Low": 1, "Insufficient Evidence": 0
}


# Prompts 

_ARBITER_SYSTEM = """You are a senior research editor adjudicating peer reviewer challenges against a preliminary research answer.

For each challenge provided, issue one of these rulings:
- "valid_critical": The challenge identifies a genuine, material problem that would change the conclusion. Synthesis must revise.
- "valid_minor": The challenge raises a real but non-material issue. Note it as a limitation but do not require revision.
- "invalid": The existing evidence in the sources already addresses the challenge. Explain why the challenge fails.
- "dismissed": The challenge is too vague, speculative, or not supported by the available evidence.

Return ONLY a valid JSON object — no markdown, no preamble:
{
  "rulings": [
    {
      "challenge_id": "<challenge_id from input>",
      "ruling": "<valid_critical|valid_minor|invalid|dismissed>",
      "reason": "<one to two sentences explaining the ruling, citing specific sources or evidence>"
    }
  ]
}

Rules:
- Every challenge in the input must receive exactly one ruling.
- "valid_critical" should only be used when the challenge would materially alter the answer's conclusion.
- "invalid" requires citing specific evidence in the sources that rebuts the challenge.
- Be rigorous: do not dismiss valid concerns just because they are inconvenient."""


def _build_arbiter_prompt(
    question: str,
    preliminary_answer: str,
    challenges_this_round: list[dict],
    scored_sources: list[dict],
    contradictions: list[dict],
    loop_round: int,
) -> str:
    lines = [
        f"Research question: {question}\n",
        f"Loop round: {loop_round}\n",
        "PRELIMINARY ANSWER:",
        preliminary_answer,
        "\nCHALLENGES TO ADJUDICATE:\n",
    ]

    for ch in challenges_this_round:
        lines.append(
            f"Challenge [{ch['challenge_id']}] — type={ch['type']} | severity={ch['severity']}\n"
            f"  Description: {ch['description']}\n"
            f"  Implicated sources: {', '.join(ch.get('source_ids_implicated', []))}\n"
            f"  Suggested revision: {ch.get('suggested_revision', 'none')}\n"
        )

    if scored_sources:
        lines.append("\nSOURCE CREDIBILITY REFERENCE:")
        for src in scored_sources:
            lines.append(
                f"  [{src['source_id']}] type={src.get('source_type','?')} | "
                f"cred={src.get('credibility',0):.2f} | "
                f"rel={src.get('relevance',0):.2f} | "
                f"bias={src.get('bias_flag','none')}"
            )

    if contradictions:
        lines.append("\nACTIVE CONTRADICTIONS:")
        for con in contradictions:
            lines.append(
                f"  [{con.get('conflict_id')}] {con.get('conflict_type')} — "
                f"{con.get('explanation','')}"
            )

    return "\n".join(lines)


# Parse arbiter rulings 

def _parse_rulings(
    raw: dict,
    challenges_this_round: list[dict],
    loop_round: int,
) -> list[dict]:
    """
    Parse and validate arbiter rulings.
    Every challenge must get a ruling; missing ones default to "dismissed".
    """
    raw_rulings = raw.get("rulings", [])
    if not isinstance(raw_rulings, list):
        raw_rulings = []

    ruled_ids = set()
    result: list[dict] = []

    for item in raw_rulings:
        if not isinstance(item, dict):
            continue
        challenge_id = item.get("challenge_id", "")
        ruling = item.get("ruling", "dismissed")
        reason = (item.get("reason", "") or "").strip()

        if ruling not in _VALID_RULINGS:
            ruling = "dismissed"

        result.append({
            "challenge_id": challenge_id,
            "ruling": ruling,
            "reason": reason,
            "round": loop_round,
        })
        ruled_ids.add(challenge_id)

    # Ensure every input challenge has a ruling (fallback to dismissed)
    for ch in challenges_this_round:
        cid = ch["challenge_id"]
        if cid not in ruled_ids:
            result.append({
                "challenge_id": cid,
                "ruling": "dismissed",
                "reason": "No ruling returned by LLM; defaulting to dismissed.",
                "round": loop_round,
            })

    return result


# Confidence finalization 

def _apply_hard_confidence_rules(
    confidence: Optional[ConfidenceTier],
    contradictions: list[dict],
    source_count_ceiling: Optional[ConfidenceTier],
    unresolved_critical_count: int,
    loop_round: int,
    max_loop_rounds: int,
) -> ConfidenceTier:
    """
    Apply code-level confidence rules after the loop terminates.
    These CANNOT be overridden by the LLM.
    """
    current: ConfidenceTier = confidence or "Low"

    # Rule 1: irreconcilable conflict → cannot be High
    has_irreconcilable = any(
        c.get("conflict_type") == "irreconcilable" for c in contradictions
    )
    if has_irreconcilable and _TIER_ORDER.get(current, 0) >= _TIER_ORDER["High"]:
        current = "Moderate"

    # Rule 2: max rounds hit AND critical challenges unresolved → Insufficient Evidence
    if loop_round >= max_loop_rounds and unresolved_critical_count > 0:
        current = "Insufficient Evidence"

    # Rule 3: source_count_ceiling is a hard cap
    if source_count_ceiling is not None:
        ceiling_order = _TIER_ORDER.get(source_count_ceiling, 0)
        if _TIER_ORDER.get(current, 0) > ceiling_order:
            current = source_count_ceiling

    return current


# Unresolved contradictions builder 

def _build_unresolved_contradictions(
    contradictions: list[dict],
    claims_lookup: dict[str, dict],
    valid_minor_challenge_ids: set[str],
    challenges_lookup: dict[str, dict],
) -> list[str]:
    """
    Build human-readable unresolved contradiction descriptions.
    Includes:
    - irreconcilable and direct_conflict contradictions
    - valid_minor challenges that were noted but not revised
    """
    descriptions: list[str] = []

    for con in contradictions:
        if con.get("conflict_type") in ("irreconcilable", "direct_conflict"):
            ca = claims_lookup.get(con.get("claim_a_id", ""), {})
            cb = claims_lookup.get(con.get("claim_b_id", ""), {})
            descriptions.append(
                f"{con['conflict_type'].upper()}: "
                f"[{con.get('claim_a_id')} from {ca.get('source_id','?')}] "
                f"vs [{con.get('claim_b_id')} from {cb.get('source_id','?')}] — "
                f"{con.get('explanation','')}"
            )

    for cid in valid_minor_challenge_ids:
        ch = challenges_lookup.get(cid, {})
        if ch:
            descriptions.append(
                f"NOTED (minor): {ch.get('description', '')} "
                f"(sources: {', '.join(ch.get('source_ids_implicated', []))})"
            )

    return descriptions


# Audit helper 

def _audit(request_id: str, event: str, data: dict) -> dict:
    return {
        "event": event,
        "stage": "arbiter",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_arbiter(state: PipelineState) -> dict:
    """
    LangGraph node function for the Arbiter agent.

    Reads:  state["skeptic_challenges"], state["arbiter_rulings"],
            state["preliminary_answer"], state["scored_sources"],
            state["contradictions"], state["claims"],
            state["confidence"], state["loop_round"],
            state["source_count_ceiling"], state["request_id"]
    Writes: arbiter_rulings (appended), loop_round (incremented on revise),
            loop_terminated, final_answer, confidence,
            unresolved_contradictions, _arbiter_route, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    question = state["question"]
    loop_round = state.get("loop_round", 0)
    max_loop_rounds = settings.max_loop_rounds
    preliminary_answer = state.get("preliminary_answer") or ""
    scored_sources = state.get("scored_sources", [])
    contradictions = state.get("contradictions", [])
    claims = state.get("claims", [])
    source_count_ceiling = state.get("source_count_ceiling")
    existing_rulings = state.get("arbiter_rulings", [])
    existing_challenges = state.get("skeptic_challenges", [])
    current_confidence = state.get("confidence")

    log = logger.bind(stage="arbiter", request_id=request_id, round=loop_round)
    log.info(
        "arbiter_start",
        round=loop_round,
        max_rounds=max_loop_rounds,
        total_challenges=len(existing_challenges),
    )

    # 1. HARD TERMINATION CHECK (code-level) 
    # This is a Python if statement — NOT an LLM instruction.
    if loop_round >= max_loop_rounds:
        log.warning(
            "arbiter_max_rounds_reached",
            round=loop_round,
            max=max_loop_rounds,
        )
        # Count unresolved critical challenges
        already_ruled_ids = {r["challenge_id"] for r in existing_rulings}
        unresolved_critical = [
            c for c in existing_challenges
            if c.get("severity") == "critical"
            and c["challenge_id"] not in already_ruled_ids
        ]

        claims_lookup = {c["claim_id"]: c for c in claims}
        challenges_lookup = {c["challenge_id"]: c for c in existing_challenges}

        final_confidence = _apply_hard_confidence_rules(
            confidence=current_confidence,
            contradictions=contradictions,
            source_count_ceiling=source_count_ceiling,
            unresolved_critical_count=len(unresolved_critical),
            loop_round=loop_round,
            max_loop_rounds=max_loop_rounds,
        )

        unresolved = _build_unresolved_contradictions(
            contradictions=contradictions,
            claims_lookup=claims_lookup,
            valid_minor_challenge_ids=set(),
            challenges_lookup=challenges_lookup,
        )

        audit = _audit(request_id, "arbiter_max_rounds_forced_finalize", {
            "round": loop_round,
            "max_rounds": max_loop_rounds,
            "unresolved_critical": len(unresolved_critical),
            "final_confidence": final_confidence,
        })

        return {
            "loop_terminated": True,
            "loop_round": loop_round,
            "final_answer": preliminary_answer,
            "confidence": final_confidence,
            "unresolved_contradictions": unresolved,
            "_arbiter_route": "finalize",
            "arbiter_rulings": existing_rulings,
            "audit_trail": state["audit_trail"] + [audit],
        }

    # 2. Identify challenges from THIS round that haven't been ruled on 
    already_ruled_ids = {r["challenge_id"] for r in existing_rulings}
    challenges_this_round = [
        c for c in existing_challenges
        if c.get("round") == loop_round
        and c["challenge_id"] not in already_ruled_ids
    ]

    log.info(
        "arbiter_challenges_this_round",
        count=len(challenges_this_round),
        round=loop_round,
    )

    # 3. If no challenges this round → finalize 
    if not challenges_this_round:
        log.info("arbiter_no_challenges_finalizing")

        claims_lookup = {c["claim_id"]: c for c in claims}
        challenges_lookup = {c["challenge_id"]: c for c in existing_challenges}

        # Count all-time unresolved criticals
        all_ruled_ids = {r["challenge_id"] for r in existing_rulings}
        unresolved_critical_count = len([
            c for c in existing_challenges
            if c.get("severity") == "critical"
            and c["challenge_id"] not in all_ruled_ids
        ])

        final_confidence = _apply_hard_confidence_rules(
            confidence=current_confidence,
            contradictions=contradictions,
            source_count_ceiling=source_count_ceiling,
            unresolved_critical_count=unresolved_critical_count,
            loop_round=loop_round,
            max_loop_rounds=max_loop_rounds,
        )

        valid_minor_ids = {
            r["challenge_id"] for r in existing_rulings
            if r.get("ruling") == "valid_minor"
        }
        unresolved = _build_unresolved_contradictions(
            contradictions=contradictions,
            claims_lookup=claims_lookup,
            valid_minor_challenge_ids=valid_minor_ids,
            challenges_lookup=challenges_lookup,
        )

        audit = _audit(request_id, "arbiter_no_challenges_finalize", {
            "round": loop_round,
            "final_confidence": final_confidence,
        })

        return {
            "loop_terminated": True,
            "loop_round": loop_round,
            "final_answer": preliminary_answer,
            "confidence": final_confidence,
            "unresolved_contradictions": unresolved,
            "_arbiter_route": "finalize",
            "arbiter_rulings": existing_rulings,
            "audit_trail": state["audit_trail"] + [audit],
        }

    # 4. LLM ruling on challenges 
    user_prompt = _build_arbiter_prompt(
        question=question,
        preliminary_answer=preliminary_answer,
        challenges_this_round=challenges_this_round,
        scored_sources=scored_sources,
        contradictions=contradictions,
        loop_round=loop_round,
    )

    try:
        raw = call_llm(
            system_prompt=_ARBITER_SYSTEM,
            user_prompt=user_prompt,
            call_type="arbiter",
            stage="arbiter",
            request_id=request_id,
            max_tokens=1200,
            temperature=0.0,
        )
        new_rulings = _parse_rulings(raw, challenges_this_round, loop_round)
    except Exception as exc:
        log.error("arbiter_llm_failed", error=str(exc))
        # On LLM failure: dismiss all challenges from this round conservatively
        new_rulings = [
            {
                "challenge_id": c["challenge_id"],
                "ruling": "dismissed",
                "reason": f"Arbiter LLM failed: {exc}. Defaulting to dismissed.",
                "round": loop_round,
            }
            for c in challenges_this_round
        ]

    all_rulings = existing_rulings + new_rulings

    # 5. Routing decision 
    valid_critical_this_round = [
        r for r in new_rulings if r.get("ruling") == "valid_critical"
    ]
    valid_minor_this_round = [
        r for r in new_rulings if r.get("ruling") == "valid_minor"
    ]

    # Build lookups for output
    claims_lookup = {c["claim_id"]: c for c in claims}
    challenges_lookup = {c["challenge_id"]: c for c in existing_challenges}
    all_valid_minor_ids = {
        r["challenge_id"] for r in all_rulings if r.get("ruling") == "valid_minor"
    }

    if valid_critical_this_round:
        # Route back to Synthesis — increment loop_round HERE (code-level)
        new_loop_round = loop_round + 1

        log.info(
            "arbiter_routing_revise",
            critical_challenges=len(valid_critical_this_round),
            new_round=new_loop_round,
        )

        audit = _audit(request_id, "arbiter_route_revise", {
            "round": loop_round,
            "new_round": new_loop_round,
            "critical_ruling_count": len(valid_critical_this_round),
            "minor_ruling_count": len(valid_minor_this_round),
            "rulings": [
                {"id": r["challenge_id"], "ruling": r["ruling"]}
                for r in new_rulings
            ],
        })

        return {
            "arbiter_rulings": all_rulings,
            "loop_round": new_loop_round,
            "loop_terminated": False,
            "_arbiter_route": "revise",
            # Don't set final_answer yet — Synthesis will revise
            "audit_trail": state["audit_trail"] + [audit],
        }

    else:
        # No critical issues — finalize
        all_ruled_ids = {r["challenge_id"] for r in all_rulings}
        unresolved_critical_count = len([
            c for c in existing_challenges
            if c.get("severity") == "critical"
            and c["challenge_id"] not in all_ruled_ids
        ])

        final_confidence = _apply_hard_confidence_rules(
            confidence=current_confidence,
            contradictions=contradictions,
            source_count_ceiling=source_count_ceiling,
            unresolved_critical_count=unresolved_critical_count,
            loop_round=loop_round,
            max_loop_rounds=max_loop_rounds,
        )

        unresolved = _build_unresolved_contradictions(
            contradictions=contradictions,
            claims_lookup=claims_lookup,
            valid_minor_challenge_ids=all_valid_minor_ids,
            challenges_lookup=challenges_lookup,
        )

        log.info(
            "arbiter_routing_finalize",
            final_confidence=final_confidence,
            minor_noted=len(valid_minor_this_round),
            unresolved_count=len(unresolved),
        )

        audit = _audit(request_id, "arbiter_route_finalize", {
            "round": loop_round,
            "final_confidence": final_confidence,
            "minor_ruling_count": len(valid_minor_this_round),
            "unresolved_count": len(unresolved),
            "rulings": [
                {"id": r["challenge_id"], "ruling": r["ruling"]}
                for r in new_rulings
            ],
        })

        return {
            "arbiter_rulings": all_rulings,
            "loop_round": loop_round,
            "loop_terminated": True,
            "final_answer": preliminary_answer,
            "confidence": final_confidence,
            "unresolved_contradictions": unresolved,
            "_arbiter_route": "finalize",
            "audit_trail": state["audit_trail"] + [audit],
        }