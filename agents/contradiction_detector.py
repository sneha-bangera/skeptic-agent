"""
Contradiction Detector agent — runs after Evidence Extractor, before Synthesis.

Responsibilities:
1. Embed all claim texts using the sentence-transformer embedder service.
2. Find all claim pairs with cosine similarity > contradiction_similarity_threshold (default 0.70).
   These pairs are "topically similar" and warrant contradiction checking.
3. For each high-similarity pair, run an LLM judge to classify the conflict type:
   - no_conflict:        claims are compatible
   - partial_conflict:   same topic, different populations or timeframes (not truly contradictory)
   - direct_conflict:    same population, same outcome, opposite conclusions
   - irreconcilable:     both sources are high credibility (> 0.70) and directly contradict
4. Pairs judged "no_conflict" are still logged (the Skeptic checks ignored conflicts).
5. "irreconcilable" flags are critical — they flow to Synthesis and Arbiter to cap confidence.
6. LLM calls are batched: up to 5 pairs per call to reduce token spend.
7. If embedding fails: log error, skip contradiction detection, continue pipeline.
8. If LLM judge fails for a pair: default to "partial_conflict" (conservative), log error.

LLM usage: One judge call per batch of up to 5 high-similarity pairs.
Writes to state: contradictions, audit_trail
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from core.state import PipelineState
from services.llm import call_llm

logger = structlog.get_logger(__name__)

# Constants 

JUDGE_BATCH_SIZE = 5          # claim pairs per LLM judge call
HIGH_CREDIBILITY_FLOOR = 0.70 # both sources above this → irreconcilable is possible


# Prompts 

_JUDGE_SYSTEM = """You are a research integrity analyst judging whether pairs of claims from different sources conflict with each other.

For each claim pair provided, classify the conflict type using exactly one of these labels:
- "no_conflict":       The claims are compatible, complementary, or about different things.
- "partial_conflict":  The claims address the same topic but differ in population, timeframe, or metric — not truly contradictory.
- "direct_conflict":   The claims address the same population and outcome but reach opposite conclusions.
- "irreconcilable":    Both sources are credible (credibility > 0.70) AND the claims are directly contradictory.

Return ONLY a valid JSON object — no markdown, no preamble:
{
  "judgments": [
    {
      "pair_id": "<pair_id from input>",
      "conflict_type": "<no_conflict|partial_conflict|direct_conflict|irreconcilable>",
      "explanation": "<one sentence: why you classified it this way, citing specific differences>"
    }
  ]
}

Rules:
- Be precise. "no_conflict" is appropriate when claims measure different things.
- "partial_conflict" is appropriate when claims use different populations or timeframes.
- "direct_conflict" and "irreconcilable" require the same population + outcome with opposite findings.
- The difference between "direct_conflict" and "irreconcilable" is source credibility — only use "irreconcilable" if BOTH sources have credibility > 0.70.
- Keep explanations under 150 characters."""


def _build_judge_prompt(
    pairs: list[dict],
    claims_lookup: dict[str, dict],
    credibility_lookup: dict[str, float],
) -> str:
    """
    Build the user-side prompt for a batch of claim pairs.

    Each pair entry: {pair_id, claim_a_id, claim_b_id, similarity_score}
    """
    lines = ["Evaluate the following claim pairs for contradictions:\n"]

    for pair in pairs:
        pair_id = pair["pair_id"]
        ca_id = pair["claim_a_id"]
        cb_id = pair["claim_b_id"]
        sim = pair["similarity_score"]

        ca = claims_lookup.get(ca_id, {})
        cb = claims_lookup.get(cb_id, {})

        ca_src = ca.get("source_id", "unknown")
        cb_src = cb.get("source_id", "unknown")
        ca_cred = credibility_lookup.get(ca_src, 0.0)
        cb_cred = credibility_lookup.get(cb_src, 0.0)

        lines.append(
            f"PAIR {pair_id} (similarity: {sim:.2f}):\n"
            f"  Claim A [{ca_id}] (source: {ca_src}, credibility: {ca_cred:.2f}):\n"
            f"    Text: {ca.get('text', '')}\n"
            f"    Population: {ca.get('population', 'not specified')}\n"
            f"    Timeframe: {ca.get('timeframe', 'not specified')}\n"
            f"    Outcome: {ca.get('outcome_metric', 'not specified')}\n"
            f"  Claim B [{cb_id}] (source: {cb_src}, credibility: {cb_cred:.2f}):\n"
            f"    Text: {cb.get('text', '')}\n"
            f"    Population: {cb.get('population', 'not specified')}\n"
            f"    Timeframe: {cb.get('timeframe', 'not specified')}\n"
            f"    Outcome: {cb.get('outcome_metric', 'not specified')}\n"
        )

    return "\n".join(lines)


# Parse judge response 

_VALID_CONFLICT_TYPES = frozenset({
    "no_conflict", "partial_conflict", "direct_conflict", "irreconcilable"
})


def _parse_judgments(
    raw: dict,
    pairs: list[dict],
    claims_lookup: dict[str, dict],
    credibility_lookup: dict[str, float],
) -> list[dict]:
    """
    Parse and validate the LLM judge response.

    - Validates conflict_type values.
    - Downgrades "irreconcilable" to "direct_conflict" if either source credibility < 0.70.
    - Returns list of judgment dicts with pair metadata re-attached.
    """
    judgments_raw = raw.get("judgments", [])
    if not isinstance(judgments_raw, list):
        return []

    # Index pairs by pair_id for fast lookup
    pair_by_id = {p["pair_id"]: p for p in pairs}
    results: list[dict] = []

    for item in judgments_raw:
        if not isinstance(item, dict):
            continue

        pair_id = item.get("pair_id", "")
        conflict_type = item.get("conflict_type", "partial_conflict")
        explanation = item.get("explanation", "No explanation provided.").strip()

        if pair_id not in pair_by_id:
            logger.warning("contradiction_unexpected_pair_id", pair_id=pair_id)
            continue

        # Validate conflict type
        if conflict_type not in _VALID_CONFLICT_TYPES:
            logger.warning(
                "contradiction_invalid_conflict_type",
                pair_id=pair_id,
                raw_type=conflict_type,
            )
            conflict_type = "partial_conflict"  # conservative default

        pair = pair_by_id[pair_id]
        ca_id = pair["claim_a_id"]
        cb_id = pair["claim_b_id"]

        # Enforce irreconcilable constraint: BOTH sources must be high credibility
        if conflict_type == "irreconcilable":
            ca_src = claims_lookup.get(ca_id, {}).get("source_id", "")
            cb_src = claims_lookup.get(cb_id, {}).get("source_id", "")
            cred_a = credibility_lookup.get(ca_src, 0.0)
            cred_b = credibility_lookup.get(cb_src, 0.0)

            if cred_a < HIGH_CREDIBILITY_FLOOR or cred_b < HIGH_CREDIBILITY_FLOOR:
                logger.info(
                    "contradiction_irreconcilable_downgraded",
                    pair_id=pair_id,
                    cred_a=cred_a,
                    cred_b=cred_b,
                    floor=HIGH_CREDIBILITY_FLOOR,
                )
                conflict_type = "direct_conflict"
                explanation = (
                    f"[Downgraded from irreconcilable: source credibility below {HIGH_CREDIBILITY_FLOOR}] "
                    + explanation
                )

        results.append({
            "pair_id": pair_id,
            "claim_a_id": ca_id,
            "claim_b_id": cb_id,
            "conflict_type": conflict_type,
            "explanation": explanation,
            "similarity_score": pair["similarity_score"],
        })

    return results


# Conflict ID counter 

def _make_conflict_id(index: int) -> str:
    return f"con_{str(index).zfill(3)}"


# Audit helper 

def _audit(request_id: str, event: str, data: dict) -> dict:
    return {
        "event": event,
        "stage": "contradiction_detector",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Main agent node 

def run_contradiction_detector(state: PipelineState) -> dict:
    """
    LangGraph node function for the Contradiction Detector agent.

    Reads:  state["claims"], state["scored_sources"], state["request_id"]
    Writes: contradictions, audit_trail
    """
    from config.settings import get_settings
    settings = get_settings()

    request_id = state["request_id"]
    claims = state.get("claims", [])
    scored_sources = state.get("scored_sources", [])

    log = logger.bind(stage="contradiction_detector", request_id=request_id)
    log.info("contradiction_detector_start", claim_count=len(claims))

    audit_records: list[dict] = []

    # Need at least 2 claims to detect contradictions
    if len(claims) < 2:
        log.info("contradiction_detector_skip", reason="fewer_than_2_claims")
        audit_records.append(_audit(request_id, "contradiction_skipped", {
            "reason": "fewer_than_2_claims",
            "claim_count": len(claims),
        }))
        return {
            "contradictions": [],
            "audit_trail": state["audit_trail"] + audit_records,
        }

    # Build fast lookups
    claims_lookup: dict[str, dict] = {c["claim_id"]: c for c in claims}
    credibility_lookup: dict[str, float] = {
        s["source_id"]: s.get("credibility", 0.0)
        for s in scored_sources
    }

    # Step 1: Embed all claims 
    claim_texts = [c["text"] for c in claims]
    claim_ids = [c["claim_id"] for c in claims]

    try:
        from services.similarity import find_potentially_contradicting_claim_pairs
        similar_pairs_raw = find_potentially_contradicting_claim_pairs(
            claim_texts=claim_texts,
            claim_ids=claim_ids,
            similarity_threshold=settings.contradiction_similarity_threshold,
        )
        log.info(
            "contradiction_similar_pairs_found",
            total_claims=len(claims),
            similar_pairs=len(similar_pairs_raw),
            threshold=settings.contradiction_similarity_threshold,
        )
        audit_records.append(_audit(request_id, "embedding_complete", {
            "claim_count": len(claims),
            "similar_pairs_found": len(similar_pairs_raw),
            "threshold": settings.contradiction_similarity_threshold,
        }))
    except Exception as exc:
        log.error("contradiction_embedding_failed", error=str(exc))
        audit_records.append(_audit(request_id, "embedding_failed", {
            "error": str(exc),
            "claim_count": len(claims),
        }))
        # Cannot detect contradictions without embeddings — return empty
        return {
            "contradictions": [],
            "audit_trail": state["audit_trail"] + audit_records,
        }

    if not similar_pairs_raw:
        log.info("contradiction_no_similar_pairs")
        audit_records.append(_audit(request_id, "no_similar_pairs", {
            "claim_count": len(claims),
        }))
        audit_records.append(_audit(request_id, "contradiction_detector_summary", {
            "total_pairs_evaluated": 0,
            "contradictions_recorded": 0,
            "type_breakdown": {},
            "irreconcilable_count": 0,
        }))
        return {
            "contradictions": [],
            "audit_trail": state["audit_trail"] + audit_records,
        }

    # Step 2: Filter out same-source pairs 
    # Claims from the same source cannot contradict each other in the spec's sense
    cross_source_pairs = []
    for ca_id, cb_id, sim in similar_pairs_raw:
        src_a = claims_lookup.get(ca_id, {}).get("source_id", "")
        src_b = claims_lookup.get(cb_id, {}).get("source_id", "")
        if src_a != src_b:
            cross_source_pairs.append({
                "pair_id": f"p_{str(len(cross_source_pairs) + 1).zfill(3)}",
                "claim_a_id": ca_id,
                "claim_b_id": cb_id,
                "similarity_score": round(sim, 4),
            })

    log.info(
        "contradiction_cross_source_pairs",
        total_similar=len(similar_pairs_raw),
        cross_source=len(cross_source_pairs),
    )

    if not cross_source_pairs:
        log.info("contradiction_all_same_source")
        audit_records.append(_audit(request_id, "no_cross_source_pairs", {}))
        audit_records.append(_audit(request_id, "contradiction_detector_summary", {
            "total_pairs_evaluated": 0,
            "contradictions_recorded": 0,
            "type_breakdown": {},
            "irreconcilable_count": 0,
        }))
        return {
            "contradictions": [],
            "audit_trail": state["audit_trail"] + audit_records,
        }

    # Step 3: LLM judge — batched 
    all_judgments: list[dict] = []

    for batch_start in range(0, len(cross_source_pairs), JUDGE_BATCH_SIZE):
        batch = cross_source_pairs[batch_start: batch_start + JUDGE_BATCH_SIZE]

        log.info(
            "contradiction_judge_batch",
            batch_start=batch_start,
            batch_size=len(batch),
        )

        try:
            user_prompt = _build_judge_prompt(batch, claims_lookup, credibility_lookup)

            raw = call_llm(
                system_prompt=_JUDGE_SYSTEM,
                user_prompt=user_prompt,
                call_type="contradictions",
                stage="contradiction_detector",
                request_id=request_id,
                max_tokens=1000,
                temperature=0.0,
            )

            batch_judgments = _parse_judgments(
                raw=raw,
                pairs=batch,
                claims_lookup=claims_lookup,
                credibility_lookup=credibility_lookup,
            )
            all_judgments.extend(batch_judgments)

            audit_records.append(_audit(request_id, "judge_batch_complete", {
                "batch_size": len(batch),
                "judgments_returned": len(batch_judgments),
                "conflict_types": [j["conflict_type"] for j in batch_judgments],
            }))

        except Exception as exc:
            log.error(
                "contradiction_judge_batch_failed",
                batch_start=batch_start,
                error=str(exc),
            )
            # Conservative fallback: treat all pairs in this batch as partial_conflict
            for pair in batch:
                all_judgments.append({
                    "pair_id": pair["pair_id"],
                    "claim_a_id": pair["claim_a_id"],
                    "claim_b_id": pair["claim_b_id"],
                    "conflict_type": "partial_conflict",
                    "explanation": f"Judge call failed ({exc}); defaulting to partial_conflict.",
                    "similarity_score": pair["similarity_score"],
                })
            audit_records.append(_audit(request_id, "judge_batch_failed", {
                "batch_start": batch_start,
                "error": str(exc),
                "fallback": "partial_conflict",
            }))

    # Step 4: Build final contradiction records 
    # Skip "no_conflict" pairs — they don't need to appear in the output
    # (the Skeptic checks for ignored conflicts separately via claim evidence)
    actual_conflicts = [j for j in all_judgments if j["conflict_type"] != "no_conflict"]

    contradictions: list[dict] = []
    for i, judgment in enumerate(actual_conflicts):
        contradictions.append({
            "conflict_id": _make_conflict_id(i + 1),
            "claim_a_id": judgment["claim_a_id"],
            "claim_b_id": judgment["claim_b_id"],
            "conflict_type": judgment["conflict_type"],
            "explanation": judgment["explanation"],
            "similarity_score": judgment["similarity_score"],
        })

    # Summarise by severity for logging
    type_counts: dict[str, int] = {}
    for j in all_judgments:
        ct = j["conflict_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    irreconcilable_count = type_counts.get("irreconcilable", 0)
    direct_count = type_counts.get("direct_conflict", 0)

    log.info(
        "contradiction_detector_complete",
        pairs_evaluated=len(all_judgments),
        contradictions_found=len(contradictions),
        irreconcilable=irreconcilable_count,
        direct_conflict=direct_count,
        type_counts=type_counts,
    )

    audit_records.append(_audit(request_id, "contradiction_detector_summary", {
        "total_pairs_evaluated": len(all_judgments),
        "contradictions_recorded": len(contradictions),
        "type_breakdown": type_counts,
        "irreconcilable_count": irreconcilable_count,
    }))

    return {
        "contradictions": contradictions,
        "audit_trail": state["audit_trail"] + audit_records,
    }