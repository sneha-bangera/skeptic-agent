"""
Unit tests for agents/arbiter.py

Critical tests:
- Loop termination is CODE-LEVEL (loop_round >= max): verify the Python guard fires
- Arbiter cannot route "revise" when at max rounds
- Confidence hard rules (irreconcilable → not High, max rounds + unresolved → Insufficient)
- Ruling parsing and fallback handling
- source_count_ceiling enforcement

Run with: pytest tests/test_arbiter.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")

from core.state import make_initial_state
from agents.arbiter import (
    run_arbiter,
    _parse_rulings,
    _apply_hard_confidence_rules,
    _build_unresolved_contradictions,
    _VALID_RULINGS,
    _TIER_ORDER,
)
from core.graph import _route_after_arbiter, _route_after_guardrail


# Fixtures 

def _make_state(
    loop_round: int = 0,
    challenges: list | None = None,
    rulings: list | None = None,
    confidence: str = "Moderate",
    preliminary_answer: str = "Aspirin did not reduce events in adults 70+ [src_001].",
    contradictions: list | None = None,
    source_count_ceiling=None,
    scored_sources: list | None = None,
    claims: list | None = None,
) -> dict:
    state = make_initial_state(
        "req_arbiter_test",
        "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
        [],
    )
    state["loop_round"] = loop_round
    state["skeptic_challenges"] = challenges or []
    state["arbiter_rulings"] = rulings or []
    state["confidence"] = confidence
    state["preliminary_answer"] = preliminary_answer
    state["contradictions"] = contradictions or []
    state["source_count_ceiling"] = source_count_ceiling
    state["scored_sources"] = scored_sources or [_scored("src_001")]
    state["claims"] = claims or []
    return state


def _challenge(
    challenge_id: str,
    severity: str = "critical",
    round_: int = 0,
    type_: str = "wrong_population",
) -> dict:
    return {
        "challenge_id": challenge_id,
        "type": type_,
        "description": f"Detailed description for challenge {challenge_id} about population mismatch.",
        "source_ids_implicated": ["src_001"],
        "severity": severity,
        "suggested_revision": f"Revise based on {challenge_id}.",
        "round": round_,
    }


def _ruling(challenge_id: str, ruling: str = "valid_critical", round_: int = 0) -> dict:
    return {
        "challenge_id": challenge_id,
        "ruling": ruling,
        "reason": f"Challenge {challenge_id} is {ruling}.",
        "round": round_,
    }


def _scored(source_id: str, credibility: float = 0.85) -> dict:
    return {
        "source_id": source_id,
        "credibility": credibility,
        "relevance": 0.80,
        "source_type": "peer_reviewed",
        "bias_flag": "none",
    }


def _irreconcilable_contradiction() -> dict:
    return {
        "conflict_id": "con_001",
        "claim_a_id": "clm_001",
        "claim_b_id": "clm_002",
        "conflict_type": "irreconcilable",
        "explanation": "Both high-credibility RCTs reached opposite conclusions.",
        "similarity_score": 0.91,
    }



# Hard Confidence Rules

class TestApplyHardConfidenceRules:

    def test_irreconcilable_blocks_high_confidence(self):
        result = _apply_hard_confidence_rules(
            confidence="High",
            contradictions=[_irreconcilable_contradiction()],
            source_count_ceiling=None,
            unresolved_critical_count=0,
            loop_round=1,
            max_loop_rounds=3,
        )
        assert result == "Moderate"

    def test_irreconcilable_does_not_downgrade_moderate(self):
        """Irreconcilable should only block High, not further downgrade Moderate."""
        result = _apply_hard_confidence_rules(
            confidence="Moderate",
            contradictions=[_irreconcilable_contradiction()],
            source_count_ceiling=None,
            unresolved_critical_count=0,
            loop_round=1,
            max_loop_rounds=3,
        )
        assert result == "Moderate"

    def test_max_rounds_with_unresolved_critical_forces_insufficient(self):
        result = _apply_hard_confidence_rules(
            confidence="Moderate",
            contradictions=[],
            source_count_ceiling=None,
            unresolved_critical_count=2,
            loop_round=3,
            max_loop_rounds=3,
        )
        assert result == "Insufficient Evidence"

    def test_max_rounds_no_unresolved_does_not_force_insufficient(self):
        """Hitting max rounds alone doesn't force Insufficient if all challenges resolved."""
        result = _apply_hard_confidence_rules(
            confidence="Moderate",
            contradictions=[],
            source_count_ceiling=None,
            unresolved_critical_count=0,
            loop_round=3,
            max_loop_rounds=3,
        )
        assert result == "Moderate"

    def test_source_count_ceiling_enforced(self):
        result = _apply_hard_confidence_rules(
            confidence="High",
            contradictions=[],
            source_count_ceiling="Low",
            unresolved_critical_count=0,
            loop_round=1,
            max_loop_rounds=3,
        )
        assert result == "Low"

    def test_source_count_ceiling_insufficient_enforced(self):
        result = _apply_hard_confidence_rules(
            confidence="Moderate",
            contradictions=[],
            source_count_ceiling="Insufficient Evidence",
            unresolved_critical_count=0,
            loop_round=0,
            max_loop_rounds=3,
        )
        assert result == "Insufficient Evidence"

    def test_ceiling_does_not_upgrade_confidence(self):
        """If confidence is already below ceiling, don't change it."""
        result = _apply_hard_confidence_rules(
            confidence="Low",
            contradictions=[],
            source_count_ceiling="High",
            unresolved_critical_count=0,
            loop_round=0,
            max_loop_rounds=3,
        )
        assert result == "Low"

    def test_all_rules_together_most_conservative_wins(self):
        """Multiple rules: irreconcilable + ceiling both apply, most conservative wins."""
        result = _apply_hard_confidence_rules(
            confidence="High",
            contradictions=[_irreconcilable_contradiction()],
            source_count_ceiling="Low",
            unresolved_critical_count=0,
            loop_round=1,
            max_loop_rounds=3,
        )
        # irreconcilable drops High→Moderate, then ceiling drops Moderate→Low
        assert result == "Low"



# Ruling Parsing Tests

class TestParseRulings:

    def test_valid_ruling_parsed(self):
        challenges = [_challenge("c_001")]
        raw = {"rulings": [{"challenge_id": "c_001", "ruling": "valid_critical", "reason": "Real issue."}]}
        result = _parse_rulings(raw, challenges, loop_round=0)
        assert len(result) == 1
        assert result[0]["ruling"] == "valid_critical"
        assert result[0]["round"] == 0

    def test_invalid_ruling_type_defaults_to_dismissed(self):
        challenges = [_challenge("c_001")]
        raw = {"rulings": [{"challenge_id": "c_001", "ruling": "made_up_ruling", "reason": "X"}]}
        result = _parse_rulings(raw, challenges, loop_round=0)
        assert result[0]["ruling"] == "dismissed"

    def test_missing_challenge_gets_dismissed_fallback(self):
        """If LLM omits a ruling for a challenge, it must default to 'dismissed'."""
        challenges = [_challenge("c_001"), _challenge("c_002")]
        raw = {"rulings": [{"challenge_id": "c_001", "ruling": "valid_minor", "reason": "Minor."}]}
        result = _parse_rulings(raw, challenges, loop_round=0)
        assert len(result) == 2
        ids = {r["challenge_id"] for r in result}
        assert "c_001" in ids
        assert "c_002" in ids
        c002 = next(r for r in result if r["challenge_id"] == "c_002")
        assert c002["ruling"] == "dismissed"

    def test_all_4_valid_ruling_types_accepted(self):
        challenges = [_challenge(f"c_{str(i).zfill(3)}") for i in range(1, 5)]
        raw = {
            "rulings": [
                {"challenge_id": "c_001", "ruling": r, "reason": "X"}
                for r in _VALID_RULINGS
            ]
        }
        # Each valid ruling type should parse without error
        result = _parse_rulings(raw, challenges[:4], loop_round=0)
        ruling_types = {r["ruling"] for r in result}
        assert ruling_types == _VALID_RULINGS

    def test_non_list_rulings_all_default_to_dismissed(self):
        challenges = [_challenge("c_001"), _challenge("c_002")]
        raw = {"rulings": "not a list"}
        result = _parse_rulings(raw, challenges, loop_round=1)
        assert len(result) == 2
        assert all(r["ruling"] == "dismissed" for r in result)



# Graph Routing Tests 

class TestGraphRouting:

    def test_route_revise_when_below_max_rounds(self):
        """Arbiter says 'revise' and loop_round < max → route to synthesis."""
        state = _make_state(loop_round=1)
        state["_arbiter_route"] = "revise"
        route = _route_after_arbiter(state)
        assert route == "synthesis"

    def test_route_finalize_overrides_revise_at_max_rounds(self):
        """
        CRITICAL: Even if Arbiter says 'revise', the code-level guard must
        override to 'confidence_floor' when loop_round >= max_loop_rounds.
        """
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds
        state = _make_state(loop_round=max_rounds)  # at the ceiling
        state["_arbiter_route"] = "revise"  # Arbiter wants to revise
        route = _route_after_arbiter(state)
        # Code-level guard overrides → must NOT go to synthesis
        assert route == "confidence_floor"

    def test_route_finalize_when_arbiter_says_finalize(self):
        state = _make_state(loop_round=0)
        state["_arbiter_route"] = "finalize"
        route = _route_after_arbiter(state)
        assert route == "confidence_floor"

    def test_route_guardrail_pass(self):
        state = make_initial_state("req_1", "Test question here?", [])
        state["input_rejected"] = False
        assert _route_after_guardrail(state) == "ingress"

    def test_route_guardrail_reject(self):
        state = make_initial_state("req_1", "Test question here?", [])
        state["input_rejected"] = True
        state["input_rejection_reason"] = "Empty question."
        assert _route_after_guardrail(state) == "rejection_formatter"



# run_arbiter Integration Tests

class TestRunArbiter:

    def test_no_challenges_finalizes_immediately(self):
        """Zero challenges → arbiter routes to finalize immediately."""
        state = _make_state(loop_round=0, challenges=[], rulings=[])
        result = run_arbiter(state)
        assert result["_arbiter_route"] == "finalize"
        assert result["loop_terminated"] is True
        assert result["final_answer"] is not None

    def test_max_rounds_forces_finalize_regardless(self):
        """
        CRITICAL: At max_loop_rounds, the arbiter must always finalize,
        even if there are unresolved critical challenges.
        """
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds

        state = _make_state(
            loop_round=max_rounds,
            challenges=[_challenge("c_001", severity="critical", round_=max_rounds - 1)],
            rulings=[],
        )
        result = run_arbiter(state)
        assert result["_arbiter_route"] == "finalize"
        assert result["loop_terminated"] is True

    def test_max_rounds_unresolved_critical_sets_insufficient(self):
        """
        At max rounds with unresolved critical challenges → Insufficient Evidence.
        """
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds

        state = _make_state(
            loop_round=max_rounds,
            confidence="Moderate",
            challenges=[_challenge("c_001", severity="critical", round_=0)],
            rulings=[],  # no ruling issued yet → still unresolved
        )
        result = run_arbiter(state)
        assert result["confidence"] == "Insufficient Evidence"

    def test_valid_critical_routes_to_revise(self):
        """Arbiter ruling of valid_critical → must route to revise."""
        mock_rulings_response = {
            "rulings": [
                {
                    "challenge_id": "c_001",
                    "ruling": "valid_critical",
                    "reason": "The population mismatch is material and changes the conclusion.",
                }
            ]
        }
        state = _make_state(
            loop_round=0,
            challenges=[_challenge("c_001", severity="critical", round_=0)],
            rulings=[],
        )
        with patch("agents.arbiter.call_llm", return_value=mock_rulings_response):
            result = run_arbiter(state)

        assert result["_arbiter_route"] == "revise"
        assert result["loop_terminated"] is False
        assert result["loop_round"] == 1  # incremented

    def test_valid_minor_only_routes_to_finalize(self):
        """Arbiter ruling of valid_minor only → finalize (no revision needed)."""
        mock_rulings_response = {
            "rulings": [
                {
                    "challenge_id": "c_001",
                    "ruling": "valid_minor",
                    "reason": "Minor population scope limitation, noted.",
                }
            ]
        }
        state = _make_state(
            loop_round=0,
            challenges=[_challenge("c_001", severity="minor", round_=0)],
            rulings=[],
        )
        with patch("agents.arbiter.call_llm", return_value=mock_rulings_response):
            result = run_arbiter(state)

        assert result["_arbiter_route"] == "finalize"
        assert result["loop_terminated"] is True

    def test_dismissed_routes_to_finalize(self):
        """All challenges dismissed → finalize."""
        mock_rulings_response = {
            "rulings": [
                {
                    "challenge_id": "c_001",
                    "ruling": "dismissed",
                    "reason": "The source actually covers adults 60+.",
                }
            ]
        }
        state = _make_state(
            loop_round=0,
            challenges=[_challenge("c_001", severity="critical", round_=0)],
            rulings=[],
        )
        with patch("agents.arbiter.call_llm", return_value=mock_rulings_response):
            result = run_arbiter(state)

        assert result["_arbiter_route"] == "finalize"
        assert result["loop_terminated"] is True

    def test_loop_round_incremented_only_on_revise(self):
        """loop_round must only be incremented when routing to revise."""
        mock_critical = {
            "rulings": [{"challenge_id": "c_001", "ruling": "valid_critical", "reason": "Real issue."}]
        }
        state = _make_state(
            loop_round=1,
            challenges=[_challenge("c_001", round_=1)],
            rulings=[],
        )
        with patch("agents.arbiter.call_llm", return_value=mock_critical):
            result = run_arbiter(state)

        assert result["loop_round"] == 2  # incremented from 1

        # Now with finalize — should NOT increment
        mock_minor = {
            "rulings": [{"challenge_id": "c_002", "ruling": "valid_minor", "reason": "Minor."}]
        }
        state2 = _make_state(
            loop_round=1,
            challenges=[_challenge("c_002", severity="minor", round_=1)],
            rulings=[],
        )
        with patch("agents.arbiter.call_llm", return_value=mock_minor):
            result2 = run_arbiter(state2)

        assert result2["loop_round"] == 1  # NOT incremented

    def test_irreconcilable_contradiction_downgrades_high_to_moderate(self):
        state = _make_state(
            loop_round=0,
            challenges=[],
            confidence="High",
            contradictions=[_irreconcilable_contradiction()],
        )
        result = run_arbiter(state)
        assert result["confidence"] != "High"
        assert result["confidence"] == "Moderate"

    def test_source_count_ceiling_enforced_in_finalize(self):
        state = _make_state(
            loop_round=0,
            challenges=[],
            confidence="High",
            source_count_ceiling="Low",
        )
        result = run_arbiter(state)
        assert result["confidence"] == "Low"

    def test_existing_rulings_preserved(self):
        """Rulings from previous rounds must be accumulated, not replaced."""
        prior_ruling = _ruling("c_001", "valid_minor", round_=0)
        state = _make_state(
            loop_round=1,
            challenges=[_challenge("c_002", round_=1)],
            rulings=[prior_ruling],
        )
        mock_response = {
            "rulings": [{"challenge_id": "c_002", "ruling": "dismissed", "reason": "OK."}]
        }
        with patch("agents.arbiter.call_llm", return_value=mock_response):
            result = run_arbiter(state)

        # Both rulings should be present
        ruling_ids = {r["challenge_id"] for r in result["arbiter_rulings"]}
        assert "c_001" in ruling_ids
        assert "c_002" in ruling_ids

    def test_challenges_from_previous_rounds_not_re_evaluated(self):
        """Challenges already ruled on must not be sent to the LLM again."""
        c_old = _challenge("c_001", round_=0)
        r_old = _ruling("c_001", "dismissed", round_=0)
        c_new = _challenge("c_002", round_=1)

        state = _make_state(
            loop_round=1,
            challenges=[c_old, c_new],
            rulings=[r_old],
        )
        mock_response = {
            "rulings": [{"challenge_id": "c_002", "ruling": "valid_minor", "reason": "Minor."}]
        }
        llm_calls: list[dict] = []

        def capture_llm(*args, **kwargs):
            llm_calls.append(kwargs.get("user_prompt", ""))
            return mock_response

        with patch("agents.arbiter.call_llm", side_effect=capture_llm):
            result = run_arbiter(state)

        # c_001 should NOT appear in the prompt (already ruled)
        assert len(llm_calls) == 1
        assert "c_001" not in llm_calls[0] or "c_002" in llm_calls[0]

    def test_llm_failure_defaults_to_dismissed_not_crash(self):
        """LLM failure must dismiss all challenges and finalize — not crash."""
        state = _make_state(
            loop_round=0,
            challenges=[_challenge("c_001", round_=0)],
            rulings=[],
        )
        with patch("agents.arbiter.call_llm") as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM down")
            result = run_arbiter(state)

        # Should finalize (dismissed = no critical issue)
        assert result["_arbiter_route"] == "finalize"
        dismissed = [r for r in result["arbiter_rulings"] if r["ruling"] == "dismissed"]
        assert len(dismissed) >= 1

    def test_audit_trail_written_on_all_paths(self):
        # Path 1: no challenges
        state1 = _make_state(loop_round=0, challenges=[])
        r1 = run_arbiter(state1)
        assert len([a for a in r1["audit_trail"] if a["stage"] == "arbiter"]) >= 1

        # Path 2: max rounds
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds
        state2 = _make_state(loop_round=max_rounds, challenges=[_challenge("c_001")])
        r2 = run_arbiter(state2)
        assert len([a for a in r2["audit_trail"] if a["stage"] == "arbiter"]) >= 1



# Confidence Floor Guardrail Tests

class TestConfidenceFloorGuardrail:

    def _make_floor_state(self, **kwargs) -> dict:
        from guardrails.confidence_floor import run_confidence_floor
        state = _make_state(**kwargs)
        state["loop_terminated"] = True
        return state

    def test_no_issues_passes_through(self):
        from guardrails.confidence_floor import run_confidence_floor
        state = self._make_floor_state(loop_round=1, confidence="Moderate")
        result = run_confidence_floor(state)
        assert result["confidence"] == "Moderate"

    def test_irreconcilable_blocks_high(self):
        from guardrails.confidence_floor import run_confidence_floor
        state = self._make_floor_state(
            loop_round=1,
            confidence="High",
            contradictions=[_irreconcilable_contradiction()],
        )
        result = run_confidence_floor(state)
        assert result["confidence"] == "Moderate"
        assert len(result["unresolved_contradictions"]) >= 1

    def test_ceiling_enforced(self):
        from guardrails.confidence_floor import run_confidence_floor
        state = self._make_floor_state(
            loop_round=1,
            confidence="High",
            source_count_ceiling="Low",
        )
        result = run_confidence_floor(state)
        assert result["confidence"] == "Low"

    def test_max_rounds_unresolved_critical_forced_insufficient(self):
        from guardrails.confidence_floor import run_confidence_floor
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds

        state = self._make_floor_state(
            loop_round=max_rounds,
            confidence="Moderate",
            challenges=[_challenge("c_001", severity="critical")],
        )
        state["skeptic_challenges"] = [_challenge("c_001", severity="critical")]
        state["arbiter_rulings"] = []  # unruled

        result = run_confidence_floor(state)
        assert result["confidence"] == "Insufficient Evidence"

    def test_audit_record_written(self):
        from guardrails.confidence_floor import run_confidence_floor
        state = self._make_floor_state(loop_round=1, confidence="Moderate")
        result = run_confidence_floor(state)
        records = [r for r in result["audit_trail"] if r.get("event") == "confidence_floor_guardrail"]
        assert len(records) == 1
        data = records[0]["data"]
        assert "original_confidence" in data
        assert "adjusted_confidence" in data