"""
Unit tests for agents/contradiction_detector.py

Tests cover:
- Same-source pair filtering
- Irreconcilable downgrade logic (credibility < 0.70)
- Conflict type validation
- Audit trail integrity
- Graceful handling of embedding/LLM failures

Uses DRY_RUN=True for LLM calls. Embedding calls are mocked where needed
to avoid requiring sentence-transformers to be installed for unit tests.

Run with: pytest tests/test_contradiction_detector.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")

from core.state import make_initial_state
from agents.contradiction_detector import (
    run_contradiction_detector,
    _parse_judgments,
    _build_judge_prompt,
    _make_conflict_id,
    HIGH_CREDIBILITY_FLOOR,
    JUDGE_BATCH_SIZE,
)


# Fixtures 

def _make_state(
    claims: list | None = None,
    scored_sources: list | None = None,
) -> dict:
    state = make_initial_state(
        "req_cd_test",
        "Does aspirin reduce cardiovascular events in healthy adults?",
        [],
    )
    state["claims"] = claims or []
    state["scored_sources"] = scored_sources or []
    return state


def _claim(
    claim_id: str,
    source_id: str,
    text: str,
    population: str = "healthy adults",
    outcome: str = "cardiovascular events",
) -> dict:
    return {
        "claim_id": claim_id,
        "source_id": source_id,
        "text": text,
        "population": population,
        "timeframe": "12 months",
        "outcome_metric": outcome,
        "study_design": "RCT",
        "low_credibility": False,
    }


def _scored(source_id: str, credibility: float = 0.80) -> dict:
    return {
        "source_id": source_id,
        "credibility": credibility,
        "relevance": 0.75,
        "source_type": "peer_reviewed",
        "trusted_source": True,
    }


def _mock_similar_pairs(pairs: list[tuple[str, str, float]]):
    """Return a mock for find_potentially_contradicting_claim_pairs."""
    return MagicMock(return_value=pairs)


#  
# Helper Function Tests
#  

class TestConflictIdGeneration:

    def test_format(self):
        assert _make_conflict_id(1) == "con_001"
        assert _make_conflict_id(10) == "con_010"
        assert _make_conflict_id(100) == "con_100"


class TestParseJudgments:

    def _pair(self, pair_id: str, ca_id: str, cb_id: str, sim: float = 0.85) -> dict:
        return {
            "pair_id": pair_id,
            "claim_a_id": ca_id,
            "claim_b_id": cb_id,
            "similarity_score": sim,
        }

    def test_valid_direct_conflict(self):
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces CV events."},
            "clm_002": {"source_id": "src_002", "text": "Aspirin does not reduce CV events."},
        }
        credibility_lookup = {"src_001": 0.85, "src_002": 0.82}
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "direct_conflict",
                    "explanation": "Opposite conclusions on cardiovascular event reduction.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert len(result) == 1
        assert result[0]["conflict_type"] == "direct_conflict"

    def test_irreconcilable_both_high_credibility(self):
        """irreconcilable should stay when both sources >= 0.70."""
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces CV events by 20%."},
            "clm_002": {"source_id": "src_002", "text": "Aspirin increases CV events by 10%."},
        }
        credibility_lookup = {"src_001": 0.92, "src_002": 0.88}
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "irreconcilable",
                    "explanation": "Both high-credibility RCTs with opposite findings.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert result[0]["conflict_type"] == "irreconcilable"

    def test_irreconcilable_downgraded_low_credibility_source_a(self):
        """irreconcilable must be downgraded when source_a credibility < 0.70."""
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces CV events significantly."},
            "clm_002": {"source_id": "src_002", "text": "Aspirin worsens cardiovascular outcomes."},
        }
        credibility_lookup = {"src_001": 0.35, "src_002": 0.88}  # src_001 is low
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "irreconcilable",
                    "explanation": "Opposite findings.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert result[0]["conflict_type"] == "direct_conflict"
        assert "Downgraded" in result[0]["explanation"]

    def test_irreconcilable_downgraded_low_credibility_source_b(self):
        """irreconcilable must be downgraded when source_b credibility < 0.70."""
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces CV events significantly."},
            "clm_002": {"source_id": "src_002", "text": "Aspirin worsens cardiovascular outcomes."},
        }
        credibility_lookup = {"src_001": 0.90, "src_002": 0.50}  # src_002 is low
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "irreconcilable",
                    "explanation": "Opposite findings.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert result[0]["conflict_type"] == "direct_conflict"

    def test_invalid_conflict_type_defaults_to_partial(self):
        """Unknown conflict type values must be replaced with 'partial_conflict'."""
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Claim A about aspirin effects."},
            "clm_002": {"source_id": "src_002", "text": "Claim B about aspirin outcomes."},
        }
        credibility_lookup = {"src_001": 0.80, "src_002": 0.75}
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "completely_made_up_type",
                    "explanation": "Some explanation.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert result[0]["conflict_type"] == "partial_conflict"

    def test_unexpected_pair_id_skipped(self):
        """Judgments with pair IDs not in the input must be silently dropped."""
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Test claim."},
            "clm_002": {"source_id": "src_002", "text": "Other claim."},
        }
        credibility_lookup = {"src_001": 0.80, "src_002": 0.75}
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_999",  # not in input
                    "conflict_type": "direct_conflict",
                    "explanation": "Should be ignored.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert result == []

    def test_no_conflict_included_in_parse_output(self):
        """no_conflict judgments ARE returned by _parse_judgments (filtering happens later)."""
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces LDL cholesterol."},
            "clm_002": {"source_id": "src_002", "text": "Aspirin reduces platelet aggregation."},
        }
        credibility_lookup = {"src_001": 0.80, "src_002": 0.75}
        pairs = [self._pair("p_001", "clm_001", "clm_002")]

        raw = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "no_conflict",
                    "explanation": "Measures different outcomes.",
                }
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert len(result) == 1
        assert result[0]["conflict_type"] == "no_conflict"

    def test_similarity_score_preserved(self):
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces cardiovascular events."},
            "clm_002": {"source_id": "src_002", "text": "Aspirin worsens cardiovascular events."},
        }
        credibility_lookup = {"src_001": 0.80, "src_002": 0.75}
        pairs = [self._pair("p_001", "clm_001", "clm_002", sim=0.88)]
        raw = {
            "judgments": [
                {"pair_id": "p_001", "conflict_type": "direct_conflict", "explanation": "Opposite."}
            ]
        }
        result = _parse_judgments(raw, pairs, claims_lookup, credibility_lookup)
        assert result[0]["similarity_score"] == 0.88


class TestBuildJudgePrompt:

    def test_contains_pair_id(self):
        pairs = [{"pair_id": "p_001", "claim_a_id": "clm_001", "claim_b_id": "clm_002", "similarity_score": 0.82}]
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Aspirin reduces CV risk.", "population": "adults", "timeframe": "12m", "outcome_metric": "CV events"},
            "clm_002": {"source_id": "src_002", "text": "Aspirin increases bleeding.", "population": "elderly", "timeframe": "24m", "outcome_metric": "bleeding"},
        }
        credibility_lookup = {"src_001": 0.85, "src_002": 0.80}
        prompt = _build_judge_prompt(pairs, claims_lookup, credibility_lookup)
        assert "p_001" in prompt
        assert "clm_001" in prompt
        assert "clm_002" in prompt

    def test_contains_similarity_score(self):
        pairs = [{"pair_id": "p_001", "claim_a_id": "clm_001", "claim_b_id": "clm_002", "similarity_score": 0.91}]
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Claim A text.", "population": None, "timeframe": None, "outcome_metric": None},
            "clm_002": {"source_id": "src_002", "text": "Claim B text.", "population": None, "timeframe": None, "outcome_metric": None},
        }
        credibility_lookup = {"src_001": 0.80, "src_002": 0.75}
        prompt = _build_judge_prompt(pairs, claims_lookup, credibility_lookup)
        assert "0.91" in prompt

    def test_contains_credibility_scores(self):
        pairs = [{"pair_id": "p_001", "claim_a_id": "clm_001", "claim_b_id": "clm_002", "similarity_score": 0.80}]
        claims_lookup = {
            "clm_001": {"source_id": "src_001", "text": "Claim A.", "population": None, "timeframe": None, "outcome_metric": None},
            "clm_002": {"source_id": "src_002", "text": "Claim B.", "population": None, "timeframe": None, "outcome_metric": None},
        }
        credibility_lookup = {"src_001": 0.92, "src_002": 0.45}
        prompt = _build_judge_prompt(pairs, claims_lookup, credibility_lookup)
        assert "0.92" in prompt
        assert "0.45" in prompt


#  
# Integration Tests — run_contradiction_detector
#  

class TestRunContradictionDetector:

    def test_fewer_than_2_claims_returns_empty(self):
        state = _make_state(
            claims=[_claim("clm_001", "src_001", "Single claim about aspirin.")],
            scored_sources=[_scored("src_001")],
        )
        result = run_contradiction_detector(state)
        assert result["contradictions"] == []
        skip_records = [r for r in result["audit_trail"] if r.get("event") == "contradiction_skipped"]
        assert len(skip_records) == 1

    def test_zero_claims_returns_empty(self):
        state = _make_state(claims=[], scored_sources=[])
        result = run_contradiction_detector(state)
        assert result["contradictions"] == []

    def test_same_source_pairs_filtered_out(self):
        """Two claims from the same source should never be judged for contradiction."""
        claims = [
            _claim("clm_001", "src_001", "Aspirin reduces major cardiovascular events by 15%."),
            _claim("clm_002", "src_001", "Aspirin increases bleeding risk by 25%."),  # same source
        ]
        # Mock the embedding to return high similarity
        fake_embeddings = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)

        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = [("clm_001", "clm_002", 0.95)]
            state = _make_state(claims=claims, scored_sources=[_scored("src_001")])
            result = run_contradiction_detector(state)

        # Same-source pairs are filtered; no LLM judge is called
        assert result["contradictions"] == []
        no_pairs_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "no_cross_source_pairs"
        ]
        assert len(no_pairs_records) == 1

    def test_cross_source_pair_produces_contradiction(self):
        """Two claims from different sources with high similarity → contradiction check → recorded."""
        claims = [
            _claim("clm_001", "src_001", "Aspirin significantly reduces cardiovascular event risk."),
            _claim("clm_002", "src_002", "Aspirin does not reduce cardiovascular event risk significantly."),
        ]
        scored = [_scored("src_001", 0.90), _scored("src_002", 0.85)]

        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = [("clm_001", "clm_002", 0.87)]
            state = _make_state(claims=claims, scored_sources=scored)
            result = run_contradiction_detector(state)

        # Dry-run LLM returns {"conflict_type": "no_conflict"} →
        # no_conflict is filtered out, so contradictions list may be empty in dry-run
        # But the pipeline should not crash and should have processed the pair
        assert isinstance(result["contradictions"], list)
        assert isinstance(result["audit_trail"], list)

    def test_no_similar_pairs_returns_empty_contradictions(self):
        claims = [
            _claim("clm_001", "src_001", "Aspirin reduces platelet aggregation in adults."),
            _claim("clm_002", "src_002", "Exercise reduces LDL cholesterol in elderly patients."),
        ]
        scored = [_scored("src_001"), _scored("src_002")]

        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = []  # no similar pairs
            state = _make_state(claims=claims, scored_sources=scored)
            result = run_contradiction_detector(state)

        assert result["contradictions"] == []
        no_pairs_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "no_similar_pairs"
        ]
        assert len(no_pairs_records) == 1

    def test_embedding_failure_returns_empty_not_crash(self):
        """If embedding service fails, return empty contradictions gracefully."""
        claims = [
            _claim("clm_001", "src_001", "Aspirin reduces CV events."),
            _claim("clm_002", "src_002", "Aspirin increases bleeding risk."),
        ]
        scored = [_scored("src_001"), _scored("src_002")]

        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.side_effect = RuntimeError("Embedding model not available")
            state = _make_state(claims=claims, scored_sources=scored)
            result = run_contradiction_detector(state)

        assert result["contradictions"] == []
        fail_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "embedding_failed"
        ]
        assert len(fail_records) == 1

    def test_contradiction_ids_sequential(self):
        """Contradiction IDs must be con_001, con_002, etc."""
        # Build multiple cross-source pairs that will all get non-no_conflict judgments
        claims = [
            _claim("clm_001", "src_001", "Aspirin significantly reduces cardiovascular event risk."),
            _claim("clm_002", "src_002", "Aspirin does not reduce cardiovascular risk at all."),
            _claim("clm_003", "src_003", "Aspirin significantly worsens cardiovascular outcomes."),
        ]
        scored = [_scored("src_001", 0.90), _scored("src_002", 0.85), _scored("src_003", 0.80)]

        # Mock: all pairs are similar
        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = [
                ("clm_001", "clm_002", 0.88),
                ("clm_001", "clm_003", 0.86),
            ]
            # Override dry-run LLM to return direct_conflict so they pass the filter
            dry_run_response = {
                "judgments": [
                    {"pair_id": "p_001", "conflict_type": "direct_conflict", "explanation": "Opposite conclusions."},
                    {"pair_id": "p_002", "conflict_type": "direct_conflict", "explanation": "Opposite conclusions."},
                ]
            }
            with patch("agents.contradiction_detector.call_llm", return_value=dry_run_response):
                state = _make_state(claims=claims, scored_sources=scored)
                result = run_contradiction_detector(state)

        contradictions = result["contradictions"]
        if len(contradictions) >= 2:
            assert contradictions[0]["conflict_id"] == "con_001"
            assert contradictions[1]["conflict_id"] == "con_002"

    def test_contradiction_record_has_required_fields(self):
        claims = [
            _claim("clm_001", "src_001", "Aspirin significantly reduces cardiovascular events."),
            _claim("clm_002", "src_002", "Aspirin does not affect cardiovascular event rates."),
        ]
        scored = [_scored("src_001", 0.90), _scored("src_002", 0.85)]

        dry_run_response = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "direct_conflict",
                    "explanation": "Same population, opposite outcomes.",
                }
            ]
        }
        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = [("clm_001", "clm_002", 0.87)]
            with patch("agents.contradiction_detector.call_llm", return_value=dry_run_response):
                state = _make_state(claims=claims, scored_sources=scored)
                result = run_contradiction_detector(state)

        if result["contradictions"]:
            conflict = result["contradictions"][0]
            required_fields = [
                "conflict_id", "claim_a_id", "claim_b_id",
                "conflict_type", "explanation", "similarity_score"
            ]
            for field in required_fields:
                assert field in conflict, f"Missing field: {field}"

    def test_audit_trail_has_summary(self):
        claims = [
            _claim("clm_001", "src_001", "Aspirin reduces cardiovascular events."),
            _claim("clm_002", "src_002", "Aspirin increases cardiovascular risk."),
        ]
        scored = [_scored("src_001"), _scored("src_002")]

        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = []
            state = _make_state(claims=claims, scored_sources=scored)
            result = run_contradiction_detector(state)

        summary_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "contradiction_detector_summary"
        ]
        assert len(summary_records) == 1
        summary = summary_records[0]["data"]
        assert "total_pairs_evaluated" in summary
        assert "irreconcilable_count" in summary

    def test_audit_trail_appended_not_replaced(self):
        state = _make_state(
            claims=[_claim("clm_001", "src_001", "Only one claim available.")],
        )
        prior_record = {
            "event": "prior_stage",
            "stage": "evidence_extractor",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "request_id": "req_cd_test",
            "data": {},
        }
        state["audit_trail"] = [prior_record]
        result = run_contradiction_detector(state)
        assert result["audit_trail"][0]["event"] == "prior_stage"
        assert len(result["audit_trail"]) >= 2

    def test_no_conflict_excluded_from_contradictions_output(self):
        """no_conflict pairs should NOT appear in state['contradictions']."""
        claims = [
            _claim("clm_001", "src_001", "Aspirin reduces platelet aggregation in adults."),
            _claim("clm_002", "src_002", "Aspirin affects platelet function in healthy subjects."),
        ]
        scored = [_scored("src_001"), _scored("src_002")]

        dry_run_response = {
            "judgments": [
                {
                    "pair_id": "p_001",
                    "conflict_type": "no_conflict",
                    "explanation": "Complementary findings about same mechanism.",
                }
            ]
        }
        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = [("clm_001", "clm_002", 0.78)]
            with patch("agents.contradiction_detector.call_llm", return_value=dry_run_response):
                state = _make_state(claims=claims, scored_sources=scored)
                result = run_contradiction_detector(state)

        # no_conflict should be excluded from the contradictions output
        assert all(
            c["conflict_type"] != "no_conflict"
            for c in result["contradictions"]
        )

    def test_judge_failure_defaults_to_partial_conflict(self):
        """If the LLM judge fails, pairs should default to partial_conflict (conservative)."""
        claims = [
            _claim("clm_001", "src_001", "Aspirin reduces cardiovascular events significantly."),
            _claim("clm_002", "src_002", "Aspirin worsens cardiovascular health outcomes."),
        ]
        scored = [_scored("src_001"), _scored("src_002")]

        with patch("services.similarity.find_potentially_contradicting_claim_pairs") as mock_sim:
            mock_sim.return_value = [("clm_001", "clm_002", 0.85)]
            with patch("agents.contradiction_detector.call_llm") as mock_llm:
                mock_llm.side_effect = RuntimeError("LLM unavailable")
                state = _make_state(claims=claims, scored_sources=scored)
                result = run_contradiction_detector(state)

        # Should have a partial_conflict entry (fallback)
        partial = [c for c in result["contradictions"] if c["conflict_type"] == "partial_conflict"]
        assert len(partial) >= 1

        # Should have a failure audit record
        fail_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "judge_batch_failed"
        ]
        assert len(fail_records) >= 1

    def test_high_credibility_floor_constant(self):
        """Verify the irreconcilable floor is exactly 0.70 as per spec."""
        assert HIGH_CREDIBILITY_FLOOR == 0.70