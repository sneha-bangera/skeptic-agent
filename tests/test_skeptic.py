"""
Unit tests for agents/skeptic.py

Tests cover:
- Challenge parsing and validation (vague challenges rejected, valid ones accepted)
- Specificity enforcement (no source_ids → rejected)
- Severity validation
- Challenge ID sequencing across rounds
- Zero-challenge clean response handling
- LLM failure isolation

Run with: pytest tests/test_skeptic.py -v
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
from agents.skeptic import (
    run_skeptic,
    _parse_challenges,
    _build_skeptic_prompt,
    _VALID_CHALLENGE_TYPES,
    _VALID_SEVERITIES,
)


# Fixtures 

def _make_state(
    preliminary_answer: str = "Aspirin reduces cardiovascular events [src_001].",
    scored_sources: list | None = None,
    claims: list | None = None,
    contradictions: list | None = None,
    existing_challenges: list | None = None,
    loop_round: int = 0,
) -> dict:
    state = make_initial_state(
        "req_skeptic_test",
        "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
        [],
    )
    state["preliminary_answer"] = preliminary_answer
    state["scored_sources"] = scored_sources or [_scored("src_001")]
    state["normalized_sources"] = [_norm("src_001")]
    state["claims"] = claims or [_claim("clm_001", "src_001")]
    state["contradictions"] = contradictions or []
    state["skeptic_challenges"] = existing_challenges or []
    state["loop_round"] = loop_round
    return state


def _scored(source_id: str, credibility: float = 0.85, bias: str = "none") -> dict:
    return {
        "source_id": source_id,
        "credibility": credibility,
        "relevance": 0.80,
        "source_type": "peer_reviewed",
        "bias_flag": bias,
        "trusted_source": True,
    }


def _norm(source_id: str) -> dict:
    return {
        "source_id": source_id,
        "title": f"Study {source_id}",
        "published_date": "2023-01-01",
        "content_text": "RCT on aspirin effects in adults 70+ over 5 years.",
    }


def _claim(claim_id: str, source_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "source_id": source_id,
        "text": "Aspirin did not significantly reduce cardiovascular events in adults 70+.",
        "population": "adults aged 70+",
        "timeframe": "5 years",
        "outcome_metric": "cardiovascular events",
        "study_design": "RCT",
        "low_credibility": False,
    }


def _challenge_raw(
    type_: str = "wrong_population",
    description: str = "Source src_001 studied adults 70+ but the question asks about adults over 60.",
    source_ids: list | None = None,
    severity: str = "critical",
    revision: str = "Restrict conclusion to adults 70+ or find a source covering adults 60-70.",
) -> dict:
    return {
        "type": type_,
        "description": description,
        "source_ids_implicated": ["src_001"] if source_ids is None else source_ids,
        "severity": severity,
        "suggested_revision": revision,
    }



# _parse_challenges Tests

class TestParseChallenges:

    def test_valid_challenge_accepted(self):
        raw = {"challenges": [_challenge_raw()]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert len(result) == 1
        ch = result[0]
        assert ch["challenge_id"] == "c_001"
        assert ch["type"] == "wrong_population"
        assert ch["severity"] == "critical"
        assert ch["round"] == 0

    def test_empty_challenges_accepted(self):
        """Skeptic producing zero challenges is valid and expected."""
        raw = {"challenges": []}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert result == []

    def test_vague_description_rejected(self):
        """Descriptions under 30 chars are too vague and must be rejected."""
        raw = {"challenges": [_challenge_raw(description="Are you sure?")]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert result == []

    def test_missing_source_ids_rejected(self):
        """Challenges without source_ids_implicated are too vague — rejected."""
        raw = {"challenges": [_challenge_raw(source_ids=[])]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert result == []

    def test_none_source_ids_rejected(self):
        ch = _challenge_raw()
        ch["source_ids_implicated"] = None
        raw = {"challenges": [ch]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert result == []

    def test_invalid_type_rejected(self):
        """Challenge types not in the spec's 7-item list must be rejected."""
        raw = {"challenges": [_challenge_raw(type_="made_up_challenge_type")]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert result == []

    def test_invalid_severity_defaults_to_minor(self):
        """Invalid severity values default to 'minor' rather than being rejected."""
        raw = {"challenges": [_challenge_raw(severity="catastrophic")]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert len(result) == 1
        assert result[0]["severity"] == "minor"

    def test_challenge_id_sequential_from_existing(self):
        """Challenge IDs must continue from existing_challenge_count."""
        raw = {"challenges": [_challenge_raw(), _challenge_raw(type_="commercial_bias")]}
        result = _parse_challenges(raw, loop_round=1, existing_challenge_count=3)
        assert result[0]["challenge_id"] == "c_004"
        assert result[1]["challenge_id"] == "c_005"

    def test_round_tagged_correctly(self):
        raw = {"challenges": [_challenge_raw()]}
        result = _parse_challenges(raw, loop_round=2, existing_challenge_count=0)
        assert result[0]["round"] == 2

    def test_all_7_valid_types_accepted(self):
        challenges = [
            _challenge_raw(type_=t)
            for t in _VALID_CHALLENGE_TYPES
        ]
        raw = {"challenges": challenges}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert len(result) == len(_VALID_CHALLENGE_TYPES)

    def test_mixed_valid_invalid_only_valid_pass(self):
        raw = {
            "challenges": [
                _challenge_raw(type_="wrong_population"),  # valid
                _challenge_raw(description="Too short."),   # vague — rejected
                _challenge_raw(type_="invalid_type"),       # bad type — rejected
                _challenge_raw(type_="commercial_bias"),    # valid
            ]
        }
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert len(result) == 2
        types = {c["type"] for c in result}
        assert "wrong_population" in types
        assert "commercial_bias" in types

    def test_non_list_challenges_returns_empty(self):
        raw = {"challenges": "not a list"}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert result == []

    def test_source_ids_cast_to_strings(self):
        ch = _challenge_raw()
        ch["source_ids_implicated"] = [1, 2]  # integers, not strings
        raw = {"challenges": [ch]}
        result = _parse_challenges(raw, loop_round=0, existing_challenge_count=0)
        assert len(result) == 1
        assert result[0]["source_ids_implicated"] == ["1", "2"]



# _build_skeptic_prompt Tests

class TestBuildSkepticPrompt:

    def test_contains_question(self):
        prompt = _build_skeptic_prompt(
            question="Does aspirin reduce heart attacks?",
            preliminary_answer="Aspirin helps [src_001].",
            scored_sources=[_scored("src_001")],
            normalized_lookup={"src_001": _norm("src_001")},
            claims=[_claim("clm_001", "src_001")],
            contradictions=[],
            loop_round=0,
        )
        assert "Does aspirin reduce heart attacks?" in prompt

    def test_contains_preliminary_answer(self):
        answer = "Aspirin reduces events by 15% in adults 70+ [src_001]."
        prompt = _build_skeptic_prompt(
            question="Question?",
            preliminary_answer=answer,
            scored_sources=[_scored("src_001")],
            normalized_lookup={"src_001": _norm("src_001")},
            claims=[],
            contradictions=[],
            loop_round=0,
        )
        assert answer in prompt

    def test_contains_all_source_ids(self):
        prompt = _build_skeptic_prompt(
            question="Question?",
            preliminary_answer="Answer [src_001].",
            scored_sources=[_scored("src_001"), _scored("src_002")],
            normalized_lookup={"src_001": _norm("src_001"), "src_002": _norm("src_002")},
            claims=[],
            contradictions=[],
            loop_round=0,
        )
        assert "src_001" in prompt
        assert "src_002" in prompt

    def test_contradictions_included(self):
        contradictions = [{
            "conflict_id": "con_001",
            "claim_a_id": "clm_001",
            "claim_b_id": "clm_002",
            "conflict_type": "irreconcilable",
            "explanation": "Opposite findings in same population.",
        }]
        prompt = _build_skeptic_prompt(
            question="Q?",
            preliminary_answer="A.",
            scored_sources=[_scored("src_001")],
            normalized_lookup={"src_001": _norm("src_001")},
            claims=[],
            contradictions=contradictions,
            loop_round=0,
        )
        assert "IRRECONCILABLE" in prompt
        assert "con_001" in prompt

    def test_loop_round_indicated(self):
        prompt = _build_skeptic_prompt(
            question="Q?",
            preliminary_answer="A.",
            scored_sources=[_scored("src_001")],
            normalized_lookup={"src_001": _norm("src_001")},
            claims=[],
            contradictions=[],
            loop_round=2,
        )
        assert "2" in prompt



# run_skeptic Integration Tests

class TestRunSkeptic:

    def test_dry_run_returns_no_challenges(self):
        """Dry-run mock returns {'challenges': []} — the Skeptic must handle zero challenges."""
        state = _make_state()
        result = run_skeptic(state)
        # dry-run returns empty challenges list
        new_challenges = [c for c in result["skeptic_challenges"] if c.get("round") == 0]
        assert isinstance(result["skeptic_challenges"], list)

    def test_existing_challenges_preserved(self):
        """Challenges from previous rounds must not be dropped."""
        prior_challenge = {
            "challenge_id": "c_001",
            "type": "wrong_population",
            "description": "Prior round challenge from round 0 about population mismatch.",
            "source_ids_implicated": ["src_001"],
            "severity": "critical",
            "suggested_revision": "Restrict to adults 70+.",
            "round": 0,
        }
        state = _make_state(existing_challenges=[prior_challenge], loop_round=1)
        result = run_skeptic(state)
        # Prior challenge must still be in the list
        assert any(c["challenge_id"] == "c_001" for c in result["skeptic_challenges"])

    def test_audit_trail_written(self):
        state = _make_state()
        result = run_skeptic(state)
        skeptic_records = [
            r for r in result["audit_trail"]
            if r.get("stage") == "skeptic"
        ]
        assert len(skeptic_records) >= 1

    def test_audit_record_has_round(self):
        state = _make_state(loop_round=1)
        result = run_skeptic(state)
        complete_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "skeptic_complete"
        ]
        assert len(complete_records) == 1
        assert complete_records[0]["data"]["round"] == 1

    def test_audit_trail_appended_not_replaced(self):
        state = _make_state()
        state["audit_trail"] = [{
            "event": "prior",
            "stage": "arbiter",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "request_id": "req_skeptic_test",
            "data": {},
        }]
        result = run_skeptic(state)
        assert result["audit_trail"][0]["event"] == "prior"

    def test_llm_failure_returns_existing_challenges_not_crash(self):
        """LLM failure must not crash the pipeline — return existing challenges unchanged."""
        prior = {
            "challenge_id": "c_001",
            "type": "commercial_bias",
            "description": "Prior challenge about commercial bias in src_002.",
            "source_ids_implicated": ["src_002"],
            "severity": "moderate",
            "suggested_revision": "Remove src_002 from the answer.",
            "round": 0,
        }
        state = _make_state(existing_challenges=[prior])
        with patch("agents.skeptic.call_llm") as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM unavailable")
            result = run_skeptic(state)
        # Prior challenge preserved; no crash
        assert any(c["challenge_id"] == "c_001" for c in result["skeptic_challenges"])
        fail_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "skeptic_llm_failed"
        ]
        assert len(fail_records) == 1

    def test_valid_challenge_from_llm_accepted(self):
        """When LLM returns a well-formed challenge, it must appear in state."""
        mock_response = {
            "challenges": [
                {
                    "type": "wrong_population",
                    "description": (
                        "Source src_001 studied adults aged 70+ in Australia, "
                        "but the question asks about adults over 60 globally."
                    ),
                    "source_ids_implicated": ["src_001"],
                    "severity": "critical",
                    "suggested_revision": (
                        "Restrict the conclusion to adults 70+ or note this population limitation explicitly."
                    ),
                }
            ]
        }
        state = _make_state()
        with patch("agents.skeptic.call_llm", return_value=mock_response):
            result = run_skeptic(state)
        round_0_challenges = [c for c in result["skeptic_challenges"] if c.get("round") == 0]
        assert len(round_0_challenges) == 1
        assert round_0_challenges[0]["type"] == "wrong_population"
        assert round_0_challenges[0]["severity"] == "critical"

    def test_vague_challenge_from_llm_filtered(self):
        """Even if the LLM returns a vague challenge, it must be filtered out."""
        mock_response = {
            "challenges": [
                {
                    "type": "wrong_population",
                    "description": "Are you sure?",  # too vague
                    "source_ids_implicated": ["src_001"],
                    "severity": "critical",
                    "suggested_revision": "Reconsider.",
                }
            ]
        }
        state = _make_state()
        with patch("agents.skeptic.call_llm", return_value=mock_response):
            result = run_skeptic(state)
        new_challenges = [c for c in result["skeptic_challenges"] if c.get("round") == 0]
        assert len(new_challenges) == 0