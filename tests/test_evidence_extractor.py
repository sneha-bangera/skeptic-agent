"""
Unit tests for agents/evidence_extractor.py

All tests use DRY_RUN=True — the LLM dry-run mock returns:
  {"claims": [{"text": "This is a mock claim...", "population": "general", ...}]}
for each source in the batch.

Run with: pytest tests/test_evidence_extractor.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")

from core.state import make_initial_state
from agents.evidence_extractor import (
    run_evidence_extractor,
    _make_claim_id,
    _parse_claims_response,
    _build_extraction_prompt,
    BATCH_SIZE,
)


# Fixtures 

def _make_state(
    scored_sources: list | None = None,
    normalized_sources: list | None = None,
) -> dict:
    state = make_initial_state(
        "req_ee_test",
        "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
        [],
    )
    state["scored_sources"] = scored_sources or []
    state["normalized_sources"] = normalized_sources or []
    return state


def _scored(source_id: str, credibility: float = 0.80, relevance: float = 0.75) -> dict:
    return {
        "source_id": source_id,
        "credibility": credibility,
        "relevance": relevance,
        "source_type": "peer_reviewed",
        "trusted_source": True,
        "bias_flag": "none",
        "score_breakdown": {},
    }


def _normalized(source_id: str, title: str = "Test Study") -> dict:
    return {
        "source_id": source_id,
        "url": f"https://nih.gov/{source_id}",
        "title": title,
        "author": "Dr. Test",
        "domain": "nih.gov",
        "published_date": "2023-01-01",
        "content_text": (
            f"This randomized controlled trial ({source_id}) enrolled 5,000 participants. "
            "The primary endpoint was major cardiovascular events over 24 months. "
            "Results showed a statistically significant reduction in events (p=0.02). "
            "Aspirin 100mg daily was associated with a 15% relative risk reduction."
        ),
        "metadata": {"citation_count": 50},
    }


#  
# Helper Function Tests
#  

class TestClaimIdGeneration:

    def test_format(self):
        assert _make_claim_id(1) == "clm_001"
        assert _make_claim_id(10) == "clm_010"
        assert _make_claim_id(100) == "clm_100"

    def test_sequential(self):
        ids = [_make_claim_id(i) for i in range(1, 6)]
        assert ids == ["clm_001", "clm_002", "clm_003", "clm_004", "clm_005"]


class TestParseClaimsResponse:

    def _settings_threshold(self):
        from config.settings import get_settings
        return get_settings().min_credibility_threshold

    def test_valid_response_parsed(self):
        raw = {
            "claims": [
                {
                    "source_id": "src_001",
                    "text": "Aspirin reduces cardiovascular events by 15% in elderly adults.",
                    "population": "adults aged 70+",
                    "timeframe": "24 months",
                    "outcome_metric": "major cardiovascular events",
                    "study_design": "RCT",
                }
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.90},
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert len(result) == 1
        claim = result[0]
        assert claim["claim_id"] == "clm_001"
        assert claim["source_id"] == "src_001"
        assert "15%" in claim["text"]
        assert claim["population"] == "adults aged 70+"
        assert claim["low_credibility"] is False

    def test_low_credibility_flagged(self):
        raw = {
            "claims": [
                {
                    "source_id": "src_001",
                    "text": "Aspirin dramatically helps heart disease in everyone.",
                    "population": None,
                    "timeframe": None,
                    "outcome_metric": "heart disease",
                    "study_design": None,
                }
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.15},  # below threshold
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert result[0]["low_credibility"] is True

    def test_unexpected_source_id_skipped(self):
        """Claims with source IDs not in the batch must be silently dropped."""
        raw = {
            "claims": [
                {
                    "source_id": "src_999",  # not in batch
                    "text": "This claim should be dropped.",
                    "population": None,
                    "timeframe": None,
                    "outcome_metric": None,
                    "study_design": None,
                }
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.80},
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert result == []

    def test_empty_text_skipped(self):
        raw = {
            "claims": [
                {
                    "source_id": "src_001",
                    "text": "",  # empty
                    "population": None,
                    "timeframe": None,
                    "outcome_metric": None,
                    "study_design": None,
                }
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.80},
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert result == []

    def test_too_short_text_skipped(self):
        raw = {
            "claims": [
                {
                    "source_id": "src_001",
                    "text": "Yes.",
                    "population": None, "timeframe": None,
                    "outcome_metric": None, "study_design": None,
                }
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.80},
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert result == []

    def test_counter_continues_from_start(self):
        raw = {
            "claims": [
                {
                    "source_id": "src_001",
                    "text": "First claim about aspirin and cardiovascular outcomes.",
                    "population": None, "timeframe": None,
                    "outcome_metric": None, "study_design": None,
                },
                {
                    "source_id": "src_001",
                    "text": "Second claim about aspirin and bleeding risk in elderly.",
                    "population": None, "timeframe": None,
                    "outcome_metric": None, "study_design": None,
                },
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.80},
            min_credibility=0.20,
            claim_counter_start=5,  # continuing from previous batch
        )
        assert result[0]["claim_id"] == "clm_005"
        assert result[1]["claim_id"] == "clm_006"

    def test_non_list_claims_returns_empty(self):
        raw = {"claims": "not a list"}
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={},
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert result == []

    def test_null_scope_tags_become_none(self):
        raw = {
            "claims": [
                {
                    "source_id": "src_001",
                    "text": "Aspirin reduces cardiovascular risk in adults significantly.",
                    "population": "",      # empty string → should become None
                    "timeframe": None,
                    "outcome_metric": None,
                    "study_design": None,
                }
            ]
        }
        result = _parse_claims_response(
            raw=raw,
            batch_source_ids={"src_001"},
            credibility_lookup={"src_001": 0.80},
            min_credibility=0.20,
            claim_counter_start=1,
        )
        assert result[0]["population"] is None


class TestBuildExtractionPrompt:

    def test_contains_question(self):
        scored = [_scored("src_001")]
        source_lookup = {"src_001": _normalized("src_001")}
        prompt = _build_extraction_prompt(
            "Does aspirin help heart disease?",
            scored,
            source_lookup,
        )
        assert "Does aspirin help heart disease?" in prompt

    def test_contains_source_id(self):
        scored = [_scored("src_001")]
        source_lookup = {"src_001": _normalized("src_001")}
        prompt = _build_extraction_prompt("Question?", scored, source_lookup)
        assert "src_001" in prompt

    def test_content_truncated_to_1200_chars(self):
        long_content = "X" * 5000
        normalized = _normalized("src_001")
        normalized["content_text"] = long_content
        scored = [_scored("src_001")]
        prompt = _build_extraction_prompt("Question?", scored, {"src_001": normalized})
        # Content in prompt should not contain all 5000 X's
        assert prompt.count("X") <= 1200



# Integration Tests — run_evidence_extractor

class TestRunEvidenceExtractor:

    def test_basic_extraction_single_source(self):
        state = _make_state(
            scored_sources=[_scored("src_001")],
            normalized_sources=[_normalized("src_001")],
        )
        result = run_evidence_extractor(state)
        # Dry-run returns at least 1 mock claim
        assert len(result["claims"]) >= 1
        claim = result["claims"][0]
        assert claim["claim_id"].startswith("clm_")
        assert claim["source_id"] == "src_001"
        assert isinstance(claim["text"], str)
        assert len(claim["text"]) >= 10

    def test_claims_have_required_fields(self):
        state = _make_state(
            scored_sources=[_scored("src_001")],
            normalized_sources=[_normalized("src_001")],
        )
        result = run_evidence_extractor(state)
        for claim in result["claims"]:
            for field in ["claim_id", "source_id", "text", "low_credibility"]:
                assert field in claim, f"Missing required field: {field}"

    def test_low_credibility_source_extracted_and_flagged(self):
        """Sources below threshold should still be extracted but flagged."""
        state = _make_state(
            scored_sources=[
                _scored("src_001", credibility=0.80, relevance=0.75),  # usable
                _scored("src_002", credibility=0.10, relevance=0.05),  # below threshold
            ],
            normalized_sources=[
                _normalized("src_001"),
                _normalized("src_002", title="Low Quality Source"),
            ],
        )
        result = run_evidence_extractor(state)
        # Both sources should have claims extracted
        source_ids_in_claims = {c["source_id"] for c in result["claims"]}
        assert "src_001" in source_ids_in_claims
        assert "src_002" in source_ids_in_claims

        # Claims from src_002 should be flagged
        src_002_claims = [c for c in result["claims"] if c["source_id"] == "src_002"]
        assert all(c["low_credibility"] is True for c in src_002_claims)

    def test_usable_source_not_flagged_as_low_credibility(self):
        state = _make_state(
            scored_sources=[_scored("src_001", credibility=0.80, relevance=0.75)],
            normalized_sources=[_normalized("src_001")],
        )
        result = run_evidence_extractor(state)
        src_001_claims = [c for c in result["claims"] if c["source_id"] == "src_001"]
        assert all(c["low_credibility"] is False for c in src_001_claims)

    def test_empty_scored_sources_returns_empty_claims(self):
        state = _make_state(scored_sources=[], normalized_sources=[])
        result = run_evidence_extractor(state)
        assert result["claims"] == []

    def test_claim_ids_are_sequential(self):
        """claim IDs must be sequential with no gaps."""
        state = _make_state(
            scored_sources=[
                _scored("src_001"),
                _scored("src_002"),
                _scored("src_003"),
            ],
            normalized_sources=[
                _normalized("src_001"),
                _normalized("src_002"),
                _normalized("src_003"),
            ],
        )
        result = run_evidence_extractor(state)
        claims = result["claims"]
        if len(claims) > 1:
            # Extract numeric parts and verify they're sequential
            nums = [int(c["claim_id"].split("_")[1]) for c in claims]
            assert nums == list(range(nums[0], nums[0] + len(nums)))

    def test_batching_multiple_sources(self):
        """5 sources = 2 batches (3+2 with BATCH_SIZE=3). No crash."""
        sources_scored = [_scored(f"src_{str(i).zfill(3)}") for i in range(1, 6)]
        sources_norm = [_normalized(f"src_{str(i).zfill(3)}") for i in range(1, 6)]
        state = _make_state(
            scored_sources=sources_scored,
            normalized_sources=sources_norm,
        )
        # Should not raise
        result = run_evidence_extractor(state)
        assert isinstance(result["claims"], list)

    def test_audit_trail_has_summary(self):
        state = _make_state(
            scored_sources=[_scored("src_001")],
            normalized_sources=[_normalized("src_001")],
        )
        result = run_evidence_extractor(state)
        summary_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "evidence_extractor_summary"
        ]
        assert len(summary_records) == 1
        summary = summary_records[0]["data"]
        assert "total_claims_extracted" in summary
        assert "total_sources_processed" in summary
        assert "usable_sources" in summary

    def test_audit_trail_appended_not_replaced(self):
        """audit_trail from previous stages must be preserved."""
        state = _make_state(
            scored_sources=[_scored("src_001")],
            normalized_sources=[_normalized("src_001")],
        )
        existing_record = {
            "event": "previous_stage",
            "stage": "ingress",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "request_id": "req_ee_test",
            "data": {},
        }
        state["audit_trail"] = [existing_record]
        result = run_evidence_extractor(state)
        assert result["audit_trail"][0]["event"] == "previous_stage"
        assert len(result["audit_trail"]) > 1

    def test_batch_audit_records_written(self):
        state = _make_state(
            scored_sources=[_scored("src_001"), _scored("src_002")],
            normalized_sources=[_normalized("src_001"), _normalized("src_002")],
        )
        result = run_evidence_extractor(state)
        batch_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "batch_extracted"
        ]
        assert len(batch_records) >= 1

    def test_scope_tags_present_in_dry_run(self):
        """Dry-run mock should return claims with scope tags."""
        state = _make_state(
            scored_sources=[_scored("src_001")],
            normalized_sources=[_normalized("src_001")],
        )
        result = run_evidence_extractor(state)
        claims = result["claims"]
        assert len(claims) >= 1
        # All four scope tag keys must be present (values may be None)
        for claim in claims:
            assert "population" in claim
            assert "timeframe" in claim
            assert "outcome_metric" in claim
            assert "study_design" in claim

    def test_no_normalized_source_content_graceful(self):
        """If normalized source is missing, extraction should still attempt (graceful)."""
        state = _make_state(
            scored_sources=[_scored("src_001")],
            normalized_sources=[],  # no normalized source available
        )
        # Should not raise
        result = run_evidence_extractor(state)
        assert isinstance(result["claims"], list)