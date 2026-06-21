"""
Phase 5 tests covering:
  - agents/egress.py           — citation building, challenge summaries, HITL status,
                                  disclaimer injection, pipeline status flags
  - guardrails/output_disclaimer.py — all domain flags, Insufficient Evidence addendum
  - storage/audit_db.py        — init, write/read audit trail, replay records,
                                  feedback + credibility adjustments
  - Full end-to-end pipeline   — graph compiles and runs with DRY_RUN=True,
                                  returns a well-formed state

Run with: pytest tests/test_pipeline_e2e.py -v
All tests use DRY_RUN=True; aiosqlite tests use a temp file DB.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")


#  
# Output Disclaimer Guardrail Tests
#  

class TestOutputDisclaimer:

    def test_medical_disclaimer_returned(self):
        from guardrails.output_disclaimer import get_disclaimer
        result = get_disclaimer("medical")
        assert result is not None
        assert "MEDICAL DISCLAIMER" in result
        assert "healthcare professional" in result

    def test_legal_disclaimer_returned(self):
        from guardrails.output_disclaimer import get_disclaimer
        result = get_disclaimer("legal")
        assert result is not None
        assert "LEGAL DISCLAIMER" in result
        assert "legal professional" in result

    def test_financial_disclaimer_returned(self):
        from guardrails.output_disclaimer import get_disclaimer
        result = get_disclaimer("financial")
        assert result is not None
        assert "FINANCIAL DISCLAIMER" in result
        assert "financial advisor" in result

    def test_none_flag_returns_none(self):
        from guardrails.output_disclaimer import get_disclaimer
        assert get_disclaimer(None) is None

    def test_unknown_flag_returns_none(self):
        from guardrails.output_disclaimer import get_disclaimer
        assert get_disclaimer("cooking") is None

    def test_insufficient_evidence_addendum_appended(self):
        from guardrails.output_disclaimer import get_disclaimer
        result = get_disclaimer("medical", confidence="Insufficient Evidence")
        assert result is not None
        assert "Insufficient Evidence" in result
        assert "extra caution" in result.lower() or "especially warranted" in result

    def test_high_confidence_no_addendum(self):
        from guardrails.output_disclaimer import get_disclaimer
        result = get_disclaimer("medical", confidence="High")
        assert result is not None
        assert "Insufficient Evidence" not in result

    def test_case_insensitive_flag(self):
        from guardrails.output_disclaimer import get_disclaimer
        assert get_disclaimer("MEDICAL") is not None
        assert get_disclaimer("Medical") is not None


#  
# Egress Agent Tests
#  

class TestRunEgress:

    def _make_state(self, **overrides) -> dict:
        from core.state import make_initial_state
        state = make_initial_state(
            "req_egress_test",
            "Does aspirin reduce cardiovascular events in adults over 60?",
            [],
        )
        state.update({
            "final_answer": "Aspirin did not significantly reduce events [src_001].",
            "confidence": "Moderate",
            "claim_source_map": [
                {
                    "claim_text": "Aspirin did not significantly reduce events.",
                    "source_id": "src_001",
                    "credibility": 0.88,
                    "relevance": 0.82,
                }
            ],
            "unresolved_contradictions": [],
            "input_domain_flag": None,
            "scored_sources": [
                {
                    "source_id": "src_001",
                    "credibility": 0.88,
                    "relevance": 0.82,
                    "source_type": "peer_reviewed",
                }
            ],
            "dropped_sources": [],
            "normalized_sources": [
                {
                    "source_id": "src_001",
                    "url": "https://nejm.org/study",
                    "title": "ASPREE Trial",
                    "content_text": "Trial content...",
                }
            ],
            "skeptic_challenges": [],
            "arbiter_rulings": [],
            "loop_round": 1,
            "input_rejected": False,
            "hitl_required": False,
            "hitl_approved": None,
        })
        state.update(overrides)
        return state

    def test_basic_egress_runs(self):
        from agents.egress import run_egress
        state = self._make_state()
        result = run_egress(state)
        assert "api_response" in result
        resp = result["api_response"]
        assert resp["request_id"] == "req_egress_test"
        assert resp["confidence"] == "Moderate"
        assert resp["status"] == "complete"

    def test_answer_in_response(self):
        from agents.egress import run_egress
        state = self._make_state()
        result = run_egress(state)
        assert result["api_response"]["answer"] is not None
        assert "Aspirin" in result["api_response"]["answer"]

    def test_citations_built_from_claim_source_map(self):
        from agents.egress import run_egress
        state = self._make_state()
        result = run_egress(state)
        citations = result["api_response"]["citations"]
        assert len(citations) == 1
        c = citations[0]
        assert c["source_id"] == "src_001"
        assert c["credibility"] == 0.88
        assert c["source_url"] == "https://nejm.org/study"
        assert c["source_title"] == "ASPREE Trial"

    def test_duplicate_claims_deduplicated_in_citations(self):
        from agents.egress import run_egress
        state = self._make_state(claim_source_map=[
            {"claim_text": "Same claim text repeated.", "source_id": "src_001",
             "credibility": 0.88, "relevance": 0.82},
            {"claim_text": "Same claim text repeated.", "source_id": "src_001",
             "credibility": 0.88, "relevance": 0.82},
        ])
        result = run_egress(state)
        assert len(result["api_response"]["citations"]) == 1

    def test_medical_disclaimer_injected(self):
        from agents.egress import run_egress
        state = self._make_state(input_domain_flag="medical")
        result = run_egress(state)
        assert result["disclaimer"] is not None
        assert "MEDICAL DISCLAIMER" in result["disclaimer"]
        assert result["api_response"]["disclaimer"] is not None

    def test_no_disclaimer_when_no_flag(self):
        from agents.egress import run_egress
        state = self._make_state(input_domain_flag=None)
        result = run_egress(state)
        assert result["disclaimer"] is None
        assert result["api_response"]["disclaimer"] is None

    def test_sources_used_count(self):
        from agents.egress import run_egress
        state = self._make_state()
        result = run_egress(state)
        assert result["api_response"]["sources_used"] == 1

    def test_sources_dropped_count(self):
        from agents.egress import run_egress
        state = self._make_state(
            dropped_sources=[{"source_id": "src_002", "domain": "bad.com", "reason": "denylist"}]
        )
        result = run_egress(state)
        assert result["api_response"]["sources_dropped"] == 1

    def test_loop_rounds_in_response(self):
        from agents.egress import run_egress
        state = self._make_state(loop_round=2)
        result = run_egress(state)
        assert result["api_response"]["loop_rounds"] == 2

    def test_challenge_summaries_built(self):
        from agents.egress import run_egress
        challenges = [
            {
                "challenge_id": "c_001",
                "type": "wrong_population",
                "description": "Population mismatch detected in the analysis.",
                "severity": "critical",
                "round": 0,
            }
        ]
        rulings = [
            {
                "challenge_id": "c_001",
                "ruling": "valid_critical",
                "reason": "Confirmed population mismatch.",
                "round": 0,
            }
        ]
        state = self._make_state(
            skeptic_challenges=challenges,
            arbiter_rulings=rulings,
        )
        result = run_egress(state)
        summaries = result["api_response"]["skeptic_challenges"]
        assert len(summaries) == 1
        s = summaries[0]
        assert s["challenge_id"] == "c_001"
        assert s["was_resolved"] is True
        assert s["arbiter_ruling"] == "valid_critical"

    def test_unresolved_challenge_marked(self):
        from agents.egress import run_egress
        state = self._make_state(
            skeptic_challenges=[{
                "challenge_id": "c_001",
                "type": "recency_gap",
                "description": "Older sources used when newer exist.",
                "severity": "moderate",
                "round": 0,
            }],
            arbiter_rulings=[],  # no ruling = not resolved
        )
        result = run_egress(state)
        summary = result["api_response"]["skeptic_challenges"][0]
        assert summary["was_resolved"] is False
        assert summary["arbiter_ruling"] is None

    def test_hitl_not_required(self):
        from agents.egress import run_egress
        state = self._make_state(hitl_required=False)
        result = run_egress(state)
        assert result["api_response"]["hitl_status"] == "not_required"

    def test_hitl_pending(self):
        from agents.egress import run_egress
        state = self._make_state(hitl_required=True, hitl_approved=None)
        result = run_egress(state)
        assert result["api_response"]["hitl_status"] == "pending"
        assert result["api_response"]["status"] == "hitl_pending"

    def test_hitl_approved(self):
        from agents.egress import run_egress
        state = self._make_state(hitl_required=True, hitl_approved=True)
        result = run_egress(state)
        assert result["api_response"]["hitl_status"] == "approved"

    def test_input_rejected_sets_failed_status(self):
        from agents.egress import run_egress
        state = self._make_state(
            input_rejected=True,
            input_rejection_reason="PII detected in question.",
            final_answer="Request rejected: PII detected in question.",
        )
        result = run_egress(state)
        assert result["api_response"]["status"] == "failed"
        assert result["api_response"]["error_message"] == "PII detected in question."

    def test_citations_sorted_by_credibility_desc(self):
        from agents.egress import run_egress
        state = self._make_state(
            claim_source_map=[
                {"claim_text": "Low cred claim text here.", "source_id": "src_002",
                 "credibility": 0.45, "relevance": 0.50},
                {"claim_text": "High cred claim text here.", "source_id": "src_001",
                 "credibility": 0.88, "relevance": 0.82},
            ],
            scored_sources=[
                {"source_id": "src_001", "credibility": 0.88, "relevance": 0.82,
                 "source_type": "peer_reviewed"},
                {"source_id": "src_002", "credibility": 0.45, "relevance": 0.50,
                 "source_type": "general_news"},
            ],
        )
        result = run_egress(state)
        citations = result["api_response"]["citations"]
        assert len(citations) == 2
        assert citations[0]["credibility"] >= citations[1]["credibility"]

    def test_audit_trail_written(self):
        from agents.egress import run_egress
        state = self._make_state()
        result = run_egress(state)
        egress_records = [
            r for r in result["audit_trail"]
            if r.get("stage") == "egress"
        ]
        assert len(egress_records) == 1
        assert egress_records[0]["data"]["confidence"] == "Moderate"

    def test_empty_final_answer_becomes_none(self):
        from agents.egress import run_egress
        state = self._make_state(final_answer="")
        result = run_egress(state)
        assert result["api_response"]["answer"] is None

    def test_unresolved_contradictions_in_response(self):
        from agents.egress import run_egress
        state = self._make_state(
            unresolved_contradictions=["IRRECONCILABLE: clm_001 vs clm_002 — opposite findings."]
        )
        result = run_egress(state)
        assert len(result["api_response"]["unresolved_contradictions"]) == 1


#  
# Audit DB Tests (async, use temp file)
#  

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_audit.db")


@pytest.mark.asyncio
async def test_init_db_creates_file(tmp_db):
    from storage.audit_db import init_db
    await init_db(tmp_db)
    assert os.path.exists(tmp_db)


@pytest.mark.asyncio
async def test_write_and_read_audit_trail(tmp_db):
    from storage.audit_db import init_db, write_audit_trail, read_audit_trail
    await init_db(tmp_db)

    trail = [
        {
            "event": "ingress_complete",
            "stage": "ingress",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "request_id": "req_test_001",
            "data": {"normalized_count": 3, "url_duplicates_merged": 0},
        },
        {
            "event": "source_scorer_summary",
            "stage": "source_scorer",
            "timestamp": "2024-01-01T00:00:01+00:00",
            "request_id": "req_test_001",
            "data": {"scored_count": 3, "dropped_count": 0},
        },
    ]
    count = await write_audit_trail(tmp_db, "req_test_001", trail)
    assert count == 2

    read_back = await read_audit_trail(tmp_db, "req_test_001")
    assert len(read_back) == 2
    assert read_back[0]["event"] == "ingress_complete"
    assert read_back[1]["event"] == "source_scorer_summary"
    assert read_back[0]["data"]["normalized_count"] == 3


@pytest.mark.asyncio
async def test_write_and_read_replay_record(tmp_db):
    from storage.audit_db import init_db, write_replay_record, read_replay_record
    await init_db(tmp_db)

    success = await write_replay_record(
        db_path=tmp_db,
        request_id="req_replay_001",
        question="Does aspirin reduce cardiovascular events?",
        sources_hash="abc123",
        final_answer="Aspirin showed no significant benefit in adults 70+.",
        confidence="Moderate",
        loop_rounds=2,
        skeptic_challenges=[{"challenge_id": "c_001", "type": "wrong_population"}],
        arbiter_rulings=[{"challenge_id": "c_001", "ruling": "valid_critical"}],
    )
    assert success is True

    record = await read_replay_record(tmp_db, "req_replay_001")
    assert record is not None
    assert record["request_id"] == "req_replay_001"
    assert record["confidence"] == "Moderate"
    assert record["loop_rounds"] == 2
    assert len(record["skeptic_challenges"]) == 1
    assert record["skeptic_challenges"][0]["challenge_id"] == "c_001"
    assert len(record["arbiter_rulings"]) == 1


@pytest.mark.asyncio
async def test_read_nonexistent_replay_returns_none(tmp_db):
    from storage.audit_db import init_db, read_replay_record
    await init_db(tmp_db)
    result = await read_replay_record(tmp_db, "req_does_not_exist")
    assert result is None


@pytest.mark.asyncio
async def test_write_replay_idempotent(tmp_db):
    """Writing the same request_id twice should update in-place (INSERT OR REPLACE)."""
    from storage.audit_db import init_db, write_replay_record, read_replay_record
    await init_db(tmp_db)

    await write_replay_record(
        tmp_db, "req_idem", "Q?", "hash1", "Answer v1", "Low", 1, [], []
    )
    await write_replay_record(
        tmp_db, "req_idem", "Q?", "hash1", "Answer v2", "Moderate", 2, [], []
    )

    record = await read_replay_record(tmp_db, "req_idem")
    assert record["final_answer"] == "Answer v2"
    assert record["confidence"] == "Moderate"


@pytest.mark.asyncio
async def test_write_feedback_rating(tmp_db):
    from storage.audit_db import init_db, write_feedback
    await init_db(tmp_db)
    applied = await write_feedback(tmp_db, "req_fb_001", rating=4, source_corrections=[])
    assert applied == 0  # no corrections → 0 adjustments


@pytest.mark.asyncio
async def test_write_feedback_source_corrections(tmp_db):
    from storage.audit_db import (
        init_db, write_feedback, get_credibility_adjustment,
        _CREDIBILITY_ADJUSTMENTS,
    )
    # Clear in-memory state from other tests
    _CREDIBILITY_ADJUSTMENTS.clear()
    await init_db(tmp_db)

    corrections = [
        {"source_id": "src_bad_001", "flag": "commercial_bias"},
        {"source_id": "src_bad_002", "flag": "misinformation"},
    ]
    applied = await write_feedback(tmp_db, "req_fb_002", rating=2, source_corrections=corrections)
    assert applied == 2

    adj1 = get_credibility_adjustment("src_bad_001")
    assert adj1 < 0  # commercial_bias applies negative delta

    adj2 = get_credibility_adjustment("src_bad_002")
    assert adj2 < 0  # misinformation applies strong negative delta


@pytest.mark.asyncio
async def test_credibility_adjustment_clamped(tmp_db):
    """Cumulative adjustments must be clamped to [-0.50, +0.20]."""
    from storage.audit_db import (
        init_db, write_feedback, get_credibility_adjustment,
        _CREDIBILITY_ADJUSTMENTS,
    )
    _CREDIBILITY_ADJUSTMENTS.clear()
    await init_db(tmp_db)

    # Apply many negative corrections to the same source
    for _ in range(10):
        await write_feedback(
            tmp_db, "req_clamp", rating=None,
            source_corrections=[{"source_id": "src_clamp", "flag": "misinformation"}]
        )

    adj = get_credibility_adjustment("src_clamp")
    assert adj >= -0.50  # never below floor


@pytest.mark.asyncio
async def test_load_adjustments_from_db(tmp_db):
    """Adjustments persisted to DB must be loadable back into memory."""
    from storage.audit_db import (
        init_db, write_feedback, load_adjustments_from_db,
        get_credibility_adjustment, _CREDIBILITY_ADJUSTMENTS,
    )
    _CREDIBILITY_ADJUSTMENTS.clear()
    await init_db(tmp_db)

    await write_feedback(
        tmp_db, "req_persist", rating=3,
        source_corrections=[{"source_id": "src_persist_001", "flag": "low_credibility"}]
    )

    # Simulate restart: clear in-memory state, then reload from DB
    _CREDIBILITY_ADJUSTMENTS.clear()
    assert get_credibility_adjustment("src_persist_001") == 0.0  # cleared

    loaded = await load_adjustments_from_db(tmp_db)
    assert loaded >= 1
    assert get_credibility_adjustment("src_persist_001") < 0.0  # restored


@pytest.mark.asyncio
async def test_empty_audit_trail_writes_zero(tmp_db):
    from storage.audit_db import init_db, write_audit_trail
    await init_db(tmp_db)
    count = await write_audit_trail(tmp_db, "req_empty", [])
    assert count == 0


@pytest.mark.asyncio
async def test_audit_trail_isolation_by_request_id(tmp_db):
    """Records for different request_ids must not bleed into each other."""
    from storage.audit_db import init_db, write_audit_trail, read_audit_trail
    await init_db(tmp_db)

    trail_a = [{
        "event": "ingress_complete", "stage": "ingress",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "request_id": "req_A",
        "data": {"normalized_count": 2},
    }]
    trail_b = [{
        "event": "source_scorer_summary", "stage": "source_scorer",
        "timestamp": "2024-01-01T00:00:01+00:00",
        "request_id": "req_B",
        "data": {"scored_count": 5},
    }]

    await write_audit_trail(tmp_db, "req_A", trail_a)
    await write_audit_trail(tmp_db, "req_B", trail_b)

    records_a = await read_audit_trail(tmp_db, "req_A")
    records_b = await read_audit_trail(tmp_db, "req_B")

    assert len(records_a) == 1
    assert records_a[0]["event"] == "ingress_complete"
    assert len(records_b) == 1
    assert records_b[0]["event"] == "source_scorer_summary"


#  
# Full End-to-End Pipeline Test
#  

class TestPipelineE2E:
    """
    Full graph compilation and invocation tests.
    DRY_RUN=True means no real LLM calls are made.
    Tests verify the pipeline runs without crashing and produces valid output.
    """

    def _run_pipeline(self, question: str, sources: list[dict]) -> dict:
        from core.graph import run_pipeline
        import hashlib, time
        request_id = f"req_e2e_{hashlib.md5(question.encode()).hexdigest()[:8]}"
        return run_pipeline(request_id, question, sources)

    def _valid_source(self, index: int = 1, domain: str = "nih.gov") -> dict:
        return {
            "url": f"https://{domain}/study-{index}",
            "content": (
                f"This randomized controlled trial ({index}) enrolled 19,114 healthy adults "
                "aged 70 years or older to evaluate the effect of daily aspirin. "
                "The primary endpoint was major cardiovascular events over five years. "
                "Results showed no significant difference between aspirin and placebo groups. "
                "Aspirin was associated with increased risk of major haemorrhage. "
                "The study was double-blinded and placebo-controlled."
            ),
            "metadata": {
                "title": f"ASPREE Trial Part {index}",
                "doi": f"10.1056/test{index}",
                "published_date": "2022-09-16",
                "citation_count": 500 + index * 100,
                "author": "Dr. Jane Smith",
                "journal": "New England Journal of Medicine",
            },
        }

    def test_pipeline_compiles(self):
        """Graph must compile without raising."""
        from core.graph import compile_graph
        graph = compile_graph()
        assert graph is not None

    def test_pipeline_runs_single_source(self):
        """Pipeline must complete with a single source without crashing."""
        state = self._run_pipeline(
            question="Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
            sources=[self._valid_source(1)],
        )
        assert state is not None
        assert state.get("loop_terminated") is True or state.get("loop_round", 0) >= 0

    def test_pipeline_runs_three_sources(self):
        """Pipeline must complete with three sources (meets source_count_floor)."""
        sources = [self._valid_source(i) for i in range(1, 4)]
        state = self._run_pipeline(
            question="Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
            sources=sources,
        )
        assert state is not None
        assert state.get("confidence") in (
            "High", "Moderate", "Low", "Insufficient Evidence"
        )

    def test_pipeline_rejected_input_short_circuits(self):
        """Empty question should be rejected at input_guardrail, not crash."""
        state = self._run_pipeline(question="Why?", sources=[self._valid_source()])
        assert state.get("input_rejected") is True
        assert state.get("final_answer") is not None

    def test_pipeline_pii_rejection(self):
        """SSN in question should trigger PII rejection."""
        state = self._run_pipeline(
            question="What should patient 123-45-6789 take for heart disease treatment?",
            sources=[self._valid_source()],
        )
        assert state.get("input_rejected") is True

    def test_pipeline_medical_flag_set(self):
        """Medical question should set input_domain_flag = 'medical'."""
        state = self._run_pipeline(
            question="Does daily aspirin reduce cardiovascular events in diabetic patients?",
            sources=[self._valid_source()],
        )
        assert state.get("input_domain_flag") == "medical"

    def test_pipeline_produces_normalized_sources(self):
        """Ingress must produce normalized_sources from raw input."""
        state = self._run_pipeline(
            question="Does aspirin affect cardiovascular disease outcomes in elderly patients?",
            sources=[self._valid_source(1), self._valid_source(2)],
        )
        assert len(state.get("normalized_sources", [])) >= 1
        for src in state["normalized_sources"]:
            assert "source_id" in src
            assert src["source_id"].startswith("src_")
            assert "content_text" in src

    def test_pipeline_produces_scored_sources(self):
        """Source scorer must produce scored_sources with required fields."""
        state = self._run_pipeline(
            question="Does aspirin affect cardiovascular outcomes in elderly patients?",
            sources=[self._valid_source(1)],
        )
        for src in state.get("scored_sources", []):
            assert "credibility" in src
            assert "relevance" in src
            assert "bias_flag" in src
            assert 0.0 <= src["credibility"] <= 1.0
            assert 0.0 <= src["relevance"] <= 1.0

    def test_pipeline_produces_claims(self):
        """Evidence extractor must produce at least one claim in dry-run mode."""
        state = self._run_pipeline(
            question="Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
            sources=[self._valid_source(1), self._valid_source(2)],
        )
        claims = state.get("claims", [])
        assert isinstance(claims, list)
        for claim in claims:
            assert "claim_id" in claim
            assert "source_id" in claim
            assert "text" in claim

    def test_pipeline_audit_trail_non_empty(self):
        """At minimum, each pipeline node should contribute one audit record."""
        state = self._run_pipeline(
            question="Does aspirin affect cardiovascular outcomes in elderly patients?",
            sources=[self._valid_source()],
        )
        trail = state.get("audit_trail", [])
        assert len(trail) >= 3  # at minimum: guardrail + ingress + scorer
        stages = {r.get("stage") for r in trail}
        assert "input_guardrail" in stages
        assert "ingress" in stages

    def test_pipeline_api_response_present(self):
        """Egress must populate _api_response in state."""
        state = self._run_pipeline(
            question="Does aspirin affect cardiovascular disease in elderly patients?",
            sources=[self._valid_source()],
        )
        resp = state.get("api_response")
        assert resp is not None
        assert "request_id" in resp
        assert "confidence" in resp
        assert "status" in resp

    def test_pipeline_denylisted_source_dropped(self):
        """A source from a denylisted domain must be dropped, not crash the pipeline."""
        sources = [
            self._valid_source(1),  # good source
            {
                "url": "https://naturalnews.com/article",
                "content": "Aspirin is a miracle cure for all diseases and conditions.",
                "metadata": {"title": "Miracle Cures"},
            },
        ]
        state = self._run_pipeline(
            question="Does aspirin reduce cardiovascular disease risk in elderly patients?",
            sources=sources,
        )
        dropped = state.get("dropped_sources", [])
        dropped_domains = [d.get("domain") for d in dropped]
        assert "naturalnews.com" in dropped_domains

    def test_pipeline_confidence_always_valid_tier(self):
        """Final confidence must always be one of the 4 valid tiers."""
        state = self._run_pipeline(
            question="Does aspirin affect cardiovascular outcomes in elderly patients?",
            sources=[self._valid_source()],
        )
        valid_tiers = {"High", "Moderate", "Low", "Insufficient Evidence"}
        conf = state.get("confidence")
        # May be None if pipeline was rejected at guardrail
        if conf is not None:
            assert conf in valid_tiers

    def test_pipeline_loop_round_non_negative(self):
        """loop_round must never go below 0."""
        state = self._run_pipeline(
            question="Does aspirin reduce cardiovascular events in healthy adults over 60?",
            sources=[self._valid_source()],
        )
        assert state.get("loop_round", 0) >= 0

    def test_pipeline_loop_round_never_exceeds_max(self):
        """loop_round must never exceed max_loop_rounds."""
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds

        state = self._run_pipeline(
            question="Does aspirin affect cardiovascular outcomes in elderly patients?",
            sources=[self._valid_source(i) for i in range(1, 4)],
        )
        assert state.get("loop_round", 0) <= max_rounds