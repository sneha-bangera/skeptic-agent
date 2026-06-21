"""
Phase 6 tests covering all 5 API endpoints via FastAPI TestClient.

All tests use DRY_RUN=True so no real LLM calls are made.
The TestClient runs requests synchronously against the real app
(including the full LangGraph pipeline).

Run with: pytest tests/test_api_routes.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")

from fastapi.testclient import TestClient


# App fixture                 

@pytest.fixture(scope="module")
def tmp_db_path(tmp_path_factory):
    """Temporary SQLite DB path, scoped to the whole test module."""
    return str(tmp_path_factory.mktemp("db") / "test_audit.db")


@pytest.fixture(scope="module")
def client(tmp_db_path):
    """
    FastAPI TestClient with a temporary DB.
    Module-scoped so the graph is compiled only once (expensive).
    """
    os.environ["AUDIT_DB_PATH"] = tmp_db_path
    # Reset settings singleton so it picks up the temp DB path
    import config.settings as cs
    cs._settings = None

    from main import app
    # Reset the compiled graph cache so each test module gets a fresh one
    import api.dependencies as deps
    deps._compiled_graph = None

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# Test data                 

def _valid_body(question: str | None = None, source_count: int = 1) -> dict:
    question = question or "Does daily aspirin reduce cardiovascular events in healthy adults over 60?"
    sources = [
        {
            "url": f"https://nih.gov/study-{i}",
            "content": (
                f"Randomized controlled trial {i} enrolled 10,000 healthy adults "
                "aged 65 or older. Participants received 100mg aspirin daily or placebo "
                "over five years. Primary endpoint was major cardiovascular events. "
                "Results showed no significant reduction (HR 0.97, 95% CI 0.85-1.10). "
                "Aspirin significantly increased major bleeding events."
            ),
            "metadata": {
                "title": f"Aspirin Trial {i}",
                "doi": f"10.1056/test{i}",
                "published_date": "2022-09-16",
                "citation_count": 400 + i * 50,
                "author": "Dr. Jane Smith",
                "journal": "New England Journal of Medicine",
            },
        }
        for i in range(1, source_count + 1)
    ]
    return {"question": question, "sources": sources}


 
# Health Check
 

class TestHealthCheck:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


 
# POST /analyze
 

class TestAnalyzeEndpoint:

    def test_analyze_returns_200(self, client):
        resp = client.post("/analyze", json=_valid_body())
        assert resp.status_code == 200

    def test_analyze_response_has_required_fields(self, client):
        resp = client.post("/analyze", json=_valid_body())
        data = resp.json()
        required = [
            "request_id", "question", "confidence", "status",
            "sources_used", "sources_dropped", "loop_rounds",
            "hitl_status", "citations", "unresolved_contradictions",
            "skeptic_challenges",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_analyze_request_id_format(self, client):
        resp = client.post("/analyze", json=_valid_body())
        data = resp.json()
        assert data["request_id"].startswith("req_")

    def test_analyze_question_echoed(self, client):
        q = "Does daily aspirin reduce cardiovascular events in healthy adults over 60?"
        resp = client.post("/analyze", json=_valid_body(question=q))
        assert resp.json()["question"] == q

    def test_analyze_valid_confidence_tier(self, client):
        resp = client.post("/analyze", json=_valid_body())
        confidence = resp.json()["confidence"]
        valid = {"High", "Moderate", "Low", "Insufficient Evidence"}
        assert confidence in valid

    def test_analyze_valid_status(self, client):
        resp = client.post("/analyze", json=_valid_body())
        status = resp.json()["status"]
        assert status in {"complete", "hitl_pending", "failed"}

    def test_analyze_valid_hitl_status(self, client):
        resp = client.post("/analyze", json=_valid_body())
        hitl = resp.json()["hitl_status"]
        assert hitl in {"not_required", "pending", "approved", "returned_for_revision"}

    def test_analyze_sources_used_count(self, client):
        resp = client.post("/analyze", json=_valid_body(source_count=2))
        data = resp.json()
        # sources_used counts scored sources; both should pass (nih.gov is trusted)
        assert data["sources_used"] >= 1

    def test_analyze_citations_are_list(self, client):
        resp = client.post("/analyze", json=_valid_body())
        assert isinstance(resp.json()["citations"], list)

    def test_analyze_citation_has_required_fields(self, client):
        resp = client.post("/analyze", json=_valid_body())
        citations = resp.json()["citations"]
        for c in citations:
            assert "claim_text" in c
            assert "source_id" in c
            assert "credibility" in c
            assert "relevance" in c

    def test_analyze_idempotent_same_request_returns_cached(self, client):
        """Identical requests must return the same response from cache."""
        body = _valid_body()
        resp1 = client.post("/analyze", json=body)
        resp2 = client.post("/analyze", json=body)
        # Both succeed
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Same request_id from cache
        assert resp1.json()["request_id"] == resp2.json()["request_id"]

    def test_analyze_empty_question_rejected(self, client):
        body = {"question": "   ", "sources": [{"url": "http://nih.gov", "content": "Valid content here."}]}
        resp = client.post("/analyze", json=body)
        # FastAPI schema validation: 422
        assert resp.status_code == 422

    def test_analyze_question_too_short_rejected(self, client):
        body = {"question": "Why?", "sources": [{"url": "http://nih.gov", "content": "Valid content here."}]}
        resp = client.post("/analyze", json=body)
        assert resp.status_code == 422

    def test_analyze_no_sources_rejected(self, client):
        body = {"question": "Does aspirin reduce cardiovascular events in adults?", "sources": []}
        resp = client.post("/analyze", json=body)
        assert resp.status_code == 422

    def test_analyze_empty_source_content_rejected(self, client):
        body = {
            "question": "Does aspirin reduce cardiovascular events in adults?",
            "sources": [{"url": "http://nih.gov", "content": ""}],
        }
        resp = client.post("/analyze", json=body)
        assert resp.status_code == 422

    def test_analyze_denylisted_source_dropped(self, client):
        body = {
            "question": "Does aspirin reduce cardiovascular events in healthy adults?",
            "sources": [
                {   # good source
                    "url": "https://nih.gov/aspirin-study",
                    "content": "RCT of aspirin in healthy adults found no significant cardiovascular benefit.",
                    "metadata": {"title": "NIH Study", "citation_count": 200},
                },
                {   # denylisted
                    "url": "https://naturalnews.com/article",
                    "content": "Big pharma conspiracy: aspirin is toxic to all humans everywhere.",
                    "metadata": {"title": "Conspiracy Article"},
                },
            ],
        }
        resp = client.post("/analyze", json=body)
        assert resp.status_code == 200
        assert resp.json()["sources_dropped"] >= 1

    def test_analyze_medical_question_response_valid(self, client):
        """Medical questions should complete and may include a disclaimer."""
        body = _valid_body(
            question="Does metformin reduce all-cause mortality in Type 2 diabetic patients?"
        )
        resp = client.post("/analyze", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] in {"High", "Moderate", "Low", "Insufficient Evidence"}

    def test_analyze_loop_rounds_non_negative(self, client):
        resp = client.post("/analyze", json=_valid_body())
        assert resp.json()["loop_rounds"] >= 0

    def test_analyze_loop_rounds_at_most_max(self, client):
        from config.settings import get_settings
        max_rounds = get_settings().max_loop_rounds
        resp = client.post("/analyze", json=_valid_body())
        assert resp.json()["loop_rounds"] <= max_rounds


 
# GET /status/{request_id}
 

class TestStatusEndpoint:

    def _get_request_id(self, client) -> str:
        resp = client.post("/analyze", json=_valid_body(
            question="Does ibuprofen affect kidney function in elderly patients over 70?"
        ))
        return resp.json()["request_id"]

    def test_status_returns_200_for_known_id(self, client):
        request_id = self._get_request_id(client)
        resp = client.get(f"/status/{request_id}")
        assert resp.status_code == 200

    def test_status_returns_404_for_unknown_id(self, client):
        resp = client.get("/status/req_doesnotexist999")
        assert resp.status_code == 404

    def test_status_response_has_required_fields(self, client):
        request_id = self._get_request_id(client)
        resp = client.get(f"/status/{request_id}")
        data = resp.json()
        assert "request_id" in data
        assert "status" in data
        assert "confidence" in data
        assert "loop_round" in data
        assert "hitl_status" in data

    def test_status_request_id_matches(self, client):
        request_id = self._get_request_id(client)
        resp = client.get(f"/status/{request_id}")
        assert resp.json()["request_id"] == request_id

    def test_status_confidence_valid_or_none(self, client):
        request_id = self._get_request_id(client)
        resp = client.get(f"/status/{request_id}")
        confidence = resp.json()["confidence"]
        valid = {"High", "Moderate", "Low", "Insufficient Evidence", None}
        assert confidence in valid


 
# POST /feedback/{request_id}
 

class TestFeedbackEndpoint:

    def _get_request_id(self, client) -> str:
        resp = client.post("/analyze", json=_valid_body(
            question="Does exercise reduce depression symptoms in adults with mild to moderate depression?"
        ))
        return resp.json()["request_id"]

    def test_feedback_returns_200(self, client):
        request_id = self._get_request_id(client)
        resp = client.post(f"/feedback/{request_id}", json={"rating": 4})
        assert resp.status_code == 200

    def test_feedback_acknowledged_true(self, client):
        request_id = self._get_request_id(client)
        resp = client.post(f"/feedback/{request_id}", json={"rating": 5})
        assert resp.json()["acknowledged"] is True

    def test_feedback_request_id_echoed(self, client):
        request_id = self._get_request_id(client)
        resp = client.post(f"/feedback/{request_id}", json={"rating": 3})
        assert resp.json()["request_id"] == request_id

    def test_feedback_source_correction_applied(self, client):
        request_id = self._get_request_id(client)
        body = {
            "rating": 2,
            "source_corrections": [
                {"source_id": "src_001", "flag": "commercial_bias"}
            ],
        }
        resp = client.post(f"/feedback/{request_id}", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["adjustments_applied"] >= 1

    def test_feedback_invalid_rating_rejected(self, client):
        request_id = self._get_request_id(client)
        resp = client.post(f"/feedback/{request_id}", json={"rating": 10})
        assert resp.status_code == 422

    def test_feedback_invalid_rating_zero_rejected(self, client):
        request_id = self._get_request_id(client)
        resp = client.post(f"/feedback/{request_id}", json={"rating": 0})
        assert resp.status_code == 422

    def test_feedback_no_rating_accepted(self, client):
        """Feedback with only source_corrections (no rating) should be accepted."""
        request_id = self._get_request_id(client)
        body = {
            "source_corrections": [
                {"source_id": "src_001", "flag": "outdated"}
            ]
        }
        resp = client.post(f"/feedback/{request_id}", json=body)
        assert resp.status_code == 200

    def test_feedback_unknown_request_id_still_accepted(self, client):
        """Feedback for unknown IDs is still written to DB (no 404 required)."""
        resp = client.post(
            "/feedback/req_unknown_xyz",
            json={"rating": 3}
        )
        assert resp.status_code == 200


 
# POST /approve/{request_id}
 

class TestApproveEndpoint:

    def _get_hitl_request_id(self, client) -> str | None:
        """
        Run pipeline until we get a hitl_pending response, or return None.
        With DRY_RUN, confidence is typically Insufficient Evidence (1 source),
        which triggers HITL if threshold is Moderate.
        """
        resp = client.post("/analyze", json=_valid_body(
            question="Does low-dose naltrexone reduce pain in fibromyalgia patients?",
            source_count=1,
        ))
        data = resp.json()
        if data.get("hitl_status") == "pending":
            return data["request_id"]
        return None

    def test_approve_404_for_non_hitl_request(self, client):
        """Approve on a request that didn't trigger HITL must return 404."""
        resp = client.post(
            "/approve/req_no_hitl_here",
            json={"approved": True, "reviewer_notes": "Looks good."}
        )
        assert resp.status_code == 404

    def test_approve_422_for_missing_approved_field(self, client):
        """Approve without the required 'approved' boolean must return 422."""
        # We need a HITL pending request for this, but we test validation first
        # by using a fake ID that's in the HITL registry
        from api.routes import _HITL_REGISTRY
        _HITL_REGISTRY["req_fake_hitl"] = {"approved": None, "notes": None}
        resp = client.post("/approve/req_fake_hitl", json={"reviewer_notes": "OK"})
        assert resp.status_code == 422
        # Clean up
        del _HITL_REGISTRY["req_fake_hitl"]

    def test_approve_returns_approved_status(self, client):
        """Approval with approved=True returns status='approved'."""
        from api.routes import _HITL_REGISTRY
        _HITL_REGISTRY["req_approve_test"] = {"approved": None, "notes": None}
        resp = client.post(
            "/approve/req_approve_test",
            json={"approved": True, "reviewer_notes": "Verified accurate."}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["reviewer_notes"] == "Verified accurate."
        del _HITL_REGISTRY["req_approve_test"]

    def test_approve_returns_returned_for_revision_status(self, client):
        """Approval with approved=False returns status='returned_for_revision'."""
        from api.routes import _HITL_REGISTRY
        _HITL_REGISTRY["req_reject_test"] = {"approved": None, "notes": None}
        resp = client.post(
            "/approve/req_reject_test",
            json={"approved": False, "reviewer_notes": "Answer needs more evidence."}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "returned_for_revision"
        del _HITL_REGISTRY["req_reject_test"]

    def test_approve_status_reflected_in_status_endpoint(self, client):
        """After approval, GET /status/{id} should reflect the approved state."""
        from api.routes import _HITL_REGISTRY, _STATE_CACHE
        from core.state import make_initial_state

        # Plant a fake state and HITL entry
        _STATE_CACHE["req_status_test"] = make_initial_state(
            "req_status_test", "Test?", []
        )
        _STATE_CACHE["req_status_test"]["confidence"] = "Low"
        _HITL_REGISTRY["req_status_test"] = {"approved": None, "notes": None}

        # Approve it
        client.post("/approve/req_status_test", json={"approved": True})

        # Check status
        resp = client.get("/status/req_status_test")
        assert resp.status_code == 200
        assert resp.json()["hitl_status"] == "approved"

        # Cleanup
        del _STATE_CACHE["req_status_test"]
        del _HITL_REGISTRY["req_status_test"]


 
# GET /replay/{request_id}
 

class TestReplayEndpoint:

    def _analyze_and_get_id(self, client, question_suffix: str = "") -> str:
        q = f"Does omega-3 supplementation reduce triglycerides in adults? {question_suffix}"
        resp = client.post("/analyze", json=_valid_body(question=q))
        return resp.json()["request_id"]

    def test_replay_404_for_unknown_id(self, client):
        resp = client.get("/replay/req_does_not_exist_xyz")
        assert resp.status_code == 404

    def test_replay_returns_200_for_completed_request(self, client):
        request_id = self._analyze_and_get_id(client, "replay_test_1")
        resp = client.get(f"/replay/{request_id}")
        assert resp.status_code == 200

    def test_replay_response_has_required_fields(self, client):
        request_id = self._analyze_and_get_id(client, "replay_test_2")
        resp = client.get(f"/replay/{request_id}")
        data = resp.json()
        assert "request_id" in data
        assert "question" in data
        assert "confidence" in data
        assert "timestamp" in data
        assert "audit_trail" in data
        assert "skeptic_challenges" in data
        assert "arbiter_rulings" in data

    def test_replay_audit_trail_is_list(self, client):
        request_id = self._analyze_and_get_id(client, "replay_test_3")
        resp = client.get(f"/replay/{request_id}")
        assert isinstance(resp.json()["audit_trail"], list)

    def test_replay_audit_trail_non_empty(self, client):
        request_id = self._analyze_and_get_id(client, "replay_test_4")
        resp = client.get(f"/replay/{request_id}")
        trail = resp.json()["audit_trail"]
        # At minimum: guardrail + ingress records
        assert len(trail) >= 2

    def test_replay_question_matches_original(self, client):
        suffix = "replay_match_test_unique_9847"
        q = f"Does omega-3 supplementation reduce triglycerides in adults? {suffix}"
        resp_analyze = client.post("/analyze", json=_valid_body(question=q))
        request_id = resp_analyze.json()["request_id"]

        resp_replay = client.get(f"/replay/{request_id}")
        assert suffix in resp_replay.json()["question"]

    def test_replay_confidence_valid(self, client):
        request_id = self._analyze_and_get_id(client, "replay_test_5")
        resp = client.get(f"/replay/{request_id}")
        confidence = resp.json()["confidence"]
        valid = {"High", "Moderate", "Low", "Insufficient Evidence", None}
        assert confidence in valid