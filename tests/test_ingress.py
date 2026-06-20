"""
Unit tests for:
- guardrails/input_guardrail.py
- agents/ingress.py

All tests use DRY_RUN=True and mock embeddings so no API keys are needed.
Run with: pytest tests/test_ingress.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure project root is on path 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force dry-run and a dummy API key for all tests
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")

from core.state import make_initial_state
from guardrails.input_guardrail import (
    run_input_guardrail,
    _detect_domain,
    _check_pii,
    _validate_sources_shape,
)
from agents.ingress import (
    run_ingress,
    _strip_html,
    _extract_domain,
    _normalize_source,
    _count_populated_fields,
    _merge_pair,
)



# Input Guardrail Tests

class TestInputGuardrailDomainDetection:

    def test_detects_medical_domain(self):
        assert _detect_domain("Does aspirin reduce cardiovascular events in diabetic patients?") == "medical"

    def test_detects_legal_domain(self):
        assert _detect_domain("What are the GDPR compliance requirements for data processors?") == "legal"

    def test_detects_financial_domain(self):
        assert _detect_domain("How does inflation affect bond portfolio returns?") == "financial"

    def test_returns_none_for_neutral(self):
        assert _detect_domain("What is the boiling point of water?") is None

    def test_returns_none_for_empty(self):
        assert _detect_domain("") is None

    def test_multiple_domain_hits_returns_strongest(self):
        # "clinical trial drug dosage treatment medication" → heavy medical
        result = _detect_domain("clinical trial drug dosage treatment medication efficacy")
        assert result == "medical"


class TestInputGuardrailPIIDetection:

    def test_detects_ssn(self):
        pii = _check_pii("Patient SSN: 123-45-6789 was admitted")
        assert "ssn" in pii

    def test_detects_credit_card(self):
        pii = _check_pii("Card number 4111 1111 1111 1111 charged")
        assert "credit_card" in pii

    def test_detects_phone(self):
        pii = _check_pii("Call us at 555-867-5309")
        assert "phone_number" in pii

    def test_email_detected(self):
        pii = _check_pii("Contact john.doe@example.com for more info")
        assert "email_address" in pii

    def test_clean_text_no_pii(self):
        pii = _check_pii("The study found statistically significant results (p < 0.05)")
        assert pii == []

    def test_general_text_about_ssn_topic_not_flagged(self):
        # Discussing SSN as a topic (no actual number) should not be flagged
        pii = _check_pii("Social Security Numbers are used for identity verification in the US.")
        assert "ssn" not in pii


class TestInputGuardrailSourceShapeValidation:

    def test_empty_sources_rejected(self):
        error = _validate_sources_shape([])
        assert error is not None
        assert "No sources" in error

    def test_source_without_content_rejected(self):
        error = _validate_sources_shape([{"url": "http://example.com", "content": ""}])
        assert error is not None
        assert "empty content" in error

    def test_whitespace_only_content_rejected(self):
        error = _validate_sources_shape([{"content": "   \n   "}])
        assert error is not None

    def test_valid_source_passes(self):
        error = _validate_sources_shape([{
            "url": "http://example.com",
            "content": "This is valid content with enough text to pass."
        }])
        assert error is None


class TestRunInputGuardrail:

    def _make_state(self, question: str, sources: list) -> dict:
        state = make_initial_state("req_test", question, sources)
        return state

    def _valid_source(self):
        return {"url": "http://nih.gov/study", "content": "Study of cardiovascular outcomes in healthy adults."}

    def test_valid_request_passes(self):
        state = self._make_state(
            "Does exercise reduce cardiovascular risk in healthy adults?",
            [self._valid_source()]
        )
        result = run_input_guardrail(state)
        assert result["input_rejected"] is False
        assert result["input_domain_flag"] == "medical"
        assert len(result["audit_trail"]) == 1

    def test_empty_question_rejected(self):
        state = self._make_state("", [self._valid_source()])
        result = run_input_guardrail(state)
        assert result["input_rejected"] is True
        assert "empty" in result["input_rejection_reason"].lower()

    def test_too_short_question_rejected(self):
        state = self._make_state("Why?", [self._valid_source()])
        result = run_input_guardrail(state)
        assert result["input_rejected"] is True

    def test_pii_in_question_rejected(self):
        state = self._make_state(
            "What medications should patient with SSN 123-45-6789 take?",
            [self._valid_source()]
        )
        result = run_input_guardrail(state)
        assert result["input_rejected"] is True
        assert "pii" in result["input_rejection_reason"].lower()

    def test_pii_in_source_content_rejected(self):
        state = self._make_state(
            "What are the effects of aspirin on blood pressure?",
            [{"url": "http://example.com", "content": "Patient 123-45-6789 experienced side effects."}]
        )
        result = run_input_guardrail(state)
        assert result["input_rejected"] is True

    def test_non_sensitive_domain_returns_none_flag(self):
        state = self._make_state(
            "What is the speed of light in a vacuum?",
            [{"url": "http://physics.org", "content": "The speed of light is approximately 299,792,458 metres per second."}]
        )
        result = run_input_guardrail(state)
        assert result["input_rejected"] is False
        assert result["input_domain_flag"] is None

    def test_audit_trail_appended(self):
        state = self._make_state(
            "Does ibuprofen affect kidney function in elderly patients?",
            [self._valid_source()]
        )
        result = run_input_guardrail(state)
        assert len(result["audit_trail"]) == 1
        record = result["audit_trail"][0]
        assert record["stage"] == "input_guardrail"
        assert record["event"] == "input_guardrail"
        assert "request_id" in record



# Ingress Agent Tests
class TestHTMLStripping:

    def test_strips_basic_tags(self):
        result = _strip_html("<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_script_tags(self):
        result = _strip_html("<script>alert('xss')</script>Some content here")
        assert "alert" not in result
        assert "Some content" in result

    def test_strips_style_tags(self):
        result = _strip_html("<style>.foo{color:red}</style>Visible text")
        assert "color" not in result
        assert "Visible text" in result

    def test_plain_text_unchanged(self):
        text = "Plain text with no HTML tags at all."
        result = _strip_html(text)
        assert "Plain text" in result

    def test_html_entities_decoded(self):
        result = _strip_html("Fish &amp; Chips")
        assert "&amp;" not in result
        assert "Fish" in result


class TestDomainExtraction:

    def test_extracts_simple_domain(self):
        assert _extract_domain("https://www.nih.gov/research/study") == "nih.gov"

    def test_strips_www(self):
        assert _extract_domain("http://www.example.com/page") == "example.com"

    def test_no_scheme(self):
        domain = _extract_domain("pubmed.ncbi.nlm.nih.gov/article/123")
        assert domain is not None
        assert "nih" in domain

    def test_none_url_returns_none(self):
        assert _extract_domain(None) is None

    def test_empty_url_returns_none(self):
        assert _extract_domain("") is None


class TestNormalizeSource:

    def _raw(self, **kwargs):
        base = {
            "url": "https://www.nih.gov/study",
            "content": "<p>This is study content about cardiovascular health outcomes.</p>",
            "metadata": {"title": "Study Title", "author": "Dr. Smith", "citation_count": 5},
        }
        base.update(kwargs)
        return base

    def test_source_id_assigned(self):
        result = _normalize_source(self._raw(), "src_001")
        assert result["source_id"] == "src_001"

    def test_html_stripped_from_content(self):
        result = _normalize_source(self._raw(), "src_001")
        assert "<p>" not in result["content_text"]
        assert "cardiovascular" in result["content_text"]

    def test_domain_extracted(self):
        result = _normalize_source(self._raw(), "src_001")
        assert result["domain"] == "nih.gov"

    def test_metadata_fields_populated(self):
        result = _normalize_source(self._raw(), "src_001")
        assert result["title"] == "Study Title"
        assert result["author"] == "Dr. Smith"

    def test_missing_url_allowed_with_title(self):
        raw = self._raw()
        raw["url"] = None
        result = _normalize_source(raw, "src_001")
        assert result["url"] is None
        # Should still normalize
        assert result["content_text"]


class TestMetadataRichness:

    def test_more_populated_fields_wins(self):
        rich = {
            "url": "http://a.com",
            "title": "Study A",
            "author": "Dr. A",
            "domain": "a.com",
            "published_date": "2023-01-01",
            "metadata": {"doi": "10.1234/abc", "journal": "Nature"},
        }
        sparse = {
            "url": "http://b.com",
            "title": None,
            "author": None,
            "domain": "b.com",
            "published_date": None,
            "metadata": {},
        }
        assert _count_populated_fields(rich) > _count_populated_fields(sparse)

    def test_merge_keeps_richer_as_base(self):
        rich = {
            "source_id": "src_001",
            "url": "http://rich.com",
            "title": "Rich Title",
            "author": "Author A",
            "domain": "rich.com",
            "published_date": "2023",
            "content_text": "Rich content",
            "metadata": {"doi": "10.1234/x"},
        }
        sparse = {
            "source_id": "src_002",
            "url": None,
            "title": None,
            "author": None,
            "domain": "sparse.com",
            "published_date": None,
            "content_text": "Sparse content",
            "metadata": {},
        }
        merged = _merge_pair(rich, sparse)
        assert merged["title"] == "Rich Title"
        assert merged["source_id"] == "src_001"  # rich was primary

    def test_merge_fills_missing_from_secondary(self):
        a = {
            "source_id": "src_001",
            "url": "http://a.com",
            "title": "Title A",
            "author": None,
            "domain": "a.com",
            "published_date": None,
            "content_text": "Content A",
            "metadata": {},
        }
        b = {
            "source_id": "src_002",
            "url": None,
            "title": None,
            "author": "Dr. B",
            "domain": "b.com",
            "published_date": "2022-06-01",
            "content_text": "Content B",
            "metadata": {"journal": "Nature"},
        }
        # a has more populated fields (url + title + domain vs author + date + domain)
        # both have 3 but a wins on url
        merged = _merge_pair(a, b)
        # Should fill author and published_date from b
        assert merged["author"] == "Dr. B"
        assert merged["published_date"] == "2022-06-01"


class TestRunIngress:

    def _make_state(self, sources: list) -> dict:
        state = make_initial_state("req_test", "Test question about health outcomes?", sources)
        return state

    def test_basic_normalization(self):
        state = self._make_state([{
            "url": "https://nih.gov/study",
            "content": "This is a study about cardiovascular outcomes in healthy adults.",
            "metadata": {"title": "Heart Study", "citation_count": 10}
        }])
        result = run_ingress(state)
        assert len(result["normalized_sources"]) == 1
        src = result["normalized_sources"][0]
        assert src["source_id"] == "src_001"
        assert src["domain"] == "nih.gov"
        assert src["title"] == "Heart Study"

    def test_sequential_ids_assigned(self):
        sources = [
            {"url": f"https://example{i}.com", "content": f"Content {i} " * 20, "metadata": {"title": f"Title {i}"}}
            for i in range(3)
        ]
        state = self._make_state(sources)
        result = run_ingress(state)
        ids = [s["source_id"] for s in result["normalized_sources"]]
        assert ids == ["src_001", "src_002", "src_003"]

    def test_url_deduplication(self):
        """Two sources with the same URL should be merged into one."""
        same_url = "https://nih.gov/study"
        state = self._make_state([
            {"url": same_url, "content": "Study content " * 10, "metadata": {"title": "Study A"}},
            {"url": same_url, "content": "Study content " * 10, "metadata": {"title": "Study A", "citation_count": 5}},
        ])
        result = run_ingress(state)
        assert len(result["normalized_sources"]) == 1

    def test_invalid_source_dropped_not_crash(self):
        """Source with empty content should be dropped, not crash the pipeline."""
        state = self._make_state([
            {"url": "https://nih.gov/study", "content": "Valid content for source one here it is.", "metadata": {"title": "Valid"}},
            {"url": "https://bad.com", "content": "", "metadata": {}},
        ])
        result = run_ingress(state)
        # Only the valid source passes
        assert len(result["normalized_sources"]) == 1
        assert result["normalized_sources"][0]["domain"] == "nih.gov"

    def test_html_stripped_in_output(self):
        state = self._make_state([{
            "url": "https://example.com/article",
            "content": "<h1>Title</h1><p>The study found <b>significant</b> results.</p>",
            "metadata": {"title": "Test Article"}
        }])
        result = run_ingress(state)
        content = result["normalized_sources"][0]["content_text"]
        assert "<h1>" not in content
        assert "<b>" not in content
        assert "significant" in content

    def test_audit_trail_appended(self):
        state = self._make_state([{
            "url": "https://nih.gov/study",
            "content": "This is substantial study content about the topic.",
            "metadata": {"title": "Test Study"}
        }])
        result = run_ingress(state)
        assert len(result["audit_trail"]) == 1
        record = result["audit_trail"][0]
        assert record["stage"] == "ingress"
        assert record["event"] == "ingress_complete"
        assert record["data"]["raw_source_count"] == 1
        assert record["data"]["normalized_count"] == 1

    def test_no_sources_returns_empty(self):
        state = self._make_state([])
        result = run_ingress(state)
        assert result["normalized_sources"] == []

    def test_internal_content_hash_not_in_output(self):
        """The _content_hash field is internal and must not leak into normalized_sources."""
        state = self._make_state([{
            "url": "https://nih.gov/study",
            "content": "Valid content for this test source to pass validation.",
            "metadata": {"title": "Study"}
        }])
        result = run_ingress(state)
        for src in result["normalized_sources"]:
            assert "_content_hash" not in src


class TestSourceCountGuardrail:

    def test_sufficient_sources_no_ceiling(self):
        from guardrails.source_count_guardrail import run_source_count_guardrail
        state = make_initial_state("req_1", "Test?", [])
        state["scored_sources"] = [
            {"source_id": "src_001", "credibility": 0.80, "relevance": 0.75},
            {"source_id": "src_002", "credibility": 0.70, "relevance": 0.65},
            {"source_id": "src_003", "credibility": 0.60, "relevance": 0.55},
        ]
        result = run_source_count_guardrail(state)
        assert result["source_count_ceiling"] is None

    def test_below_floor_sets_low_ceiling(self):
        from guardrails.source_count_guardrail import run_source_count_guardrail
        state = make_initial_state("req_1", "Test?", [])
        state["scored_sources"] = [
            {"source_id": "src_001", "credibility": 0.80, "relevance": 0.75},
            {"source_id": "src_002", "credibility": 0.70, "relevance": 0.65},
            # Only 2 usable sources (floor is 3)
        ]
        result = run_source_count_guardrail(state)
        assert result["source_count_ceiling"] == "Low"

    def test_zero_usable_sources_sets_insufficient(self):
        from guardrails.source_count_guardrail import run_source_count_guardrail
        state = make_initial_state("req_1", "Test?", [])
        state["scored_sources"] = [
            {"source_id": "src_001", "credibility": 0.10, "relevance": 0.05},
        ]
        result = run_source_count_guardrail(state)
        assert result["source_count_ceiling"] == "Insufficient Evidence"

    def test_sources_below_threshold_not_counted(self):
        from guardrails.source_count_guardrail import run_source_count_guardrail
        state = make_initial_state("req_1", "Test?", [])
        state["scored_sources"] = [
            {"source_id": "src_001", "credibility": 0.80, "relevance": 0.75},  # usable
            {"source_id": "src_002", "credibility": 0.80, "relevance": 0.10},  # relevance too low
            {"source_id": "src_003", "credibility": 0.10, "relevance": 0.75},  # credibility too low
        ]
        result = run_source_count_guardrail(state)
        # Only 1 usable → below floor of 3 → Low ceiling
        assert result["source_count_ceiling"] == "Low"

    def test_audit_record_written(self):
        from guardrails.source_count_guardrail import run_source_count_guardrail
        state = make_initial_state("req_1", "Test?", [])
        state["scored_sources"] = [
            {"source_id": "src_001", "credibility": 0.80, "relevance": 0.75},
        ]
        result = run_source_count_guardrail(state)
        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["stage"] == "source_count_guardrail"