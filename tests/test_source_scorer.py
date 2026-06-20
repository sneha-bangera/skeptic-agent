"""
Unit tests for agents/source_scorer.py

All tests use DRY_RUN=True so no API keys are required.
The LLM service returns typed mock responses (defined in services/llm.py _DRY_RUN_MOCKS).

Run with: pytest tests/test_source_scorer.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("DRY_RUN", "true")

from core.state import make_initial_state
from agents.source_scorer import (
    run_source_scorer,
    _classify_metadata,
    _compute_credibility,
    _compute_relevance,
    _apply_bias_cap,
    _parse_year,
    _years_old,
    _is_institutional_domain,
)
from registry.allowlist import is_trusted_domain
from registry.denylist import is_denylisted


# Fixtures 

def _make_scored_state(sources: list) -> dict:
    """Build a state with normalized_sources already populated."""
    state = make_initial_state(
        "req_scorer_test",
        "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
        [],
    )
    state["normalized_sources"] = sources
    return state


def _base_source(**overrides) -> dict:
    """Minimal valid normalized source."""
    src = {
        "source_id": "src_001",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
        "title": "Effect of Aspirin on Cardiovascular Events",
        "author": "Dr. Jane Smith",
        "domain": "pubmed.ncbi.nlm.nih.gov",
        "published_date": "2022-03-15",
        "content_text": (
            "This randomized controlled trial enrolled 19,114 healthy adults "
            "aged 70 years or older. Participants received 100mg aspirin daily "
            "or placebo. Primary outcome was cardiovascular disease events over "
            "5 years. Results showed no significant difference between groups "
            "(HR 0.95, 95% CI 0.83–1.08). Aspirin was associated with increased "
            "risk of major hemorrhage."
        ),
        "metadata": {
            "doi": "10.1056/NEJMoa1805819",
            "journal": "New England Journal of Medicine",
            "citation_count": 1847,
            "published_date": "2022-03-15",
        },
    }
    src.update(overrides)
    return src



# Helper Function Tests

class TestDateHelpers:

    def test_parse_year_iso_date(self):
        assert _parse_year("2022-03-15") == 2022

    def test_parse_year_year_only(self):
        assert _parse_year("2019") == 2019

    def test_parse_year_none(self):
        assert _parse_year(None) is None

    def test_parse_year_invalid(self):
        assert _parse_year("unknown date") is None

    def test_parse_year_with_text(self):
        assert _parse_year("Published in 2021 by Elsevier") == 2021

    def test_years_old_recent(self):
        import datetime
        current = datetime.datetime.now().year
        age = _years_old(current - 1)
        assert 0.9 <= age <= 1.1

    def test_years_old_none(self):
        assert _years_old(None) is None


class TestInstitutionalDomainCheck:

    def test_gov_domain(self):
        assert _is_institutional_domain("nih.gov") is True

    def test_edu_domain(self):
        assert _is_institutional_domain("harvard.edu") is True

    def test_ac_uk_domain(self):
        assert _is_institutional_domain("ox.ac.uk") is True

    def test_who_int(self):
        assert _is_institutional_domain("who.int") is True

    def test_commercial_domain(self):
        assert _is_institutional_domain("webmd.com") is False

    def test_none_domain(self):
        assert _is_institutional_domain(None) is False



# Step 1 & 2: Denylist / Allowlist

class TestDenylistAllowlistIntegration:

    def test_denylisted_source_dropped(self):
        state = _make_scored_state([
            {
                **_base_source(),
                "source_id": "src_001",
                "url": "https://naturalnews.com/article",
                "domain": "naturalnews.com",
                "metadata": {},
            }
        ])
        result = run_source_scorer(state)
        assert len(result["scored_sources"]) == 0
        assert len(result["dropped_sources"]) == 1
        assert result["dropped_sources"][0]["source_id"] == "src_001"
        assert "misinformation" in result["dropped_sources"][0]["reason"]

    def test_allowlisted_source_gets_high_credibility_floor(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        # pubmed.ncbi.nlm.nih.gov is trusted — base credibility is 0.90
        # Modifiers (recency, citation, author) adjust from there — floor is 0.75+
        assert len(result["scored_sources"]) == 1
        src = result["scored_sources"][0]
        assert src["trusted_source"] is True
        assert src["credibility"] >= 0.75  # base 0.90 with modifiers applied
        assert src["credibility"] > 0.60   # definitively above general_news range

    def test_denylisted_source_in_audit_trail(self):
        state = _make_scored_state([
            {
                **_base_source(),
                "domain": "omicsonline.org",
                "url": "https://omicsonline.org/paper",
            }
        ])
        result = run_source_scorer(state)
        drop_events = [
            r for r in result["audit_trail"]
            if r.get("data", {}).get("reason", "").startswith("denylist")
        ]
        assert len(drop_events) >= 1

    def test_multiple_sources_denylist_does_not_crash_others(self):
        """A denylist hit on one source should not prevent scoring of others."""
        state = _make_scored_state([
            _base_source(source_id="src_001"),  # trusted (nih.gov)
            {
                **_base_source(),
                "source_id": "src_002",
                "domain": "naturalnews.com",
                "url": "https://naturalnews.com/article",
                "metadata": {},
            },
        ])
        result = run_source_scorer(state)
        assert len(result["scored_sources"]) == 1
        assert len(result["dropped_sources"]) == 1
        assert result["scored_sources"][0]["source_id"] == "src_001"



# Step 3: Metadata Classification

class TestMetadataClassification:

    def _settings(self):
        from config.settings import get_settings
        return get_settings()

    def test_peer_reviewed_with_doi_and_journal(self):
        src = _base_source()
        source_type, ambiguity = _classify_metadata(src, self._settings())
        assert source_type == "peer_reviewed"
        assert ambiguity == 0

    def test_institutional_gov_domain(self):
        src = _base_source(
            domain="cdc.gov",
            url="https://cdc.gov/report",
            metadata={},  # no DOI
        )
        source_type, ambiguity = _classify_metadata(src, self._settings())
        assert source_type == "institutional"

    def test_established_press_domain(self):
        src = _base_source(
            domain="reuters.com",
            url="https://reuters.com/article",
            metadata={"published_date": "2023-01-01"},
            author="Jane Reporter",
        )
        source_type, _ = _classify_metadata(src, self._settings())
        assert source_type == "established_press"

    def test_commercial_url_signals(self):
        src = _base_source(
            domain="aspirin-shop.com",
            url="https://aspirin-shop.com/buy-aspirin-online",
            metadata={},
        )
        source_type, _ = _classify_metadata(src, self._settings())
        assert source_type == "commercial"

    def test_blog_opinion_first_person(self):
        src = _base_source(
            domain="healthblog.com",
            url="https://healthblog.com/my-thoughts",
            content_text=(
                "I have been taking aspirin for years. In my experience, it has helped "
                "me tremendously. I believe everyone should try it. My doctor disagrees "
                "but I think I know better. I've read extensively on this topic."
            ),
            metadata={},
        )
        source_type, _ = _classify_metadata(src, self._settings())
        assert source_type == "blog_opinion"

    def test_ambiguous_triggers_llm_path(self):
        """Source with no DOI, no institutional domain, no established press, no byline."""
        src = _base_source(
            domain="randomsite.io",
            url="https://randomsite.io/article",
            author=None,
            published_date=None,
            metadata={},
        )
        source_type, ambiguity = _classify_metadata(src, self._settings())
        assert ambiguity >= 3



# Step 5: Credibility Score

class TestCredibilityScore:

    def _settings(self):
        from config.settings import get_settings
        return get_settings()

    def test_trusted_source_floor(self):
        """Allowlisted sources start from base 0.90; modifiers can reduce this slightly."""
        src = _base_source()
        score, breakdown = _compute_credibility("trusted_source", src, self._settings(), trusted_source=True)
        assert score >= 0.75  # base 0.90 after recency/citation/author modifiers
        assert breakdown["trusted_source"] is True
        assert breakdown["base"] == 0.90

    def test_peer_reviewed_base(self):
        src = _base_source()
        score, breakdown = _compute_credibility("peer_reviewed", src, self._settings(), trusted_source=False)
        assert 0.70 <= score <= 1.0
        assert breakdown["base"] == 0.85

    def test_blog_opinion_base(self):
        src = _base_source(
            published_date="2023",
            metadata={"citation_count": 0},
            author=None,
        )
        score, breakdown = _compute_credibility("blog_opinion", src, self._settings(), trusted_source=False)
        assert score < 0.35  # base 0.25 × modifiers
        assert breakdown["base"] == 0.25

    def test_commercial_base(self):
        src = _base_source(metadata={"citation_count": 0}, author=None)
        score, breakdown = _compute_credibility("commercial", src, self._settings(), trusted_source=False)
        assert score < 0.15

    def test_high_citation_modifier_increases_score(self):
        src_high = _base_source(metadata={"citation_count": 100, "doi": "10.1000/x", "published_date": "2023"})
        src_low = _base_source(metadata={"citation_count": 0, "doi": "10.1000/x", "published_date": "2023"})
        score_high, _ = _compute_credibility("peer_reviewed", src_high, self._settings())
        score_low, _ = _compute_credibility("peer_reviewed", src_low, self._settings())
        assert score_high > score_low

    def test_recent_source_no_recency_penalty(self):
        import datetime
        current_year = datetime.datetime.now().year
        src = _base_source(
            published_date=str(current_year - 1),
            metadata={"citation_count": 5, "published_date": str(current_year - 1)},
        )
        _, breakdown = _compute_credibility("peer_reviewed", src, self._settings())
        assert breakdown["recency_weight"] == 1.0

    def test_old_source_gets_recency_penalty(self):
        src = _base_source(
            published_date="2015",
            metadata={"citation_count": 5, "published_date": "2015"},
        )
        _, breakdown = _compute_credibility("peer_reviewed", src, self._settings())
        assert breakdown["recency_weight"] < 1.0

    def test_anonymous_author_modifier(self):
        src = _base_source(author="Anonymous", metadata={"citation_count": 0})
        _, breakdown = _compute_credibility("general_news", src, self._settings())
        assert breakdown["author_modifier"] == 0.90

    def test_score_clamped_to_one(self):
        """Score must never exceed 1.0 regardless of modifiers."""
        src = _base_source(
            metadata={
                "citation_count": 999,
                "orcid": "0000-0001-2345-6789",
                "published_date": "2024",
            },
            published_date="2024",
        )
        score, _ = _compute_credibility("trusted_source", src, self._settings(), trusted_source=True)
        assert score <= 1.0

    def test_score_clamped_to_zero(self):
        """Score must never go below 0.0."""
        src = _base_source(
            published_date="1980",
            author="Anonymous",
            metadata={"citation_count": 0, "published_date": "1980"},
        )
        score, _ = _compute_credibility("commercial", src, self._settings(), trusted_source=False)
        assert score >= 0.0



# Step 6: Relevance Score

class TestRelevanceScore:

    def test_formula_correct(self):
        data = {"topic_overlap": 1.0, "population_match": 1.0, "outcome_match": 1.0}
        assert _compute_relevance(data) == 1.0

    def test_formula_weighted(self):
        data = {"topic_overlap": 0.8, "population_match": 0.6, "outcome_match": 0.4}
        expected = round(0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.4, 4)
        assert _compute_relevance(data) == expected

    def test_zero_relevance(self):
        data = {"topic_overlap": 0.0, "population_match": 0.0, "outcome_match": 0.0}
        assert _compute_relevance(data) == 0.0

    def test_missing_fields_default_to_half(self):
        # Missing fields should not crash; they default to 0.5
        data = {}
        result = _compute_relevance(data)
        expected = round(0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5, 4)
        assert result == expected

    def test_clamped_to_one(self):
        data = {"topic_overlap": 2.0, "population_match": 2.0, "outcome_match": 2.0}
        assert _compute_relevance(data) == 1.0



# Step 7: Bias Cap

class TestBiasCap:

    def _settings(self):
        from config.settings import get_settings
        return get_settings()

    def test_high_bias_caps_at_040(self):
        result = _apply_bias_cap(0.90, "high", self._settings())
        assert result == 0.40

    def test_moderate_bias_caps_at_060(self):
        result = _apply_bias_cap(0.90, "moderate", self._settings())
        assert result == 0.60

    def test_low_bias_no_cap(self):
        result = _apply_bias_cap(0.80, "low", self._settings())
        assert result == 0.80

    def test_none_bias_no_cap(self):
        result = _apply_bias_cap(0.75, "none", self._settings())
        assert result == 0.75

    def test_cap_only_applies_when_score_exceeds_cap(self):
        """If credibility is already below the cap, don't reduce it further."""
        result = _apply_bias_cap(0.30, "high", self._settings())
        assert result == 0.30  # already below 0.40 cap

    def test_cap_reduces_exactly_to_ceiling(self):
        result = _apply_bias_cap(0.95, "moderate", self._settings())
        assert result == 0.60



# Full run_source_scorer Integration Tests

class TestRunSourceScorerIntegration:

    def test_trusted_source_scored_correctly(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        assert len(result["scored_sources"]) == 1
        src = result["scored_sources"][0]
        assert src["source_id"] == "src_001"
        assert src["trusted_source"] is True
        assert 0.0 <= src["credibility"] <= 1.0
        assert 0.0 <= src["relevance"] <= 1.0
        assert src["bias_flag"] in ("none", "low", "moderate", "high")

    def test_scored_source_has_required_fields(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        src = result["scored_sources"][0]
        required_fields = [
            "source_id", "credibility", "relevance", "bias_flag",
            "source_type", "trusted_source", "score_breakdown"
        ]
        for field in required_fields:
            assert field in src, f"Missing field: {field}"

    def test_score_breakdown_contains_components(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        breakdown = result["scored_sources"][0]["score_breakdown"]
        assert "base" in breakdown
        assert "recency_weight" in breakdown
        assert "citation_modifier" in breakdown
        assert "author_modifier" in breakdown

    def test_empty_normalized_sources_returns_empty(self):
        state = _make_scored_state([])
        result = run_source_scorer(state)
        assert result["scored_sources"] == []
        assert result["dropped_sources"] == []

    def test_agent_failure_on_one_source_does_not_crash_others(self):
        """
        If scoring logic errors on one source (e.g., weird metadata),
        the pipeline should still score other sources.
        """
        good = _base_source(source_id="src_001")
        weird = _base_source(
            source_id="src_002",
            domain="cdc.gov",
            url="https://cdc.gov/report",
            published_date=None,
            metadata={"citation_count": "not-a-number"},  # malformed
        )
        state = _make_scored_state([good, weird])
        # Should not raise
        result = run_source_scorer(state)
        # Both should be scored (neither is denylisted)
        assert len(result["scored_sources"]) == 2

    def test_audit_trail_has_summary_record(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        summary_records = [
            r for r in result["audit_trail"]
            if r.get("event") == "source_scorer_summary"
        ]
        assert len(summary_records) == 1
        summary = summary_records[0]["data"]
        assert "scored_count" in summary
        assert "dropped_count" in summary

    def test_dropped_sources_accumulated_in_state(self):
        """dropped_sources from previous stages should be preserved."""
        state = _make_scored_state([
            {
                **_base_source(),
                "domain": "naturalnews.com",
                "url": "https://naturalnews.com/article",
                "metadata": {},
            }
        ])
        # Pre-populate dropped_sources as if a previous stage dropped something
        state["dropped_sources"] = [{"source_id": "src_000", "domain": "old.com", "reason": "validation_failed"}]

        result = run_source_scorer(state)
        # Should have 2 total dropped: the pre-existing one + the new denylist hit
        assert len(result["dropped_sources"]) == 2

    def test_multiple_sources_all_scored(self):
        sources = [
            _base_source(source_id="src_001"),
            {
                **_base_source(),
                "source_id": "src_002",
                "domain": "cdc.gov",
                "url": "https://cdc.gov/report",
                "metadata": {"citation_count": 3, "published_date": "2023"},
            },
            {
                **_base_source(),
                "source_id": "src_003",
                "domain": "reuters.com",
                "url": "https://reuters.com/article",
                "author": "Health Reporter",
                "metadata": {"published_date": "2023-06-01"},
            },
        ]
        state = _make_scored_state(sources)
        result = run_source_scorer(state)
        assert len(result["scored_sources"]) == 3
        ids = {s["source_id"] for s in result["scored_sources"]}
        assert ids == {"src_001", "src_002", "src_003"}

    def test_bias_flag_present_in_output(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        bias = result["scored_sources"][0]["bias_flag"]
        assert bias in ("none", "low", "moderate", "high")

    def test_source_type_present_in_output(self):
        state = _make_scored_state([_base_source()])
        result = run_source_scorer(state)
        src_type = result["scored_sources"][0]["source_type"]
        valid_types = {
            "peer_reviewed", "institutional", "established_press",
            "general_news", "blog_opinion", "commercial", "trusted_source", "unknown"
        }
        assert src_type in valid_types