"""
Output disclaimer guardrail — generates domain-specific disclaimer text
to append to responses touching sensitive domains.

Called by the Egress agent when state["input_domain_flag"] is set.
Returns a plain-text disclaimer string (no LLM needed).

Domains handled: medical, legal, financial
"""

from __future__ import annotations

from typing import Optional

_DISCLAIMERS: dict[str, str] = {
    "medical": (
        "MEDICAL DISCLAIMER: This analysis is generated from published sources "
        "for informational purposes only and does not constitute medical advice, "
        "diagnosis, or treatment recommendations. Medical evidence evolves rapidly "
        "and individual circumstances vary significantly. Always consult a qualified "
        "healthcare professional before making any medical decisions. The confidence "
        "tier assigned reflects the quality of available sources, not clinical certainty."
    ),
    "legal": (
        "LEGAL DISCLAIMER: This analysis is generated from published sources "
        "for informational purposes only and does not constitute legal advice. "
        "Laws vary by jurisdiction and change over time. The analysis may not "
        "reflect the most current legal interpretations or apply to your specific "
        "situation. Always consult a qualified legal professional before taking "
        "any action based on legal information."
    ),
    "financial": (
        "FINANCIAL DISCLAIMER: This analysis is generated from published sources "
        "for informational purposes only and does not constitute financial, "
        "investment, or tax advice. Past performance does not guarantee future "
        "results. Financial markets, regulations, and circumstances change "
        "frequently. Always consult a qualified financial advisor before making "
        "investment or financial decisions."
    ),
}

_INSUFFICIENT_EVIDENCE_ADDENDUM = (
    " Given the Insufficient Evidence confidence rating, extra caution is "
    "especially warranted — the available sources were insufficient to support "
    "a reliable conclusion on this question."
)


def get_disclaimer(
    domain_flag: Optional[str],
    confidence: Optional[str] = None,
) -> Optional[str]:
    """
    Return the appropriate disclaimer string for the given domain flag.

    Args:
        domain_flag:  "medical" | "legal" | "financial" | None
        confidence:   Current confidence tier (used to add extra caution note)

    Returns:
        Disclaimer string, or None if no flag is set.
    """
    if not domain_flag:
        return None

    base = _DISCLAIMERS.get(domain_flag.lower())
    if not base:
        return None

    if confidence == "Insufficient Evidence":
        return base + _INSUFFICIENT_EVIDENCE_ADDENDUM

    return base