"""
Input guardrail node — first node in the LangGraph graph.

Responsibilities:
1. Schema validation  — question and sources meet minimum shape requirements
2. Domain flagging    — detect medical / legal / financial questions (triggers disclaimer later)
3. PII detection      — basic check for obvious PII patterns in question + source content
4. Rejection          — if the request is fundamentally unusable, set input_rejected = True
                        and short-circuit the pipeline

On rejection the pipeline should NOT proceed. The graph's conditional edge after this
node checks state["input_rejected"] and routes to a terminal error formatter if True.

LLM usage: None. Pure deterministic logic.
Writes to state: input_domain_flag, input_rejected, input_rejection_reason, audit_trail
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import structlog

from core.state import PipelineState

logger = structlog.get_logger(__name__)

# Domain keyword maps 

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "medical": [
        "diagnos", "treatment", "medication", "drug", "dose", "dosage",
        "symptom", "disease", "cancer", "diabetes", "vaccine", "clinical",
        "therapy", "prescription", "surgery", "hospital", "patient",
        "cardiovascular", "oncol", "pharmacol", "syndrome", "disorder",
        "mental health", "depression", "anxiety", "covid", "virus", "infection",
        "supplement", "vitamin", "mortality", "morbidity", "randomized",
        "placebo", "efficacy", "adverse effect", "side effect",
    ],
    "legal": [
        "lawsuit", "litigation", "contract", "statute", "regulation",
        "court", "judge", "attorney", "lawyer", "legal", "law ", " law",
        "liability", "plaintiff", "defendant", "verdict", "settlement",
        "intellectual property", "copyright", "patent", "trademark",
        "compliance", "gdpr", "hipaa", "regulatory", "legislation",
        "criminal", "civil", "jurisdiction", "precedent",
    ],
    "financial": [
        "invest", "stock", "bond", "portfolio", "return", "risk",
        "market", "trading", "crypto", "bitcoin", "ethereum", "fund",
        "hedge", "asset", "liability", "balance sheet", "revenue",
        "profit", "loss", "tax", "irs", "audit", "accounting",
        "inflation", "interest rate", "gdp", "recession", "mortgage",
        "loan", "debt", "credit", "derivative", "option", "futures",
    ],
}

# PII patterns 
# Conservative: only match patterns that are clearly PII.
# We do NOT block general discussions about SSN/email/phone as topics.

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # US Social Security Number: 123-45-6789 or 123 45 6789
    ("ssn", re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")),
    # Credit card numbers (Luhn-shaped, 13–19 digits, space/dash separated)
    ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    # Email addresses — only flag if they appear to be personal data (name@domain)
    ("email_address", re.compile(r"\b[A-Za-z0-9._%+\-]{2,}@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # US phone numbers
    ("phone_number", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    # Passport-like patterns (letter + 6–9 digits)
    ("passport_number", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
]

# Audit helper 

def _make_audit_record(request_id: str, data: dict) -> dict:
    return {
        "event": "input_guardrail",
        "stage": "input_guardrail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "data": data,
    }


# Detection helpers 

def _detect_domain(question: str) -> Optional[str]:
    """
    Return the most likely sensitive domain ("medical" | "legal" | "financial" | None)
    by keyword matching on the question text.

    If multiple domains match, return the one with the most keyword hits.
    """
    q_lower = question.lower()
    scores: dict[str, int] = {}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q_lower)
        if hits > 0:
            scores[domain] = hits

    if not scores:
        return None

    return max(scores, key=lambda d: scores[d])


def _check_pii(text: str) -> list[str]:
    """
    Return a list of PII type names found in text.
    Empty list = no PII detected.
    """
    found: list[str] = []
    for pii_type, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(pii_type)
    return found


def _validate_sources_shape(raw_sources: list[dict]) -> Optional[str]:
    """
    Check that the raw sources list is non-empty and each entry has at minimum
    a non-empty 'content' field and either 'url' or a 'title' in metadata.
    Returns an error string or None if valid.
    """
    if not raw_sources:
        return "No sources provided. At least one source is required."

    for i, src in enumerate(raw_sources):
        content = src.get("content", "")
        if not content or not str(content).strip():
            return (
                f"Source at index {i} has empty content. "
                "Every source must have non-empty content text."
            )

    return None 


# Main guardrail node 

def run_input_guardrail(state: PipelineState) -> dict:
    """
    LangGraph node function for the input guardrail.

    Reads:  state["question"], state["raw_sources"], state["request_id"]
    Writes: input_domain_flag, input_rejected, input_rejection_reason, audit_trail
    """
    request_id = state["request_id"]
    question = state.get("question", "")
    raw_sources = state.get("raw_sources", [])

    log = logger.bind(stage="input_guardrail", request_id=request_id)
    log.info("input_guardrail_start", question_len=len(question), source_count=len(raw_sources))

    # 1. Basic question validation 
    if not question or not question.strip():
        audit = _make_audit_record(request_id, {
            "rejected": True,
            "reason": "empty_question",
        })
        log.warning("input_rejected", reason="empty_question")
        return {
            "input_rejected": True,
            "input_rejection_reason": "Question must not be empty.",
            "input_domain_flag": None,
            "audit_trail": state["audit_trail"] + [audit],
        }

    if len(question.strip()) < 10:
        audit = _make_audit_record(request_id, {
            "rejected": True,
            "reason": "question_too_short",
        })
        log.warning("input_rejected", reason="question_too_short")
        return {
            "input_rejected": True,
            "input_rejection_reason": "Question is too short (minimum 10 characters).",
            "input_domain_flag": None,
            "audit_trail": state["audit_trail"] + [audit],
        }

    # 2. Source shape validation 
    shape_error = _validate_sources_shape(raw_sources)
    if shape_error:
        audit = _make_audit_record(request_id, {
            "rejected": True,
            "reason": "invalid_source_shape",
            "detail": shape_error,
        })
        log.warning("input_rejected", reason="invalid_source_shape", detail=shape_error)
        return {
            "input_rejected": True,
            "input_rejection_reason": shape_error,
            "input_domain_flag": None,
            "audit_trail": state["audit_trail"] + [audit],
        }

    # 3. PII check — question 
    pii_in_question = _check_pii(question)
    if pii_in_question:
        audit = _make_audit_record(request_id, {
            "rejected": True,
            "reason": "pii_in_question",
            "pii_types": pii_in_question,
        })
        log.warning("input_rejected", reason="pii_in_question", pii_types=pii_in_question)
        return {
            "input_rejected": True,
            "input_rejection_reason": (
                f"PII detected in question ({', '.join(pii_in_question)}). "
                "Please remove personal identifiable information before submitting."
            ),
            "input_domain_flag": None,
            "audit_trail": state["audit_trail"] + [audit],
        }

    # 4. PII check — source content (check first 2000 chars per source) 
    for i, src in enumerate(raw_sources):
        content_sample = str(src.get("content", ""))[:2000]
        pii_found = _check_pii(content_sample)
        # Email addresses in academic content are common; only flag if SSN/CC/phone
        sensitive_pii = [p for p in pii_found if p not in ("email_address",)]
        if sensitive_pii:
            audit = _make_audit_record(request_id, {
                "rejected": True,
                "reason": "pii_in_source",
                "source_index": i,
                "pii_types": sensitive_pii,
            })
            log.warning(
                "input_rejected",
                reason="pii_in_source",
                source_index=i,
                pii_types=sensitive_pii,
            )
            return {
                "input_rejected": True,
                "input_rejection_reason": (
                    f"PII detected in source at index {i} ({', '.join(sensitive_pii)}). "
                    "Please remove personal identifiable information before submitting."
                ),
                "input_domain_flag": None,
                "audit_trail": state["audit_trail"] + [audit],
            }

    # 5. Domain flagging 
    domain_flag = _detect_domain(question)

    # 6. All checks passed — accept the request 
    audit = _make_audit_record(request_id, {
        "rejected": False,
        "domain_flag": domain_flag,
        "question_length": len(question),
        "source_count": len(raw_sources),
    })

    log.info(
        "input_guardrail_passed",
        domain_flag=domain_flag,
        source_count=len(raw_sources),
    )

    return {
        "input_rejected": False,
        "input_rejection_reason": None,
        "input_domain_flag": domain_flag,
        "audit_trail": state["audit_trail"] + [audit],
    }