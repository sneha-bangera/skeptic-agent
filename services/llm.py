"""
Single gateway for all LLM calls in the pipeline.

Rules:
- ALL LLM calls in every agent must go through call_llm() — never directly.
- Handles: retry with exponential backoff (3 attempts), structured logging,
  dry-run mode (returns mock response without hitting the API).
- Always requests JSON output; callers receive the parsed dict or raise LLMParseError.
- On parse failure: re-prompts once with an explicit JSON correction instruction,
  then raises LLMParseError if still failing.
"""

from __future__ import annotations

import json
import time
import re
import logging
from typing import Any, Optional

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from core.exceptions import LLMCallError, LLMParseError

logger = structlog.get_logger(__name__)

# Lazy Groq client
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        from config.settings import get_settings
        settings = get_settings()
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# Mock responses for dry-run mode 

_DRY_RUN_MOCKS: dict[str, Any] = {
    "default": {"result": "dry_run_mock_response", "dry_run": True},
    "source_classification": {
        "source_type": "general_news",
        "has_methods": False,
        "has_citations": False,
        "has_funding_disclosure": False,
        "promotional_language_score": 0.1,
    },
    "relevance": {
        "topic_overlap": 0.7,
        "population_match": 0.6,
        "outcome_match": 0.5,
    },
    "bias": {
        "funding_conflict": "none",
        "commercial_interest": "none",
        "framing_language": "none",
        "overall_bias": "none",
    },
    "claims": {
        "claims": [
            {
                # source_id is intentionally left as a sentinel;
                # evidence_extractor replaces this at call time via _inject_dry_run_source_id
                "source_id": "__DRY_RUN_SOURCE__",
                "text": "This is a mock claim for dry-run mode. No real LLM was called.",
                "population": "general population",
                "timeframe": "12 months",
                "outcome_metric": "primary outcome measure",
                "study_design": "randomized controlled trial",
            }
        ]
    },
    "contradictions": {
        "conflict_type": "no_conflict",
        "explanation": "Dry-run: no conflict assessed.",
    },
    "synthesis": {
        "answer": "Dry-run synthesis answer.",
        "confidence": "Low",
        "reasoning": "Dry-run mode — no real LLM call made.",
    },
    "skeptic": {
        "challenges": []
    },
    "arbiter": {
        "rulings": []
    },
}


def _get_mock(call_type: str) -> dict:
    return _DRY_RUN_MOCKS.get(call_type, _DRY_RUN_MOCKS["default"])


# JSON extraction helpers 

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that LLMs sometimes emit."""
    text = text.strip()
    # Remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _try_parse_json(text: str) -> dict:
    """Attempt to parse text as JSON, stripping fences first."""
    cleaned = _strip_markdown_fences(text)
    return json.loads(cleaned)


# Core LLM call (with tenacity retry) 

def _call_groq_raw(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Make a single raw Groq API call and return the assistant message text.
    This function is wrapped by tenacity for retry logic.
    """
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)
def _call_groq_with_retry(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> str:
    return _call_groq_raw(system_prompt, user_prompt, model_name, max_tokens, temperature)


# Public API 

def call_llm(
    system_prompt: str,
    user_prompt: str,
    call_type: str = "default",
    stage: str = "unknown",
    request_id: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    model_name: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> dict:
    """
    Make a structured LLM call and return a parsed JSON dict.

    Args:
        system_prompt:  The system instruction (should always instruct JSON-only output).
        user_prompt:    The user-side content.
        call_type:      Label used for dry-run mock selection and logging.
        stage:          Pipeline stage name (for error attribution).
        request_id:     Request ID (for error attribution).
        max_tokens:     Max tokens for the response.
        temperature:    Sampling temperature (default 0.1 for determinism).
        model_name:     Override the model from settings.
        dry_run:        Override the dry_run setting for this call.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        LLMCallError:   If the API call fails after all retries.
        LLMParseError:  If the response cannot be parsed as JSON after re-prompt.
    """
    from config.settings import get_settings
    settings = get_settings()

    # Resolve dry_run: explicit override > env setting
    is_dry_run = dry_run if dry_run is not None else settings.dry_run

    if is_dry_run:
        mock = _get_mock(call_type)
        logger.debug(
            "llm_dry_run",
            call_type=call_type,
            stage=stage,
            request_id=request_id,
        )
        return mock

    _model = model_name or settings.model_name

    log = logger.bind(
        call_type=call_type,
        stage=stage,
        request_id=request_id,
        model=_model,
        max_tokens=max_tokens,
    )

    # Attempt 1: direct call 
    t0 = time.monotonic()
    try:
        raw = _call_groq_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        log.error("llm_call_failed", error=str(exc))
        raise LLMCallError(
            message=f"LLM call failed at stage '{stage}': {exc}",
            stage=stage,
            attempts=3,
            request_id=request_id,
        ) from exc

    elapsed = time.monotonic() - t0
    log.info("llm_call_success", elapsed_s=round(elapsed, 3), response_len=len(raw))

    # Parse attempt 1 
    try:
        return _try_parse_json(raw)
    except json.JSONDecodeError:
        log.warning("llm_parse_failed_attempt_1", raw_preview=raw[:200])

    # Re-prompt: ask the model to fix its JSON 
    correction_prompt = (
        f"Your previous response was not valid JSON. "
        f"Here is what you returned:\n\n{raw[:1000]}\n\n"
        f"Return ONLY the corrected JSON object. No markdown, no explanation."
    )
    try:
        raw2 = _call_groq_with_retry(
            system_prompt=system_prompt,
            user_prompt=correction_prompt,
            model_name=_model,
            max_tokens=max_tokens,
            temperature=0.0,  # zero temperature for correction
        )
        result = _try_parse_json(raw2)
        log.info("llm_parse_success_after_reprompt")
        return result
    except (json.JSONDecodeError, Exception) as exc:
        log.error("llm_parse_failed_final", error=str(exc), raw_preview=raw[:200])
        raise LLMParseError(
            stage=stage,
            raw_response=raw,
            request_id=request_id,
        ) from exc


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    call_type: str = "default",
    stage: str = "unknown",
    request_id: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    model_name: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> str:
    """
    Like call_llm but returns raw text instead of parsed JSON.
    Used by Synthesis agent which produces prose, not JSON.
    """
    from config.settings import get_settings
    settings = get_settings()

    is_dry_run = dry_run if dry_run is not None else settings.dry_run

    if is_dry_run:
        return "Dry-run synthesis: this is a placeholder answer generated without an LLM call."

    _model = model_name or settings.model_name

    log = logger.bind(
        call_type=call_type,
        stage=stage,
        request_id=request_id,
        model=_model,
    )

    t0 = time.monotonic()
    try:
        raw = _call_groq_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.monotonic() - t0
        log.info("llm_text_call_success", elapsed_s=round(elapsed, 3))
        return raw
    except Exception as exc:
        log.error("llm_text_call_failed", error=str(exc))
        raise LLMCallError(
            message=f"LLM text call failed at stage '{stage}': {exc}",
            stage=stage,
            attempts=3,
            request_id=request_id,
        ) from exc