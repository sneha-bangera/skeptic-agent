"""
Custom exception hierarchy for The Skeptic's Brief.

Design rules:
- Every exception carries a `stage` so audit_trail records can pinpoint where failures occur.
- Agent-level exceptions are caught inside agents and written to audit_trail.
- Only pipeline-level exceptions propagate to the FastAPI error handler.
"""

from __future__ import annotations
from typing import Optional


class SkepticsBriefError(Exception):
    """Base exception for all Skeptic's Brief errors."""

    def __init__(self, message: str, stage: str = "unknown", request_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.request_id = request_id

    def to_audit_record(self) -> dict:
        return {
            "event": "exception",
            "stage": self.stage,
            "error_type": type(self).__name__,
            "message": self.message,
            "request_id": self.request_id,
        }


# Input / Validation Errors 

class InputValidationError(SkepticsBriefError):
    """Raised when incoming API request fails schema or guardrail validation."""

    def __init__(self, message: str, reason: str, request_id: Optional[str] = None):
        super().__init__(message, stage="input_guardrail", request_id=request_id)
        self.reason = reason


class PIIDetectedError(InputValidationError):
    """Raised when PII is detected in the question or sources."""

    def __init__(self, field: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"PII detected in field: {field}",
            reason="pii_detected",
            request_id=request_id,
        )
        self.field = field


# Source / Ingress Errors  

class SourceValidationError(SkepticsBriefError):
    """Raised when a source fails minimum validation (missing content or URL)."""

    def __init__(self, source_index: int, reason: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"Source at index {source_index} failed validation: {reason}",
            stage="ingress",
            request_id=request_id,
        )
        self.source_index = source_index
        self.reason = reason


class DuplicateSourceError(SkepticsBriefError):
    """Raised (logged only) when duplicate sources are detected and merged."""

    def __init__(self, source_id: str, duplicate_of: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"Source {source_id} is a duplicate of {duplicate_of} and was merged",
            stage="ingress",
            request_id=request_id,
        )
        self.source_id = source_id
        self.duplicate_of = duplicate_of


class DenylistedSourceError(SkepticsBriefError):
    """Raised (logged only) when a source domain hits the denylist."""

    def __init__(self, source_id: str, domain: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"Source {source_id} from domain '{domain}' is denylisted and was dropped",
            stage="source_scorer",
            request_id=request_id,
        )
        self.source_id = source_id
        self.domain = domain


# LLM / Service Errors

class LLMCallError(SkepticsBriefError):
    """Raised when an LLM call fails after all retries are exhausted."""

    def __init__(
        self,
        message: str,
        stage: str,
        attempts: int = 3,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, stage=stage, request_id=request_id)
        self.attempts = attempts


class LLMParseError(SkepticsBriefError):
    """Raised when the LLM response cannot be parsed as valid JSON."""

    def __init__(
        self,
        stage: str,
        raw_response: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Failed to parse LLM JSON response at stage '{stage}'",
            stage=stage,
            request_id=request_id,
        )
        self.raw_response = raw_response[:500]  # truncate for safety


class EmbeddingError(SkepticsBriefError):
    """Raised when the embedding service fails."""

    def __init__(self, message: str, request_id: Optional[str] = None):
        super().__init__(message, stage="embedder", request_id=request_id)


# Pipeline / Graph Errors  

class PipelineStateError(SkepticsBriefError):
    """Raised when required state fields are missing at a node boundary."""

    def __init__(self, field: str, stage: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"Required state field '{field}' missing at stage '{stage}'",
            stage=stage,
            request_id=request_id,
        )
        self.field = field


class LoopLimitExceededError(SkepticsBriefError):
    """Raised (internally) when the adversarial loop hits max_loop_rounds."""

    def __init__(self, rounds: int, request_id: Optional[str] = None):
        super().__init__(
            message=f"Adversarial loop exceeded {rounds} rounds — forcing termination",
            stage="arbiter",
            request_id=request_id,
        )
        self.rounds = rounds


class InsufficientSourcesError(SkepticsBriefError):
    """Raised when fewer than source_count_floor usable sources are available."""

    def __init__(self, count: int, floor: int, request_id: Optional[str] = None):
        super().__init__(
            message=f"Only {count} usable sources (floor is {floor}) — confidence capped",
            stage="source_count_guardrail",
            request_id=request_id,
        )
        self.count = count
        self.floor = floor


# Storage Errors  

class AuditWriteError(SkepticsBriefError):
    """Raised when the audit DB write fails. Non-fatal — pipeline still returns."""

    def __init__(self, message: str, request_id: Optional[str] = None):
        super().__init__(message, stage="audit", request_id=request_id)


# HITL Errors

class HITLPendingError(SkepticsBriefError):
    """Raised when a response is held pending HITL review."""

    def __init__(self, request_id: str, confidence: str):
        super().__init__(
            message=f"Response for {request_id} is pending HITL review (confidence: {confidence})",
            stage="hitl_gate",
            request_id=request_id,
        )
        self.confidence = confidence