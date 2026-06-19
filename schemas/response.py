"""
Pydantic schemas for all API response shapes.
"""

from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


ConfidenceTier = Literal["High", "Moderate", "Low", "Insufficient Evidence"]
HITLStatus = Literal["pending", "approved", "returned_for_revision", "not_required"]
PipelineStatus = Literal["processing", "complete", "failed", "hitl_pending"]


class CitationEntry(BaseModel):
    """A single entry in the claim-to-source citation map."""

    claim_text: str = Field(..., description="The specific claim being cited")
    source_id: str = Field(..., description="Which source supports this claim")
    credibility: float = Field(..., ge=0.0, le=1.0, description="Credibility score of that source")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score of that source")
    source_url: Optional[str] = Field(None, description="URL of the source, if available")
    source_title: Optional[str] = Field(None, description="Title of the source, if available")


class SkepticChallengeSummary(BaseModel):
    """Summarized Skeptic challenge for inclusion in the response."""

    challenge_id: str
    type: str
    description: str
    severity: str
    was_resolved: bool
    arbiter_ruling: Optional[str] = None


class APIResponse(BaseModel):
    """
    The full response returned by POST /analyze.
    This is the contract the Egress agent formats output to.
    """

    request_id: str = Field(..., description="Unique identifier for this analysis request")
    question: str = Field(..., description="The original question asked")

    # Core answer
    answer: Optional[str] = Field(None, description="The final synthesized answer")
    confidence: Optional[ConfidenceTier] = Field(
        None,
        description="Confidence tier: High | Moderate | Low | Insufficient Evidence",
    )

    # Supporting detail
    citations: List[CitationEntry] = Field(
        default_factory=list,
        description="Claim-to-source mapping for every statement in the answer",
    )
    unresolved_contradictions: List[str] = Field(
        default_factory=list,
        description="Contradictions between sources that could not be resolved",
    )
    disclaimer: Optional[str] = Field(
        None,
        description="Domain-specific disclaimer (medical / legal / financial)",
    )

    # Process metadata
    sources_used: int = Field(0, description="Number of sources that passed scoring")
    sources_dropped: int = Field(0, description="Number of sources dropped (denylist / validation)")
    loop_rounds: int = Field(0, description="Number of adversarial loop rounds completed")
    skeptic_challenges: List[SkepticChallengeSummary] = Field(
        default_factory=list,
        description="Summary of Skeptic agent challenges this round",
    )

    # HITL
    hitl_status: HITLStatus = Field(
        default="not_required",
        description="Human-in-the-loop review status",
    )

    # Pipeline status
    status: PipelineStatus = Field(
        default="complete",
        description="Overall pipeline status",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if status is 'failed'",
    )


class StatusResponse(BaseModel):
    """Response for GET /status/{request_id}."""

    request_id: str
    status: PipelineStatus
    confidence: Optional[ConfidenceTier] = None
    loop_round: int = 0
    hitl_status: HITLStatus = "not_required"
    message: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response for POST /feedback/{request_id}."""

    acknowledged: bool = True
    request_id: str
    adjustments_applied: int = Field(0, description="Number of credibility adjustments applied")


class ApprovalResponse(BaseModel):
    """Response for POST /approve/{request_id}."""

    request_id: str
    status: Literal["approved", "returned_for_revision"]
    reviewer_notes: Optional[str] = None


class ReplayResponse(BaseModel):
    """Response for GET /replay/{request_id}."""

    request_id: str
    question: str
    final_answer: Optional[str] = None
    confidence: Optional[ConfidenceTier] = None
    timestamp: str
    audit_trail: List[dict] = Field(default_factory=list)
    skeptic_challenges: List[dict] = Field(default_factory=list)
    arbiter_rulings: List[dict] = Field(default_factory=list)