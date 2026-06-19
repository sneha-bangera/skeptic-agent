"""
Pydantic schemas for source documents as they flow through the pipeline.
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


BiasFlag = Literal["none", "low", "moderate", "high"]
SourceType = Literal[
    "peer_reviewed",
    "institutional",
    "established_press",
    "general_news",
    "blog_opinion",
    "commercial",
    "trusted_source",
    "unknown",
]


class Source(BaseModel):
    """A source after normalization by the Ingress agent."""

    source_id: str = Field(..., description="Stable identifier: src_001, src_002, …")
    url: Optional[str] = Field(None, description="Source URL if available")
    title: Optional[str] = Field(None, description="Article/page title")
    author: Optional[str] = Field(None, description="Author name(s)")
    domain: Optional[str] = Field(None, description="Extracted domain from URL")
    published_date: Optional[str] = Field(None, description="ISO date string or raw date text")
    content_text: str = Field(..., min_length=1, description="Cleaned plain-text content")
    metadata: dict = Field(default_factory=dict, description="Any additional metadata")

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not v.startswith("src_"):
            raise ValueError("source_id must start with 'src_'")
        return v


class ScoredSource(Source):
    """A source after scoring by the Source Scorer agent."""

    credibility: float = Field(..., ge=0.0, le=1.0, description="Credibility score 0–1")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0–1")
    bias_flag: BiasFlag = Field(default="none", description="Bias severity flag")
    source_type: SourceType = Field(default="unknown", description="Classified source type")
    trusted_source: bool = Field(
        default=False,
        description="True if domain is on the allowlist",
    )
    score_breakdown: dict = Field(
        default_factory=dict,
        description="Detailed component scores for audit purposes",
    )

    @property
    def combined_score(self) -> float:
        """Simple average of credibility and relevance, used for fallback ranking."""
        return (self.credibility + self.relevance) / 2

    @property
    def is_usable(self) -> bool:
        """True if the source meets minimum thresholds for evidence extraction."""
        # Thresholds are checked against config at runtime; this is a convenience property
        return self.credibility >= 0.20 and self.relevance >= 0.30