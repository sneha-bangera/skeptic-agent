"""
Pydantic schema for atomic, falsifiable claims extracted from sources.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    """
    An atomic, falsifiable claim extracted from a scored source.
    Every claim is anchored to exactly one source_id.
    """

    claim_id: str = Field(..., description="Stable identifier: clm_001, clm_002, …")
    source_id: str = Field(..., description="The source this claim was extracted from")
    text: str = Field(..., min_length=5, description="The claim text — specific and falsifiable")

    # Scope tags (set by Evidence Extractor via LLM)
    population: Optional[str] = Field(
        None,
        description="Who the finding applies to, e.g. 'healthy adults 40–70'",
    )
    timeframe: Optional[str] = Field(
        None,
        description="Measurement duration, e.g. '12 months', 'cross-sectional'",
    )
    outcome_metric: Optional[str] = Field(
        None,
        description="What was measured, e.g. 'LDL reduction', 'cardiovascular events'",
    )
    study_design: Optional[str] = Field(
        None,
        description="How it was measured, e.g. 'RCT', 'observational', 'meta-analysis'",
    )

    # Quality flags
    low_credibility: bool = Field(
        default=False,
        description="True if the source credibility is below min_credibility_threshold",
    )

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, v: str) -> str:
        if not v.startswith("clm_"):
            raise ValueError("claim_id must start with 'clm_'")
        return v

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not v.startswith("src_"):
            raise ValueError("source_id must start with 'src_'")
        return v

    def to_dict(self) -> dict:
        return self.model_dump()