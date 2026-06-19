"""
Pydantic schema for the POST /analyze API request body.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class SourceInput(BaseModel):
    """A single source provided by the API caller."""

    url: Optional[str] = Field(None, description="URL of the source (optional if content provided)")
    content: str = Field(
        ...,
        min_length=20,
        description="Raw content text or HTML. Must be non-empty.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Optional metadata: title, author, published_date, doi, "
            "journal, issn, citation_count, etc."
        ),
    )

    @field_validator("content")
    @classmethod
    def content_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""

    question: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The research question to answer",
    )
    sources: List[SourceInput] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of sources (1–20). Each must have non-empty content.",
    )

    @field_validator("question")
    @classmethod
    def question_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return v

    @model_validator(mode="after")
    def at_least_one_valid_source(self) -> "AnalyzeRequest":
        if not self.sources:
            raise ValueError("At least one source must be provided")
        return self