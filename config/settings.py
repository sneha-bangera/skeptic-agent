"""
Central configuration using Pydantic Settings v2.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM 
    groq_api_key: str = Field(..., description="Groq API key (required)")
    model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier",
    )

    # Pipeline Tuning 
    max_loop_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum Synthesis ↔ Skeptic ↔ Arbiter loop rounds",
    )
    min_credibility_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum credibility for a source to be used in evidence extraction",
    )
    min_relevance_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum relevance for a source to be used in evidence extraction",
    )
    source_count_floor: int = Field(
        default=3,
        ge=1,
        description="Below this number of usable sources, confidence is capped at Low",
    )

    #   Embeddings 
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )
    contradiction_similarity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which two claims are checked for contradiction",
    )
    dedup_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which two sources are considered duplicates",
    )

    #   HITL                                
    hitl_confidence_threshold: str = Field(
        default="Moderate",
        description="Confidence tier below which HITL review is triggered",
    )

    #   Storage                              
    audit_db_path: str = Field(
        default="storage/audit.db",
        description="Path to the SQLite audit database",
    )

    #   Domain classification                       
    sensitive_domains: List[str] = Field(
        default=["medical", "legal", "financial"],
        description="Domains that trigger domain disclaimer on output",
    )
    fast_moving_domains: List[str] = Field(
        default=["AI", "pharmaceuticals", "policy"],
        description="Domains where recency weighting is applied aggressively",
    )

    #   Logging                              
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR",
    )

    #   Testing                              
    dry_run: bool = Field(
        default=False,
        description="If True, LLM calls return mock responses without hitting the API",
    )

    #   Scoring constants (not env-overridable, defined here for DRY)    
    @property
    def base_credibility_scores(self) -> dict:
        return {
            "peer_reviewed": 0.85,
            "institutional": 0.75,
            "established_press": 0.60,
            "general_news": 0.45,
            "blog_opinion": 0.25,
            "commercial": 0.10,
            "trusted_source": 0.90,   # allowlist
            "unknown": 0.20,
        }

    @property
    def bias_credibility_caps(self) -> dict:
        return {
            "high": 0.40,
            "moderate": 0.60,
            "low": 1.0,
            "none": 1.0,
        }

    @property
    def known_established_press(self) -> set:
        return {
            "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
            "theguardian.com", "nytimes.com", "washingtonpost.com",
            "wsj.com", "ft.com", "economist.com", "bloomberg.com",
            "npr.org", "pbs.org", "abc.net.au",
        }

    @property
    def institutional_tlds_and_domains(self) -> set:
        return {
            ".gov", ".edu", ".mil", ".ac.uk", ".ac.in",
            "who.int", "nih.gov", "cdc.gov", "cochrane.org",
            "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
        }

    @field_validator("hitl_confidence_threshold")
    @classmethod
    def validate_hitl_tier(cls, v: str) -> str:
        valid = {"High", "Moderate", "Low", "Insufficient Evidence"}
        if v not in valid:
            raise ValueError(f"hitl_confidence_threshold must be one of {valid}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()


#   Singleton accessor                             
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings() 
    return _settings