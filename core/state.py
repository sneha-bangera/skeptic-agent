"""
The single shared PipelineState TypedDict that flows through every node
in the LangGraph graph.

Design rules:
- Every agent reads from state and returns a dict of only the fields it modifies.
- Lists are APPEND-ONLY. Never mutate existing entries — add new records.
- Optional fields start as None; agents must guard against None before reading.
- audit_trail is the append-only log of every agent step and is written by every agent.
"""

from __future__ import annotations

from typing import List, Optional, Literal, TypedDict


# Type Aliases  

ConfidenceTier = Literal["High", "Moderate", "Low", "Insufficient Evidence"]
BiasFlag = Literal["none", "low", "moderate", "high"]
ConflictType = Literal["no_conflict", "partial_conflict", "direct_conflict", "irreconcilable"]
ChallengeType = Literal[
    "wrong_population",
    "commercial_bias",
    "outcome_mismatch",
    "sample_size",
    "recency_gap",
    "missing_counter_evidence",
    "ignored_conflict",
]
ChallengeSeverity = Literal["critical", "moderate", "minor"]
ArbiterRuling = Literal["valid_critical", "valid_minor", "invalid", "dismissed"]


# PipelineState  

class PipelineState(TypedDict):
    """
    The central data contract for the entire pipeline.
    Flows through every LangGraph node.
    """

    # Identity       
    request_id: str
    question: str
    raw_sources: List[dict]

    # Input guardrail output    
    input_domain_flag: Optional[str]         # "medical" | "legal" | "financial" | None
    input_rejected: bool
    input_rejection_reason: Optional[str]

    # Source count guardrail output   
    source_count_ceiling: Optional[ConfidenceTier]  # set if < source_count_floor usable sources

    # After Ingress agent     
    normalized_sources: List[dict]
    # Each entry: {
    #   source_id: str,       # "src_001", "src_002", …
    #   url: Optional[str],
    #   title: Optional[str],
    #   author: Optional[str],
    #   domain: Optional[str],
    #   published_date: Optional[str],
    #   content_text: str,
    #   metadata: dict,
    # }

    # After Source Scorer agent    
    scored_sources: List[dict]
    # Each entry: {
    #   source_id: str,
    #   credibility: float,       # 0.0–1.0
    #   relevance: float,         # 0.0–1.0
    #   bias_flag: BiasFlag,
    #   source_type: str,
    #   trusted_source: bool,
    #   score_breakdown: dict,    # detailed component scores for audit
    # }
    dropped_sources: List[dict]
    # Each entry: {source_id, domain, reason: "denylist" | "validation_failed"}

    # After Evidence Extractor agent   
    claims: List[dict]
    # Each entry: {
    #   claim_id: str,            # "clm_001", …
    #   source_id: str,
    #   text: str,
    #   population: Optional[str],
    #   timeframe: Optional[str],
    #   outcome_metric: Optional[str],
    #   study_design: Optional[str],
    #   low_credibility: bool,    # True if source credibility < min_credibility_threshold
    # }

    # After Contradiction Detector agent  
    contradictions: List[dict]
    # Each entry: {
    #   conflict_id: str,         # "con_001", …
    #   claim_a_id: str,
    #   claim_b_id: str,
    #   conflict_type: ConflictType,
    #   explanation: str,
    #   similarity_score: float,
    # }

    # Synthesis ↔ Skeptic ↔ Arbiter loop  
    preliminary_answer: Optional[str]

    skeptic_challenges: List[dict]
    # Each entry: {
    #   challenge_id: str,        # "c_001", …
    #   type: ChallengeType,
    #   description: str,
    #   source_ids_implicated: List[str],
    #   severity: ChallengeSeverity,
    #   suggested_revision: str,
    #   round: int,               # which loop round produced this challenge
    # }

    arbiter_rulings: List[dict]
    # Each entry: {
    #   challenge_id: str,
    #   ruling: ArbiterRuling,
    #   reason: str,
    #   round: int,
    # }

    loop_round: int               # 0-indexed; incremented by Arbiter before routing back
    loop_terminated: bool

    # Final output fields 
    final_answer: Optional[str]
    confidence: Optional[ConfidenceTier]
    unresolved_contradictions: List[str]  # human-readable descriptions

    claim_source_map: List[dict]
    # Each entry: {
    #   claim_text: str,
    #   source_id: str,
    #   credibility: float,
    #   relevance: float,
    # }

    disclaimer: Optional[str]     # appended by Egress if domain flag is set
    api_response: Optional[dict]  # formatted APIResponse dict set by Egress

    # HITL  
    hitl_required: bool
    hitl_approved: Optional[bool]
    hitl_reviewer_notes: Optional[str]

    # Audit  
    audit_trail: List[dict]
    # Each entry appended by every agent: {
    #   event: str,
    #   stage: str,
    #   timestamp: str,           # ISO 8601
    #   request_id: str,
    #   data: dict,               # agent-specific payload
    # }


# State factory 

def make_initial_state(request_id: str, question: str, raw_sources: List[dict]) -> PipelineState:
    """
    Build a fresh PipelineState with all required fields initialized to safe defaults.
    Called once per request at the API layer before the graph is invoked.
    """
    return PipelineState(
        # Identity
        request_id=request_id,
        question=question,
        raw_sources=raw_sources,

        # Guardrail flags
        input_domain_flag=None,
        input_rejected=False,
        input_rejection_reason=None,
        source_count_ceiling=None,

        # Ingress
        normalized_sources=[],

        # Source scorer
        scored_sources=[],
        dropped_sources=[],

        # Evidence extractor
        claims=[],

        # Contradiction detector
        contradictions=[],

        # Loop
        preliminary_answer=None,
        skeptic_challenges=[],
        arbiter_rulings=[],
        loop_round=0,
        loop_terminated=False,

        # Final output
        final_answer=None,
        confidence=None,
        unresolved_contradictions=[],
        claim_source_map=[],
        disclaimer=None,
        api_response=None,

        # HITL
        hitl_required=False,
        hitl_approved=None,
        hitl_reviewer_notes=None,

        # Audit
        audit_trail=[],
    )