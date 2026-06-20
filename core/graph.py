"""
LangGraph graph builder — assembles the full pipeline with the adversarial loop.

Graph topology:
  START
    → input_guardrail          (guardrails/input_guardrail.py)
        ↓ conditional: rejected? → rejection_formatter → END
        ↓ passed
    → ingress                  (agents/ingress.py)
    → source_scorer            (agents/source_scorer.py)
    → source_count_guardrail   (guardrails/source_count_guardrail.py)
    → evidence_extractor       (agents/evidence_extractor.py)
    → contradiction_detector   (agents/contradiction_detector.py)
    → synthesis                (agents/synthesis.py)         ←──────────────┐
    → skeptic                  (agents/skeptic.py)                          │
    → arbiter                  (agents/arbiter.py)                          │
        ↓ conditional edge (CODE-LEVEL — NOT LLM):                          │
            "revise"   ─────────────────────────────────────────────────────┘
            "finalize"
    → confidence_floor         (guardrails/confidence_floor.py)
    → egress                   (agents/egress.py)
    → audit                    (agents/audit.py)
  END

CRITICAL: The "revise" vs "finalize" routing is a Python function that reads
state["_arbiter_route"] AND enforces loop_round < max_loop_rounds.
The LLM is never asked to make the termination decision.
"""

from __future__ import annotations

import structlog
from langgraph.graph import StateGraph, END

from core.state import PipelineState, make_initial_state
from guardrails.input_guardrail import run_input_guardrail
from guardrails.source_count_guardrail import run_source_count_guardrail
from guardrails.confidence_floor import run_confidence_floor
from agents.ingress import run_ingress
from agents.source_scorer import run_source_scorer
from agents.evidence_extractor import run_evidence_extractor
from agents.contradiction_detector import run_contradiction_detector
from agents.synthesis import run_synthesis
from agents.skeptic import run_skeptic
from agents.arbiter import run_arbiter
from agents.egress import run_egress
from agents.audit import run_audit

logger = structlog.get_logger(__name__)


# Node name constants 

NODE_INPUT_GUARDRAIL = "input_guardrail"
NODE_INGRESS = "ingress"
NODE_SOURCE_SCORER = "source_scorer"
NODE_SOURCE_COUNT_GUARDRAIL = "source_count_guardrail"
NODE_EVIDENCE_EXTRACTOR = "evidence_extractor"
NODE_CONTRADICTION_DETECTOR = "contradiction_detector"
NODE_SYNTHESIS = "synthesis"
NODE_SKEPTIC = "skeptic"
NODE_ARBITER = "arbiter"
NODE_CONFIDENCE_FLOOR = "confidence_floor"
NODE_EGRESS = "egress"
NODE_AUDIT = "audit"
NODE_REJECTION = "rejection_formatter"


# Rejection formatter 

def _format_rejection(state: PipelineState) -> dict:
    """
    Terminal node: formats a rejected-input response and short-circuits the pipeline.
    Sets final_answer and confidence directly so Egress can still format output.
    """
    from datetime import datetime, timezone
    reason = state.get("input_rejection_reason", "Input rejected by guardrail.")
    return {
        "final_answer": f"Request rejected: {reason}",
        "confidence": "Insufficient Evidence",
        "loop_terminated": True,
        "audit_trail": state["audit_trail"] + [{
            "event": "pipeline_rejected",
            "stage": "rejection_formatter",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": state["request_id"],
            "data": {"reason": reason},
        }],
    }


# Conditional edge functions 

def _route_after_guardrail(state: PipelineState) -> str:
    """After input_guardrail: rejected → rejection_formatter, else → ingress."""
    if state.get("input_rejected"):
        logger.info(
            "graph_route_rejected",
            request_id=state.get("request_id"),
            reason=state.get("input_rejection_reason"),
        )
        return NODE_REJECTION
    return NODE_INGRESS


def _route_after_arbiter(state: PipelineState) -> str:
    """
    After arbiter: "revise" → synthesis, "finalize" → confidence_floor.

    CRITICAL: This is a Python function — NOT an LLM call. The loop_round
    ceiling is enforced HERE as a hard code-level guard.
    """
    from config.settings import get_settings
    settings = get_settings()

    route = state.get("_arbiter_route", "finalize")
    loop_round = state.get("loop_round", 0)
    request_id = state.get("request_id", "")

    logger.info(
        "graph_arbiter_route",
        request_id=request_id,
        arbiter_route=route,
        loop_round=loop_round,
        max_rounds=settings.max_loop_rounds,
    )

    # Hard ceiling — even if Arbiter says "revise", stop if at max rounds
    if route == "revise" and loop_round < settings.max_loop_rounds:
        return NODE_SYNTHESIS

    # Any other case (finalize, or revise-but-at-ceiling) → confidence_floor
    return NODE_CONFIDENCE_FLOOR


# Graph builder 

def build_graph() -> StateGraph:
    """
    Build and compile the full LangGraph pipeline.
    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(PipelineState)

    # Register all nodes 
    graph.add_node(NODE_INPUT_GUARDRAIL, run_input_guardrail)
    graph.add_node(NODE_REJECTION, _format_rejection)
    graph.add_node(NODE_INGRESS, run_ingress)
    graph.add_node(NODE_SOURCE_SCORER, run_source_scorer)
    graph.add_node(NODE_SOURCE_COUNT_GUARDRAIL, run_source_count_guardrail)
    graph.add_node(NODE_EVIDENCE_EXTRACTOR, run_evidence_extractor)
    graph.add_node(NODE_CONTRADICTION_DETECTOR, run_contradiction_detector)
    graph.add_node(NODE_SYNTHESIS, run_synthesis)
    graph.add_node(NODE_SKEPTIC, run_skeptic)
    graph.add_node(NODE_ARBITER, run_arbiter)
    graph.add_node(NODE_CONFIDENCE_FLOOR, run_confidence_floor)
    graph.add_node(NODE_EGRESS, run_egress)
    graph.add_node(NODE_AUDIT, run_audit)

    # Set entry point 
    graph.set_entry_point(NODE_INPUT_GUARDRAIL)

    # Linear edges (pre-loop) 
    graph.add_conditional_edges(
        NODE_INPUT_GUARDRAIL,
        _route_after_guardrail,
        {
            NODE_REJECTION: NODE_REJECTION,
            NODE_INGRESS: NODE_INGRESS,
        },
    )
    graph.add_edge(NODE_REJECTION, NODE_EGRESS)
    graph.add_edge(NODE_INGRESS, NODE_SOURCE_SCORER)
    graph.add_edge(NODE_SOURCE_SCORER, NODE_SOURCE_COUNT_GUARDRAIL)
    graph.add_edge(NODE_SOURCE_COUNT_GUARDRAIL, NODE_EVIDENCE_EXTRACTOR)
    graph.add_edge(NODE_EVIDENCE_EXTRACTOR, NODE_CONTRADICTION_DETECTOR)
    graph.add_edge(NODE_CONTRADICTION_DETECTOR, NODE_SYNTHESIS)

    # Adversarial loop edges 
    graph.add_edge(NODE_SYNTHESIS, NODE_SKEPTIC)
    graph.add_edge(NODE_SKEPTIC, NODE_ARBITER)
    graph.add_conditional_edges(
        NODE_ARBITER,
        _route_after_arbiter,
        {
            NODE_SYNTHESIS: NODE_SYNTHESIS,        # revise → loop back
            NODE_CONFIDENCE_FLOOR: NODE_CONFIDENCE_FLOOR,  # finalize → proceed
        },
    )

    # Post-loop edges 
    graph.add_edge(NODE_CONFIDENCE_FLOOR, NODE_EGRESS)
    graph.add_edge(NODE_EGRESS, NODE_AUDIT)
    graph.add_edge(NODE_AUDIT, END)

    return graph


def compile_graph():
    """Build and compile the graph. Call this once at app startup."""
    graph = build_graph()
    return graph.compile()


# Convenience runner 

def run_pipeline(
    request_id: str,
    question: str,
    raw_sources: list[dict],
) -> PipelineState:
    """
    End-to-end pipeline runner. Takes raw API inputs and returns the final state.

    Args:
        request_id:   Unique identifier for this request.
        question:     The research question.
        raw_sources:  List of raw source dicts (url, content, metadata).

    Returns:
        The final PipelineState after all nodes have executed.
    """
    compiled = compile_graph()
    initial_state = make_initial_state(request_id, question, raw_sources)

    logger.info(
        "pipeline_start",
        request_id=request_id,
        question_len=len(question),
        source_count=len(raw_sources),
    )

    final_state = compiled.invoke(initial_state)

    logger.info(
        "pipeline_complete",
        request_id=request_id,
        confidence=final_state.get("confidence"),
        loop_rounds=final_state.get("loop_round", 0),
    )

    return final_state