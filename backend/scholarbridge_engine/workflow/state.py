"""
ScholarBridge LangGraph State Definitions
==========================================
Defines the strongly-typed graph state model for Phase 5 workflow orchestration.

Distinguishes:
  1. Input State: user query, profile data, selected schemes, session identifiers.
  2. Derived State: deterministic engine output, RAG evidence, LLM explanations, citations.
  3. Control State: intent, multi-step research iteration counters, routing flags, errors, traces.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, TypedDict


class WorkflowIntent(str, Enum):
    """Supported workflow intents for the LangGraph state machine."""
    RECOMMEND = "RECOMMEND"
    SEARCH = "SEARCH"
    COMPARE = "COMPARE"
    EXPLAIN = "EXPLAIN"
    UNKNOWN = "UNKNOWN"


class ScholarBridgeWorkflowState(TypedDict, total=False):
    """
    Typed graph state passed between all nodes in the ScholarBridge LangGraph workflow.
    """
    # 1. Input State
    user_query: str
    student_profile: Optional[Dict[str, Any]]
    selected_scholarships: List[str]
    thread_id: Optional[str]
    intent_override: Optional[str]

    # 2. Derived State (Produced by Engine, RAG, and LLM nodes)
    intent: str
    engine_result: Optional[Dict[str, Any]]
    retrieved_evidence: List[Dict[str, Any]]
    comparison_result: Optional[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]]
    citations: List[str]
    grounding_status: str  # "PASSED", "FAILED", "PENDING", "NOT_REQUIRED"

    # 3. Control & Multi-Step Research State
    research_iterations: int
    max_research_iterations: int
    evidence_sufficient: bool
    current_step: str
    errors: List[str]
    trace: Dict[str, Any]
    final_response: Optional[Dict[str, Any]]
