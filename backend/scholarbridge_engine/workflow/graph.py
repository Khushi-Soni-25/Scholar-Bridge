"""
ScholarBridge LangGraph StateGraph Definition & Compilation
============================================================
Constructs the complete Phase 5 LangGraph state graph with:
  1. Intent Routing (RECOMMEND / SEARCH / COMPARE / EXPLAIN)
  2. Deterministic Engine Authority
  3. Multi-Step Research Refinement Loop (Bounded Iterations)
  4. Conditional Evidence Quality Routing
  5. Checkpointed Conversational State Management (MemorySaver)
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ScholarBridgeWorkflowState, WorkflowIntent
from .router import route_intent
from .nodes import (
    node_recommend,
    node_retrieve_evidence,
    node_evaluate_evidence,
    node_refine_search,
    node_explain_recommendations,
    node_synthesize_search,
    node_compare_scholarships,
    node_explain_eligibility,
    node_validate_and_finalize,
)


def _route_by_intent(state: ScholarBridgeWorkflowState) -> str:
    """Conditional edge router branching from route_intent."""
    intent = state.get("intent", WorkflowIntent.SEARCH.value)

    if intent == WorkflowIntent.RECOMMEND.value:
        return "node_recommend"
    elif intent == WorkflowIntent.SEARCH.value:
        return "node_retrieve_evidence"
    elif intent == WorkflowIntent.COMPARE.value:
        return "node_compare_scholarships"
    elif intent == WorkflowIntent.EXPLAIN.value:
        return "node_explain_eligibility"
    else:
        return "node_validate_and_finalize"


def _route_after_evidence_eval(state: ScholarBridgeWorkflowState) -> str:
    """Conditional edge router for multi-step research and evidence sufficiency."""
    sufficient = state.get("evidence_sufficient", True)
    iterations = state.get("research_iterations", 0)
    max_iterations = state.get("max_research_iterations", 2)
    intent = state.get("intent", WorkflowIntent.SEARCH.value)

    # Trigger research refinement if evidence is insufficient and under iteration limit
    if not sufficient and iterations < max_iterations:
        return "node_refine_search"

    # Otherwise route to the appropriate synthesis node
    if intent == WorkflowIntent.SEARCH.value:
        return "node_synthesize_search"
    elif intent == WorkflowIntent.RECOMMEND.value:
        return "node_explain_recommendations"
    else:
        return "node_validate_and_finalize"


def create_scholarbridge_graph(checkpointer: Optional[MemorySaver] = None) -> Any:
    """
    Constructs and compiles the ScholarBridge LangGraph workflow.
    """
    builder = StateGraph(ScholarBridgeWorkflowState)

    # 1. Add all nodes
    builder.add_node("route_intent", route_intent)
    builder.add_node("node_recommend", node_recommend)
    builder.add_node("node_retrieve_evidence", node_retrieve_evidence)
    builder.add_node("node_evaluate_evidence", node_evaluate_evidence)
    builder.add_node("node_refine_search", node_refine_search)
    builder.add_node("node_explain_recommendations", node_explain_recommendations)
    builder.add_node("node_synthesize_search", node_synthesize_search)
    builder.add_node("node_compare_scholarships", node_compare_scholarships)
    builder.add_node("node_explain_eligibility", node_explain_eligibility)
    builder.add_node("node_validate_and_finalize", node_validate_and_finalize)

    # 2. Add edges from START
    builder.add_edge(START, "route_intent")

    # 3. Intent Branching conditional edge
    builder.add_conditional_edges(
        "route_intent",
        _route_by_intent,
        {
            "node_recommend": "node_recommend",
            "node_retrieve_evidence": "node_retrieve_evidence",
            "node_compare_scholarships": "node_compare_scholarships",
            "node_explain_eligibility": "node_explain_eligibility",
            "node_validate_and_finalize": "node_validate_and_finalize",
        }
    )

    # 4. RECOMMEND workflow edges
    builder.add_edge("node_recommend", "node_retrieve_evidence")

    # 5. Evidence Evaluation & Multi-Step Research Refinement Loop
    builder.add_edge("node_retrieve_evidence", "node_evaluate_evidence")
    builder.add_conditional_edges(
        "node_evaluate_evidence",
        _route_after_evidence_eval,
        {
            "node_refine_search": "node_refine_search",
            "node_synthesize_search": "node_synthesize_search",
            "node_explain_recommendations": "node_explain_recommendations",
            "node_validate_and_finalize": "node_validate_and_finalize",
        }
    )
    # Loop refinement back to evaluation
    builder.add_edge("node_refine_search", "node_evaluate_evidence")

    # 6. Downstream synthesis edges to finalization
    builder.add_edge("node_synthesize_search", "node_validate_and_finalize")
    builder.add_edge("node_explain_recommendations", "node_validate_and_finalize")
    builder.add_edge("node_compare_scholarships", "node_validate_and_finalize")
    builder.add_edge("node_explain_eligibility", "node_validate_and_finalize")

    # 7. Finalization edge to END
    builder.add_edge("node_validate_and_finalize", END)

    # 8. Compile graph with memory checkpointer
    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    compiled_graph = builder.compile(checkpointer=active_checkpointer)
    return compiled_graph


# Global compiled singleton
_COMPILED_WORKFLOW: Optional[Any] = None

def get_compiled_workflow() -> Any:
    """Returns or creates the global compiled LangGraph workflow instance."""
    global _COMPILED_WORKFLOW
    if _COMPILED_WORKFLOW is None:
        _COMPILED_WORKFLOW = create_scholarbridge_graph()
    return _COMPILED_WORKFLOW
