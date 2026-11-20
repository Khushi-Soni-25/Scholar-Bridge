"""
ScholarBridge LangGraph Workflow Service Facade
================================================
Central entrypoint for executing LangGraph-orchestrated workflows.
Handles state initialization, checkpointer session configs, execution tracing,
and structured response formatting.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional

from .state import ScholarBridgeWorkflowState, WorkflowIntent
from .graph import get_compiled_workflow, create_scholarbridge_graph

logger = logging.getLogger(__name__)


class ScholarBridgeWorkflowService:
    """Service wrapper for executing compiled ScholarBridge LangGraph workflows."""

    def __init__(self, workflow: Optional[Any] = None):
        self.workflow = workflow or get_compiled_workflow()

    def run(
        self,
        user_query: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        selected_scholarships: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        intent_override: Optional[str] = None,
        max_research_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Executes end-to-end workflow through LangGraph state machine.
        """
        active_thread_id = thread_id or f"session-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Build initial input state for this turn
        # Omit None keys (such as student_profile when not passed on Turn 2)
        # so LangGraph's checkpointer preserves existing checkpointed state values for this thread_id.
        initial_state: Dict[str, Any] = {
            "user_query": user_query or "",
            "thread_id": active_thread_id,
            "research_iterations": 0,
            "max_research_iterations": max_research_iterations,
            "evidence_sufficient": True,
            "errors": [],
            "trace": {
                "session_id": active_thread_id,
                "start_timestamp": time.time(),
            }
        }
        if profile is not None:
            initial_state["student_profile"] = profile
        if selected_scholarships is not None and len(selected_scholarships) > 0:
            initial_state["selected_scholarships"] = selected_scholarships
        if intent_override is not None:
            initial_state["intent_override"] = intent_override

        config = {"configurable": {"thread_id": active_thread_id}}

        try:
            final_state = self.workflow.invoke(initial_state, config=config)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            trace = final_state.get("trace", {})
            trace["total_workflow_ms"] = round(duration_ms, 2)

            return {
                "thread_id": active_thread_id,
                "intent": final_state.get("intent", WorkflowIntent.UNKNOWN.value),
                "status": "COMPLETED" if not final_state.get("errors") else "COMPLETED_WITH_ERRORS",
                "final_response": final_state.get("final_response"),
                "citations": final_state.get("citations", []),
                "grounding_status": final_state.get("grounding_status", "PENDING"),
                "research_iterations": final_state.get("research_iterations", 0),
                "errors": final_state.get("errors", []),
                "trace": trace
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "thread_id": active_thread_id,
                "intent": intent_override or WorkflowIntent.UNKNOWN.value,
                "status": "FAILED",
                "final_response": None,
                "citations": [],
                "grounding_status": "FAILED",
                "research_iterations": 0,
                "errors": [str(e)],
                "trace": {
                    "session_id": active_thread_id,
                    "total_workflow_ms": round(duration_ms, 2),
                    "error": str(e)
                }
            }
