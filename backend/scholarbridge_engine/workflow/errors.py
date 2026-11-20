"""
ScholarBridge Workflow Exception Hierarchy
==========================================
Domain-specific exceptions for Phase 5 LangGraph workflow execution.
"""


class ScholarBridgeWorkflowError(Exception):
    """Base exception for all workflow orchestration errors."""
    pass


class IntentRoutingError(ScholarBridgeWorkflowError):
    """Raised when intent routing fails or encounters ambiguous unroutable state."""
    pass


class ResearchRefinementLimitError(ScholarBridgeWorkflowError):
    """Raised when research iteration bounds are violated."""
    pass


class InvalidWorkflowStateError(ScholarBridgeWorkflowError):
    """Raised when graph state lacks mandatory attributes for a specific node."""
    pass
