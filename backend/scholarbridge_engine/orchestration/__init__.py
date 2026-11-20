"""
ScholarBridge LangChain Orchestration Package
==============================================
Provides controlled, deterministic LangChain orchestration over Phase 1-3 components.

Exported Components:
  - LangChainOrchestrationService: Central service facade
  - ScholarBridgeTools: Tool bindings for deterministic engine, Hybrid RAG, and Citation Validator
  - GroundedExplanationChain: LangChain chain for grounded eligibility explanations
  - ScholarshipComparisonChain: LangChain chain for multi-scholarship comparative analysis
  - ScholarBridgeCallbackHandler: Lightweight observability tracing callback
  - Schemas: Data contracts for traces and structured outputs
"""

from .service import LangChainOrchestrationService
from .tools import ScholarBridgeTools
from .chains import GroundedExplanationChain, ScholarshipComparisonChain
from .callbacks import ScholarBridgeCallbackHandler
from .schemas import (
    ScholarshipExplanationSchema,
    ScholarshipComparisonResponse,
    ComparisonItemSchema,
    OrchestrationTrace,
)

__all__ = [
    "LangChainOrchestrationService",
    "ScholarBridgeTools",
    "GroundedExplanationChain",
    "ScholarshipComparisonChain",
    "ScholarBridgeCallbackHandler",
    "ScholarshipExplanationSchema",
    "ScholarshipComparisonResponse",
    "ComparisonItemSchema",
    "OrchestrationTrace",
]