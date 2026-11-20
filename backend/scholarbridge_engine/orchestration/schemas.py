"""
ScholarBridge Orchestration Schemas
====================================
Data contracts and schemas for LangChain tools, chains, structured outputs,
and observability telemetry.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time


class ScholarshipExplanationSchema(BaseModel):
    """Structured output contract for LLM-generated scholarship explanations."""
    answer: str = Field(
        description="Clear, student-friendly explanation of the scholarship result."
    )
    explanation_points: List[str] = Field(
        default_factory=list,
        description="Bulleted breakdown grounded strictly in deterministic rules and official evidence."
    )
    reasoning_summary: str = Field(
        description="High-level reasoning grounded in authoritative scheme guidelines."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Important statutory compliance warnings or prerequisite notices."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of citation IDs (e.g. ['EV-MYSY-01']) strictly referenced from retrieved evidence."
    )


class ComparisonItemSchema(BaseModel):
    """Detailed comparison between evaluated scholarships."""
    scholarship_id: str
    scholarship_name: str
    eligibility_status: str
    benefit_summary: str
    key_requirements: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class ScholarshipComparisonResponse(BaseModel):
    """Structured response for scholarship comparison."""
    student_summary: str
    compared_scholarships: List[ComparisonItemSchema] = Field(default_factory=list)
    comparative_analysis: str
    recommendation_verdict: str
    citations: List[str] = Field(default_factory=list)
    trace: Optional[Dict[str, Any]] = None


class OrchestrationTrace(BaseModel):
    """
    Lightweight orchestration-level observability trace.
    Captures stage latencies, execution path, provider details, and validation state.
    """
    request_id: str = Field(default_factory=lambda: f"trace-{int(time.time()*1000)}")
    chain_name: str = "ScholarBridgeOrchestrationChain"
    llm_provider: str = "unknown"
    llm_model: str = "unknown"
    
    # Timing breakdown (milliseconds)
    deterministic_engine_ms: float = 0.0
    hybrid_rag_ms: float = 0.0
    cross_feature_rerank_ms: float = 0.0
    llm_generation_ms: float = 0.0
    citation_validation_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    
    # Validation telemetry
    evidence_count: int = 0
    citations_referenced: int = 0
    citations_valid: int = 0
    validation_status: str = "PENDING"
    authority_invariant_preserved: bool = True
    errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()