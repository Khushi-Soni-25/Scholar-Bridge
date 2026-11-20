"""
ScholarBridge LLM Structured Schemas
====================================
Defines strict Pydantic models for LLM prompt payloads, raw generation outputs,
and validated, authoritative ScholarshipResponses.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ScholarshipRecommendationItem(BaseModel):
    """A ranked scholarship recommendation item."""
    model_config = ConfigDict(extra="forbid")

    scholarship_id: str = Field(..., description="Unique scheme identifier, e.g. MYSY, BCK-10")
    scholarship_name: str = Field(..., description="Official title of the scholarship")
    rank: int = Field(..., ge=1, description="1-indexed recommendation ranking position")
    score: float = Field(..., ge=0.0, le=100.0, description="Multi-factor heuristic recommendation score")
    eligibility_status: str = Field(..., description="Deterministic eligibility outcome: eligible, conditional, ineligible, suspended")
    reason: str = Field(..., description="Explanation of recommendation ranking and eligibility outcome")


class DocumentItem(BaseModel):
    """Authoritative document requirement item."""
    model_config = ConfigDict(extra="forbid")

    document_name: str = Field(..., description="Official title of the document")
    document_category: str = Field("General", description="Category: Identity_Proof, Income_Proof, Academic_Proof, etc.")
    is_mandatory: bool = Field(True, description="True if mandatory; False if conditional")
    purpose: str = Field("", description="Purpose or condition for requesting this document")


class LLMGeneratedContent(BaseModel):
    """
    Contract for raw generation from LLM providers.
    Captures natural language explanations, reasoning summaries, and citations.
    All authoritative fields (status, rank, score, documents, next_steps, metadata)
    are strictly constructed and enforced by LLMService.
    """
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(..., description="Clear, concise natural language answer explaining eligibility grounded in facts")
    explanation_points: List[str] = Field(default_factory=list, description="Specific rule-by-rule or criterion explanations")
    reasoning_summary: str = Field("", description="High-level reasoning synthesis grounded in authoritative evidence")
    warnings: List[str] = Field(default_factory=list, description="Important cautions, exclusions, or verification notes")
    citations: List[str] = Field(default_factory=list, description="List of referenced Evidence Citation IDs, e.g. ['EV-...']")


class GroundingMetadata(BaseModel):
    """Service-generated provenance and audit metadata."""
    model_config = ConfigDict(extra="forbid")

    engine_status: str = Field(..., description="Authoritative deterministic status: ELIGIBLE, CONDITIONAL, INELIGIBLE, SUSPENDED")
    engine_score: float = Field(..., description="Authoritative composite score from RecommendationScorer")
    evidence_count: int = Field(..., description="Number of validated evidence items retrieved")
    citation_ids: List[str] = Field(default_factory=list, description="Valid citation IDs present in the retrieved evidence pack")
    provider: str = Field(..., description="Name of the LLM provider used")
    model: str = Field(..., description="Name of the model invoked")
    validation_status: str = Field(..., description="PASSED or FAILED")
    authority_model: str = Field("Engine Decides. RAG Proves. LLM Explains.", description="Architecture authority declaration")


class ScholarshipResponse(BaseModel):
    """
    Standardized, authoritative response contract returned by ScholarBridge LLM Service.
    Enforces complete service-side ownership of authoritative fields (status, rank, score, documents, next_steps, citations, metadata).
    """
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="Natural language explanation grounded in evidence and engine decisions")
    scholarship_id: str = Field(..., description="Authoritative scholarship ID from Knowledge Base")
    scholarship_name: str = Field(..., description="Authoritative scholarship name from Knowledge Base")
    eligibility_status: str = Field(..., description="Authoritative eligibility status: eligible, conditional, ineligible, suspended")
    recommendations: List[ScholarshipRecommendationItem] = Field(default_factory=list, description="Rank-ordered recommendations from deterministic engine")
    eligibility_explanation: List[str] = Field(default_factory=list, description="Grounded explanation points")
    documents: List[DocumentItem] = Field(default_factory=list, description="Authoritative required documents from deterministic knowledge base")
    next_steps: List[str] = Field(default_factory=list, description="Authoritative next steps derived from portal, deadline, and rule data")
    warnings: List[str] = Field(default_factory=list, description="Authoritative warnings and exclusions")
    citations: List[str] = Field(default_factory=list, description="Validated citation IDs linking to official evidence")
    grounding_metadata: Dict[str, Any] = Field(default_factory=dict, description="Service-generated audit and provenance metadata")
