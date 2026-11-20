"""
ScholarBridge LangChain Tools
=============================
Wraps existing Phase 1 (Deterministic Engine), Phase 2 (Hybrid RAG),
and Citation Validation without modifying or replacing underlying logic.

Authority Hierarchy Enforced:
  1. Engine decides eligibility and ranking
  2. Hybrid RAG retrieves and proves evidence
  3. LLM explains and tools validate
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

from scholarbridge_engine.models import StudentProfile, EngineOutput, MatchResult, RecommendationItem
from scholarbridge_engine.engine import ScholarBridgeEngine
from scholarbridge_engine.evidence.retriever import HybridEvidenceRetriever, get_evidence_retriever
from scholarbridge_engine.evidence.models import EvidenceItem
from scholarbridge_engine.evidence.citations import CitationValidator

logger = logging.getLogger(__name__)


class ScholarBridgeTools:
    """Encapsulates LangChain tool bindings to the existing Phase 1-3 infrastructure."""

    def __init__(
        self,
        engine: Optional[ScholarBridgeEngine] = None,
        retriever: Optional[HybridEvidenceRetriever] = None,
        validator: Optional[CitationValidator] = None
    ):
        self.engine = engine or ScholarBridgeEngine()
        self.retriever = retriever or get_evidence_retriever()
        self.validator = validator or CitationValidator()

    def get_recommend_tool(self):
        """Returns LangChain tool for deterministic scholarship recommendation."""
        engine_instance = self.engine

        @tool("recommend_scholarships")
        def recommend_scholarships(profile_dict: Dict[str, Any]) -> Dict[str, Any]:
            """
            Evaluates a student profile against all statutory scholarship eligibility rules
            and computes multi-factor recommendation rankings deterministically.
            The LLM is NOT consulted for eligibility; the deterministic engine is the sole authority.
            """
            profile = StudentProfile(**profile_dict)
            output: EngineOutput = engine_instance.recommend(profile)

            return {
                "total_evaluated": output.total_scholarships_evaluated,
                "eligible_count": output.eligible_count,
                "conditional_count": output.conditional_count,
                "ineligible_count": output.ineligible_count,
                "ranked_recommendations": [
                    {
                        "scheme_code": r.scholarship.scheme_code,
                        "scholarship_name": r.scholarship.scholarship_name,
                        "rank": r.rank,
                        "final_score": r.final_score,
                        "eligibility_status": r.match_result.status.value.lower(),
                        "benefit_score": r.benefit_score,
                        "confidence_score": r.confidence_score,
                        "urgency_score": r.urgency_score,
                        "specificity_score": r.specificity_score,
                        "timeline_info": r.timeline_info,
                    }
                    for r in output.ranked_recommendations
                ]
            }

        return recommend_scholarships

    def get_search_evidence_tool(self):
        """Returns LangChain tool for Hybrid RAG evidence retrieval."""
        retriever_instance = self.retriever

        @tool("search_scholarship_evidence")
        def search_scholarship_evidence(
            query: str,
            scholarship_id: Optional[str] = None,
            top_k: int = 5
        ) -> List[Dict[str, Any]]:
            """
            Retrieves grounded official scholarship documentation chunks using Hybrid RAG
            (Qdrant dense vector search + BM25 keyword matching + RRF fusion + CrossFeature reranker).
            """
            items: List[EvidenceItem] = retriever_instance.retrieve(
                query=query,
                scheme_id=scholarship_id,
                top_k=top_k
            )

            output_items = []
            for item in items:
                cid = getattr(item, "citation_id", None) or getattr(getattr(item, "citation", None), "citation_id", "EV-UNKNOWN")
                chash = getattr(item, "chunk_hash", "") or getattr(getattr(item, "citation", None), "chunk_hash", "")
                ayear = getattr(item, "academic_year", "") or getattr(getattr(item, "citation", None), "academic_year", "2024-25")
                src = getattr(item, "source_authority", "") or getattr(getattr(item, "citation", None), "document_title", "Official Portal")
                output_items.append({
                    "citation_id": cid,
                    "scheme_id": item.scheme_id,
                    "section": item.section,
                    "text": item.text,
                    "retrieval_score": item.retrieval_score,
                    "source_authority": src,
                    "chunk_hash": chash,
                    "academic_year": ayear
                })
            return output_items

        return search_scholarship_evidence

    def get_validate_citations_tool(self):
        """Returns LangChain tool for citation and provenance validation."""
        validator_instance = self.validator

        @tool("validate_citations")
        def validate_citations(
            response_text: str,
            citations: List[str],
            retrieved_evidence_dicts: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """
            Verifies that all citations referenced in generated text exist in the retrieved evidence,
            have intact cryptographic provenance hashes, and are not fabricated.
            """
            # Pass dicts directly — CitationValidator.validate_response handles dict items
            # by checking for citation_id key, so no need to construct EvidenceItem objects
            result = validator_instance.validate_response(
                response_text=response_text,
                citations=citations,
                retrieved_evidence=retrieved_evidence_dicts
            )

            return {
                "is_valid": result.is_valid,
                "valid_citations": result.valid_citations,
                "invalid_citations": result.invalid_citations,
                "hallucinated_citations": result.hallucinated_citations,
                "provenance_verified": result.provenance_verified,
                "validation_error": result.validation_error
            }

        return validate_citations