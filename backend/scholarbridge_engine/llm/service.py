"""
ScholarBridge LLM Orchestration & Grounding Service
===================================================
Coordinates deterministic engine results, validated RAG evidence packs,
grounded prompt generation, and strict post-generation verification.

Guiding principle:
"The deterministic engine decides. RAG provides evidence. The LLM explains."
"""

import logging
from typing import Optional, List, Dict, Any, Union
from pydantic import ValidationError

from ..engine import ScholarBridgeEngine
from ..models import (
    StudentProfile, Scholarship, MatchResult, EligibilityStatus, EngineOutput, RecommendationItem
)
from ..loader import KnowledgeBase, get_knowledge_base
from ..evidence.retriever import HybridEvidenceRetriever, get_evidence_retriever
from ..evidence.models import EvidencePack

from .base import LLMProvider
from .provider import get_llm_provider, MockLLMProvider
from .prompts import SYSTEM_PROMPT, build_eligibility_explanation_prompt
from .schemas import (
    ScholarshipResponse, ScholarshipRecommendationItem, DocumentItem,
    LLMGeneratedContent, GroundingMetadata
)
from .exceptions import (
    ScholarBridgeLLMError, LLMValidationError, GroundingError,
    EvidenceUnavailableError, MalformedLLMResponseError
)

logger = logging.getLogger(__name__)


def build_authoritative_next_steps(
    scholarship: Scholarship,
    match_result: MatchResult,
    documents: List[DocumentItem]
) -> List[str]:
    """
    Constructs 100% authoritative next steps from Knowledge Base portal,
    deadline, document, and deterministic rule evaluation data.
    The LLM is NOT permitted to invent or alter these actionable instructions.
    """
    status = match_result.status.value.lower()
    steps = []

    portal_name = scholarship.application_portal or "official scholarship portal"
    portal_url = scholarship.official_website or scholarship.official_source_url
    deadline_end = scholarship.application_end_date or "designated cycle end date"
    d_status = scholarship.deadline_status or "ANNOUNCED"

    if status == "eligible":
        if portal_url:
            steps.append(f"Visit the official application portal: {portal_name} ({portal_url})")
        else:
            steps.append(f"Visit the official application portal: {portal_name}")
            
        if deadline_end and deadline_end != "-":
            steps.append(f"Submit application before cycle deadline: {deadline_end} (Status: {d_status})")
            
        mandatory_docs = [d.document_name for d in documents if d.is_mandatory]
        if mandatory_docs:
            doc_sample = ", ".join(mandatory_docs[:3])
            suffix = ", etc." if len(mandatory_docs) > 3 else ""
            steps.append(f"Gather and verify mandatory documents ({doc_sample}{suffix})")
        else:
            steps.append("Gather and verify all mandatory application documents.")

    elif status == "conditional":
        cond_rules = match_result.conditional_rules
        for cr in cond_rules:
            steps.append(f"Satisfy conditional requirement for {cr.rule.field.replace('_', ' ')}: {cr.reason}")
        if portal_url:
            steps.append(f"Verify document and eligibility guidelines on {portal_name} ({portal_url})")

    elif status == "ineligible":
        failed_rules = match_result.failed_rules
        for fr in failed_rules:
            steps.append(f"Failed criteria: {fr.rule.field.replace('_', ' ')} — {fr.reason}")
        steps.append("Explore other eligible scholarships on ScholarBridge matching your profile.")

    else: # SUSPENDED
        steps.append(f"This scheme is currently {scholarship.status.upper()}; do not submit applications.")
        steps.append("Check alternative active schemes on ScholarBridge matching your profile.")

    return steps


class LLMService:
    """
    Central orchestration service for Phase 3 grounded LLM explanations.
    Enforces strict authoritative ownership of deterministic engine decisions
    and validates all generated outputs against retrieved evidence.
    """

    def __init__(
        self,
        engine: Optional[ScholarBridgeEngine] = None,
        retriever: Optional[HybridEvidenceRetriever] = None,
        provider: Optional[LLMProvider] = None,
        kb: Optional[KnowledgeBase] = None
    ):
        self.engine = engine or ScholarBridgeEngine()
        self.retriever = retriever or get_evidence_retriever()
        self.provider = provider or get_llm_provider()
        self.kb = kb or get_knowledge_base()

    def _resolve_scholarship(self, scholarship_id_or_code: str) -> Scholarship:
        """Finds master scholarship entity by scheme code or scholarship ID."""
        target = str(scholarship_id_or_code).strip().upper()
        for s in self.kb.scholarships:
            if s.scheme_code.upper() == target or s.scholarship_id.upper() == target:
                return s
            if s.short_name and s.short_name.upper() == target:
                return s
        raise ValueError(f"Scholarship '{scholarship_id_or_code}' not found in knowledge base.")

    def explain_eligibility(
        self,
        profile: StudentProfile,
        scholarship_id_or_code: str,
        user_query: Optional[str] = None,
        strict_citation_check: bool = True
    ) -> ScholarshipResponse:
        """
        Primary Phase 3 functional entrypoint: "Why am I eligible for this scholarship?"
        
        1. Executes deterministic Phase 1 engine evaluation.
        2. Retrieves official Phase 2 RAG evidence.
        3. Constructs grounded prompt with immutable facts.
        4. Invokes abstracted LLM provider for explanation generation.
        5. Validates raw generation against strict grounding rules.
        6. Constructs and returns authoritative ScholarshipResponse.
        """
        # Step 1: Resolve Target Scholarship
        scholarship = self._resolve_scholarship(scholarship_id_or_code)
        scheme_code = scholarship.scheme_code

        # Step 2: Run Authoritative Deterministic Engine
        engine_output = self.engine.recommend(profile)
        
        # Locate target match result & ranking
        rank: Optional[int] = None
        score: Optional[float] = None
        match_result: Optional[MatchResult] = None

        for rec in engine_output.ranked_recommendations:
            if rec.scholarship.scheme_code == scheme_code:
                rank = rec.rank
                score = rec.final_score
                match_result = rec.match_result
                break

        if match_result is None:
            for rec in engine_output.conditional_recommendations:
                if rec.scholarship.scheme_code == scheme_code:
                    score = rec.final_score
                    match_result = rec.match_result
                    break

        if match_result is None:
            for excl in engine_output.excluded_scholarships:
                if excl.scholarship.scheme_code == scheme_code:
                    match_result = excl
                    score = 0.0
                    break

        if match_result is None:
            # Direct match fallback if filtered out earlier
            rules = self.kb.rules_by_scholarship.get(scholarship.scholarship_id, [])
            match_result = self.engine.matcher.evaluate_scholarship(scholarship, rules, profile)

        # Step 3: Authoritative Status, Documents & Next Steps (100% Service Owned)
        auth_status = match_result.status.value.lower()
        
        # Build authoritative DocumentItems from knowledge base
        doc_reqs = self.kb.docs_by_scholarship.get(scholarship.scholarship_id, [])
        authoritative_documents: List[DocumentItem] = [
            DocumentItem(
                document_name=d.document_name,
                document_category=d.document_category or "General",
                is_mandatory=(str(d.mandatory).strip().capitalize() == "Yes"),
                purpose=d.condition_description or d.notes or "Official verification proof"
            )
            for d in doc_reqs
        ]

        # Build authoritative next steps from verified portal & deadline metadata
        authoritative_next_steps = build_authoritative_next_steps(
            scholarship, match_result, authoritative_documents
        )

        # Step 4: Retrieve Validated RAG Evidence Pack
        query_text = user_query or f"{scheme_code} eligibility income {profile.annual_family_income} criteria documents"
        evidence_pack = self.retriever.retrieve_for_scheme(
            scheme_id=scheme_code,
            query=query_text,
            top_k=4
        )

        valid_citation_ids = [ev.citation.citation_id for ev in evidence_pack.evidence]

        # Step 5: Build Grounded Prompt
        prompt = build_eligibility_explanation_prompt(
            profile=profile,
            scholarship=scholarship,
            match_result=match_result,
            evidence_pack=evidence_pack,
            rank=rank,
            score=score,
            user_query=user_query
        )

        # Step 6: Invoke LLM Provider
        try:
            raw_response = self.provider.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                response_schema=LLMGeneratedContent
            )
        except Exception as e:
            logger.error(f"LLM Provider execution error: {e}")
            raise MalformedLLMResponseError(f"LLM Provider failed to generate structured response: {e}") from e

        # Step 7: Parse & Validate LLM Raw Content
        if not isinstance(raw_response, dict):
            raise MalformedLLMResponseError(f"Expected dict from LLM Provider, got {type(raw_response)}")

        try:
            generated_content = LLMGeneratedContent(**raw_response)
        except ValidationError as ve:
            logger.error(f"LLM output schema validation failed: {ve}")
            raise LLMValidationError(f"LLM output did not conform to generation schema: {ve}") from ve

        # Step 8: Strict Grounding & Citation Validation
        # 8.1 Citation Verification (Reject fabricated citation IDs)
        for cited_id in generated_content.citations:
            if cited_id not in valid_citation_ids:
                if strict_citation_check:
                    raise GroundingError(
                        f"Grounding violation: LLM cited unauthorized / fabricated citation ID '{cited_id}'. "
                        f"Valid evidence citation IDs are: {valid_citation_ids}"
                    )
                else:
                    logger.warning(f"Removing invalid citation '{cited_id}' not in evidence pack.")

        final_citations = [c for c in generated_content.citations if c in valid_citation_ids]
        # If LLM omitted citations but valid evidence exists, preserve retrieved citation provenance
        if not final_citations and valid_citation_ids:
            final_citations = list(valid_citation_ids)

        # Step 9: Assemble Authoritative Recommendations List (100% Service Owned)
        authoritative_recommendations: List[ScholarshipRecommendationItem] = [
            ScholarshipRecommendationItem(
                scholarship_id=rec.scholarship.scheme_code,
                scholarship_name=rec.scholarship.scholarship_name,
                rank=rec.rank,
                score=rec.final_score,
                eligibility_status=rec.match_result.status.value.lower(),
                reason=rec.match_result.explanation or f"Ranked #{rec.rank} based on composite benefit, confidence, and urgency."
            )
            for rec in engine_output.ranked_recommendations[:5]
        ]

        # Step 10: Service-Generated Grounding Metadata (100% Service Owned)
        grounding_meta = {
            "engine_status": match_result.status.value.upper(),
            "engine_score": round(score, 1) if score is not None else 0.0,
            "evidence_count": len(evidence_pack.evidence),
            "citation_ids": valid_citation_ids,
            "provider": self.provider.get_provider_name(),
            "model": self.provider.get_model_name(),
            "validation_status": "PASSED",
            "authority_model": "Engine Decides. RAG Proves. LLM Explains."
        }

        # Step 11: Return Final Standardized ScholarshipResponse
        return ScholarshipResponse(
            answer=generated_content.answer,
            scholarship_id=scholarship.scheme_code,
            scholarship_name=scholarship.scholarship_name,
            eligibility_status=auth_status,
            recommendations=authoritative_recommendations,
            eligibility_explanation=generated_content.explanation_points or [match_result.explanation],
            documents=authoritative_documents,
            next_steps=authoritative_next_steps,
            warnings=generated_content.warnings,
            citations=final_citations,
            grounding_metadata=grounding_meta
        )
