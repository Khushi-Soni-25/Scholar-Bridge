"""
ScholarBridge LangChain Orchestration Chains
=============================================
Defines controlled, deterministic LangChain Runnable pipelines:
  1. GroundedExplanationChain: Synthesizes grounded explanations from deterministic results + RAG evidence.
  2. ScholarshipComparisonChain: Compares multiple scholarships based on deterministic rules and evidence.

Preserves the Authority Hierarchy:
  - Deterministic Engine decides
  - Hybrid RAG proves
  - LLM explains
"""

import logging
import json
from typing import Dict, Any, List, Optional

from scholarbridge_engine.models import (
    StudentProfile,
    Scholarship,
    MatchResult,
    EngineOutput,
)
from scholarbridge_engine.llm.schemas import (
    ScholarshipResponse,
    ScholarshipRecommendationItem,
    DocumentItem,
    LLMGeneratedContent,
)
from scholarbridge_engine.evidence.models import EvidenceItem, EvidencePack
from scholarbridge_engine.evidence.citations import CitationValidator
from scholarbridge_engine.llm.provider import LLMProvider, get_llm_provider
from scholarbridge_engine.llm.exceptions import LLMProviderError, MalformedLLMResponseError
from scholarbridge_engine.llm.service import build_authoritative_next_steps

from .prompts import explanation_prompt_template, comparison_prompt_template
from .schemas import (
    ScholarshipExplanationSchema,
    ScholarshipComparisonResponse,
    ComparisonItemSchema,
    OrchestrationTrace,
)
from .callbacks import ScholarBridgeCallbackHandler

logger = logging.getLogger(__name__)


class GroundedExplanationChain:
    """
    LangChain orchestration chain for synthesizing grounded eligibility explanations.
    Receives deterministic evaluation + RAG evidence + applicant profile.
    Calls LLM via existing Phase 3 provider and validates citations against evidence.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        validator: Optional[CitationValidator] = None
    ):
        self.provider = provider or get_llm_provider()
        self.validator = validator or CitationValidator()

    def run(
        self,
        profile: StudentProfile,
        scholarship: Scholarship,
        match_result: MatchResult,
        evidence: List[EvidenceItem],
        engine_output: Optional[EngineOutput] = None,
        user_query: Optional[str] = None,
        callback_handler: Optional[ScholarBridgeCallbackHandler] = None
    ) -> ScholarshipResponse:
        """Executes the explanation chain with strict grounding and citation validation."""
        trace = callback_handler.trace if callback_handler else OrchestrationTrace()
        trace.llm_provider = self.provider.get_provider_name()
        trace.llm_model = self.provider.get_model_name()
        trace.evidence_count = len(evidence)

        # 1. Format Evaluated Rules Text
        rules_lines = []
        for r in match_result.passed_rules:
            rules_lines.append(f"  [PASSED] {r.rule.field}: Satisfied requirement ({r.rule.operator} {r.rule.value})")
        for r in match_result.conditional_rules:
            rules_lines.append(f"  [CONDITIONAL] {r.rule.field}: Additional proof needed ({r.rule.rule_description or r.rule.rule_id})")
        for r in match_result.failed_rules:
            rules_lines.append(f"  [FAILED] {r.rule.field}: Did not meet criteria ({r.reason or r.rule.rule_description})")
        evaluated_rules_text = "\n".join(rules_lines) if rules_lines else "  All standard criteria evaluated."

        # 2. Format Evidence Text
        evidence_lines = []
        for item in evidence:
            cid = getattr(item, "citation_id", None) or getattr(getattr(item, "citation", None), "citation_id", "EV-UNKNOWN")
            src = getattr(item, "source_authority", None) or getattr(getattr(item, "citation", None), "document_title", "Official Portal")
            evidence_lines.append(
                f"[{cid}] (Section: {item.section}, Source: {src}):\n{item.text.strip()}"
            )
        evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No official evidence excerpts retrieved."

        # 3. Format Prompt Inputs
        prompt_inputs = {
            "education_level": profile.education_level or "Not specified",
            "stream": profile.stream or "Not specified",
            "course_name": profile.course_name or "Not specified",
            "social_category": profile.social_category or "Not specified",
            "gender": profile.gender or "Not specified",
            "annual_family_income": f"{profile.annual_family_income:,.2f}" if profile.annual_family_income is not None else "Not specified",
            "domicile_state": profile.domicile_state or "Not specified",
            "study_state": profile.study_state or "Not specified",
            "marks_percentile": profile.marks_percentile if profile.marks_percentile is not None else "Not specified",
            "is_pwd": "Yes" if profile.is_pwd else "No",
            "disability_percentage": profile.disability_percentage if profile.disability_percentage is not None else 0.0,
            "hostel_status": profile.hostel_status or "Day Scholar",
            "scholarship_name": scholarship.scholarship_name,
            "scholarship_id": scholarship.scheme_code,
            "authoritative_status": match_result.status.value.upper(),
            "match_confidence": f"{scholarship.confidence_score or 0.85}",
            "composite_score": "85.0",
            "evaluated_rules_text": evaluated_rules_text,
            "evidence_text": evidence_text,
            "user_query": user_query or f"Why am I {match_result.status.value} for {scholarship.scholarship_name}?"
        }

        # 4. Generate Prompt through LangChain PromptTemplate
        formatted_messages = explanation_prompt_template.format_messages(**prompt_inputs)
        system_prompt = formatted_messages[0].content
        user_prompt = formatted_messages[1].content

        # 5. Call Existing Phase 3 LLM Provider
        if callback_handler:
            callback_handler.on_llm_start({"name": self.provider.get_provider_name()}, [user_prompt])

        try:
            raw_llm_output = self.provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_schema=ScholarshipExplanationSchema,
                temperature=0.0
            )
            if callback_handler:
                callback_handler.on_llm_end(raw_llm_output)

        except Exception as e:
            if callback_handler:
                callback_handler.on_llm_error(e)
            logger.error(f"LLM generation failed in GroundedExplanationChain: {e}")
            raise LLMProviderError(f"Explanation chain LLM call failed: {e}") from e

        # 6. Parse and Validate LLM Structured Output
        try:
            if isinstance(raw_llm_output, dict):
                explanation_data = ScholarshipExplanationSchema(**raw_llm_output)
            else:
                explanation_data = ScholarshipExplanationSchema.model_validate_json(raw_llm_output)
        except Exception as e:
            logger.error(f"Failed to parse LLM structured output: {e}")
            raise MalformedLLMResponseError(f"LLM output violated explanation schema: {e}") from e

        # 7. Perform Citation Validation
        citations = list(explanation_data.citations)
        validation_result = self.validator.validate_response(
            response_text=explanation_data.answer + " " + " ".join(explanation_data.explanation_points),
            citations=citations,
            retrieved_evidence=evidence
        )

        trace.citations_referenced = len(citations)
        trace.citations_valid = len(validation_result.valid_citations)
        trace.validation_status = "PASSED" if validation_result.is_valid else "FAILED"

        # 8. Build Authoritative Documents & Next Steps (100% Service Owned)
        auth_documents: List[DocumentItem] = [
            DocumentItem(
                document_name=d.document_name,
                document_category=d.document_category or "General",
                is_mandatory=(str(d.mandatory).strip().capitalize() == "Yes"),
                purpose=d.condition_description or d.notes or "Official verification proof"
            )
            for d in match_result.documents
        ]
        auth_next_steps = build_authoritative_next_steps(scholarship, match_result, auth_documents)

        # Authoritative Recommendations
        auth_recs: List[ScholarshipRecommendationItem] = []
        if engine_output:
            auth_recs = [
                ScholarshipRecommendationItem(
                    scholarship_id=rec.scholarship.scheme_code,
                    scholarship_name=rec.scholarship.scholarship_name,
                    rank=rec.rank,
                    score=rec.final_score,
                    eligibility_status=rec.match_result.status.value.lower(),
                    reason=rec.match_result.explanation or f"Ranked #{rec.rank} based on composite heuristic score."
                )
                for rec in engine_output.ranked_recommendations[:5]
            ]

        # 9. Build Final Structured ScholarshipResponse
        grounding_meta = {
            "engine_status": match_result.status.value.upper(),
            "provider": self.provider.get_provider_name(),
            "model": self.provider.get_model_name(),
            "evidence_count": len(evidence),
            "citation_ids": validation_result.valid_citations,
            "validation_status": "PASSED" if validation_result.is_valid else "FAILED",
            "authority_model": "Engine Decides. RAG Proves. LLM Explains."
        }

        return ScholarshipResponse(
            scholarship_id=scholarship.scheme_code,
            scholarship_name=scholarship.scholarship_name,
            eligibility_status=match_result.status.value.lower(),
            recommendations=auth_recs,
            answer=explanation_data.answer,
            eligibility_explanation=explanation_data.explanation_points or [match_result.explanation],
            documents=auth_documents,
            next_steps=auth_next_steps,
            warnings=explanation_data.warnings,
            citations=validation_result.valid_citations,
            grounding_metadata=grounding_meta
        )


class ScholarshipComparisonChain:
    """
    LangChain orchestration chain for comparing multiple scholarships deterministically.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        validator: Optional[CitationValidator] = None
    ):
        self.provider = provider or get_llm_provider()
        self.validator = validator or CitationValidator()

    def run(
        self,
        profile: StudentProfile,
        evaluated_scholarships: List[Dict[str, Any]],
        evidence_by_scheme: Dict[str, List[EvidenceItem]],
        callback_handler: Optional[ScholarBridgeCallbackHandler] = None
    ) -> ScholarshipComparisonResponse:
        """Runs multi-scholarship comparative analysis."""
        trace = callback_handler.trace if callback_handler else OrchestrationTrace()
        trace.chain_name = "ScholarshipComparisonChain"

        # Format applicant context
        context_str = (
            f"- Stream: {profile.stream} ({profile.course_name})\n"
            f"- Category: {profile.social_category}, Gender: {profile.gender}\n"
            f"- Income: Rs. {profile.annual_family_income:,.2f}\n"
            f"- Domicile: {profile.domicile_state}, Marks: {profile.marks_percentile}%"
        )

        # Format scholarships
        sch_lines = []
        for item in evaluated_scholarships:
            sch = item["scholarship"]
            mr = item["match_result"]
            sch_lines.append(
                f"[{sch.scheme_code}] {sch.scholarship_name}\n"
                f"  - Status: {mr.status.value.upper()}\n"
                f"  - Benefit: Rs. {sch.max_benefit_amount_inr or 0.0:,.2f} ({sch.benefit_type or 'Financial Aid'})\n"
                f"  - Deadline: {sch.application_end_date or 'Ongoing'}\n"
                f"  - Portal: {sch.application_portal or 'Official Portal'}"
            )
        scholarships_data = "\n\n".join(sch_lines)

        # Format all evidence
        ev_lines = []
        all_evidence: List[EvidenceItem] = []
        for scheme_id, items in evidence_by_scheme.items():
            all_evidence.extend(items)
            for it in items:
                cid = getattr(it, "citation_id", None) or getattr(getattr(it, "citation", None), "citation_id", "EV-UNKNOWN")
                ev_lines.append(f"[{cid}] ({scheme_id} - {it.section}): {it.text.strip()}")
        evidence_text = "\n\n".join(ev_lines) if ev_lines else "No evidence retrieved."

        # Format prompt
        formatted_messages = comparison_prompt_template.format_messages(
            applicant_context=context_str,
            scholarships_data=scholarships_data,
            evidence_text=evidence_text
        )
        system_prompt = formatted_messages[0].content
        user_prompt = formatted_messages[1].content

        if callback_handler:
            callback_handler.on_llm_start({"name": self.provider.get_provider_name()}, [user_prompt])

        raw_llm_output = self.provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0
        )

        if callback_handler:
            callback_handler.on_llm_end(raw_llm_output)

        # Synthesize structured response and enforce citation validation
        if isinstance(raw_llm_output, dict):
            comparative_analysis = raw_llm_output.get(
                "comparative_analysis",
                raw_llm_output.get("answer", "Comparative evaluation completed.")
            )
            recommendation_verdict = raw_llm_output.get(
                "recommendation_verdict",
                raw_llm_output.get("reasoning_summary", "Review the eligible schemes first.")
            )
            claimed_citations = list(raw_llm_output.get("citations", []))

            # Perform strict citation validation against all retrieved evidence
            combined_text = f"{comparative_analysis}\n{recommendation_verdict}"
            validation_result = self.validator.validate_response(
                response_text=combined_text,
                citations=claimed_citations,
                retrieved_evidence=all_evidence
            )

            trace.evidence_count = len(all_evidence)
            trace.citations_referenced = len(validation_result.valid_citations) + len(validation_result.invalid_citations)
            trace.citations_valid = len(validation_result.valid_citations)
            trace.validation_status = "PASSED" if validation_result.is_valid else "FAILED"
            if not validation_result.is_valid and validation_result.validation_error:
                trace.errors.append(validation_result.validation_error)

            compared_items = []
            for item in evaluated_scholarships:
                sch = item["scholarship"]
                mr = item["match_result"]
                scheme_ev_items = evidence_by_scheme.get(sch.scheme_code, [])
                scheme_cids = [
                    getattr(it, "citation_id", None) or getattr(getattr(it, "citation", None), "citation_id", "EV-UNKNOWN")
                    for it in scheme_ev_items
                ]
                item_cits = [cid for cid in validation_result.valid_citations if cid in scheme_cids]
                if not item_cits and scheme_cids:
                    item_cits = scheme_cids[:2]

                compared_items.append(
                    ComparisonItemSchema(
                        scholarship_id=sch.scheme_code,
                        scholarship_name=sch.scholarship_name,
                        eligibility_status=mr.status.value.lower(),
                        benefit_summary=f"Up to Rs. {sch.max_benefit_amount_inr or 0.0:,.2f} ({sch.benefit_type or 'Financial Aid'})",
                        key_requirements=[r.rule.field for r in mr.failed_rules + mr.conditional_rules],
                        citations=item_cits[:2]
                    )
                )

            return ScholarshipComparisonResponse(
                student_summary=context_str,
                compared_scholarships=compared_items,
                comparative_analysis=comparative_analysis,
                recommendation_verdict=recommendation_verdict,
                citations=validation_result.valid_citations,
                trace=trace.to_dict()
            )
        else:
            raise MalformedLLMResponseError("Comparison output was not a valid dictionary.")