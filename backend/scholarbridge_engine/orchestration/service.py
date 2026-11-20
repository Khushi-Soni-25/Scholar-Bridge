"""
ScholarBridge Central LangChain Orchestration Service
======================================================
Unified orchestration facade coordinating:
  1. Deterministic Recommendation Engine (sole authority for eligibility and ranking)
  2. Hybrid RAG (Qdrant + BM25 + RRF + CrossFeature reranking)
  3. Production LLM Provider (Gemini / OpenAI / Ollama / Mock)
  4. Citation & Provenance Validation
  5. Lightweight observability tracing

Hard Architectural Invariants Enforced:
  - Deterministic Engine decides
  - Hybrid RAG proves
  - LLM explains
  - Observability captures all latencies and stages
"""

import logging
from typing import Dict, Any, List, Optional

from scholarbridge_engine.models import (
    StudentProfile,
    EngineOutput,
    MatchResult,
    Scholarship,
    EligibilityStatus,
)
from scholarbridge_engine.llm.schemas import ScholarshipResponse
from scholarbridge_engine.engine import ScholarBridgeEngine
from scholarbridge_engine.loader import KnowledgeBase, get_knowledge_base
from scholarbridge_engine.evidence.retriever import HybridEvidenceRetriever, get_evidence_retriever
from scholarbridge_engine.evidence.models import EvidenceItem
from scholarbridge_engine.evidence.citations import CitationValidator
from scholarbridge_engine.llm.provider import LLMProvider, get_llm_provider

from .tools import ScholarBridgeTools
from .chains import GroundedExplanationChain, ScholarshipComparisonChain
from .callbacks import ScholarBridgeCallbackHandler
from .schemas import (
    OrchestrationTrace,
    ScholarshipComparisonResponse,
)

logger = logging.getLogger(__name__)


class LangChainOrchestrationService:
    """
    Central orchestration service coordinating ScholarBridge's deterministic,
    retrieval, and LLM explanation components through modern LangChain interfaces.
    """

    def __init__(
        self,
        engine: Optional[ScholarBridgeEngine] = None,
        retriever: Optional[HybridEvidenceRetriever] = None,
        provider: Optional[LLMProvider] = None,
        validator: Optional[CitationValidator] = None,
        kb: Optional[KnowledgeBase] = None
    ):
        self.engine = engine or ScholarBridgeEngine()
        self.kb = kb or getattr(self.engine, "kb", None) or get_knowledge_base()
        self.retriever = retriever or get_evidence_retriever()
        self.provider = provider or get_llm_provider()
        self.validator = validator or CitationValidator()

        # Tools and Chains
        self.tools = ScholarBridgeTools(
            engine=self.engine,
            retriever=self.retriever,
            validator=self.validator
        )
        self.explanation_chain = GroundedExplanationChain(
            provider=self.provider,
            validator=self.validator
        )
        self.comparison_chain = ScholarshipComparisonChain(
            provider=self.provider,
            validator=self.validator
        )

    def _resolve_scholarship(self, identifier: str) -> Scholarship:
        """Resolves scholarship by scholarship_id, scheme_code, or short_name."""
        clean_id = identifier.strip().upper()
        if clean_id in self.kb.scholarships_by_id:
            return self.kb.scholarships_by_id[clean_id]

        for s in self.kb.scholarships:
            if s.scheme_code.upper() == clean_id or s.scholarship_name.upper() == clean_id or s.short_name.upper() == clean_id:
                return s

        for s in self.kb.scholarships:
            if clean_id in s.scheme_code.upper() or clean_id in s.scholarship_name.upper():
                return s

        raise ValueError(f"Scholarship '{identifier}' not found in Knowledge Base.")

    def recommend(
        self,
        profile: StudentProfile,
        trace: Optional[OrchestrationTrace] = None
    ) -> EngineOutput:
        """
        Executes deterministic scholarship recommendation via LangChain tool.
        Preserves 100% deterministic authority.
        """
        cb = ScholarBridgeCallbackHandler(trace=trace)
        cb.on_tool_start({"name": "recommend_scholarships"}, str(profile))

        output: EngineOutput = self.engine.recommend(profile)

        cb.on_tool_end(output)
        cb.finalize()
        return output

    def search_evidence(
        self,
        query: str,
        scholarship_id: Optional[str] = None,
        top_k: int = 5,
        trace: Optional[OrchestrationTrace] = None
    ) -> List[EvidenceItem]:
        """
        Executes Hybrid RAG retrieval via LangChain tool.
        """
        cb = ScholarBridgeCallbackHandler(trace=trace)
        cb.on_tool_start({"name": "search_scholarship_evidence"}, query)

        items = self.retriever.retrieve(
            query=query,
            scheme_id=scholarship_id,
            top_k=top_k
        )

        cb.on_tool_end(items)
        cb.finalize()
        return items

    def explain(
        self,
        profile: StudentProfile,
        scholarship_id_or_code: str,
        user_query: Optional[str] = None,
        trace: Optional[OrchestrationTrace] = None
    ) -> ScholarshipResponse:
        """
        Full 3-phase grounded explanation orchestrated via LangChain:
          Phase 1: Deterministic evaluation of the specific scholarship
          Phase 2: Hybrid RAG retrieval for relevant scheme rules/benefits
          Phase 3: LLM generation + Citation validation
        """
        cb = ScholarBridgeCallbackHandler(trace=trace)

        # 1. Resolve Target Scholarship
        scholarship = self._resolve_scholarship(scholarship_id_or_code)
        scheme_code = scholarship.scheme_code

        # 2. Run Deterministic Engine
        cb.on_tool_start({"name": "deterministic_engine_evaluation"}, scheme_code)
        engine_output = self.engine.recommend(profile)

        match_result: Optional[MatchResult] = None
        for rec in engine_output.ranked_recommendations + engine_output.conditional_recommendations:
            if rec.scholarship.scheme_code == scheme_code:
                match_result = rec.match_result
                break
        if match_result is None:
            for excl in engine_output.excluded_scholarships:
                if excl.scholarship.scheme_code == scheme_code:
                    match_result = excl
                    break
        if match_result is None:
            rules = self.kb.rules_by_scholarship.get(scholarship.scholarship_id, [])
            match_result = self.engine.matcher.evaluate_scholarship(scholarship, rules, profile)
            match_result.documents = self.kb.docs_by_scholarship.get(scholarship.scholarship_id, [])
            match_result.benefits = self.kb.benefits_by_scholarship.get(scholarship.scholarship_id, [])

        cb.on_tool_end(match_result)

        # 3. Hybrid RAG Evidence Retrieval
        cb.on_tool_start({"name": "hybrid_rag_retrieval"}, f"eligibility {scheme_code}")
        evidence_query = user_query or f"eligibility criteria benefits documents {scholarship.scholarship_name}"
        evidence = self.retriever.retrieve(
            query=evidence_query,
            scheme_id=scholarship.scheme_code,
            top_k=4
        )
        cb.on_tool_end(evidence)

        # 4. Grounded Explanation Chain
        response = self.explanation_chain.run(
            profile=profile,
            scholarship=scholarship,
            match_result=match_result,
            evidence=evidence,
            engine_output=engine_output,
            user_query=user_query,
            callback_handler=cb
        )

        cb.finalize()
        return response

    def compare(
        self,
        profile: StudentProfile,
        scholarship_ids: List[str],
        trace: Optional[OrchestrationTrace] = None
    ) -> ScholarshipComparisonResponse:
        """
        Compares multiple scholarships deterministically with evidence-backed analysis.
        """
        cb = ScholarBridgeCallbackHandler(trace=trace)

        evaluated: List[Dict[str, Any]] = []
        evidence_by_scheme: Dict[str, List[EvidenceItem]] = {}

        for sch_id in scholarship_ids:
            sch = self._resolve_scholarship(sch_id)
            rules = self.kb.rules_by_scholarship.get(sch.scholarship_id, [])
            mr = self.engine.matcher.evaluate_scholarship(sch, rules, profile)
            mr.documents = self.kb.docs_by_scholarship.get(sch.scholarship_id, [])
            mr.benefits = self.kb.benefits_by_scholarship.get(sch.scholarship_id, [])
            evaluated.append({"scholarship": sch, "match_result": mr})

            ev = self.retriever.retrieve(
                query=f"benefits requirements {sch.scholarship_name}",
                scheme_id=sch.scheme_code,
                top_k=3
            )
            evidence_by_scheme[sch.scheme_code] = ev

        result = self.comparison_chain.run(
            profile=profile,
            evaluated_scholarships=evaluated,
            evidence_by_scheme=evidence_by_scheme,
            callback_handler=cb
        )

        cb.finalize()
        return result

    def orchestrate_recommendation_and_explanation(
        self,
        profile: StudentProfile,
        top_n: int = 3,
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Composite orchestration pipeline:
          1. Deterministic recommendation across all 100 schemes
          2. RAG evidence retrieval for top N recommendations
          3. Grounded LLM explanations synthesized for top recommendations
          4. Citation validation and telemetry trace collection
        """
        trace = OrchestrationTrace(chain_name="CompositeRecommendationOrchestrator")
        cb = ScholarBridgeCallbackHandler(trace=trace)

        # Step 1: Deterministic Engine Recommendation
        rec_result = self.recommend(profile=profile, trace=trace)

        # Step 2 & 3: Grounded explanations for top candidates
        top_candidates = rec_result.ranked_recommendations[:top_n]
        explained_scholarships: List[Dict[str, Any]] = []

        for cand in top_candidates:
            try:
                single_expl = self.explain(
                    profile=profile,
                    scholarship_id_or_code=cand.scholarship.scheme_code,
                    user_query=user_query,
                    trace=trace
                )
                explained_scholarships.append(single_expl.model_dump())
            except Exception as e:
                logger.warning(f"Failed to generate explanation for {cand.scholarship.scheme_code}: {e}")
                explained_scholarships.append({
                    "scholarship_id": cand.scholarship.scheme_code,
                    "scholarship_name": cand.scholarship.scholarship_name,
                    "eligibility_status": cand.match_result.status.value.lower(),
                    "final_score": cand.final_score,
                    "answer": f"Evaluated as {cand.match_result.status.value.lower()} with score {cand.final_score:.2f}."
                })

        final_trace = cb.finalize()

        return {
            "applicant_summary": {
                "education_level": profile.education_level,
                "stream": profile.stream,
                "social_category": profile.social_category,
                "annual_family_income": profile.annual_family_income
            },
            "recommendation_summary": {
                "total_evaluated": rec_result.total_scholarships_evaluated,
                "eligible_count": rec_result.eligible_count,
                "conditional_count": rec_result.conditional_count,
                "ineligible_count": rec_result.ineligible_count,
            },
            "top_recommendations": explained_scholarships,
            "all_ranked_recommendations": [
                {
                    "scheme_code": r.scholarship.scheme_code,
                    "scholarship_name": r.scholarship.scholarship_name,
                    "rank": r.rank,
                    "final_score": r.final_score,
                    "eligibility_status": r.match_result.status.value.lower(),
                    "benefit_score": r.benefit_score,
                    "urgency_score": r.urgency_score,
                }
                for r in rec_result.ranked_recommendations
            ],
            "trace": final_trace.to_dict()
        }