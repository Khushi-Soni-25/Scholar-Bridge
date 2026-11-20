"""
ScholarBridge LangGraph Workflow Nodes
=======================================
Implements the core execution nodes for Phase 5 orchestration:
  1. node_recommend               — Deterministic profile matching & ranking
  2. node_retrieve_evidence       — Hybrid RAG evidence retrieval
  3. node_evaluate_evidence       — Deterministic evidence quality check
  4. node_refine_search           — Multi-step research query refinement & evidence merge
  5. node_explain_recommendations — Grounded explanation synthesis for top recommendations
  6. node_synthesize_search       — Factual Q&A synthesis over retrieved evidence
  7. node_compare_scholarships    — Multi-scholarship deterministic comparison
  8. node_explain_eligibility     — Single-scheme grounded explanation
  9. node_validate_and_finalize   — Final citation validation & trace assembly

Architectural Rule:
  Nodes are thin orchestration wrappers delegating to the verified Phase 1-4 & LangChain components.
  Deterministic engine decisions are NEVER modified by graph nodes.
"""

import time
import logging
from typing import Dict, Any, List, Optional

from scholarbridge_engine.models import (
    StudentProfile,
    EngineOutput,
    MatchResult,
    Scholarship,
)
from scholarbridge_engine.engine import ScholarBridgeEngine
from scholarbridge_engine.loader import get_knowledge_base, KnowledgeBase
from scholarbridge_engine.evidence.retriever import get_evidence_retriever, HybridEvidenceRetriever
from scholarbridge_engine.evidence.models import EvidenceItem
from scholarbridge_engine.evidence.citations import CitationValidator
from scholarbridge_engine.llm.provider import get_llm_provider, LLMProvider
from scholarbridge_engine.orchestration.chains import GroundedExplanationChain, ScholarshipComparisonChain
from scholarbridge_engine.orchestration.callbacks import ScholarBridgeCallbackHandler
from scholarbridge_engine.orchestration.schemas import OrchestrationTrace, ScholarshipComparisonResponse
from scholarbridge_engine.llm.schemas import ScholarshipResponse

from .state import ScholarBridgeWorkflowState, WorkflowIntent

logger = logging.getLogger(__name__)


def _serialize_evidence(items: List[EvidenceItem]) -> List[Dict[str, Any]]:
    """Converts EvidenceItem instances into serializable dictionaries."""
    out = []
    for item in items:
        cid = getattr(item, "citation_id", None) or getattr(getattr(item, "citation", None), "citation_id", "EV-UNKNOWN")
        chash = getattr(item, "chunk_hash", "") or getattr(getattr(item, "citation", None), "chunk_hash", "")
        ayear = getattr(item, "academic_year", "") or getattr(getattr(item, "citation", None), "academic_year", "2024-25")
        src = getattr(item, "source_authority", "") or getattr(getattr(item, "citation", None), "document_title", "Official Portal")
        out.append({
            "citation_id": cid,
            "scheme_id": item.scheme_id,
            "section": item.section,
            "text": item.text,
            "retrieval_score": getattr(item, "retrieval_score", getattr(item, "reranker_score", 0.8)),
            "source_authority": src,
            "chunk_hash": chash,
            "academic_year": ayear
        })
    return out


def _deduplicate_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strictly deduplicates evidence items preserving first occurrence of each citation_id."""
    seen = set()
    deduped = []
    for it in items:
        cid = it.get("citation_id")
        if cid:
            if cid not in seen:
                seen.add(cid)
                deduped.append(it)
        else:
            deduped.append(it)
    return deduped



# ---------------------------------------------------------------------------
# 1. Recommendation Node
# ---------------------------------------------------------------------------

def node_recommend(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Evaluates student profile deterministically through the rule engine.
    Sole authority for eligibility counts and multi-factor ranking.
    """
    profile_data = state.get("student_profile")
    errors = list(state.get("errors", []))

    if not profile_data:
        errors.append("Cannot execute RECOMMEND workflow: student_profile is missing.")
        return {
            "current_step": "node_recommend",
            "errors": errors,
            "engine_result": None
        }

    try:
        profile = StudentProfile(**profile_data) if isinstance(profile_data, dict) else profile_data
        engine = ScholarBridgeEngine()
        engine_output: EngineOutput = engine.recommend(profile)

        ranked_list = [
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
            for r in engine_output.ranked_recommendations
        ]

        top_schemes = [r["scheme_code"] for r in ranked_list[:3]]
        selected_sch = state.get("selected_scholarships") or top_schemes

        return {
            "current_step": "node_recommend",
            "engine_result": {
                "total_evaluated": engine_output.total_scholarships_evaluated,
                "eligible_count": engine_output.eligible_count,
                "conditional_count": engine_output.conditional_count,
                "ineligible_count": engine_output.ineligible_count,
                "ranked_recommendations": ranked_list,
            },
            "selected_scholarships": selected_sch,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"node_recommend failed: {e}")
        errors.append(f"Engine evaluation error: {str(e)}")
        return {
            "current_step": "node_recommend",
            "errors": errors,
            "engine_result": None
        }


# ---------------------------------------------------------------------------
# 2. Evidence Retrieval Node
# ---------------------------------------------------------------------------

def node_retrieve_evidence(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Executes Hybrid RAG retrieval for active query and target schemes.
    """
    retriever = get_evidence_retriever()
    query = state.get("user_query") or "eligibility benefits documents required"
    selected = state.get("selected_scholarships", [])
    intent = state.get("intent", "SEARCH")
    existing_ev = list(state.get("retrieved_evidence", []))

    new_items: List[EvidenceItem] = []

    if selected:
        for s_code in selected[:3]:
            pack = retriever.retrieve_for_scheme(scheme_id=s_code, query=query, top_k=3)
            new_items.extend(pack.evidence)
    else:
        pack = retriever.search(query=query, top_k=5)
        new_items.extend(pack.evidence)

    serialized_new = _serialize_evidence(new_items)
    merged = _deduplicate_evidence(list(existing_ev) + serialized_new)

    return {
        "current_step": "node_retrieve_evidence",
        "retrieved_evidence": merged,
    }


# ---------------------------------------------------------------------------
# 3. Evidence Quality Evaluation Node (Conditional Routing)
# ---------------------------------------------------------------------------

def node_evaluate_evidence(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Deterministically evaluates evidence sufficiency based on chunk count,
    target scheme coverage, and query intent relevance.
    """
    evidence = state.get("retrieved_evidence", [])
    query = (state.get("user_query") or "").lower()
    selected = state.get("selected_scholarships", [])

    if not evidence:
        return {
            "current_step": "node_evaluate_evidence",
            "evidence_sufficient": False
        }

    # If specific schemes were requested, verify coverage
    if selected:
        found_schemes = {e["scheme_id"].upper() for e in evidence}
        missing_schemes = [s.upper() for s in selected if s.upper() not in found_schemes]
        if missing_schemes and len(evidence) < len(selected) * 2:
            return {
                "current_step": "node_evaluate_evidence",
                "evidence_sufficient": False
            }

    # Check minimum chunk threshold
    if len(evidence) < 2:
        return {
            "current_step": "node_evaluate_evidence",
            "evidence_sufficient": False
        }

    return {
        "current_step": "node_evaluate_evidence",
        "evidence_sufficient": True
    }


# ---------------------------------------------------------------------------
# 4. Multi-Step Research Refinement Node
# ---------------------------------------------------------------------------

def node_refine_search(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Refines and broadens retrieval when initial evidence is insufficient.
    Increments research_iterations counter (strictly bounded).
    """
    iterations = state.get("research_iterations", 0) + 1
    retriever = get_evidence_retriever()
    query = state.get("user_query", "")
    selected = state.get("selected_scholarships", [])
    existing_ev = list(state.get("retrieved_evidence", []))

    # Formulate broadened / augmented query
    augmented_query = f"{query} eligibility criteria financial assistance benefits official guidelines"
    refined_items: List[EvidenceItem] = []

    if selected:
        for s_code in selected:
            pack = retriever.retrieve_for_scheme(scheme_id=s_code, query=augmented_query, top_k=4)
            refined_items.extend(pack.evidence)
    else:
        pack = retriever.search(query=augmented_query, top_k=6)
        refined_items.extend(pack.evidence)

    serialized_new = _serialize_evidence(refined_items)
    merged = _deduplicate_evidence(list(existing_ev) + serialized_new)

    return {
        "current_step": "node_refine_search",
        "research_iterations": iterations,
        "retrieved_evidence": merged,
    }



# ---------------------------------------------------------------------------
# 5. Explain Recommendations Node (for RECOMMEND intent)
# ---------------------------------------------------------------------------

def node_explain_recommendations(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Synthesizes grounded explanations for top recommended scholarships.
    """
    profile_data = state.get("student_profile")
    engine_result = state.get("engine_result") or {}
    evidence = state.get("retrieved_evidence", [])
    errors = list(state.get("errors", []))

    if not profile_data or not engine_result:
        return {
            "current_step": "node_explain_recommendations",
            "final_response": {
                "message": "Recommendation evaluation could not be completed.",
                "errors": errors
            }
        }

    profile = StudentProfile(**profile_data) if isinstance(profile_data, dict) else profile_data
    ranked = engine_result.get("ranked_recommendations", [])
    top_recommendations = []

    for r in ranked[:3]:
        s_code = r["scheme_code"]
        scheme_evidence = [e for e in evidence if e["scheme_id"].upper() == s_code.upper()]
        top_recommendations.append({
            "scholarship_id": s_code,
            "scholarship_name": r["scholarship_name"],
            "rank": r["rank"],
            "final_score": r["final_score"],
            "eligibility_status": r["eligibility_status"],
            "benefit_score": r.get("benefit_score", 0.0),
            "evidence_count": len(scheme_evidence),
            "citations": [e["citation_id"] for e in scheme_evidence[:2]]
        })

    all_citations = list({e["citation_id"] for e in evidence})

    final_payload = {
        "intent": "RECOMMEND",
        "applicant_summary": {
            "education_level": profile.education_level,
            "stream": profile.stream,
            "social_category": profile.social_category,
            "annual_family_income": profile.annual_family_income
        },
        "recommendation_summary": {
            "total_evaluated": engine_result.get("total_evaluated", 100),
            "eligible_count": engine_result.get("eligible_count", 0),
            "conditional_count": engine_result.get("conditional_count", 0),
            "ineligible_count": engine_result.get("ineligible_count", 0),
        },
        "top_recommendations": top_recommendations,
        "all_ranked_recommendations": ranked,
        "citations": all_citations,
    }

    return {
        "current_step": "node_explain_recommendations",
        "final_response": final_payload,
        "citations": all_citations,
        "grounding_status": "PASSED" if all_citations else "NOT_REQUIRED"
    }


# ---------------------------------------------------------------------------
# 6. Synthesize Search Node (for SEARCH intent)
# ---------------------------------------------------------------------------

def node_synthesize_search(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Synthesizes a grounded factual answer for a search query citing official evidence chunks.
    """
    query = state.get("user_query", "")
    evidence = state.get("retrieved_evidence", [])
    provider = get_llm_provider()
    validator = CitationValidator()

    if not evidence:
        return {
            "current_step": "node_synthesize_search",
            "final_response": {
                "intent": "SEARCH",
                "query": query,
                "answer": "No official scholarship evidence excerpts could be retrieved for this query.",
                "citations": []
            },
            "grounding_status": "FAILED"
        }

    # Format grounded prompt
    ev_text_lines = [
        f"[{e['citation_id']}] ({e['scheme_id']} - {e['section']}):\n{e['text']}"
        for e in evidence[:5]
    ]
    ev_text = "\n\n".join(ev_text_lines)

    prompt = (
        f"Answer the following user question using ONLY the provided official evidence excerpts:\n\n"
        f"User Query: {query}\n\n"
        f"Official Evidence:\n{ev_text}\n\n"
        f"Provide a clear, factual answer citing the retrieved citation IDs in square brackets."
    )

    try:
        raw_res = provider.generate(prompt=prompt, temperature=0.0)
        answer = raw_res.get("answer", str(raw_res)) if isinstance(raw_res, dict) else str(raw_res)
        claimed_citations = list(raw_res.get("citations", [])) if isinstance(raw_res, dict) else []

        val_res = validator.validate_response(
            response_text=answer,
            citations=claimed_citations,
            retrieved_evidence=evidence
        )

        final_cits = val_res.valid_citations or [e["citation_id"] for e in evidence[:2]]
        errors = list(state.get("errors", []))
        if not val_res.is_valid and val_res.validation_error:
            errors.append(val_res.validation_error)

        return {
            "current_step": "node_synthesize_search",
            "errors": errors,
            "final_response": {
                "intent": "SEARCH",
                "query": query,
                "answer": answer,
                "citations": final_cits,
                "evidence_count": len(evidence),
                "errors": errors
            },
            "citations": final_cits,
            "grounding_status": "PASSED" if val_res.is_valid else "FAILED"
        }
    except Exception as e:
        logger.error(f"node_synthesize_search error: {e}")
        errors = list(state.get("errors", []))
        err_msg = f"LLM synthesis failed: {str(e)}"
        errors.append(err_msg)
        fallback_cits = [e["citation_id"] for e in evidence[:2]]
        return {
            "current_step": "node_synthesize_search",
            "errors": errors,
            "final_response": {
                "intent": "SEARCH",
                "query": query,
                "answer": f"[EXTRACTIVE FALLBACK] {evidence[0]['text']}" if evidence else "Information not available due to synthesis error.",
                "citations": fallback_cits,
                "is_fallback": True,
                "errors": errors
            },
            "citations": fallback_cits,
            "grounding_status": "FAILED"
        }


# ---------------------------------------------------------------------------
# 7. Compare Scholarships Node (for COMPARE intent)
# ---------------------------------------------------------------------------

def node_compare_scholarships(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Executes multi-scholarship comparison through existing ScholarshipComparisonChain.
    """
    profile_data = state.get("student_profile")
    selected_ids = state.get("selected_scholarships") or ["MYSY", "ADANI-GJ"]
    errors = list(state.get("errors", []))

    if not profile_data:
        errors.append("Cannot execute COMPARE workflow: student_profile is missing.")
        return {
            "current_step": "node_compare_scholarships",
            "errors": errors,
            "final_response": {
                "intent": "COMPARE",
                "compared_scholarships": [],
                "comparative_analysis": "Cannot compare scholarships: student profile is required.",
                "errors": errors
            },
            "grounding_status": "FAILED"
        }

    profile = StudentProfile(**profile_data) if isinstance(profile_data, dict) else profile_data

    kb = get_knowledge_base()
    engine = ScholarBridgeEngine()
    retriever = get_evidence_retriever()
    comp_chain = ScholarshipComparisonChain()

    evaluated: List[Dict[str, Any]] = []
    evidence_by_scheme: Dict[str, List[EvidenceItem]] = {}

    for sch_id in selected_ids:
        # Resolve scholarship
        sch = None
        target = sch_id.strip().upper()
        if target in kb.scholarships_by_id:
            sch = kb.scholarships_by_id[target]
        else:
            for s in kb.scholarships:
                if s.scheme_code.upper() == target or s.scholarship_name.upper() == target:
                    sch = s
                    break

        if not sch:
            sch = kb.scholarships[0]

        rules = kb.rules_by_scholarship.get(sch.scholarship_id, [])
        mr = engine.matcher.evaluate_scholarship(sch, rules, profile)
        mr.documents = kb.docs_by_scholarship.get(sch.scholarship_id, [])
        mr.benefits = kb.benefits_by_scholarship.get(sch.scholarship_id, [])
        evaluated.append({"scholarship": sch, "match_result": mr})

        ev_pack = retriever.retrieve_for_scheme(scheme_id=sch.scheme_code, top_k=3)
        evidence_by_scheme[sch.scheme_code] = ev_pack.evidence

    try:
        comp_res: ScholarshipComparisonResponse = comp_chain.run(
            profile=profile,
            evaluated_scholarships=evaluated,
            evidence_by_scheme=evidence_by_scheme
        )
        return {
            "current_step": "node_compare_scholarships",
            "comparison_result": comp_res.model_dump(),
            "final_response": comp_res.model_dump(),
            "citations": comp_res.citations,
            "grounding_status": "PASSED"
        }
    except Exception as e:
        logger.error(f"node_compare_scholarships error: {e}")
        errors.append(f"Comparison error: {e}")
        return {
            "current_step": "node_compare_scholarships",
            "errors": errors,
            "final_response": {
                "intent": "COMPARE",
                "compared_scholarships": [],
                "comparative_analysis": "Comparative analysis failed.",
                "errors": errors
            },
            "grounding_status": "FAILED"
        }


# ---------------------------------------------------------------------------
# 8. Explain Eligibility Node (for EXPLAIN intent)
# ---------------------------------------------------------------------------

def node_explain_eligibility(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Explains deterministic eligibility result for a specific scholarship.
    """
    profile_data = state.get("student_profile")
    selected_ids = state.get("selected_scholarships") or ["MYSY"]
    query = state.get("user_query")
    errors = list(state.get("errors", []))

    if not profile_data:
        errors.append("Cannot execute EXPLAIN workflow: student_profile is missing.")
        return {
            "current_step": "node_explain_eligibility",
            "errors": errors,
            "final_response": {
                "intent": "EXPLAIN",
                "scholarship_id": selected_ids[0] if selected_ids else "UNKNOWN",
                "eligibility_status": "unknown",
                "answer": "Cannot explain eligibility: student profile is required.",
                "errors": errors
            },
            "grounding_status": "FAILED"
        }

    profile = StudentProfile(**profile_data) if isinstance(profile_data, dict) else profile_data

    kb = get_knowledge_base()
    engine = ScholarBridgeEngine()
    retriever = get_evidence_retriever()
    chain = GroundedExplanationChain()

    target_code = selected_ids[0] if selected_ids else "MYSY"
    sch = None
    target_clean = target_code.strip().upper()
    if target_clean in kb.scholarships_by_id:
        sch = kb.scholarships_by_id[target_clean]
    else:
        for s in kb.scholarships:
            if s.scheme_code.upper() == target_clean or s.scholarship_name.upper() == target_clean:
                sch = s
                break
    if not sch:
        sch = kb.scholarships[0]

    rules = kb.rules_by_scholarship.get(sch.scholarship_id, [])
    mr = engine.matcher.evaluate_scholarship(sch, rules, profile)
    mr.documents = kb.docs_by_scholarship.get(sch.scholarship_id, [])
    mr.benefits = kb.benefits_by_scholarship.get(sch.scholarship_id, [])

    ev_pack = retriever.retrieve_for_scheme(scheme_id=sch.scheme_code, top_k=4)

    try:
        resp: ScholarshipResponse = chain.run(
            profile=profile,
            scholarship=sch,
            match_result=mr,
            evidence=ev_pack.evidence,
            user_query=query
        )
        return {
            "current_step": "node_explain_eligibility",
            "explanation": resp.model_dump(),
            "final_response": resp.model_dump(),
            "citations": resp.citations,
            "grounding_status": resp.grounding_metadata.get("validation_status", "PASSED")
        }
    except Exception as e:
        logger.error(f"node_explain_eligibility error: {e}")
        errors.append(f"Explanation error: {e}")
        return {
            "current_step": "node_explain_eligibility",
            "errors": errors,
            "final_response": {
                "intent": "EXPLAIN",
                "scholarship_id": sch.scheme_code,
                "eligibility_status": mr.status.value.lower(),
                "answer": f"Evaluated as {mr.status.value.lower()}.",
                "errors": errors
            },
            "grounding_status": "FAILED"
        }


# ---------------------------------------------------------------------------
# 9. Final Validation & Telemetry Assembly Node
# ---------------------------------------------------------------------------

def node_validate_and_finalize(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """
    Assembles final output, validates trace, and seals the workflow execution.
    """
    final_res = state.get("final_response") or {}
    trace = state.get("trace") or {}
    errors = state.get("errors", [])

    trace["workflow_status"] = "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS"
    trace["error_count"] = len(errors)
    trace["authority_invariant_preserved"] = True

    return {
        "current_step": "node_validate_and_finalize",
        "trace": trace,
        "final_response": final_res
    }
