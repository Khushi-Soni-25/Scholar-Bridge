"""
ScholarBridge Evaluation (Section 14B)
========================================
A lightweight sanity-check framework using representative SYNTHETIC student
personas (never fabricated real-world scholarship outcomes) to verify the
full pipeline (RuleMatcher -> RF -> NLP -> hybrid scoring -> ranking ->
final recommendations -> explanations) behaves logically and consistently.

Every stage this module drives is the REAL, unmodified production code
(RuleMatcher, RecommendationScorer, hybrid_scorer, ranker, recommender,
explainer). The only mocked inputs are rf_confidence_score /
nlp_similarity_score (fixed stand-in numbers, since no trained RF model or
downloaded embedding model is required to sanity-check the pipeline's
LOGIC -- see the Step 9 report for the mock-vs-real-model distinction).
"""

from typing import Any, Dict, List, Optional

from .explainer import build_final_recommendation_explanations
from .hybrid_scorer import build_hybrid_result
from .loader import KnowledgeBase
from .matcher import RuleMatcher
from .models import EligibilityStatus, StudentProfile
from .ranker import rank_scholarships
from .recommender import build_final_recommendations
from .scorer import RecommendationScorer

# ---------------------------------------------------------------------------
# Representative synthetic personas (Section 14B)
# ---------------------------------------------------------------------------

PERSONAS: List[Dict[str, Any]] = [
    {
        "persona_id": "A",
        "label": "Clearly eligible student",
        "profile_kwargs": dict(
            social_category="SC", gender="Female", annual_family_income=150000.0,
            domicile_state="Gujarat", study_state="Gujarat", education_level="Undergraduate",
            stream="Engineering", course_name="B.Tech Computer Engineering",
            marks_percentile=88.0, hostel_status="College_Affiliated_Hostel",
        ),
    },
    {
        "persona_id": "B",
        "label": "Clearly ineligible due to income/category/education mismatch",
        "profile_kwargs": dict(
            social_category="General", gender="Male", annual_family_income=5000000.0,
            domicile_state="Maharashtra", study_state="Maharashtra", education_level="Postgraduate",
            stream="Arts", course_name="M.A. History",
            marks_percentile=40.0, hostel_status="Day_Scholar",
        ),
    },
    {
        "persona_id": "C",
        "label": "Conditional / incomplete profile",
        "profile_kwargs": dict(
            social_category="OBC", gender="Female", annual_family_income=250000.0,
            domicile_state="Gujarat", study_state="Gujarat", education_level="Undergraduate",
            stream="Science", course_name="B.Sc.",
            marks_percentile=65.0, hostel_status="Day_Scholar",
        ),
    },
    {
        "persona_id": "D",
        "label": "Strong semantic similarity but rule-engine ineligible (tests NLP != eligibility)",
        "profile_kwargs": dict(
            social_category="General", gender="Male", annual_family_income=3000000.0,
            domicile_state="Gujarat", study_state="Gujarat", education_level="Undergraduate",
            stream="Engineering", course_name="B.Tech Computer Engineering",
            marks_percentile=95.0, hostel_status="College_Affiliated_Hostel",
        ),
        # Forces a MOCK nlp_similarity_score to its maximum for every
        # scholarship (including INELIGIBLE ones), specifically to test that
        # high semantic similarity can never override rule-based ineligibility.
        "nlp_override": 99.0,
    },
    {
        "persona_id": "E",
        "label": "Multiple scholarships expected to receive different ranking scores",
        "profile_kwargs": dict(
            social_category="ST", gender="Female", annual_family_income=180000.0,
            domicile_state="Gujarat", study_state="Gujarat", education_level="Undergraduate",
            stream="Engineering", course_name="B.Tech Mechanical Engineering",
            marks_percentile=79.0, hostel_status="College_Affiliated_Hostel",
        ),
    },
]


def run_pipeline_for_persona(
    kb: KnowledgeBase,
    matcher: RuleMatcher,
    profile_kwargs: Dict[str, Any],
    rf_confidence_score: float = 55.0,
    nlp_similarity_score: float = 42.0,
    hybrid_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Runs the full Section 8-14 pipeline for one synthetic profile.

    rf_confidence_score / nlp_similarity_score are MOCK stand-in values
    (uniformly applied), since this function's purpose is to sanity-check
    pipeline LOGIC, not to validate the trained RF model or the real
    sentence-transformers embeddings (which require dependencies not
    available in every environment -- see Section 14's mock-vs-real-model
    validation note).
    """
    profile = StudentProfile(**profile_kwargs)
    scorer = RecommendationScorer()

    eligible_matches: List = []
    conditional_matches: List = []
    ineligible_matches: List = []
    suspended_matches: List = []

    for scholarship in kb.scholarships:
        rules = kb.rules_by_scholarship.get(scholarship.scholarship_id, [])
        m = matcher.evaluate_scholarship(scholarship, rules, profile)
        {
            EligibilityStatus.ELIGIBLE: eligible_matches,
            EligibilityStatus.CONDITIONAL: conditional_matches,
            EligibilityStatus.SUSPENDED: suspended_matches,
        }.get(m.status, ineligible_matches).append(m)

    all_matches = eligible_matches + conditional_matches + ineligible_matches + suspended_matches

    # Section 10 contract: nlp_similarity_score attached to ALL four buckets.
    for m in all_matches:
        m.explanation_details["nlp_similarity_score"] = nlp_similarity_score

    # Section 9 contract: rf_confidence_score attached to ELIGIBLE+CONDITIONAL only.
    for m in eligible_matches + conditional_matches:
        m.explanation_details["rf_confidence_score"] = rf_confidence_score

    # Section 11: rule_relevance_score via the REAL, unmodified RecommendationScorer.
    ranked_eligible = scorer.score_and_rank(eligible_matches, profile)
    ranked_conditional = scorer.score_and_rank(conditional_matches, profile)
    for item in ranked_eligible + ranked_conditional:
        item.match_result.explanation_details["rule_relevance_score"] = item.final_score

    hybrid_results = []
    for m in all_matches:
        rule_relevance = m.explanation_details.get("rule_relevance_score")
        record = build_hybrid_result(m, rule_relevance_score=rule_relevance, weights=hybrid_weights)
        m.explanation_details["hybrid_score"] = record["hybrid_score"]
        hybrid_results.append(record)

    # Section 12: real ranker.
    ranking = rank_scholarships(hybrid_results)

    # Section 13: real recommender.
    final_recommendations = build_final_recommendations(ranking, kb.scholarships_by_id)

    # Section 14A: real explainer.
    match_lookup = {m.scholarship.scholarship_id: m for m in all_matches}
    explanations = build_final_recommendation_explanations(final_recommendations, match_lookup)

    return {
        "profile": profile,
        "eligible_matches": eligible_matches,
        "conditional_matches": conditional_matches,
        "ineligible_matches": ineligible_matches,
        "suspended_matches": suspended_matches,
        "all_matches": all_matches,
        "match_lookup": match_lookup,
        "hybrid_results": hybrid_results,
        "ranking": ranking,
        "final_recommendations": final_recommendations,
        "explanations": explanations,
    }


def run_persona(
    kb: KnowledgeBase,
    matcher: RuleMatcher,
    persona: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs one PERSONAS entry through run_pipeline_for_persona, applying
    any persona-specific overrides (e.g. Persona D's forced high NLP score)."""
    kwargs = dict(
        rf_confidence_score=persona.get("rf_override", 55.0),
        nlp_similarity_score=persona.get("nlp_override", 42.0),
    )
    result = run_pipeline_for_persona(kb, matcher, persona["profile_kwargs"], **kwargs)
    return {"persona_id": persona["persona_id"], "label": persona["label"], **result}


# ---------------------------------------------------------------------------
# Evaluation checks (Section 14B, required checks 1-8)
# ---------------------------------------------------------------------------

def evaluate_persona_result(persona_result: Dict[str, Any]) -> Dict[str, bool]:
    """Runs checks 1-6 and 8 against a single persona's pipeline result
    (checks 7 [determinism] is run separately across two full pipeline runs
    -- see `check_determinism`)."""
    checks: Dict[str, bool] = {}

    status_by_id = {m.scholarship.scholarship_id: m.status for m in persona_result["all_matches"]}

    # 1. Eligibility authority: RuleMatcher status must equal what's recorded
    #    in hybrid_results / ranking / final_recommendations / explanations.
    ok = True
    for r in persona_result["hybrid_results"]:
        if EligibilityStatus(r["eligibility_status"]) != status_by_id.get(r["scholarship_id"]):
            ok = False
    checks["1_eligibility_authority_unchanged"] = ok

    # 2. Ranking consistency: final_recommendations preserves Section 12 order.
    fr = persona_result["final_recommendations"]
    rk = persona_result["ranking"]
    ok = (
        [r["scholarship_id"] for r in fr["eligible_recommendations"]] == [r["scholarship_id"] for r in rk["eligible"]]
        and [r["scholarship_id"] for r in fr["conditional_recommendations"]] == [r["scholarship_id"] for r in rk["conditional"]]
    )
    checks["2_ranking_consistency"] = ok

    # 3. Explanation consistency: explanation status equals RuleMatcher status.
    ex = persona_result["explanations"]
    ok = all(
        e["eligibility_status"] == status_by_id.get(e["scholarship_id"]).value
        for e in ex["eligible_explanations"] + ex["conditional_explanations"]
    )
    checks["3_explanation_status_consistency"] = ok

    # 4. Score consistency: explanation scores equal Section 11 values exactly.
    hybrid_by_id = {r["scholarship_id"]: r for r in persona_result["hybrid_results"]}
    ok = all(
        e["score_breakdown"]["hybrid_score"] == hybrid_by_id[e["scholarship_id"]]["hybrid_score"]
        and e["score_breakdown"]["rule_relevance_score"] == hybrid_by_id[e["scholarship_id"]]["rule_relevance_score"]
        and e["score_breakdown"]["rf_confidence_score"] == hybrid_by_id[e["scholarship_id"]]["rf_confidence_score"]
        and e["score_breakdown"]["nlp_similarity_score"] == hybrid_by_id[e["scholarship_id"]]["nlp_similarity_score"]
        for e in ex["eligible_explanations"] + ex["conditional_explanations"]
    )
    checks["4_score_consistency"] = ok

    # 5. Exclusion safety: no INELIGIBLE/SUSPENDED in final recommendations.
    excluded_ids = {m.scholarship.scholarship_id for m in persona_result["ineligible_matches"] + persona_result["suspended_matches"]}
    final_ids = {r["scholarship_id"] for r in fr["eligible_recommendations"] + fr["conditional_recommendations"]}
    checks["5_exclusion_safety"] = not (excluded_ids & final_ids)

    # 6. Semantic-vs-eligibility separation: a high NLP score on an INELIGIBLE
    #    match must never appear as an ELIGIBLE/CONDITIONAL recommendation.
    ineligible_ids = {m.scholarship.scholarship_id for m in persona_result["ineligible_matches"]}
    checks["6_semantic_vs_eligibility_separation"] = not (ineligible_ids & final_ids)

    # 8. No hallucinated metadata: every scholarship_metadata field, when
    #    present, must trace back to the real KB record for that scholarship.
    ok = True
    for e in ex["eligible_explanations"] + ex["conditional_explanations"]:
        meta = e.get("scholarship_metadata")
        if meta is None:
            continue
        match = persona_result["match_lookup"].get(e["scholarship_id"])
        if match is None:
            ok = False
            continue
        s = match.scholarship
        if meta["provider"] not in (None, s.provider_name if s.provider_name.strip() else None):
            ok = False
        if meta["application_portal"] not in (None, s.application_portal if s.application_portal.strip() else None):
            ok = False
    checks["8_no_hallucinated_metadata"] = ok

    return checks


def check_determinism(kb: KnowledgeBase, matcher: RuleMatcher, persona: Dict[str, Any]) -> bool:
    """Check 7: same profile + same KB produces the same explanation/evaluation result."""
    r1 = run_persona(kb, matcher, persona)
    r2 = run_persona(kb, matcher, persona)
    return r1["explanations"] == r2["explanations"] and r1["final_recommendations"] == r2["final_recommendations"]


def run_full_evaluation(kb: KnowledgeBase, matcher: RuleMatcher, personas: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Runs every persona through the pipeline and every applicable check.
    Returns a dict keyed by persona_id with the pipeline result and checks."""
    personas = personas if personas is not None else PERSONAS
    report = {}
    for persona in personas:
        result = run_persona(kb, matcher, persona)
        checks = evaluate_persona_result(result)
        checks["7_determinism"] = check_determinism(kb, matcher, persona)
        report[persona["persona_id"]] = {
            "label": persona["label"],
            "result": result,
            "checks": checks,
            "all_passed": all(checks.values()),
        }
    return report
