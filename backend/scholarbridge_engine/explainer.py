"""
ScholarBridge Explainability Engine
===================================
Generates clear, human-readable explanations for eligible matches,
conditional requirements, and specific rejection reasons.

Section 14A extends this module to build a full, structured explanation for
each Section 13 final recommendation (rule-relevance / RF confidence / NLP
similarity / hybrid score, matched factors, missing-or-uncertain factors,
and scholarship metadata) WITHOUT recalculating eligibility, RF, NLP,
hybrid_score, or ranking -- those are trusted exactly as produced by
Sections 8-13.

CRITICAL SEMANTIC RULE (see module-level constants below): the RF score is a
confidence signal that the rule-engine logic would classify the profile as
eligible -- it is NOT an award/approval/success probability. The NLP score
is a semantic-relevance/similarity signal -- it is NOT an eligibility
probability. The hybrid score is a ranking/relevance score, not a
probability of anything. RuleMatcher remains the sole authority for
eligibility_status; nothing in this module can change it.
"""

from typing import Any, Dict, List, Optional

from .models import EligibilityStatus, MatchResult, StudentProfile

# ---------------------------------------------------------------------------
# Section 14A: structured explanation for Section 13 final recommendations
# ---------------------------------------------------------------------------

_FRIENDLY_FIELD_NAMES = {
    "education_level": "Education Level",
    "education_level_normalized": "Education Level",
    "category": "Category",
    "social_category": "Category",
    "gender": "Gender",
    "income": "Income",
    "annual_family_income": "Income",
    "marks": "Marks / Percentile",
    "marks_percentile": "Marks / Percentile",
    "course": "Course / Stream",
    "stream": "Course / Stream",
    "course_name": "Course / Stream",
    "domicile_state": "Domicile / Geography",
    "study_state": "Domicile / Geography",
    "district": "District",
    "disability": "Disability / PwD Status",
    "pwd": "Disability / PwD Status",
    "beneficiary_group": "Special Beneficiary Group",
    "entrance_exam": "Entrance Exam",
    "hostel_status": "Hostel Status",
}


def _friendly_field(field_name: Optional[str]) -> str:
    if not field_name:
        return "Requirement"
    return _FRIENDLY_FIELD_NAMES.get(field_name, field_name.replace("_", " ").title())


def _band_key(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 85:
        return "very_high"
    if score >= 70:
        return "high"
    if score >= 50:
        return "moderate"
    if score >= 30:
        return "low"
    return "very_low"


# Adjective used per component per band -- chosen to match the wording style
# requested in the spec ("strong rule-based match", "high semantic
# relevance", "strong agreement with the rule-engine eligibility pattern",
# "high overall ranking score") while never implying a probability.
_COMPONENT_ADJECTIVES = {
    "rule_relevance": {
        "very_high": "very strong", "high": "strong", "moderate": "moderate",
        "low": "weak", "very_low": "very weak",
    },
    "rf_confidence": {
        "very_high": "very strong", "high": "strong", "moderate": "moderate",
        "low": "weak", "very_low": "very weak",
    },
    "nlp_similarity": {
        "very_high": "very high", "high": "high", "moderate": "moderate",
        "low": "low", "very_low": "very low",
    },
    "hybrid_score": {
        "very_high": "very high", "high": "high", "moderate": "moderate",
        "low": "low", "very_low": "very low",
    },
}

# Fixed, non-probabilistic phrasing per component. These strings are the
# semantic contract enforced by Step 9 Test 9/10: none of them describe an
# award/approval/success probability, and none turn NLP relevance into an
# eligibility probability.
_COMPONENT_DESCRIPTIONS = {
    "rule_relevance": "how closely this scholarship's rule-engine criteria align with the profile",
    "rf_confidence": "a model confidence signal indicating agreement with the rule engine's own eligibility pattern -- separate from, and never a substitute for, the rule engine's actual decision",
    "nlp_similarity": "the semantic/textual similarity between the profile and this scholarship's description -- a relevance signal only",
    "hybrid_score": "a combined ranking score used only to order recommendations",
}

SCORE_SEMANTICS_DISCLAIMER = (
    "These four scores describe relevance and internal model agreement for ranking purposes only. "
    "None of them measure or estimate the likelihood of actually being awarded a scholarship. "
    "Eligibility is decided solely by the deterministic rule engine (RuleMatcher); the RF and NLP "
    "scores do not represent eligibility and cannot change the eligibility_status shown above."
)


def _component_text(component: str, score: Optional[float]) -> str:
    if score is None:
        return f"No {component.replace('_', ' ')} value is available for this scholarship."
    band = _band_key(score)
    adjective = _COMPONENT_ADJECTIVES[component][band]
    description = _COMPONENT_DESCRIPTIONS[component]
    return f"{adjective.capitalize()} {component.replace('_', ' ')} ({score:.1f}/100) -- {description}."


def _matched_factors_from_match(match: Optional[MatchResult]) -> List[Dict[str, str]]:
    """Derives matched factors ONLY from the rule engine's own passed_rules
    -- never fabricated, never inferred beyond what RuleMatcher recorded."""
    if match is None:
        return []
    factors = []
    for ev in match.passed_rules:
        if not ev.reason:
            continue
        factors.append({"factor": _friendly_field(ev.rule.field if ev.rule else None), "detail": ev.reason})
    return factors


def _missing_or_uncertain_factors_from_match(match: Optional[MatchResult]) -> List[Dict[str, str]]:
    """Derives missing/uncertain factors ONLY from the rule engine's own
    conditional_rules (items that require verification). Does not treat
    missing profile information as ineligibility -- that determination
    belongs solely to RuleMatcher."""
    if match is None:
        return []
    factors = []
    for ev in match.conditional_rules:
        if not ev.reason:
            continue
        factors.append({"factor": _friendly_field(ev.rule.field if ev.rule else None), "detail": ev.reason})
    return factors


def _eligibility_explanation(status: str, matched_factors: List[Dict[str, str]], missing_factors: List[Dict[str, str]]) -> str:
    if status == "ELIGIBLE":
        if matched_factors:
            joined = "; ".join(f"{f['factor']}: {f['detail']}" for f in matched_factors)
            return f"You meet the scholarship's eligibility requirements based on the rule engine's evaluation. Matched factors: {joined}."
        return "This scholarship was classified as ELIGIBLE by the rule engine."
    if status == "CONDITIONAL":
        if missing_factors:
            joined = "; ".join(f"{f['factor']}: {f['detail']}" for f in missing_factors)
            return f"Your profile matches the available conditions, but some information requires verification: {joined}."
        return "Your profile matches the available conditions, but some information requires verification before this scholarship can be confirmed as eligible."
    if status == "INELIGIBLE":
        return "This scholarship was classified as INELIGIBLE by the rule engine and is excluded from recommendations."
    if status == "SUSPENDED":
        return "This scholarship is currently SUSPENDED by the rule engine and is excluded from recommendations."
    return f"Eligibility status: {status}."


def build_recommendation_explanation(
    recommendation: Dict[str, Any],
    match_result: Optional[MatchResult] = None,
) -> Dict[str, Any]:
    """Builds one structured explanation for a single Section 13 final
    recommendation record. Reads scores/status/metadata as-is; recalculates
    nothing. `match_result` (optional) supplies the underlying RuleMatcher
    evaluation for that scholarship, used only to extract matched/missing
    factors it already recorded -- never to re-derive eligibility."""
    status = recommendation.get("eligibility_status")
    rule_relevance = recommendation.get("rule_relevance_score")
    rf_confidence = recommendation.get("rf_confidence_score")
    nlp_similarity = recommendation.get("nlp_similarity_score")
    hybrid_score = recommendation.get("hybrid_score")

    matched_factors = _matched_factors_from_match(match_result)
    missing_factors = _missing_or_uncertain_factors_from_match(match_result)

    score_breakdown = {
        "rule_relevance_score": rule_relevance,
        "rule_relevance_explanation": _component_text("rule_relevance", rule_relevance),
        "rf_confidence_score": rf_confidence,
        "rf_confidence_explanation": _component_text("rf_confidence", rf_confidence),
        "nlp_similarity_score": nlp_similarity,
        "nlp_similarity_explanation": _component_text("nlp_similarity", nlp_similarity),
        "hybrid_score": hybrid_score,
        "hybrid_score_explanation": _component_text("hybrid_score", hybrid_score),
        "disclaimer": SCORE_SEMANTICS_DISCLAIMER,
    }

    hybrid_band = _band_key(hybrid_score)
    hybrid_adj = _COMPONENT_ADJECTIVES["hybrid_score"].get(hybrid_band, "n/a") if hybrid_band else "n/a"
    if status == "ELIGIBLE":
        summary = (f"Ranked #{recommendation.get('rank')} -- {hybrid_adj} overall ranking score "
                    f"({hybrid_score}/100) with {len(matched_factors)} matched factor(s).")
    elif status == "CONDITIONAL":
        summary = (f"Ranked #{recommendation.get('rank')} -- conditional match; "
                    f"{len(missing_factors)} item(s) require verification.")
    else:
        summary = f"Excluded from recommendations (status: {status})."

    return {
        "scholarship_id": recommendation.get("scholarship_id"),
        "scholarship_name": recommendation.get("scholarship_name"),
        "eligibility_status": status,
        "rank": recommendation.get("rank"),
        "hybrid_score": hybrid_score,
        "summary": summary,
        "eligibility_explanation": _eligibility_explanation(status, matched_factors, missing_factors),
        "score_breakdown": score_breakdown,
        "matched_factors": matched_factors,
        "missing_or_uncertain_factors": missing_factors,
        "scholarship_metadata": recommendation.get("scholarship_details"),
    }


def build_final_recommendation_explanations(
    final_recommendations: Dict[str, Any],
    match_lookup: Optional[Dict[str, MatchResult]] = None,
) -> Dict[str, Any]:
    """Builds explanations for every Section 13 final recommendation,
    preserving Section 12/13's exact order (never re-sorted). `match_lookup`
    (scholarship_id -> MatchResult) is optional; when omitted, matched/missing
    factors are empty lists rather than fabricated."""
    match_lookup = match_lookup or {}

    eligible_explanations = [
        build_recommendation_explanation(rec, match_lookup.get(rec.get("scholarship_id")))
        for rec in final_recommendations.get("eligible_recommendations", [])
    ]
    conditional_explanations = [
        build_recommendation_explanation(rec, match_lookup.get(rec.get("scholarship_id")))
        for rec in final_recommendations.get("conditional_recommendations", [])
    ]

    return {
        "eligible_explanations": eligible_explanations,
        "conditional_explanations": conditional_explanations,
        "metadata": dict(final_recommendations.get("metadata", {})),
    }


def build_excluded_explanation(match: MatchResult) -> Dict[str, Any]:
    """Debug/evaluation-only helper: explains an INELIGIBLE or SUSPENDED
    match WITHOUT ever using recommendation/positive language, and WITHOUT
    ever changing the original status. Never called from the Section 13
    recommendation flow (those buckets never contain excluded records)."""
    status = match.status.value
    if status not in ("INELIGIBLE", "SUSPENDED"):
        raise ValueError(f"build_excluded_explanation is only for INELIGIBLE/SUSPENDED, got {status!r}.")

    failed_factors = [
        {"factor": _friendly_field(ev.rule.field if ev.rule else None), "detail": ev.reason}
        for ev in match.failed_rules if ev.reason and not ev.is_conditional
    ]

    return {
        "scholarship_id": match.scholarship.scholarship_id,
        "scholarship_name": match.scholarship.short_name,
        "eligibility_status": status,  # preserved exactly, never altered
        "summary": f"Excluded from recommendations (status: {status}). This is not a recommendation.",
        "exclusion_reason": match.exclusion_reason or match.explanation or None,
        "failed_factors": failed_factors,
    }


class ExplainabilityEngine:
    """Produces explainable rationales for eligibility and ineligibility."""

    @staticmethod
    def format_match_explanation(match: MatchResult, profile: StudentProfile) -> Dict[str, Any]:
        """Generates detailed explanation dictionary for a match result."""
        s = match.scholarship
        
        if match.status == EligibilityStatus.ELIGIBLE:
            satisfied_criteria = [f"✓ {ev.reason}" for ev in match.passed_rules if ev.reason]
            return {
                "status": "ELIGIBLE",
                "summary": s.eligibility_explanation_template or f"You meet all criteria for {s.short_name}.",
                "satisfied_criteria": satisfied_criteria,
                "benefit_summary": s.benefit_description,
                "portal_url": s.official_website or s.official_source_url
            }
            
        elif match.status == EligibilityStatus.CONDITIONAL:
            pending_actions = [f"⚠️ {ev.reason}" for ev in match.conditional_rules if ev.reason]
            satisfied_criteria = [f"✓ {ev.reason}" for ev in match.passed_rules if ev.reason]
            return {
                "status": "CONDITIONAL",
                "summary": match.explanation,
                "pending_requirements": pending_actions,
                "satisfied_criteria": satisfied_criteria,
                "benefit_summary": s.benefit_description,
                "portal_url": s.official_website or s.official_source_url
            }
            
        elif match.status == EligibilityStatus.SUSPENDED:
            return {
                "status": "SUSPENDED",
                "summary": match.explanation,
                "reason": match.exclusion_reason
            }
            
        else: # INELIGIBLE
            failure_reasons = [
                {
                    "field": ev.rule.field,
                    "expected": ev.expected_value,
                    "actual": ev.actual_value,
                    "detail": ev.reason
                }
                for ev in match.failed_rules if not ev.passed and not ev.is_conditional
            ]
            return {
                "status": "INELIGIBLE",
                "summary": s.ineligibility_exclusion_template or match.explanation,
                "failed_criteria": failure_reasons
            }
