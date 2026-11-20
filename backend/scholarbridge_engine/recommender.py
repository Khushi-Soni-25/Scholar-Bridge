"""
ScholarBridge Final Recommendations (Section 13)
==================================================
Turns the Section 12 ranked output into the final recommendation payload
consumable by the ScholarBridge application/UI.

CRITICAL INVARIANT: this module recomputes NOTHING. It does not touch
eligibility, RF confidence, NLP similarity, hybrid_score, or ranking order --
it only reads the already-ranked Section 12 records (rank, eligibility_status,
scholarship_id, scholarship_name, rule_relevance_score, rf_confidence_score,
nlp_similarity_score, hybrid_score), enriches them with read-only scholarship
metadata already present in the knowledge base, and slices to top-N. The
RuleMatcher-assigned eligibility_status is trusted exactly as it arrives from
Section 12 -- INELIGIBLE and SUSPENDED scholarships are never present in that
input (Section 12 already excludes them), and this module never re-derives
or second-guesses that exclusion.
"""

from typing import Any, Dict, List, Optional

from .models import Scholarship

_RANKED_FIELDS = (
    "rank", "scholarship_id", "scholarship_name", "eligibility_status",
    "hybrid_score", "rule_relevance_score", "rf_confidence_score", "nlp_similarity_score",
)


def _clean(value: Any) -> Any:
    """Normalizes a raw knowledge-base field into a safe display value.
    Blank strings (common for unset CSV columns) become None rather than
    being fabricated into something else. Never invents a replacement value."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _application_status_from_timeline(timeline_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Derives a presentational application/deadline status from the
    EXISTING computed fields in a DeadlineTracker.get_timeline_info() dict --
    never invents new data, and is entirely distinct from RuleMatcher's
    eligibility_status (e.g. a scholarship can be eligibility_status=ELIGIBLE
    while application_status=CLOSED for the current cycle). Returns None
    when no timeline_info is available, rather than guessing."""
    if not timeline_info:
        return None
    status = timeline_info.get("status")
    if status == "SUSPENDED":
        return "SUSPENDED"  # scheme-level suspension, as already flagged by DeadlineTracker
    if timeline_info.get("is_open"):
        return "OPEN"
    display = (timeline_info.get("timeline_display") or "").lower()
    if display.startswith("closed"):
        return "CLOSED"
    if display.startswith("opens in"):
        return "UPCOMING"
    return "UNKNOWN"


def build_scholarship_lookup(scholarships: List[Scholarship]) -> Dict[str, Scholarship]:
    """Convenience helper: builds a scholarship_id -> Scholarship dict from
    whatever scholarship list the notebook already has loaded (mongo_kb.scholarships,
    kb.scholarships, etc.). Purely a read-only index; does not copy or mutate
    the Scholarship objects themselves."""
    return {s.scholarship_id: s for s in scholarships}


def _scholarship_details(scholarship: Optional[Scholarship]) -> Optional[Dict[str, Any]]:
    """Extracts only fields already present on the Scholarship record. Never
    fabricates a value: an unavailable/blank field is represented as None."""
    if scholarship is None:
        return None
    return {
        "provider": _clean(scholarship.provider_name),
        "department": _clean(scholarship.department),
        "education_level": _clean(scholarship.education_level_normalized),
        "target_category": _clean(scholarship.eligible_categories),
        "financial_benefit": {
            "benefit_type": _clean(scholarship.benefit_type),
            "description": _clean(scholarship.benefit_description),
            "max_amount_inr": _clean(scholarship.max_benefit_amount_inr),
        },
        "annual_income_limit": _clean(scholarship.annual_income_limit_inr),
        "application_portal": _clean(scholarship.application_portal),
        "deadline": {
            "application_end_date": _clean(scholarship.application_end_date),
            "status": _clean(scholarship.deadline_status),
        },
        "geographic_applicability": _clean(scholarship.gujarat_applicability),
        "gender_eligibility": _clean(scholarship.gender),
    }


def _build_recommendation(
    ranked_item: Dict[str, Any],
    scholarship_lookup: Dict[str, Scholarship],
    timeline_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Builds one recommendation record. Returns a brand-new dict -- never
    mutates `ranked_item` or the looked-up Scholarship object."""
    record = {field: ranked_item.get(field) for field in _RANKED_FIELDS}
    scholarship = scholarship_lookup.get(ranked_item.get("scholarship_id"))
    record["scholarship_details"] = _scholarship_details(scholarship)
    timeline_info = (timeline_lookup or {}).get(ranked_item.get("scholarship_id"))
    record["application_status"] = _application_status_from_timeline(timeline_info)
    return record


def build_final_recommendations(
    ranking: Dict[str, Any],
    scholarship_lookup: Optional[Dict[str, Scholarship]] = None,
    top_n: Optional[int] = None,
    top_n_eligible: Optional[int] = None,
    top_n_conditional: Optional[int] = None,
    timeline_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Builds the Section 13 final recommendation payload from Section 12's
    `ranking` dict (as returned by `scholarbridge_engine.ranker.rank_scholarships`).

    Args:
        ranking: the Section 12 output -- must contain 'eligible' and
            'conditional' lists, already sorted and ranked. This function
            does not re-sort them; it only slices and enriches, in the exact
            order Section 12 produced.
        scholarship_lookup: optional scholarship_id -> Scholarship dict (see
            `build_scholarship_lookup`) used only to enrich the output with
            read-only KB metadata. If omitted, `scholarship_details` is None
            for every recommendation rather than crashing.
        top_n: optional convenience limit applied to BOTH eligible and
            conditional unless overridden below. `None` (default) returns
            every ranked record Section 12 produced.
        top_n_eligible / top_n_conditional: per-bucket overrides of `top_n`.
        timeline_lookup: optional scholarship_id -> DeadlineTracker timeline
            dict (see `deadline_tracker.DeadlineTracker.get_timeline_info`),
            used only to derive `application_status` (OPEN/CLOSED/UPCOMING/
            SUSPENDED/UNKNOWN) -- a presentational field entirely separate
            from `eligibility_status`. If omitted, `application_status` is
            None for every recommendation rather than guessed.

    Returns:
        {
            "eligible_recommendations": [...],
            "conditional_recommendations": [...],
            "metadata": {
                "total_eligible": int, "total_conditional": int,
                "returned_eligible": int, "returned_conditional": int,
                "excluded_count": int,  # INELIGIBLE + SUSPENDED, from Section 12 -- never a recommendation
                "has_recommendations": bool,
            },
        }
    """
    scholarship_lookup = scholarship_lookup or {}

    ranked_eligible = ranking.get("eligible") or []
    ranked_conditional = ranking.get("conditional") or []

    limit_eligible = top_n_eligible if top_n_eligible is not None else top_n
    limit_conditional = top_n_conditional if top_n_conditional is not None else top_n

    # Slice the ALREADY-sorted Section 12 lists -- no re-sorting, no re-ranking.
    sliced_eligible = ranked_eligible[:limit_eligible] if limit_eligible is not None else ranked_eligible
    sliced_conditional = ranked_conditional[:limit_conditional] if limit_conditional is not None else ranked_conditional

    eligible_recommendations = [_build_recommendation(r, scholarship_lookup, timeline_lookup) for r in sliced_eligible]
    conditional_recommendations = [_build_recommendation(r, scholarship_lookup, timeline_lookup) for r in sliced_conditional]

    metadata = {
        "total_eligible": len(ranked_eligible),
        "total_conditional": len(ranked_conditional),
        "returned_eligible": len(eligible_recommendations),
        "returned_conditional": len(conditional_recommendations),
        "excluded_count": ranking.get("excluded_count", 0),  # INELIGIBLE + SUSPENDED -- informational only, never a recommendation
        "has_recommendations": bool(eligible_recommendations or conditional_recommendations),
    }

    return {
        "eligible_recommendations": eligible_recommendations,
        "conditional_recommendations": conditional_recommendations,
        "metadata": metadata,
    }
