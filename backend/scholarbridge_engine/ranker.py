"""
ScholarBridge Ranking (Section 12)
===================================
Consumes the Section 11 hybrid-scoring output (one dict per scholarship, as
produced by `scholarbridge_engine.hybrid_scorer.build_hybrid_result`) and
produces two independently sorted, deterministic rankings: ELIGIBLE and
CONDITIONAL. INELIGIBLE and SUSPENDED scholarships are always excluded.

CRITICAL INVARIANT: this module never recomputes hybrid_score and never
reads/derives eligibility. It only sorts and slices the records RuleMatcher
(via Section 8) and hybrid_scorer (Section 11) already produced. A
scholarship's `eligibility_status` string is trusted as-is; no score of any
kind -- rule_relevance_score, rf_confidence_score, nlp_similarity_score, or
hybrid_score -- can move a record between buckets, no matter how high.
"""

from typing import Any, Dict, List, Optional

_RANKABLE_STATUSES = ("ELIGIBLE", "CONDITIONAL")
_EXCLUDED_STATUSES = ("INELIGIBLE", "SUSPENDED")


class MissingHybridScoreError(ValueError):
    """Raised only in strict mode when a rankable record has no usable
    hybrid_score. In default (non-strict) mode such records are skipped
    instead -- see `rank_scholarships(..., strict=False)`."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sort_key(record: Dict[str, Any]):
    """Deterministic sort key, descending on every numeric field, with a
    final ascending name/id tie-breaker so ties never depend on input order.

    Order of precedence (all descending except the final tie-breaker):
        1. hybrid_score
        2. rule_relevance_score
        3. nlp_similarity_score
        4. rf_confidence_score
        5. scholarship_name (ascending, case-insensitive)
        6. scholarship_id (ascending) -- final deterministic tie-breaker

    A missing/non-numeric component sorts as the *worst* possible value for
    that key (never crashes, never wins a tie it has no data for).
    """
    def desc(value: Any) -> float:
        return -float(value) if _is_number(value) else float("inf")

    name = (record.get("scholarship_name") or "").lower()
    sid = record.get("scholarship_id") or ""

    return (
        desc(record.get("hybrid_score")),
        desc(record.get("rule_relevance_score")),
        desc(record.get("nlp_similarity_score")),
        desc(record.get("rf_confidence_score")),
        name,
        sid,
    )


def _to_ranked_item(record: Dict[str, Any], rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "eligibility_status": record["eligibility_status"],
        "scholarship_id": record["scholarship_id"],
        "scholarship_name": record.get("scholarship_name") or record.get("short_name"),
        "rule_relevance_score": record.get("rule_relevance_score"),
        "rf_confidence_score": record.get("rf_confidence_score"),
        "nlp_similarity_score": record.get("nlp_similarity_score"),
        "hybrid_score": record["hybrid_score"],
    }


def rank_scholarships(
    hybrid_results: List[Dict[str, Any]],
    top_n_eligible: Optional[int] = None,
    top_n_conditional: Optional[int] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Ranks Section 11 hybrid results into ELIGIBLE / CONDITIONAL buckets.

    Args:
        hybrid_results: records as produced by `hybrid_scorer.build_hybrid_result`
            (or any dict with the same keys), one per scholarship, covering
            all four eligibility buckets.
        top_n_eligible / top_n_conditional: optional caps applied AFTER full
            sorting. `None` (the default) returns every valid ranked record.
        strict: if True, a rankable (ELIGIBLE/CONDITIONAL) record with a
            missing or non-numeric hybrid_score raises MissingHybridScoreError.
            If False (default), such records are safely skipped and reported
            in the returned `skipped` list instead of crashing the ranking run.

    Returns a dict:
        eligible            -- ranked list, rank 1..N, sorted by hybrid_score DESC
        conditional         -- ranked list, rank 1..N, sorted independently
        excluded_count      -- number of INELIGIBLE + SUSPENDED records (never ranked)
        skipped             -- rankable records dropped due to missing/invalid
                                hybrid_score (list of {scholarship_id, eligibility_status, reason})
    """
    eligible_pool: List[Dict[str, Any]] = []
    conditional_pool: List[Dict[str, Any]] = []
    excluded_count = 0
    skipped: List[Dict[str, Any]] = []

    for record in hybrid_results:
        status = record.get("eligibility_status")

        if status in _EXCLUDED_STATUSES:
            # RuleMatcher's own decision -- never ranked, regardless of hybrid_score.
            excluded_count += 1
            continue

        if status not in _RANKABLE_STATUSES:
            # Unrecognized status: fail safe by excluding rather than guessing.
            skipped.append({
                "scholarship_id": record.get("scholarship_id"),
                "eligibility_status": status,
                "reason": f"unrecognized eligibility_status: {status!r}",
            })
            continue

        hybrid_score = record.get("hybrid_score")
        if not _is_number(hybrid_score):
            if strict:
                raise MissingHybridScoreError(
                    f"Scholarship {record.get('scholarship_id')!r} ({status}) has no usable "
                    f"hybrid_score ({hybrid_score!r}); cannot be ranked."
                )
            skipped.append({
                "scholarship_id": record.get("scholarship_id"),
                "eligibility_status": status,
                "reason": f"missing/invalid hybrid_score: {hybrid_score!r}",
            })
            continue

        (eligible_pool if status == "ELIGIBLE" else conditional_pool).append(record)

    # Sort in full BEFORE any top-N truncation, per spec.
    eligible_sorted = sorted(eligible_pool, key=_sort_key)
    conditional_sorted = sorted(conditional_pool, key=_sort_key)

    eligible_ranked = [_to_ranked_item(r, i + 1) for i, r in enumerate(eligible_sorted)]
    conditional_ranked = [_to_ranked_item(r, i + 1) for i, r in enumerate(conditional_sorted)]

    if top_n_eligible is not None:
        eligible_ranked = eligible_ranked[:top_n_eligible]
    if top_n_conditional is not None:
        conditional_ranked = conditional_ranked[:top_n_conditional]

    return {
        "eligible": eligible_ranked,
        "conditional": conditional_ranked,
        "excluded_count": excluded_count,
        "skipped": skipped,
    }
