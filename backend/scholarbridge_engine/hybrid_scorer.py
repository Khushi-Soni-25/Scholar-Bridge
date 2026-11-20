"""
ScholarBridge Hybrid Scoring (Section 11)
==========================================
Combines three independent 0-100 signals into a single hybrid_score:

    1. rule_relevance_score  -- existing rule-engine relevance score
                                 (RecommendationScorer.final_score, unchanged)
    2. rf_confidence_score   -- Section 9 Random Forest distillation signal
    3. nlp_similarity_score  -- Section 10 NLP semantic similarity signal

CRITICAL INVARIANT: eligibility_status (ELIGIBLE / CONDITIONAL / INELIGIBLE /
SUSPENDED) is decided exclusively by RuleMatcher. This module NEVER reads,
writes, or derives an eligibility status -- it only produces an additional
ranking score alongside whatever status the rule engine already assigned.
Callers must never use hybrid_score to promote/demote eligibility.

Weights live in `scholarbridge_engine.config.HYBRID_SCORING_WEIGHTS` so they
can be tuned in one place instead of being hard-coded across the codebase.
"""

from typing import Any, Dict, List, Optional

from .config import HYBRID_SCORING_WEIGHTS
from .models import MatchResult

# Keys expected in HYBRID_SCORING_WEIGHTS / used throughout this module.
_WEIGHT_KEYS = ("rule_relevance", "rf_confidence", "nlp_similarity")


def _clamp_0_100(value: float) -> float:
    """Clamps a numeric score defensively into the documented 0-100 range."""
    return max(0.0, min(100.0, float(value)))


def _validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Ensures the weight dict has the three expected keys with numeric values.

    Does not force the weights to sum to 1.0: a caller experimenting with an
    intentionally scaled configuration is allowed to do so; this only guards
    against missing keys / non-numeric values that would otherwise crash the
    weighted sum below.
    """
    missing = [k for k in _WEIGHT_KEYS if k not in weights]
    if missing:
        raise ValueError(f"HYBRID_SCORING_WEIGHTS is missing required key(s): {missing}")
    for k in _WEIGHT_KEYS:
        if not isinstance(weights[k], (int, float)):
            raise ValueError(f"HYBRID_SCORING_WEIGHTS['{k}'] must be numeric, got {type(weights[k])!r}")
    return weights


def compute_hybrid_score(
    rule_relevance_score: Optional[float],
    rf_confidence_score: Optional[float],
    nlp_similarity_score: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Combines the three 0-100 component signals into a single hybrid_score.

    Any component may be `None` (e.g. rf_confidence_score is only computed
    for the ELIGIBLE/CONDITIONAL pool in Section 9). Missing components are
    handled safely: they contribute 0 to the weighted sum rather than raising
    an exception or silently reweighting the remaining components, so a
    missing signal is always visible in `missing_components` and the score
    stays fully deterministic and auditable.

    Returns a dict with:
        hybrid_score        -- float, clamped to [0, 100]
        missing_components  -- list of component names that were None
        weights_used         -- the weight dict actually applied
    """
    w = _validate_weights(dict(weights) if weights is not None else dict(HYBRID_SCORING_WEIGHTS))

    components = {
        "rule_relevance": rule_relevance_score,
        "rf_confidence": rf_confidence_score,
        "nlp_similarity": nlp_similarity_score,
    }

    missing_components: List[str] = [name for name, val in components.items() if val is None]

    weighted_sum = 0.0
    for name, val in components.items():
        contribution = _clamp_0_100(val) if val is not None else 0.0
        weighted_sum += w[name] * contribution

    hybrid_score = _clamp_0_100(weighted_sum)

    return {
        "hybrid_score": round(hybrid_score, 2),
        "missing_components": missing_components,
        "weights_used": w,
    }


def build_hybrid_result(
    match_result: MatchResult,
    rule_relevance_score: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Builds the full Section 11 hybrid result record for one scholarship.

    `eligibility_status` is read directly from `match_result.status` (the
    RuleMatcher's own decision) and is never recomputed or altered here.
    `rf_confidence_score` / `nlp_similarity_score` are read from
    `match_result.explanation_details`, where Sections 9 and 10 already
    attached them (they may be absent/None for some buckets, e.g.
    rf_confidence_score is only present for ELIGIBLE/CONDITIONAL matches).
    """
    details = match_result.explanation_details or {}
    rf_confidence_score = details.get("rf_confidence_score")
    nlp_similarity_score = details.get("nlp_similarity_score")

    hybrid = compute_hybrid_score(
        rule_relevance_score=rule_relevance_score,
        rf_confidence_score=rf_confidence_score,
        nlp_similarity_score=nlp_similarity_score,
        weights=weights,
    )

    return {
        "scholarship_id": match_result.scholarship.scholarship_id,
        "short_name": match_result.scholarship.short_name,
        "eligibility_status": match_result.status.value,  # RuleMatcher's own decision, untouched
        "rule_relevance_score": (
            round(float(rule_relevance_score), 2) if rule_relevance_score is not None else None
        ),
        "rf_confidence_score": (
            round(float(rf_confidence_score), 2) if rf_confidence_score is not None else None
        ),
        "nlp_similarity_score": (
            round(float(nlp_similarity_score), 2) if nlp_similarity_score is not None else None
        ),
        "hybrid_score": hybrid["hybrid_score"],
        "missing_components": hybrid["missing_components"],
    }
