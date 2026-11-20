"""
ScholarBridge Eligibility & Recommendation Engine (Unified Facade)
==================================================================
Ties together Knowledge Base loading, 3-state deterministic rule matching,
RF confidence, NLP similarity, hybrid scoring, ranking, final recommendation
assembly, explainability, and document aggregation.

Step 10 integration note
-------------------------
Sections 11-14 (hybrid_scorer, ranker, recommender, explainer) previously
existed only as notebook cells and standalone modules exercised via the
notebook/tests -- this production facade now calls them directly, in the
same order the notebook demonstrates:

    StudentProfile -> KnowledgeBase -> RuleMatcher -> RF -> NLP
        -> hybrid_scorer -> ranker -> recommender -> explainer

RuleMatcher (Stage 1, unchanged) remains the SOLE authority for
eligibility_status. RF confidence, NLP similarity, and hybrid_score are
ranking/relevance signals only and can NEVER move a scholarship between
ELIGIBLE / CONDITIONAL / INELIGIBLE / SUSPENDED -- ranker.py's own
exclusion of INELIGIBLE/SUSPENDED from both ranked buckets is the
enforcement point, and this facade never second-guesses it.

Backward compatibility: `EngineOutput.ranked_recommendations` /
`.conditional_recommendations` / `.excluded_scholarships` /
`.eligible_count` / `.conditional_count` / `.ineligible_count` /
`.total_scholarships_evaluated` are produced EXACTLY as before (same
`RecommendationScorer.score_and_rank` call, same values) so existing
consumers -- notably `scholarbridge_engine.evidence.engine_bridge.
EngineEvidenceBridge`, which reads `base_output.ranked_recommendations`
and iterates `rec.scholarship.scheme_code` -- keep working unmodified.
Sections 11-14 output is exposed via NEW, additive `EngineOutput` fields
(`hybrid_results`, `ranking`, `final_recommendations`, `explanations`).

RF/NLP note: Section 9 (Random Forest) and Section 10 (NLP similarity)
exist today only as notebook cells (trained/run against a live MongoDB
knowledge base + a downloaded sentence-transformers model) -- there was no
existing standalone production RF/NLP module to call, and this integration
step does not add a new trained model (that would be new ML development,
out of scope here). Instead, `ScholarBridgeEngine` accepts optional
`rf_confidence_provider` / `nlp_similarity_provider` callables so a real
model can be plugged in without touching this file again. Without one
supplied, a clearly-labeled deterministic STUB is used (mirroring the same
stub already used and documented in `run_pipeline_mock_validation.py`) --
NOT a trained RandomForestClassifier, NOT real sentence-transformers
embeddings. hybrid_scorer.compute_hybrid_score already handles a missing/
None component safely, so this is a graceful degradation, not a crash risk.
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import HYBRID_SCORING_WEIGHTS
from .deadline_tracker import DeadlineTracker
from .doc_aggregator import DocumentAggregator
from .explainer import ExplainabilityEngine, build_final_recommendation_explanations
from .hybrid_scorer import build_hybrid_result
from .loader import KnowledgeBase, get_knowledge_base
from .matcher import RuleMatcher
from .models import (
    EligibilityStatus, EngineOutput, MatchResult, RecommendationItem, StudentProfile,
)
from .mongo_loader import MongoUnavailableError, get_mongo_knowledge_base
from .ranker import rank_scholarships
from .recommender import build_final_recommendations
from .scorer import RecommendationScorer
from .validators import ValidationError, validate_student_profile

RFConfidenceProvider = Callable[[MatchResult, StudentProfile], Optional[float]]
NLPSimilarityProvider = Callable[[MatchResult, StudentProfile], Optional[float]]


def default_rf_confidence_provider(match: MatchResult, profile: StudentProfile) -> float:
    """STUB, NOT a trained RandomForestClassifier. Deterministic 0-100
    stand-in with the same shape/contract Section 9 hands to Section 11
    (rf_confidence_score attached only to ELIGIBLE/CONDITIONAL matches --
    the caller decides which matches to call this for). Plug in a real
    provider via ScholarBridgeEngine(rf_confidence_provider=...) once a
    trained model is available as a production module."""
    h = int(hashlib.sha256(f"rf:{match.scholarship.scholarship_id}".encode()).hexdigest(), 16)
    return round((h % 10000) / 100.0, 2)


def default_nlp_similarity_provider(match: MatchResult, profile: StudentProfile) -> float:
    """STUB, NOT a real sentence-transformers embedding model. Deterministic
    0-100 stand-in with the same shape/contract Section 10 hands to Section
    11 (nlp_similarity_score attached to ALL four eligibility buckets).
    Plug in a real provider via ScholarBridgeEngine(nlp_similarity_provider=...)
    once a downloaded embedding model is available as a production module."""
    h = int(hashlib.sha256(f"nlp:{match.scholarship.scholarship_id}".encode()).hexdigest(), 16)
    return round((h % 10000) / 100.0, 2)


class ScholarBridgeEngine:
    """Production facade for ScholarBridge recommendation pipeline."""

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        rf_confidence_provider: Optional[RFConfidenceProvider] = None,
        nlp_similarity_provider: Optional[NLPSimilarityProvider] = None,
        use_mongo: bool = False,
        mongo_uri: Optional[str] = None,
        mongo_db_name: Optional[str] = None,
    ):
        """
        kb_dir / custom_weights: unchanged from prior versions (CSV knowledge
            base directory override, RecommendationScorer benefit/confidence/
            urgency/specificity weight override).
        hybrid_weights: Section 11 weights (rule_relevance/rf_confidence/
            nlp_similarity). Defaults to config.HYBRID_SCORING_WEIGHTS
            (0.60/0.20/0.20) -- pass a dict to override, same contract as
            hybrid_scorer.compute_hybrid_score's `weights` parameter.
        rf_confidence_provider / nlp_similarity_provider: optional callables
            `(match_result, profile) -> Optional[float]`. Default to the
            documented STUBs above when not supplied.
        use_mongo / mongo_uri / mongo_db_name: if use_mongo=True, the engine
            uses scholarbridge_engine.mongo_loader.get_mongo_knowledge_base()
            instead of the CSV-backed KnowledgeBase. Defaults to False (CSV),
            so existing zero-argument callers (e.g. EngineEvidenceBridge's
            `ScholarBridgeEngine()`) are completely unaffected. If Mongo is
            requested but unavailable (pymongo missing, no URI/DB configured,
            or connection failure), raises MongoUnavailableError rather than
            silently falling back -- callers that want automatic CSV fallback
            should use `ScholarBridgeEngine.from_preferred_source` instead.
        """
        if use_mongo:
            self.kb: KnowledgeBase = get_mongo_knowledge_base(mongo_uri, mongo_db_name)
        else:
            self.kb = get_knowledge_base(kb_dir)

        self.matcher = RuleMatcher()
        self.scorer = RecommendationScorer()
        self.explainer = ExplainabilityEngine()
        self.doc_aggregator = DocumentAggregator()
        self.deadline_tracker = DeadlineTracker()
        self.weights = custom_weights
        self.hybrid_weights = hybrid_weights or HYBRID_SCORING_WEIGHTS
        self.rf_confidence_provider = rf_confidence_provider or default_rf_confidence_provider
        self.nlp_similarity_provider = nlp_similarity_provider or default_nlp_similarity_provider

    @classmethod
    def from_preferred_source(
        cls,
        mongo_uri: Optional[str] = None,
        mongo_db_name: Optional[str] = None,
        **kwargs,
    ) -> "ScholarBridgeEngine":
        """Convenience constructor: tries MongoDB first (if a URI/DB name is
        given), falls back to the CSV-backed KnowledgeBase on
        MongoUnavailableError. Preserves the CSV path as the safety net --
        never crashes just because Mongo isn't reachable."""
        if mongo_uri and mongo_db_name:
            try:
                return cls(use_mongo=True, mongo_uri=mongo_uri, mongo_db_name=mongo_db_name, **kwargs)
            except MongoUnavailableError:
                pass  # fall through to CSV
        return cls(**kwargs)

    def recommend(
        self,
        profile: StudentProfile,
        top_n_eligible: Optional[int] = None,
        top_n_conditional: Optional[int] = None,
    ) -> EngineOutput:
        """Executes the full evaluation pipeline for a student profile:
        RuleMatcher -> RF -> NLP -> hybrid_scorer -> ranker -> recommender -> explainer.

        `top_n_eligible` / `top_n_conditional` (both default None = no
        limit) are passed straight through to Section 13's
        `recommender.build_final_recommendations` -- they only affect
        `EngineOutput.final_recommendations`, never `ranked_recommendations`
        / `conditional_recommendations` (which remain the full, unmodified
        Stage 2 output, exactly as before, for backward compatibility).
        """

        # 1. Validate & Normalize Student Profile (unchanged)
        is_valid, errors = validate_student_profile(profile)
        if not is_valid:
            raise ValidationError(f"Invalid student profile: {'; '.join(errors)}")

        eligible_matches: List[MatchResult] = []
        conditional_matches: List[MatchResult] = []
        excluded_matches: List[MatchResult] = []

        # 2. Stage 1: Deterministic Rule Matching (unchanged -- RuleMatcher is
        #    the sole eligibility authority; nothing below this stage can
        #    alter match_res.status).
        for scholarship in self.kb.scholarships:
            sid = scholarship.scholarship_id
            rules = self.kb.rules_by_scholarship.get(sid, [])

            match_res = self.matcher.evaluate_scholarship(scholarship, rules, profile)

            # Attach related auxiliary data (unchanged)
            match_res.documents = self.kb.docs_by_scholarship.get(sid, [])
            match_res.benefits = self.kb.benefits_by_scholarship.get(sid, [])

            # Attach explainability and timeline details (unchanged -- this
            # is the pre-existing per-match explanation dict/timeline info;
            # Sections 11/14 below ADD keys into the same explanation_details
            # dict rather than replacing it).
            match_res.explanation_details = self.explainer.format_match_explanation(match_res, profile)
            match_res.timeline_info = self.deadline_tracker.get_timeline_info(scholarship)

            if match_res.status == EligibilityStatus.ELIGIBLE:
                eligible_matches.append(match_res)
            elif match_res.status == EligibilityStatus.CONDITIONAL:
                conditional_matches.append(match_res)
            else:
                excluded_matches.append(match_res)

        all_matches = eligible_matches + conditional_matches + excluded_matches

        # 3. Section 9 (RF confidence): attached ONLY to ELIGIBLE/CONDITIONAL,
        #    matching Section 9's documented contract (RF is not computed for
        #    excluded matches in the existing design either).
        for m in eligible_matches + conditional_matches:
            m.explanation_details["rf_confidence_score"] = self.rf_confidence_provider(m, profile)

        # 4. Section 10 (NLP similarity): attached to ALL FOUR eligibility
        #    buckets, matching Section 10's documented contract. A high NLP
        #    score on an INELIGIBLE/SUSPENDED match is intentional -- it must
        #    never influence eligibility_status (see Step 9 Persona D).
        for m in all_matches:
            m.explanation_details["nlp_similarity_score"] = self.nlp_similarity_provider(m, profile)

        # 5. Stage 2 (unchanged): Multi-Factor Recommendation Scoring &
        #    Ranking via the existing, unmodified RecommendationScorer. This
        #    IS the "existing rule/relevance score from the current engine"
        #    that Section 11 combines into hybrid_score -- reused here
        #    exactly as engine.py always did, never duplicated.
        ranked_eligible: List[RecommendationItem] = self.scorer.score_and_rank(eligible_matches, profile, self.weights)
        ranked_conditional: List[RecommendationItem] = self.scorer.score_and_rank(conditional_matches, profile, self.weights)

        # Ensure timeline and explanation are propagated to recommendation items (unchanged)
        for item in ranked_eligible + ranked_conditional:
            item.timeline_info = item.match_result.timeline_info
            item.explanation_details = item.match_result.explanation_details
            # Section 11 input: rule_relevance_score = the existing scorer's
            # final_score, attached back onto the shared match_result so it's
            # visible to hybrid_scorer below (mirrors the notebook exactly).
            item.match_result.explanation_details["rule_relevance_score"] = item.final_score

        # 6. Section 11 (hybrid_scorer, REUSED, not duplicated): compute
        #    hybrid_score for EVERY scholarship in all four buckets.
        #    rule_relevance_score is None for INELIGIBLE/SUSPENDED (the
        #    existing engine never scores excluded matches with
        #    RecommendationScorer either) -- hybrid_scorer treats that
        #    missing component safely (0 contribution), never crashes.
        hybrid_results: List[Dict[str, Any]] = []
        for m in all_matches:
            rule_relevance = m.explanation_details.get("rule_relevance_score")
            record = build_hybrid_result(m, rule_relevance_score=rule_relevance, weights=self.hybrid_weights)
            m.explanation_details["hybrid_score"] = record["hybrid_score"]
            hybrid_results.append(record)

        # 7. Section 12 (ranker, REUSED, not duplicated): INELIGIBLE/SUSPENDED
        #    are excluded here -- unconditionally, regardless of hybrid_score.
        #    This is the actual enforcement point for "RF/NLP/hybrid can
        #    never override RuleMatcher".
        ranking: Dict[str, Any] = rank_scholarships(
            hybrid_results, top_n_eligible=None, top_n_conditional=None,
        )

        # 8. Section 13 (recommender, REUSED, not duplicated): final
        #    recommendation payload, enriched with KB metadata and the
        #    deadline-derived application_status (kept distinct from
        #    eligibility_status -- see recommender._application_status_from_timeline).
        timeline_lookup = {m.scholarship.scholarship_id: m.timeline_info for m in all_matches}
        final_recommendations: Dict[str, Any] = build_final_recommendations(
            ranking, self.kb.scholarships_by_id,
            top_n_eligible=top_n_eligible, top_n_conditional=top_n_conditional,
            timeline_lookup=timeline_lookup,
        )

        # 9. Section 14A (explainer, REUSED, not duplicated): structured
        #    explanations for the final recommendations only. Recalculates
        #    nothing; matched/missing factors come straight from RuleMatcher's
        #    own passed_rules/conditional_rules.
        match_lookup = {m.scholarship.scholarship_id: m for m in all_matches}
        explanations: Dict[str, Any] = build_final_recommendation_explanations(final_recommendations, match_lookup)

        # 10. Stage 3 (unchanged): Document Checklist Aggregation
        all_positive_recs = ranked_eligible + ranked_conditional
        unified_docs = self.doc_aggregator.build_unified_checklist(
            all_positive_recs, self.kb.docs_by_scholarship
        )

        return EngineOutput(
            student_profile=profile,
            total_scholarships_evaluated=len(self.kb.scholarships),
            eligible_count=len(ranked_eligible),
            conditional_count=len(ranked_conditional),
            ineligible_count=len(excluded_matches),
            ranked_recommendations=ranked_eligible,
            conditional_recommendations=ranked_conditional,
            excluded_scholarships=excluded_matches,
            unified_checklist=unified_docs,
            hybrid_results=hybrid_results,
            ranking=ranking,
            final_recommendations=final_recommendations,
            explanations=explanations,
        )
