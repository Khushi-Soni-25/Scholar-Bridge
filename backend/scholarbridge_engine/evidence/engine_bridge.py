"""
ScholarBridge Engine ↔ Evidence Integration Bridge
==================================================
Binds the Phase 1 deterministic engine with the Phase 2 RAG & Evidence Layer.
Enriches eligibility and recommendation results with authoritative Evidence Packs
while preserving 100% of the deterministic engine's decision authority.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..models import StudentProfile, EngineOutput, RecommendationItem, MatchResult
from ..engine import ScholarBridgeEngine
from .models import EvidencePack
from .retriever import HybridEvidenceRetriever, get_evidence_retriever


@dataclass
class EnrichedRecommendation:
    """A recommendation item enriched with authoritative official evidence."""
    item: RecommendationItem
    evidence_pack: EvidencePack
    citations_summary: List[str] = field(default_factory=list)


@dataclass
class EnrichedEngineOutput:
    """Complete output with deterministic decisions AND authoritative citations."""
    engine_output: EngineOutput
    enriched_recommendations: List[EnrichedRecommendation] = field(default_factory=list)
    overall_evidence_summary: Dict[str, Any] = field(default_factory=dict)


class EngineEvidenceBridge:
    """Bridge service connecting deterministic decisions with retrieved evidence."""

    def __init__(
        self,
        engine: Optional[ScholarBridgeEngine] = None,
        retriever: Optional[HybridEvidenceRetriever] = None
    ):
        self.engine = engine or ScholarBridgeEngine()
        self.retriever = retriever or get_evidence_retriever()

    def recommend_with_evidence(
        self,
        profile: StudentProfile,
        evidence_per_scheme: int = 3
    ) -> EnrichedEngineOutput:
        """
        1. Executes deterministic Phase 1 eligibility & ranking (authoritative).
        2. Retrieves official evidence packs for each top recommendation.
        3. Returns enriched output with traceable citations.
        """
        # Step 1: Deterministic Engine Execution
        base_output = self.engine.recommend(profile)

        # Step 2: Retrieve Evidence for Top Ranked Recommendations
        enriched_items: List[EnrichedRecommendation] = []
        for rec in base_output.ranked_recommendations:
            scheme_code = rec.scholarship.scheme_code
            
            # Formulate query based on student profile context
            query = f"{scheme_code} eligibility income {profile.annual_family_income} category {profile.social_category} documents"
            
            evidence_pack = self.retriever.retrieve_for_scheme(
                scheme_id=scheme_code,
                query=query,
                top_k=evidence_per_scheme
            )

            citation_summaries = [
                f"[{ev.citation.citation_id}] {ev.citation.document_title} (p. {ev.page}, {ev.section}): {ev.citation.source_url}"
                for ev in evidence_pack.evidence
            ]

            enriched_items.append(EnrichedRecommendation(
                item=rec,
                evidence_pack=evidence_pack,
                citations_summary=citation_summaries
            ))

        return EnrichedEngineOutput(
            engine_output=base_output,
            enriched_recommendations=enriched_items,
            overall_evidence_summary={
                "total_recommendations_with_evidence": len(enriched_items),
                "evidence_provider": "ScholarBridge Phase 2 RAG Layer",
                "authority_model": "Engine Decides. RAG Proves. LLM Explains."
            }
        )
