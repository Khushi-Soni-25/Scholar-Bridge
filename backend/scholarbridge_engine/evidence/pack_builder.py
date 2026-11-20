"""
ScholarBridge Standardized Evidence Pack Builder
================================================
Constructs standardized EvidencePack contracts with traceable citations,
score breakdowns, applied filters, and diagnostic metadata.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .models import EvidencePack, EvidenceItem, SearchFilter, RetrievalDiagnostics
from .fusion import FusedCandidate
from .citations import CitationTracker


class EvidencePackBuilder:
    """Assembles ranked candidates and citations into an EvidencePack."""

    def __init__(self, citation_tracker: Optional[CitationTracker] = None):
        self.citation_tracker = citation_tracker or CitationTracker()

    def build_pack(
        self,
        query: str,
        ranked_candidates: List[FusedCandidate],
        search_filter: Optional[SearchFilter] = None,
        diagnostics: Optional[RetrievalDiagnostics] = None
    ) -> EvidencePack:
        """Assembles a full standardized EvidencePack."""
        evidence_items: List[EvidenceItem] = []

        for rank, cand in enumerate(ranked_candidates, start=1):
            citation = self.citation_tracker.create_citation(cand.chunk, index=rank)
            
            final_score = cand.rerank_score if cand.rerank_score is not None else cand.fusion_score
            evidence_items.append(EvidenceItem(
                citation=citation,
                scheme_id=cand.chunk.scheme_id,
                section=cand.chunk.section,
                subsection=cand.chunk.subsection,
                page=cand.chunk.page,
                text=cand.chunk.text,
                retrieval_score=final_score,
                dense_score=cand.dense_score,
                bm25_score=cand.bm25_score,
                fusion_score=cand.fusion_score,
                reranker_score=cand.rerank_score,
                rank=rank,
                retrieval_source=cand.retrieval_source
            ))

        filter_dict: Dict[str, Any] = {}
        if search_filter:
            if search_filter.scheme_ids:
                filter_dict["scheme_ids"] = search_filter.scheme_ids
            if search_filter.academic_year:
                filter_dict["academic_year"] = search_filter.academic_year
            if search_filter.section:
                filter_dict["section"] = search_filter.section
            if search_filter.category:
                filter_dict["category"] = search_filter.category
            if search_filter.state:
                filter_dict["state"] = search_filter.state

        retrieval_meta = {
            "total_evidence_returned": len(evidence_items),
            "retrieval_pipeline": "Dense(Qdrant) + Sparse(BM25) -> RRF Fusion -> CrossFeature Rerank",
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }
        if diagnostics:
            retrieval_meta.update({
                "dense_candidates": diagnostics.dense_candidates_count,
                "bm25_candidates": diagnostics.bm25_candidates_count,
                "fused_candidates": diagnostics.fused_candidates_count,
                "reranked_candidates": diagnostics.reranked_candidates_count,
                "execution_time_ms": round(diagnostics.execution_time_ms, 2)
            })

        return EvidencePack(
            query=query,
            generated_at=datetime.now(timezone.utc).isoformat(),
            filters=filter_dict,
            evidence=evidence_items,
            retrieval_metadata=retrieval_meta,
            diagnostics=diagnostics
        )
