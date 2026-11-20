"""
ScholarBridge Hybrid Retrieval Result Fusion
============================================
Combines dense vector and BM25 keyword retrieval rankings using
Reciprocal Rank Fusion (RRF) and weighted score normalization.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from .models import DocumentChunk
from .config import RRF_K_CONSTANT, DENSE_FUSION_WEIGHT, BM25_FUSION_WEIGHT, FUSION_METHOD


@dataclass
class FusedCandidate:
    """Represents a fused candidate chunk with provenance scores from both retrieval branches."""
    chunk: DocumentChunk
    fusion_score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    retrieval_source: str = "hybrid" # 'dense', 'bm25', 'hybrid'


class ResultFusion:
    """Merges and ranks candidate chunks from dense and keyword retrieval."""

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Tuple[DocumentChunk, float]],
        bm25_results: List[Tuple[DocumentChunk, float]],
        k: int = RRF_K_CONSTANT
    ) -> List[FusedCandidate]:
        """
        Combines rankings using standard Reciprocal Rank Fusion:
        RRF_Score(d) = sum_{m in models} 1 / (k + rank_m(d))
        """
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Process Dense Results
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores[cid] = score
            dense_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

        # 2. Process BM25 Results
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            bm25_scores[cid] = score
            bm25_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

        # 3. Assemble Fused Candidates
        fused_list: List[FusedCandidate] = []
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

        for cid, rrf_raw in rrf_scores.items():
            chunk = chunk_map[cid]
            has_dense = cid in dense_scores
            has_bm25 = cid in bm25_scores

            if has_dense and has_bm25:
                source = "hybrid"
            elif has_dense:
                source = "dense"
            else:
                source = "bm25"

            # Normalized fusion score to [0.0, 1.0]
            normalized_score = min(1.0, rrf_raw / max_rrf)

            fused_list.append(FusedCandidate(
                chunk=chunk,
                fusion_score=normalized_score,
                dense_score=dense_scores.get(cid),
                bm25_score=bm25_scores.get(cid),
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                retrieval_source=source
            ))

        # Sort descending by fusion score
        fused_list.sort(key=lambda x: x.fusion_score, reverse=True)
        return fused_list

    @staticmethod
    def weighted_score_fusion(
        dense_results: List[Tuple[DocumentChunk, float]],
        bm25_results: List[Tuple[DocumentChunk, float]],
        dense_weight: float = DENSE_FUSION_WEIGHT,
        bm25_weight: float = BM25_FUSION_WEIGHT
    ) -> List[FusedCandidate]:
        """Combines normalized scores using weighted linear combination."""
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for chunk, score in dense_results:
            chunk_map[chunk.chunk_id] = chunk
            dense_scores[chunk.chunk_id] = score

        for chunk, score in bm25_results:
            chunk_map[chunk.chunk_id] = chunk
            bm25_scores[chunk.chunk_id] = score

        fused_list: List[FusedCandidate] = []
        for cid, chunk in chunk_map.items():
            d_score = dense_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)
            
            has_d = cid in dense_scores
            has_b = cid in bm25_scores
            source = "hybrid" if (has_d and has_b) else ("dense" if has_d else "bm25")

            combined = (d_score * dense_weight) + (b_score * bm25_weight)
            fused_list.append(FusedCandidate(
                chunk=chunk,
                fusion_score=min(1.0, combined),
                dense_score=dense_scores.get(cid),
                bm25_score=bm25_scores.get(cid),
                retrieval_source=source
            ))

        fused_list.sort(key=lambda x: x.fusion_score, reverse=True)
        return fused_list

    def fuse(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        bm25_results: List[Tuple[DocumentChunk, float]],
        method: str = FUSION_METHOD
    ) -> List[FusedCandidate]:
        """Entrypoint for fusing dense and keyword retrieval results."""
        if method == "weighted_sum":
            return self.weighted_score_fusion(dense_results, bm25_results)
        return self.reciprocal_rank_fusion(dense_results, bm25_results)
