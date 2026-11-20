"""
ScholarBridge BM25 Keyword Retrieval Index
==========================================
Provides high-precision keyword retrieval over normalized chunks with
custom tokenization for currency figures, percentage cutoffs, and scheme codes.
"""

import re
from typing import List, Tuple, Optional, Dict
from rank_bm25 import BM25Plus
from .models import DocumentChunk, SearchFilter
from .config import BM25_K1, BM25_B


class BM25KeywordIndex:
    """Inverted index implementing BM25Plus for exact scholarship term search."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None
        self._chunk_lookup: Dict[str, DocumentChunk] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Scholarship-domain aware tokenizer.
        Preserves currency symbols, percentage figures, hyphens in codes, and alphanumeric terms.
        """
        clean = text.lower().replace("₹", "rs ").replace("rs.", "rs ").replace("rs ", "rs ")
        tokens = re.findall(r"[a-z0-9_\-\.\,\%]+", clean)
        processed: List[str] = []
        for t in tokens:
            t_clean = t.strip(".-_")
            if t_clean:
                processed.append(t_clean)
                if "," in t_clean:
                    processed.append(t_clean.replace(",", ""))
        return processed

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Builds or replaces the BM25 inverted index."""
        self.chunks = list(chunks)
        self._chunk_lookup = {c.chunk_id: c for c in chunks}
        self.corpus_tokens = [self.tokenize(c.text) for c in chunks]
        
        if self.corpus_tokens:
            self.bm25 = BM25Plus(self.corpus_tokens, k1=self.k1, b=self.b)
        else:
            self.bm25 = None

    def search_bm25(
        self,
        query: str,
        search_filter: Optional[SearchFilter] = None,
        top_k: int = 30
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches BM25 index with optional metadata filtering."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0

        candidates: List[Tuple[DocumentChunk, float]] = []
        for idx, score in enumerate(raw_scores):
            if score <= 0.0:
                continue

            chunk = self.chunks[idx]
            
            # Apply deterministic metadata pre-filters
            if search_filter:
                if search_filter.scheme_ids and chunk.scheme_id not in search_filter.scheme_ids:
                    continue
                if search_filter.academic_year and chunk.academic_year != search_filter.academic_year:
                    continue
                if search_filter.section and chunk.section.lower() != search_filter.section.lower():
                    continue

            # Normalized BM25 score in [0.0, 1.0]
            norm_score = min(1.0, float(score / max_score))
            candidates.append((chunk, norm_score))

        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def count(self) -> int:
        return len(self.chunks)
