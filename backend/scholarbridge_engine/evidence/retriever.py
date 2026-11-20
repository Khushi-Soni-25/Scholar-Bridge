"""
ScholarBridge Unified Hybrid Evidence Retriever
================================================
Coordinates Qdrant dense vector search, BM25 keyword retrieval,
RRF fusion, section-intent reranking, validation, and EvidencePack assembly.
"""

import time
from typing import List, Optional, Dict, Any
from .models import EvidencePack, SearchFilter, RetrievalDiagnostics, DocumentChunk, EvidenceItem
from .registry import SourceRegistry, get_source_registry
from .ingestion import DocumentIngestionPipeline
from .chunker import StructureAwareChunker
from .embeddings import BaseEmbeddingModel, get_embedding_model
from .vector_store import QdrantVectorStore
from .bm25_index import BM25KeywordIndex
from .fusion import ResultFusion
from .reranker import BaseReranker, get_reranker
from .validator import EvidenceValidator
from .pack_builder import EvidencePackBuilder
from .config import (
    RETRIEVAL_TOP_K_DENSE, RETRIEVAL_TOP_K_BM25,
    RERANKER_TOP_N, DEFAULT_ACADEMIC_YEAR
)
from ..loader import KnowledgeBase, get_knowledge_base


class HybridEvidenceRetriever:
    """Unified facade for scholarship evidence ingestion, indexing, and hybrid retrieval."""

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_index: Optional[BM25KeywordIndex] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        reranker: Optional[BaseReranker] = None
    ):
        self.registry = registry or get_source_registry()
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or QdrantVectorStore()
        self.bm25_index = bm25_index or BM25KeywordIndex()
        self.fusion = ResultFusion()
        self.reranker = reranker or get_reranker()
        self.validator = EvidenceValidator(self.registry)
        self.pack_builder = EvidencePackBuilder()
        self.chunker = StructureAwareChunker()
        self.ingestion_pipeline = DocumentIngestionPipeline()
        self._is_indexed = False

    def index_knowledge_base(self, kb: Optional[KnowledgeBase] = None, force_reindex: bool = False) -> int:
        """
        Ingests and indexes all official scholarship sources from the knowledge base
        into both Qdrant vector database and BM25 keyword index.
        """
        active_kb = kb or get_knowledge_base()
        all_chunks: List[DocumentChunk] = []

        for s in active_kb.scholarships:
            # 1. Determine official vs private source
            is_official_source = s.government_level in ("State", "Central") or s.provider_type in ("State_Government", "Central_Government")

            # 2. Register Source in SourceRegistry (defer disk save until full batch completes)
            source = self.registry.register_source(
                scheme_id=s.scheme_code,
                title=f"{s.scholarship_name} Official Guidelines",
                source_url=s.official_source_url or s.official_website,
                raw_content=f"{s.scholarship_name} {s.benefit_description} {s.annual_income_limit_inr}",
                publisher=s.provider_name or "Government of Gujarat",
                academic_year=DEFAULT_ACADEMIC_YEAR,
                version="1.0",
                is_official=is_official_source,
                auto_save=False
            )

            # 2. Ingest & Normalize Document
            rules = active_kb.rules_by_scholarship.get(s.scholarship_id, [])
            bens = active_kb.benefits_by_scholarship.get(s.scholarship_id, [])
            docs = active_kb.docs_by_scholarship.get(s.scholarship_id, [])
            
            doc = self.ingestion_pipeline.build_normalized_document_from_kb(
                s, rules, bens, docs, source
            )

            # 3. Structure-Aware Chunking
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        # Batch persist the sources registry once after full KB loop
        self.registry.save_registry()

        # 4. Check if vector store is already populated (for persistent disk storage)
        existing_count = self.vector_store.count()
        if force_reindex or existing_count != len(all_chunks):
            # Generate Dense Embeddings & Index in Qdrant
            texts = [c.text for c in all_chunks]
            embeddings = self.embedding_model.embed_batch(texts)
            self.vector_store.index_chunks(all_chunks, embeddings)
        else:
            # Rehydrate chunk lookup for vector store
            for c in all_chunks:
                self.vector_store._chunk_lookup[c.chunk_id] = c

        # 5. Index in BM25 Keyword Index
        self.bm25_index.index_chunks(all_chunks)
        
        self._is_indexed = True
        return len(all_chunks)

    def search(
        self,
        query: str,
        search_filter: Optional[SearchFilter] = None,
        top_k: int = RERANKER_TOP_N
    ) -> EvidencePack:
        """
        Executes end-to-end hybrid retrieval:
        Query -> Dense Vector Search + BM25 Keyword Search -> RRF Fusion -> CrossFeature Rerank -> Validation -> EvidencePack.
        """
        start_time = time.perf_counter()
        diagnostics = RetrievalDiagnostics(query=query, filters_applied=vars(search_filter) if search_filter else {})

        # 1. Dense Vector Retrieval (Qdrant)
        query_vector = self.embedding_model.embed_text(query)
        dense_results = self.vector_store.search_dense(
            query_vector=query_vector,
            search_filter=search_filter,
            top_k=RETRIEVAL_TOP_K_DENSE
        )
        diagnostics.dense_candidates_count = len(dense_results)

        # 2. Keyword Retrieval (BM25)
        bm25_results = self.bm25_index.search_bm25(
            query=query,
            search_filter=search_filter,
            top_k=RETRIEVAL_TOP_K_BM25
        )
        diagnostics.bm25_candidates_count = len(bm25_results)

        # 3. Result Fusion (Reciprocal Rank Fusion)
        fused_candidates = self.fusion.fuse(dense_results, bm25_results)
        diagnostics.fused_candidates_count = len(fused_candidates)

        # 4. Evidence Validation
        valid_candidates = [
            c for c in fused_candidates
            if self.validator.validate_chunk(c.chunk)[0]
        ]

        # 5. Cross-Feature Reranking
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=valid_candidates,
            top_n=top_k
        )
        diagnostics.reranked_candidates_count = len(reranked_candidates)

        # 6. Assemble Standardized Evidence Pack
        exec_time = (time.perf_counter() - start_time) * 1000.0
        diagnostics.execution_time_ms = exec_time

        return self.pack_builder.build_pack(
            query=query,
            ranked_candidates=reranked_candidates,
            search_filter=search_filter,
            diagnostics=diagnostics
        )

    def retrieve_for_scheme(
        self, scheme_id: str, query: str = "", top_k: int = 5
    ) -> EvidencePack:
        """Retrieves authoritative evidence specifically for a single scholarship scheme."""
        filter_obj = SearchFilter(scheme_ids=[scheme_id])
        search_q = query.strip() if query.strip() else f"Eligibility, Benefits, and Documents for {scheme_id}"
        return self.search(query=search_q, search_filter=filter_obj, top_k=top_k)

    def retrieve(
        self, query: str, scheme_id: Optional[str] = None, top_k: int = 5
    ) -> List[EvidenceItem]:
        """Convenience method returning List[EvidenceItem] from search or retrieve_for_scheme."""
        if scheme_id:
            pack = self.retrieve_for_scheme(scheme_id=scheme_id, query=query, top_k=top_k)
        else:
            pack = self.search(query=query, top_k=top_k)
        return pack.evidence


# Singleton instance
_GLOBAL_RETRIEVER: Optional[HybridEvidenceRetriever] = None

def get_evidence_retriever() -> HybridEvidenceRetriever:
    """Returns cached global retriever, ensuring knowledge base is indexed."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridEvidenceRetriever()
        _GLOBAL_RETRIEVER.index_knowledge_base()
    return _GLOBAL_RETRIEVER
