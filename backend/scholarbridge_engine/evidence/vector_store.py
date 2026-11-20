"""
ScholarBridge Qdrant Vector Store Manager
=========================================
Manages collection lifecycle, payload schemas, dense indexing, and
structured metadata filtering with the Qdrant vector database.
"""

import uuid
from typing import List, Tuple, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from .models import DocumentChunk, SearchFilter
from .config import (
    QDRANT_URL, QDRANT_COLLECTION_NAME, QDRANT_PATH,
    EMBEDDING_DIMENSIONS
)


# Global client cache to prevent lock conflicts when multiple instances access the same local path
_CLIENT_CACHE: Dict[str, QdrantClient] = {}


def _get_or_create_qdrant_client(url: Optional[str], path: Optional[str]) -> QdrantClient:
    key = str(url or path or ":memory:")
    if key not in _CLIENT_CACHE:
        if url == ":memory:":
            _CLIENT_CACHE[key] = QdrantClient(":memory:")
        elif url and url.startswith(("http://", "https://", "grpc://")):
            _CLIENT_CACHE[key] = QdrantClient(url=url)
        elif path:
            _CLIENT_CACHE[key] = QdrantClient(path=path)
        elif url:
            _CLIENT_CACHE[key] = QdrantClient(url=url)
        else:
            _CLIENT_CACHE[key] = QdrantClient(":memory:")
    return _CLIENT_CACHE[key]


class QdrantVectorStore:
    """Manages Qdrant vector database indexing and dense vector search."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimensions: int = EMBEDDING_DIMENSIONS
    ):
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.dimensions = dimensions
        target_url = url or QDRANT_URL
        target_path = QDRANT_PATH if (not target_url or target_url != ":memory:") else None

        self.client = _get_or_create_qdrant_client(target_url, target_path)

        self._ensure_collection()
        self._chunk_lookup: Dict[str, DocumentChunk] = {}

    def _ensure_collection(self):
        """Creates the Qdrant evidence collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimensions,
                    distance=qmodels.Distance.COSINE
                )
            )

    def _build_qdrant_filter(self, search_filter: Optional[SearchFilter]) -> Optional[qmodels.Filter]:
        """Translates high-level SearchFilter into native Qdrant filter conditions."""
        if not search_filter:
            return None

        must_conditions: List[qmodels.FieldCondition] = []

        # 1. Scheme IDs Filter
        if search_filter.scheme_ids:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="scheme_id",
                    match=qmodels.MatchAny(any=search_filter.scheme_ids)
                )
            )

        # 2. Academic Year Filter
        if search_filter.academic_year:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="academic_year",
                    match=qmodels.MatchValue(value=search_filter.academic_year)
                )
            )

        # 3. Section Filter
        if search_filter.section:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="section",
                    match=qmodels.MatchValue(value=search_filter.section)
                )
            )

        # 4. Jurisdiction / State Filter
        if search_filter.state:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="jurisdiction",
                    match=qmodels.MatchValue(value=search_filter.state)
                )
            )

        if not must_conditions:
            return None

        return qmodels.Filter(must=must_conditions)

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Upserts document chunks and their dense embeddings into Qdrant."""
        if not chunks:
            return

        points: List[qmodels.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            self._chunk_lookup[chunk.chunk_id] = chunk
            # Deterministic point UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            payload = {
                "chunk_id": chunk.chunk_id,
                "scheme_id": chunk.scheme_id,
                "source_id": chunk.source_id,
                "academic_year": chunk.academic_year,
                "document_version": chunk.document_version,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "page": chunk.page,
                "text": chunk.text,
                "token_count": chunk.token_count,
                **chunk.metadata_payload
            }

            points.append(qmodels.PointStruct(
                id=point_id,
                vector=emb,
                payload=payload
            ))

        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_dense(
        self,
        query_vector: List[float],
        search_filter: Optional[SearchFilter] = None,
        top_k: int = 30
    ) -> List[Tuple[DocumentChunk, float]]:
        """Performs dense vector cosine search with metadata pre-filtering."""
        q_filter = self._build_qdrant_filter(search_filter)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=top_k
        )

        results: List[Tuple[DocumentChunk, float]] = []
        for hit in res.points:
            chunk_id = hit.payload.get("chunk_id")
            chunk = self._chunk_lookup.get(chunk_id)
            if not chunk:
                # Reconstruct chunk from payload
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    scheme_id=hit.payload.get("scheme_id", ""),
                    source_id=hit.payload.get("source_id", ""),
                    document_version=hit.payload.get("document_version", "1.0"),
                    academic_year=hit.payload.get("academic_year", ""),
                    section=hit.payload.get("section", ""),
                    subsection=hit.payload.get("subsection", ""),
                    page=hit.payload.get("page", 1),
                    text=hit.payload.get("text", ""),
                    chunk_index=hit.payload.get("chunk_index", 0),
                    token_count=hit.payload.get("token_count", 0),
                    metadata_payload=hit.payload
                )
                self._chunk_lookup[chunk_id] = chunk

            # Cosine similarity score normalized between 0.0 and 1.0
            score = max(0.0, min(1.0, float(hit.score)))
            results.append((chunk, score))

        return results

    def count(self) -> int:
        """Returns total number of points in collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    def clear(self):
        """Clears the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        self._chunk_lookup.clear()
