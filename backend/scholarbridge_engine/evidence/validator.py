"""
ScholarBridge Evidence Validation Guard
=======================================
Validates candidate evidence chunks before inclusion in Evidence Packs,
enforcing cryptographic SHA-256 chunk hashes, source registration, and academic-year isolation.
"""

import hashlib
from typing import List, Tuple, Optional
from .models import DocumentChunk
from .registry import SourceRegistry, get_source_registry


class EvidenceValidator:
    """Validates structural, cryptographic, and provenance integrity of retrieved evidence."""

    def __init__(self, registry: Optional[SourceRegistry] = None):
        self.registry = registry or get_source_registry()

    @staticmethod
    def verify_chunk_hash(chunk: DocumentChunk) -> bool:
        """
        Verifies that chunk content matches its claimed SHA-256 hash.
        Strict verification: Missing hash strictly returns False.
        """
        supplied_hash = chunk.chunk_hash or chunk.metadata_payload.get("chunk_hash")
        if not supplied_hash or not str(supplied_hash).strip():
            return False

        expected_hash = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        return supplied_hash == expected_hash

    def validate_chunk(self, chunk: DocumentChunk, require_active: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validates individual chunk against registered sources and cryptographic provenance rules.
        Returns (is_valid, error_reason).
        """
        # 1. Non-empty text check
        if not chunk.text or not chunk.text.strip():
            return False, f"Empty content in chunk {chunk.chunk_id}"

        # 2. Chunk ID validity
        if not chunk.chunk_id or len(chunk.chunk_id) < 5:
            return False, f"Malformed chunk ID: {chunk.chunk_id}"

        # 3. Cryptographic Chunk Hash Verification (Strict: missing or mismatched hash fails)
        if not self.verify_chunk_hash(chunk):
            supplied_hash = chunk.chunk_hash or chunk.metadata_payload.get("chunk_hash", "")
            if not supplied_hash:
                return False, f"Cryptographic integrity failure: chunk {chunk.chunk_id} is missing required SHA-256 chunk_hash"
            return False, f"Cryptographic integrity failure: chunk content does not match SHA-256 hash in {chunk.chunk_id}"

        # 4. Source registration check
        source = self.registry.get_source(chunk.source_id)
        if not source:
            return False, f"Orphan chunk {chunk.chunk_id}: source {chunk.source_id} not registered"

        # 5. Academic year consistency check
        if not chunk.academic_year or chunk.academic_year != source.academic_year:
            return False, f"Academic year mismatch in chunk {chunk.chunk_id} ({chunk.academic_year} vs source {source.academic_year})"

        # 6. Active / Superseded status check
        if require_active and source.status != "active":
            return False, f"Source {chunk.source_id} is {source.status} (superseded)"

        # 7. Page and Section completeness
        if chunk.page <= 0:
            return False, f"Invalid page number ({chunk.page}) in chunk {chunk.chunk_id}"
        if not chunk.section:
            return False, f"Missing section name in chunk {chunk.chunk_id}"

        return True, None

    def filter_valid_chunks(
        self, chunks: List[DocumentChunk], require_active: bool = True
    ) -> List[DocumentChunk]:
        """Filters a list of chunks, returning only validated, verified chunks."""
        valid_chunks: List[DocumentChunk] = []
        for c in chunks:
            is_valid, _ = self.validate_chunk(c, require_active=require_active)
            if is_valid:
                valid_chunks.append(c)
        return valid_chunks
