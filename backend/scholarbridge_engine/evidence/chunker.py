"""
ScholarBridge Structure-Aware Chunking & Cryptographic Provenance
================================================================
Splits normalized documents along semantic section and table boundaries,
preserving tables, headers, numerical constraints, and computing SHA-256 chunk hashes.
"""

import hashlib
import re
from typing import List, Dict, Any
from .models import DocumentChunk, SourceMetadata
from .ingestion import NormalizedDocument, DocumentSection
from .config import DEFAULT_CHUNK_SIZE_TOKENS


class StructureAwareChunker:
    """Chunks documents along semantic section and table boundaries with hash provenance."""

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^A-Z0-9_]+", "_", text.upper().strip()).strip("_")

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def compute_chunk_hash(text: str) -> str:
        """Computes cryptographic SHA-256 hash of normalized chunk content."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def chunk_document(self, document: NormalizedDocument) -> List[DocumentChunk]:
        """Splits a normalized document into structure-aware, section-preserved chunks."""
        chunks: List[DocumentChunk] = []
        source: SourceMetadata = document.source_metadata
        chunk_idx = 0

        for sec in document.sections:
            sec_name = sec.section_name
            subsec_name = sec.subsection_name
            page = sec.page_number
            content = sec.content.strip()
            
            if not content:
                continue

            sec_slug = self._slugify(sec_name)
            est_tokens = self._estimate_token_count(content)
            
            if est_tokens <= DEFAULT_CHUNK_SIZE_TOKENS or sec.is_table:
                chunk_idx += 1
                chunk_id = f"{source.scheme_id}-{source.academic_year}-{sec_slug}-{chunk_idx:03d}"
                c_hash = self.compute_chunk_hash(content)
                
                payload = {
                    "chunk_id": chunk_id,
                    "chunk_hash": c_hash,
                    "scheme_id": source.scheme_id,
                    "source_id": source.source_id,
                    "document_title": source.title,
                    "source_url": source.source_url,
                    "academic_year": source.academic_year,
                    "document_version": source.version,
                    "section": sec_name,
                    "subsection": subsec_name,
                    "page": page,
                    "is_official": source.is_official,
                    "jurisdiction": source.jurisdiction,
                    "is_table": sec.is_table,
                    **sec.metadata
                }

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    scheme_id=source.scheme_id,
                    source_id=source.source_id,
                    document_version=source.version,
                    academic_year=source.academic_year,
                    section=sec_name,
                    subsection=subsec_name,
                    page=page,
                    text=content,
                    chunk_index=chunk_idx,
                    chunk_hash=c_hash,
                    token_count=est_tokens,
                    metadata_payload=payload
                ))
            else:
                lines = content.split("\n\n")
                current_buffer = []
                current_count = 0

                for line in lines:
                    line_count = self._estimate_token_count(line)
                    if current_count + line_count > DEFAULT_CHUNK_SIZE_TOKENS and current_buffer:
                        chunk_idx += 1
                        chunk_id = f"{source.scheme_id}-{source.academic_year}-{sec_slug}-{chunk_idx:03d}"
                        chunk_text = "\n\n".join(current_buffer)
                        c_hash = self.compute_chunk_hash(chunk_text)
                        
                        payload = {
                            "chunk_id": chunk_id,
                            "chunk_hash": c_hash,
                            "scheme_id": source.scheme_id,
                            "source_id": source.source_id,
                            "document_title": source.title,
                            "source_url": source.source_url,
                            "academic_year": source.academic_year,
                            "document_version": source.version,
                            "section": sec_name,
                            "subsection": subsec_name,
                            "page": page,
                            "is_official": source.is_official,
                            "jurisdiction": source.jurisdiction,
                            "is_table": False,
                            **sec.metadata
                        }

                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            scheme_id=source.scheme_id,
                            source_id=source.source_id,
                            document_version=source.version,
                            academic_year=source.academic_year,
                            section=sec_name,
                            subsection=subsec_name,
                            page=page,
                            text=chunk_text,
                            chunk_index=chunk_idx,
                            chunk_hash=c_hash,
                            token_count=current_count,
                            metadata_payload=payload
                        ))
                        current_buffer = [line]
                        current_count = line_count
                    else:
                        current_buffer.append(line)
                        current_count += line_count

                if current_buffer:
                    chunk_idx += 1
                    chunk_id = f"{source.scheme_id}-{source.academic_year}-{sec_slug}-{chunk_idx:03d}"
                    chunk_text = "\n\n".join(current_buffer)
                    c_hash = self.compute_chunk_hash(chunk_text)
                    payload = {
                        "chunk_id": chunk_id,
                        "chunk_hash": c_hash,
                        "scheme_id": source.scheme_id,
                        "source_id": source.source_id,
                        "document_title": source.title,
                        "source_url": source.source_url,
                        "academic_year": source.academic_year,
                        "document_version": source.version,
                        "section": sec_name,
                        "subsection": subsec_name,
                        "page": page,
                        "is_official": source.is_official,
                        "jurisdiction": source.jurisdiction,
                        "is_table": False,
                        **sec.metadata
                    }
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        scheme_id=source.scheme_id,
                        source_id=source.source_id,
                        document_version=source.version,
                        academic_year=source.academic_year,
                        section=sec_name,
                        subsection=subsec_name,
                        page=page,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        chunk_hash=c_hash,
                        token_count=current_count,
                        metadata_payload=payload
                    ))

        return chunks
