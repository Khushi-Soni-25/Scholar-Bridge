"""
ScholarBridge Evidence Data Models
==================================
Defines typed models for official sources, structure-aware chunks with
cryptographic SHA-256 chunk hashes, traceable citations, and Evidence Packs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any


class SourceStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    NEEDS_REVIEW = "needs_review"


class SourceType(str, Enum):
    OFFICIAL_GUIDELINE = "official_guideline"
    GOVERNMENT_RESOLUTION = "government_resolution"
    PORTAL_POLICY = "portal_policy"
    OFFICIAL_NOTIFICATION = "official_notification"
    INSTITUTION_CIRCULAR = "institution_circular"
    CSR_POLICY = "csr_policy"


@dataclass
class SourceMetadata:
    """Represents a registered, version-controlled official scholarship document source."""
    source_id: str
    scheme_id: str
    title: str
    source_url: str
    source_type: str = SourceType.OFFICIAL_GUIDELINE.value
    publisher: str = "Government of Gujarat"
    jurisdiction: str = "Gujarat"
    academic_year: str = "2026-27"
    version: str = "1.0"
    effective_from: str = "2026-06-01"
    effective_until: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_hash: str = ""
    is_official: bool = True
    status: str = SourceStatus.ACTIVE.value
    notes: str = ""


@dataclass
class DocumentChunk:
    """Represents a structure-aware, section-preserved chunk of a scholarship document."""
    chunk_id: str
    scheme_id: str
    source_id: str
    document_version: str
    academic_year: str
    section: str                   # 'Eligibility', 'Benefits', 'Documents Required', 'Important Dates', 'Renewal', etc.
    subsection: str                # 'Income Limit', 'Academic Cutoff', 'Tuition Waiver', etc.
    page: int                      # Page number (1-indexed) or 1 for web documents
    text: str                      # Exact chunk text
    chunk_index: int               # Sequential index within document
    chunk_hash: str = ""           # Cryptographic SHA-256 hash of chunk content
    token_count: int = 0
    metadata_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Traceable, verifiable citation linking evidence to an official source."""
    citation_id: str
    chunk_id: str
    source_id: str
    scheme_id: str
    document_title: str
    page: int
    section: str
    subsection: str
    academic_year: str
    version: str
    source_url: str
    quoted_text: str
    chunk_hash: str = ""           # Cryptographic content hash verifying provenance


@dataclass
class SearchFilter:
    """Structured deterministic filter applied before/during retrieval."""
    scheme_ids: Optional[List[str]] = None
    academic_year: Optional[str] = None
    state: Optional[str] = None
    category: Optional[str] = None
    education_level: Optional[str] = None
    stream: Optional[str] = None
    section: Optional[str] = None
    is_official_only: bool = True


@dataclass
class EvidenceItem:
    """A validated, ranked evidence chunk with full provenance and score breakdown."""
    citation: Citation
    scheme_id: str
    section: str
    subsection: str
    page: int
    text: str
    retrieval_score: float         # Final score (0 - 1)
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fusion_score: Optional[float] = None
    reranker_score: Optional[float] = None
    rank: int = 0
    retrieval_source: str = "hybrid" # 'dense', 'bm25', 'hybrid'


@dataclass
class RetrievalDiagnostics:
    """Structured observability log for every retrieval execution."""
    query: str
    filters_applied: Dict[str, Any]
    dense_candidates_count: int = 0
    bm25_candidates_count: int = 0
    fused_candidates_count: int = 0
    reranked_candidates_count: int = 0
    execution_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvidencePack:
    """Standardized, LLM-independent Evidence Pack contract."""
    query: str
    generated_at: str
    filters: Dict[str, Any]
    evidence: List[EvidenceItem]
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Optional[RetrievalDiagnostics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts EvidencePack to a clean serializable dictionary."""
        return {
            "query": self.query,
            "generated_at": self.generated_at,
            "filters": self.filters,
            "evidence": [
                {
                    "citation_id": item.citation.citation_id,
                    "scheme_id": item.scheme_id,
                    "document_title": item.citation.document_title,
                    "source_id": item.citation.source_id,
                    "source_url": item.citation.source_url,
                    "academic_year": item.citation.academic_year,
                    "version": item.citation.version,
                    "chunk_hash": item.citation.chunk_hash,
                    "section": item.section,
                    "subsection": item.subsection,
                    "page": item.page,
                    "text": item.text,
                    "retrieval_score": round(item.retrieval_score, 4),
                    "rank": item.rank
                }
                for item in self.evidence
            ],
            "retrieval_metadata": self.retrieval_metadata
        }
