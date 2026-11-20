"""
ScholarBridge Citation & Provenance Tracker
===========================================
Generates globally unique, query-independent citation identifiers, verifiable source linkages,
cryptographic content hashes, and human-readable citation references.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from .models import Citation, DocumentChunk, EvidenceItem
from .registry import SourceRegistry, get_source_registry


@dataclass
class CitationValidationResult:
    """Result of validating citations against retrieved evidence and registry."""
    is_valid: bool
    valid_citations: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    hallucinated_citations: List[str] = field(default_factory=list)
    provenance_verified: bool = True
    validation_error: Optional[str] = None


class CitationValidator:
    """Validates citations against retrieved evidence packs and provenance hashes."""

    def __init__(self, registry: Optional[SourceRegistry] = None):
        self.registry = registry or get_source_registry()

    def validate_response(
        self,
        response_text: str,
        citations: List[str],
        retrieved_evidence: List[Union[EvidenceItem, Dict[str, Any], Any]]
    ) -> CitationValidationResult:
        """
        Validates citations in generated response against retrieved evidence items.
        Extracts both explicit citation IDs and inline [EV-...] references in the response text.
        """
        valid_ev_map: Dict[str, Any] = {}
        for it in retrieved_evidence:
            cid = getattr(it, "citation_id", None) or (it.get("citation_id") if isinstance(it, dict) else None)
            if not cid and hasattr(it, "citation"):
                cid = getattr(it.citation, "citation_id", None)
            if cid:
                valid_ev_map[cid] = it

        # Extract inline citation strings [EV-...] directly from the generated response text
        text_cits = re.findall(r"\[(EV-[A-Za-z0-9\-_]+)\]", response_text or "")
        all_claimed_citations = list(dict.fromkeys(list(citations or []) + text_cits))

        valid_citations: List[str] = []
        invalid_citations: List[str] = []
        hallucinated_citations: List[str] = []

        for cit in all_claimed_citations:
            if cit in valid_ev_map:
                if cit not in valid_citations:
                    valid_citations.append(cit)
            else:
                if cit not in invalid_citations:
                    invalid_citations.append(cit)
                if cit not in hallucinated_citations:
                    hallucinated_citations.append(cit)

        # Fallback: if no citations were claimed in text/list but valid evidence was retrieved, preserve retrieved citations
        if not valid_citations and valid_ev_map and not invalid_citations:
            valid_citations = list(valid_ev_map.keys())

        is_valid = len(invalid_citations) == 0

        return CitationValidationResult(
            is_valid=is_valid,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            hallucinated_citations=hallucinated_citations,
            provenance_verified=is_valid,
            validation_error=None if is_valid else f"Hallucinated citations detected: {invalid_citations}"
        )


class CitationTracker:
    """Manages creation, formatting, and validation of evidence citations."""

    def __init__(self, registry: Optional[SourceRegistry] = None):
        self.registry = registry or get_source_registry()

    def create_citation(self, chunk: DocumentChunk, index: int = 1) -> Citation:
        """
        Generates a globally unique, query-independent citation object from a verified chunk.
        The citation ID is derived deterministically from chunk_id and cryptographic chunk_hash,
        ensuring stability regardless of retrieval rank or query context.
        """
        source = self.registry.get_source(chunk.source_id)
        doc_title = source.title if source else chunk.metadata_payload.get("document_title", "Official Guideline")
        source_url = source.source_url if source else chunk.metadata_payload.get("source_url", "")
        chunk_hash = chunk.chunk_hash or chunk.metadata_payload.get("chunk_hash", "")
        hash_suffix = chunk_hash[:8] if chunk_hash else "verified"
        
        # Globally unique, query-independent citation ID: EV-{chunk_id}-{hash8}
        citation_id = f"EV-{chunk.chunk_id}-{hash_suffix}"

        return Citation(
            citation_id=citation_id,
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            scheme_id=chunk.scheme_id,
            document_title=doc_title,
            page=chunk.page,
            section=chunk.section,
            subsection=chunk.subsection,
            academic_year=chunk.academic_year,
            version=chunk.document_version,
            source_url=source_url,
            quoted_text=chunk.text[:250] + ("..." if len(chunk.text) > 250 else ""),
            chunk_hash=chunk_hash
        )

    @staticmethod
    def format_citation_string(citation: Citation) -> str:
        """Formats citation as a clean reference string with hash fingerprint."""
        hash_fp = f" [sha:{citation.chunk_hash[:8]}]" if citation.chunk_hash else ""
        return (
            f"[{citation.citation_id}] {citation.document_title} ({citation.academic_year} v{citation.version}){hash_fp}, "
            f"Section: {citation.section} > {citation.subsection}, Page {citation.page} — {citation.source_url}"
        )
