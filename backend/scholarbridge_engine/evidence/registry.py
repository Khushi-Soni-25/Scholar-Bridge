"""
ScholarBridge Official Source Registry & Versioning
===================================================
Manages registered official scholarship guideline sources, cryptographic
hashes, semantic version history, academic-year isolation, and lifecycle states.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .models import SourceMetadata, SourceStatus, SourceType
from .config import EVIDENCE_DATA_DIR, DEFAULT_ACADEMIC_YEAR

logger = logging.getLogger(__name__)


def parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """
    Parses semantic version string into a numeric tuple for accurate comparison.
    E.g. '1.9' -> (1, 9), '1.10' -> (1, 10), '2.0.1' -> (2, 0, 1).
    """
    parts = re.findall(r"\d+", version_str)
    if parts:
        return tuple(int(p) for p in parts)
    return (0,)


class SourceRegistry:
    """Registry maintaining version-controlled, official scholarship sources."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (EVIDENCE_DATA_DIR / "sources_registry.json")
        self.sources: Dict[str, SourceMetadata] = {}
        self._load_registry()

    @staticmethod
    def compute_document_hash(content: str) -> str:
        """Computes SHA-256 hash of document text for duplicate and integrity checks."""
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def register_source(
        self,
        scheme_id: str,
        title: str,
        source_url: str,
        raw_content: str,
        source_type: str = SourceType.OFFICIAL_GUIDELINE.value,
        publisher: str = "Government of Gujarat",
        jurisdiction: str = "Gujarat",
        academic_year: str = DEFAULT_ACADEMIC_YEAR,
        version: str = "1.0",
        effective_from: str = "2026-06-01",
        is_official: bool = True,
        notes: str = "",
        auto_save: bool = True
    ) -> SourceMetadata:
        """Registers a new authoritative source or updates version history."""
        doc_hash = self.compute_document_hash(raw_content)
        source_id = f"{scheme_id}-{academic_year}-V{version.replace('.', '_')}"
        new_v_tuple = parse_version_tuple(version)
        
        # Check if identical document hash already exists for this scheme and year
        for existing in self.sources.values():
            if existing.scheme_id == scheme_id and existing.academic_year == academic_year:
                if existing.document_hash == doc_hash:
                    return existing

        # Mark previous active versions for this scheme/year as superseded using semantic version comparison
        for existing in self.sources.values():
            if existing.scheme_id == scheme_id and existing.academic_year == academic_year:
                if existing.status == SourceStatus.ACTIVE.value:
                    existing_v_tuple = parse_version_tuple(existing.version)
                    if existing_v_tuple < new_v_tuple:
                        existing.status = SourceStatus.SUPERSEDED.value
                        existing.effective_until = effective_from

        metadata = SourceMetadata(
            source_id=source_id,
            scheme_id=scheme_id,
            title=title,
            source_url=source_url,
            source_type=source_type,
            publisher=publisher,
            jurisdiction=jurisdiction,
            academic_year=academic_year,
            version=version,
            effective_from=effective_from,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            document_hash=doc_hash,
            is_official=is_official,
            status=SourceStatus.ACTIVE.value,
            notes=notes
        )
        
        self.sources[source_id] = metadata
        if auto_save:
            self.save_registry()
        return metadata

    def get_source(self, source_id: str) -> Optional[SourceMetadata]:
        """Retrieves source metadata by source_id."""
        return self.sources.get(source_id)

    def get_active_source_for_scheme(
        self, scheme_id: str, academic_year: str = DEFAULT_ACADEMIC_YEAR
    ) -> Optional[SourceMetadata]:
        """
        Retrieves the latest active source for a given scheme and academic year.
        Strict academic-year isolation: NEVER falls back to a different academic year.
        """
        candidates = [
            s for s in self.sources.values()
            if s.scheme_id == scheme_id and s.academic_year == academic_year and s.status == SourceStatus.ACTIVE.value
        ]
        if not candidates:
            return None
        # Sort by parsed semantic version tuple descending to guarantee highest version returned
        candidates.sort(key=lambda x: parse_version_tuple(x.version), reverse=True)
        return candidates[0]

    def get_version_history(
        self, scheme_id: str, academic_year: Optional[str] = None
    ) -> List[SourceMetadata]:
        """Returns the full version history for a given scheme and optional academic year."""
        history = [s for s in self.sources.values() if s.scheme_id == scheme_id]
        if academic_year:
            history = [s for s in history if s.academic_year == academic_year]
        history.sort(key=lambda x: parse_version_tuple(x.version))
        return history

    def list_sources(
        self,
        scheme_id: Optional[str] = None,
        academic_year: Optional[str] = None,
        active_only: bool = True
    ) -> List[SourceMetadata]:
        """Lists sources with optional filtering."""
        res = list(self.sources.values())
        if scheme_id:
            res = [s for s in res if s.scheme_id == scheme_id]
        if academic_year:
            res = [s for s in res if s.academic_year == academic_year]
        if active_only:
            res = [s for s in res if s.status == SourceStatus.ACTIVE.value]
        return res

    def save_registry(self):
        """Persists sources registry to disk."""
        if str(self.storage_path) == ":memory:":
            return
        data = {sid: vars(sm) for sid, sm in self.sources.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_registry(self):
        """Loads sources registry from disk if present."""
        if str(self.storage_path) == ":memory:":
            return
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, sm_dict in data.items():
                        self.sources[sid] = SourceMetadata(**sm_dict)
            except Exception as e:
                logger.warning(
                    f"Failed to load sources registry from {self.storage_path}: {e}. "
                    "Starting with empty in-memory registry."
                )


# Global singleton registry
_GLOBAL_REGISTRY: Optional[SourceRegistry] = None

def get_source_registry() -> SourceRegistry:
    """Returns the cached global source registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SourceRegistry()
    return _GLOBAL_REGISTRY
