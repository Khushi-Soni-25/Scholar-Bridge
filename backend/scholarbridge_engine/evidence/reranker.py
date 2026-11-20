"""
ScholarBridge Configurable Evidence Reranker
============================================
Refines top candidate chunks using query-scheme intent matching,
query-section intent alignment, lexical density, and source trustworthiness.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .fusion import FusedCandidate
from .config import RERANKER_TOP_N, RERANKER_TYPE


class BaseReranker(ABC):
    """Abstract base class for all reranker implementations."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[FusedCandidate],
        top_n: int = RERANKER_TOP_N
    ) -> List[FusedCandidate]:
        """Reranks candidate evidence chunks and returns top_n items."""
        pass


class CrossFeatureReranker(BaseReranker):
    """
    Deterministic cross-feature reranker.
    Scores candidates by combining fusion rank, exact scheme intent alignment,
    section intent alignment, lexical n-gram density, and official source trust tier.
    """

    INTENT_SECTION_MAP = {
        "income": ["Eligibility", "Income & Domicile", "Income Limit", "Overview"],
        "ceiling": ["Eligibility", "Income & Domicile", "Income Limit", "Overview"],
        "percentage": ["Eligibility", "Academic & Course Criteria", "Academic Cutoff", "Overview"],
        "percentile": ["Eligibility", "Academic & Course Criteria", "Academic Cutoff", "Overview"],
        "marks": ["Eligibility", "Academic & Course Criteria", "Academic Cutoff", "Overview"],
        "cutoff": ["Eligibility", "Academic & Course Criteria", "Academic Cutoff", "Overview"],
        "document": ["Documents Required", "Checklist & Proofs", "Overview"],
        "certificate": ["Documents Required", "Checklist & Proofs", "Overview"],
        "proof": ["Documents Required", "Checklist & Proofs", "Eligibility", "Benefits"],
        "deadline": ["Important Dates", "Application Timeline", "Overview"],
        "last date": ["Important Dates", "Application Timeline", "Overview"],
        "portal": ["Overview", "Important Dates", "Scheme Details"],
        "website": ["Overview", "Important Dates", "Scheme Details"],
        "hostel": ["Eligibility", "Benefits", "Income & Domicile", "Overview"],
        "food bill": ["Benefits", "Financial Assistance Schedule", "Overview"],
        "benefit": ["Benefits", "Financial Assistance Schedule", "Overview"],
        "stipend": ["Benefits", "Financial Assistance Schedule", "Overview"],
        "fellowship": ["Benefits", "Financial Assistance Schedule", "Overview"],
        "amount": ["Benefits", "Financial Assistance Schedule", "Overview"],
        "grant": ["Benefits", "Financial Assistance Schedule", "Overview"],
        "fee": ["Benefits", "Financial Assistance Schedule", "Tuition Fee Waiver", "Overview"],
        "waiver": ["Benefits", "Financial Assistance Schedule", "Tuition Fee Waiver", "Overview"],
        "renewal": ["Terms & Conditions", "Renewal & Progression", "Important Dates", "Overview"],
        "status": ["Overview", "Terms & Conditions", "Scheme Details"],
        "eligibility": ["Eligibility", "Overview"]
    }

    def _compute_scheme_intent_boost(self, query: str, candidate: FusedCandidate) -> float:
        q_lower = query.lower()
        q_tokens = [t.strip("?,.!:\"'") for t in q_lower.split() if len(t.strip("?,.!:\"'")) >= 2]
        
        scheme_id_lower = candidate.chunk.scheme_id.lower()
        
        # 1. Exact Scheme ID in query (e.g. "MYSY", "BCK-10", "SHODH", "NTSE")
        if scheme_id_lower in q_tokens or scheme_id_lower in q_lower:
            return 0.50

        # Also check scheme ID with hyphens stripped (e.g. "bck10" vs "bck-10", "rfug" vs "rf-ug")
        if scheme_id_lower.replace("-", "") in q_lower.replace("-", "").split():
            return 0.50

        # 2. Document Title / Scheme Name match
        doc_title_lower = candidate.chunk.metadata_payload.get("document_title", "").lower()
        
        # Check high-value unique scheme names (e.g. "mukhyamantri yuva", "namo laxmi", "namo saraswati", "shodh", "vidyalaxmi", "adani", "reliance", "google generation", "adobe", "santoor", "amazon")
        for key_scheme in [
            "mukhyamantri yuva", "namo laxmi", "namo saraswati", "shodh", "vidyalaxmi",
            "adani", "reliance", "inspire", "pankh", "saraswati", "shramik puraskar",
            "swami vivekanand", "kanya", "google generation", "google", "adobe",
            "santoor", "amazon future engineer", "amazon", "pragati", "saksham", "kotak"
        ]:
            if key_scheme in q_lower and key_scheme in doc_title_lower:
                return 0.45

        # Word-level intersection with title (excluding generic category and stop words)
        title_tokens = set(doc_title_lower.split())
        generic_words = {
            "scholarship", "scheme", "gujarat", "government", "assistance",
            "what", "which", "women", "tech", "technology", "education",
            "students", "higher", "portal", "grant", "amount", "stipend"
        }
        matched_title_words = [
            t for t in q_tokens
            if t in title_tokens and len(t) >= 4 and t not in generic_words
        ]
        if matched_title_words:
            return min(0.40, 0.20 + 0.10 * len(matched_title_words))

        return 0.0

    def _compute_section_intent_boost(self, query: str, candidate: FusedCandidate) -> float:
        q_lower = query.lower()
        chunk = candidate.chunk
        boost = 0.0

        for keyword, target_sections in self.INTENT_SECTION_MAP.items():
            if keyword in q_lower:
                if chunk.section in target_sections or chunk.subsection in target_sections:
                    boost += 0.20
                    break
        return min(0.25, boost)

    def _compute_lexical_density(self, query: str, text: str) -> float:
        q_tokens = [t.lower().strip("?,.!") for t in query.split() if len(t.strip("?,.!")) >= 3]
        if not q_tokens:
            return 0.0
        text_lower = text.lower()
        matched = sum(1 for tok in q_tokens if tok in text_lower)
        return matched / len(q_tokens)

    def rerank(
        self,
        query: str,
        candidates: List[FusedCandidate],
        top_n: int = RERANKER_TOP_N
    ) -> List[FusedCandidate]:
        if not candidates:
            return []

        reranked: List[FusedCandidate] = []
        for cand in candidates:
            base_score = cand.fusion_score
            scheme_boost = self._compute_scheme_intent_boost(query, cand)
            intent_boost = self._compute_section_intent_boost(query, cand)
            lexical_density = self._compute_lexical_density(query, cand.chunk.text)
            
            # Official source bonus
            is_official = cand.chunk.metadata_payload.get("is_official", True)
            trust_bonus = 0.05 if is_official else 0.0

            # Composite rerank score (0.0 to 1.0)
            final_rerank_score = (
                0.25 * base_score +
                0.40 * scheme_boost +
                0.20 * intent_boost +
                0.10 * lexical_density +
                0.05 * trust_bonus
            )

            cand.rerank_score = round(min(1.0, final_rerank_score), 4)
            reranked.append(cand)

        reranked.sort(key=lambda x: (x.rerank_score if x.rerank_score is not None else x.fusion_score), reverse=True)
        return reranked[:top_n]


class NoOpReranker(BaseReranker):
    """Fallback pass-through reranker."""

    def rerank(
        self,
        query: str,
        candidates: List[FusedCandidate],
        top_n: int = RERANKER_TOP_N
    ) -> List[FusedCandidate]:
        return candidates[:top_n]


def get_reranker(reranker_type: Optional[str] = None) -> BaseReranker:
    """Factory returning configured reranker instance."""
    selected = reranker_type or RERANKER_TYPE
    if selected == "cross_feature":
        return CrossFeatureReranker()
    return NoOpReranker()
