"""
ScholarBridge Deterministic Intent Router
==========================================
Classifies incoming user requests into one of four supported workflow intents:
  1. RECOMMEND — Deterministic profile evaluation + multi-factor recommendation + RAG explanation
  2. SEARCH    — Factual inquiry over official scholarship documents via Hybrid RAG
  3. COMPARE   — Multi-scheme deterministic comparison and evidence synthesis
  4. EXPLAIN   — Grounded explanation of an authoritative eligibility decision

Invariants Enforced:
  - The router ONLY classifies workflow intent; it NEVER makes eligibility decisions.
  - Tightly constrained pattern/keyword matching prevents arbitrary agentic execution.
"""

import re
from typing import Dict, Any, Optional
from .state import WorkflowIntent, ScholarBridgeWorkflowState


class IntentRouter:
    """Deterministic, rule-based intent router for ScholarBridge workflows."""

    COMPARE_PATTERNS = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bvs\b",
        r"\bversus\b",
        r"\bdifference\b",
        r"\bbetter between\b",
        r"\bwhich is better\b",
    ]

    EXPLAIN_PATTERNS = [
        r"\bwhy am i\b",
        r"\bwhy was i\b",
        r"\bwhy not\b",
        r"\bwhy ineligible\b",
        r"\bwhy eligible\b",
        r"\bwhy rejected\b",
        r"\bexplain (my |the )?(eligibility|result|decision)\b",
        r"\breason for (ineligibility|rejection)\b",
        r"\bwhy failed\b",
    ]

    RECOMMEND_PATTERNS = [
        r"\brecommend\b",
        r"\bfind (me )?scholarships\b",
        r"\bwhich scholarship(s)? can i\b",
        r"\bwhat scholarship(s)? (can|should) i\b",
        r"\bsearch scholarships for me\b",
        r"\bbest scholarship(s)?\b",
        r"\bmatch(ing)? scholarship(s)?\b",
        r"\bsuggest scholarship(s)?\b",
        r"\bcheck (my )?eligibility\b",
        r"\bam i eligible for anything\b",
        r"\bwhat am i eligible for\b",
    ]

    SEARCH_PATTERNS = [
        r"\bwhat is the (amount|grant|stipend|income limit|deadline|portal|last date)\b",
        r"\bhow much\b",
        r"\bwhen is the deadline\b",
        r"\bwhat documents\b",
        r"\bwho (is eligible|can apply)\b",
        r"\bofficial (portal|website|link)\b",
        r"\bincome ceiling\b",
        r"\bgrant amount\b",
        r"\bfee waiver\b",
    ]

    @classmethod
    def classify(
        cls,
        user_query: Optional[str] = None,
        has_profile: bool = False,
        selected_scholarships: Optional[list] = None,
        intent_override: Optional[str] = None
    ) -> WorkflowIntent:
        """Classifies intent deterministically based on query text and context."""
        # 1. Respect validated explicit intent override
        if intent_override:
            clean_override = intent_override.strip().upper()
            if clean_override in WorkflowIntent.__members__:
                return WorkflowIntent(clean_override)

        q = (user_query or "").strip().lower()
        selected = selected_scholarships or []

        # 2. Check COMPARE patterns or multiple selected scholarships with comparison keywords
        if len(selected) >= 2:
            if not q or any(re.search(p, q) for p in cls.COMPARE_PATTERNS):
                return WorkflowIntent.COMPARE
        for pattern in cls.COMPARE_PATTERNS:
            if re.search(pattern, q):
                return WorkflowIntent.COMPARE

        # 3. Check EXPLAIN patterns ("Why am I ineligible?", "Explain why I qualified")
        for pattern in cls.EXPLAIN_PATTERNS:
            if re.search(pattern, q):
                return WorkflowIntent.EXPLAIN

        # 4. Check RECOMMEND patterns
        for pattern in cls.RECOMMEND_PATTERNS:
            if re.search(pattern, q):
                return WorkflowIntent.RECOMMEND

        # 5. Check SEARCH patterns (factual lookup)
        for pattern in cls.SEARCH_PATTERNS:
            if re.search(pattern, q):
                return WorkflowIntent.SEARCH

        # 6. Context-driven fallback
        if has_profile:
            return WorkflowIntent.RECOMMEND
        if q:
            return WorkflowIntent.SEARCH

        return WorkflowIntent.UNKNOWN


def route_intent(state: ScholarBridgeWorkflowState) -> Dict[str, Any]:
    """LangGraph node function that determines intent and writes it to state."""
    q = state.get("user_query", "")
    profile = state.get("student_profile")
    selected = state.get("selected_scholarships", [])
    override = state.get("intent_override")

    intent_enum = IntentRouter.classify(
        user_query=q,
        has_profile=bool(profile),
        selected_scholarships=selected,
        intent_override=override
    )

    return {
        "intent": intent_enum.value,
        "current_step": "route_intent",
    }
