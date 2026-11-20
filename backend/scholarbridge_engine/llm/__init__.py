"""
ScholarBridge Phase 3 LLM Integration Layer
===========================================
Grounded explanation service adhering strictly to:
"The deterministic engine decides. RAG provides evidence. The LLM explains."
"""

from .base import LLMProvider
from .provider import MockLLMProvider, HttpLLMProvider, get_llm_provider
from .schemas import (
    ScholarshipResponse,
    ScholarshipRecommendationItem,
    DocumentItem,
    LLMGeneratedContent,
    GroundingMetadata
)
from .exceptions import (
    ScholarBridgeLLMError,
    LLMProviderError,
    LLMValidationError,
    GroundingError,
    EvidenceUnavailableError,
    MalformedLLMResponseError
)
from .service import LLMService

__all__ = [
    "LLMService",
    "LLMProvider",
    "MockLLMProvider",
    "HttpLLMProvider",
    "get_llm_provider",
    "ScholarshipResponse",
    "ScholarshipRecommendationItem",
    "DocumentItem",
    "LLMGeneratedContent",
    "GroundingMetadata",
    "ScholarBridgeLLMError",
    "LLMProviderError",
    "LLMValidationError",
    "GroundingError",
    "EvidenceUnavailableError",
    "MalformedLLMResponseError"
]
