"""
ScholarBridge LLM Layer Exceptions
==================================
Structured exception hierarchy for LLM providers, output validation,
grounding integrity, and evidence availability.
"""


class ScholarBridgeLLMError(Exception):
    """Base exception for all ScholarBridge LLM layer errors."""
    pass


class LLMProviderError(ScholarBridgeLLMError):
    """Raised when an external or mock LLM provider encounters an execution error."""
    pass


class LLMValidationError(ScholarBridgeLLMError):
    """Raised when LLM output violates schema contracts or type requirements."""
    pass


class GroundingError(ScholarBridgeLLMError):
    """Raised when LLM output contradicts deterministic engine output or invents citations."""
    pass


class EvidenceUnavailableError(ScholarBridgeLLMError):
    """Raised when authoritative evidence is strictly required but unavailable."""
    pass


class MalformedLLMResponseError(ScholarBridgeLLMError):
    """Raised when LLM returns invalid JSON or unparseable text."""
    pass
