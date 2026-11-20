"""
ScholarBridge LLM Provider Base Interface
=========================================
Vendor-agnostic abstraction for language model inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract interface for LLM execution."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes generation against the model and returns a parsed dictionary
        matching the requested response_schema structure.
        
        Raises:
            LLMProviderError: On connection/network/API failure.
            MalformedLLMResponseError: If response cannot be parsed as JSON.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns provider identifier, e.g. 'mock', 'http_openai', 'http_gemini'."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns model identifier, e.g. 'mock-deterministic-v1', 'gpt-4o', 'gemini-1.5-pro'."""
        pass
