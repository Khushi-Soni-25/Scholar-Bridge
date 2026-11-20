"""
ScholarBridge Concrete LLM Provider Adapters
============================================
Implements:
1. MockLLMProvider: Fast, deterministic, network-free mock provider for test suites.
2. HttpLLMProvider: OpenAI-compatible HTTP REST provider (e.g. OpenAI, vLLM, Ollama, LiteLLM).
3. Factory function `get_llm_provider()`.
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel

from .base import LLMProvider
from .exceptions import (
    LLMProviderError, MalformedLLMResponseError
)

logger = logging.getLogger(__name__)


class MockLLMProvider(LLMProvider):
    """
    Deterministic, offline Mock LLM Provider for unit testing and CI/CD.
    Extracts deterministic facts and evidence IDs from the grounded prompt to synthesize
    consistent, schema-compliant JSON explanations.
    """

    def __init__(self, canned_responses: Optional[List[Dict[str, Any]]] = None, model_name: str = "mock-deterministic-v1"):
        self._canned_responses = list(canned_responses) if canned_responses else []
        self._model_name = model_name
        self._call_history: List[Dict[str, Any]] = []

    def get_provider_name(self) -> str:
        return "mock"

    def get_model_name(self) -> str:
        return self._model_name

    def set_canned_response(self, response: Dict[str, Any]):
        """Queues a specific canned response for the next generate() call."""
        self._canned_responses.append(response)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Generates a grounded response or returns a queued canned response."""
        self._call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature
        })

        # 1. Check if a canned response is queued (for adversarial/edge-case tests)
        if self._canned_responses:
            canned = self._canned_responses.pop(0)
            if isinstance(canned, Exception):
                raise canned
            return canned

        # 2. Extract Evidence Citation IDs from prompt
        evidence_ids = re.findall(r"\[(EV-[A-Za-z0-9\-_]+)\]", prompt)
        unique_ev_ids = list(dict.fromkeys(evidence_ids))

        # 3. Detect Status from Prompt
        status_match = re.search(r"Authoritative Status:\s*([A-Z_]+)", prompt)
        status = status_match.group(1).upper() if status_match else "ELIGIBLE"

        # 4. Detect Scheme Name from Prompt
        scheme_match = re.search(r"Scholarship Name:\s*([^\n]+)", prompt)
        scheme_name = scheme_match.group(1).strip() if scheme_match else "Scholarship"

        # 5. Extract passed/failed/conditional rules
        passed_rules = re.findall(r"\[PASSED\]\s*([^:\n]+):\s*([^\n]+)", prompt)
        failed_rules = re.findall(r"\[FAILED\]\s*([^:\n]+):\s*([^\n]+)", prompt)
        conditional_rules = re.findall(r"\[CONDITIONAL\]\s*([^:\n]+):\s*([^\n]+)", prompt)

        # 6. Synthesize Grounded Explanations based on Authoritative Status
        if status == "ELIGIBLE":
            answer = f"You are eligible for {scheme_name} because your profile satisfies all required criteria."
            explanation_points = [
                f"Your {field.replace('_', ' ')} meets the criteria: {reason}"
                for field, reason in passed_rules
            ] or [f"Your profile satisfies all eligibility rules for {scheme_name}."]
            reasoning_summary = "All academic, financial, and demographic requirements are satisfied as verified in official guidelines."
            warnings = ["Ensure all information submitted matches your official government certificates."]

        elif status == "INELIGIBLE":
            failed_reasons = "; ".join([f"{f}: {r}" for f, r in failed_rules]) or "one or more criteria were not met"
            answer = f"You are currently not eligible for {scheme_name} because {failed_reasons}."
            explanation_points = [
                f"Failed Criterion: {field.replace('_', ' ')} — {reason}"
                for field, reason in failed_rules
            ] or ["Your profile does not satisfy the mandatory criteria."]
            reasoning_summary = f"Eligibility evaluation failed on {len(failed_rules)} rule(s) based on authoritative scheme guidelines."
            warnings = ["Government scholarship eligibility criteria are strictly enforced and cannot be overridden."]

        elif status == "CONDITIONAL":
            cond_reasons = "; ".join([f"{f}: {r}" for f, r in conditional_rules]) or "additional details are required"
            answer = f"You are conditionally eligible for {scheme_name}. {cond_reasons}."
            explanation_points = [
                f"Conditional Requirement: {field.replace('_', ' ')} — {reason}"
                for field, reason in conditional_rules
            ] or ["Additional information is required to confirm full eligibility."]
            reasoning_summary = "Your profile is conditionally matched pending submission or verification of required fields."
            warnings = ["Conditional scholarships require additional proof before final approval."]

        else: # SUSPENDED or DISCONTINUED
            answer = f"{scheme_name} is currently suspended or inactive for the current cycle."
            explanation_points = ["This scheme is not accepting new applications."]
            reasoning_summary = "Scheme status in official government portal is marked SUSPENDED."
            warnings = ["Do not submit applications to suspended schemes."]

        # If no evidence was retrieved in prompt
        if not unique_ev_ids and "No official evidence retrieved" in prompt:
            warnings.append("Note: Detailed official evidence text could not be retrieved from the repository.")

        return {
            "answer": answer,
            "explanation_points": explanation_points,
            "reasoning_summary": reasoning_summary,
            "warnings": warnings,
            "citations": unique_ev_ids
        }


class GeminiLLMProvider(LLMProvider):
    """
    Google Gemini REST provider for ScholarBridge's production grounded explanation layer.
    Uses Google Generative Language API with native JSON response mode.

    Configured via environment variables:
    - GEMINI_API_KEY / GOOGLE_API_KEY / SCHOLARBRIDGE_GEMINI_API_KEY
    - SCHOLARBRIDGE_GEMINI_MODEL (default: gemini-2.5-flash)
    - SCHOLARBRIDGE_GEMINI_TIMEOUT (default: 30 seconds)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("SCHOLARBRIDGE_GEMINI_API_KEY")
            or os.environ.get("SCHOLARBRIDGE_LLM_API_KEY", "")
        )
        self._model_name = (
            model_name
            or os.environ.get("SCHOLARBRIDGE_GEMINI_MODEL")
            or os.environ.get("SCHOLARBRIDGE_LLM_MODEL")
            or "gemini-2.5-flash"
        )
        self._timeout = float(timeout or os.environ.get("SCHOLARBRIDGE_GEMINI_TIMEOUT", "30"))

    def get_provider_name(self) -> str:
        return "gemini"

    def get_model_name(self) -> str:
        return self._model_name

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
        Sends structured generateContent request to Gemini REST API with JSON mode.
        """
        if not self._api_key:
            raise LLMProviderError(
                "Gemini API key is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env.local. "
                "For local testing without an API key, set SCHOLARBRIDGE_LLM_PROVIDER=mock or SCHOLARBRIDGE_LLM_PROVIDER=ollama."
            )

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self._api_key}"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                
                candidates = resp_data.get("candidates", [])
                if not candidates:
                    raise MalformedLLMResponseError("Gemini returned no candidates in response.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise MalformedLLMResponseError("Gemini candidate contains no parts.")

                raw_text = parts[0].get("text", "")
                if not raw_text:
                    raise MalformedLLMResponseError("Gemini returned empty text.")

                # Clean markdown codeblocks if present
                clean_text = raw_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                return json.loads(clean_text)

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            msg = f"Gemini API returned HTTP {e.code}: {e.reason}. {error_body}"
            logger.error(f"Gemini API error: {msg}")
            raise LLMProviderError(msg) from e
        except json.JSONDecodeError as e:
            logger.error(f"Gemini output is not valid JSON: {e}")
            raise MalformedLLMResponseError(f"Gemini output is not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"GeminiLLMProvider error: {e}")
            raise LLMProviderError(f"Gemini LLM Provider request failed: {e}") from e


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI Chat Completions REST provider for production grounded explanations.
    Uses official OpenAI API or OpenAI-compatible gateways.

    Configured via environment variables:
    - OPENAI_API_KEY / SCHOLARBRIDGE_OPENAI_API_KEY / SCHOLARBRIDGE_LLM_API_KEY
    - SCHOLARBRIDGE_OPENAI_MODEL (default: gpt-4o-mini)
    - SCHOLARBRIDGE_OPENAI_BASE_URL (default: https://api.openai.com/v1/chat/completions)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self._base_url = (
            base_url
            or os.environ.get("SCHOLARBRIDGE_OPENAI_BASE_URL")
            or os.environ.get("SCHOLARBRIDGE_LLM_URL", "https://api.openai.com/v1/chat/completions")
        )
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SCHOLARBRIDGE_OPENAI_API_KEY")
            or os.environ.get("SCHOLARBRIDGE_LLM_API_KEY", "")
        )
        self._model_name = (
            model_name
            or os.environ.get("SCHOLARBRIDGE_OPENAI_MODEL")
            or os.environ.get("SCHOLARBRIDGE_LLM_MODEL", "gpt-4o-mini")
        )
        self._timeout = float(timeout or os.environ.get("SCHOLARBRIDGE_OPENAI_TIMEOUT", "30"))

    def get_provider_name(self) -> str:
        return "openai"

    def get_model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Sends structured request to OpenAI chat completions endpoint."""
        if not self._api_key:
            raise LLMProviderError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY in your environment or .env.local. "
                "For local testing without an API key, set SCHOLARBRIDGE_LLM_PROVIDER=mock."
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }

        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                self._base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choices = resp_data.get("choices", [])
                if not choices:
                    raise MalformedLLMResponseError("OpenAI returned no choices.")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise MalformedLLMResponseError("OpenAI returned empty message content.")
                return json.loads(content)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            msg = f"OpenAI API returned HTTP {e.code}: {e.reason}. {error_body}"
            logger.error(f"OpenAI API error: {msg}")
            raise LLMProviderError(msg) from e
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI output is not valid JSON: {e}")
            raise MalformedLLMResponseError(f"OpenAI output is not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"OpenAILLMProvider generation failed: {e}")
            raise LLMProviderError(f"OpenAI LLM Provider request failed: {e}") from e


class HttpLLMProvider(OpenAILLMProvider):
    """Alias for OpenAILLMProvider supporting generic OpenAI-compatible REST endpoints."""
    def get_provider_name(self) -> str:
        return "http_rest"


class OllamaLLMProvider(LLMProvider):
    """
    Local Ollama provider for local offline development.
    Communicates with local Ollama runtime without requiring external API keys.

    Configured via environment variables:
    - SCHOLARBRIDGE_OLLAMA_URL (default: http://localhost:11434)
    - SCHOLARBRIDGE_OLLAMA_MODEL (default: llama3.2, fallback: llama3)
    - SCHOLARBRIDGE_OLLAMA_TIMEOUT (default: 60 seconds)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        raw_url = base_url or os.environ.get("SCHOLARBRIDGE_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        if raw_url.endswith(("/api/chat", "/v1/chat/completions")):
            raw_url = raw_url.rsplit("/", 2)[0] if "/v1/" in raw_url else raw_url.rsplit("/", 1)[0]
        self._base_url = raw_url
        self._model_name = (
            model_name
            or os.environ.get("SCHOLARBRIDGE_OLLAMA_MODEL")
            or os.environ.get("SCHOLARBRIDGE_LLM_MODEL")
            or "llama3.2"
        )
        self._timeout = float(timeout or os.environ.get("SCHOLARBRIDGE_OLLAMA_TIMEOUT", "60"))

    def get_provider_name(self) -> str:
        return "ollama"

    def get_model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Sends structured chat request to Ollama /api/chat endpoint with JSON format enforcement."""
        endpoint = f"{self._base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                message_content = resp_data.get("message", {}).get("content", "")
                if not message_content:
                    raise MalformedLLMResponseError("Ollama returned an empty response message.")

                clean_content = message_content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                elif clean_content.startswith("```"):
                    clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()

                return json.loads(clean_content)

        except urllib.error.URLError as e:
            msg = (
                f"Cannot connect to Ollama at '{self._base_url}'. "
                f"Ensure Ollama is running (`ollama serve`) and model '{self._model_name}' is available (`ollama pull {self._model_name}`). "
                f"For testing without Ollama, set SCHOLARBRIDGE_LLM_PROVIDER=mock."
            )
            logger.error(f"Ollama connection error: {msg} (details: {e})")
            raise LLMProviderError(msg) from e
        except json.JSONDecodeError as e:
            logger.error(f"Ollama output is not valid JSON: {e}")
            raise MalformedLLMResponseError(f"Ollama output is not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"OllamaLLMProvider generation error: {e}")
            raise LLMProviderError(f"Ollama LLM Provider error: {e}") from e


def get_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """
    Factory creating configured LLMProvider instance.

    Provider Resolution:
    1. Explicit provider_type argument or SCHOLARBRIDGE_LLM_PROVIDER env var
    2. 'gemini' / 'google' -> GeminiLLMProvider (Production Default)
    3. 'openai' / 'http'   -> OpenAILLMProvider / HttpLLMProvider
    4. 'ollama' / 'local'  -> OllamaLLMProvider (Local Development)
    5. 'mock' / 'test'     -> MockLLMProvider (CI / Unit Tests)
    """
    raw_type = provider_type or os.environ.get("SCHOLARBRIDGE_LLM_PROVIDER", "").strip().lower()

    if not raw_type:
        # Auto-detect from available API keys or default to Gemini production provider
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("SCHOLARBRIDGE_GEMINI_API_KEY"):
            return GeminiLLMProvider()
        elif os.environ.get("OPENAI_API_KEY") or os.environ.get("SCHOLARBRIDGE_OPENAI_API_KEY"):
            return OpenAILLMProvider()
        else:
            return GeminiLLMProvider()

    if raw_type in ["gemini", "google"]:
        return GeminiLLMProvider()
    elif raw_type in ["openai"]:
        return OpenAILLMProvider()
    elif raw_type in ["http", "rest", "vllm", "litellm"]:
        return HttpLLMProvider()
    elif raw_type in ["ollama", "local", "llama"]:
        return OllamaLLMProvider()
    elif raw_type in ["mock", "fake", "test", "deterministic"]:
        return MockLLMProvider()
    else:
        logger.warning(f"Unknown LLM provider '{raw_type}', defaulting to GeminiLLMProvider.")
        return GeminiLLMProvider()
