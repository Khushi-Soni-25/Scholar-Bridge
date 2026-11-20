"""
ScholarBridge LangChain Observability Callbacks
================================================
Lightweight callback handler capturing tool executions, latencies,
model parameters, and grounding validation state without heavy external tracing infra.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from .schemas import OrchestrationTrace

logger = logging.getLogger(__name__)


class ScholarBridgeCallbackHandler(BaseCallbackHandler):
    """
    Lightweight orchestration tracing callback.
    Captures stage-by-stage execution metrics:
      - Deterministic Engine latency
      - Hybrid RAG retrieval latency
      - CrossFeature reranker latency
      - LLM generation latency
      - Citation validation outcome
    """

    def __init__(self, trace: Optional[OrchestrationTrace] = None):
        super().__init__()
        self.trace = trace or OrchestrationTrace()
        self._start_time: float = time.perf_counter()
        self._tool_starts: Dict[str, float] = {}
        self._llm_start: float = 0.0

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_starts[tool_name] = time.perf_counter()
        logger.debug(f"[Trace {self.trace.request_id}] Tool started: {tool_name}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any
    ) -> None:
        # Check if we can identify tool by kwargs or most recent tool
        # Default fallback to duration recording
        now = time.perf_counter()
        for tool_name, start_t in list(self._tool_starts.items()):
            duration_ms = (now - start_t) * 1000
            if "recommend" in tool_name.lower() or "engine" in tool_name.lower():
                self.trace.deterministic_engine_ms = duration_ms
            elif "evidence" in tool_name.lower() or "search" in tool_name.lower() or "rag" in tool_name.lower():
                self.trace.hybrid_rag_ms = duration_ms
            elif "validate" in tool_name.lower() or "citation" in tool_name.lower():
                self.trace.citation_validation_ms = duration_ms
            logger.debug(f"[Trace {self.trace.request_id}] Tool ended: {tool_name} ({duration_ms:.2f}ms)")
            del self._tool_starts[tool_name]

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        self._llm_start = time.perf_counter()
        logger.debug(f"[Trace {self.trace.request_id}] LLM generation started")

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any
    ) -> None:
        if self._llm_start > 0:
            self.trace.llm_generation_ms = (time.perf_counter() - self._llm_start) * 1000
            logger.debug(f"[Trace {self.trace.request_id}] LLM generation finished ({self.trace.llm_generation_ms:.2f}ms)")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any
    ) -> None:
        err_msg = f"Tool execution failed: {error}"
        self.trace.errors.append(err_msg)
        logger.error(f"[Trace {self.trace.request_id}] {err_msg}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any
    ) -> None:
        err_msg = f"LLM generation failed: {error}"
        self.trace.errors.append(err_msg)
        logger.error(f"[Trace {self.trace.request_id}] {err_msg}")

    def finalize(self) -> OrchestrationTrace:
        """Finalizes total pipeline duration and returns the trace."""
        self.trace.total_pipeline_ms = (time.perf_counter() - self._start_time) * 1000
        return self.trace