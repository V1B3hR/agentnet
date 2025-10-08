"""
Enhanced, production-ready base provider adapter interface.

This module provides a robust, async-first abstract base class for all LLM
provider adapters. It has observability (metrics, tracing, logging) built-in,
making all provider implementations automatically instrumented.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# --- Observability Imports ---
# The base class is now directly responsible for instrumentation.
from ..performance.latency import get_latency_tracker, LatencyComponent
from ..performance.tokens import get_token_tracker
from ..observability.tracing import get_global_tracer, trace_agent_operation
from ..observability.logging import get_correlation_logger

logger = get_correlation_logger("agentnet.providers")

# Define common, transient API errors that should be retried.
RETRYABLE_EXCEPTIONS = (IOError,)


@dataclass
class InferenceResponse:
    """A standardized data structure for all provider inference responses."""
    content: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    """
    Abstract base class for LLM provider adapters with built-in resilience
    and observability.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get("model", "default-model")
        self._client = None
        
        # ### INSTRUMENTATION ###
        # Observability components are initialized directly in the base class.
        self.latency_tracker = get_latency_tracker()
        self.token_tracker = get_token_tracker()
        self.tracer = get_global_tracer()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> Any:
        pass

    @abstractmethod
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        pass

    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True
    )
    async def async_infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """
        Public-facing async inference method with built-in retries and observability.
        """
        # ### INSTRUMENTATION ###
        # Context for logging and tracing is extracted from kwargs.
        turn_id = kwargs.get("turn_id", f"turn_{time.time_ns()}")
        agent_name = kwargs.get("agent_name", "unknown_agent")
        session_id = kwargs.get("session_id")
        
        logger.set_correlation_context(session_id=session_id, agent_name=agent_name, operation="provider_inference")
        
        start_time = time.monotonic()
        response = None
        
        try:
            # ### INSTRUMENTATION ###
            # OpenTelemetry tracing is wrapped around the core logic.
            with trace_agent_operation(agent_name, self.model_name, self.__class__.__name__, session_id) as span:
                span.set_attribute("prompt_length", len(prompt))

                # The latency tracker is used via its automated context manager.
                async with self.latency_tracker.measure(turn_id, LatencyComponent.INFERENCE):
                    response = await self._infer(prompt, **kwargs)

                latency_ms = (time.monotonic() - start_time) * 1000
                
                # Enrich the response with final metrics
                response.latency_ms = latency_ms
                response.cost_usd = self._calculate_cost(response.input_tokens, response.output_tokens)
                
                # ### INSTRUMENTATION ###
                # Accurate metrics are recorded using the standardized response object.
                self.token_tracker.record_token_usage(
                    agent_id=agent_name,
                    turn_id=turn_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    model_name=response.model_name,
                    processing_time_seconds=latency_ms / 1000,
                )
                
                # Update the trace with the final results.
                span.set_attribute("output_tokens", response.output_tokens)
                span.set_attribute("cost_usd", response.cost_usd)
                span.set_attribute("latency_ms", latency_ms)
                
                logger.log_agent_inference(
                    model_name=response.model_name,
                    provider_name=self.__class__.__name__,
                    total_tokens=response.input_tokens + response.output_tokens,
                    duration_ms=latency_ms,
                    cost_usd=response.cost_usd,
                )
                
                return response

        except Exception as e:
            logger.error(f"Inference failed after retries: {e}", error_type=type(e).__name__)
            # Here you could add metrics for failed calls if desired.
            raise
        finally:
            logger.clear_context()

    def infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """Synchronous wrapper for the async inference method."""
        return asyncio.run(self.async_infer(prompt, **kwargs))

    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError(f"Streaming is not supported by {self.__class__.__name__}.")
        yield {}

    def validate_config(self) -> bool:
        return "model" in self.config

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider_name": self.__class__.__name__.replace("Adapter", ""),
            "configured_model": self.model_name,
            "supports_streaming": not self.stream_infer.__doc__.startswith("Streaming is not supported"),
        }
