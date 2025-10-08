"""
Enhanced, production-ready base provider adapter interface.

This module provides a robust, async-first abstract base class for all LLM
provider adapters. It includes built-in support for:
- Standardized response objects (InferenceResponse).
- Automatic, configurable retries with exponential backoff.
- Deep integration with observability for latency and token/cost tracking.
- Lazy client initialization and a simplified implementation interface.
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

from ..observability.latency import get_latency_tracker, LatencyComponent
from ..observability.tokens import get_token_tracker

# Define common, transient API errors that should be retried.
RETRYABLE_EXCEPTIONS = (
    IOError,
    # Add provider-specific exceptions here if needed, e.g.,
    # openai.RateLimitError, openai.APIConnectionError
)


@dataclass
class InferenceResponse:
    """A standardized data structure for all provider inference responses."""
    content: str
    model_name: str
    
    # Observability metrics
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    
    # For debugging and special cases
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    """
    Abstract base class for LLM provider adapters with built-in resilience
    and observability.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the provider adapter.

        Args:
            config: Provider-specific configuration, including model name, API keys, etc.
        """
        self.config = config or {}
        self.model_name = self.config.get("model", "default-model")
        self._client = None # Lazy initialization
        self.latency_tracker = get_latency_tracker()
        self.token_tracker = get_token_tracker()

    @property
    def client(self) -> Any:
        """Lazy initializer for the provider-specific API client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> Any:
        """
        Create and return an instance of the provider's API client.
        (e.g., `return openai.OpenAI(api_key=self.config['api_key'])`)
        """
        pass

    @abstractmethod
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """
        Core asynchronous inference logic to be implemented by subclasses.
        This method contains the actual provider API call.
        """
        pass

    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of an operation based on the provider's pricing model.
        """
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
        turn_id = kwargs.get("turn_id", f"turn_{time.time_ns()}")
        agent_name = kwargs.get("agent_name", "unknown_agent")
        
        start_time = time.monotonic()
        
        async with self.latency_tracker.measure(turn_id, LatencyComponent.INFERENCE):
            response = await self._infer(prompt, **kwargs)
        
        latency_ms = (time.monotonic() - start_time) * 1000
        
        # Standardize and enrich the response
        response.latency_ms = latency_ms
        response.cost_usd = self._calculate_cost(response.input_tokens, response.output_tokens)
        
        # Record metrics
        self.token_tracker.record_token_usage(
            agent_id=agent_name,
            turn_id=turn_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model_name=response.model_name,
            processing_time_seconds=latency_ms / 1000,
            prompt_text=prompt,
            response_text=response.content,
        )
        
        return response

    def infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """
        Synchronous wrapper for the async inference method.
        """
        # This provides a simple sync interface for environments that need it.
        return asyncio.run(self.async_infer(prompt, **kwargs))

    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming inference (optional). Subclasses should override this if they
        support streaming.
        """
        raise NotImplementedError(f"Streaming is not supported by the {self.__class__.__name__} provider.")
        yield {} # Required for async generator type hint

    def validate_config(self) -> bool:
        """
        Validate the provider configuration (e.g., check for API keys).
        Subclasses should override this with specific checks.
        """
        return "model" in self.config

    def get_provider_info(self) -> Dict[str, Any]:
        """Get standardized information about the provider."""
        return {
            "provider_name": self.__class__.__name__.replace("Adapter", ""),
            "configured_model": self.model_name,
            "supports_streaming": not self.stream_infer.__doc__.startswith("Streaming inference (optional)"),
            "required_config_keys": ["model"], # Subclasses can extend this
        }
