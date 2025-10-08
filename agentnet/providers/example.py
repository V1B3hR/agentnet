"""
Enhanced example provider implementation for testing and development.

This script serves as a best-practice template for creating new provider adapters,
demonstrating the correct implementation of the async-first interface,
standardized responses, cost calculation, and optional streaming.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class ExampleEngine(ProviderAdapter):
    """
    Example engine for testing, demonstrations, and as a template for new providers.

    This adapter simulates the behavior of a real LLM provider, including latency,
    token counting, and streaming, while inheriting automatic retries and
    observability from the base class.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the example engine.

        Args:
            config: Configuration dictionary. Can include:
                - model (str): The name of the model to simulate.
                - simulated_delay_ms (int): The delay to simulate for API calls.
        """
        # It's crucial to call super().__init__ to set up the base adapter.
        super().__init__(config)
        self.simulated_delay_s = self.config.get("simulated_delay_ms", 20) / 1000

    def _create_client(self) -> Any:
        """
        Demonstrates the client creation pattern.
        For this example, we don't need a real client, so we return a mock object.
        """
        class MockApiClient:
            def __init__(self):
                self.info = "Example Mock Client v1.0"
        
        return MockApiClient()

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Implements the required cost calculation method.
        This simulates a simple, dual-rate pricing model.
        """
        # Example pricing: $0.0005/1K input tokens, $0.0015/1K output tokens
        input_cost = (input_tokens / 1000) * 0.0005
        output_cost = (output_tokens / 1000) * 0.0015
        return input_cost + output_cost

    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """
        Core asynchronous inference logic for the example engine.
        This is the primary method to be implemented by any new provider.
        """
        agent_name = kwargs.get("agent_name", "Agent")
        
        # 1. Simulate network/processing latency
        await asyncio.sleep(self.simulated_delay_s)

        # 2. Generate a predictable response
        content = f"[{agent_name}] successfully processed the prompt: '{prompt[:50]}...'"

        # 3. Simulate token calculation
        # A real implementation would get this from the provider's API response.
        input_tokens = len(prompt.split())
        output_tokens = len(content.split())

        # 4. Construct and return the standardized InferenceResponse object
        return InferenceResponse(
            content=content,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response={"simulated": True, "timestamp": time.time()},
            metadata={"confidence_score": 0.95} # Example of extra metadata
        )

    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Demonstrates an implementation of the optional streaming method.
        """
        agent_name = kwargs.get("agent_name", "Agent")
        
        # Simulate generating a response word by word
        response_words = f"[{agent_name}] is streaming a response for: '{prompt[:50]}...'".split()
        
        for i, word in enumerate(response_words):
            await asyncio.sleep(self.simulated_delay_s / 5) # Faster delay for chunks
            
            yield {
                "chunk": f"{word} ",
                "is_final": i == len(response_words) - 1,
                "metadata": {"chunk_index": i}
            }

    def validate_config(self) -> bool:
        """Demonstrates a simple configuration validation."""
        # Ensure the base validation passes and our specific keys are valid.
        base_valid = super().validate_config()
        delay_valid = isinstance(self.config.get("simulated_delay_ms", 0), int)
        return base_valid and delay_valid
