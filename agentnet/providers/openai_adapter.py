"""
OpenAI provider adapter implementation.

Provides integration with OpenAI's API (GPT-4, GPT-3.5, etc.) with built-in
retry logic, cost tracking, and observability.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class OpenAIAdapter(ProviderAdapter):
    """
    Provider adapter for OpenAI's API.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models with automatic
    cost calculation and token tracking.
    """
    
    # Pricing per 1K tokens (as of 2024, subject to change)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI adapter.
        
        Args:
            config: Configuration dictionary with keys:
                - model (str): Model name (default: gpt-4o-mini)
                - api_key (str): OpenAI API key (default: from OPENAI_API_KEY env)
                - organization (str): Optional organization ID
                - temperature (float): Sampling temperature
                - max_tokens (int): Max tokens to generate
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass in config.")
        
        self.organization = self.config.get("organization")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
    
    def _create_client(self) -> Any:
        """Create OpenAI client instance."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI adapter. "
                "Install with: pip install openai"
            )
        
        client_kwargs = {"api_key": self.api_key}
        if self.organization:
            client_kwargs["organization"] = self.organization
        
        return openai.AsyncOpenAI(**client_kwargs)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on OpenAI pricing."""
        # Get pricing for the model, default to gpt-4o pricing if unknown
        pricing = self.PRICING.get(self.model_name, self.PRICING["gpt-4o"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """Core inference logic using OpenAI API."""
        import time
        
        start_time = time.time()
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Optional system message
        if "system_message" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response data
        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)
        
        return InferenceResponse(
            content=content,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_response=response,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "model_used": response.model,
            }
        )
    
    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream inference results from OpenAI."""
        messages = [{"role": "user", "content": prompt}]
        
        if "system_message" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})
        
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "content": chunk.choices[0].delta.content,
                    "finish_reason": chunk.choices[0].finish_reason,
                }
