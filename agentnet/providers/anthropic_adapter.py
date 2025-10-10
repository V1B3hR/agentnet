"""
Anthropic (Claude) provider adapter implementation.

Provides integration with Anthropic's Claude API with built-in
retry logic, cost tracking, and observability.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class AnthropicAdapter(ProviderAdapter):
    """
    Provider adapter for Anthropic's Claude API.
    
    Supports Claude 3 family (Opus, Sonnet, Haiku) with automatic
    cost calculation and token tracking.
    """
    
    # Pricing per 1M tokens (as of 2024, subject to change)
    PRICING = {
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-2.1": {"input": 8.00, "output": 24.00},
        "claude-2.0": {"input": 8.00, "output": 24.00},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Anthropic adapter.
        
        Args:
            config: Configuration dictionary with keys:
                - model (str): Model name (default: claude-3-sonnet-20240229)
                - api_key (str): Anthropic API key (default: from ANTHROPIC_API_KEY env)
                - temperature (float): Sampling temperature
                - max_tokens (int): Max tokens to generate
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass in config.")
        
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
    
    def _create_client(self) -> Any:
        """Create Anthropic client instance."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic adapter. "
                "Install with: pip install anthropic"
            )
        
        return anthropic.AsyncAnthropic(api_key=self.api_key)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on Anthropic pricing (per 1M tokens)."""
        # Get pricing for the model, default to sonnet if unknown
        pricing = self.PRICING.get(
            self.model_name, 
            self.PRICING["claude-3-sonnet-20240229"]
        )
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """Core inference logic using Anthropic API."""
        import time
        
        start_time = time.time()
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Anthropic uses system parameter separately
        system_message = kwargs.get("system_message", "You are a helpful AI assistant.")
        
        # Call Anthropic API
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system_message,
            messages=messages,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response data
        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
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
                "stop_reason": response.stop_reason,
                "model_used": response.model,
            }
        )
    
    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream inference results from Anthropic."""
        messages = [{"role": "user", "content": prompt}]
        system_message = kwargs.get("system_message", "You are a helpful AI assistant.")
        
        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system_message,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield {
                    "content": text,
                    "finish_reason": None,
                }
