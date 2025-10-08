"""
Anthropic (Claude) provider adapter for AgentNet.

This adapter provides integration with Anthropic's Claude API.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class AnthropicProvider(ProviderAdapter):
    """
    Provider adapter for Anthropic's Claude API.
    
    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2 models.
    Requires anthropic package to be installed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Anthropic provider.
        
        Args:
            config: Configuration dictionary with:
                - api_key (str): Anthropic API key (or set ANTHROPIC_API_KEY env var)
                - model (str): Model name (default: claude-3-sonnet-20240229)
                - max_tokens (int): Maximum tokens to generate (required by Anthropic)
                - temperature (float): Sampling temperature (0-1)
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set via config or ANTHROPIC_API_KEY env var")
        
        self.model = self.config.get("model", "claude-3-sonnet-20240229")
        self.max_tokens = self.config.get("max_tokens", 1024)
        
    def _create_client(self) -> Any:
        """Create Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install anthropic>=0.18.0"
            )
        
        return AsyncAnthropic(api_key=self.api_key)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on Anthropic pricing.
        
        Prices as of 2024 (may need updating):
        - Claude 3 Opus: $15/1M input, $75/1M output
        - Claude 3 Sonnet: $3/1M input, $15/1M output
        - Claude 3 Haiku: $0.25/1M input, $1.25/1M output
        - Claude 2.1: $8/1M input, $24/1M output
        """
        pricing = {
            "claude-3-opus": (15, 75),
            "claude-3-sonnet": (3, 15),
            "claude-3-haiku": (0.25, 1.25),
            "claude-2.1": (8, 24),
            "claude-2.0": (8, 24),
        }
        
        # Find matching pricing (check if model name contains key)
        input_price, output_price = (3, 15)  # Default to Sonnet pricing
        for model_key, prices in pricing.items():
            if model_key in self.model:
                input_price, output_price = prices
                break
        
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost
    
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """
        Execute inference against Anthropic API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        
        Returns:
            InferenceResponse with results
        """
        # Build request parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.config.get("temperature", 1.0)),
        }
        
        # Add optional parameters
        if "system" in kwargs:
            params["system"] = kwargs["system"]
        
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        
        if "top_k" in kwargs:
            params["top_k"] = kwargs["top_k"]
        
        # Execute request
        response = await self.client.messages.create(**params)
        
        # Extract response data
        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        # Build response
        return InferenceResponse(
            content=content,
            model_name=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            metadata={
                "stop_reason": response.stop_reason,
                "response_id": response.id,
                "role": response.role,
                "provider": "anthropic",
            }
        )
    
    async def _infer_stream(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[InferenceResponse]:
        """
        Execute streaming inference against Anthropic API.
        
        Yields partial responses as they arrive.
        """
        # Build request parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.config.get("temperature", 1.0)),
            "stream": True,
        }
        
        # Add optional parameters
        if "system" in kwargs:
            params["system"] = kwargs["system"]
        
        # Execute streaming request
        accumulated_content = ""
        input_tokens = 0
        output_tokens = 0
        
        async with self.client.messages.stream(**params) as stream:
            async for event in stream:
                # Handle different event types
                if hasattr(event, 'type'):
                    if event.type == "message_start":
                        if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                            input_tokens = event.message.usage.input_tokens
                    
                    elif event.type == "content_block_delta":
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            delta_content = event.delta.text
                            accumulated_content += delta_content
                            
                            # Yield incremental response
                            yield InferenceResponse(
                                content=accumulated_content,
                                model_name=self.model,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cost_usd=self._calculate_cost(input_tokens, output_tokens) if output_tokens > 0 else 0.0,
                                metadata={
                                    "is_streaming": True,
                                    "provider": "anthropic",
                                }
                            )
                    
                    elif event.type == "message_delta":
                        if hasattr(event, 'usage'):
                            output_tokens = event.usage.output_tokens
