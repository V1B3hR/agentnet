"""
OpenAI provider adapter for AgentNet.

This adapter provides integration with OpenAI's API including GPT-4, GPT-3.5, and other models.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class OpenAIProvider(ProviderAdapter):
    """
    Provider adapter for OpenAI API.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    Requires openai package to be installed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            config: Configuration dictionary with:
                - api_key (str): OpenAI API key (or set OPENAI_API_KEY env var)
                - model (str): Model name (default: gpt-3.5-turbo)
                - organization (str): Optional organization ID
                - temperature (float): Sampling temperature (0-2)
                - max_tokens (int): Maximum tokens to generate
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set via config or OPENAI_API_KEY env var")
        
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.organization = self.config.get("organization")
        
    def _create_client(self) -> Any:
        """Create OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI provider. "
                "Install with: pip install openai>=1.0.0"
            )
        
        client_args = {"api_key": self.api_key}
        if self.organization:
            client_args["organization"] = self.organization
            
        return AsyncOpenAI(**client_args)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on OpenAI pricing.
        
        Prices as of 2024 (may need updating):
        - GPT-4-turbo: $0.01/1K input, $0.03/1K output
        - GPT-4: $0.03/1K input, $0.06/1K output
        - GPT-3.5-turbo: $0.0005/1K input, $0.0015/1K output
        """
        pricing = {
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            "gpt-3.5-turbo-16k": (0.003, 0.004),
        }
        
        # Default to GPT-3.5-turbo pricing if model not found
        input_price, output_price = pricing.get(
            self.model,
            pricing["gpt-3.5-turbo"]
        )
        
        input_cost = (input_tokens / 1000) * input_price
        output_cost = (output_tokens / 1000) * output_price
        
        return input_cost + output_cost
    
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """
        Execute inference against OpenAI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        
        Returns:
            InferenceResponse with results
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build request parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
        }
        
        if "max_tokens" in kwargs or "max_tokens" in self.config:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.get("max_tokens"))
        
        # Execute request
        response = await self.client.chat.completions.create(**params)
        
        # Extract response data
        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Build response
        return InferenceResponse(
            content=content,
            model_name=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
                "provider": "openai",
            }
        )
    
    async def _infer_stream(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[InferenceResponse]:
        """
        Execute streaming inference against OpenAI API.
        
        Yields partial responses as they arrive.
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build request parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
            "stream": True,
        }
        
        if "max_tokens" in kwargs or "max_tokens" in self.config:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.get("max_tokens"))
        
        # Execute streaming request
        stream = await self.client.chat.completions.create(**params)
        
        accumulated_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta_content = chunk.choices[0].delta.content
                accumulated_content += delta_content
                
                # Yield incremental response
                # Note: Token counts not available in streaming, estimate or mark as 0
                yield InferenceResponse(
                    content=accumulated_content,
                    model_name=chunk.model,
                    input_tokens=0,  # Not available during streaming
                    output_tokens=0,  # Not available during streaming
                    cost_usd=0.0,  # Will be calculated at end
                    metadata={
                        "is_streaming": True,
                        "response_id": chunk.id,
                        "provider": "openai",
                    }
                )
