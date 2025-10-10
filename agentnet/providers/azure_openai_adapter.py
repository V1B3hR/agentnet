"""
Azure OpenAI provider adapter implementation.

Provides integration with Azure OpenAI Service with built-in
retry logic, cost tracking, and observability.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class AzureOpenAIAdapter(ProviderAdapter):
    """
    Provider adapter for Azure OpenAI Service.
    
    Supports GPT-4, GPT-3.5-turbo, and other models deployed on Azure with
    automatic cost calculation and token tracking.
    """
    
    # Pricing per 1K tokens (same as OpenAI, but may vary by region)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-35-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-35-turbo-16k": {"input": 0.003, "output": 0.004},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Azure OpenAI adapter.
        
        Args:
            config: Configuration dictionary with keys:
                - model (str): Deployment name in Azure
                - api_key (str): Azure OpenAI API key (default: from AZURE_OPENAI_API_KEY env)
                - azure_endpoint (str): Azure endpoint URL (default: from AZURE_OPENAI_ENDPOINT env)
                - api_version (str): API version (default: 2024-02-15-preview)
                - temperature (float): Sampling temperature
                - max_tokens (int): Max tokens to generate
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = self.config.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY env var or pass in config.")
        if not self.azure_endpoint:
            raise ValueError("Azure endpoint required. Set AZURE_OPENAI_ENDPOINT env var or pass in config.")
        
        self.api_version = self.config.get("api_version", "2024-02-15-preview")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
    
    def _create_client(self) -> Any:
        """Create Azure OpenAI client instance."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for Azure OpenAI adapter. "
                "Install with: pip install openai"
            )
        
        return openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on Azure OpenAI pricing."""
        # Try to map deployment name to base model for pricing
        # Azure deployments often include version numbers
        base_model = self.model_name
        for model_key in self.PRICING.keys():
            if model_key in self.model_name.lower():
                base_model = model_key
                break
        
        pricing = self.PRICING.get(base_model, self.PRICING["gpt-4o"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """Core inference logic using Azure OpenAI API."""
        import time
        
        start_time = time.time()
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Optional system message
        if "system_message" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})
        
        # Call Azure OpenAI API (uses deployment name as model parameter)
        response = await self.client.chat.completions.create(
            model=self.model_name,  # This is the deployment name in Azure
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
                "deployment_name": self.model_name,
            }
        )
    
    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream inference results from Azure OpenAI."""
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
