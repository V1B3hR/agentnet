"""OpenAI provider adapter."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """OpenAI API adapter for GPT models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI adapter.

        Args:
            config: Configuration with optional 'api_key', 'model', 'organization'
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = self.config.get("model", "gpt-4")
        self.organization = self.config.get("organization")
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    organization=self.organization
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        return self._client

    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Synchronous inference using OpenAI API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with 'content', 'confidence', and metadata
        """
        client = self._get_client()
        
        # Extract parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        model = kwargs.get("model", self.model)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "confidence": 1.0 - (temperature / 2.0),  # Rough estimate
                "model": model,
                "provider": "openai",
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "confidence": 0.0,
                "model": model,
                "provider": "openai",
                "error": str(e)
            }

    async def async_infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Asynchronous inference using OpenAI API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'content', 'confidence', and metadata
        """
        # For now, use sync client wrapped
        # In production, use AsyncOpenAI
        return self.infer(prompt, **kwargs)

    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Streaming inference using OpenAI API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Yields:
            Partial results as dictionaries
        """
        client = self._get_client()
        
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        model = kwargs.get("model", self.model)
        
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "partial": True,
                        "model": model,
                        "provider": "openai"
                    }
        except Exception as e:
            yield {
                "content": f"Error: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
                "partial": False
            }

    def get_cost_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost information for OpenAI result.

        Args:
            result: Inference result

        Returns:
            Cost information dictionary
        """
        # Pricing as of 2024 (approximate, should be configurable)
        pricing = {
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
        }
        
        model = result.get("model", "gpt-4")
        usage = result.get("usage", {})
        
        # Find matching pricing
        model_pricing = None
        for key in pricing:
            if key in model:
                model_pricing = pricing[key]
                break
        
        if not model_pricing:
            model_pricing = pricing["gpt-4"]  # Default
        
        prompt_cost = usage.get("prompt_tokens", 0) * model_pricing["prompt"]
        completion_cost = usage.get("completion_tokens", 0) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "cost": total_cost,
            "tokens": usage.get("total_tokens", 0),
            "provider": "openai",
            "model": model,
            "breakdown": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost
            }
        }

    def validate_config(self) -> bool:
        """Validate OpenAI configuration.

        Returns:
            True if configuration is valid
        """
        return bool(self.api_key)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get OpenAI provider information.

        Returns:
            Provider metadata
        """
        return {
            "name": "OpenAI",
            "supports_streaming": True,
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "configured": self.validate_config(),
            "default_model": self.model
        }
