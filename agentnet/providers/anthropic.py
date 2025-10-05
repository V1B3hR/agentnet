"""Anthropic provider adapter."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    """Anthropic API adapter for Claude models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Anthropic adapter.

        Args:
            config: Configuration with optional 'api_key', 'model'
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.model = self.config.get("model", "claude-3-opus-20240229")
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Synchronous inference using Anthropic API.

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
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            return {
                "content": content,
                "confidence": 1.0 - (temperature / 2.0),  # Rough estimate
                "model": model,
                "provider": "anthropic",
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "confidence": 0.0,
                "model": model,
                "provider": "anthropic",
                "error": str(e)
            }

    async def async_infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Asynchronous inference using Anthropic API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'content', 'confidence', and metadata
        """
        # For now, use sync client wrapped
        # In production, use AsyncAnthropic
        return self.infer(prompt, **kwargs)

    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Streaming inference using Anthropic API.

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
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield {
                        "content": text,
                        "partial": True,
                        "model": model,
                        "provider": "anthropic"
                    }
        except Exception as e:
            yield {
                "content": f"Error: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
                "partial": False
            }

    def get_cost_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost information for Anthropic result.

        Args:
            result: Inference result

        Returns:
            Cost information dictionary
        """
        # Pricing as of 2024 (approximate, should be configurable)
        pricing = {
            "claude-3-opus": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
            "claude-3-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
            "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
        }
        
        model = result.get("model", "claude-3-opus")
        usage = result.get("usage", {})
        
        # Find matching pricing
        model_pricing = None
        for key in pricing:
            if key in model:
                model_pricing = pricing[key]
                break
        
        if not model_pricing:
            model_pricing = pricing["claude-3-opus"]  # Default
        
        input_cost = usage.get("input_tokens", 0) * model_pricing["input"]
        output_cost = usage.get("output_tokens", 0) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "cost": total_cost,
            "tokens": usage.get("total_tokens", 0),
            "provider": "anthropic",
            "model": model,
            "breakdown": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "input_cost": input_cost,
                "output_cost": output_cost
            }
        }

    def validate_config(self) -> bool:
        """Validate Anthropic configuration.

        Returns:
            True if configuration is valid
        """
        return bool(self.api_key)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get Anthropic provider information.

        Returns:
            Provider metadata
        """
        return {
            "name": "Anthropic",
            "supports_streaming": True,
            "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "configured": self.validate_config(),
            "default_model": self.model
        }
