"""
HuggingFace provider adapter implementation.

Provides integration with HuggingFace Inference API and local models
with built-in retry logic and observability.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Optional

from .base import ProviderAdapter, InferenceResponse


class HuggingFaceAdapter(ProviderAdapter):
    """
    Provider adapter for HuggingFace models.
    
    Supports both HuggingFace Inference API and locally hosted models
    with token tracking and cost estimation.
    """
    
    # Estimated pricing for popular models (API usage, per 1K tokens)
    # For local models, cost can be set to 0 or compute cost
    PRICING = {
        "gpt2": {"input": 0.0, "output": 0.0},  # Free/local
        "mistralai/Mistral-7B-Instruct-v0.2": {"input": 0.0002, "output": 0.0002},
        "meta-llama/Llama-2-7b-chat-hf": {"input": 0.0002, "output": 0.0002},
        "meta-llama/Llama-2-13b-chat-hf": {"input": 0.0004, "output": 0.0004},
        "meta-llama/Llama-2-70b-chat-hf": {"input": 0.001, "output": 0.001},
        "default": {"input": 0.0002, "output": 0.0002},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HuggingFace adapter.
        
        Args:
            config: Configuration dictionary with keys:
                - model (str): Model name or path
                - api_key (str): HF API token (default: from HF_TOKEN env)
                - use_local (bool): Use local model instead of API (default: False)
                - api_url (str): Custom inference endpoint URL
                - temperature (float): Sampling temperature
                - max_tokens (int): Max tokens to generate
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("HF_TOKEN")
        self.use_local = self.config.get("use_local", False)
        self.api_url = self.config.get("api_url")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
        
        if not self.use_local and not self.api_key:
            raise ValueError(
                "HuggingFace API token required for API usage. "
                "Set HF_TOKEN env var, pass in config, or set use_local=True."
            )
    
    def _create_client(self) -> Any:
        """Create HuggingFace client instance."""
        if self.use_local:
            # For local models, return a placeholder
            # Actual implementation would load the model locally
            return {"type": "local", "model": self.model_name}
        else:
            try:
                from huggingface_hub import InferenceClient
            except ImportError:
                raise ImportError(
                    "huggingface_hub package required for HuggingFace adapter. "
                    "Install with: pip install huggingface_hub"
                )
            
            return InferenceClient(
                model=self.api_url or self.model_name,
                token=self.api_key,
            )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model pricing."""
        if self.use_local:
            # Local models have no API cost
            return 0.0
        
        pricing = self.PRICING.get(self.model_name, self.PRICING["default"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    async def _infer(self, prompt: str, **kwargs) -> InferenceResponse:
        """Core inference logic using HuggingFace API."""
        import time
        
        start_time = time.time()
        
        if self.use_local:
            # Placeholder for local inference
            # Real implementation would use transformers pipeline
            content = f"[Local HF Model {self.model_name}] Response to: {prompt[:50]}..."
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content)
        else:
            # Use HuggingFace Inference API
            response = await self.client.text_generation(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                return_full_text=False,
            )
            
            content = response
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content)
        
        latency_ms = (time.time() - start_time) * 1000
        cost = self._calculate_cost(input_tokens, output_tokens)
        
        return InferenceResponse(
            content=content,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_response=None,
            metadata={
                "use_local": self.use_local,
                "estimated_tokens": True,
            }
        )
    
    async def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream inference results from HuggingFace."""
        if self.use_local:
            # Placeholder for local streaming
            words = f"[Local HF] Response for: {prompt[:50]}...".split()
            for word in words:
                yield {"content": word + " ", "finish_reason": None}
        else:
            # Use HuggingFace streaming API
            async for token in self.client.text_generation(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                stream=True,
            ):
                yield {
                    "content": token.token.text if hasattr(token, 'token') else str(token),
                    "finish_reason": None,
                }
