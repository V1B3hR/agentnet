"""Base engine interface for AgentNet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Dict, List, Optional


class BaseEngine(ABC):
    """Abstract base class for LLM providers and inference engines."""

    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Synchronous inference method.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with 'content', 'confidence', and other metadata
        """
        pass

    @abstractmethod
    async def async_infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Asynchronous inference method.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with 'content', 'confidence', and other metadata
        """
        pass

    def get_cost_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost information for a result.

        Args:
            result: The inference result

        Returns:
            Dictionary with cost information
        """
        return {"cost": 0.0, "tokens": 0}

    def supports_streaming(self) -> bool:
        """Check if engine supports streaming responses."""
        return False
