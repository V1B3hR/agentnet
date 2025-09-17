"""Base provider adapter interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncIterator


class ProviderAdapter(ABC):
    """Abstract base class for LLM provider adapters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize provider adapter.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Synchronous inference.
        
        Args:
            prompt: Input prompt
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary with 'content', 'confidence', and metadata
        """
        pass
    
    @abstractmethod
    async def async_infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Asynchronous inference.
        
        Args:
            prompt: Input prompt
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary with 'content', 'confidence', and metadata
        """
        pass
    
    def stream_infer(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Streaming inference (optional).
        
        Args:
            prompt: Input prompt
            **kwargs: Provider-specific parameters
            
        Yields:
            Partial results as dictionaries
        """
        raise NotImplementedError("Streaming not supported by this provider")
    
    def get_cost_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost information for a result.
        
        Args:
            result: Inference result
            
        Returns:
            Cost information dictionary
        """
        return {"cost": 0.0, "tokens": 0, "provider": self.__class__.__name__}
    
    def validate_config(self) -> bool:
        """Validate provider configuration.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Provider metadata
        """
        return {
            "name": self.__class__.__name__,
            "supports_streaming": hasattr(self, 'stream_infer'),
            "config_keys": list(self.config.keys())
        }