"""Example provider implementation for testing."""

from __future__ import annotations
import asyncio
import time
from typing import Any, Dict

from .base import ProviderAdapter


class ExampleEngine(ProviderAdapter):
    """Example engine implementation for testing and demonstrations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize example engine.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "ExampleEngine"
    
    def infer(self, prompt: str, agent_name: str = "Agent", **kwargs) -> Dict[str, Any]:
        """Synchronous inference with example response.
        
        Args:
            prompt: Input prompt/task
            agent_name: Name of the requesting agent
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with content and confidence
        """
        time.sleep(0.02)  # Simulate processing time
        conf = 0.6 if "Round" in prompt else 0.9
        content = f"[{agent_name}] Thoughts about: {prompt}"
        
        return {
            "content": content, 
            "confidence": conf,
            "runtime": 0.02,
            "provider": "ExampleEngine"
        }
    
    async def async_infer(self, prompt: str, agent_name: str = "Agent", **kwargs) -> Dict[str, Any]:
        """Asynchronous inference with example response.
        
        Args:
            prompt: Input prompt/task
            agent_name: Name of the requesting agent
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with content and confidence
        """
        await asyncio.sleep(0.02)  # Simulate async processing time
        conf = 0.65 if "Round" in prompt else 0.92
        content = f"[{agent_name}] (async) Thoughts about: {prompt}"
        
        return {
            "content": content, 
            "confidence": conf,
            "runtime": 0.02,
            "provider": "ExampleEngine"
        }
    
    def safe_think(self, agent_name: str, task: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.infer(task, agent_name=agent_name)
    
    async def safe_think_async(self, agent_name: str, task: str) -> Dict[str, Any]:
        """Legacy async method for backward compatibility."""
        return await self.async_infer(task, agent_name=agent_name)
    
    def get_cost_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost information for the example engine."""
        return {
            "cost": 0.001,  # Example cost
            "tokens": len(result.get("content", "").split()),
            "provider": "ExampleEngine"
        }