"""
LangChain Compatibility Layer

This module provides seamless migration support for LangChain projects,
allowing existing LangChain code to work with AgentNet with minimal changes.
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
from abc import ABC, abstractmethod
import warnings
import logging

try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.llms.base import BaseLLM
    from langchain.chat_models.base import BaseChatModel
    from langchain.embeddings.base import Embeddings
    from langchain.vectorstores.base import VectorStore
    from langchain.tools.base import BaseTool
    from langchain.agents.agent import Agent

    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Define minimal interface for type hints when LangChain is not available
    class BaseMessage:
        pass

    class HumanMessage:
        pass

    class AIMessage:
        pass

    class SystemMessage:
        pass

    class BaseLLM:
        pass

    class BaseChatModel:
        pass

    class Embeddings:
        pass

    class VectorStore:
        pass

    class BaseTool:
        pass

    class Agent:
        pass

    LANGCHAIN_AVAILABLE = False

from ..providers.base import ProviderAdapter
from ..core.agent import AgentNet
from ..core.types import InferenceResult

logger = logging.getLogger(__name__)


class LangChainCompatibilityLayer:
    """
    Compatibility layer that allows LangChain components to work with AgentNet.

    Features:
    - Automatic conversion between LangChain and AgentNet message formats
    - Wrapper for LangChain LLMs to work as AgentNet providers
    - Migration utilities for existing LangChain code
    - Vector store integration
    """

    def __init__(self, agent_net: Optional[AgentNet] = None):
        """Initialize the compatibility layer."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for compatibility layer. Install with: pip install langchain>=0.1.0"
            )

        self.agent_net = agent_net
        self._message_converters = {
            "human": HumanMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }

    def convert_message_to_langchain(self, message: Dict[str, Any]) -> BaseMessage:
        """Convert AgentNet message format to LangChain message."""
        message_type = message.get("type", "human").lower()
        content = message.get("content", "")

        if message_type in ["user", "human"]:
            return HumanMessage(content=content)
        elif message_type in ["assistant", "ai"]:
            return AIMessage(content=content)
        elif message_type == "system":
            return SystemMessage(content=content)
        else:
            logger.warning(
                f"Unknown message type: {message_type}, defaulting to HumanMessage"
            )
            return HumanMessage(content=content)

    def convert_message_from_langchain(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert LangChain message to AgentNet format."""
        if isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"type": "ai", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"type": "system", "content": message.content}
        else:
            return {"type": "human", "content": str(message.content)}

    def wrap_langchain_llm(self, llm: Union[BaseLLM, BaseChatModel]) -> ProviderAdapter:
        """
        Wrap a LangChain LLM to work as an AgentNet provider.

        Args:
            llm: LangChain LLM or ChatModel instance

        Returns:
            ProviderAdapter that can be used with AgentNet
        """
        return LangChainProviderAdapter(llm, self)

    def create_agent_from_langchain(
        self, langchain_agent: Agent, name: str = "LangChainAgent", **kwargs
    ) -> AgentNet:
        """
        Create an AgentNet agent from a LangChain agent.

        Args:
            langchain_agent: LangChain Agent instance
            name: Name for the AgentNet agent
            **kwargs: Additional arguments for AgentNet initialization

        Returns:
            AgentNet agent configured with LangChain compatibility
        """
        # Extract LLM from LangChain agent
        llm = getattr(langchain_agent, "llm", None)
        if llm is None:
            raise ValueError("LangChain agent must have an 'llm' attribute")

        # Wrap the LLM as a provider
        provider = self.wrap_langchain_llm(llm)

        # Create AgentNet with wrapped provider
        agent = AgentNet(
            name=name,
            style=kwargs.get("style", {"analytical": 0.7, "creative": 0.3}),
            engine=provider,
            **kwargs,
        )

        # Store reference to original LangChain agent for advanced features
        agent._langchain_agent = langchain_agent

        return agent

    def migrate_langchain_tools(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """
        Convert LangChain tools to AgentNet format.

        Args:
            tools: List of LangChain BaseTool instances

        Returns:
            List of tool configurations for AgentNet
        """
        agentnet_tools = []

        for tool in tools:
            agentnet_tool = {
                "name": tool.name,
                "description": tool.description,
                "function": tool._run,  # Direct function reference
                "async_function": getattr(tool, "_arun", None),
                "schema": getattr(tool, "args_schema", None),
            }
            agentnet_tools.append(agentnet_tool)

        return agentnet_tools

    def create_migration_guide(self, langchain_code: str) -> str:
        """
        Generate a migration guide for converting LangChain code to AgentNet.

        Args:
            langchain_code: String containing LangChain code

        Returns:
            Migration guide with suggested changes
        """
        suggestions = []

        # Basic pattern matching for common LangChain patterns
        if "from langchain" in langchain_code:
            suggestions.append("1. Replace LangChain imports with AgentNet equivalents")

        if "ChatOpenAI" in langchain_code or "OpenAI" in langchain_code:
            suggestions.append(
                "2. Use AgentNet's OpenAI provider adapter instead of LangChain's ChatOpenAI"
            )

        if "LLMChain" in langchain_code:
            suggestions.append(
                "3. Replace LLMChain with AgentNet's reasoning capabilities"
            )

        if "VectorStore" in langchain_code:
            suggestions.append("4. Use AgentNet's vector database integrations")

        guide = f"""
# LangChain to AgentNet Migration Guide

## Suggested Changes:
{chr(10).join(suggestions)}

## Example Migration:

### Before (LangChain):
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI()
response = llm([HumanMessage(content="Hello")])
```

### After (AgentNet):
```python
from agentnet import AgentNet
from agentnet.integrations import get_langchain_compatibility

# Option 1: Direct AgentNet usage
agent = AgentNet(name="Assistant", style={{"analytical": 0.7}})
response = agent.reason("Hello")

# Option 2: Using compatibility layer
compat = get_langchain_compatibility()
llm = ChatOpenAI()  # Your existing LangChain LLM
provider = compat.wrap_langchain_llm(llm)
agent = AgentNet(name="Assistant", style={{"analytical": 0.7}}, engine=provider)
```

## Benefits of Migration:
- Enhanced reasoning capabilities with style modulation
- Built-in monitoring and governance
- Better performance tracking
- More flexible agent orchestration
"""

        return guide


class LangChainProviderAdapter(ProviderAdapter):
    """
    Adapter that wraps LangChain LLMs to work with AgentNet's provider interface.
    """

    def __init__(
        self,
        langchain_llm: Union[BaseLLM, BaseChatModel],
        compat_layer: LangChainCompatibilityLayer,
    ):
        """Initialize the adapter with a LangChain LLM."""
        super().__init__()
        self.langchain_llm = langchain_llm
        self.compat_layer = compat_layer
        self.model_name = getattr(langchain_llm, "model_name", "langchain_llm")

    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """Synchronous inference using LangChain LLM."""
        try:
            # Handle both LLMs and ChatModels
            if hasattr(self.langchain_llm, "predict"):
                # Chat model
                response = self.langchain_llm.predict(prompt, **kwargs)
            else:
                # Regular LLM
                response = self.langchain_llm(prompt, **kwargs)

            return InferenceResult(
                content=response,
                confidence=0.8,  # Default confidence for LangChain responses
                tokens_used=len(prompt.split())
                + len(response.split()),  # Rough estimate
                cost=0.0,  # Cost tracking would need LangChain's token counting
                model=self.model_name,
                provider="langchain",
                metadata={
                    "langchain_model": self.model_name,
                    "original_response": response,
                },
            )
        except Exception as e:
            logger.error(f"LangChain inference error: {e}")
            return InferenceResult(
                content=f"Error: {str(e)}",
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                model=self.model_name,
                provider="langchain",
                error=str(e),
            )

    async def ainfer(self, prompt: str, **kwargs) -> InferenceResult:
        """Asynchronous inference using LangChain LLM."""
        try:
            # Check if async methods are available
            if hasattr(self.langchain_llm, "apredict"):
                response = await self.langchain_llm.apredict(prompt, **kwargs)
            elif hasattr(self.langchain_llm, "acall"):
                response = await self.langchain_llm.acall(prompt, **kwargs)
            else:
                # Fallback to sync method
                logger.warning(
                    "LangChain LLM doesn't support async, falling back to sync"
                )
                return self.infer(prompt, **kwargs)

            return InferenceResult(
                content=response,
                confidence=0.8,
                tokens_used=len(prompt.split()) + len(response.split()),
                cost=0.0,
                model=self.model_name,
                provider="langchain",
                metadata={
                    "langchain_model": self.model_name,
                    "original_response": response,
                    "async": True,
                },
            )
        except Exception as e:
            logger.error(f"LangChain async inference error: {e}")
            return InferenceResult(
                content=f"Error: {str(e)}",
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                model=self.model_name,
                provider="langchain",
                error=str(e),
            )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream inference using LangChain LLM."""
        try:
            if hasattr(self.langchain_llm, "stream"):
                for chunk in self.langchain_llm.stream(prompt, **kwargs):
                    yield chunk.content if hasattr(chunk, "content") else str(chunk)
            else:
                # Fallback to regular inference
                result = self.infer(prompt, **kwargs)
                yield result.content
        except Exception as e:
            logger.error(f"LangChain streaming error: {e}")
            yield f"Error: {str(e)}"

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Async stream inference using LangChain LLM."""
        try:
            if hasattr(self.langchain_llm, "astream"):
                async for chunk in self.langchain_llm.astream(prompt, **kwargs):
                    yield chunk.content if hasattr(chunk, "content") else str(chunk)
            else:
                # Fallback to regular async inference
                result = await self.ainfer(prompt, **kwargs)
                yield result.content
        except Exception as e:
            logger.error(f"LangChain async streaming error: {e}")
            yield f"Error: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the wrapped LangChain model."""
        return {
            "name": self.model_name,
            "provider": "langchain",
            "type": type(self.langchain_llm).__name__,
            "supports_streaming": hasattr(self.langchain_llm, "stream"),
            "supports_async": hasattr(self.langchain_llm, "apredict")
            or hasattr(self.langchain_llm, "acall"),
            "langchain_version": getattr(self.langchain_llm, "__version__", "unknown"),
        }


# Utility functions for easy migration
def migrate_from_langchain(
    langchain_agent: Agent, name: str = "MigratedAgent", **agentnet_kwargs
) -> AgentNet:
    """
    Quick migration utility to convert a LangChain agent to AgentNet.

    Args:
        langchain_agent: LangChain Agent instance
        name: Name for the new AgentNet agent
        **agentnet_kwargs: Additional arguments for AgentNet

    Returns:
        AgentNet agent with LangChain compatibility
    """
    compat = LangChainCompatibilityLayer()
    return compat.create_agent_from_langchain(langchain_agent, name, **agentnet_kwargs)


def wrap_langchain_llm(llm: Union[BaseLLM, BaseChatModel]) -> ProviderAdapter:
    """
    Quick utility to wrap a LangChain LLM for use with AgentNet.

    Args:
        llm: LangChain LLM or ChatModel

    Returns:
        ProviderAdapter for use with AgentNet
    """
    compat = LangChainCompatibilityLayer()
    return compat.wrap_langchain_llm(llm)
