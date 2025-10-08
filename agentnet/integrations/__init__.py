"""
AgentNet Integrations Module

This module provides compatibility layers and integrations with popular ML/AI frameworks
and services, enabling seamless migration and interoperability.

Available integrations:
- LangChain compatibility layer
- OpenAI Assistants API
- Hugging Face Hub
- Vector databases (Pinecone, Weaviate, Milvus)
- Monitoring stack (Grafana, Prometheus)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .langchain import LangChainCompatibilityLayer
    from .openai_assistants import OpenAIAssistantsAdapter
    from .huggingface import HuggingFaceHubAdapter
    from .vector_databases import PineconeAdapter, WeaviateAdapter, MilvusAdapter
    from .monitoring import GrafanaIntegration, PrometheusIntegration

__all__ = [
    "LangChainCompatibilityLayer",
    "OpenAIAssistantsAdapter",
    "HuggingFaceHubAdapter",
    "PineconeAdapter",
    "WeaviateAdapter",
    "MilvusAdapter",
    "GrafanaIntegration",
    "PrometheusIntegration",
    # Utility functions
    "get_langchain_compatibility",
    "get_openai_assistants",
    "get_huggingface_hub",
    "get_vector_database_adapter",
    "get_monitoring_integration",
]


# Lazy imports to avoid import errors when optional dependencies are missing
def get_langchain_compatibility():
    """Get LangChain compatibility layer with lazy import."""
    try:
        from .langchain import LangChainCompatibilityLayer

        return LangChainCompatibilityLayer
    except ImportError as e:
        raise ImportError(
            "LangChain integration requires: pip install 'agentnet[langchain]' or 'langchain>=0.1.0'"
        ) from e


def get_openai_assistants():
    """Get OpenAI Assistants adapter with lazy import."""
    try:
        from .openai_assistants import OpenAIAssistantsAdapter

        return OpenAIAssistantsAdapter
    except ImportError as e:
        raise ImportError(
            "OpenAI Assistants integration requires: pip install 'agentnet[openai]' or 'openai>=1.0.0'"
        ) from e


def get_huggingface_hub():
    """Get Hugging Face Hub adapter with lazy import."""
    try:
        from .huggingface import HuggingFaceHubAdapter

        return HuggingFaceHubAdapter
    except ImportError as e:
        raise ImportError(
            "Hugging Face integration requires: pip install 'agentnet[huggingface]' or 'huggingface_hub>=0.16.0'"
        ) from e


def get_vector_database_adapter(provider: str):
    """Get vector database adapter with lazy import."""
    if provider.lower() == "pinecone":
        try:
            from .vector_databases import PineconeAdapter

            return PineconeAdapter
        except ImportError as e:
            raise ImportError(
                "Pinecone integration requires: pip install 'agentnet[pinecone]' or 'pinecone-client>=2.2.0'"
            ) from e
    elif provider.lower() == "weaviate":
        try:
            from .vector_databases import WeaviateAdapter

            return WeaviateAdapter
        except ImportError as e:
            raise ImportError(
                "Weaviate integration requires: pip install 'agentnet[weaviate]' or 'weaviate-client>=3.15.0'"
            ) from e
    elif provider.lower() == "milvus":
        try:
            from .vector_databases import MilvusAdapter

            return MilvusAdapter
        except ImportError as e:
            raise ImportError(
                "Milvus integration requires: pip install 'agentnet[milvus]' or 'pymilvus>=2.3.0'"
            ) from e
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")


def get_monitoring_integration(provider: str):
    """Get monitoring integration with lazy import."""
    if provider.lower() == "grafana":
        try:
            from .monitoring import GrafanaIntegration

            return GrafanaIntegration
        except ImportError as e:
            raise ImportError(
                "Grafana integration requires: pip install 'agentnet[monitoring]' or 'grafana-api>=1.5.0'"
            ) from e
    elif provider.lower() == "prometheus":
        try:
            from .monitoring import PrometheusIntegration

            return PrometheusIntegration
        except ImportError as e:
            raise ImportError(
                "Prometheus integration requires: pip install 'agentnet[monitoring]' or 'prometheus-client>=0.16.0'"
            ) from e
    else:
        raise ValueError(f"Unsupported monitoring provider: {provider}")
