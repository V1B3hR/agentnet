"""
Tests for AgentNet integrations module.
"""

import pytest
from unittest.mock import patch, MagicMock

from agentnet.integrations import (
    get_langchain_compatibility,
    get_openai_assistants,
    get_huggingface_hub,
    get_vector_database_adapter,
    get_monitoring_integration,
)


def test_integration_imports_with_missing_dependencies():
    """Test that integration functions raise appropriate ImportError when dependencies are missing."""

    # Test LangChain integration
    with pytest.raises(ImportError) as exc_info:
        get_langchain_compatibility()
    assert "langchain>=0.1.0" in str(exc_info.value)

    # Test OpenAI Assistants integration
    with pytest.raises(ImportError) as exc_info:
        get_openai_assistants()
    assert "openai>=1.0.0" in str(exc_info.value)

    # Test Hugging Face integration
    with pytest.raises(ImportError) as exc_info:
        get_huggingface_hub()
    assert "huggingface_hub>=0.16.0" in str(exc_info.value)


def test_vector_database_adapters_available():
    """Test that vector database adapters are available (no external dependencies)."""

    # These should work because they don't require external dependencies to import
    PineconeAdapter = get_vector_database_adapter("pinecone")
    assert PineconeAdapter.__name__ == "PineconeAdapter"

    WeaviateAdapter = get_vector_database_adapter("weaviate")
    assert WeaviateAdapter.__name__ == "WeaviateAdapter"

    MilvusAdapter = get_vector_database_adapter("milvus")
    assert MilvusAdapter.__name__ == "MilvusAdapter"


def test_monitoring_integrations_available():
    """Test that monitoring integrations are available."""

    PrometheusIntegration = get_monitoring_integration("prometheus")
    assert PrometheusIntegration.__name__ == "PrometheusIntegration"

    GrafanaIntegration = get_monitoring_integration("grafana")
    assert GrafanaIntegration.__name__ == "GrafanaIntegration"


def test_invalid_provider_names():
    """Test that invalid provider names raise ValueError."""

    with pytest.raises(ValueError) as exc_info:
        get_vector_database_adapter("invalid_provider")
    assert "Unsupported vector database provider: invalid_provider" in str(
        exc_info.value
    )

    with pytest.raises(ValueError) as exc_info:
        get_monitoring_integration("invalid_provider")
    assert "Unsupported monitoring provider: invalid_provider" in str(exc_info.value)


@patch("agentnet.integrations.langchain.LANGCHAIN_AVAILABLE", True)
def test_langchain_compatibility_with_mocked_dependency():
    """Test LangChain compatibility layer with mocked dependencies."""

    # Mock the langchain module
    mock_langchain = MagicMock()
    mock_langchain.schema.BaseMessage = MagicMock()
    mock_langchain.schema.HumanMessage = MagicMock()
    mock_langchain.schema.AIMessage = MagicMock()

    with patch.dict("sys.modules", {"langchain": mock_langchain}):
        with patch(
            "agentnet.integrations.langchain.LangChainCompatibilityLayer"
        ) as mock_class:
            LangChainCompatibilityLayer = get_langchain_compatibility()
            assert LangChainCompatibilityLayer == mock_class


def test_vector_database_adapter_initialization():
    """Test that vector database adapters can be initialized with basic config."""

    # Test Pinecone adapter initialization (should fail due to missing dependencies)
    PineconeAdapter = get_vector_database_adapter("pinecone")

    with pytest.raises(ImportError):
        PineconeAdapter(api_key="test", environment="test")

    # Test Weaviate adapter initialization (should fail due to missing dependencies)
    WeaviateAdapter = get_vector_database_adapter("weaviate")

    with pytest.raises(ImportError):
        WeaviateAdapter(url="http://localhost:8080")

    # Test Milvus adapter initialization (should fail due to missing dependencies)
    MilvusAdapter = get_vector_database_adapter("milvus")

    with pytest.raises(ImportError):
        MilvusAdapter(host="localhost", port=19530)


def test_monitoring_integration_initialization():
    """Test monitoring integration initialization."""

    # Test Prometheus integration (should fail due to missing prometheus_client)
    PrometheusIntegration = get_monitoring_integration("prometheus")

    with pytest.raises(ImportError):
        PrometheusIntegration()

    # Test Grafana integration (should fail due to missing requests)
    GrafanaIntegration = get_monitoring_integration("grafana")

    with pytest.raises(ImportError):
        GrafanaIntegration(url="http://localhost:3000", api_key="test")


def test_integration_module_exports():
    """Test that integration module exports expected functions."""

    import agentnet.integrations as integrations_module

    expected_exports = [
        "get_langchain_compatibility",
        "get_openai_assistants",
        "get_huggingface_hub",
        "get_vector_database_adapter",
        "get_monitoring_integration",
    ]

    for export in expected_exports:
        assert hasattr(integrations_module, export), f"Missing export: {export}"
        assert callable(
            getattr(integrations_module, export)
        ), f"Export not callable: {export}"


if __name__ == "__main__":
    # Run some basic tests
    print("Testing AgentNet integrations...")

    try:
        test_vector_database_adapters_available()
        print("✅ Vector database adapters available")
    except Exception as e:
        print(f"❌ Vector database test failed: {e}")

    try:
        test_monitoring_integrations_available()
        print("✅ Monitoring integrations available")
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")

    try:
        test_invalid_provider_names()
        print("✅ Invalid provider validation works")
    except Exception as e:
        print(f"❌ Provider validation test failed: {e}")

    try:
        test_integration_imports_with_missing_dependencies()
        print("✅ Dependency checking works")
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")

    print("Integration tests completed!")
