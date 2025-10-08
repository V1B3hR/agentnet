"""
Tests for provider adapter implementations.
"""

import pytest
from agentnet.providers import ProviderAdapter, ExampleEngine, get_available_providers


def test_example_provider_initialization():
    """Test that ExampleEngine can be initialized."""
    provider = ExampleEngine()
    assert provider is not None
    assert isinstance(provider, ProviderAdapter)
    

def test_example_provider_with_config():
    """Test ExampleEngine with custom config."""
    config = {
        "model": "example-model-v2",
        "simulated_delay_ms": 50
    }
    provider = ExampleEngine(config)
    assert provider.config["model"] == "example-model-v2"
    assert provider.simulated_delay_s == 0.05


@pytest.mark.asyncio
async def test_example_provider_inference():
    """Test basic inference with ExampleEngine."""
    provider = ExampleEngine({"simulated_delay_ms": 1})
    
    result = await provider.async_infer("Test prompt")
    
    assert result is not None
    assert result.content is not None
    assert isinstance(result.content, str)
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.model_name is not None  # ExampleEngine provides model_name


@pytest.mark.asyncio
async def test_example_provider_cost_calculation():
    """Test cost calculation in ExampleEngine."""
    provider = ExampleEngine()
    
    # Test cost calculation directly
    cost = provider._calculate_cost(input_tokens=1000, output_tokens=500)
    
    # Based on ExampleEngine pricing: $0.0005/1K input, $0.0015/1K output
    expected_cost = (1000/1000 * 0.0005) + (500/1000 * 0.0015)
    assert abs(cost - expected_cost) < 0.0001


def test_get_available_providers():
    """Test that available providers are listed."""
    providers = get_available_providers()
    
    assert isinstance(providers, list)
    assert "example" in providers
    # OpenAI and Anthropic may or may not be available depending on dependencies


@pytest.mark.skipif(
    "openai" not in get_available_providers(),
    reason="OpenAI provider not available (missing openai package)"
)
def test_openai_provider_import():
    """Test that OpenAIProvider can be imported when available."""
    from agentnet.providers import OpenAIProvider
    assert OpenAIProvider is not None


@pytest.mark.skipif(
    "anthropic" not in get_available_providers(),
    reason="Anthropic provider not available (missing anthropic package)"
)
def test_anthropic_provider_import():
    """Test that AnthropicProvider can be imported when available."""
    from agentnet.providers import AnthropicProvider
    assert AnthropicProvider is not None


def test_provider_adapter_is_abstract():
    """Test that ProviderAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ProviderAdapter({"test": "config"})


@pytest.mark.asyncio
async def test_provider_retry_mechanism():
    """Test that retry mechanism is present in base adapter."""
    provider = ExampleEngine()
    
    # The async_infer method should have retry logic from tenacity
    # This test just verifies the method is callable
    result = await provider.async_infer("Test with potential retry")
    assert result is not None


def test_inference_response_structure():
    """Test the structure of InferenceResponse."""
    from agentnet.providers.base import InferenceResponse
    
    response = InferenceResponse(
        content="Test response",
        model_name="test-model",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.01
    )
    
    assert response.content == "Test response"
    assert response.model_name == "test-model"
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.cost_usd == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
