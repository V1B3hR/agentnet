#!/usr/bin/env python3
"""Tests for provider adapters (OpenAI, Anthropic, Azure)."""

import os
import sys
from unittest.mock import Mock, patch

import pytest


def test_openai_adapter_import():
    """Test that OpenAI adapter can be imported."""
    try:
        from agentnet.providers.openai import OpenAIAdapter
        assert OpenAIAdapter is not None
    except ImportError as e:
        pytest.skip(f"OpenAI adapter requires openai package: {e}")


def test_anthropic_adapter_import():
    """Test that Anthropic adapter can be imported."""
    try:
        from agentnet.providers.anthropic import AnthropicAdapter
        assert AnthropicAdapter is not None
    except ImportError as e:
        pytest.skip(f"Anthropic adapter requires anthropic package: {e}")


def test_azure_adapter_import():
    """Test that Azure adapter can be imported."""
    try:
        from agentnet.providers.azure import AzureOpenAIAdapter
        assert AzureOpenAIAdapter is not None
    except ImportError as e:
        pytest.skip(f"Azure adapter requires openai package: {e}")


def test_openai_adapter_config():
    """Test OpenAI adapter configuration."""
    try:
        from agentnet.providers.openai import OpenAIAdapter
    except ImportError:
        pytest.skip("OpenAI adapter not available")

    # Test with explicit config
    adapter = OpenAIAdapter(config={
        "api_key": "test-key",
        "model": "gpt-4",
        "organization": "test-org"
    })
    
    assert adapter.api_key == "test-key"
    assert adapter.model == "gpt-4"
    assert adapter.organization == "test-org"
    assert adapter.validate_config() is True

    # Test without API key
    adapter_no_key = OpenAIAdapter(config={})
    assert adapter_no_key.validate_config() is False


def test_anthropic_adapter_config():
    """Test Anthropic adapter configuration."""
    try:
        from agentnet.providers.anthropic import AnthropicAdapter
    except ImportError:
        pytest.skip("Anthropic adapter not available")

    # Test with explicit config
    adapter = AnthropicAdapter(config={
        "api_key": "test-key",
        "model": "claude-3-opus-20240229"
    })
    
    assert adapter.api_key == "test-key"
    assert adapter.model == "claude-3-opus-20240229"
    assert adapter.validate_config() is True

    # Test without API key
    adapter_no_key = AnthropicAdapter(config={})
    assert adapter_no_key.validate_config() is False


def test_azure_adapter_config():
    """Test Azure OpenAI adapter configuration."""
    try:
        from agentnet.providers.azure import AzureOpenAIAdapter
    except ImportError:
        pytest.skip("Azure adapter not available")

    # Test with explicit config
    adapter = AzureOpenAIAdapter(config={
        "api_key": "test-key",
        "endpoint": "https://test.openai.azure.com/",
        "deployment": "gpt-4",
        "api_version": "2024-02-15-preview"
    })
    
    assert adapter.api_key == "test-key"
    assert adapter.endpoint == "https://test.openai.azure.com/"
    assert adapter.deployment == "gpt-4"
    assert adapter.validate_config() is True

    # Test without API key or endpoint
    adapter_no_config = AzureOpenAIAdapter(config={})
    assert adapter_no_config.validate_config() is False


def test_openai_provider_info():
    """Test OpenAI provider info."""
    try:
        from agentnet.providers.openai import OpenAIAdapter
    except ImportError:
        pytest.skip("OpenAI adapter not available")

    adapter = OpenAIAdapter(config={"api_key": "test-key"})
    info = adapter.get_provider_info()
    
    assert info["name"] == "OpenAI"
    assert info["supports_streaming"] is True
    assert "gpt-4" in info["models"]
    assert info["configured"] is True


def test_anthropic_provider_info():
    """Test Anthropic provider info."""
    try:
        from agentnet.providers.anthropic import AnthropicAdapter
    except ImportError:
        pytest.skip("Anthropic adapter not available")

    adapter = AnthropicAdapter(config={"api_key": "test-key"})
    info = adapter.get_provider_info()
    
    assert info["name"] == "Anthropic"
    assert info["supports_streaming"] is True
    assert "claude-3-opus" in info["models"][0]
    assert info["configured"] is True


def test_azure_provider_info():
    """Test Azure provider info."""
    try:
        from agentnet.providers.azure import AzureOpenAIAdapter
    except ImportError:
        pytest.skip("Azure adapter not available")

    adapter = AzureOpenAIAdapter(config={
        "api_key": "test-key",
        "endpoint": "https://test.openai.azure.com/"
    })
    info = adapter.get_provider_info()
    
    assert info["name"] == "Azure OpenAI"
    assert info["supports_streaming"] is True
    assert info["configured"] is True


def test_openai_cost_calculation():
    """Test OpenAI cost calculation."""
    try:
        from agentnet.providers.openai import OpenAIAdapter
    except ImportError:
        pytest.skip("OpenAI adapter not available")

    adapter = OpenAIAdapter(config={"api_key": "test-key"})
    
    result = {
        "model": "gpt-4",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
    
    cost_info = adapter.get_cost_info(result)
    
    assert cost_info["provider"] == "openai"
    assert cost_info["tokens"] == 150
    assert cost_info["cost"] > 0
    assert "breakdown" in cost_info


def test_anthropic_cost_calculation():
    """Test Anthropic cost calculation."""
    try:
        from agentnet.providers.anthropic import AnthropicAdapter
    except ImportError:
        pytest.skip("Anthropic adapter not available")

    adapter = AnthropicAdapter(config={"api_key": "test-key"})
    
    result = {
        "model": "claude-3-opus-20240229",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    }
    
    cost_info = adapter.get_cost_info(result)
    
    assert cost_info["provider"] == "anthropic"
    assert cost_info["tokens"] == 150
    assert cost_info["cost"] > 0
    assert "breakdown" in cost_info


def test_azure_cost_calculation():
    """Test Azure cost calculation."""
    try:
        from agentnet.providers.azure import AzureOpenAIAdapter
    except ImportError:
        pytest.skip("Azure adapter not available")

    adapter = AzureOpenAIAdapter(config={
        "api_key": "test-key",
        "endpoint": "https://test.openai.azure.com/"
    })
    
    result = {
        "model": "gpt-4",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
    
    cost_info = adapter.get_cost_info(result)
    
    assert cost_info["provider"] == "azure"
    assert cost_info["tokens"] == 150
    assert cost_info["cost"] > 0
    assert "breakdown" in cost_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
