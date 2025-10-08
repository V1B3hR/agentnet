"""Provider adapters for AgentNet."""

from .base import ProviderAdapter
from .example import ExampleEngine

# Optional providers - only export if dependencies are available
try:
    from .openai_provider import OpenAIProvider
    _has_openai = True
except ImportError:
    OpenAIProvider = None
    _has_openai = False

try:
    from .anthropic_provider import AnthropicProvider
    _has_anthropic = True
except ImportError:
    AnthropicProvider = None
    _has_anthropic = False

# Build exports list dynamically
__all__ = [
    "ProviderAdapter",
    "ExampleEngine",
]

if _has_openai:
    __all__.append("OpenAIProvider")

if _has_anthropic:
    __all__.append("AnthropicProvider")


def get_available_providers():
    """Return list of available provider names."""
    providers = ["example"]
    if _has_openai:
        providers.append("openai")
    if _has_anthropic:
        providers.append("anthropic")
    return providers
