"""Provider adapters for AgentNet."""

from .base import ProviderAdapter
from .example import ExampleEngine
from .instrumented import (
    InstrumentedProviderAdapter,
    InstrumentedProviderMixin,
    instrument_provider,
)

# Import optional provider adapters (may require additional dependencies)
try:
    from .openai import OpenAIAdapter
    _has_openai = True
except ImportError:
    OpenAIAdapter = None
    _has_openai = False

try:
    from .anthropic import AnthropicAdapter
    _has_anthropic = True
except ImportError:
    AnthropicAdapter = None
    _has_anthropic = False

try:
    from .azure import AzureOpenAIAdapter
    _has_azure = True
except ImportError:
    AzureOpenAIAdapter = None
    _has_azure = False

__all__ = [
    "ProviderAdapter",
    "ExampleEngine",
    "InstrumentedProviderAdapter",
    "InstrumentedProviderMixin",
    "instrument_provider",
]

# Add optional providers to __all__ if available
if _has_openai:
    __all__.append("OpenAIAdapter")
if _has_anthropic:
    __all__.append("AnthropicAdapter")
if _has_azure:
    __all__.append("AzureOpenAIAdapter")
