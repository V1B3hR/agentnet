"""Provider adapters for AgentNet."""

from .base import ProviderAdapter
from .example import ExampleEngine
from .instrumented import (
    InstrumentedProviderAdapter,
    InstrumentedProviderMixin,
    instrument_provider,
)

# Import concrete provider implementations
# These are optional and will only work if their dependencies are installed
try:
    from .openai_adapter import OpenAIAdapter
    _openai_available = True
except ImportError:
    OpenAIAdapter = None
    _openai_available = False

try:
    from .anthropic_adapter import AnthropicAdapter
    _anthropic_available = True
except ImportError:
    AnthropicAdapter = None
    _anthropic_available = False

try:
    from .azure_openai_adapter import AzureOpenAIAdapter
    _azure_available = True
except ImportError:
    AzureOpenAIAdapter = None
    _azure_available = False

try:
    from .huggingface_adapter import HuggingFaceAdapter
    _huggingface_available = True
except ImportError:
    HuggingFaceAdapter = None
    _huggingface_available = False

__all__ = [
    "ProviderAdapter",
    "ExampleEngine",
    "InstrumentedProviderAdapter",
    "InstrumentedProviderMixin",
    "instrument_provider",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "AzureOpenAIAdapter",
    "HuggingFaceAdapter",
]

