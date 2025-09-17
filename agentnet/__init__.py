"""AgentNet: A governed multi-agent reasoning platform.

This package provides policy-aware LLM orchestration, dialogue strategies 
(debate/brainstorm/consensus), task graph execution, memory layers, monitoring, 
and extensible tool & provider adapters.
"""

# Import refactored modules
from .core.agent import AgentNet
from .core.types import Severity, CognitiveFault
from .monitors.factory import MonitorFactory
from .monitors.manager import MonitorManager
from .monitors.base import MonitorSpec
from .persistence.session import SessionManager
from .providers.base import ProviderAdapter
from .providers.example import ExampleEngine

__version__ = "0.1.0"
__all__ = [
    "AgentNet",
    "Severity", 
    "CognitiveFault",
    "MonitorFactory", 
    "MonitorManager",
    "MonitorSpec",
    "SessionManager",
    "ProviderAdapter", 
    "ExampleEngine",
]
