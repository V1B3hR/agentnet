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

# Import P2 features: Memory & Tools
from .memory.manager import MemoryManager
from .memory.base import MemoryLayer, MemoryEntry, MemoryRetrieval, MemoryType
from .memory.short_term import ShortTermMemory
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .tools.registry import ToolRegistry
from .tools.executor import ToolExecutor
from .tools.base import Tool, ToolResult, ToolSpec, ToolStatus
from .tools.examples import WebSearchTool, CalculatorTool, FileWriteTool, StatusCheckTool

# Import P3 features: DAG & Eval
from .core.orchestration import DAGPlanner, TaskNode, TaskGraph, TaskScheduler, ExecutionResult
from .core.eval import EvaluationRunner, EvaluationScenario, EvaluationSuite, MetricsCalculator, EvaluationMetrics, SuccessCriteria

# Import P4 features: Governance++, Cost Engine, RBAC
from .core.cost import PricingEngine, CostRecorder, CostAggregator, TenantCostTracker
from .core.auth import Role, Permission, RBACManager, User, AuthMiddleware

__version__ = "0.4.0"
__all__ = [
    # Core
    "AgentNet",
    "Severity", 
    "CognitiveFault",
    "MonitorFactory", 
    "MonitorManager",
    "MonitorSpec",
    "SessionManager",
    "ProviderAdapter", 
    "ExampleEngine",
    # Memory
    "MemoryManager",
    "MemoryLayer",
    "MemoryEntry", 
    "MemoryRetrieval",
    "MemoryType",
    "ShortTermMemory",
    "EpisodicMemory", 
    "SemanticMemory",
    # Tools
    "ToolRegistry",
    "ToolExecutor",
    "Tool",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
    "WebSearchTool",
    "CalculatorTool",
    "FileWriteTool",
    "StatusCheckTool",
    # P3: DAG & Eval
    "DAGPlanner",
    "TaskNode",
    "TaskGraph", 
    "TaskScheduler",
    "ExecutionResult",
    "EvaluationRunner",
    "EvaluationScenario",
    "EvaluationSuite",
    "MetricsCalculator",
    "EvaluationMetrics",
    "SuccessCriteria",
    # P4 Cost Engine
    "PricingEngine",
    "CostRecorder",
    "CostAggregator", 
    "TenantCostTracker",
    # P4 RBAC
    "Role",
    "Permission",
    "RBACManager",
    "User",
    "AuthMiddleware",
]
