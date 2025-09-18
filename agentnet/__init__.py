"""AgentNet: A governed multi-agent reasoning platform.

This package provides policy-aware LLM orchestration, dialogue strategies
(debate/brainstorm/consensus), task graph execution, memory layers, monitoring,
and extensible tool & provider adapters.
"""

# Phase 0 Core imports - Essential functionality only
from .core.agent import AgentNet
from .core.types import CognitiveFault, Severity
from .monitors.base import MonitorSpec
from .monitors.factory import MonitorFactory
from .monitors.manager import MonitorManager
from .persistence.session import SessionManager
from .providers.base import ProviderAdapter
from .providers.example import ExampleEngine

# Phase 1+ imports (with graceful degradation for missing dependencies)
_p2_available = True
_p3_available = True
_p4_available = True
_p5_available = True
_p6_available = True

try:
    # Import P2 features: Memory & Tools
    from .memory.base import MemoryEntry, MemoryLayer, MemoryRetrieval, MemoryType
    from .memory.episodic import EpisodicMemory
    from .memory.manager import MemoryManager
    from .memory.semantic import SemanticMemory
    from .memory.short_term import ShortTermMemory
    from .tools.base import Tool, ToolResult, ToolSpec, ToolStatus
    from .tools.examples import (
        CalculatorTool,
        FileWriteTool,
        StatusCheckTool,
        WebSearchTool,
    )
    from .tools.executor import ToolExecutor
    from .tools.registry import ToolRegistry
except ImportError:
    _p2_available = False
    # Stub classes for P2 functionality
    MemoryManager = None
    MemoryLayer = MemoryEntry = MemoryRetrieval = MemoryType = None
    ShortTermMemory = EpisodicMemory = SemanticMemory = None
    ToolRegistry = ToolExecutor = None
    Tool = ToolResult = ToolSpec = ToolStatus = None
    WebSearchTool = CalculatorTool = FileWriteTool = StatusCheckTool = None

try:
    # Import P3 features: DAG & Eval
    from .core.eval import (
        EvaluationMetrics,
        EvaluationRunner,
        EvaluationScenario,
        EvaluationSuite,
        MetricsCalculator,
        SuccessCriteria,
    )
    from .core.orchestration import (
        DAGPlanner,
        ExecutionResult,
        TaskGraph,
        TaskNode,
        TaskScheduler,
    )
except ImportError:
    _p3_available = False
    # Stub classes for P3 functionality
    DAGPlanner = TaskNode = TaskGraph = TaskScheduler = ExecutionResult = None
    EvaluationRunner = EvaluationScenario = EvaluationSuite = None
    MetricsCalculator = EvaluationMetrics = SuccessCriteria = None

try:
    # Import P4 features: Governance++, Cost Engine, RBAC
    from .core.auth import AuthMiddleware, Permission, RBACManager, Role, User
    from .core.cost import (
        CostAggregator,
        CostRecorder,
        PricingEngine,
        TenantCostTracker,
    )
except ImportError:
    _p4_available = False
    # Stub classes for P4 functionality
    PricingEngine = CostRecorder = CostAggregator = TenantCostTracker = None
    Role = Permission = RBACManager = User = AuthMiddleware = None

try:
    # Import P5 features: Observability
    from .observability import (
        AgentNetMetrics,
        MetricsCollector,
        TracingManager,
        create_tracer,
        get_correlation_logger,
        setup_structured_logging,
    )
except ImportError:
    _p5_available = False
    # Stub functions for P5 functionality
    MetricsCollector = AgentNetMetrics = TracingManager = create_tracer = None
    setup_structured_logging = get_correlation_logger = None

try:
    # Import P6 features: Enterprise Hardening
    from .audit import AuditDashboard, AuditLogger, AuditStorage, AuditWorkflow
    from .compliance import (
        ComplianceReporter,
        ContentRedactor,
        DataClassifier,
        ExportControlManager,
    )
    from .plugins import (
        Plugin,
        PluginInfo,
        PluginManager,
        PluginSandbox,
        SecurityPolicy,
    )
except ImportError:
    _p6_available = False
    # Stub classes for P6 functionality
    ExportControlManager = DataClassifier = ContentRedactor = ComplianceReporter = None
    AuditWorkflow = AuditLogger = AuditStorage = AuditDashboard = None
    PluginManager = Plugin = PluginInfo = PluginSandbox = SecurityPolicy = None

__version__ = "0.5.0"

# Phase availability flags
__phase_status__ = {
    "P0": True,  # Core functionality always available
    "P1": True,  # Basic multi-agent features
    "P2": _p2_available,  # Memory & Tools
    "P3": _p3_available,  # DAG & Eval
    "P4": _p4_available,  # Governance++
    "P5": _p5_available,  # Observability
    "P6": _p6_available,  # Enterprise Hardening
}

# Build dynamic __all__ based on available features
__all__ = [
    # Phase 0 Core - Always available
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

# Add phase-specific exports if available
if _p2_available:
    __all__.extend(
        [
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
        ]
    )

if _p3_available:
    __all__.extend(
        [
            # DAG & Eval
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
        ]
    )

if _p4_available:
    __all__.extend(
        [
            # Cost Engine & RBAC
            "PricingEngine",
            "CostRecorder",
            "CostAggregator",
            "TenantCostTracker",
            "Role",
            "Permission",
            "RBACManager",
            "User",
            "AuthMiddleware",
        ]
    )

if _p5_available:
    __all__.extend(
        [
            # Observability
            "MetricsCollector",
            "AgentNetMetrics",
            "TracingManager",
            "create_tracer",
            "setup_structured_logging",
            "get_correlation_logger",
        ]
    )

if _p6_available:
    __all__.extend(
        [
            # Enterprise Hardening
            "ExportControlManager",
            "DataClassifier",
            "ContentRedactor",
            "ComplianceReporter",
            "AuditWorkflow",
            "AuditLogger",
            "AuditStorage",
            "AuditDashboard",
            "PluginManager",
            "Plugin",
            "PluginInfo",
            "PluginSandbox",
            "SecurityPolicy",
        ]
    )

# Add phase status to __all__ for introspection
__all__.append("__phase_status__")
