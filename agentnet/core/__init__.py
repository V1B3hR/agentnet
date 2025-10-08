"""Core AgentNet components."""

from .auth import AuthMiddleware, Permission, RBACManager, Role, User
from .cost import CostAggregator, CostRecorder, PricingEngine, TenantCostTracker
from .engine import BaseEngine
from .types import CognitiveFault, Severity

# Phase 3 & 4 features with graceful fallback
try:
    from .cache import CacheManager, CacheEntry, InMemoryCache, FileCache
    from .telemetry import (
        TelemetryCollector,
        TelemetryEvent,
        EventType,
        MetricType,
        init_telemetry,
        get_telemetry,
        record_event,
        record_metric,
    )
    from .multimodal import (
        MultiModalManager,
        MultiModalMessage,
        ModalityContent,
        ModalityType,
        ModalityProcessor,
        get_multimodal_manager,
    )

    _PHASE3_4_AVAILABLE = True
except ImportError:
    # Stub classes if dependencies are missing
    CacheManager = CacheEntry = InMemoryCache = FileCache = None
    TelemetryCollector = TelemetryEvent = EventType = MetricType = None
    init_telemetry = get_telemetry = record_event = record_metric = None
    MultiModalManager = MultiModalMessage = ModalityContent = ModalityType = None
    ModalityProcessor = get_multimodal_manager = None
    _PHASE3_4_AVAILABLE = False

# Phase 6 advanced features with graceful fallback
try:
    from .metacontroller import (
        MetaController,
        AgentRole,
        ReconfigurationTrigger,
        AgentNode,
        ReconfigurationEvent,
    )
    from .human_loop import (
        HumanApprovalGate,
        ApprovalStatus,
        EscalationLevel,
        RiskLevel,
        ApprovalRequest,
        Approver,
    )
    from .reward_modeling import (
        RewardModel,
        FeedbackType,
        RewardSignal,
        FeedbackEntry,
        EvaluationBatch,
    )
    from .adaptive_orchestration import (
        PerformanceFeedbackCollector,
        AdaptiveOrchestrator,
        PerformanceMetric,
        OrchestrationStrategy,
        OptimizationObjective,
        PerformanceSnapshot,
        StrategyPerformanceProfile,
    )
    from .multilingual_safety import (
        MultiLingualPolicyTranslator,
        SupportedLanguage,
        PolicyViolationType,
        SafetyRule,
        PolicyViolation,
        LanguageDetector,
    )

    _PHASE6_AVAILABLE = True
except ImportError:
    # Stub classes if dependencies are missing
    MetaController = AgentRole = ReconfigurationTrigger = AgentNode = (
        ReconfigurationEvent
    ) = None
    HumanApprovalGate = ApprovalStatus = EscalationLevel = RiskLevel = None
    ApprovalRequest = Approver = None
    RewardModel = FeedbackType = RewardSignal = FeedbackEntry = EvaluationBatch = None
    PerformanceFeedbackCollector = AdaptiveOrchestrator = None
    PerformanceMetric = OrchestrationStrategy = OptimizationObjective = None
    PerformanceSnapshot = StrategyPerformanceProfile = None
    MultiLingualPolicyTranslator = SupportedLanguage = PolicyViolationType = None
    SafetyRule = PolicyViolation = LanguageDetector = None
    _PHASE6_AVAILABLE = False

__all__ = [
    "Severity",
    "CognitiveFault",
    "BaseEngine",
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

# Add Phase 3 & 4 exports if available
if _PHASE3_4_AVAILABLE:
    __all__.extend(
        [
            # Cache
            "CacheManager",
            "CacheEntry",
            "InMemoryCache",
            "FileCache",
            # Telemetry
            "TelemetryCollector",
            "TelemetryEvent",
            "EventType",
            "MetricType",
            "init_telemetry",
            "get_telemetry",
            "record_event",
            "record_metric",
            # Multi-modal
            "MultiModalManager",
            "MultiModalMessage",
            "ModalityContent",
            "ModalityType",
            "ModalityProcessor",
            "get_multimodal_manager",
        ]
    )

# Add Phase 6 exports if available
if _PHASE6_AVAILABLE:
    __all__.extend(
        [
            # Meta-controller
            "MetaController",
            "AgentRole",
            "ReconfigurationTrigger",
            "AgentNode",
            "ReconfigurationEvent",
            # Human-in-loop
            "HumanApprovalGate",
            "ApprovalStatus",
            "EscalationLevel",
            "RiskLevel",
            "ApprovalRequest",
            "Approver",
            # Reward modeling
            "RewardModel",
            "FeedbackType",
            "RewardSignal",
            "FeedbackEntry",
            "EvaluationBatch",
            # Adaptive orchestration
            "PerformanceFeedbackCollector",
            "AdaptiveOrchestrator",
            "PerformanceMetric",
            "OrchestrationStrategy",
            "OptimizationObjective",
            "PerformanceSnapshot",
            "StrategyPerformanceProfile",
            # Multi-lingual safety
            "MultiLingualPolicyTranslator",
            "SupportedLanguage",
            "PolicyViolationType",
            "SafetyRule",
            "PolicyViolation",
            "LanguageDetector",
        ]
    )
