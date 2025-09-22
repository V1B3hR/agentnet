"""Core AgentNet components."""

from .auth import AuthMiddleware, Permission, RBACManager, Role, User
from .cost import CostAggregator, CostRecorder, PricingEngine, TenantCostTracker
from .engine import BaseEngine
from .types import CognitiveFault, Severity

# Phase 3 & 4 features with graceful fallback
try:
    from .cache import CacheManager, CacheEntry, InMemoryCache, FileCache
    from .telemetry import (
        TelemetryCollector, TelemetryEvent, EventType, MetricType,
        init_telemetry, get_telemetry, record_event, record_metric
    )
    from .multimodal import (
        MultiModalManager, MultiModalMessage, ModalityContent, ModalityType,
        ModalityProcessor, get_multimodal_manager
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
    __all__.extend([
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
    ])
