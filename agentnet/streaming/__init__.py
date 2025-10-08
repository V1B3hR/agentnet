"""
Streaming Partial-Output Collaboration for AgentNet

Implements streaming response handling with partial JSON parsing
and real-time collaboration features as specified in Phase 6 requirements.
"""

from .collaboration import (
    StreamingCollaborator,
    CollaborationSession,
    PartialResponse,
    CollaborationMode,
    PartialResponseType,
)
from .parser import PartialJSONParser, StreamingParser, ParseResult
from .handlers import (
    StreamHandler,
    CollaborationHandler,
    ErrorHandler,
    FilterHandler,
    TransformHandler,
    HandlerType,
)

# Phase 6 enhanced features with graceful fallback
try:
    from .enhanced_collaboration import (
        EnhancedStreamingCollaborator,
        StreamingIntervention,
        StreamingMetrics,
        InterventionType,
        InterventionTrigger,
        coherence_monitor,
        relevance_monitor,
        safety_monitor,
    )

    _ENHANCED_AVAILABLE = True
except ImportError:
    # Stub classes if dependencies are missing
    EnhancedStreamingCollaborator = None
    StreamingIntervention = StreamingMetrics = None
    InterventionType = InterventionTrigger = None
    coherence_monitor = relevance_monitor = safety_monitor = None
    _ENHANCED_AVAILABLE = False

__all__ = [
    "StreamingCollaborator",
    "CollaborationSession",
    "PartialResponse",
    "CollaborationMode",
    "PartialResponseType",
    "PartialJSONParser",
    "StreamingParser",
    "ParseResult",
    "StreamHandler",
    "CollaborationHandler",
    "ErrorHandler",
    "FilterHandler",
    "TransformHandler",
    "HandlerType",
]

# Add enhanced features if available
if _ENHANCED_AVAILABLE:
    __all__.extend(
        [
            "EnhancedStreamingCollaborator",
            "StreamingIntervention",
            "StreamingMetrics",
            "InterventionType",
            "InterventionTrigger",
            "coherence_monitor",
            "relevance_monitor",
            "safety_monitor",
        ]
    )
