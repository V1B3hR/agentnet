"""Memory system for AgentNet.

Provides multi-layer memory architecture:
- Short-term: In-memory sliding window
- Episodic: Persisted conversation chunks with tags
- Semantic: Vector similarity search for content retrieval
- Retention: Policies for memory lifecycle management
"""

from .base import MemoryLayer, MemoryRetrieval, MemoryEntry, MemoryType
from .episodic import EpisodicMemory
from .manager import MemoryManager
from .semantic import SemanticMemory
from .short_term import ShortTermMemory

# Phase 3 retention features with graceful fallback
try:
    from .retention import (
        RetentionManager, RetentionPolicy, RetentionStrategy,
        LRURetentionPolicy, LFURetentionPolicy, SemanticSalienceRetentionPolicy,
        TimeDecayRetentionPolicy, HybridRetentionPolicy
    )
    _RETENTION_AVAILABLE = True
except ImportError:
    RetentionManager = RetentionPolicy = RetentionStrategy = None
    LRURetentionPolicy = LFURetentionPolicy = None
    SemanticSalienceRetentionPolicy = TimeDecayRetentionPolicy = None
    HybridRetentionPolicy = None
    _RETENTION_AVAILABLE = False

__all__ = [
    "MemoryLayer",
    "MemoryRetrieval",
    "MemoryEntry",
    "MemoryType",
    "ShortTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "MemoryManager",
]

# Add retention exports if available
if _RETENTION_AVAILABLE:
    __all__.extend([
        "RetentionManager",
        "RetentionPolicy", 
        "RetentionStrategy",
        "LRURetentionPolicy",
        "LFURetentionPolicy",
        "SemanticSalienceRetentionPolicy",
        "TimeDecayRetentionPolicy",
        "HybridRetentionPolicy",
    ])
