"""Memory system for AgentNet.

Provides multi-layer memory architecture:
- Short-term: In-memory sliding window
- Episodic: Persisted conversation chunks with tags
- Semantic: Vector similarity search for content retrieval
- Retention: Policies for memory lifecycle management
- Enhanced: Phase 7 enhanced memory with temporal reasoning and cross-modal linking
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

# Phase 7 enhanced memory features with graceful fallback
try:
    from .enhanced import (
        EnhancedEpisodicMemory,
        HierarchicalKnowledgeOrganizer,
        CrossModalMemoryLinker,
        MemoryConsolidationEngine,
        ModalityType,
        ConsolidationStrategy,
        CrossModalLink,
        MemoryCluster,
        ConsolidationRule
    )
    _ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    EnhancedEpisodicMemory = None
    HierarchicalKnowledgeOrganizer = None
    CrossModalMemoryLinker = None
    MemoryConsolidationEngine = None
    ModalityType = ConsolidationStrategy = None
    CrossModalLink = MemoryCluster = ConsolidationRule = None
    _ENHANCED_MEMORY_AVAILABLE = False

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

# Add Phase 7 enhanced memory exports if available
if _ENHANCED_MEMORY_AVAILABLE:
    __all__.extend([
        "EnhancedEpisodicMemory",
        "HierarchicalKnowledgeOrganizer",
        "CrossModalMemoryLinker",
        "MemoryConsolidationEngine",
        "ModalityType",
        "ConsolidationStrategy",
        "CrossModalLink",
        "MemoryCluster",
        "ConsolidationRule",
    ])
