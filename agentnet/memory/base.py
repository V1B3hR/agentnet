"""Base memory interfaces and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MemoryType(str, Enum):
    """Types of memory layers."""

    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryEntry:
    """A single memory entry."""

    content: str
    metadata: Dict[str, Any]
    timestamp: float
    agent_name: Optional[str] = None
    tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None


@dataclass
class MemoryRetrieval:
    """Result of memory retrieval operation."""

    entries: List[MemoryEntry]
    total_tokens: int
    retrieval_time: float
    source_layers: List[MemoryType]


class MemoryLayer(ABC):
    """Abstract base class for memory layers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        pass

    @abstractmethod
    def retrieve(
        self, query: str, max_entries: int = 10, threshold: float = 0.7, **kwargs
    ) -> List[MemoryEntry]:
        """Retrieve relevant memory entries."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries in this layer."""
        pass

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """Return the type of this memory layer."""
        pass
