"""Memory retention policies for AgentNet."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .base import MemoryEntry, MemoryType

logger = logging.getLogger("agentnet.memory.retention")


class RetentionStrategy(str, Enum):
    """Memory retention strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SEMANTIC_SALIENCE = "semantic_salience"  # Based on semantic importance
    TIME_DECAY = "time_decay"  # Based on age with decay function
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class RetentionMetrics:
    """Metrics for retention decisions."""

    access_count: int = 0
    last_accessed: float = 0.0
    creation_time: float = 0.0
    semantic_score: float = 0.0
    importance_score: float = 0.0
    decay_factor: float = 1.0

    def age_seconds(self) -> float:
        """Calculate age in seconds."""
        return time.time() - self.creation_time

    def recency_score(self) -> float:
        """Calculate recency score (higher = more recent)."""
        age = self.age_seconds()
        if age <= 0:
            return 1.0
        # Exponential decay with half-life of 1 day
        half_life = 86400  # 24 hours
        return 2 ** (-age / half_life)


class RetentionPolicy(ABC):
    """Abstract base class for memory retention policies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_entries = config.get("max_entries", 1000)
        self.metrics: Dict[str, RetentionMetrics] = {}

    @abstractmethod
    def should_retain(self, entry: MemoryEntry, entry_id: str) -> bool:
        """Determine if an entry should be retained."""
        pass

    @abstractmethod
    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], count: int
    ) -> List[str]:
        """Select entries for eviction."""
        pass

    def update_access(self, entry_id: str) -> None:
        """Update access metrics for an entry."""
        if entry_id not in self.metrics:
            self.metrics[entry_id] = RetentionMetrics(creation_time=time.time())

        metrics = self.metrics[entry_id]
        metrics.access_count += 1
        metrics.last_accessed = time.time()

    def add_entry(self, entry: MemoryEntry, entry_id: str) -> None:
        """Add metrics for a new entry."""
        self.metrics[entry_id] = RetentionMetrics(
            creation_time=entry.timestamp,
            semantic_score=self._calculate_semantic_score(entry),
            importance_score=self._calculate_importance_score(entry),
        )

    def remove_entry(self, entry_id: str) -> None:
        """Remove metrics for an entry."""
        if entry_id in self.metrics:
            del self.metrics[entry_id]

    def _calculate_semantic_score(self, entry: MemoryEntry) -> float:
        """Calculate semantic importance score."""
        # Base implementation - can be overridden
        score = 0.0

        # Length factor (longer content often more important)
        content_length = len(entry.content)
        score += min(content_length / 1000, 1.0) * 0.3

        # Tag-based importance
        if entry.tags:
            important_tags = {"decision", "error", "warning", "critical", "summary"}
            tag_score = len(set(entry.tags) & important_tags) / len(important_tags)
            score += tag_score * 0.4

        # Agent importance (some agents may be more important)
        if entry.agent_name:
            # This could be configured per agent
            score += 0.3

        return min(score, 1.0)

    def _calculate_importance_score(self, entry: MemoryEntry) -> float:
        """Calculate general importance score."""
        score = 0.0

        # Check for important keywords
        important_keywords = {
            "error",
            "critical",
            "fail",
            "success",
            "decision",
            "conclusion",
            "result",
            "summary",
            "important",
            "warning",
            "alert",
        }

        content_lower = entry.content.lower()
        keyword_matches = sum(
            1 for keyword in important_keywords if keyword in content_lower
        )
        score += min(keyword_matches / len(important_keywords), 1.0) * 0.5

        # Metadata importance
        if entry.metadata:
            priority = entry.metadata.get("priority", "normal")
            if priority == "high":
                score += 0.3
            elif priority == "critical":
                score += 0.5

        return min(score, 1.0)


class LRURetentionPolicy(RetentionPolicy):
    """Least Recently Used retention policy."""

    def should_retain(self, entry: MemoryEntry, entry_id: str) -> bool:
        """LRU keeps entries based on recent access."""
        return True  # Let eviction handle the logic

    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], count: int
    ) -> List[str]:
        """Select least recently used entries for eviction."""
        if count <= 0:
            return []

        # Sort by last accessed time (oldest first)
        entry_ids = list(entries.keys())
        entry_ids.sort(
            key=lambda eid: self.metrics.get(eid, RetentionMetrics()).last_accessed
        )

        return entry_ids[:count]


class LFURetentionPolicy(RetentionPolicy):
    """Least Frequently Used retention policy."""

    def should_retain(self, entry: MemoryEntry, entry_id: str) -> bool:
        """LFU keeps entries based on access frequency."""
        return True  # Let eviction handle the logic

    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], count: int
    ) -> List[str]:
        """Select least frequently used entries for eviction."""
        if count <= 0:
            return []

        # Sort by access count (least accessed first)
        entry_ids = list(entries.keys())
        entry_ids.sort(
            key=lambda eid: self.metrics.get(eid, RetentionMetrics()).access_count
        )

        return entry_ids[:count]


class SemanticSalienceRetentionPolicy(RetentionPolicy):
    """Semantic salience-based retention policy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_semantic_score = config.get("min_semantic_score", 0.3)
        self.min_importance_score = config.get("min_importance_score", 0.2)

    def should_retain(self, entry: MemoryEntry, entry_id: str) -> bool:
        """Retain entries based on semantic importance."""
        if entry_id not in self.metrics:
            self.add_entry(entry, entry_id)

        metrics = self.metrics[entry_id]
        return (
            metrics.semantic_score >= self.min_semantic_score
            or metrics.importance_score >= self.min_importance_score
        )

    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], count: int
    ) -> List[str]:
        """Select entries with lowest semantic salience for eviction."""
        if count <= 0:
            return []

        # Calculate combined score for each entry
        def combined_score(entry_id: str) -> float:
            metrics = self.metrics.get(entry_id, RetentionMetrics())
            return metrics.semantic_score * 0.6 + metrics.importance_score * 0.4

        # Sort by combined score (lowest first)
        entry_ids = list(entries.keys())
        entry_ids.sort(key=combined_score)

        return entry_ids[:count]


class TimeDecayRetentionPolicy(RetentionPolicy):
    """Time-based decay retention policy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_age_seconds = config.get("max_age_seconds", 7 * 24 * 3600)  # 7 days
        self.decay_half_life = config.get("decay_half_life", 24 * 3600)  # 1 day

    def should_retain(self, entry: MemoryEntry, entry_id: str) -> bool:
        """Retain entries based on age and decay."""
        if entry_id not in self.metrics:
            self.add_entry(entry, entry_id)

        metrics = self.metrics[entry_id]
        age = metrics.age_seconds()

        # Always evict if too old
        if age > self.max_age_seconds:
            return False

        # Keep recent entries
        if age < self.decay_half_life:
            return True

        # Use decay function combined with importance
        decay_score = metrics.recency_score()
        importance_boost = metrics.importance_score * 0.5

        return (decay_score + importance_boost) > 0.3

    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], count: int
    ) -> List[str]:
        """Select oldest entries for eviction."""
        if count <= 0:
            return []

        # Calculate retention score for each entry
        def retention_score(entry_id: str) -> float:
            metrics = self.metrics.get(entry_id, RetentionMetrics())
            return metrics.recency_score() + metrics.importance_score * 0.3

        # Sort by retention score (lowest first)
        entry_ids = list(entries.keys())
        entry_ids.sort(key=retention_score)

        return entry_ids[:count]


class HybridRetentionPolicy(RetentionPolicy):
    """Hybrid retention policy combining multiple strategies."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weights = config.get(
            "weights", {"recency": 0.3, "frequency": 0.3, "semantic": 0.4}
        )
        self.min_retention_score = config.get("min_retention_score", 0.4)

    def should_retain(self, entry: MemoryEntry, entry_id: str) -> bool:
        """Retain entries based on hybrid score."""
        if entry_id not in self.metrics:
            self.add_entry(entry, entry_id)

        score = self._calculate_hybrid_score(entry_id)
        return score >= self.min_retention_score

    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], count: int
    ) -> List[str]:
        """Select entries with lowest hybrid scores for eviction."""
        if count <= 0:
            return []

        # Sort by hybrid score (lowest first)
        entry_ids = list(entries.keys())
        entry_ids.sort(key=self._calculate_hybrid_score)

        return entry_ids[:count]

    def _calculate_hybrid_score(self, entry_id: str) -> float:
        """Calculate hybrid retention score."""
        metrics = self.metrics.get(entry_id, RetentionMetrics())

        # Normalize frequency score
        max_access = max((m.access_count for m in self.metrics.values()), default=1)
        frequency_score = metrics.access_count / max_access if max_access > 0 else 0

        # Combine scores
        score = (
            self.weights.get("recency", 0.3) * metrics.recency_score()
            + self.weights.get("frequency", 0.3) * frequency_score
            + self.weights.get("semantic", 0.4) * metrics.semantic_score
        )

        return score


class RetentionManager:
    """Manager for memory retention policies."""

    def __init__(self, strategy: RetentionStrategy = RetentionStrategy.HYBRID):
        self.strategy = strategy
        self.policies: Dict[MemoryType, RetentionPolicy] = {}

    def set_policy(self, memory_type: MemoryType, policy: RetentionPolicy) -> None:
        """Set retention policy for a memory type."""
        self.policies[memory_type] = policy
        logger.info(f"Set {policy.__class__.__name__} for {memory_type}")

    def get_policy(self, memory_type: MemoryType) -> Optional[RetentionPolicy]:
        """Get retention policy for a memory type."""
        return self.policies.get(memory_type)

    def should_retain(
        self, entry: MemoryEntry, entry_id: str, memory_type: MemoryType
    ) -> bool:
        """Check if entry should be retained."""
        policy = self.get_policy(memory_type)
        if not policy:
            return True  # No policy = retain everything

        return policy.should_retain(entry, entry_id)

    def select_for_eviction(
        self, entries: Dict[str, MemoryEntry], memory_type: MemoryType, count: int
    ) -> List[str]:
        """Select entries for eviction."""
        policy = self.get_policy(memory_type)
        if not policy:
            return []  # No policy = evict nothing

        return policy.select_for_eviction(entries, count)

    def update_access(self, entry_id: str, memory_type: MemoryType) -> None:
        """Update access metrics."""
        policy = self.get_policy(memory_type)
        if policy:
            policy.update_access(entry_id)

    def add_entry(
        self, entry: MemoryEntry, entry_id: str, memory_type: MemoryType
    ) -> None:
        """Add entry to retention tracking."""
        policy = self.get_policy(memory_type)
        if policy:
            policy.add_entry(entry, entry_id)

    def remove_entry(self, entry_id: str, memory_type: MemoryType) -> None:
        """Remove entry from retention tracking."""
        policy = self.get_policy(memory_type)
        if policy:
            policy.remove_entry(entry_id)

    def get_retention_stats(self) -> Dict[str, Any]:
        """Get retention statistics across all policies."""
        stats = {}

        for memory_type, policy in self.policies.items():
            total_entries = len(policy.metrics)
            if total_entries > 0:
                avg_access_count = (
                    sum(m.access_count for m in policy.metrics.values()) / total_entries
                )
                avg_age = (
                    sum(m.age_seconds() for m in policy.metrics.values())
                    / total_entries
                )
                avg_semantic_score = (
                    sum(m.semantic_score for m in policy.metrics.values())
                    / total_entries
                )
            else:
                avg_access_count = avg_age = avg_semantic_score = 0

            stats[memory_type.value] = {
                "total_entries": total_entries,
                "avg_access_count": avg_access_count,
                "avg_age_seconds": avg_age,
                "avg_semantic_score": avg_semantic_score,
                "max_entries": policy.max_entries,
            }

        return stats
