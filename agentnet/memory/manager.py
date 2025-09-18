"""Memory manager orchestrating multiple memory layers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import MemoryEntry, MemoryLayer, MemoryRetrieval, MemoryType
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .short_term import ShortTermMemory


class MemoryManager:
    """Manages multiple memory layers and orchestrates retrieval pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.layers: Dict[MemoryType, MemoryLayer] = {}

        # Initialize enabled memory layers
        self._initialize_layers()

        # Pipeline configuration
        self.retrieval_config = config.get("retrieval", {})
        self.max_total_tokens = self.retrieval_config.get("max_total_tokens", 8000)
        self.summarize_threshold = self.retrieval_config.get(
            "summarize_threshold", 6000
        )

    def _initialize_layers(self) -> None:
        """Initialize configured memory layers."""
        memory_config = self.config.get("memory", {})

        # Short-term memory (always enabled)
        if memory_config.get("short_term", {}).get("enabled", True):
            short_term_config = memory_config.get("short_term", {})
            self.layers[MemoryType.SHORT_TERM] = ShortTermMemory(short_term_config)

        # Episodic memory
        if memory_config.get("episodic", {}).get("enabled", False):
            episodic_config = memory_config.get("episodic", {})
            self.layers[MemoryType.EPISODIC] = EpisodicMemory(episodic_config)

        # Semantic memory
        if memory_config.get("semantic", {}).get("enabled", False):
            semantic_config = memory_config.get("semantic", {})
            self.layers[MemoryType.SEMANTIC] = SemanticMemory(semantic_config)

    def store(
        self,
        content: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store content across appropriate memory layers."""
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            timestamp=time.time(),
            agent_name=agent_name,
            tags=tags,
        )

        success = True

        # Always store in short-term
        if MemoryType.SHORT_TERM in self.layers:
            if not self.layers[MemoryType.SHORT_TERM].store(entry):
                success = False

        # Store in episodic if tags provided or marked as important
        if MemoryType.EPISODIC in self.layers:
            should_store_episodic = (
                tags
                or metadata
                and metadata.get("important", False)
                or len(content) > 200  # Store longer content in episodic
            )
            if should_store_episodic:
                if not self.layers[MemoryType.EPISODIC].store(entry):
                    success = False

        # Store in semantic memory
        if MemoryType.SEMANTIC in self.layers:
            if not self.layers[MemoryType.SEMANTIC].store(entry):
                success = False

        return success

    def retrieve(
        self,
        query: str,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MemoryRetrieval:
        """Execute memory retrieval pipeline."""
        start_time = time.time()
        all_entries = []
        source_layers = []
        total_tokens = 0

        # 1. Collect short-term tail
        if MemoryType.SHORT_TERM in self.layers:
            short_term_entries = self.layers[MemoryType.SHORT_TERM].retrieve(
                query, max_entries=self.retrieval_config.get("short_term_count", 10)
            )
            all_entries.extend(short_term_entries)
            source_layers.append(MemoryType.SHORT_TERM)

            for entry in short_term_entries:
                total_tokens += len(entry.content.split())

        # 2. Add semantic top-k (if enabled and under token budget)
        if MemoryType.SEMANTIC in self.layers and total_tokens < self.max_total_tokens:
            remaining_tokens = self.max_total_tokens - total_tokens
            semantic_entries = self.layers[MemoryType.SEMANTIC].retrieve(
                query,
                max_entries=self.retrieval_config.get("semantic_count", 5),
                threshold=self.retrieval_config.get("semantic_threshold", 0.7),
            )

            # Add entries until token budget reached
            for entry in semantic_entries:
                entry_tokens = len(entry.content.split())
                if total_tokens + entry_tokens <= self.max_total_tokens:
                    all_entries.append(entry)
                    total_tokens += entry_tokens
                else:
                    break

            if semantic_entries:
                source_layers.append(MemoryType.SEMANTIC)

        # 3. Optional episodic tag matches
        if MemoryType.EPISODIC in self.layers and total_tokens < self.max_total_tokens:
            # Extract potential tags from query
            query_tags = self._extract_tags_from_query(query, context)

            if query_tags:
                episodic_entries = self.layers[MemoryType.EPISODIC].retrieve(
                    query,
                    max_entries=self.retrieval_config.get("episodic_count", 3),
                    tags=query_tags,
                    agent_name=agent_name,
                )

                # Add entries until token budget reached
                for entry in episodic_entries:
                    entry_tokens = len(entry.content.split())
                    if total_tokens + entry_tokens <= self.max_total_tokens:
                        all_entries.append(entry)
                        total_tokens += entry_tokens
                    else:
                        break

                if episodic_entries:
                    source_layers.append(MemoryType.EPISODIC)

        # 4. Check if summarization needed
        needs_summary = total_tokens > self.summarize_threshold

        retrieval_time = time.time() - start_time

        return MemoryRetrieval(
            entries=all_entries,
            total_tokens=total_tokens,
            retrieval_time=retrieval_time,
            source_layers=source_layers,
        )

    def _extract_tags_from_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract potential tags from query and context."""
        tags = []

        # Simple keyword extraction
        keywords = [
            "error",
            "bug",
            "issue",
            "solution",
            "important",
            "critical",
            "urgent",
        ]
        query_lower = query.lower()

        for keyword in keywords:
            if keyword in query_lower:
                tags.append(keyword)

        # Add context tags if provided
        if context and "tags" in context:
            tags.extend(context["tags"])

        return list(set(tags))  # Remove duplicates

    def clear_all(self) -> None:
        """Clear all memory layers."""
        for layer in self.layers.values():
            layer.clear()

    def clear_layer(self, memory_type: MemoryType) -> None:
        """Clear specific memory layer."""
        if memory_type in self.layers:
            self.layers[memory_type].clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        stats = {"enabled_layers": list(self.layers.keys()), "layers": {}}

        for memory_type, layer in self.layers.items():
            layer_stats = {"type": memory_type.value}

            if hasattr(layer, "entry_count"):
                layer_stats["entry_count"] = layer.entry_count

            if hasattr(layer, "current_token_count"):
                layer_stats["token_count"] = layer.current_token_count

            if hasattr(layer, "episode_count"):
                layer_stats["episode_count"] = layer.episode_count

            stats["layers"][memory_type.value] = layer_stats

        return stats

    def add_episodic_tags(self, tags: List[str], count: int = 1) -> None:
        """Add tags to recent episodic entries."""
        if MemoryType.EPISODIC in self.layers:
            episodic_layer = self.layers[MemoryType.EPISODIC]
            if hasattr(episodic_layer, "add_tags_to_recent"):
                episodic_layer.add_tags_to_recent(tags, count)
