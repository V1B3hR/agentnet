"""Short-term memory implementation."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

from .base import MemoryEntry, MemoryLayer, MemoryType


class ShortTermMemory(MemoryLayer):
    """In-memory sliding window for recent interactions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_entries = config.get("max_entries", 50)
        self.max_tokens = config.get("max_tokens", 5000)
        self.truncate_strategy = config.get(
            "truncate_strategy", "head"
        )  # "head" or "tail"

        self._entries: deque[MemoryEntry] = deque(maxlen=self.max_entries)
        self._current_tokens = 0

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SHORT_TERM

    def store(self, entry: MemoryEntry) -> bool:
        """Store entry in short-term memory with token limit management."""
        entry_tokens = len(entry.content.split())  # Simple token estimation

        # Add entry
        self._entries.append(entry)
        self._current_tokens += entry_tokens

        # Enforce token limits
        self._enforce_token_limit()

        return True

    def retrieve(
        self,
        query: str,
        max_entries: int = 10,
        threshold: float = 0.0,  # Not used for short-term
        **kwargs,
    ) -> List[MemoryEntry]:
        """Retrieve recent entries (no semantic filtering for short-term)."""
        # Return most recent entries up to max_entries
        entries = list(self._entries)
        return entries[-max_entries:] if len(entries) > max_entries else entries

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._entries.clear()
        self._current_tokens = 0

    def get_recent_entries(self, count: int = None) -> List[MemoryEntry]:
        """Get the most recent entries."""
        if count is None:
            return list(self._entries)
        return (
            list(self._entries)[-count:]
            if len(self._entries) > count
            else list(self._entries)
        )

    def _enforce_token_limit(self) -> None:
        """Enforce token limits by removing entries based on strategy."""
        if self._current_tokens <= self.max_tokens:
            return

        if self.truncate_strategy == "head":
            # Remove oldest entries
            while self._current_tokens > self.max_tokens and self._entries:
                removed = self._entries.popleft()
                self._current_tokens -= len(removed.content.split())
        elif self.truncate_strategy == "tail":
            # Remove newest entries (keep oldest)
            while self._current_tokens > self.max_tokens and self._entries:
                removed = self._entries.pop()
                self._current_tokens -= len(removed.content.split())

    @property
    def current_token_count(self) -> int:
        """Get current token count."""
        return self._current_tokens

    @property
    def entry_count(self) -> int:
        """Get current entry count."""
        return len(self._entries)
