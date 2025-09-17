"""Episodic memory implementation."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MemoryLayer, MemoryEntry, MemoryType


class EpisodicMemory(MemoryLayer):
    """Persistent storage for important conversation episodes with tags."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_path = Path(config.get("storage_path", "sessions/episodic_memory.json"))
        self.max_episodes = config.get("max_episodes", 1000)
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing episodes
        self._episodes: List[Dict[str, Any]] = self._load_episodes()
    
    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.EPISODIC
    
    def store(self, entry: MemoryEntry) -> bool:
        """Store episodic memory entry with tags."""
        episode = {
            "id": f"ep_{int(time.time() * 1000)}_{len(self._episodes)}",
            "content": entry.content,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp,
            "agent_name": entry.agent_name,
            "tags": entry.tags or [],
        }
        
        self._episodes.append(episode)
        
        # Enforce episode limit
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]
        
        # Persist to storage
        self._save_episodes()
        return True
    
    def retrieve(
        self, 
        query: str, 
        max_entries: int = 10,
        threshold: float = 0.7,
        tags: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
        **kwargs
    ) -> List[MemoryEntry]:
        """Retrieve episodic entries by tag matching and content similarity."""
        matching_episodes = []
        
        for episode in self._episodes:
            # Tag filtering
            if tags:
                episode_tags = set(episode.get("tags", []))
                query_tags = set(tags)
                if not query_tags.intersection(episode_tags):
                    continue
            
            # Agent filtering
            if agent_name and episode.get("agent_name") != agent_name:
                continue
            
            # Simple content matching (could be enhanced with embeddings)
            if query.lower() in episode["content"].lower():
                matching_episodes.append(episode)
        
        # Sort by timestamp (most recent first)
        matching_episodes.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Convert to MemoryEntry objects
        entries = []
        for episode in matching_episodes[:max_entries]:
            entry = MemoryEntry(
                content=episode["content"],
                metadata=episode["metadata"],
                timestamp=episode["timestamp"],
                agent_name=episode.get("agent_name"),
                tags=episode.get("tags")
            )
            entries.append(entry)
        
        return entries
    
    def retrieve_by_tags(self, tags: List[str], max_entries: int = 10) -> List[MemoryEntry]:
        """Retrieve episodes that match any of the given tags."""
        return self.retrieve("", max_entries=max_entries, tags=tags)
    
    def clear(self) -> None:
        """Clear all episodic memory."""
        self._episodes.clear()
        self._save_episodes()
    
    def add_tags_to_recent(self, tags: List[str], count: int = 1) -> None:
        """Add tags to the most recent episodes."""
        for i in range(min(count, len(self._episodes))):
            episode = self._episodes[-(i+1)]
            existing_tags = set(episode.get("tags", []))
            episode["tags"] = list(existing_tags.union(set(tags)))
        
        self._save_episodes()
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episodes from storage."""
        if not self.storage_path.exists():
            return []
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return data.get("episodes", [])
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_episodes(self) -> None:
        """Save episodes to storage."""
        data = {
            "episodes": self._episodes,
            "saved_at": time.time()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @property
    def episode_count(self) -> int:
        """Get current episode count."""
        return len(self._episodes)