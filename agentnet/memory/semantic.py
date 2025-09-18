"""Semantic memory implementation with vector similarity search."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import MemoryEntry, MemoryLayer, MemoryType


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing (uses simple hash-based vectors)."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, text: str) -> List[float]:
        # Simple deterministic "embedding" based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to normalized vector
        vector = []
        for i in range(self.dimensions):
            byte_idx = i % len(hash_bytes)
            vector.append((hash_bytes[byte_idx] - 127.5) / 127.5)

        # Normalize to unit vector
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def store(
        self, entry_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
        """Store vector with metadata."""
        pass

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int = 10, threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors. Returns (entry_id, similarity, metadata)."""
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self):
        self._vectors: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    def store(
        self, entry_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
        self._vectors[entry_id] = (vector, metadata)
        return True

    def search(
        self, query_vector: List[float], top_k: int = 10, threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using cosine similarity."""
        results = []

        for entry_id, (vector, metadata) in self._vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity >= threshold:
                results.append((entry_id, similarity, metadata))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._vectors:
            del self._vectors[entry_id]
            return True
        return False

    def clear(self) -> None:
        self._vectors.clear()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)


class SemanticMemory(MemoryLayer):
    """Vector-based semantic memory for content similarity search."""

    def __init__(
        self,
        config: Dict[str, Any],
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        super().__init__(config)

        # Use provided or default implementations
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self.vector_store = vector_store or InMemoryVectorStore()

        # Configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_entries = config.get("max_entries", 500)

        # Content storage (vector store only handles vectors + metadata)
        self.storage_path = Path(
            config.get("storage_path", "sessions/semantic_memory.json")
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._content_store: Dict[str, MemoryEntry] = self._load_content()

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SEMANTIC

    def store(self, entry: MemoryEntry) -> bool:
        """Store entry with vector embedding."""
        entry_id = f"sem_{int(time.time() * 1000)}_{hash(entry.content) & 0x7FFFFFFF}"

        # Generate embedding
        try:
            embedding = self.embedding_provider.encode(entry.content)
            entry.embedding = embedding
        except Exception as e:
            # If embedding fails, skip semantic storage
            return False

        # Store in vector store
        metadata = {
            "timestamp": entry.timestamp,
            "agent_name": entry.agent_name,
            "tags": entry.tags or [],
            "content_length": len(entry.content),
        }

        if not self.vector_store.store(entry_id, embedding, metadata):
            return False

        # Store content
        self._content_store[entry_id] = entry

        # Enforce size limits
        self._enforce_size_limit()

        # Persist content store
        self._save_content()

        return True

    def retrieve(
        self, query: str, max_entries: int = 10, threshold: float = None, **kwargs
    ) -> List[MemoryEntry]:
        """Retrieve semantically similar entries."""
        if threshold is None:
            threshold = self.similarity_threshold

        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.encode(query)
        except Exception:
            return []

        # Search vector store
        search_results = self.vector_store.search(
            query_embedding, top_k=max_entries, threshold=threshold
        )

        # Build results
        entries = []
        for entry_id, similarity, metadata in search_results:
            if entry_id in self._content_store:
                entry = self._content_store[entry_id]
                # Add similarity to metadata
                entry.metadata["similarity_score"] = similarity
                entries.append(entry)

        return entries

    def clear(self) -> None:
        """Clear all semantic memory."""
        self.vector_store.clear()
        self._content_store.clear()
        self._save_content()

    def _enforce_size_limit(self) -> None:
        """Enforce maximum entry limits."""
        if len(self._content_store) <= self.max_entries:
            return

        # Remove oldest entries
        entries_by_time = sorted(
            self._content_store.items(), key=lambda x: x[1].timestamp
        )

        entries_to_remove = len(self._content_store) - self.max_entries
        for i in range(entries_to_remove):
            entry_id, _ = entries_by_time[i]
            self.vector_store.delete(entry_id)
            del self._content_store[entry_id]

    def _load_content(self) -> Dict[str, MemoryEntry]:
        """Load content store from disk."""
        if not self.storage_path.exists():
            return {}

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            content_store = {}
            for entry_id, entry_data in data.get("entries", {}).items():
                entry = MemoryEntry(
                    content=entry_data["content"],
                    metadata=entry_data["metadata"],
                    timestamp=entry_data["timestamp"],
                    agent_name=entry_data.get("agent_name"),
                    tags=entry_data.get("tags"),
                    embedding=entry_data.get("embedding"),
                )
                content_store[entry_id] = entry

            return content_store
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return {}

    def _save_content(self) -> None:
        """Save content store to disk."""
        data = {"entries": {}, "saved_at": time.time()}

        for entry_id, entry in self._content_store.items():
            data["entries"][entry_id] = {
                "content": entry.content,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp,
                "agent_name": entry.agent_name,
                "tags": entry.tags,
                "embedding": entry.embedding,
            }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def entry_count(self) -> int:
        """Get current entry count."""
        return len(self._content_store)
