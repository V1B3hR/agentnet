"""Caching system for embeddings and responses."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("agentnet.cache")


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    tags: Optional[List[str]] = None
    ttl: Optional[float] = None  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self) -> None:
        """Mark the entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve a cache entry."""
        pass
    
    @abstractmethod
    def set(self, entry: CacheEntry) -> bool:
        """Store a cache entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """List all cache keys."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        
    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._cache.get(key)
        if entry and not entry.is_expired():
            entry.access()
            return entry
        elif entry and entry.is_expired():
            # Remove expired entry
            del self._cache[key]
        return None
    
    def set(self, entry: CacheEntry) -> bool:
        # Evict if at capacity
        if len(self._cache) >= self.max_size and entry.key not in self._cache:
            self._evict_lru()
        
        self._cache[entry.key] = entry
        return True
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        self._cache.clear()
    
    def keys(self) -> List[str]:
        return list(self._cache.keys())
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
            
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]


class FileCache(CacheBackend):
    """File-based cache implementation."""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self._index_file = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                return json.loads(self._index_file.read_text())
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            self._index_file.write_text(json.dumps(self._index, indent=2))
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid file name issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        if key not in self._index:
            return None
        
        file_path = self._get_file_path(key)
        if not file_path.exists():
            # Clean up stale index entry
            del self._index[key]
            self._save_index()
            return None
        
        try:
            data = json.loads(file_path.read_text())
            entry = CacheEntry(**data)
            
            if entry.is_expired():
                self.delete(key)
                return None
            
            entry.access()
            # Update index with new access info
            self._index[key].update({
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed
            })
            self._save_index()
            
            return entry
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load cache entry {key}: {e}")
            return None
    
    def set(self, entry: CacheEntry) -> bool:
        # Evict if at capacity
        if len(self._index) >= self.max_size and entry.key not in self._index:
            self._evict_lru()
        
        file_path = self._get_file_path(entry.key)
        try:
            file_path.write_text(json.dumps({
                "key": entry.key,
                "value": entry.value,
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "tags": entry.tags,
                "ttl": entry.ttl
            }, indent=2))
            
            # Update index
            self._index[entry.key] = {
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "file_path": str(file_path)
            }
            self._save_index()
            return True
        except IOError as e:
            logger.error(f"Failed to save cache entry {entry.key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if key not in self._index:
            return False
        
        file_path = self._get_file_path(key)
        try:
            if file_path.exists():
                file_path.unlink()
            del self._index[key]
            self._save_index()
            return True
        except IOError as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False
    
    def clear(self) -> None:
        for key in list(self._index.keys()):
            self.delete(key)
    
    def keys(self) -> List[str]:
        return list(self._index.keys())
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._index:
            return
        
        lru_key = min(
            self._index.keys(),
            key=lambda k: self._index[k].get("last_accessed", 0)
        )
        self.delete(lru_key)


class CacheManager:
    """High-level cache manager for embeddings and responses."""
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: Optional[float] = None
    ):
        self.backend = backend or InMemoryCache()
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    def _make_key(self, prefix: str, content: Union[str, Dict[str, Any]]) -> str:
        """Create a cache key from content."""
        if isinstance(content, str):
            content_str = content
        else:
            content_str = json.dumps(content, sort_keys=True)
        
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        return f"{prefix}:{content_hash}"
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._make_key("embedding", text)
        entry = self.backend.get(key)
        
        if entry:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for embedding: {key[:16]}...")
            return entry.value
        
        self.stats["misses"] += 1
        logger.debug(f"Cache miss for embedding: {key[:16]}...")
        return None
    
    def cache_embedding(
        self, text: str, embedding: List[float], ttl: Optional[float] = None
    ) -> bool:
        """Cache an embedding."""
        key = self._make_key("embedding", text)
        entry = CacheEntry(
            key=key,
            value=embedding,
            timestamp=time.time(),
            tags=["embedding"],
            ttl=ttl or self.default_ttl
        )
        
        success = self.backend.set(entry)
        if success:
            self.stats["sets"] += 1
            logger.debug(f"Cached embedding: {key[:16]}...")
        
        return success
    
    def get_response(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for request."""
        key = self._make_key("response", request_data)
        entry = self.backend.get(key)
        
        if entry:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for response: {key[:16]}...")
            return entry.value
        
        self.stats["misses"] += 1
        logger.debug(f"Cache miss for response: {key[:16]}...")
        return None
    
    def cache_response(
        self,
        request_data: Dict[str, Any],
        response: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> bool:
        """Cache a response."""
        key = self._make_key("response", request_data)
        entry = CacheEntry(
            key=key,
            value=response,
            timestamp=time.time(),
            tags=["response"],
            ttl=ttl or self.default_ttl
        )
        
        success = self.backend.set(entry)
        if success:
            self.stats["sets"] += 1
            logger.debug(f"Cached response: {key[:16]}...")
        
        return success
    
    def clear_by_tag(self, tag: str) -> int:
        """Clear cache entries by tag."""
        cleared = 0
        for key in self.backend.keys():
            entry = self.backend.get(key)
            if entry and entry.tags and tag in entry.tags:
                if self.backend.delete(key):
                    cleared += 1
                    self.stats["deletes"] += 1
        
        logger.info(f"Cleared {cleared} cache entries with tag '{tag}'")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_size": len(self.backend.keys())
        }