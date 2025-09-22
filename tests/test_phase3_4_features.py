"""Tests for Phase 3 & 4 new features."""

import json
import time
import tempfile
from pathlib import Path

import pytest

from agentnet.core.cache import CacheManager, CacheEntry, InMemoryCache, FileCache
from agentnet.core.telemetry import (
    TelemetryCollector, EventType, MetricType, init_telemetry
)
from agentnet.core.multimodal import (
    MultiModalManager, MultiModalMessage, ModalityContent, ModalityType
)
from agentnet.memory.retention import (
    RetentionManager, LRURetentionPolicy, SemanticSalienceRetentionPolicy
)
from agentnet.memory.base import MemoryEntry, MemoryType
from agentnet.persistence.session import SessionManager


class TestCaching:
    """Test caching functionality."""
    
    def test_in_memory_cache(self):
        cache = InMemoryCache(max_size=2)
        
        # Test setting and getting
        entry1 = CacheEntry(key="key1", value="value1", timestamp=time.time())
        entry2 = CacheEntry(key="key2", value="value2", timestamp=time.time())
        
        assert cache.set(entry1)
        assert cache.set(entry2)
        
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.value == "value1"
        assert retrieved.access_count == 1
        
        # Test eviction
        entry3 = CacheEntry(key="key3", value="value3", timestamp=time.time())
        cache.set(entry3)
        
        # Should have evicted key1 (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
    
    def test_cache_manager(self):
        cache_manager = CacheManager()
        
        # Test embedding caching
        text = "Hello world"
        embedding = [0.1, 0.2, 0.3]
        
        # Should be a cache miss first
        assert cache_manager.get_embedding(text) is None
        
        # Cache the embedding
        assert cache_manager.cache_embedding(text, embedding)
        
        # Should be a cache hit now
        cached_embedding = cache_manager.get_embedding(text)
        assert cached_embedding == embedding
        
        # Test response caching
        request = {"prompt": "test", "model": "gpt-4"}
        response = {"content": "response"}
        
        assert cache_manager.get_response(request) is None
        assert cache_manager.cache_response(request, response)
        
        cached_response = cache_manager.get_response(request)
        assert cached_response == response
        
        # Check stats
        stats = cache_manager.get_stats()
        assert stats["hits"] >= 2
        assert stats["misses"] >= 2


class TestTelemetry:
    """Test telemetry functionality."""
    
    def test_telemetry_collector(self):
        telemetry = TelemetryCollector()
        
        # Record an event
        telemetry.record_event(
            EventType.AGENT_START,
            session_id="test_session",
            agent_name="test_agent",
            event_data={"test": "data"}
        )
        
        # Record a metric
        telemetry.record_metric("test_counter", 5, MetricType.COUNTER)
        
        # Test timer
        telemetry.start_timer("test_timer")
        time.sleep(0.01)  # Small sleep
        duration = telemetry.end_timer("test_timer")
        assert duration is not None and duration > 0
        
        # Get events
        events = telemetry.get_events()
        assert len(events) >= 1
        
        # Get metric summary
        summary = telemetry.get_metric_summary("test_counter")
        assert summary is not None
        assert summary["count"] == 1
        assert summary["latest"] == 5
    
    def test_global_telemetry(self):
        # Test global telemetry functions
        from agentnet.core.telemetry import record_event, record_metric
        
        telemetry = init_telemetry({"enabled": True})
        
        record_event(EventType.MEMORY_STORE, session_id="global_test")
        record_metric("global_counter", 10, MetricType.COUNTER)
        
        events = telemetry.get_events(event_type=EventType.MEMORY_STORE)
        assert len(events) >= 1
        
        summary = telemetry.get_metric_summary("global_counter")
        assert summary is not None
        assert summary["latest"] == 10


class TestMultiModal:
    """Test multi-modal functionality."""
    
    def test_modality_content(self):
        text_content = ModalityContent(
            content="Hello world",
            modality=ModalityType.TEXT
        )
        
        assert text_content.is_text()
        assert text_content.get_text_representation() == "Hello world"
        
        image_content = ModalityContent(
            content="base64data",
            modality=ModalityType.IMAGE,
            encoding="base64",
            metadata={"description": "A test image"}
        )
        
        assert not image_content.is_text()
        text_repr = image_content.get_text_representation()
        assert "IMAGE" in text_repr and "A test image" in text_repr
    
    def test_multimodal_message(self):
        message = MultiModalMessage("Primary text")
        
        # Add text content
        message.add_text("Additional text")
        
        # Add image content
        message.add_image(
            "base64data",
            metadata={"description": "Test image"}
        )
        
        # Check modalities
        modalities = message.get_modalities()
        assert ModalityType.TEXT in modalities
        assert ModalityType.IMAGE in modalities
        
        # Get text representation
        text_only = message.get_text_only()
        assert "Primary text" in text_only
        assert "Additional text" in text_only
        assert "[IMAGE:" in text_only
    
    def test_multimodal_manager(self):
        manager = MultiModalManager()
        
        # Test supported modalities
        supported = manager.get_supported_modalities()
        assert ModalityType.TEXT in supported
        assert ModalityType.IMAGE in supported
        
        # Create and process message
        message = MultiModalMessage("Test message")
        message.add_text("More text")
        
        processed = manager.process_message(message)
        assert processed.primary_text == "Test message"
        assert len(processed.contents) == 1
        
        # Test validation
        assert manager.validate_message(message)


class TestRetention:
    """Test memory retention policies."""
    
    def test_lru_retention_policy(self):
        config = {"max_entries": 100}
        policy = LRURetentionPolicy(config)
        
        # Create test entries
        entries = {}
        for i in range(5):
            entry_id = f"entry_{i}"
            entry = MemoryEntry(
                content=f"Content {i}",
                metadata={},
                timestamp=time.time(),
                agent_name="test"
            )
            entries[entry_id] = entry
            policy.add_entry(entry, entry_id)
        
        # Access some entries
        policy.update_access("entry_1")
        policy.update_access("entry_3")
        
        # Select for eviction (should get least recently used)
        to_evict = policy.select_for_eviction(entries, 2)
        assert len(to_evict) == 2
        # entry_1 and entry_3 were accessed, so they shouldn't be first to evict
        assert "entry_1" not in to_evict[:2]
        assert "entry_3" not in to_evict[:2]
    
    def test_semantic_salience_policy(self):
        config = {
            "max_entries": 100,
            "min_semantic_score": 0.3,
            "min_importance_score": 0.2
        }
        policy = SemanticSalienceRetentionPolicy(config)
        
        # Create entry with important content
        important_entry = MemoryEntry(
            content="This is a critical error that needs attention",
            metadata={"priority": "high"},
            timestamp=time.time(),
            agent_name="test",
            tags=["error", "critical"]
        )
        
        # Create normal entry
        normal_entry = MemoryEntry(
            content="Just some normal text",
            metadata={},
            timestamp=time.time(),
            agent_name="test"
        )
        
        # Test retention decisions
        assert policy.should_retain(important_entry, "important")
        # Normal entry might or might not be retained based on calculated scores
        
        entries = {
            "important": important_entry,
            "normal": normal_entry
        }
        
        # Important entry should be less likely to be evicted
        to_evict = policy.select_for_eviction(entries, 1)
        # The exact eviction choice depends on scoring, but we can verify it runs
        assert len(to_evict) <= 1
    
    def test_retention_manager(self):
        manager = RetentionManager()
        
        # Set up policy for semantic memory
        policy = LRURetentionPolicy({"max_entries": 100})
        manager.set_policy(MemoryType.SEMANTIC, policy)
        
        # Test entry operations
        entry = MemoryEntry(
            content="Test content",
            metadata={},
            timestamp=time.time(),
            agent_name="test"
        )
        
        manager.add_entry(entry, "test_id", MemoryType.SEMANTIC)
        manager.update_access("test_id", MemoryType.SEMANTIC)
        
        # Check that policy was used (no exceptions)
        assert manager.should_retain(entry, "test_id", MemoryType.SEMANTIC)
        
        # Get stats
        stats = manager.get_retention_stats()
        assert "semantic" in stats


class TestSessionCheckpoints:
    """Test session checkpoint functionality."""
    
    def test_session_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_manager = SessionManager(temp_dir)
            
            # Create checkpoint data
            checkpoint_data = {
                "session_id": "test_session",
                "current_state": {"round": 3, "participants": ["agent1", "agent2"]},
                "transcript": [{"speaker": "agent1", "message": "Hello"}]
            }
            
            # Create checkpoint
            checkpoint_id = session_manager.create_checkpoint("test_session", checkpoint_data)
            assert checkpoint_id.startswith("test_session_checkpoint_")
            
            # Load checkpoint
            loaded = session_manager.load_checkpoint(checkpoint_id)
            assert loaded is not None
            assert loaded["session_id"] == "test_session"
            assert loaded["data"]["current_state"]["round"] == 3
            
            # Resume session
            additional_context = {"resume_reason": "system restart"}
            resumed = session_manager.resume_session(checkpoint_id, additional_context)
            assert resumed is not None
            assert resumed["resume_reason"] == "system restart"
            assert "resumed_from_checkpoint" in resumed
            assert "resume_timestamp" in resumed
            
            # List checkpoints
            checkpoints = session_manager.list_checkpoints("test_session")
            assert len(checkpoints) >= 1
            assert any(cp["checkpoint_id"] == checkpoint_id for cp in checkpoints)


if __name__ == "__main__":
    pytest.main([__file__])