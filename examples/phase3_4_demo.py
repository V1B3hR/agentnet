#!/usr/bin/env python3
"""
Demo script showcasing Phase 3 & 4 features in AgentNet.

Features demonstrated:
- Caching system for embeddings and responses
- Memory retention policies  
- Telemetry and observability
- Multi-modal content handling
- Session checkpoints and resume
"""

import time
from agentnet import AgentNet, init_telemetry
from agentnet.core.cache import CacheManager, InMemoryCache
from agentnet.core.telemetry import EventType, MetricType, record_event, record_metric
from agentnet.core.multimodal import (
    MultiModalMessage, ModalityContent, ModalityType, get_multimodal_manager
)
from agentnet.memory.retention import RetentionManager, HybridRetentionPolicy
from agentnet.memory.base import MemoryEntry, MemoryType
from agentnet.persistence.session import SessionManager


def demo_caching():
    """Demonstrate caching system."""
    print("=== Caching System Demo ===")
    
    cache = CacheManager(backend=InMemoryCache(max_size=10))
    
    # Simulate embedding caching
    texts = ["Hello world", "AgentNet is awesome", "Caching improves performance"]
    
    print("Testing embedding cache...")
    for i, text in enumerate(texts):
        # First call - cache miss
        embedding = cache.get_embedding(text)
        if embedding is None:
            # Simulate generating embedding
            embedding = [0.1 * j + i for j in range(5)]  # Mock embedding
            cache.cache_embedding(text, embedding)
            print(f"  Cache MISS for '{text}' - generated and cached")
        else:
            print(f"  Cache HIT for '{text}'")
    
    # Second pass - should all be hits
    print("\nSecond pass (should be cache hits):")
    for text in texts:
        embedding = cache.get_embedding(text)
        if embedding:
            print(f"  Cache HIT for '{text}'")
    
    # Show stats
    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")


def demo_telemetry():
    """Demonstrate telemetry system."""
    print("\n=== Telemetry System Demo ===")
    
    # Initialize telemetry
    telemetry = init_telemetry({
        "enabled": True,
        "buffer_size": 100,
        "export_dir": "demo_telemetry"
    })
    
    # Record various events
    session_id = f"demo_session_{int(time.time())}"
    
    record_event(EventType.AGENT_START, 
                session_id=session_id, 
                agent_name="DemoAgent",
                event_data={"version": "0.5.0"})
    
    # Simulate some operations with timing
    telemetry.start_timer("reasoning_task")
    time.sleep(0.01)  # Simulate work
    duration = telemetry.end_timer("reasoning_task", 
                                  event_type=EventType.AGENT_END,
                                  session_id=session_id)
    
    # Record metrics
    record_metric("tasks_completed", 1, MetricType.COUNTER)
    record_metric("memory_usage_mb", 45.2, MetricType.GAUGE)
    
    print(f"Recorded reasoning task duration: {duration:.2f}ms")
    
    # Show events
    events = telemetry.get_events(session_id=session_id)
    print(f"Recorded {len(events)} events for session {session_id}")
    
    # Export audit bundle
    bundle_path = telemetry.export_audit_bundle(session_id)
    print(f"Exported audit bundle to: {bundle_path}")


def demo_multimodal():
    """Demonstrate multi-modal system."""
    print("\n=== Multi-Modal System Demo ===")
    
    manager = get_multimodal_manager()
    print(f"Supported modalities: {[m.value for m in manager.get_supported_modalities()]}")
    
    # Create a multi-modal message
    message = MultiModalMessage("This is a message with multiple content types.")
    
    # Add different types of content
    message.add_text("Here's some additional text content.")
    
    message.add_image(
        "base64_encoded_image_data_here",
        metadata={"description": "A sample chart showing performance metrics"}
    )
    
    # Add structured data
    structured_data = ModalityContent(
        content={"metrics": {"accuracy": 0.95, "latency": "50ms"}},
        modality=ModalityType.STRUCTURED,
        metadata={"format": "json"}
    )
    message.add_content(structured_data)
    
    print(f"Message modalities: {[m.value for m in message.get_modalities()]}")
    
    # Process the message
    processed = manager.process_message(message)
    print("Message processed successfully")
    
    # Convert to text-only representation
    text_only = manager.convert_to_text_only(message)
    print(f"Text-only representation:\n{text_only}")


def demo_retention():
    """Demonstrate memory retention policies."""
    print("\n=== Memory Retention Demo ===")
    
    # Create retention manager with hybrid policy
    retention_manager = RetentionManager()
    policy = HybridRetentionPolicy({
        "max_entries": 10,
        "weights": {"recency": 0.3, "frequency": 0.3, "semantic": 0.4},
        "min_retention_score": 0.3
    })
    retention_manager.set_policy(MemoryType.SEMANTIC, policy)
    
    # Create sample memory entries
    entries = {}
    important_contents = [
        "CRITICAL ERROR: System failure detected",
        "SUCCESS: Mission completed successfully", 
        "Just some regular conversation text",
        "WARNING: Performance degraded",
        "Normal operational status update",
        "ERROR: Connection timeout occurred"
    ]
    
    # Add entries and simulate access patterns
    for i, content in enumerate(important_contents):
        entry_id = f"entry_{i}"
        entry = MemoryEntry(
            content=content,
            metadata={"priority": "high" if "ERROR" in content or "CRITICAL" in content else "normal"},
            timestamp=time.time() - (len(important_contents) - i) * 3600,  # Spread over time
            agent_name="DemoAgent",
            tags=["error"] if "ERROR" in content else ["normal"]
        )
        entries[entry_id] = entry
        retention_manager.add_entry(entry, entry_id, MemoryType.SEMANTIC)
        
        # Simulate access for important entries
        if "ERROR" in content or "SUCCESS" in content:
            retention_manager.update_access(entry_id, MemoryType.SEMANTIC)
            retention_manager.update_access(entry_id, MemoryType.SEMANTIC)  # Access twice
    
    # Test retention decisions
    print("Retention decisions:")
    for entry_id, entry in entries.items():
        should_retain = retention_manager.should_retain(entry, entry_id, MemoryType.SEMANTIC)
        print(f"  {entry_id}: {'RETAIN' if should_retain else 'EVICT'} - {entry.content[:50]}...")
    
    # Select entries for eviction
    to_evict = retention_manager.select_for_eviction(entries, MemoryType.SEMANTIC, 3)
    print(f"\nWould evict {len(to_evict)} entries: {to_evict}")
    
    # Show retention stats
    stats = retention_manager.get_retention_stats()
    print(f"Retention stats: {stats}")


def demo_checkpoints():
    """Demonstrate session checkpoints."""
    print("\n=== Session Checkpoints Demo ===")
    
    session_manager = SessionManager("demo_sessions")
    
    # Create session state
    session_state = {
        "session_id": "demo_conversation",
        "participants": ["Alice", "Bob", "Charlie"],
        "current_round": 3,
        "topic": "AI Safety Protocols",
        "transcript": [
            {"speaker": "Alice", "message": "We need better safety measures"},
            {"speaker": "Bob", "message": "I agree, what do you propose?"},
            {"speaker": "Charlie", "message": "Let me suggest a framework..."}
        ],
        "metadata": {"start_time": time.time() - 1800}  # Started 30 minutes ago
    }
    
    # Create checkpoint
    checkpoint_id = session_manager.create_checkpoint("demo_conversation", session_state)
    print(f"Created checkpoint: {checkpoint_id}")
    
    # Simulate some time passing and state changes
    time.sleep(0.1)
    
    # Load checkpoint
    loaded_checkpoint = session_manager.load_checkpoint(checkpoint_id)
    print(f"Loaded checkpoint successfully: {loaded_checkpoint is not None}")
    
    # Resume session with additional context
    additional_context = {
        "resume_reason": "System restart after maintenance",
        "new_participant": "David"
    }
    
    resumed_state = session_manager.resume_session(checkpoint_id, additional_context)
    print(f"Resumed session with {len(resumed_state)} state keys")
    print(f"Resume metadata: resumed_from_checkpoint={resumed_state.get('resumed_from_checkpoint')}")
    
    # List checkpoints
    checkpoints = session_manager.list_checkpoints("demo_conversation")
    print(f"Found {len(checkpoints)} checkpoints for this session")


def demo_integration():
    """Demonstrate integration with AgentNet."""
    print("\n=== AgentNet Integration Demo ===")
    
    # Initialize telemetry for the agent
    init_telemetry({"enabled": True})
    
    # Create AgentNet instance
    agent = AgentNet(
        name="EnhancedAgent",
        style={"logic": 0.8, "creativity": 0.6, "analytical": 0.9}
    )
    
    # Record agent creation
    record_event(EventType.AGENT_START, 
                agent_name="EnhancedAgent",
                event_data={"style": agent.style})
    
    print(f"Created agent: {agent.name}")
    
    # Test basic reasoning with telemetry
    start_time = time.time()
    result = agent.generate_reasoning_tree("How can we improve AI safety?")
    duration = (time.time() - start_time) * 1000
    
    # Record metrics
    record_metric("reasoning_latency_ms", duration, MetricType.TIMER)
    record_metric("reasoning_requests", 1, MetricType.COUNTER)
    
    print(f"Generated reasoning in {duration:.2f}ms")
    print(f"Result: {result['result']['content'][:100]}...")


def main():
    """Run all demos."""
    print("🚀 AgentNet Phase 3 & 4 Features Demo")
    print("=" * 50)
    
    try:
        demo_caching()
        demo_telemetry()
        demo_multimodal()
        demo_retention()
        demo_checkpoints()
        demo_integration()
        
        print("\n✅ All demos completed successfully!")
        print("\nPhase 3 & 4 features are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()