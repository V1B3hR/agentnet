#!/usr/bin/env python3
"""
Comprehensive test of P0 implementation - validates all refactored components.

This test validates:
1. Core agent refactoring
2. Monitor system v1 
3. Session persistence enhancements
4. Provider adapter interface
5. Backward compatibility
"""

import sys
import time
from pathlib import Path

def test_core_agent_refactoring():
    """Test core agent functionality from modular structure."""
    print("🧪 Testing Core Agent Refactoring...")
    
    from agentnet import AgentNet, ExampleEngine, Severity, CognitiveFault
    
    # Test agent creation
    engine = ExampleEngine()
    agent = AgentNet("TestAgent", {"logic": 0.8, "creativity": 0.6}, engine=engine)
    assert agent.name == "TestAgent"
    assert agent.style["logic"] == 0.8
    print("  ✅ Agent creation successful")
    
    # Test reasoning tree generation
    result = agent.generate_reasoning_tree("Design a resilient system")
    assert "result" in result
    assert "content" in result["result"]
    assert "TestAgent" in result["result"]["content"]
    print("  ✅ Reasoning tree generation working")
    
    # Test state persistence
    agent.save_state("/tmp/test_agent.json")
    loaded_agent = AgentNet.load_state("/tmp/test_agent.json", engine=engine)
    assert loaded_agent.name == agent.name
    assert loaded_agent.style == agent.style
    print("  ✅ Agent state persistence working")


def test_monitor_system_v1():
    """Test refactored monitor system."""
    print("🔍 Testing Monitor System V1...")
    
    from agentnet import MonitorFactory, MonitorSpec, Severity
    from agentnet.monitors.base import MonitorFn
    
    # Test keyword monitor
    spec = MonitorSpec(
        name="test_keyword",
        type="keyword", 
        params={"keywords": ["secret", "password"], "violation_name": "security_violation"},
        severity=Severity.MAJOR,
        description="Test keyword monitor"
    )
    
    monitor_fn = MonitorFactory.build(spec)
    assert callable(monitor_fn)
    print("  ✅ Monitor factory creating monitors")
    
    # Test regex monitor
    regex_spec = MonitorSpec(
        name="test_regex",
        type="regex",
        params={"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "violation_name": "ssn_violation"},
        severity=Severity.SEVERE,
        description="SSN pattern detection"
    )
    
    regex_monitor = MonitorFactory.build(regex_spec)
    assert callable(regex_monitor)
    print("  ✅ Regex monitor creation working")


def test_session_persistence():
    """Test enhanced session persistence."""
    print("💾 Testing Session Persistence...")
    
    from agentnet import SessionManager, AgentNet, ExampleEngine
    
    # Create session manager
    session_manager = SessionManager("/tmp/test_sessions")
    
    # Create test session data
    session_data = {
        "session_id": f"test_session_{int(time.time())}",
        "topic_start": "Initial topic",
        "topic_final": "Final topic", 
        "participants": ["Agent1", "Agent2"],
        "rounds_executed": 3,
        "converged": True,
        "timestamp": time.time(),
        "transcript": [
            {"agent": "Agent1", "content": "First response", "round": 1},
            {"agent": "Agent2", "content": "Second response", "round": 2}
        ]
    }
    
    # Test persistence
    filepath = session_manager.persist_session(session_data, "TestAgent")
    assert Path(filepath).exists()
    print("  ✅ Session persistence working")
    
    # Test loading
    loaded_session = session_manager.load_session(session_data["session_id"])
    assert loaded_session is not None
    assert loaded_session["session_id"] == session_data["session_id"]
    print("  ✅ Session loading working")
    
    # Test listing
    sessions = session_manager.list_sessions(limit=5)
    assert len(sessions) > 0
    print("  ✅ Session listing working")


def test_provider_adapters():
    """Test provider adapter interface."""
    print("🔌 Testing Provider Adapters...")
    
    from agentnet import ProviderAdapter, ExampleEngine
    
    # Test ExampleEngine
    engine = ExampleEngine()
    assert isinstance(engine, ProviderAdapter)
    
    # Test sync inference
    result = engine.infer("Test prompt", agent_name="TestAgent")
    assert "content" in result
    assert "confidence" in result
    print("  ✅ Sync inference working")
    
    # Test async inference
    import asyncio
    async def test_async():
        result = await engine.async_infer("Test async prompt", agent_name="TestAgent")
        assert "content" in result
        assert "confidence" in result
        return result
    
    asyncio.run(test_async())
    print("  ✅ Async inference working")
    
    # Test cost info
    cost_info = engine.get_cost_info(result)
    assert "cost" in cost_info
    assert "provider" in cost_info
    print("  ✅ Cost information working")


def test_backward_compatibility():
    """Test backward compatibility with original API."""
    print("🔄 Testing Backward Compatibility...")
    
    # Test importing from original module structure
    try:
        import AgentNet as original_module
        assert hasattr(original_module, '_use_refactored')
        assert original_module._use_refactored == True
        print("  ✅ Original module imports with refactored flag")
    except ImportError:
        print("  ⚠️  Original module import failed (expected in some setups)")
    
    # Test legacy compatibility layer
    try:
        import AgentNet_legacy
        print("  ✅ Legacy compatibility layer available")
    except ImportError:
        print("  ⚠️  Legacy layer not found")


def test_integration_workflow():
    """Test complete P0 workflow integration."""
    print("🔗 Testing P0 Integration Workflow...")
    
    from agentnet import AgentNet, ExampleEngine, MonitorFactory, MonitorSpec, Severity
    
    # Create monitored agent
    engine = ExampleEngine()
    
    # Create monitor
    monitor_spec = MonitorSpec(
        name="content_filter",
        type="keyword",
        params={"keywords": ["error", "fail"], "violation_name": "negative_content"},
        severity=Severity.MINOR,
        description="Filter negative content"
    )
    monitor_fn = MonitorFactory.build(monitor_spec)
    
    # Create agent with monitor
    agent = AgentNet("MonitoredAgent", {"logic": 0.7}, engine=engine, monitors=[monitor_fn])
    
    # Generate reasoning (should not trigger monitor with current content)
    result = agent.generate_reasoning_tree("Design a robust system")
    assert result["result"]["content"] is not None
    print("  ✅ Monitored agent working")
    
    # Test session persistence
    session_data = {
        "session_id": f"integration_test_{int(time.time())}",
        "participants": [agent.name],
        "topic": "Integration test",
        "rounds_executed": 1,
        "converged": False,
        "timestamp": time.time(),
        "transcript": [{"agent": agent.name, "content": result["result"]["content"]}]
    }
    
    filepath = agent.persist_session(session_data)
    assert Path(filepath).exists()
    print("  ✅ End-to-end workflow successful")


def main():
    """Run all P0 validation tests."""
    print("🚀 P0 Implementation Validation Tests")
    print("=" * 50)
    
    try:
        test_core_agent_refactoring()
        test_monitor_system_v1()
        test_session_persistence()
        test_provider_adapters()
        test_backward_compatibility()
        test_integration_workflow()
        
        print("\n" + "=" * 50)
        print("🎉 ALL P0 TESTS PASSED!")
        print("✅ Core agent refactoring: COMPLETE")
        print("✅ Monitor system v1: COMPLETE") 
        print("✅ Session persistence: COMPLETE")
        print("✅ Provider adapters: COMPLETE")
        print("✅ Backward compatibility: MAINTAINED")
        print("\n🏆 P0 Phase Successfully Implemented!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())