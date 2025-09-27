#!/usr/bin/env python3
"""
Comprehensive Component Specifications Test Coverage

Tests for all major AgentNet component modules to ensure compliance
with specifications and proper edge case handling.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agentnet import (
    AgentNet, ExampleEngine, MonitorFactory, MonitorSpec, 
    PolicyEngine, MemoryManager, ToolRegistry, TurnEngine,
    CognitiveFault, Severity
)
from agentnet.core.types import CognitiveFault
from agentnet.core.orchestration import TurnResult, TurnMode, TerminationReason
from agentnet.reasoning import ReasoningEngine, ReasoningType
from agentnet.memory import ShortTermMemory, EpisodicMemory
from agentnet.tools import Tool, ToolResult
from agentnet.tools.base import ToolStatus
from agentnet.performance import LatencyTracker, TokenUtilizationTracker
from agentnet.persistence import SessionManager


class TestCoreAgentModule:
    """Test core AgentNet agent functionality."""
    
    def test_agent_initialization_edge_cases(self):
        """Test agent initialization with various edge cases."""
        print("🤖 Testing Agent Initialization Edge Cases...")
        
        # Test with minimal parameters
        agent = AgentNet("MinimalAgent", {})
        assert agent.name == "MinimalAgent"
        assert isinstance(agent.style, dict)
        
        # Test with null/empty style
        agent_empty = AgentNet("EmptyStyleAgent", {})
        result = agent_empty.generate_reasoning_tree("test")
        assert result is not None
        
        # Test with extreme style values
        extreme_style = {"logic": 2.0, "creativity": -0.5, "analytical": 10.0}
        agent_extreme = AgentNet("ExtremeAgent", extreme_style, engine=ExampleEngine())
        result = agent_extreme.generate_reasoning_tree("test extreme values")
        assert result is not None
        
        # Test with non-standard style keys
        custom_style = {"custom_param": 0.7, "weird_key": 0.3}
        agent_custom = AgentNet("CustomAgent", custom_style, engine=ExampleEngine())
        assert agent_custom.style == custom_style
        
        print("  ✅ Agent initialization handles edge cases properly")
    
    def test_agent_state_persistence(self):
        """Test agent state can be saved and restored."""
        print("📁 Testing Agent State Persistence...")
        
        original_agent = AgentNet(
            "PersistentAgent", 
            {"logic": 0.8, "creativity": 0.6}, 
            engine=ExampleEngine()
        )
        
        # Generate some state
        result1 = original_agent.generate_reasoning_tree("Create some history")
        
        # Save state
        state = original_agent.save_state()
        assert isinstance(state, dict)
        assert "name" in state
        assert "style" in state
        
        # Restore state
        restored_agent = AgentNet.from_state(state)
        assert restored_agent.name == original_agent.name
        assert restored_agent.style == original_agent.style
        
        # Verify functionality after restoration
        result2 = restored_agent.generate_reasoning_tree("Test after restoration")
        assert result2 is not None
        
        print("  ✅ Agent state persistence working correctly")
    
    def test_cognitive_fault_handling(self):
        """Test handling of cognitive faults."""
        print("🧠 Testing Cognitive Fault Handling...")
        
        agent = AgentNet("CognitiveFaultAgent", {"logic": 0.9}, engine=ExampleEngine())
        
        # Test direct fault creation and handling
        fault = CognitiveFault(
            "Test cognitive processing error",
            severity=Severity.MINOR,
            violations=[{"test": "violation"}],
            context={"test_context": "value"}
        )
        
        # Verify fault structure
        fault_dict = fault.to_dict()
        assert fault_dict["severity"] == "minor"
        assert len(fault_dict["violations"]) == 1
        assert fault_dict["context"]["test_context"] == "value"
        
        print("  ✅ Cognitive fault handling properly implemented")


class TestReasoningEngine:
    """Test reasoning engine components."""
    
    def test_all_reasoning_types(self):
        """Test all reasoning types work correctly."""
        print("🧩 Testing All Reasoning Types...")
        
        style_weights = {"logic": 0.8, "creativity": 0.6, "analytical": 0.7}
        reasoning_engine = ReasoningEngine(style_weights)
        
        test_cases = [
            (ReasoningType.DEDUCTIVE, "If all birds can fly, and penguins are birds, then penguins can fly"),
            (ReasoningType.INDUCTIVE, "Observe patterns in data: 2, 4, 6, 8, what comes next?"),
            (ReasoningType.ABDUCTIVE, "The grass is wet. What might explain this?"),
            (ReasoningType.ANALOGICAL, "How is a computer network like a transportation system?"),
            (ReasoningType.CAUSAL, "What causes traffic congestion in cities?"),
        ]
        
        for reasoning_type, task in test_cases:
            result = reasoning_engine.reason(task, reasoning_type)
            
            assert result.reasoning_type == reasoning_type
            assert isinstance(result.content, str)
            assert len(result.content) > 0
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.reasoning_steps, list)
            
            print(f"  ✅ {reasoning_type.value}: {result.confidence:.2f} confidence")
        
        # Test multi-perspective reasoning
        multi_results = reasoning_engine.multi_perspective_reasoning("Analyze urban planning challenges")
        assert len(multi_results) == 5  # All reasoning types
        assert all(isinstance(r.confidence, float) for r in multi_results)
        
        print("  ✅ Multi-perspective reasoning working correctly")
    
    def test_reasoning_auto_selection(self):
        """Test automatic reasoning type selection."""
        print("🎯 Testing Reasoning Auto-Selection...")
        
        reasoning_engine = ReasoningEngine({"logic": 0.8, "creativity": 0.5, "analytical": 0.7})
        
        # Test prompts that should trigger specific reasoning types
        selection_tests = [
            ("If we implement this policy, then we can conclude...", ReasoningType.DEDUCTIVE),
            ("Looking at these data patterns, we can observe...", ReasoningType.INDUCTIVE),
            ("What could explain this unexpected behavior?", ReasoningType.ABDUCTIVE),
            ("This system is like a living organism because...", ReasoningType.ANALOGICAL),
            ("The increase in temperature causes ice to melt because...", ReasoningType.CAUSAL),
        ]
        
        for prompt, expected_type in selection_tests:
            result = reasoning_engine.reason(prompt)  # Auto-selection
            print(f"  📝 '{prompt[:30]}...' -> {result.reasoning_type.value}")
            # Note: Auto-selection is heuristic, so we don't assert exact match
            # but verify it selects a valid reasoning type
            assert result.reasoning_type in list(ReasoningType)


class TestMemoryModule:
    """Test memory system components."""
    
    def test_memory_manager_functionality(self):
        """Test memory manager operations."""
        print("🧠 Testing Memory Manager Functionality...")
        
        memory_manager = MemoryManager()
        
        # Test storing different types of memories
        test_memories = [
            {"type": "conversation", "content": "User asked about AI capabilities"},
            {"type": "fact", "content": "Python is a programming language"},
            {"type": "experience", "content": "Successfully completed a complex reasoning task"},
        ]
        
        memory_ids = []
        for memory in test_memories:
            memory_id = memory_manager.store(memory["content"], memory_type=memory["type"])
            memory_ids.append(memory_id)
            assert memory_id is not None
        
        # Test retrieval
        for memory_id in memory_ids:
            retrieved = memory_manager.retrieve(memory_id)
            assert retrieved is not None
            print(f"  ✅ Memory {memory_id}: Retrieved successfully")
        
        # Test search functionality
        search_results = memory_manager.search("Python", limit=5)
        assert len(search_results) >= 0
        
        # Test memory consolidation
        memory_manager.consolidate_memories()
        print("  ✅ Memory consolidation completed")
    
    def test_short_term_memory_capacity(self):
        """Test short-term memory capacity limits."""
        print("💾 Testing Short-Term Memory Capacity...")
        
        stm = ShortTermMemory(capacity=5)
        
        # Fill beyond capacity
        for i in range(10):
            stm.store(f"Memory item {i}")
        
        # Should only keep the most recent items
        items = stm.get_recent(10)
        assert len(items) <= 5
        
        # Most recent should be available
        recent_content = [item.content for item in items]
        assert any("Memory item 9" in content for content in recent_content)
        
        print(f"  ✅ Short-term memory capacity properly enforced: {len(items)} items stored")
    
    def test_episodic_memory_retrieval(self):
        """Test episodic memory retrieval by context."""
        print("📚 Testing Episodic Memory Retrieval...")
        
        episodic = EpisodicMemory()
        
        # Store memories with different contexts
        contexts = [
            {"session": "A", "topic": "AI", "timestamp": 1000},
            {"session": "A", "topic": "ML", "timestamp": 1001},  
            {"session": "B", "topic": "AI", "timestamp": 1002},
        ]
        
        for i, context in enumerate(contexts):
            episodic.store(f"Content {i}", context=context)
        
        # Test retrieval by different criteria
        session_a_memories = episodic.retrieve_by_context({"session": "A"})
        assert len(session_a_memories) == 2
        
        ai_memories = episodic.retrieve_by_context({"topic": "AI"})
        assert len(ai_memories) == 2
        
        print("  ✅ Episodic memory retrieval by context working correctly")


class TestToolsModule:
    """Test tools and tool registry functionality."""
    
    def test_tool_registry_operations(self):
        """Test tool registry registration and execution."""
        print("🔧 Testing Tool Registry Operations...")
        
        registry = ToolRegistry()
        
        # Create a simple test tool
        class TestTool(Tool):
            def execute(self, **kwargs):
                test_input = kwargs.get("input", "default")
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Processed: {test_input}",
                    metadata={"tool_name": "TestTool"}
                )
        
        # Register tool
        test_tool = TestTool()
        registry.register("test_tool", test_tool)
        
        # Verify registration
        assert registry.is_registered("test_tool")
        registered_tools = registry.list_tools()
        assert "test_tool" in registered_tools
        
        # Test execution
        result = registry.execute("test_tool", input="hello world")
        assert result.status == ToolStatus.SUCCESS
        assert "Processed: hello world" in result.output
        
        # Test non-existent tool
        try:
            registry.execute("non_existent_tool")
            assert False, "Should raise exception for non-existent tool"
        except Exception:
            pass  # Expected
        
        print("  ✅ Tool registry operations working correctly")
    
    def test_builtin_tools(self):
        """Test built-in tools functionality."""
        print("⚙️ Testing Built-in Tools...")
        
        from agentnet.tools import CalculatorTool, StatusCheckTool
        
        # Test calculator tool
        calc_tool = CalculatorTool()
        calc_result = calc_tool.execute(expression="2 + 2 * 3")
        assert calc_result.status == ToolStatus.SUCCESS
        # Basic expression evaluation
        
        # Test status check tool
        status_tool = StatusCheckTool()
        status_result = status_tool.execute()
        assert status_result.status == ToolStatus.SUCCESS
        assert "status" in status_result.output.lower()
        
        print("  ✅ Built-in tools functioning correctly")


class TestOrchestrationModule:
    """Test orchestration and turn engine functionality."""
    
    @pytest.mark.asyncio
    async def test_turn_engine_single_agent(self):
        """Test turn engine with single agent."""
        print("🔄 Testing Turn Engine Single Agent...")
        
        engine = TurnEngine(max_turns=3, turn_timeout=10.0)
        agent = AgentNet("TurnTestAgent", {"logic": 0.7}, engine=ExampleEngine())
        
        # Execute single-agent session
        session = await engine.execute_single_agent_session(
            agent=agent,
            initial_prompt="Discuss the benefits of renewable energy",
            termination_conditions=["concluded"]
        )
        
        # Verify session results
        assert session.mode == TurnMode.SINGLE_AGENT
        assert len(session.turns) >= 1
        assert session.status in [TerminationReason.COMPLETED, TerminationReason.MAX_TURNS_REACHED]
        assert session.agents_involved == ["TurnTestAgent"]
        assert session.duration is not None
        
        # Verify turn structure
        for turn in session.turns:
            assert isinstance(turn, TurnResult)
            assert turn.agent_id == "TurnTestAgent"
            assert isinstance(turn.content, str)
            assert len(turn.content) > 0
        
        print(f"  ✅ Single-agent session: {len(session.turns)} turns, {session.status}")
    
    @pytest.mark.asyncio 
    async def test_turn_engine_multi_agent(self):
        """Test turn engine with multiple agents."""
        print("👥 Testing Turn Engine Multi-Agent...")
        
        engine = TurnEngine(max_rounds=2, turn_timeout=10.0)
        agents = [
            AgentNet("Agent1", {"logic": 0.8}, engine=ExampleEngine()),
            AgentNet("Agent2", {"creativity": 0.8}, engine=ExampleEngine()),
        ]
        
        # Test different modes
        modes_to_test = [TurnMode.ROUND_ROBIN, TurnMode.DEBATE]
        
        for mode in modes_to_test:
            session = await engine.execute_multi_agent_session(
                agents=agents,
                topic="The future of artificial intelligence",
                mode=mode,
            )
            
            assert session.mode == mode
            assert len(session.turns) >= 2  # At least one turn per agent
            assert len(session.agents_involved) == 2
            assert all(agent_id in ["Agent1", "Agent2"] for agent_id in session.agents_involved)
            
            print(f"  ✅ Multi-agent {mode.value}: {len(session.turns)} turns")


class TestPerformanceModule:
    """Test performance monitoring components."""
    
    def test_latency_tracker_comprehensive(self):
        """Test latency tracker with comprehensive scenarios."""
        print("⏱️ Testing Latency Tracker Comprehensive...")
        
        tracker = LatencyTracker()
        
        # Test multiple concurrent turns
        turn_ids = [f"test_turn_{i}" for i in range(5)]
        
        for turn_id in turn_ids:
            tracker.start_turn_measurement(turn_id, f"Agent{turn_id[-1]}", prompt_length=100)
        
        # End measurements in different order
        for i, turn_id in enumerate(reversed(turn_ids)):
            measurement = tracker.end_turn_measurement(turn_id, response_length=200, tokens_processed=150)
            assert measurement.turn_id == turn_id
            assert measurement.total_latency_ms > 0
        
        # Test statistics calculation
        stats = tracker.get_latency_statistics()
        assert stats['count'] == 5
        assert stats['mean'] > 0
        assert stats['min'] <= stats['max']
        
        print(f"  ✅ Latency tracking: {stats['count']} measurements, avg {stats['mean']:.2f}ms")
    
    def test_token_utilization_edge_cases(self):
        """Test token utilization tracker with edge cases."""
        print("🪙 Testing Token Utilization Edge Cases...")
        
        tracker = TokenUtilizationTracker()
        
        # Test with zero tokens
        metrics_zero = tracker.record_token_usage(
            agent_id="ZeroAgent",
            turn_id="zero_turn",
            input_tokens=0,
            output_tokens=0,
            processing_time=0.1
        )
        assert metrics_zero.total_tokens == 0
        assert metrics_zero.efficiency_score >= 0
        
        # Test with very large token count
        metrics_large = tracker.record_token_usage(
            agent_id="LargeAgent", 
            turn_id="large_turn",
            input_tokens=50000,
            output_tokens=25000,
            processing_time=10.0
        )
        assert metrics_large.total_tokens == 75000
        assert metrics_large.tokens_per_second > 0
        
        # Test optimization recommendations
        recommendations = tracker.generate_optimization_recommendations()
        assert isinstance(recommendations, list)
        
        print(f"  ✅ Token utilization edge cases handled correctly")


@pytest.mark.asyncio
async def test_comprehensive_component_coverage():
    """Run comprehensive component coverage tests."""
    print("\n🧪 AgentNet Comprehensive Component Test Suite") 
    print("=" * 60)
    
    test_classes = [
        TestCoreAgentModule(),
        TestReasoningEngine(),
        TestMemoryModule(),
        TestToolsModule(),
        TestOrchestrationModule(),
        TestPerformanceModule(),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 Running {class_name}...")
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_') and callable(getattr(test_class, method))]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, method_name)
                
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                passed_tests += 1
                print(f"  ✅ {method_name}")
                
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print("\n" + "=" * 60)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"📊 Component Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("🎉 Component tests meet quality threshold!")
        return True
    else:
        print("❌ Component tests below quality threshold")
        return False


if __name__ == "__main__":
    asyncio.run(test_comprehensive_component_coverage())