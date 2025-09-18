#!/usr/bin/env python3
"""
Basic test suite for enhanced dialogue modes and advanced reasoning types.
Tests core functionality without external dependencies.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_import():
    """Test that all modules can be imported."""
    print("📦 Testing Module Imports...")
    
    try:
        from AgentNet import AgentNet
        print("  ✅ AgentNet imported")
        
        from agentnet.reasoning.types import ReasoningType, ReasoningEngine
        print("  ✅ Reasoning types imported")
        
        from agentnet.reasoning.modulation import ReasoningStyleModulator
        print("  ✅ Reasoning modulation imported")
        
        from agentnet.dialogue.enhanced_modes import DialogueMode, OuterDialogue
        print("  ✅ Enhanced dialogue modes imported")
        
        from agentnet.dialogue.solution_focused import SolutionFocusedDialogue
        print("  ✅ Solution-focused dialogue imported")
        
        from agentnet.dialogue.mapping import DialogueMapper
        print("  ✅ Dialogue mapping imported")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_reasoning_engine_basic():
    """Test basic reasoning engine functionality."""
    print("\n🧠 Testing Basic Reasoning Engine...")
    
    try:
        from agentnet.reasoning.types import ReasoningEngine, ReasoningType
        
        # Create reasoning engine
        style_weights = {"logic": 0.8, "creativity": 0.6, "analytical": 0.9}
        engine = ReasoningEngine(style_weights)
        print("  ✅ Reasoning engine created")
        
        # Test reasoning
        task = "All software bugs are fixable. This is a software bug. Therefore, this bug is fixable."
        result = engine.reason(task)
        
        print(f"  Task: {task[:50]}...")
        print(f"  Reasoning type: {result.reasoning_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Steps: {len(result.reasoning_steps)}")
        
        # Test specific reasoning type
        deductive_result = engine.reason(task, ReasoningType.DEDUCTIVE)
        print(f"  Deductive reasoning confidence: {deductive_result.confidence:.2f}")
        
        print("  ✅ Basic reasoning engine test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Reasoning engine test failed: {e}")
        return False


def test_style_modulation_basic():
    """Test basic style modulation."""
    print("\n🎛️ Testing Basic Style Modulation...")
    
    try:
        from agentnet.reasoning.modulation import ReasoningStyleModulator
        from agentnet.reasoning.types import ReasoningType
        
        modulator = ReasoningStyleModulator()
        print("  ✅ Style modulator created")
        
        # Test style modulation
        base_style = {"logic": 0.9, "creativity": 0.3, "analytical": 0.7}
        modulated = modulator.modulate_style_for_reasoning(
            base_style, ReasoningType.DEDUCTIVE, 0.5
        )
        
        print(f"  Base style: {base_style}")
        print(f"  Modulated for deductive: {modulated}")
        
        # Test suggestions
        suggestions = modulator.suggest_reasoning_types(base_style)
        print(f"  Suggested reasoning types: {[rt.value for rt in suggestions[:3]]}")
        
        print("  ✅ Basic style modulation test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Style modulation test failed: {e}")
        return False


def test_dialogue_modes_basic():
    """Test basic dialogue modes."""
    print("\n💬 Testing Basic Dialogue Modes...")
    
    try:
        from AgentNet import AgentNet
        from agentnet.dialogue.enhanced_modes import OuterDialogue
        
        # Create test agents
        agents = [
            AgentNet("TestAgent1", {"logic": 0.8, "creativity": 0.5, "analytical": 0.9}),
            AgentNet("TestAgent2", {"logic": 0.6, "creativity": 0.9, "analytical": 0.4})
        ]
        print("  ✅ Test agents created")
        
        # Test outer dialogue
        outer_dialogue = OuterDialogue()
        result = outer_dialogue.conduct_dialogue(
            agents, "Test topic", rounds=2, conversation_pattern="collaborative"
        )
        
        print(f"  Dialogue completed: {result['rounds_completed']} rounds")
        print(f"  Participants: {result['participants']}")
        print(f"  Transcript length: {len(result['transcript'])}")
        
        assert len(result['transcript']) > 0
        assert result['mode'] == 'outer'
        
        print("  ✅ Basic dialogue modes test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Dialogue modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solution_focused_basic():
    """Test basic solution-focused dialogue."""
    print("\n🎯 Testing Basic Solution-Focused Dialogue...")
    
    try:
        from AgentNet import AgentNet
        from agentnet.dialogue.solution_focused import SolutionFocusedDialogue, ConversationPhase
        
        # Create test agents
        agents = [
            AgentNet("Manager", {"logic": 0.7, "creativity": 0.6, "analytical": 0.8})
        ]
        print("  ✅ Test agents created")
        
        # Test solution-focused dialogue
        solution_dialogue = SolutionFocusedDialogue()
        result = solution_dialogue.conduct_solution_focused_dialogue(
            agents, "Our system has performance issues", rounds=3
        )
        
        print(f"  Phases: {result['phase_progression']}")
        print(f"  Problems identified: {result['analysis']['problems_identified']}")
        print(f"  Solutions proposed: {result['analysis']['solutions_proposed']}")
        
        assert result['rounds_completed'] == 3
        assert len(result['transcript']) > 0
        
        print("  ✅ Basic solution-focused dialogue test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Solution-focused dialogue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dialogue_mapping_basic():
    """Test basic dialogue mapping."""
    print("\n🗺️ Testing Basic Dialogue Mapping...")
    
    try:
        from agentnet.dialogue.mapping import DialogueMapper, NodeType, EdgeType
        from agentnet.dialogue.enhanced_modes import DialogueState, DialogueTurn, DialogueMode
        import time
        
        # Create dialogue mapper
        mapper = DialogueMapper()
        print("  ✅ Dialogue mapper created")
        
        # Create mock dialogue state
        dialogue_state = DialogueState(
            session_id="test_basic",
            mode=DialogueMode.OUTER,
            participants=["Agent1", "Agent2"],
            topic="Test mapping",
            current_round=2,
            transcript=[]
        )
        
        # Create mock turns
        turns = [
            DialogueTurn("Agent1", "Let's discuss the problem", 0.8, DialogueMode.OUTER, 1, time.time()),
            DialogueTurn("Agent2", "What are the requirements?", 0.7, DialogueMode.OUTER, 1, time.time())
        ]
        dialogue_state.transcript = turns
        
        # Create dialogue map
        dialogue_map = mapper.create_dialogue_map(dialogue_state)
        
        print(f"  Map created with {len(dialogue_map['nodes'])} nodes")
        print(f"  Map has {len(dialogue_map['edges'])} edges")
        print(f"  Map topic: {dialogue_map['topic']}")
        
        assert len(dialogue_map['nodes']) >= 3  # At least topic + 2 turns  
        assert len(dialogue_map['edges']) >= 2  # At least 2 connections
        
        print("  ✅ Basic dialogue mapping test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Dialogue mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_basic():
    """Test basic integration with existing AgentNet."""
    print("\n🔗 Testing Basic Integration...")
    
    try:
        from AgentNet import AgentNet
        from agentnet.reasoning.types import ReasoningEngine
        
        # Create agent
        agent = AgentNet("IntegrationTest", {"logic": 0.8, "creativity": 0.6, "analytical": 0.9})
        print("  ✅ Agent created")
        
        # Test existing functionality still works
        result = agent.generate_reasoning_tree("Test task")
        print(f"  Existing reasoning tree: {result['agent']}")
        
        # Test new reasoning functionality
        reasoning_engine = ReasoningEngine(agent.style)
        reasoning_result = reasoning_engine.reason("Test reasoning task")
        print(f"  New reasoning type: {reasoning_result.reasoning_type.value}")
        
        print("  ✅ Basic integration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all basic tests."""
    print("🚀 Enhanced Dialogue Modes and Advanced Reasoning Types - Basic Test Suite")
    print("=" * 80)
    
    tests = [
        test_basic_import,
        test_reasoning_engine_basic,
        test_style_modulation_basic,
        test_dialogue_modes_basic,
        test_solution_focused_basic,
        test_dialogue_mapping_basic,
        test_integration_basic
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed! Enhanced dialogue modes and reasoning types are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)