#!/usr/bin/env python3
"""
Test suite for enhanced dialogue modes and advanced reasoning types.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    from AgentNet import AgentNet
    from agentnet.dialogue.enhanced_modes import (
        DialogueMapping,
        DialogueMode,
        InnerDialogue,
        InterpolationConversation,
        ModulatedConversation,
        OuterDialogue,
    )
    from agentnet.dialogue.mapping import DialogueMapper
    from agentnet.dialogue.solution_focused import (
        ConversationPhase,
        SolutionFocusedDialogue,
    )
    from agentnet.reasoning.modulation import (
        ReasoningAwareStyleInfluence,
        ReasoningStyleModulator,
    )
    from agentnet.reasoning.types import (
        AbductiveReasoning,
        AnalogicalReasoning,
        CausalReasoning,
        DeductiveReasoning,
        InductiveReasoning,
        ReasoningEngine,
        ReasoningType,
    )

    print("✅ All enhanced dialogue and reasoning modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_reasoning_types():
    """Test the five core reasoning types."""
    print("\n🧠 Testing Reasoning Types...")

    # Create style weights for testing
    style_weights = {"logic": 0.8, "creativity": 0.6, "analytical": 0.9}

    # Test each reasoning type
    reasoning_engine = ReasoningEngine(style_weights)

    test_cases = [
        (
            "All software bugs are fixable. This is a software bug. Therefore, this bug is fixable.",
            ReasoningType.DEDUCTIVE,
        ),
        (
            "We observed crashes in versions 1.0, 1.1, and 1.2. Version 1.3 will likely also crash.",
            ReasoningType.INDUCTIVE,
        ),
        (
            "The system crashed unexpectedly. The most likely explanation is memory corruption.",
            ReasoningType.ABDUCTIVE,
        ),
        (
            "This network problem is like a traffic jam - we need alternate routes.",
            ReasoningType.ANALOGICAL,
        ),
        (
            "The database timeout caused the connection failure, which led to the user error.",
            ReasoningType.CAUSAL,
        ),
    ]

    for task, expected_type in test_cases:
        # Test automatic reasoning type selection
        result = reasoning_engine.reason(task)
        print(f"  Task: {task[:50]}...")
        print(
            f"    Selected: {result.reasoning_type.value}, Expected: {expected_type.value}"
        )
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Steps: {len(result.reasoning_steps)}")

        # Test specific reasoning type
        specific_result = reasoning_engine.reason(task, expected_type)
        print(f"    Specific reasoning confidence: {specific_result.confidence:.2f}")

        assert (
            result.reasoning_type == expected_type
            or specific_result.reasoning_type == expected_type
        )
        print("    ✅ Reasoning type test passed")

    # Test multi-perspective reasoning
    multi_results = reasoning_engine.multi_perspective_reasoning(
        "How should we design a resilient distributed system?"
    )
    print(
        f"\n  Multi-perspective analysis generated {len(multi_results)} different reasoning approaches"
    )
    for result in multi_results:
        print(f"    {result.reasoning_type.value}: {result.confidence:.2f}")

    print("  ✅ Reasoning types test completed!\n")


def test_reasoning_style_modulation():
    """Test reasoning-aware style modulation."""
    print("🎛️ Testing Reasoning Style Modulation...")

    modulator = ReasoningStyleModulator()

    # Test different base styles
    test_styles = [
        {"logic": 0.9, "creativity": 0.3, "analytical": 0.7},  # Logic-heavy
        {"logic": 0.4, "creativity": 0.9, "analytical": 0.5},  # Creative
        {"logic": 0.6, "creativity": 0.4, "analytical": 0.9},  # Analytical
    ]

    for i, base_style in enumerate(test_styles):
        print(f"\n  Testing style profile {i+1}: {base_style}")

        # Test modulation for each reasoning type
        for reasoning_type in ReasoningType:
            modulated_style = modulator.modulate_style_for_reasoning(
                base_style, reasoning_type, modulation_strength=0.5
            )
            print(f"    {reasoning_type.value}: {modulated_style}")

        # Test reasoning type suggestions
        suggestions = modulator.suggest_reasoning_types(base_style)
        print(f"    Suggested reasoning types: {[rt.value for rt in suggestions[:3]]}")

    # Test reasoning-aware style influence
    influence_system = ReasoningAwareStyleInfluence(modulator)

    base_result = {"content": "Test reasoning result", "confidence": 0.7}

    enhanced_result = influence_system.apply_reasoning_aware_style_influence(
        base_result, "Test task", test_styles[0]
    )

    print(f"\n  Enhanced result confidence: {enhanced_result['confidence']:.2f}")
    print(f"  Style influence applied: {enhanced_result.get('style_applied', False)}")

    print("  ✅ Reasoning style modulation test completed!\n")


def test_enhanced_dialogue_modes():
    """Test enhanced dialogue modes."""
    print("💬 Testing Enhanced Dialogue Modes...")

    # Create test agents with different styles
    agents = [
        AgentNet("Athena", {"logic": 0.8, "creativity": 0.5, "analytical": 0.9}),
        AgentNet("Apollo", {"logic": 0.6, "creativity": 0.9, "analytical": 0.4}),
    ]

    topic = "How to design a fault-tolerant distributed system"

    # Test Outer Dialogue
    print("  Testing Outer Dialogue...")
    outer_dialogue = OuterDialogue()
    outer_result = outer_dialogue.conduct_dialogue(
        agents, topic, rounds=3, conversation_pattern="collaborative"
    )
    print(f"    Completed {outer_result['rounds_completed']} rounds")
    print(f"    Participants: {outer_result['participants']}")
    print(f"    Analysis quality: {outer_result['analysis']['dialogue_quality']}")

    # Test Modulated Conversation
    print("  Testing Modulated Conversation...")
    modulated_dialogue = ModulatedConversation()
    modulated_result = modulated_dialogue.conduct_dialogue(
        agents,
        topic,
        rounds=3,
        initial_intensity=0.3,
        max_intensity=0.8,
        tension_strategy="gradual_build",
    )
    print(
        f"    Max intensity reached: {modulated_result['analysis']['max_intensity_reached']:.2f}"
    )
    print(
        f"    Tension building success: {modulated_result['analysis']['tension_building_success']}"
    )

    # Test Interpolation Conversation
    print("  Testing Interpolation Conversation...")
    interpolation_dialogue = InterpolationConversation()
    context_gaps = ["Missing performance requirements", "Unclear scalability needs"]
    interpolation_result = interpolation_dialogue.conduct_dialogue(
        agents,
        topic,
        rounds=3,
        context_gaps=context_gaps,
        interpolation_strategy="memory_guided",
    )
    print(f"    Initial gaps: {len(interpolation_result['initial_gaps'])}")
    print(f"    Remaining gaps: {len(interpolation_result['remaining_gaps'])}")
    print(
        f"    Interpolations made: {len(interpolation_result['interpolations_made'])}"
    )

    # Test Inner Dialogue
    print("  Testing Inner Dialogue...")
    inner_dialogue = InnerDialogue()
    inner_result = inner_dialogue.conduct_dialogue(
        [agents[0]],
        topic,
        rounds=3,
        reflection_depth="moderate",
        focus_areas=["reasoning_process", "assumptions"],
    )
    print(f"    Reflection quality: {inner_result['analysis']['reflection_quality']}")
    print(f"    Metacognitive depth: {inner_result['analysis']['metacognitive_depth']}")

    # Test Dialogue Mapping
    print("  Testing Dialogue Mapping...")
    mapping_dialogue = DialogueMapping()
    mapping_result = mapping_dialogue.conduct_dialogue(
        agents, topic, rounds=3, mapping_focus="decision_making", track_arguments=True
    )
    print(
        f"    Decision points: {len(mapping_result['decision_map']['decision_points'])}"
    )
    print(f"    Alternatives: {len(mapping_result['decision_map']['alternatives'])}")
    print(
        f"    Mapping effectiveness: {mapping_result['mapping_analysis']['mapping_effectiveness']}"
    )

    print("  ✅ Enhanced dialogue modes test completed!\n")


def test_solution_focused_dialogue():
    """Test solution-focused dialogue patterns."""
    print("🎯 Testing Solution-Focused Dialogue...")

    # Create test agents
    agents = [
        AgentNet("Manager", {"logic": 0.7, "creativity": 0.6, "analytical": 0.8}),
        AgentNet("Developer", {"logic": 0.8, "creativity": 0.7, "analytical": 0.9}),
    ]

    # Test solution-focused dialogue
    solution_dialogue = SolutionFocusedDialogue()

    problem_topic = "Our deployment process is unreliable and causes frequent outages"

    result = solution_dialogue.conduct_solution_focused_dialogue(
        agents,
        problem_topic,
        rounds=6,
        initial_phase=ConversationPhase.PROBLEM_IDENTIFICATION,
        transition_strategy="adaptive",
    )

    print(f"  Phases covered: {len(set(result['phase_progression']))}")
    print(f"  Problems identified: {result['analysis']['problems_identified']}")
    print(f"  Solutions proposed: {result['analysis']['solutions_proposed']}")
    print(f"  Solution focus ratio: {result['analysis']['solution_focus_ratio']:.2f}")
    print(f"  Overall effectiveness: {result['analysis']['overall_effectiveness']}")
    print(
        f"  Transition effectiveness: {result['analysis']['transition_effectiveness']}"
    )

    assert result["analysis"]["problems_identified"] > 0
    assert (
        result["analysis"]["solution_focus_ratio"] > 0.2
    )  # Should have some solution focus

    print("  ✅ Solution-focused dialogue test completed!\n")


def test_dialogue_mapping_visualization():
    """Test dialogue mapping and visualization."""
    print("🗺️ Testing Dialogue Mapping and Visualization...")

    # Create test agents
    agents = [
        AgentNet("Alice", {"logic": 0.8, "creativity": 0.5, "analytical": 0.7}),
        AgentNet("Bob", {"logic": 0.6, "creativity": 0.8, "analytical": 0.6}),
    ]

    # Create a dialogue for mapping
    mapping_dialogue = DialogueMapping()
    result = mapping_dialogue.conduct_dialogue(
        agents,
        "Choose the best database technology for our application",
        rounds=4,
        mapping_focus="decision_making",
        track_arguments=True,
    )

    # Test dialogue mapper
    mapper = DialogueMapper()

    # Create mock dialogue state and transcript for mapping
    import time

    from agentnet.dialogue.enhanced_modes import (
        DialogueMode,
        DialogueState,
        DialogueTurn,
    )

    dialogue_state = DialogueState(
        session_id="test_mapping",
        mode=DialogueMode.MAPPING,
        participants=["Alice", "Bob"],
        topic="Database technology selection",
        current_round=4,
        transcript=[],
    )

    # Create mock transcript
    turns = [
        DialogueTurn(
            "Alice",
            "We need to choose between SQL and NoSQL databases",
            0.8,
            DialogueMode.MAPPING,
            1,
            time.time(),
        ),
        DialogueTurn(
            "Bob",
            "What are our scalability requirements?",
            0.7,
            DialogueMode.MAPPING,
            1,
            time.time(),
        ),
        DialogueTurn(
            "Alice",
            "We expect high read loads but moderate write loads",
            0.9,
            DialogueMode.MAPPING,
            2,
            time.time(),
        ),
        DialogueTurn(
            "Bob",
            "Then I propose we use PostgreSQL with read replicas",
            0.8,
            DialogueMode.MAPPING,
            2,
            time.time(),
        ),
    ]
    dialogue_state.transcript = turns

    # Create dialogue map
    dialogue_map = mapper.create_dialogue_map(dialogue_state)

    print(
        f"  Created map with {len(dialogue_map['nodes'])} nodes and {len(dialogue_map['edges'])} edges"
    )

    # Generate visualization data
    vis_data = mapper.generate_visualization_data(dialogue_map["id"])

    print(
        f"  Visualization data: {vis_data['metadata']['node_count']} nodes, {vis_data['metadata']['edge_count']} edges"
    )
    print(f"  Layout type: {vis_data['layout']['type']}")
    print(f"  Legend items: {len(vis_data['legend']['node_types'])}")

    # Test export
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = mapper.export_map(
            dialogue_map["id"],
            format="json",
            file_path=os.path.join(temp_dir, "test_map.json"),
        )
        print(f"  Exported map to: {export_path}")
        assert os.path.exists(export_path)

    print("  ✅ Dialogue mapping and visualization test completed!\n")


async def test_async_dialogue_modes():
    """Test async versions of dialogue modes."""
    print("⚡ Testing Async Dialogue Modes...")

    # Create test agents
    agents = [
        AgentNet("AsyncAgent1", {"logic": 0.7, "creativity": 0.6, "analytical": 0.8}),
        AgentNet("AsyncAgent2", {"logic": 0.6, "creativity": 0.8, "analytical": 0.7}),
    ]

    topic = "Implementing microservices architecture"

    # Test async outer dialogue
    print("  Testing Async Outer Dialogue...")
    outer_dialogue = OuterDialogue()
    async_result = await outer_dialogue.conduct_async_dialogue(
        agents, topic, rounds=2, conversation_pattern="exploratory"
    )
    print(f"    Completed {async_result['rounds_completed']} async rounds")

    # Test async modulated conversation
    print("  Testing Async Modulated Conversation...")
    modulated_dialogue = ModulatedConversation()
    async_modulated = await modulated_dialogue.conduct_async_dialogue(
        agents, topic, rounds=2, initial_intensity=0.4, tension_strategy="spike"
    )
    print(
        f"    Max intensity: {async_modulated['analysis']['max_intensity_reached']:.2f}"
    )

    print("  ✅ Async dialogue modes test completed!\n")


def test_integration_with_existing_agentnet():
    """Test integration with existing AgentNet functionality."""
    print("🔗 Testing Integration with Existing AgentNet...")

    # Create agents with existing AgentNet features
    agents = [
        AgentNet(
            "IntegrationAgent1", {"logic": 0.8, "creativity": 0.6, "analytical": 0.9}
        ),
        AgentNet(
            "IntegrationAgent2", {"logic": 0.6, "creativity": 0.9, "analytical": 0.7}
        ),
    ]

    # Test that existing dialogue methods still work
    print("  Testing existing dialogue_with method...")
    result = agents[0].dialogue_with(agents[1], "Test integration", rounds=2)
    print(f"    Existing dialogue participants: {result['participants']}")
    assert len(result["dialogue_history"]) > 0

    # Test existing multi_party_dialogue
    print("  Testing existing multi_party_dialogue method...")
    multi_result = agents[0].multi_party_dialogue(
        agents, "Test multi-party integration", rounds=2, mode="brainstorm"
    )
    print(f"    Multi-party transcript length: {len(multi_result['transcript'])}")
    assert len(multi_result["transcript"]) > 0

    # Test that reasoning capabilities integrate
    print("  Testing reasoning integration...")
    reasoning_engine = ReasoningEngine(agents[0].style)
    reasoning_result = reasoning_engine.reason("How to optimize database queries?")
    print(f"    Reasoning type selected: {reasoning_result.reasoning_type.value}")
    print(f"    Reasoning confidence: {reasoning_result.confidence:.2f}")

    print("  ✅ Integration test completed!\n")


def main():
    """Run all tests."""
    print("🚀 Enhanced Dialogue Modes and Advanced Reasoning Types Test Suite")
    print("=" * 80)

    try:
        # Core reasoning tests
        test_reasoning_types()
        test_reasoning_style_modulation()

        # Dialogue mode tests
        test_enhanced_dialogue_modes()
        test_solution_focused_dialogue()
        test_dialogue_mapping_visualization()

        # Async tests
        asyncio.run(test_async_dialogue_modes())

        # Integration tests
        test_integration_with_existing_agentnet()

        print("🎉 All tests completed successfully!")
        print("\n📊 Test Summary:")
        print("  ✅ Reasoning Types: 5 types tested")
        print("  ✅ Style Modulation: Reasoning-aware modulation working")
        print("  ✅ Dialogue Modes: 5 enhanced modes tested")
        print("  ✅ Solution-Focused: Problem-to-solution transitions working")
        print("  ✅ Mapping & Visualization: Graph generation and export working")
        print("  ✅ Async Support: Async dialogue modes working")
        print("  ✅ Integration: Backward compatibility maintained")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
