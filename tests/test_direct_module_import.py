#!/usr/bin/env python3
"""
Direct module test - imports modules directly without going through agentnet package.
"""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_direct_reasoning_import():
    """Test direct import of reasoning modules."""
    print("🧠 Testing Direct Reasoning Module Import...")

    try:
        # Direct import without going through agentnet package
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agentnet"
            ),
        )

        from reasoning.types import DeductiveReasoning, ReasoningEngine, ReasoningType

        print("  ✅ Reasoning types imported directly")

        from reasoning.modulation import ReasoningStyleModulator

        print("  ✅ Reasoning modulation imported directly")

        # Test basic functionality
        style_weights = {"logic": 0.8, "creativity": 0.6, "analytical": 0.9}
        engine = ReasoningEngine(style_weights)
        print("  ✅ Reasoning engine created")

        task = "All software bugs are fixable. This is a software bug."
        result = engine.reason(task)
        print(f"  Reasoning type: {result.reasoning_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")

        # Assert success instead of returning
        assert result is not None
        assert hasattr(result, "reasoning_type")
        assert hasattr(result, "confidence")

    except Exception as e:
        print(f"  ❌ Direct reasoning import failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Direct reasoning import test failed: {e}")


def test_direct_dialogue_import():
    """Test direct import of dialogue modules."""
    print("\n💬 Testing Direct Dialogue Module Import...")

    try:
        from dialogue.enhanced_modes import DialogueMode, OuterDialogue

        print("  ✅ Enhanced dialogue modes imported directly")

        from dialogue.solution_focused import SolutionFocusedDialogue

        print("  ✅ Solution-focused dialogue imported directly")

        from dialogue.mapping import DialogueMapper

        print("  ✅ Dialogue mapping imported directly")

        # Test basic functionality
        outer_dialogue = OuterDialogue()
        print("  ✅ Outer dialogue instance created")

        solution_dialogue = SolutionFocusedDialogue()
        print("  ✅ Solution-focused dialogue instance created")

        mapper = DialogueMapper()
        print("  ✅ Dialogue mapper instance created")

        # Assert success instead of returning
        assert outer_dialogue is not None
        assert solution_dialogue is not None
        assert mapper is not None

    except Exception as e:
        print(f"  ❌ Direct dialogue import failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Direct dialogue import test failed: {e}")


def test_reasoning_functionality():
    """Test reasoning functionality directly."""
    print("\n🧪 Testing Reasoning Functionality...")

    try:
        from reasoning.types import ReasoningEngine, ReasoningType

        style_weights = {"logic": 0.8, "creativity": 0.6, "analytical": 0.9}
        engine = ReasoningEngine(style_weights)

        # Test different reasoning types
        test_cases = [
            (
                "All birds can fly. Penguins are birds. Therefore penguins can fly.",
                "deductive",
            ),
            (
                "I've seen 5 swans and they were all white. All swans might be white.",
                "inductive",
            ),
            (
                "The lights went out. The most likely cause is a power outage.",
                "abductive",
            ),
            ("The brain is like a computer processing information.", "analogical"),
            ("Heavy rain caused flooding which led to road closures.", "causal"),
        ]

        for task, expected_category in test_cases:
            result = engine.reason(task)
            print(f"  Task: {task[:50]}...")
            print(
                f"    Type: {result.reasoning_type.value} (expected: {expected_category})"
            )
            print(f"    Confidence: {result.confidence:.2f}")
            print(f"    Steps: {len(result.reasoning_steps)}")

        # Test multi-perspective reasoning
        multi_results = engine.multi_perspective_reasoning(
            "How to design a secure system?"
        )
        print(
            f"\n  Multi-perspective reasoning generated {len(multi_results)} perspectives"
        )

        # Assert success instead of returning
        assert multi_results is not None
        assert len(multi_results) > 0

    except Exception as e:
        print(f"  ❌ Reasoning functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Reasoning functionality test failed: {e}")


def test_style_modulation_functionality():
    """Test style modulation functionality."""
    print("\n🎛️ Testing Style Modulation Functionality...")

    try:
        from reasoning.modulation import ReasoningStyleModulator
        from reasoning.types import ReasoningEngine, ReasoningType

        modulator = ReasoningStyleModulator()

        # Test different base styles
        test_styles = [
            {"logic": 0.9, "creativity": 0.3, "analytical": 0.7},  # Logic-heavy
            {"logic": 0.4, "creativity": 0.9, "analytical": 0.5},  # Creative
            {"logic": 0.6, "creativity": 0.4, "analytical": 0.9},  # Analytical
        ]

        for i, base_style in enumerate(test_styles):
            print(f"\n  Style profile {i+1}: {base_style}")

            # Test modulation for deductive reasoning
            modulated = modulator.modulate_style_for_reasoning(
                base_style, ReasoningType.DEDUCTIVE, 0.5
            )
            print(f"    Modulated for deductive: {modulated}")

            # Test reasoning type suggestions
            suggestions = modulator.suggest_reasoning_types(base_style)
            print(f"    Suggested types: {[rt.value for rt in suggestions[:3]]}")

        # Assert success instead of returning
        assert suggestions is not None
        assert len(suggestions) > 0

    except Exception as e:
        print(f"  ❌ Style modulation functionality test failed: {e}")
        pytest.fail(f"Style modulation functionality test failed: {e}")


def test_standalone_agentnet():
    """Test AgentNet functionality standalone."""
    print("\n🤖 Testing Standalone AgentNet...")

    try:
        # Import AgentNet directly from the file
        from AgentNet import AgentNet

        # Create test agents
        agent1 = AgentNet(
            "TestAgent1", {"logic": 0.8, "creativity": 0.5, "analytical": 0.9}
        )
        agent2 = AgentNet(
            "TestAgent2", {"logic": 0.6, "creativity": 0.9, "analytical": 0.4}
        )

        print("  ✅ Test agents created")

        # Test existing dialogue functionality
        result = agent1.dialogue_with(agent2, "Test dialogue", rounds=2)
        print(f"  Dialogue participants: {result['participants']}")
        print(f"  Dialogue rounds: {len(result['dialogue_history'])}")

        # Test multi-party dialogue
        multi_result = agent1.multi_party_dialogue(
            [agent1, agent2], "Test topic", rounds=2
        )
        print(f"  Multi-party transcript: {len(multi_result['transcript'])} turns")

        # Assert success instead of returning
        assert multi_result is not None
        assert "transcript" in multi_result
        assert len(multi_result["transcript"]) > 0

    except Exception as e:
        print(f"  ❌ Standalone AgentNet test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Standalone AgentNet test failed: {e}")


def main():
    """Run direct module tests."""
    print("🚀 Direct Module Test Suite - Enhanced Dialogue and Reasoning")
    print("=" * 70)

    tests = [
        test_direct_reasoning_import,
        test_direct_dialogue_import,
        test_reasoning_functionality,
        test_style_modulation_functionality,
        test_standalone_agentnet,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()  # Run test function - will raise assertion error if fails
            passed += 1
            print("  ✅ PASSED")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1

    print(f"\n📊 Final Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All direct module tests passed!")
        print(
            "\n✨ Enhanced dialogue modes and reasoning types are successfully implemented!"
        )
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
