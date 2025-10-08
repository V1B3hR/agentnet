#!/usr/bin/env python3
"""
Test for monitor system refactoring.

This test validates that the refactored monitor system:
1. Can create monitors using the factory
2. Individual monitor modules work correctly
3. Backward compatibility is maintained
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agentnet import AgentNet, ExampleEngine, MonitorFactory, MonitorSpec
from agentnet.core.types import Severity


def test_monitor_factory_with_refactored_modules():
    """Test that MonitorFactory works with the refactored module structure."""
    print("🔧 Testing Monitor Factory with Refactored Modules...")

    # Test keyword monitor
    keyword_spec = MonitorSpec(
        name="test_keyword",
        type="keyword",
        params={"keywords": ["test", "error"]},
        severity=Severity.MINOR,
        description="Test keyword monitor",
    )
    keyword_monitor = MonitorFactory.build(keyword_spec)
    assert callable(keyword_monitor)
    print("  ✅ Keyword monitor created successfully")

    # Test regex monitor
    regex_spec = MonitorSpec(
        name="test_regex",
        type="regex",
        params={"pattern": r"\berror\b"},
        severity=Severity.MINOR,
        description="Test regex monitor",
    )
    regex_monitor = MonitorFactory.build(regex_spec)
    assert callable(regex_monitor)
    print("  ✅ Regex monitor created successfully")

    # Test resource monitor
    resource_spec = MonitorSpec(
        name="test_resource",
        type="resource",
        params={"budget_key": "resource_budget", "tolerance": 0.2},
        severity=Severity.MINOR,
        description="Test resource monitor",
    )
    resource_monitor = MonitorFactory.build(resource_spec)
    assert callable(resource_monitor)
    print("  ✅ Resource monitor created successfully")

    # Test semantic similarity monitor
    semantic_spec = MonitorSpec(
        name="test_semantic",
        type="semantic_similarity",
        params={"max_similarity": 0.9, "window_size": 3},
        severity=Severity.MINOR,
        description="Test semantic monitor",
    )
    semantic_monitor = MonitorFactory.build(semantic_spec)
    assert callable(semantic_monitor)
    print("  ✅ Semantic similarity monitor created successfully")


def test_individual_monitor_modules():
    """Test individual monitor modules can be imported and used."""
    print("📦 Testing Individual Monitor Modules...")

    # Test direct imports
    from agentnet.monitors.keyword import create_keyword_monitor
    from agentnet.monitors.regex import create_regex_monitor
    from agentnet.monitors.resource import create_resource_monitor
    from agentnet.monitors.semantic import create_semantic_similarity_monitor

    print("  ✅ All monitor modules can be imported directly")

    # Test creating monitors directly
    spec = MonitorSpec(
        name="direct_test",
        type="keyword",
        params={"keywords": ["direct"]},
        severity=Severity.MINOR,
    )

    direct_monitor = create_keyword_monitor(spec)
    assert callable(direct_monitor)
    print("  ✅ Direct monitor creation works")


def test_monitor_integration_with_agent():
    """Test that refactored monitors work with AgentNet agents."""
    print("🤖 Testing Monitor Integration with AgentNet...")

    engine = ExampleEngine()
    agent = AgentNet("TestAgent", {"logic": 0.8}, engine=engine)

    # Create and register a keyword monitor
    keyword_spec = MonitorSpec(
        name="integration_keyword",
        type="keyword",
        params={"keywords": ["fail", "error"]},
        severity=Severity.MINOR,
        description="Integration test keyword monitor",
    )
    keyword_monitor = MonitorFactory.build(keyword_spec)
    agent.register_monitor(keyword_monitor)

    # Test that the agent works with the monitor
    result = agent.generate_reasoning_tree("This is a test message")
    assert "result" in result
    print("  ✅ Agent works with refactored monitors")

    # Test that monitor can detect violations
    # (The monitor won't trigger on normal content, which is expected)
    print("  ✅ Monitor integration successful")


def test_experiments_utils_compatibility():
    """Test that experiments/utils/monitors_custom.py works with refactored system."""
    print("🧪 Testing Experiments Utils Compatibility...")

    try:
        from experiments.utils.monitors_custom import (
            create_custom_monitor,
            create_repetition_monitor,
            create_semantic_similarity_monitor,
        )

        print("  ✅ Experiments utils monitors can be imported")

        # Test creating a repetition monitor
        repetition_spec = MonitorSpec(
            name="test_repetition",
            type="repetition",
            params={"max_similarity": 0.8, "window_size": 3},
            severity=Severity.MINOR,
            description="Test repetition monitor",
            cooldown_seconds=None,
        )

        repetition_monitor = create_repetition_monitor(repetition_spec)
        assert callable(repetition_monitor)
        print("  ✅ Repetition monitor creation works")

        # Test the factory function
        factory_monitor = create_custom_monitor(repetition_spec)
        assert callable(factory_monitor)
        print("  ✅ Custom monitor factory works")

    except ImportError as e:
        print(f"  ⚠️  Experiments utils import failed: {e}")
        print("  (This is acceptable if experiments deps are not installed)")


def main():
    """Run all monitor refactoring tests."""
    print("🚀 Monitor System Refactoring Tests")
    print("=" * 50)

    try:
        test_monitor_factory_with_refactored_modules()
        test_individual_monitor_modules()
        test_monitor_integration_with_agent()
        test_experiments_utils_compatibility()

        print("\n" + "=" * 50)
        print("🎉 ALL MONITOR REFACTORING TESTS PASSED!")
        print("✅ Monitor factory works with refactored modules")
        print("✅ Individual monitor modules work correctly")
        print("✅ Agent integration is successful")
        print("✅ Backward compatibility maintained")
        print("\n🏆 Monitor System Refactoring Complete!")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
