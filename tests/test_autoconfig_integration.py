"""
Integration tests for AutoConfig with AgentNet core functionality.

Tests the full integration of AutoConfig with generate_reasoning_tree methods
and verifies observability data injection.
"""

import pytest

try:
    from agentnet import AgentNet, ExampleEngine
    from agentnet.core.autoconfig import get_global_autoconfig

    _modular_available = True
except ImportError:
    _modular_available = False


@pytest.mark.skipif(not _modular_available, reason="Modular AgentNet not available")
class TestAutoConfigCoreIntegration:
    """Test AutoConfig integration with core AgentNet functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ExampleEngine()
        self.agent = AgentNet(
            name="TestAgent",
            style={"logic": 0.8, "creativity": 0.3},
            engine=self.engine,
        )

    def test_reasoning_tree_with_autoconfig_enabled(self):
        """Test generate_reasoning_tree with AutoConfig enabled."""
        simple_task = "What is Python?"

        # Test with auto_config enabled (default)
        result = self.agent.generate_reasoning_tree(
            task=simple_task, metadata={"auto_config": True}
        )

        # Should have autoconfig data injected
        assert "autoconfig" in result
        autoconfig_data = result["autoconfig"]

        assert autoconfig_data["difficulty"] == "simple"
        assert autoconfig_data["configured_rounds"] == 3
        assert (
            autoconfig_data["configured_max_depth"] == 3
        )  # max(3 default, 2 autoconfig) = 3
        assert (
            autoconfig_data["configured_confidence_threshold"] == 0.6
        )  # auto-configured for simple task
        assert autoconfig_data["enabled"] is True
        assert "reasoning" in autoconfig_data

    def test_reasoning_tree_with_autoconfig_disabled(self):
        """Test generate_reasoning_tree with AutoConfig disabled."""
        task = "What is Python?"

        # Test with auto_config disabled
        result = self.agent.generate_reasoning_tree(
            task=task, metadata={"auto_config": False}
        )

        # Should NOT have autoconfig data injected
        assert "autoconfig" not in result

    def test_reasoning_tree_hard_task_configuration(self):
        """Test generate_reasoning_tree with hard task auto-configuration."""
        hard_task = (
            "Develop a comprehensive framework for ethical AI decision-making "
            "that considers multiple stakeholder perspectives, addresses bias mitigation, "
            "and evaluates long-term societal implications"
        )

        result = self.agent.generate_reasoning_tree(
            task=hard_task,
            confidence_threshold=0.6,  # Lower initial threshold
            max_depth=2,  # Lower initial depth
        )

        # Should have autoconfig data for hard task
        assert "autoconfig" in result
        autoconfig_data = result["autoconfig"]

        assert autoconfig_data["difficulty"] == "hard"
        assert autoconfig_data["configured_rounds"] == 5
        assert autoconfig_data["configured_max_depth"] == 4  # Should be raised from 2
        # Confidence should be raised from 0.6 to at least 0.7 (0.6 + 0.1)
        assert autoconfig_data["configured_confidence_threshold"] >= 0.7

    def test_reasoning_tree_enhanced_with_autoconfig(self):
        """Test generate_reasoning_tree_enhanced with AutoConfig."""
        medium_task = "Explain the benefits of microservices architecture"

        result = self.agent.generate_reasoning_tree_enhanced(
            task=medium_task, metadata={"auto_config": True}
        )

        # Should have autoconfig data injected
        assert "autoconfig" in result
        autoconfig_data = result["autoconfig"]

        assert autoconfig_data["difficulty"] == "medium"
        assert autoconfig_data["configured_rounds"] == 4
        assert autoconfig_data["configured_max_depth"] == 3
        assert autoconfig_data["configured_confidence_threshold"] == 0.7

    def test_confidence_threshold_preservation(self):
        """Test that confidence thresholds are preserved/raised, never lowered."""
        task = "Simple task"
        high_confidence = 0.95

        result = self.agent.generate_reasoning_tree(
            task=task,
            confidence_threshold=high_confidence,
            metadata={"auto_config": True},
        )

        # Should preserve the high confidence threshold
        autoconfig_data = result["autoconfig"]
        # For simple task, auto-config would suggest 0.6, but we should preserve 0.95
        assert autoconfig_data["configured_confidence_threshold"] >= high_confidence

    def test_autoconfig_reasoning_explanation(self):
        """Test that reasoning explanations are properly generated."""
        tasks = [
            ("What is AI?", "SIMPLE", "3 rounds", "depth 2", "confidence 0.6"),
            (
                "Explain machine learning algorithms",
                "MEDIUM",
                "4 rounds",
                "depth 3",
                "confidence 0.7",
            ),
            (
                "Develop comprehensive AI governance framework",
                "HARD",
                "5 rounds",
                "depth 4",
                "confidence 0.8",
            ),
        ]

        for (
            task,
            expected_difficulty,
            expected_rounds,
            expected_depth,
            expected_confidence,
        ) in tasks:
            result = self.agent.generate_reasoning_tree(task=task)

            autoconfig_data = result["autoconfig"]
            reasoning = autoconfig_data["reasoning"]

            assert expected_difficulty in reasoning
            assert expected_rounds in reasoning
            assert expected_depth in reasoning
            assert expected_confidence in reasoning

    def test_autoconfig_with_custom_metadata(self):
        """Test AutoConfig with custom metadata and context."""
        task = "Implement a solution"

        # With technical domain context
        result = self.agent.generate_reasoning_tree(
            task=task,
            metadata={
                "auto_config": True,
                "domain": "technical research",
                "custom_field": "test_value",
            },
        )

        # Should have autoconfig data and preserve custom metadata
        assert "autoconfig" in result
        assert result["metadata"]["custom_field"] == "test_value"

        # Technical domain should influence difficulty
        autoconfig_data = result["autoconfig"]
        # Should be at least medium due to domain boost
        assert autoconfig_data["difficulty"] in ["medium", "hard"]


@pytest.mark.skipif(not _modular_available, reason="Modular AgentNet not available")
class TestAutoConfigGlobalInstance:
    """Test global AutoConfig instance behavior."""

    def test_global_autoconfig_consistency(self):
        """Test that global AutoConfig instance is used consistently."""
        autoconfig1 = get_global_autoconfig()
        autoconfig2 = get_global_autoconfig()

        # Should be the same instance
        assert autoconfig1 is autoconfig2

        # Should provide consistent results
        task = "Analyze complex system architecture"
        result1 = autoconfig1.analyze_task_difficulty(task)
        result2 = autoconfig2.analyze_task_difficulty(task)

        assert result1 == result2

    def test_autoconfig_without_engine(self):
        """Test AutoConfig works even without inference engine."""
        agent_no_engine = AgentNet(
            name="NoEngineAgent",
            style={"logic": 0.5},
            # No engine provided
        )

        task = "What is machine learning?"
        result = agent_no_engine.generate_reasoning_tree(task=task)

        # Should still have autoconfig data even without engine
        assert "autoconfig" in result
        autoconfig_data = result["autoconfig"]
        assert autoconfig_data["enabled"] is True
        assert autoconfig_data["difficulty"] == "simple"


def test_autoconfig_import():
    """Test that AutoConfig can be imported and used standalone."""
    from agentnet.core.autoconfig import AutoConfig, TaskDifficulty

    autoconfig = AutoConfig()
    difficulty = autoconfig.analyze_task_difficulty("Test task")
    assert isinstance(difficulty, TaskDifficulty)

    params = autoconfig.configure_scenario("Test task")
    assert hasattr(params, "rounds")
    assert hasattr(params, "max_depth")
    assert hasattr(params, "confidence_threshold")


if __name__ == "__main__":
    pytest.main([__file__])
