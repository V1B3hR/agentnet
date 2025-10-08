"""
Tests for AutoConfig functionality.

Tests task difficulty analysis, parameter configuration, observability integration,
confidence threshold preservation, and backward compatibility.
"""

import pytest
from agentnet.core.autoconfig import (
    AutoConfig,
    AutoConfigParams,
    TaskDifficulty,
    get_global_autoconfig,
    set_global_autoconfig,
)


class TestTaskDifficultyAnalysis:
    """Test task difficulty analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autoconfig = AutoConfig()

    def test_simple_task_detection(self):
        """Test detection of simple tasks."""
        simple_tasks = [
            "What is Python?",
            "List the benefits",
            "Define machine learning",
            "Name three colors",
            "Choose yes or no",
            "Select the correct answer",
        ]

        for task in simple_tasks:
            difficulty = self.autoconfig.analyze_task_difficulty(task)
            assert (
                difficulty == TaskDifficulty.SIMPLE
            ), f"Task '{task}' should be SIMPLE"

    def test_medium_task_detection(self):
        """Test detection of medium complexity tasks."""
        medium_tasks = [
            "Explain the benefits of using microservices architecture",
            "Compare and contrast SQL and NoSQL databases",
            "Describe the process of implementing CI/CD pipeline",
            "Outline a plan for team collaboration",
            "Summarize the main approaches to data modeling",
        ]

        for task in medium_tasks:
            difficulty = self.autoconfig.analyze_task_difficulty(task)
            assert (
                difficulty == TaskDifficulty.MEDIUM
            ), f"Task '{task}' should be MEDIUM"

    def test_hard_task_detection(self):
        """Test detection of hard/complex tasks."""
        hard_tasks = [
            "Develop a comprehensive framework for ethical AI decision-making that considers multiple stakeholder perspectives and addresses bias mitigation",
            "Analyze the complex relationship between distributed system architecture and performance optimization while evaluating trade-offs",
            "Create an in-depth analysis of policy implications for climate change mitigation considering economic, social, and technical factors",
            "Synthesize research findings to develop a sophisticated methodology for risk assessment in multi-criteria decision making",
        ]

        for task in hard_tasks:
            difficulty = self.autoconfig.analyze_task_difficulty(task)
            assert difficulty == TaskDifficulty.HARD, f"Task '{task}' should be HARD"

    def test_empty_task_handling(self):
        """Test handling of empty or invalid tasks."""
        assert self.autoconfig.analyze_task_difficulty("") == TaskDifficulty.SIMPLE
        assert self.autoconfig.analyze_task_difficulty("   ") == TaskDifficulty.SIMPLE

    def test_context_based_difficulty_adjustment(self):
        """Test context-based difficulty adjustments."""
        task = "Implement a solution"

        # Low confidence context suggests higher difficulty
        context = {"confidence": 0.3}
        difficulty = self.autoconfig.analyze_task_difficulty(task, context)

        # Without context, this might be medium, but low confidence should push it higher
        no_context_difficulty = self.autoconfig.analyze_task_difficulty(task)

        # Domain-specific context should increase difficulty
        domain_context = {"domain": "technical research"}
        domain_difficulty = self.autoconfig.analyze_task_difficulty(
            task, domain_context
        )

        # At least one should be elevated due to context
        assert (
            difficulty != TaskDifficulty.SIMPLE
            or domain_difficulty != TaskDifficulty.SIMPLE
        )


class TestParameterConfiguration:
    """Test parameter configuration based on difficulty."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autoconfig = AutoConfig()

    def test_simple_task_configuration(self):
        """Test configuration for simple tasks."""
        params = self.autoconfig.configure_scenario("What is AI?")

        assert params.difficulty == TaskDifficulty.SIMPLE
        assert params.rounds == 3
        assert params.max_depth == 2
        assert params.confidence_threshold == 0.6
        assert params.confidence_adjustment == -0.1

    def test_medium_task_configuration(self):
        """Test configuration for medium tasks."""
        params = self.autoconfig.configure_scenario(
            "Explain machine learning algorithms"
        )

        assert params.difficulty == TaskDifficulty.MEDIUM
        assert params.rounds == 4
        assert params.max_depth == 3
        assert params.confidence_threshold == 0.7
        assert params.confidence_adjustment == 0.0

    def test_hard_task_configuration(self):
        """Test configuration for hard tasks."""
        params = self.autoconfig.configure_scenario(
            "Develop a comprehensive framework for ethical AI governance considering stakeholder implications"
        )

        assert params.difficulty == TaskDifficulty.HARD
        assert params.rounds == 5
        assert params.max_depth == 4
        assert params.confidence_threshold == 0.8
        assert params.confidence_adjustment == 0.1

    def test_base_parameter_override(self):
        """Test that base parameters are respected when provided."""
        params = self.autoconfig.configure_scenario(
            "Simple task",
            base_rounds=10,
            base_max_depth=5,
            base_confidence_threshold=0.9,
        )

        # Should use higher of base and auto-configured values
        assert params.rounds >= 10  # Should be at least the base
        assert params.max_depth >= 5  # Should be at least the base
        # Confidence should be adjusted from base, not replaced
        assert params.confidence_threshold >= 0.9

    def test_confidence_threshold_adjustment(self):
        """Test confidence threshold adjustment logic."""
        # For hard task with base threshold
        params = self.autoconfig.configure_scenario(
            "Complex analysis of distributed systems", base_confidence_threshold=0.75
        )

        # Should be adjusted upward for hard task
        expected = 0.75 + 0.1  # base + hard task adjustment
        assert abs(params.confidence_threshold - expected) < 0.01


class TestObservabilityIntegration:
    """Test observability and session data integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autoconfig = AutoConfig()

    def test_autoconfig_data_injection(self):
        """Test injection of autoconfig data into session data."""
        params = self.autoconfig.configure_scenario("Complex task requiring analysis")
        session_data = {}

        self.autoconfig.inject_autoconfig_data(session_data, params)

        assert "autoconfig" in session_data
        autoconfig_data = session_data["autoconfig"]

        assert autoconfig_data["difficulty"] == params.difficulty.value
        assert autoconfig_data["configured_rounds"] == params.rounds
        assert autoconfig_data["configured_max_depth"] == params.max_depth
        assert (
            autoconfig_data["configured_confidence_threshold"]
            == params.confidence_threshold
        )
        assert "reasoning" in autoconfig_data
        assert autoconfig_data["enabled"] is True
        assert "confidence_adjustment" in autoconfig_data

    def test_reasoning_generation(self):
        """Test reasoning explanation generation."""
        hard_params = self.autoconfig.configure_scenario("Complex analysis task")
        assert "HARD" in hard_params.reasoning
        assert "5 rounds" in hard_params.reasoning
        assert "depth 4" in hard_params.reasoning
        assert "confidence 0.8" in hard_params.reasoning

        simple_params = self.autoconfig.configure_scenario("What is AI?")
        assert "SIMPLE" in simple_params.reasoning
        assert "3 rounds" in simple_params.reasoning
        assert "depth 2" in simple_params.reasoning


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autoconfig = AutoConfig()

    def test_auto_config_enable_disable(self):
        """Test auto_config enable/disable functionality."""
        # Default should be enabled
        assert self.autoconfig.should_auto_configure() is True
        assert self.autoconfig.should_auto_configure({}) is True

        # Explicitly enabled
        assert self.autoconfig.should_auto_configure({"auto_config": True}) is True

        # Explicitly disabled
        assert self.autoconfig.should_auto_configure({"auto_config": False}) is False

        # None should default to enabled
        assert self.autoconfig.should_auto_configure({"auto_config": None}) is True

    def test_confidence_threshold_preservation(self):
        """Test confidence threshold preservation (never lower original)."""
        config_params = AutoConfigParams(
            rounds=5,
            max_depth=4,
            confidence_threshold=0.8,
            difficulty=TaskDifficulty.HARD,
            reasoning="Test",
            confidence_adjustment=0.1,
        )

        # Should preserve higher original threshold
        preserved = self.autoconfig.preserve_confidence_threshold(0.9, config_params)
        assert preserved == 0.9

        # Should use config threshold when it's higher
        preserved = self.autoconfig.preserve_confidence_threshold(0.6, config_params)
        assert preserved == 0.8

        # Should handle equal thresholds
        preserved = self.autoconfig.preserve_confidence_threshold(0.8, config_params)
        assert preserved == 0.8


class TestGlobalInstance:
    """Test global AutoConfig instance management."""

    def test_global_instance_singleton(self):
        """Test global instance is singleton."""
        instance1 = get_global_autoconfig()
        instance2 = get_global_autoconfig()
        assert instance1 is instance2

    def test_global_instance_replacement(self):
        """Test global instance can be replaced."""
        original = get_global_autoconfig()
        custom = AutoConfig()

        set_global_autoconfig(custom)
        current = get_global_autoconfig()

        assert current is custom
        assert current is not original


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autoconfig = AutoConfig()

    def test_mixed_complexity_indicators(self):
        """Test tasks with mixed complexity indicators."""
        mixed_task = (
            "List the complex algorithms used in advanced machine learning research"
        )

        # Has both simple ("list") and complex ("advanced", "research") indicators
        difficulty = self.autoconfig.analyze_task_difficulty(mixed_task)

        # Should favor complexity indicators
        assert difficulty in [TaskDifficulty.MEDIUM, TaskDifficulty.HARD]

    def test_long_simple_task(self):
        """Test long but simple tasks."""
        long_simple = "What is the name of the capital city of France and what is the name of the famous tower there and what is the name of the river that flows through it?"

        difficulty = self.autoconfig.analyze_task_difficulty(long_simple)

        # Length might push it up, but simple indicators should dominate
        # This tests the balance of indicators
        assert difficulty in [TaskDifficulty.SIMPLE, TaskDifficulty.MEDIUM]

    def test_technical_domain_boost(self):
        """Test technical domain context boost."""
        task = "Implement a basic solution"

        regular_difficulty = self.autoconfig.analyze_task_difficulty(task)
        technical_difficulty = self.autoconfig.analyze_task_difficulty(
            task, {"domain": "technical research"}
        )

        # Technical domain should push difficulty up
        difficulty_levels = {"simple": 0, "medium": 1, "hard": 2}
        assert (
            difficulty_levels[technical_difficulty.value]
            >= difficulty_levels[regular_difficulty.value]
        )


if __name__ == "__main__":
    pytest.main([__file__])
