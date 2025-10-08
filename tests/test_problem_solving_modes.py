"""
Tests for problem-solving modes, styles, and techniques integration.
"""

import pytest
from agentnet import AgentNet, ExampleEngine
from agentnet.core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from agentnet.core.autoconfig import AutoConfig, TaskDifficulty
from agentnet.metrics.flow import calculate_flow_metrics, FlowMetrics
from agentnet.orchestrator.modes import (
    BrainstormStrategy,
    DebateStrategy,
    ConsensusStrategy,
    WorkflowStrategy,
    DialogueStrategy,
)


class TestCoreEnums:
    """Test the core enums for modes, styles, and techniques."""

    def test_mode_enum_values(self):
        """Test that Mode enum has correct values."""
        assert Mode.BRAINSTORM.value == "brainstorm"
        assert Mode.DEBATE.value == "debate"
        assert Mode.CONSENSUS.value == "consensus"
        assert Mode.WORKFLOW.value == "workflow"
        assert Mode.DIALOGUE.value == "dialogue"

        # Test that all expected modes are present
        expected_modes = {"brainstorm", "debate", "consensus", "workflow", "dialogue"}
        actual_modes = {mode.value for mode in Mode}
        assert actual_modes == expected_modes

    def test_problem_solving_style_enum_values(self):
        """Test that ProblemSolvingStyle enum has correct values."""
        assert ProblemSolvingStyle.CLARIFIER.value == "clarifier"
        assert ProblemSolvingStyle.IDEATOR.value == "ideator"
        assert ProblemSolvingStyle.DEVELOPER.value == "developer"
        assert ProblemSolvingStyle.IMPLEMENTOR.value == "implementor"

        # Test that all expected styles are present
        expected_styles = {"clarifier", "ideator", "developer", "implementor"}
        actual_styles = {style.value for style in ProblemSolvingStyle}
        assert actual_styles == expected_styles

    def test_problem_technique_enum_values(self):
        """Test that ProblemTechnique enum has correct values."""
        assert ProblemTechnique.SIMPLE.value == "simple"
        assert ProblemTechnique.COMPLEX.value == "complex"
        assert ProblemTechnique.TROUBLESHOOTING.value == "troubleshooting"
        assert ProblemTechnique.GAP_FROM_STANDARD.value == "gap_from_standard"
        assert ProblemTechnique.TARGET_STATE.value == "target_state"
        assert ProblemTechnique.OPEN_ENDED.value == "open_ended"

        # Test that all expected techniques are present
        expected_techniques = {
            "simple",
            "complex",
            "troubleshooting",
            "gap_from_standard",
            "target_state",
            "open_ended",
        }
        actual_techniques = {technique.value for technique in ProblemTechnique}
        assert actual_techniques == expected_techniques


class TestFlowMetrics:
    """Test flow metrics calculation."""

    def test_basic_flow_metrics_calculation(self):
        """Test basic flow metrics calculation."""
        reasoning_tree = {
            "runtime": 2.0,
            "cost_record": {"tokens_output": 100},
            "metadata": {},
            "violations": [],
            "tool_calls": [],
            "nodes": [{"confidence": 0.8}],
        }

        metrics = calculate_flow_metrics(reasoning_tree)

        assert isinstance(metrics, FlowMetrics)
        assert metrics.current == 50.0  # 100 tokens / 2.0 seconds
        assert metrics.voltage == 5.0  # Default voltage
        assert metrics.resistance >= 0.1  # Minimum resistance
        assert metrics.power == metrics.voltage * metrics.current
        assert metrics.runtime_seconds == 2.0
        assert metrics.tokens_output == 100

    def test_flow_metrics_with_metadata_voltage(self):
        """Test flow metrics with explicit voltage in metadata."""
        reasoning_tree = {
            "runtime": 1.0,
            "cost_record": {"tokens_output": 50},
            "metadata": {"voltage": 7.5},
            "violations": [],
            "tool_calls": [],
            "nodes": [],
        }

        metrics = calculate_flow_metrics(reasoning_tree)

        assert metrics.voltage == 7.5
        assert metrics.metadata_voltage == 7.5

    def test_flow_metrics_with_technique_voltage(self):
        """Test flow metrics with technique-based voltage."""
        reasoning_tree = {
            "runtime": 1.0,
            "cost_record": {"tokens_output": 50},
            "metadata": {},
            "violations": [],
            "tool_calls": [],
            "nodes": [],
        }

        metrics = calculate_flow_metrics(
            reasoning_tree, technique=ProblemTechnique.COMPLEX
        )

        assert metrics.voltage == 8.0  # Complex technique voltage
        assert metrics.technique_voltage == 8.0

    def test_flow_metrics_with_difficulty_voltage(self):
        """Test flow metrics with difficulty-based voltage."""
        reasoning_tree = {
            "runtime": 1.0,
            "cost_record": {"tokens_output": 50},
            "metadata": {},
            "violations": [],
            "tool_calls": [],
            "nodes": [],
        }

        metrics = calculate_flow_metrics(reasoning_tree, difficulty=TaskDifficulty.HARD)

        assert metrics.voltage == 8.0  # Hard difficulty voltage
        assert metrics.difficulty_voltage == 8.0

    def test_flow_metrics_voltage_clamping(self):
        """Test that voltage is clamped to 0-10 range."""
        reasoning_tree = {
            "runtime": 1.0,
            "cost_record": {"tokens_output": 50},
            "metadata": {"voltage": 15.0},  # Out of range
            "violations": [],
            "tool_calls": [],
            "nodes": [],
        }

        metrics = calculate_flow_metrics(reasoning_tree)

        assert metrics.voltage == 10.0  # Clamped to maximum

        # Test negative voltage clamping
        reasoning_tree["metadata"]["voltage"] = -5.0
        metrics = calculate_flow_metrics(reasoning_tree)

        assert metrics.voltage == 0.0  # Clamped to minimum


class TestAutoConfigIntegration:
    """Test AutoConfig integration with modes, styles, and techniques."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autoconfig = AutoConfig()

    def test_mode_recommendations(self):
        """Test mode recommendations based on task content."""
        # Test brainstorm task
        brainstorm_task = "Generate creative ideas for new product features"
        params = self.autoconfig.configure_scenario(brainstorm_task)
        assert params.recommended_mode == Mode.BRAINSTORM

        # Test debate task
        debate_task = "Analyze and debate the pros and cons of microservices"
        params = self.autoconfig.configure_scenario(debate_task)
        assert params.recommended_mode == Mode.DEBATE

        # Test consensus task
        consensus_task = "Find common ground and agree on team objectives"
        params = self.autoconfig.configure_scenario(consensus_task)
        assert params.recommended_mode == Mode.CONSENSUS

        # Test workflow task
        workflow_task = "Implement a step-by-step deployment process"
        params = self.autoconfig.configure_scenario(workflow_task)
        assert params.recommended_mode == Mode.WORKFLOW

        # Test dialogue task
        dialogue_task = "Discuss and explore user requirements"
        params = self.autoconfig.configure_scenario(dialogue_task)
        assert params.recommended_mode == Mode.DIALOGUE

    def test_style_recommendations(self):
        """Test style recommendations based on task content."""
        # Test clarifier task
        clarifier_task = "Clarify the requirements and define the scope"
        params = self.autoconfig.configure_scenario(clarifier_task)
        assert params.recommended_style == ProblemSolvingStyle.CLARIFIER

        # Test ideator task
        ideator_task = "Generate innovative concepts for the platform"
        params = self.autoconfig.configure_scenario(ideator_task)
        assert params.recommended_style == ProblemSolvingStyle.IDEATOR

        # Test developer task
        developer_task = "Design and build a scalable architecture"
        params = self.autoconfig.configure_scenario(developer_task)
        assert params.recommended_style == ProblemSolvingStyle.DEVELOPER

        # Test implementor task
        implementor_task = "Execute and deploy the new service"
        params = self.autoconfig.configure_scenario(implementor_task)
        assert params.recommended_style == ProblemSolvingStyle.IMPLEMENTOR

    def test_technique_recommendations(self):
        """Test technique recommendations based on task content."""
        # Test troubleshooting task
        troubleshooting_task = "Fix the broken authentication system"
        params = self.autoconfig.configure_scenario(troubleshooting_task)
        assert params.recommended_technique == ProblemTechnique.TROUBLESHOOTING

        # Test gap analysis task
        gap_task = "Identify gaps from security compliance standards"
        params = self.autoconfig.configure_scenario(gap_task)
        assert params.recommended_technique == ProblemTechnique.GAP_FROM_STANDARD

        # Test target state task
        target_task = "Achieve the goal of 99.9% uptime"
        params = self.autoconfig.configure_scenario(target_task)
        assert params.recommended_technique == ProblemTechnique.TARGET_STATE

        # Test open-ended task
        open_task = "Explore possibilities for improving user experience"
        params = self.autoconfig.configure_scenario(open_task)
        assert params.recommended_technique == ProblemTechnique.OPEN_ENDED

    def test_autoconfig_data_injection(self):
        """Test that autoconfig data includes new fields."""
        task = "Generate creative solutions for complex problems"
        params = self.autoconfig.configure_scenario(task)

        session_data = {}
        self.autoconfig.inject_autoconfig_data(session_data, params)

        autoconfig_data = session_data["autoconfig"]
        assert "recommended_mode" in autoconfig_data
        assert "recommended_style" in autoconfig_data
        assert "recommended_technique" in autoconfig_data

        # Verify values are properly serialized
        if params.recommended_mode:
            assert autoconfig_data["recommended_mode"] == params.recommended_mode.value
        if params.recommended_style:
            assert (
                autoconfig_data["recommended_style"] == params.recommended_style.value
            )
        if params.recommended_technique:
            assert (
                autoconfig_data["recommended_technique"]
                == params.recommended_technique.value
            )


class TestStrategyClasses:
    """Test the strategy classes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ExampleEngine()
        self.agent = AgentNet("TestAgent", {"logic": 0.8}, engine=self.engine)

    def test_brainstorm_strategy_creation(self):
        """Test BrainstormStrategy creation and configuration."""
        strategy = BrainstormStrategy(
            style=ProblemSolvingStyle.IDEATOR, technique=ProblemTechnique.OPEN_ENDED
        )

        assert strategy.mode == Mode.BRAINSTORM
        assert strategy.style == ProblemSolvingStyle.IDEATOR
        assert strategy.technique == ProblemTechnique.OPEN_ENDED

    def test_debate_strategy_creation(self):
        """Test DebateStrategy creation and configuration."""
        strategy = DebateStrategy(
            style=ProblemSolvingStyle.DEVELOPER, technique=ProblemTechnique.COMPLEX
        )

        assert strategy.mode == Mode.DEBATE
        assert strategy.style == ProblemSolvingStyle.DEVELOPER
        assert strategy.technique == ProblemTechnique.COMPLEX

    def test_strategy_execution(self):
        """Test strategy execution returns proper structure."""
        strategy = BrainstormStrategy(
            style=ProblemSolvingStyle.IDEATOR, technique=ProblemTechnique.OPEN_ENDED
        )

        result = strategy.execute(agent=self.agent, task="Generate ideas for testing")

        # Verify result structure
        assert "result" in result
        assert "strategy" in result
        assert "runtime" in result

        # Verify strategy metadata
        strategy_info = result["strategy"]
        assert strategy_info["mode"] == "brainstorm"
        assert strategy_info["style"] == "ideator"
        assert strategy_info["technique"] == "open_ended"
        assert strategy_info["focus"] == "idea_generation"
        assert "execution_time" in strategy_info

    def test_all_strategy_types(self):
        """Test that all strategy types can be created and executed."""
        strategies = [
            (BrainstormStrategy, "brainstorm", "idea_generation"),
            (DebateStrategy, "debate", "critical_analysis"),
            (ConsensusStrategy, "consensus", "shared_agreement"),
            (WorkflowStrategy, "workflow", "process_execution"),
            (DialogueStrategy, "dialogue", "conversational_exploration"),
        ]

        for StrategyClass, expected_mode, expected_focus in strategies:
            strategy = StrategyClass()
            result = strategy.execute(
                agent=self.agent, task=f"Test {expected_mode} mode"
            )

            assert result["strategy"]["mode"] == expected_mode
            assert result["strategy"]["focus"] == expected_focus

    def test_metadata_preparation(self):
        """Test that strategies properly prepare metadata."""
        strategy = BrainstormStrategy(
            style=ProblemSolvingStyle.IDEATOR, technique=ProblemTechnique.OPEN_ENDED
        )

        # Test with no initial metadata
        metadata = strategy._prepare_metadata()
        assert metadata["mode"] == "brainstorm"
        assert metadata["problem_solving_style"] == "ideator"
        assert metadata["problem_technique"] == "open_ended"

        # Test with existing metadata
        existing_metadata = {"custom_field": "value"}
        metadata = strategy._prepare_metadata(metadata=existing_metadata)
        assert metadata["custom_field"] == "value"
        assert metadata["mode"] == "brainstorm"


if __name__ == "__main__":
    pytest.main([__file__])
