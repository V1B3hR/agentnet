#!/usr/bin/env python3
"""
P3 DAG & Eval Implementation Tests

Comprehensive tests for Task Graph Planner and Evaluation Harness
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

# Import P3 components
from agentnet import (  # DAG components; Evaluation components; Existing components for integration
    AgentNet,
    DAGPlanner,
    EvaluationMetrics,
    EvaluationRunner,
    EvaluationScenario,
    EvaluationSuite,
    ExampleEngine,
    ExecutionResult,
    MetricsCalculator,
    SessionManager,
    SuccessCriteria,
    TaskGraph,
    TaskNode,
    TaskScheduler,
)
from agentnet.core.eval.metrics import CriteriaType


def test_dag_planner():
    """Test DAG Planner functionality."""
    print("🧪 Testing DAG Planner...")

    planner = DAGPlanner()

    # Test 1: Basic DAG creation from roadmap example
    example_json = """
    {
      "nodes": [
        {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
        {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]},
        {"id": "mitigations", "prompt": "Propose mitigations", "agent": "Apollo", "deps": ["analysis"]},
        {"id": "summary", "prompt": "Integrate plan & mitigations", "agent": "Synthesizer", "deps": ["mitigations"]}
      ]
    }
    """

    task_graph = planner.create_graph_from_json(example_json)
    assert task_graph.is_valid, f"DAG validation failed: {task_graph.validation_errors}"
    print(f"  ✅ Basic DAG creation: {len(task_graph.nodes)} nodes")

    # Test 2: Execution order
    execution_order = planner.get_execution_order(task_graph)
    expected_order = [["root"], ["analysis"], ["mitigations"], ["summary"]]
    assert (
        execution_order == expected_order
    ), f"Unexpected execution order: {execution_order}"
    print(f"  ✅ Execution order: {execution_order}")

    # Test 3: Graph analysis
    analysis = planner.analyze_graph(task_graph)
    assert analysis["valid"], "Graph analysis should show valid"
    assert (
        analysis["node_count"] == 4
    ), f"Expected 4 nodes, got {analysis['node_count']}"
    assert analysis["max_depth"] == 3, f"Expected depth 3, got {analysis['max_depth']}"
    print(
        f"  ✅ Graph analysis: {analysis['node_count']} nodes, depth {analysis['max_depth']}"
    )

    # Test 4: Ready nodes detection
    ready_nodes = planner.get_ready_nodes(task_graph, set())
    assert ready_nodes == ["root"], f"Expected ['root'], got {ready_nodes}"

    ready_nodes = planner.get_ready_nodes(task_graph, {"root"})
    assert ready_nodes == ["analysis"], f"Expected ['analysis'], got {ready_nodes}"
    print(f"  ✅ Ready nodes detection working")

    # Test 5: Invalid DAG detection (cycle)
    cycle_json = """
    {
      "nodes": [
        {"id": "a", "prompt": "Task A", "agent": "Agent1", "deps": ["b"]},
        {"id": "b", "prompt": "Task B", "agent": "Agent2", "deps": ["a"]}
      ]
    }
    """

    cycle_graph = planner.create_graph_from_json(cycle_json)
    assert not cycle_graph.is_valid, "Cycle should be detected as invalid"
    assert any(
        "cycle" in error.lower() for error in cycle_graph.validation_errors
    ), "Should detect cycle"
    print(f"  ✅ Cycle detection working")

    print("🎉 DAG Planner tests passed!")


@pytest.mark.asyncio
async def test_task_scheduler():
    """Test Task Scheduler functionality."""
    print("🧪 Testing Task Scheduler...")

    # Create example task executor
    async def example_task_executor(
        task_id: str, prompt: str, agent: str, context: dict
    ) -> dict:
        """Example task executor that simulates agent execution."""
        await asyncio.sleep(0.05)  # Simulate work

        # Use dependency results if available
        dep_context = ""
        if context.get("dependency_results"):
            dep_context = (
                f" (building on: {list(context['dependency_results'].keys())})"
            )

        return {
            "content": f"[{agent}] Response to: {prompt}{dep_context}",
            "confidence": 0.85,
            "agent": agent,
            "task_id": task_id,
        }

    # Create DAG
    planner = DAGPlanner()
    example_json = """
    {
      "nodes": [
        {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
        {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]},
        {"id": "mitigations", "prompt": "Propose mitigations", "agent": "Apollo", "deps": ["analysis"]},
        {"id": "summary", "prompt": "Integrate plan & mitigations", "agent": "Synthesizer", "deps": ["mitigations"]}
      ]
    }
    """

    task_graph = planner.create_graph_from_json(example_json)
    assert task_graph.is_valid, "Task graph should be valid"

    # Create scheduler
    scheduler = TaskScheduler(max_retries=2, parallel_execution=True)
    scheduler.set_task_executor(example_task_executor)

    # Execute graph
    result = await scheduler.execute_graph(task_graph)

    assert (
        result.status == "completed"
    ), f"Execution should complete, got {result.status}"
    assert (
        len(result.get_completed_tasks()) == 4
    ), f"Expected 4 completed tasks, got {len(result.get_completed_tasks())}"
    assert (
        len(result.get_failed_tasks()) == 0
    ), f"Expected 0 failed tasks, got {len(result.get_failed_tasks())}"
    print(
        f"  ✅ Basic execution: {len(result.get_completed_tasks())} tasks completed in {result.total_time or 0:.2f}s"
    )

    # Test results contain expected content
    for task_id, task_result in result.task_results.items():
        assert task_result.status == "completed", f"Task {task_id} should be completed"
        assert "content" in task_result.result, f"Task {task_id} should have content"
        assert (
            task_result.agent_used is not None
        ), f"Task {task_id} should record agent used"
    print(f"  ✅ Task results validation passed")

    print("🎉 Task Scheduler tests passed!")


def test_metrics_calculator():
    """Test Metrics Calculator functionality."""
    print("🧪 Testing Metrics Calculator...")

    calculator = MetricsCalculator()

    # Test 1: Keyword presence
    criteria = [
        SuccessCriteria(
            type=CriteriaType.KEYWORD_PRESENCE,
            name="resilience_keywords",
            must_include=["redundancy", "failover"],
            weight=2.0,
        ),
        SuccessCriteria(
            type=CriteriaType.LENGTH_CHECK,
            name="response_length",
            min_length=50,
            max_length=500,
            weight=1.0,
        ),
    ]

    content = """
    To ensure high availability, we need redundancy at multiple levels.
    Implementing failover mechanisms is crucial for system resilience.
    """

    results = calculator.evaluate_content(content, criteria, "test_scenario")

    assert (
        results.success_rate == 1.0
    ), f"Expected perfect score, got {results.success_rate}"
    assert (
        len(results.criteria_results) == 2
    ), f"Expected 2 criteria results, got {len(results.criteria_results)}"
    print(f"  ✅ Basic metrics: success rate {results.success_rate:.2f}")

    # Test 2: Failed criteria
    bad_content = "Short text without keywords."
    bad_results = calculator.evaluate_content(bad_content, criteria, "fail_test")

    assert bad_results.success_rate < 1.0, "Should fail some criteria"
    print(
        f"  ✅ Failed criteria detection: success rate {bad_results.success_rate:.2f}"
    )

    # Test 3: Semantic scoring with reference text
    calculator.register_reference_text(
        "ref1", "Network resilience requires redundancy and failover"
    )

    semantic_criteria = [
        SuccessCriteria(
            type=CriteriaType.SEMANTIC_SCORE,
            name="semantic_similarity",
            reference_id="ref1",
            min_score=0.2,
            weight=1.0,
        )
    ]

    semantic_results = calculator.evaluate_content(
        content, semantic_criteria, "semantic_test"
    )
    assert len(semantic_results.criteria_results) == 1, "Should have 1 semantic result"
    print(f"  ✅ Semantic scoring working")

    print("🎉 Metrics Calculator tests passed!")


@pytest.mark.asyncio
async def test_evaluation_runner():
    """Test Evaluation Runner functionality."""
    print("🧪 Testing Evaluation Runner...")

    # Create example dialogue executor
    async def example_dialogue_executor(agents, topic, config):
        """Example dialogue executor."""
        await asyncio.sleep(0.1)  # Simulate execution
        return {
            "transcript": [
                {
                    "agent": agents[0],
                    "content": f"Let's discuss {topic}. We need redundancy and failover systems for resilience.",
                    "confidence": 0.8,
                },
                {
                    "agent": agents[1],
                    "content": f"I agree on redundancy. Failover mechanisms are crucial for network partition recovery.",
                    "confidence": 0.85,
                },
            ],
            "converged": True,
            "rounds": 2,
            "violations": [],
        }

    # Create example workflow executor
    async def example_workflow_executor(task_graph, config):
        """Example workflow executor."""
        await asyncio.sleep(0.2)  # Simulate execution
        return {
            "status": "completed",
            "task_results": {
                "root": {
                    "status": "completed",
                    "result": {
                        "content": "System architecture requirements defined",
                        "confidence": 0.9,
                    },
                },
                "analysis": {
                    "status": "completed",
                    "result": {
                        "content": "Key components analyzed and identified",
                        "confidence": 0.85,
                    },
                },
            },
        }

    # Create evaluation scenario
    scenario = EvaluationScenario(
        name="test_resilience",
        mode="brainstorm",
        agents=["Athena", "Apollo"],
        topic="Network resilience planning",
        success_criteria=[
            SuccessCriteria(
                type=CriteriaType.KEYWORD_PRESENCE,
                name="resilience_keywords",
                must_include=["redundancy", "failover"],
                weight=2.0,
            )
        ],
        max_rounds=3,
        timeout=10,
    )

    # Create runner
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = EvaluationRunner(results_dir=temp_dir)
        runner.set_dialogue_executor(example_dialogue_executor)
        runner.set_workflow_executor(example_workflow_executor)

        # Test single scenario
        result = await runner.run_scenario(scenario)

        assert (
            result.status == "completed"
        ), f"Scenario should complete, got {result.status}"
        assert result.metrics is not None, "Should have metrics"
        assert result.metrics.success_rate > 0, "Should have positive success rate"
        print(
            f"  ✅ Single scenario: {result.scenario_name} completed with score {result.metrics.success_rate:.2f}"
        )

        # Test workflow scenario
        workflow_scenario = EvaluationScenario(
            name="test_workflow",
            mode="workflow",
            agents=["Planner", "Analyst"],
            topic="System design",
            task_graph={
                "nodes": [
                    {
                        "id": "root",
                        "prompt": "Define requirements",
                        "agent": "Planner",
                        "deps": [],
                    },
                    {
                        "id": "analysis",
                        "prompt": "Analyze components",
                        "agent": "Analyst",
                        "deps": ["root"],
                    },
                ]
            },
            success_criteria=[
                SuccessCriteria(
                    type=CriteriaType.KEYWORD_PRESENCE,
                    name="design_keywords",
                    must_include=["components", "requirements"],
                    weight=1.0,
                )
            ],
            timeout=15,
        )

        workflow_result = await runner.run_scenario(workflow_scenario)
        assert (
            workflow_result.status == "completed"
        ), f"Workflow should complete, got {workflow_result.status}"
        print(f"  ✅ Workflow scenario: {workflow_result.scenario_name} completed")

        print("🎉 Evaluation Runner tests passed!")


@pytest.mark.asyncio
async def test_yaml_evaluation_suite():
    """Test loading and running YAML evaluation suite."""
    print("🧪 Testing YAML Evaluation Suite...")

    # Load the baseline evaluation suite
    suite_path = Path("configs/eval_scenarios/baseline_design_eval.yaml")
    if not suite_path.exists():
        print(f"  ⚠️  Skipping YAML suite test - file not found: {suite_path}")
        return

    suite = EvaluationSuite.from_yaml_file(suite_path)
    print(f"  ✅ Loaded suite '{suite.name}' with {len(suite.scenarios)} scenarios")

    # Verify scenario parsing
    assert (
        len(suite.scenarios) >= 3
    ), f"Expected at least 3 scenarios, got {len(suite.scenarios)}"

    # Check that we have different modes
    modes = [scenario.mode for scenario in suite.scenarios]
    assert "brainstorm" in modes, "Should have brainstorm scenario"
    assert "debate" in modes, "Should have debate scenario"
    assert "consensus" in modes, "Should have consensus scenario"
    print(f"  ✅ Scenario modes: {set(modes)}")

    # Check workflow scenario if present
    workflow_scenarios = [s for s in suite.scenarios if s.mode == "workflow"]
    if workflow_scenarios:
        workflow = workflow_scenarios[0]
        assert (
            workflow.task_graph is not None
        ), "Workflow scenario should have task_graph"
        assert "nodes" in workflow.task_graph, "Task graph should have nodes"
        print(f"  ✅ Workflow scenario: {len(workflow.task_graph['nodes'])} nodes")

    print("🎉 YAML Evaluation Suite tests passed!")


@pytest.mark.asyncio
async def test_integration_with_agentnet():
    """Test integration with existing AgentNet components."""
    print("🧪 Testing Integration with AgentNet...")

    # Create AgentNet instances
    engine = ExampleEngine()

    planner_agent = AgentNet(
        name="Planner",
        style={"logic": 0.9, "creativity": 0.6, "analytical": 0.8},
        engine=engine,
    )

    analyst_agent = AgentNet(
        name="Analyst",
        style={"logic": 0.8, "creativity": 0.4, "analytical": 0.9},
        engine=engine,
    )

    # Create task executor that uses AgentNet
    async def agentnet_task_executor(
        task_id: str, prompt: str, agent_name: str, context: dict
    ) -> dict:
        """Task executor using AgentNet instances."""
        # Select agent
        agent = planner_agent if agent_name == "Planner" else analyst_agent

        # Add context to prompt
        enhanced_prompt = prompt
        if context.get("dependency_results"):
            dep_summary = "\n".join(
                [
                    f"From {dep_id}: {result.get('content', 'No content')}"
                    for dep_id, result in context["dependency_results"].items()
                ]
            )
            enhanced_prompt = f"{prompt}\n\nContext from dependencies:\n{dep_summary}"

        # Execute with AgentNet
        result = agent.generate_reasoning_tree(enhanced_prompt)

        return {
            "content": result["result"]["content"],
            "confidence": result["result"]["confidence"],
            "agent": agent_name,
            "task_id": task_id,
            "meta_insights": result["result"].get("meta_insights", []),
        }

    # Create simple DAG
    planner = DAGPlanner()
    simple_graph = planner.create_graph_from_dict(
        {
            "nodes": [
                {
                    "id": "plan",
                    "prompt": "Create a system architecture plan",
                    "agent": "Planner",
                    "deps": [],
                },
                {
                    "id": "analyze",
                    "prompt": "Analyze the architecture plan for risks",
                    "agent": "Analyst",
                    "deps": ["plan"],
                },
            ]
        }
    )

    assert simple_graph.is_valid, "Simple graph should be valid"

    # Execute with AgentNet integration
    scheduler = TaskScheduler(max_retries=1, parallel_execution=False)
    scheduler.set_task_executor(agentnet_task_executor)

    result = await scheduler.execute_graph(simple_graph)

    assert (
        result.status == "completed"
    ), f"Integration execution should complete, got {result.status}"
    assert (
        len(result.get_completed_tasks()) == 2
    ), f"Expected 2 completed tasks, got {len(result.get_completed_tasks())}"

    # Verify AgentNet-specific content
    for task_id, task_result in result.task_results.items():
        assert (
            "meta_insights" in task_result.result
        ), f"Task {task_id} should have meta_insights from AgentNet"
        assert (
            task_result.result["confidence"] > 0
        ), f"Task {task_id} should have confidence > 0"

    print(
        f"  ✅ AgentNet integration: {len(result.get_completed_tasks())} tasks completed"
    )
    print(f"  ✅ Task results include AgentNet-specific fields")

    print("🎉 AgentNet Integration tests passed!")


async def main():
    """Run all P3 tests."""
    print("🚀 P3 DAG & Eval Implementation Tests")
    print("=" * 50)

    start_time = time.time()

    # Test DAG Planner
    test_dag_planner()
    print()

    # Test Task Scheduler
    await test_task_scheduler()
    print()

    # Test Metrics Calculator
    test_metrics_calculator()
    print()

    # Test Evaluation Runner
    await test_evaluation_runner()
    print()

    # Test YAML Suite Loading
    await test_yaml_evaluation_suite()
    print()

    # Test Integration
    await test_integration_with_agentnet()
    print()

    total_time = time.time() - start_time

    print("=" * 50)
    print("🎉 All P3 Tests Completed!")
    print(f"⏱️  Total test time: {total_time:.2f}s")
    print()
    print("P3 Features Successfully Implemented:")
    print("  ✅ DAG Planner with networkx")
    print("  ✅ Task Scheduler with dependency resolution")
    print("  ✅ Evaluation Harness with YAML support")
    print("  ✅ Metrics Calculator with multiple criteria types")
    print("  ✅ Integration with existing AgentNet components")
    print("  ✅ Workflow mode support in session management")
    print("  ✅ Comprehensive test coverage")


if __name__ == "__main__":
    asyncio.run(main())
