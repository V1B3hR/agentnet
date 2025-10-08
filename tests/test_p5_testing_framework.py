#!/usr/bin/env python3
"""
Phase 5 Testing Framework Tests

Tests the comprehensive test matrix, integration testing, and regression
testing capabilities implemented for Phase 5.
"""

import asyncio
import tempfile
from pathlib import Path

from agentnet import ExampleEngine
from agentnet.testing import (
    TestMatrix,
    TestConfiguration,
    FeatureSet,
    AgentType,
    IntegrationTestSuite,
    MultiAgentTestCase,
    RegressionTestSuite,
    PerformanceBaseline,
    AgentFixtures,
    ScenarioGenerator,
    AgentProfile,
    TestScenario,
)


def test_test_matrix():
    """Test the test matrix framework."""
    print("🧪 Testing Test Matrix Framework...")

    async def run_test():
        # Create test matrix
        matrix = TestMatrix()

        # Generate standard configurations
        configs = matrix.generate_standard_matrix()
        assert len(configs) > 0
        print(f"  📋 Generated {len(configs)} standard test configurations")

        # Add configurations to matrix
        matrix.add_configurations(configs[:3])  # Test with first 3 for speed

        # Execute matrix (sequential for predictable testing)
        results = await matrix.execute_matrix(parallel_execution=False)

        # Verify results
        assert len(results) == 3
        assert all(r.configuration is not None for r in results)
        assert all(r.duration_seconds >= 0 for r in results)

        # Get summary
        summary = matrix.get_results_summary()
        assert summary["total_tests"] == 3
        assert "success_rate" in summary
        assert "by_feature_set" in summary

        print(f"  ✅ Executed {len(results)} tests")
        print(f"  📊 Success rate: {summary['success_rate']:.1%}")
        print(f"  ⏱️ Average duration: {summary['avg_duration_seconds']:.2f}s")

        return True

    return run_test()


def test_integration_test_suite():
    """Test the integration test suite."""
    print("🔗 Testing Integration Test Suite...")

    async def run_test():
        # Create integration test suite
        suite = IntegrationTestSuite()

        # Create standard test cases
        test_cases = suite.create_standard_test_cases()
        assert len(test_cases) > 0
        print(f"  📝 Created {len(test_cases)} integration test cases")

        # Add test cases to suite
        for test_case in test_cases[:2]:  # Test with first 2 for speed
            suite.add_test_case(test_case)

        # Run tests
        summary = await suite.run_all_tests(
            include_observability=True, parallel_execution=False
        )

        # Verify results
        assert summary["total_tests"] == 2
        assert "success_rate" in summary
        assert "feature_success_rates" in summary

        # Check test results details
        results = suite.get_results()
        assert len(results) == 2
        assert all(r["test_case"] is not None for r in results)

        print(f"  ✅ Executed {summary['total_tests']} integration tests")
        print(f"  📈 Success rate: {summary['success_rate']:.1%}")
        print(f"  🎯 Feature coverage: {len(summary['feature_success_rates'])} phases")

        return True

    return run_test()


def test_regression_test_suite():
    """Test the regression testing suite."""
    print("📉 Testing Regression Test Suite...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regression test suite
        suite = RegressionTestSuite(baseline_dir=temp_dir)

        # Create baseline performance data
        baseline_data = {
            "avg_latency_ms": 150.0,
            "p95_latency_ms": 300.0,
            "throughput_ops_per_sec": 50.0,
            "success_rate": 0.95,
            "token_efficiency_score": 0.75,
            "metadata": {"test_type": "standard"},
        }

        test_config = {
            "agent_count": 1,
            "feature_set": "phase_1",
            "agent_type": "single",
        }

        # Create baseline
        baseline = suite.create_baseline(
            version="v1.0.0",
            performance_data=baseline_data,
            test_configuration=test_config,
        )

        assert baseline.version == "v1.0.0"
        assert baseline.avg_latency_ms == 150.0
        print(f"  📊 Created baseline for {baseline.version}")

        # Test regression detection with no regression
        no_regression = suite.detect_regressions(
            current_version="v1.0.1",
            current_performance={
                "avg_latency_ms": 145.0,  # Slight improvement
                "p95_latency_ms": 295.0,
                "throughput_ops_per_sec": 52.0,
                "success_rate": 0.96,
                "token_efficiency_score": 0.76,
            },
            test_configuration=test_config,
        )

        assert len(no_regression) == 0
        print(f"  ✅ No regressions detected with improved performance")

        # Test regression detection with actual regression
        regressions = suite.detect_regressions(
            current_version="v1.1.0",
            current_performance={
                "avg_latency_ms": 220.0,  # 47% increase - should trigger
                "p95_latency_ms": 450.0,  # 50% increase - should trigger
                "throughput_ops_per_sec": 35.0,  # 30% decrease - should trigger
                "success_rate": 0.88,  # 7% decrease - should trigger
                "token_efficiency_score": 0.60,  # 20% decrease - should trigger
            },
            test_configuration=test_config,
        )

        assert len(regressions) > 0
        print(f"  ⚠️ Detected {len(regressions)} performance regressions")

        # Test report generation
        report = suite.generate_regression_report(regressions, "v1.1.0")
        assert "Performance Regression Report" in report
        assert "v1.1.0" in report
        print(f"  📄 Generated regression report ({len(report)} characters)")

        # Test baseline management
        baselines = suite.list_baselines()
        assert len(baselines) == 1
        print(f"  💾 Baseline management: {len(baselines)} baselines stored")

        return True


def test_agent_fixtures():
    """Test agent fixtures and profile management."""
    print("🤖 Testing Agent Fixtures...")

    # Create agent fixtures
    fixtures = AgentFixtures()

    # Test profile retrieval
    profiles = fixtures.get_all_profiles()
    assert len(profiles) > 0
    print(f"  👥 Available profiles: {len(profiles)}")

    # Test specific profile retrieval
    analyst = fixtures.get_profile("DataAnalyst")
    assert analyst is not None
    assert analyst.role == "analyst"
    assert analyst.specialization == "data_analysis"
    print(f"  🔍 Retrieved profile: {analyst.name} ({analyst.role})")

    # Test profile filtering
    creative_profiles = fixtures.get_profiles_by_role("creative")
    security_profiles = fixtures.get_profiles_by_specialization("cybersecurity")
    assert len(creative_profiles) > 0
    assert len(security_profiles) > 0
    print(f"  🎨 Creative profiles: {len(creative_profiles)}")
    print(f"  🔒 Security profiles: {len(security_profiles)}")

    # Test agent creation
    engine = ExampleEngine()
    agent = fixtures.create_agent_from_profile(analyst, engine)
    assert agent is not None
    assert agent.name == "DataAnalyst"
    print(f"  ⚙️ Created agent: {agent.name}")

    # Test agent group creation
    debate_group = fixtures.create_agent_group("debate_pair", engine=engine)
    assert len(debate_group) == 2
    print(f"  👥 Created debate group: {len(debate_group)} agents")

    analysis_team = fixtures.create_agent_group("analysis_team", engine=engine)
    assert len(analysis_team) >= 2
    print(f"  📊 Created analysis team: {len(analysis_team)} agents")

    # Test custom profile
    custom_profile = AgentProfile(
        name="CustomTester",
        style={"logic": 0.8, "creativity": 0.5},
        role="tester",
        specialization="test_automation",
    )
    fixtures.add_custom_profile(custom_profile)

    retrieved_custom = fixtures.get_profile("CustomTester")
    assert retrieved_custom is not None
    assert retrieved_custom.specialization == "test_automation"
    print(f"  ➕ Added custom profile: {retrieved_custom.name}")

    return True


def test_scenario_generator():
    """Test scenario generation and management."""
    print("📝 Testing Scenario Generator...")

    # Create scenario generator
    generator = ScenarioGenerator()

    # Test scenario retrieval
    scenarios = generator.get_all_scenarios()
    assert len(scenarios) > 0
    print(f"  📚 Available scenarios: {len(scenarios)}")

    # Test scenario filtering
    reasoning_scenarios = generator.get_scenarios_by_category("reasoning")
    complex_scenarios = generator.get_scenarios_by_complexity("complex")
    tool_scenarios = generator.get_scenarios_requiring_features(tools=True)

    assert len(reasoning_scenarios) > 0
    assert len(complex_scenarios) > 0
    assert len(tool_scenarios) > 0

    print(f"  🧠 Reasoning scenarios: {len(reasoning_scenarios)}")
    print(f"  🔧 Complex scenarios: {len(complex_scenarios)}")
    print(f"  🛠️ Tool-requiring scenarios: {len(tool_scenarios)}")

    # Test specific scenario retrieval
    basic_scenario = generator.get_scenario("Basic Problem Solving")
    assert basic_scenario is not None
    assert basic_scenario.category == "reasoning"
    assert len(basic_scenario.prompts) > 0
    print(f"  🎯 Retrieved scenario: {basic_scenario.name}")

    # Test custom scenario generation
    custom_scenario = generator.generate_custom_scenario(
        category="analysis", topic="distributed systems", complexity="medium"
    )

    assert custom_scenario.category == "analysis"
    assert "distributed systems" in custom_scenario.name
    assert custom_scenario.complexity_level == "medium"
    print(f"  🆕 Generated custom scenario: {custom_scenario.name}")

    # Test scenario matrix generation
    scenario_matrix = generator.generate_scenario_matrix(
        categories=["analysis", "design"],
        complexity_levels=["simple", "medium"],
        topics=["API design", "database optimization"],
    )

    expected_count = 2 * 2 * 2  # categories * complexity * topics
    assert len(scenario_matrix) == expected_count
    print(f"  🔢 Generated scenario matrix: {len(scenario_matrix)} scenarios")

    # Test duration filtering
    quick_scenarios = generator.get_scenarios_for_duration(20.0)
    assert all(s.estimated_duration_seconds <= 20.0 for s in quick_scenarios)
    print(f"  ⏱️ Quick scenarios (≤20s): {len(quick_scenarios)}")

    # Test custom scenario addition
    custom_test_scenario = TestScenario(
        name="Custom Integration Test",
        description="Custom scenario for testing",
        category="integration",
        prompts=["Test custom scenario functionality"],
        complexity_level="simple",
        tags=["custom", "test"],
    )

    generator.add_custom_scenario(custom_test_scenario)
    retrieved_custom = generator.get_scenario("Custom Integration Test")
    assert retrieved_custom is not None
    print(f"  ➕ Added custom scenario: {retrieved_custom.name}")

    return True


async def test_end_to_end_testing_workflow():
    """Test complete end-to-end testing workflow."""
    print("🔄 Testing End-to-End Testing Workflow...")

    # Setup components
    fixtures = AgentFixtures()
    generator = ScenarioGenerator()
    matrix = TestMatrix()

    # Create test configuration using fixtures and scenarios
    analyst_profile = fixtures.get_profile("DataAnalyst")
    scenario = generator.get_scenario("Basic Problem Solving")

    assert analyst_profile is not None
    assert scenario is not None

    # Create test configuration
    config = TestConfiguration(
        name="E2E_Workflow_Test",
        feature_set=FeatureSet.PHASE_2,
        agent_type=AgentType.SINGLE,
        agent_styles=[analyst_profile.style],
        test_scenario=scenario.category,
        test_prompts=scenario.prompts[:1],  # Use first prompt
        enable_memory=scenario.requires_memory,
        enable_tools=scenario.requires_tools,
        priority="high",
        tags=["e2e", "workflow"],
    )

    # Register custom test function
    async def custom_test_function(test_config):
        # Create agent using fixtures
        engine = ExampleEngine()
        agent = fixtures.create_agent_from_profile(analyst_profile, engine)

        # Execute scenario
        import time

        start = time.time()
        result = agent.generate_reasoning_tree(test_config.test_prompts[0])
        duration = time.time() - start

        return {
            "success": result is not None,
            "agent_results": [{"agent": agent.name, "result": result}],
            "avg_latency_ms": duration * 1000,
            "total_cost_usd": 0.001,
            "assertions_passed": 1 if result else 0,
            "assertions_failed": 0 if result else 1,
        }

    matrix.register_test_function(scenario.category, custom_test_function)
    matrix.add_configuration(config)

    # Execute test
    results = await matrix.execute_matrix()

    # Verify results
    assert len(results) == 1
    result = results[0]
    assert result.configuration.name == "E2E_Workflow_Test"
    assert result.success is not None

    print(f"  ✅ E2E test executed: {result.configuration.name}")
    print(f"  🎯 Success: {result.success}")
    print(f"  ⏱️ Duration: {result.duration_seconds:.2f}s")
    print(f"  📊 Latency: {result.avg_latency_ms:.2f}ms")

    return True


async def main():
    """Run all testing framework tests."""
    print("🚀 AgentNet Phase 5 Testing Framework Test Suite")
    print("=" * 60)

    tests = [
        test_test_matrix,
        test_integration_test_suite,
        test_regression_test_suite,
        test_agent_fixtures,
        test_scenario_generator,
        test_end_to_end_testing_workflow,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                # Check if the function returns a coroutine
                result = test_func()
                if asyncio.iscoroutine(result):
                    result = await result

            if result:
                passed += 1
                print("  ✅ PASSED\n")
            else:
                failed += 1
                print("  ❌ FAILED\n")
        except Exception as e:
            print(f"  ❌ CRASHED: {e}\n")
            import traceback

            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"📊 Final Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All testing framework tests passed!")
        print("✨ Phase 5 Testing Framework is ready for comprehensive validation!")
        return True
    else:
        print(f"❌ {failed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    asyncio.run(main())
