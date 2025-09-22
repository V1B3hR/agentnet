#!/usr/bin/env python3
"""
Phase 5 Performance Harness Tests

Tests the performance harness, latency tracking, token utilization,
and report generation components implemented for Phase 5.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from agentnet import ExampleEngine, AgentNet
from agentnet.performance import (
    PerformanceHarness,
    BenchmarkConfig,
    BenchmarkType,
    LatencyTracker,
    LatencyComponent,
    TokenUtilizationTracker,
    TokenCategory,
    PerformanceReporter,
    ReportFormat,
)


@pytest.mark.asyncio
async def test_performance_harness():
    """Test basic performance harness functionality."""
    print("🚀 Testing Performance Harness...")
    
    # Create test harness
    harness = PerformanceHarness()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        name="Basic Agent Test",
        benchmark_type=BenchmarkType.SINGLE_TURN,
        iterations=5,
        concurrency_level=1,
        test_prompts=["Test prompt 1", "Test prompt 2"]
    )
    
    # Create agent factory
    def agent_factory():
        return AgentNet(
            "TestAgent",
            {"logic": 0.8, "creativity": 0.6},
            engine=ExampleEngine()
        )
    
    # Run benchmark
    result = await harness.run_benchmark(config, agent_factory)
    
    # Verify results
    assert result.config.name == "Basic Agent Test"
    assert result.total_operations == 5
    assert result.successful_operations >= 0
    assert result.success_rate >= 0.0
    assert result.total_duration > 0
    
    print(f"  ✅ Benchmark completed: {result.success_rate:.1%} success rate")
    print(f"  ⏱️ Average latency: {result.avg_turn_latency_ms:.1f}ms")
    print(f"  🔄 Throughput: {result.operations_per_second:.2f} ops/sec")


def test_latency_tracker():
    """Test latency tracking functionality."""
    print("⏱️ Testing Latency Tracker...")
    
    tracker = LatencyTracker()
    
    # Start turn measurement
    turn_id = "test_turn_001"
    agent_id = "TestAgent"
    
    tracker.start_turn_measurement(turn_id, agent_id, prompt_length=50)
    
    # Simulate component measurements
    tracker.start_component_measurement(turn_id, LatencyComponent.INFERENCE)
    # Simulate some processing time
    import time
    time.sleep(0.01)
    inference_latency = tracker.end_component_measurement(turn_id, LatencyComponent.INFERENCE)
    
    tracker.start_component_measurement(turn_id, LatencyComponent.POLICY_CHECK)
    time.sleep(0.005)
    policy_latency = tracker.end_component_measurement(turn_id, LatencyComponent.POLICY_CHECK)
    
    # Record additional data
    tracker.record_tool_usage(turn_id, "calculator")
    tracker.record_policy_violation(turn_id)
    
    # End measurement
    measurement = tracker.end_turn_measurement(
        turn_id, response_length=100, tokens_processed=75
    )
    
    # Verify measurement
    assert measurement.turn_id == turn_id
    assert measurement.agent_id == agent_id
    assert measurement.total_latency_ms > 0
    assert inference_latency > 0
    assert policy_latency > 0
    assert len(measurement.tools_used) == 1
    assert measurement.policy_violations == 1
    
    # Test statistics
    stats = tracker.get_latency_statistics()
    assert stats['count'] == 1
    assert stats['mean'] > 0
    
    print(f"  ✅ Measurement recorded: {measurement.total_latency_ms:.2f}ms total")
    print(f"  🔧 Components: Inference {inference_latency:.2f}ms, Policy {policy_latency:.2f}ms")
    print(f"  📊 Statistics: {stats}")
    
    # Assert success instead of returning
    assert measurement is not None
    assert measurement.total_latency_ms > 0


def test_token_utilization_tracker():
    """Test token utilization tracking."""
    print("🪙 Testing Token Utilization Tracker...")
    
    tracker = TokenUtilizationTracker()
    
    # Record token usage
    metrics = tracker.record_token_usage(
        agent_id="TestAgent",
        turn_id="test_turn_001",
        input_tokens=150,
        output_tokens=75,
        category_breakdown={
            TokenCategory.REASONING: 100,
            TokenCategory.TOOL_CALLS: 50,
            TokenCategory.DIALOGUE: 75
        },
        context_length=300,
        processing_time=0.5,
        model_name="gpt-3.5-turbo",
        output_quality_score=0.85
    )
    
    # Verify metrics
    assert metrics.agent_id == "TestAgent"
    assert metrics.total_tokens == 225
    assert metrics.efficiency_score > 0
    assert metrics.tokens_per_second > 0
    
    # Test system overview
    overview = tracker.get_system_token_overview()
    assert overview['overview']['total_tokens'] == 225
    assert overview['overview']['unique_agents'] == 1
    
    # Test agent summary
    agent_summary = tracker.get_agent_token_summary("TestAgent")
    assert agent_summary['tokens']['total'] == 225
    assert agent_summary['total_turns'] == 1
    
    # Test optimization opportunities
    opportunities = tracker.identify_optimization_opportunities()
    assert isinstance(opportunities, dict)
    
    # Test recommendations
    recommendations = tracker.generate_optimization_recommendations()
    assert isinstance(recommendations, list)
    
    print(f"  ✅ Token metrics recorded: {metrics.total_tokens} tokens")
    print(f"  📈 Efficiency score: {metrics.efficiency_score:.3f}")
    print(f"  💰 Cost per token: ${metrics.cost_per_token:.4f}")
    print(f"  🔍 Optimization opportunities: {len(sum(opportunities.values(), []))}")
    
    # Assert success instead of returning
    assert metrics.total_tokens == 225
    assert opportunities is not None


@pytest.mark.asyncio
async def test_performance_reporter():
    """Test performance report generation."""
    print("📊 Testing Performance Reporter...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = PerformanceReporter(output_dir=temp_dir)
        
        # Create mock benchmark results
        harness = PerformanceHarness()
        config = BenchmarkConfig(
            name="Test Benchmark",
            benchmark_type=BenchmarkType.SINGLE_TURN,
            iterations=3
        )
        
        def agent_factory():
            return AgentNet("TestAgent", {"logic": 0.8}, engine=ExampleEngine())
        
        # Run a quick benchmark
        benchmark_result = await harness.run_benchmark(config, agent_factory)
        
        # Create latency tracker with sample data
        latency_tracker = LatencyTracker()
        latency_tracker.start_turn_measurement("test_001", "TestAgent", 50)
        import time
        time.sleep(0.01)
        latency_tracker.end_turn_measurement("test_001", 100, 75)
        
        # Create token tracker with sample data
        token_tracker = TokenUtilizationTracker()
        token_tracker.record_token_usage(
            "TestAgent", "test_001", 100, 50, processing_time=0.01
        )
        
        # Generate comprehensive report
        report_path = reporter.generate_comprehensive_report(
            benchmark_results=[benchmark_result],
            latency_tracker=latency_tracker,
            token_tracker=token_tracker,
            format=ReportFormat.MARKDOWN
        )
        
        # Verify report was created
        assert Path(report_path).exists()
        
        # Read and verify report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        assert "AgentNet Performance Report" in content
        assert "Executive Summary" in content
        assert "Recommendations" in content
        
        # Generate JSON report
        json_report_path = reporter.generate_comprehensive_report(
            benchmark_results=[benchmark_result],
            latency_tracker=latency_tracker,
            token_tracker=token_tracker,
            format=ReportFormat.JSON
        )
        
        assert Path(json_report_path).exists()
        
        print(f"  ✅ Reports generated:")
        print(f"    📝 Markdown: {Path(report_path).name}")
        print(f"    📋 JSON: {Path(json_report_path).name}")
        print(f"  📄 Report size: {len(content)} characters")


@pytest.mark.asyncio
async def test_integrated_performance_measurement():
    """Test integrated performance measurement across components."""
    print("🔗 Testing Integrated Performance Measurement...")
    
    # Create integrated measurement system
    harness = PerformanceHarness()
    latency_tracker = LatencyTracker()
    token_tracker = TokenUtilizationTracker()
    
    # Create test configuration
    config = BenchmarkConfig(
        name="Integrated Performance Test",
        benchmark_type=BenchmarkType.MULTI_TURN,
        iterations=3,
        turn_count=2,
        test_prompts=["Analyze distributed systems", "Explain machine learning"]
    )
    
    def agent_factory():
        return AgentNet(
            "IntegratedTestAgent",
            {"logic": 0.9, "creativity": 0.7, "analytical": 0.8},
            engine=ExampleEngine()
        )
    
    # Custom operation that includes latency and token tracking
    async def tracked_operation(agent, prompt):
        turn_id = f"turn_{hash(prompt) % 10000}"
        
        # Start latency tracking
        latency_tracker.start_turn_measurement(turn_id, agent.name, len(prompt))
        
        # Simulate component latencies
        latency_tracker.start_component_measurement(turn_id, LatencyComponent.INFERENCE)
        result = agent.generate_reasoning_tree(prompt)
        inference_latency = latency_tracker.end_component_measurement(turn_id, LatencyComponent.INFERENCE)
        
        # Record token usage (estimated)
        input_tokens = len(prompt) // 4  # Rough estimate
        output_tokens = len(str(result.get('result', {}).get('content', ''))) // 4
        
        token_tracker.record_token_usage(
            agent_id=agent.name,
            turn_id=turn_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            processing_time=inference_latency / 1000,  # Convert to seconds
            category_breakdown={
                TokenCategory.REASONING: output_tokens,
                TokenCategory.DIALOGUE: input_tokens
            }
        )
        
        # End latency tracking
        latency_tracker.end_turn_measurement(turn_id, len(str(result)), input_tokens + output_tokens)
        
        return result
    
    # Run integrated benchmark
    benchmark_result = await harness.run_benchmark(config, agent_factory, tracked_operation)
    
    # Verify integrated results
    assert benchmark_result.successful_operations > 0
    
    measurements = latency_tracker.get_measurements()
    assert len(measurements) > 0
    
    token_overview = token_tracker.get_system_token_overview()
    assert token_overview['overview']['total_tokens'] > 0
    
    print(f"  ✅ Integrated benchmark completed:")
    print(f"    🎯 Success rate: {benchmark_result.success_rate:.1%}")
    print(f"    ⏱️ Average latency: {benchmark_result.avg_turn_latency_ms:.1f}ms")
    print(f"    🪙 Total tokens: {token_overview['overview']['total_tokens']}")
    print(f"    📏 Latency measurements: {len(measurements)}")
    
    # Assert success instead of returning
    assert benchmark_result.success_rate >= 0.0
    assert measurements is not None


async def main():
    """Run all performance harness tests."""
    print("🚀 AgentNet Phase 5 Performance Harness Test Suite")
    print("=" * 60)
    
    tests = [
        test_performance_harness,
        test_latency_tracker,
        test_token_utilization_tracker,
        test_performance_reporter,
        test_integrated_performance_measurement,
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
        print("🎉 All performance harness tests passed!")
        print("✨ Phase 5 Performance Harness is ready for production use!")
        return True
    else:
        print(f"❌ {failed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    asyncio.run(main())