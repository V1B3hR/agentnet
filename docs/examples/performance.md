# Performance Testing Examples

Learn how to use AgentNet's performance harness and testing framework for comprehensive performance analysis.

## Performance Harness

The performance harness provides configurable benchmarking for measuring turn latency, token utilization, and system throughput.

### Basic Performance Benchmark

```python
import asyncio
from agentnet import AgentNet, ExampleEngine
from agentnet.performance import (
    PerformanceHarness,
    BenchmarkConfig,
    BenchmarkType,
    LatencyTracker,
    TokenUtilizationTracker,
    PerformanceReporter,
    ReportFormat
)

async def basic_performance_test():
    """Run a basic performance benchmark."""
    
    # Create performance harness
    harness = PerformanceHarness()
    
    # Configure benchmark
    config = BenchmarkConfig(
        name="Basic Agent Performance",
        benchmark_type=BenchmarkType.SINGLE_TURN,
        iterations=50,
        concurrency_level=1,
        test_prompts=[
            "Analyze system architecture patterns",
            "Optimize database query performance", 
            "Design a microservices deployment strategy"
        ],
        # Performance thresholds
        max_turn_latency_ms=2000.0,
        min_success_rate=0.95,
        max_cost_usd=0.05
    )
    
    # Agent factory function
    def create_agent():
        return AgentNet(
            name="PerformanceTestAgent",
            style={"logic": 0.8, "creativity": 0.6, "analytical": 0.9},
            engine=ExampleEngine()
        )
    
    # Run benchmark
    result = await harness.run_benchmark(config, create_agent)
    
    # Print results
    print(f"🎯 Success Rate: {result.success_rate:.1%}")
    print(f"⏱️ Average Latency: {result.avg_turn_latency_ms:.1f}ms")
    print(f"🔄 Throughput: {result.operations_per_second:.2f} ops/sec")
    print(f"🪙 Token Efficiency: {result.token_efficiency_score:.3f}")
    print(f"✅ Passed Thresholds: {result.passed_thresholds}")
    
    return result

# Run the benchmark
asyncio.run(basic_performance_test())
```

### Multi-Agent Performance Testing

```python
async def multi_agent_performance_test():
    """Test multi-agent performance scenarios."""
    
    harness = PerformanceHarness()
    
    # Multi-agent configuration
    config = BenchmarkConfig(
        name="Multi-Agent Collaboration",
        benchmark_type=BenchmarkType.MULTI_AGENT,
        iterations=25,
        concurrency_level=2,
        agent_count=3,
        turn_count=4,
        test_prompts=[
            "Collaboratively design a distributed system",
            "Plan a software development sprint together",
            "Review and optimize API architecture"
        ],
        max_turn_latency_ms=5000.0,  # Higher threshold for multi-agent
        timeout_seconds=60.0
    )
    
    # Multi-agent factory
    def create_agent_team():
        """Create a team of agents with different specializations."""
        from agentnet.testing import AgentFixtures
        
        fixtures = AgentFixtures()
        return fixtures.create_agent_group("analysis_team")
    
    # Custom multi-agent operation
    async def multi_agent_operation(agents, prompt):
        """Custom operation for multi-agent testing."""
        
        # Simulate collaborative work
        results = []
        for i, agent in enumerate(agents):
            specialized_prompt = f"As {agent.name}, contribute to: {prompt}"
            result = agent.generate_reasoning_tree(specialized_prompt)
            results.append({
                'agent': agent.name,
                'contribution': result
            })
        
        return {
            'collaboration_result': results,
            'agent_count': len(agents),
            'success': all(r['contribution'] for r in results)
        }
    
    result = await harness.run_benchmark(
        config, 
        create_agent_team,
        multi_agent_operation
    )
    
    print(f"👥 Multi-Agent Results:")
    print(f"  Success Rate: {result.success_rate:.1%}")
    print(f"  Avg Latency: {result.avg_turn_latency_ms:.1f}ms") 
    print(f"  Throughput: {result.operations_per_second:.2f} ops/sec")
    
    return result

asyncio.run(multi_agent_performance_test())
```

## Latency Analysis

Track detailed latency across different components of agent processing.

### Component Latency Tracking

```python
from agentnet.performance import LatencyTracker, LatencyComponent

def detailed_latency_analysis():
    """Analyze latency across different processing components."""
    
    tracker = LatencyTracker()
    agent = AgentNet(
        "LatencyTestAgent",
        {"logic": 0.8, "creativity": 0.6},
        engine=ExampleEngine()
    )
    
    # Start comprehensive latency tracking
    turn_id = "detailed_analysis_001"
    tracker.start_turn_measurement(turn_id, agent.name, prompt_length=150)
    
    # Track inference component
    tracker.start_component_measurement(turn_id, LatencyComponent.INFERENCE)
    result = agent.generate_reasoning_tree(
        "Analyze the performance implications of microservices architecture"
    )
    inference_latency = tracker.end_component_measurement(turn_id, LatencyComponent.INFERENCE)
    
    # Simulate other components
    tracker.start_component_measurement(turn_id, LatencyComponent.POLICY_CHECK)
    # ... policy checks would happen here ...
    policy_latency = tracker.end_component_measurement(turn_id, LatencyComponent.POLICY_CHECK)
    
    tracker.start_component_measurement(turn_id, LatencyComponent.RESPONSE_PROCESSING)
    # ... response processing would happen here ...
    processing_latency = tracker.end_component_measurement(turn_id, LatencyComponent.RESPONSE_PROCESSING)
    
    # Record metadata
    tracker.record_tool_usage(turn_id, "reasoning_engine")
    
    # Complete measurement
    measurement = tracker.end_turn_measurement(
        turn_id, 
        response_length=len(str(result)),
        tokens_processed=200
    )
    
    # Analyze results
    print(f"🔍 Detailed Latency Analysis:")
    print(f"  Total Latency: {measurement.total_latency_ms:.2f}ms")
    print(f"  Component Breakdown:")
    for component, latency in measurement.component_latencies.items():
        print(f"    {component.value}: {latency:.2f}ms")
    
    # Get latency breakdown percentages
    breakdown = measurement.latency_breakdown
    for component, percentage in breakdown.items():
        print(f"    {component}: {percentage:.1f}%")
    
    # System-wide statistics
    stats = tracker.get_latency_statistics()
    print(f"📊 Statistics: {stats}")
    
    return measurement, stats

detailed_latency_analysis()
```

## Token Utilization Optimization

Analyze and optimize token usage for cost efficiency and performance.

### Token Usage Analysis

```python
from agentnet.performance import TokenUtilizationTracker, TokenCategory

def token_optimization_analysis():
    """Comprehensive token utilization analysis."""
    
    tracker = TokenUtilizationTracker()
    agent = AgentNet(
        "TokenOptAgent", 
        {"logic": 0.8, "efficiency": 0.9},
        engine=ExampleEngine()
    )
    
    # Test scenarios with different token patterns
    scenarios = [
        {
            "name": "Concise Analysis",
            "prompt": "Briefly explain REST API design principles",
            "expected_category": TokenCategory.REASONING
        },
        {
            "name": "Detailed Architecture", 
            "prompt": "Provide a comprehensive analysis of microservices architecture including pros, cons, implementation strategies, monitoring approaches, and scaling considerations",
            "expected_category": TokenCategory.REASONING
        },
        {
            "name": "Tool Usage Scenario",
            "prompt": "Calculate system performance metrics and generate recommendations", 
            "expected_category": TokenCategory.TOOL_CALLS
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        
        # Execute scenario
        import time
        start_time = time.time()
        result = agent.generate_reasoning_tree(scenario['prompt'])
        processing_time = time.time() - start_time
        
        # Estimate tokens (in production, use actual tokenizer)
        input_tokens = len(scenario['prompt']) // 4
        output_tokens = len(str(result.get('result', {}).get('content', ''))) // 4
        
        # Record token usage
        metrics = tracker.record_token_usage(
            agent_id=agent.name,
            turn_id=f"{scenario['name'].lower().replace(' ', '_')}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            category_breakdown={
                scenario['expected_category']: output_tokens,
                TokenCategory.SYSTEM: input_tokens // 10  # System overhead
            },
            processing_time=processing_time,
            model_name="gpt-3.5-turbo",
            output_quality_score=0.85  # Simulated quality score
        )
        
        print(f"  Input Tokens: {metrics.input_tokens}")
        print(f"  Output Tokens: {metrics.output_tokens}")
        print(f"  Efficiency Score: {metrics.efficiency_score:.3f}")
        print(f"  Tokens/Second: {metrics.tokens_per_second:.2f}")
        print(f"  Cost Estimate: ${metrics.total_tokens * metrics.cost_per_token / 1000:.4f}")
    
    # Get comprehensive analysis
    overview = tracker.get_system_token_overview()
    print(f"\n📊 System Overview:")
    print(f"  Total Tokens: {overview['overview']['total_tokens']:,}")
    print(f"  Total Turns: {overview['overview']['total_turns']}")
    print(f"  Avg Efficiency: {overview['overview']['avg_efficiency']:.3f}")
    print(f"  Total Cost: ${overview['overview']['total_cost_usd']:.4f}")
    
    # Optimization opportunities
    opportunities = tracker.identify_optimization_opportunities()
    print(f"\n🔧 Optimization Opportunities:")
    for category, issues in opportunities.items():
        if issues:
            print(f"  {category.title()}: {len(issues)} issues")
            for issue in issues[:2]:  # Show first 2
                print(f"    - {issue}")
    
    # Get recommendations
    recommendations = tracker.generate_optimization_recommendations()
    print(f"\n💡 Recommendations:")
    for rec in recommendations[:3]:  # Show top 3
        print(f"  - {rec}")
    
    return overview, opportunities, recommendations

token_optimization_analysis()
```

## Performance Regression Testing

Set up automated regression testing to catch performance degradations.

### Baseline Management

```python
import tempfile
from agentnet.testing import RegressionTestSuite

def performance_regression_testing():
    """Demonstrate performance regression testing."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regression test suite
        suite = RegressionTestSuite(baseline_dir=temp_dir)
        
        # Run baseline performance test
        async def create_baseline():
            harness = PerformanceHarness()
            config = BenchmarkConfig(
                name="Regression Baseline",
                benchmark_type=BenchmarkType.SINGLE_TURN,
                iterations=20
            )
            
            def agent_factory():
                return AgentNet(
                    "BaselineAgent", 
                    {"logic": 0.8, "creativity": 0.6},
                    engine=ExampleEngine()
                )
            
            return await harness.run_benchmark(config, agent_factory)
        
        # Create baseline from benchmark results
        baseline_result = asyncio.run(create_baseline())
        
        baseline_data = {
            'avg_latency_ms': baseline_result.avg_turn_latency_ms,
            'p95_latency_ms': baseline_result.p95_turn_latency_ms,
            'throughput_ops_per_sec': baseline_result.operations_per_second,
            'success_rate': baseline_result.success_rate,
            'token_efficiency_score': baseline_result.token_efficiency_score
        }
        
        test_config = {
            'benchmark_type': 'single_turn',
            'agent_count': 1,
            'iterations': 20
        }
        
        # Create performance baseline
        baseline = suite.create_baseline(
            version="v1.0.0",
            performance_data=baseline_data,
            test_configuration=test_config
        )
        
        print(f"📊 Created baseline for v1.0.0:")
        print(f"  Avg Latency: {baseline.avg_latency_ms:.2f}ms")
        print(f"  Success Rate: {baseline.success_rate:.1%}")
        
        # Simulate performance regression
        current_data = baseline_data.copy()
        current_data.update({
            'avg_latency_ms': baseline_data['avg_latency_ms'] * 1.3,  # 30% slower
            'throughput_ops_per_sec': baseline_data['throughput_ops_per_sec'] * 0.8,  # 20% slower
            'success_rate': baseline_data['success_rate'] * 0.95  # 5% less reliable
        })
        
        # Detect regressions
        regressions = suite.detect_regressions(
            current_version="v1.1.0",
            current_performance=current_data,
            test_configuration=test_config
        )
        
        print(f"\n⚠️ Regression Analysis:")
        print(f"  Detected {len(regressions)} regressions")
        
        for regression in regressions:
            print(f"  - {regression.metric_name}: {regression.regression_percentage:.1%} regression")
            print(f"    Severity: {regression.severity}")
            print(f"    Recommendation: {regression.recommendation}")
        
        # Generate regression report
        if regressions:
            report = suite.generate_regression_report(regressions, "v1.1.0")
            print(f"\n📄 Generated regression report ({len(report)} chars)")
            
            # Save report to file
            with open("performance_regression_report.md", "w") as f:
                f.write(report)
            print("  Saved to: performance_regression_report.md")
        
        return suite, regressions

performance_regression_testing()
```

## Comprehensive Performance Reports

Generate detailed performance reports with actionable insights.

### Report Generation

```python
async def comprehensive_performance_report():
    """Generate a comprehensive performance report."""
    
    # Run multiple benchmarks
    harness = PerformanceHarness()
    latency_tracker = LatencyTracker()
    token_tracker = TokenUtilizationTracker()
    
    # Different benchmark configurations
    configs = [
        BenchmarkConfig(
            name="Single Agent Basic",
            benchmark_type=BenchmarkType.SINGLE_TURN,
            iterations=30
        ),
        BenchmarkConfig(
            name="Multi-Agent Collaboration", 
            benchmark_type=BenchmarkType.MULTI_AGENT,
            iterations=15,
            agent_count=2
        ),
        BenchmarkConfig(
            name="Concurrent Processing",
            benchmark_type=BenchmarkType.CONCURRENT_AGENTS,
            iterations=20,
            concurrency_level=3
        )
    ]
    
    # Run benchmarks and collect data
    results = []
    for config in configs:
        def agent_factory():
            return AgentNet(
                f"ReportAgent_{len(results)}",
                {"logic": 0.8, "creativity": 0.6},
                engine=ExampleEngine()
            )
        
        result = await harness.run_benchmark(config, agent_factory)
        results.append(result)
        
        # Simulate some latency and token tracking
        for i in range(5):
            turn_id = f"report_turn_{len(results)}_{i}"
            latency_tracker.start_turn_measurement(turn_id, f"Agent_{len(results)}")
            
            # Simulate processing
            import time
            time.sleep(0.01)  # Simulate work
            
            measurement = latency_tracker.end_turn_measurement(turn_id, 100, 75)
            
            # Record token usage
            token_tracker.record_token_usage(
                agent_id=f"Agent_{len(results)}",
                turn_id=turn_id,
                input_tokens=50 + i * 10,
                output_tokens=75 + i * 15,
                processing_time=measurement.total_latency_ms / 1000
            )
    
    # Generate comprehensive report
    reporter = PerformanceReporter()
    
    # Generate reports in different formats
    markdown_report = reporter.generate_comprehensive_report(
        benchmark_results=results,
        latency_tracker=latency_tracker,
        token_tracker=token_tracker,
        format=ReportFormat.MARKDOWN
    )
    
    json_report = reporter.generate_comprehensive_report(
        benchmark_results=results,
        latency_tracker=latency_tracker,
        token_tracker=token_tracker,
        format=ReportFormat.JSON
    )
    
    print(f"📊 Generated Performance Reports:")
    print(f"  Markdown: {markdown_report}")
    print(f"  JSON: {json_report}")
    
    # Display key findings
    print(f"\n🔍 Key Performance Findings:")
    for i, result in enumerate(results):
        print(f"  {result.config.name}:")
        print(f"    Success Rate: {result.success_rate:.1%}")
        print(f"    Avg Latency: {result.avg_turn_latency_ms:.1f}ms")
        print(f"    Throughput: {result.operations_per_second:.2f} ops/sec")
    
    return markdown_report, json_report

# Generate comprehensive report
asyncio.run(comprehensive_performance_report())
```

## Next Steps

- **[Testing Framework](../development/testing.md)** - Learn about systematic testing
- **[API Reference - Performance](../api/performance.md)** - Detailed API documentation
- **[Architecture - Performance Model](../architecture/performance.md)** - Understanding the performance model