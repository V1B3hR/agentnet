# Performance API Reference

Comprehensive API documentation for AgentNet's performance monitoring and benchmarking capabilities.

## Performance Harness

::: agentnet.performance.harness.PerformanceHarness
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.harness.BenchmarkConfig
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.harness.BenchmarkResult
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Latency Tracking

::: agentnet.performance.latency.LatencyTracker
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.latency.TurnLatencyMeasurement
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.latency.LatencyComponent
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Token Utilization

::: agentnet.performance.tokens.TokenUtilizationTracker
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.tokens.TokenMetrics
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.tokens.TokenCategory
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Performance Reporting

::: agentnet.performance.reports.PerformanceReporter
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: agentnet.performance.reports.ReportFormat
    handler: python
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Usage Examples

### Basic Performance Benchmark

```python
from agentnet.performance import PerformanceHarness, BenchmarkConfig, BenchmarkType

async def run_benchmark():
    harness = PerformanceHarness()
    
    config = BenchmarkConfig(
        name="Basic Test",
        benchmark_type=BenchmarkType.SINGLE_TURN,
        iterations=10
    )
    
    def agent_factory():
        return AgentNet("TestAgent", {"logic": 0.8}, engine=ExampleEngine())
    
    result = await harness.run_benchmark(config, agent_factory)
    print(f"Success rate: {result.success_rate:.1%}")
    return result
```

### Latency Analysis

```python
from agentnet.performance import LatencyTracker, LatencyComponent

def track_latency():
    tracker = LatencyTracker()
    
    # Start measurement
    tracker.start_turn_measurement("turn_001", "agent_1", prompt_length=100)
    
    # Track component
    tracker.start_component_measurement("turn_001", LatencyComponent.INFERENCE)
    # ... perform inference ...
    latency = tracker.end_component_measurement("turn_001", LatencyComponent.INFERENCE)
    
    # Complete measurement
    measurement = tracker.end_turn_measurement("turn_001", response_length=200)
    
    return measurement
```

### Token Optimization

```python
from agentnet.performance import TokenUtilizationTracker, TokenCategory

def analyze_tokens():
    tracker = TokenUtilizationTracker()
    
    # Record usage
    metrics = tracker.record_token_usage(
        agent_id="agent_1",
        turn_id="turn_001",
        input_tokens=100,
        output_tokens=150,
        category_breakdown={
            TokenCategory.REASONING: 120,
            TokenCategory.SYSTEM: 30
        }
    )
    
    # Get insights
    opportunities = tracker.identify_optimization_opportunities()
    recommendations = tracker.generate_optimization_recommendations()
    
    return metrics, opportunities, recommendations
```

### Performance Reports

```python
from agentnet.performance import PerformanceReporter, ReportFormat

def generate_report(benchmark_results, latency_tracker, token_tracker):
    reporter = PerformanceReporter()
    
    report_path = reporter.generate_comprehensive_report(
        benchmark_results=benchmark_results,
        latency_tracker=latency_tracker,
        token_tracker=token_tracker,
        format=ReportFormat.MARKDOWN
    )
    
    return report_path
```

## Performance Thresholds

Configure performance expectations and alerts:

| Metric | Recommended Threshold | Critical Threshold |
|--------|----------------------|-------------------|
| Turn Latency | < 2000ms | < 5000ms |
| Success Rate | > 95% | > 90% |
| Token Efficiency | > 0.7 | > 0.5 |
| Cost per Turn | < $0.01 | < $0.05 |

## Best Practices

### Benchmark Configuration

1. **Iterations**: Use at least 20 iterations for reliable statistics
2. **Warmup**: Always include warmup iterations to stabilize performance
3. **Concurrency**: Test both sequential and concurrent scenarios
4. **Prompts**: Use representative test prompts from your use case

### Latency Optimization

1. **Component Tracking**: Monitor individual components to identify bottlenecks
2. **Baseline Comparison**: Establish baselines for regression detection  
3. **Environment Consistency**: Run tests in consistent environments
4. **Resource Monitoring**: Track CPU, memory, and network usage

### Token Efficiency

1. **Category Analysis**: Understand token distribution across categories
2. **Quality vs Efficiency**: Balance output quality with token consumption
3. **Model Selection**: Choose appropriate models for different tasks
4. **Prompt Optimization**: Refine prompts for better token efficiency

## Error Handling

Handle common performance measurement errors:

```python
from agentnet.performance import PerformanceHarness
import asyncio

async def robust_benchmarking():
    harness = PerformanceHarness()
    
    try:
        result = await asyncio.wait_for(
            harness.run_benchmark(config, agent_factory),
            timeout=300.0  # 5 minute timeout
        )
        return result
    except asyncio.TimeoutError:
        print("Benchmark timed out - consider reducing iterations")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        # Handle gracefully or retry with reduced scope
```

## See Also

- [Performance Examples](../examples/performance.md) - Working examples
- [Testing API](testing.md) - Testing framework API
- [Architecture](../architecture/performance.md) - Performance model details