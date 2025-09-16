# AgentNet Experimental Features

This document describes the new experimental features added to AgentNet for advanced multi-agent reasoning, performance analysis, and safety monitoring.

## Overview

Five major experimental feature sets have been implemented:

1. **Fault Injection & Resilience** - Test system robustness under failure conditions
2. **Async vs Sync Performance Benchmarking** - Compare execution modes and optimize performance  
3. **Analytics Index Generation** - Extract insights and create searchable indices from agent interactions
4. **Extensibility Experiments (Custom Monitors)** - Create custom monitoring and policy enforcement
5. **Policy/Safety Modeling Extensions** - Advanced hierarchical governance and safety analysis

## Quick Start

```bash
# Run comprehensive experimental demo
python AgentNet.py --demo experimental

# Run specific demos
python AgentNet.py --demo sync
python AgentNet.py --demo async
python AgentNet.py --demo both
```

## 1. Fault Injection & Resilience

### Features
- Circuit breaker patterns
- Automatic retry with exponential backoff
- Fault injection for testing (network, processing, memory faults)
- Resilience metrics and monitoring

### Usage Example
```python
from AgentNet import *

# Create agent with resilience capabilities
engine = ExampleEngine()
agent = AgentNet('ResilientAgent', {'logic': 0.8}, engine)

# Use resilience-enabled reasoning
result = agent.generate_reasoning_tree_with_resilience("Analyze system robustness")

# Check resilience metrics
metrics = agent.get_resilience_metrics()
print(f"Success rate: {metrics['success_rate']:.2f}")
print(f"Recovery operations: {metrics['recovered_operations']}")
```

### Configuration
```python
# Configure fault injection for testing
fault_config = FaultConfig(
    fault_type=FaultType.PROCESSING_ERROR,
    probability=0.1,  # 10% chance of fault
    delay_ms=100,
    recovery_attempts=3
)

fault_monitor = FaultInjectionMonitor('test_faults', [fault_config])
```

## 2. Performance Benchmarking

### Features
- Sync vs async performance comparison
- Concurrent load testing
- Throughput and latency analysis
- Scalability assessment
- Automated report generation

### Usage Example
```python
# Create benchmark suite
benchmark = PerformanceBenchmark(agent)

# Define test tasks
tasks = [
    "Analyze distributed systems",
    "Design microservices architecture",
    "Evaluate cloud deployment"
]

# Run comprehensive benchmark
results = benchmark.benchmark_reasoning_tree(
    tasks, 
    concurrency_levels=[1, 5, 10, 20]
)

# Analyze results
comparison = results["comparison"]
print(f"Async advantage at scale: {comparison['scalability_analysis']['async_advantage_at_scale']:.2f}x")

# Export detailed report
benchmark.export_benchmark_report("performance_report.json")
```

## 3. Analytics Index Generation

### Features
- Keyword extraction and topic modeling
- Sentiment and complexity scoring
- Interaction pattern analysis
- Search and filtering capabilities
- Comprehensive analytics reports

### Usage Example
```python
# Create analytics indexer
indexer = AnalyticsIndexer()

# Index a session
session_data = {
    "session_id": "analysis_session_001",
    "participants": ["Agent1", "Agent2"],
    "transcript": [
        {"agent": "Agent1", "content": "Let's analyze the performance bottlenecks..."},
        {"agent": "Agent2", "content": "The optimization strategies show promising results..."}
    ]
}

# Generate index
index = indexer.index_session(session_data)
print(f"Keywords: {index.keywords}")
print(f"Sentiment: {index.sentiment_score:.2f}")
print(f"Complexity: {index.complexity_score:.2f}")

# Search indexed sessions
results = indexer.search_sessions("performance optimization", filters={
    "min_sentiment": 0.4,
    "agents": ["Agent1"]
})

# Generate analytics report
report = indexer.generate_analytics_report()
```

## 4. Custom Monitors (Extensibility)

### Features
- Template-based monitor creation
- Plugin system for dynamic loading
- Built-in monitors: sentiment, complexity, domain-specific
- Monitor chaining and composition
- Performance metrics for monitors

### Usage Example
```python
# Create plugin system
plugin = MonitorPlugin()

# Load built-in monitors
sentiment_monitor = plugin.load_monitor_from_config({
    "type": "sentiment",
    "name": "content_sentiment",
    "min_sentiment": 0.3,
    "max_sentiment": 0.8
})

domain_monitor = plugin.load_monitor_from_config({
    "type": "domain_specific", 
    "name": "tech_domain",
    "domain_keywords": ["algorithm", "optimization", "scalability"],
    "required_keywords": ["performance"],
    "forbidden_keywords": ["hack", "exploit"]
})

# Create monitor chain
monitor_chain = MonitorChain(
    [sentiment_monitor, domain_monitor],
    chain_strategy="all_pass"  # or "any_pass", "majority_pass"
)

# Evaluate content
outcome = {"result": {"content": "Performance optimization requires algorithmic improvements"}}
passed, violations, elapsed = monitor_chain.evaluate(outcome)
```

### Creating Custom Monitors
```python
class CustomMonitor(MonitorTemplate):
    def __init__(self, name: str, threshold: float):
        super().__init__(name)
        self.threshold = threshold
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        start_time = time.perf_counter()
        
        # Your custom logic here
        content = outcome.get("result", {}).get("content", "")
        score = len(content.split())  # Example: word count
        
        passed = score >= self.threshold
        rationale = f"Word count {score} below threshold {self.threshold}" if not passed else None
        
        elapsed = time.perf_counter() - start_time
        return passed, rationale, elapsed

# Register with plugin system
plugin.register_monitor_class("word_count", CustomMonitor)
```

## 5. Policy/Safety Modeling Extensions

### Features
- Hierarchical policy levels (organizational, team, individual, session)
- Policy versioning and rollback
- A/B testing for policy configurations
- Safety impact analysis
- Comprehensive governance reporting

### Usage Example
```python
# Create hierarchical policy engine
policy_engine = HierarchicalPolicyEngine()

# Add organizational-level policies
org_rule = ConstraintRule(
    name="no_harmful_content",
    check=lambda outcome: "harmful" not in outcome.get("result", {}).get("content", "").lower(),
    severity=Severity.SEVERE
)
policy_engine.add_policy_level(PolicyLevel.ORGANIZATIONAL, [org_rule])

# Add team-level policies
team_rule = ConstraintRule(
    name="technical_accuracy",
    check=lambda outcome: len(outcome.get("result", {}).get("content", "")) > 50,
    severity=Severity.MINOR
)
policy_engine.add_policy_level(PolicyLevel.TEAM, [team_rule])

# Evaluate with hierarchy
violations = policy_engine.evaluate_hierarchical(outcome, PolicyLevel.SESSION)

# Version policies
policy_engine.version_policy(PolicyLevel.ORGANIZATIONAL, "v1.0", {"author": "admin"})

# A/B test policies
ab_test = PolicyABTesting()
ab_test.create_experiment("policy_test", policy_a=[org_rule], policy_b=[org_rule, team_rule])

# Evaluate with A/B test
result = ab_test.evaluate_with_experiment("policy_test", outcome)
print(f"Used policy variant: {result['policy_variant']}")

# Safety impact analysis
safety_analyzer = SafetyImpactAnalyzer()
safety_analyzer.establish_baseline("baseline_v1", {"violation_rate": 0.05})
safety_analyzer.record_policy_impact("policy_v1", outcome, violations)
trends = safety_analyzer.analyze_safety_trends(time_window_hours=24)
```

## Report Generation

All features support comprehensive reporting:

```python
# Export benchmark results
benchmark.export_benchmark_report("benchmark_report.json")

# Export analytics indices
indexer.export_indices("analytics_report.json") 

# Export safety analysis
safety_analyzer.export_safety_report("safety_report.json")

# Get policy metrics
metrics = policy_engine.get_policy_metrics()
```

## Integration with Existing AgentNet

All experimental features integrate seamlessly with existing AgentNet functionality:

```python
# Standard agent creation
agent = AgentNet("ExperimentalAgent", {"logic": 0.9, "creativity": 0.6}, engine)

# Add resilience
result = agent.generate_reasoning_tree_with_resilience("Task with resilience")

# Add custom monitors to existing agent
custom_monitors = [sentiment_monitor, complexity_monitor]
agent.monitors.extend(custom_monitors)  # If using function-based monitors

# Benchmark existing agent operations
benchmark = PerformanceBenchmark(agent)
results = benchmark.benchmark_reasoning_tree(["Task 1", "Task 2"])
```

## Configuration Files

Monitors can be configured via YAML:

```yaml
# monitors_experimental.yaml
monitors:
  - name: sentiment_analysis
    type: sentiment
    severity: minor
    params:
      min_sentiment: 0.3
      max_sentiment: 0.8
      
  - name: domain_compliance
    type: domain_specific
    severity: major
    params:
      domain_keywords: ["security", "performance", "scalability"]
      required_keywords: ["analysis"]
      forbidden_keywords: ["hack", "exploit"]
```

## Performance Considerations

- **Resilience**: Adds retry overhead but improves reliability
- **Benchmarking**: Minimal overhead when not actively benchmarking
- **Analytics**: Indexing has O(n) complexity with content size
- **Custom Monitors**: Performance depends on monitor complexity
- **Policy Engine**: Evaluation time scales with number of rules

## Best Practices

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Monitor Performance**: Use benchmarking to understand impact
3. **Version Policies**: Always version policy changes for rollback capability
4. **Test Resilience**: Use fault injection to validate system robustness
5. **Analyze Trends**: Regularly review analytics and safety reports

## Future Enhancements

- Integration with external ML models for advanced analytics
- Real-time streaming analysis capabilities
- Advanced statistical analysis for A/B testing
- Integration with monitoring platforms (Prometheus, Grafana)
- Enhanced fault injection scenarios