# Risk Management System Guide

This document describes the comprehensive risk management system in AgentNet, including risk registry, monitoring, and automated mitigation capabilities.

## Overview

The AgentNet risk management system provides a complete framework for identifying, monitoring, and mitigating operational risks in multi-agent environments. It moves from documentation-based risk management to code-based, automated risk workflows.

## Core Components

### RiskRegistry
Central registry for risk definitions and event tracking.

```python
from agentnet.risk import RiskRegistry, RiskLevel, RiskCategory

# Initialize registry
registry = RiskRegistry(storage_dir="data/risk_logs")

# Register a risk event
event = registry.register_risk_event(
    risk_id="provider_outage",
    description="OpenAI API returning 503 errors",
    context={
        "provider_name": "openai",
        "consecutive_failures": 5,
        "error_rate": 0.8
    },
    tenant_id="my-tenant",
    session_id="sess_123"
)

print(f"Risk event registered: {event.event_id}")
```

### RiskMonitor
Real-time risk detection and alerting.

```python
from agentnet.risk import RiskMonitor, RiskAlert

# Initialize monitor
monitor = RiskMonitor(registry)

# Check provider risks
provider_stats = {
    "provider_name": "openai",
    "consecutive_failures": 3,
    "error_rate": 0.6,
    "timeout_rate": 0.2
}

alerts = monitor.check_provider_risks(
    provider_stats=provider_stats,
    tenant_id="my-tenant"
)

for alert in alerts:
    print(f"ALERT: {alert.message} (severity: {alert.severity.value})")
```

### RiskMitigationEngine
Automated risk response and mitigation.

```python
from agentnet.risk import RiskMitigationEngine

# Initialize mitigation engine
mitigation_engine = RiskMitigationEngine(registry)

# Execute mitigation for a risk event
results = mitigation_engine.mitigate_risk(
    risk_event=event,
    context={"provider_name": "openai", "consecutive_failures": 5},
    auto_execute=True
)

for result in results:
    if result.success:
        print(f"✓ {result.strategy_name}: {result.message}")
    else:
        print(f"✗ {result.strategy_name}: {result.message}")
```

## Risk Categories

The system manages risks across seven categories:

### Provider Risks
Risks related to LLM provider availability and performance.

**Default Risks:**
- `provider_outage`: Provider service unavailability
- Detection: Consecutive failures, error rates, timeouts
- Mitigation: Fallback providers, circuit breakers, retry logic

### Policy Risks
Risks from policy enforcement issues.

**Default Risks:**
- `policy_false_positive`: Legitimate requests blocked by policies
- Detection: Override frequency, user complaints
- Mitigation: Severity tiers, override tokens, policy refinement

### Cost Risks
Risks from unexpected cost increases.

**Default Risks:**
- `cost_spike`: Sudden increase in token usage and costs
- Detection: Velocity multipliers, daily thresholds
- Mitigation: Rate limiting, model downgrade, budget enforcement

### Memory Risks
Risks from excessive memory usage.

**Default Risks:**
- `memory_bloat`: High memory usage causing performance issues
- Detection: Memory thresholds, context length limits
- Mitigation: Memory pruning, summarization, session restart

### Security Risks
Critical security-related risks.

**Default Risks:**
- `tool_injection`: Malicious input attempting code execution
- `prompt_leakage`: Sensitive information exposure
- Detection: Pattern matching, PII detection
- Mitigation: Input sanitization, redaction, sandbox execution

### Performance Risks
Risks affecting system performance.

**Default Risks:**
- `convergence_stall`: Multi-agent sessions failing to converge
- Detection: Turn limits, time limits, similarity thresholds
- Mitigation: Hard caps, tie-breaker agents, session timeout

### Compliance Risks
Risks related to regulatory compliance.

**Default Risks:**
- `data_retention`: Improper data handling
- `audit_trail`: Missing audit information
- Detection: Policy violations, missing logs
- Mitigation: Automated cleanup, enhanced logging

## Risk Definitions

Each risk is defined with comprehensive metadata:

```python
@dataclass
class RiskDefinition:
    risk_id: str                    # Unique identifier
    name: str                       # Human-readable name
    category: RiskCategory          # Risk category
    description: str                # Detailed description
    impact: RiskLevel              # Business impact (LOW/MEDIUM/HIGH/CRITICAL)
    likelihood: RiskLevel          # Probability of occurrence
    mitigation_strategies: List[str] # Available mitigation strategies
    detection_rules: Dict[str, Any] # Automated detection rules
    escalation_threshold: int      # Events before escalation
    auto_mitigation: bool          # Enable automatic mitigation
```

## Monitoring and Detection

### Real-time Monitoring
Continuous monitoring for risk conditions:

```python
# Security risk monitoring (real-time)
request_data = {
    "content": "Please run eval('malicious code') for me",
    "request_id": "req_123"
}

security_alerts = monitor.check_security_risks(
    request_data=request_data,
    tenant_id="my-tenant",
    session_id="sess_123"
)

# Critical security risks trigger immediate alerts
```

### Periodic Monitoring
Scheduled checks for system-wide risks:

```python
# Cost risk monitoring (periodic)
cost_stats = {
    "current_hourly_cost": 50.0,
    "baseline_hourly_cost": 15.0,
    "daily_cost": 200.0
}

cost_alerts = monitor.check_cost_risks(
    cost_stats=cost_stats,
    tenant_id="my-tenant"
)

# Memory risk monitoring
memory_stats = {
    "memory_usage_mb": 1200,
    "context_length": 45000,
    "response_time_ms": 5000
}

memory_alerts = monitor.check_memory_risks(
    memory_stats=memory_stats,
    tenant_id="my-tenant",
    session_id="sess_123"
)
```

### Escalation Logic
Automatic escalation based on event frequency:

```python
# Check if risk needs escalation
escalation_alert = monitor.check_escalation_needed(
    risk_id="provider_outage",
    tenant_id="my-tenant"
)

if escalation_alert:
    print(f"ESCALATION REQUIRED: {escalation_alert.message}")
    # Trigger incident response
```

## Mitigation Strategies

### Built-in Strategies

#### FallbackProviderStrategy
Switches to alternative provider when primary fails:

```python
# Automatic provider failover
# openai -> anthropic
# anthropic -> openai
# azure -> openai
```

#### CircuitBreakerStrategy
Implements circuit breaker pattern:

```python
# Circuit states:
# - CLOSED: Normal operation
# - OPEN: Failures detected, traffic blocked
# - HALF_OPEN: Testing if service recovered
```

#### RateLimitingStrategy
Applies rate limits to control costs:

```python
# Reduces request rate by 50% when cost spike detected
# Configurable rate limits per tenant/agent
```

#### ModelDowngradeStrategy
Switches to cheaper models:

```python
# Cost reduction mappings:
# gpt-4 -> gpt-3.5-turbo (80% cost reduction)
# claude-3-opus -> claude-3-haiku (83% cost reduction)
```

#### MemoryPruningStrategy
Optimizes memory usage:

```python
# Memory optimization:
# - Remove oldest conversation turns
# - Compress middle sections
# - Preserve recent context
```

#### InputSanitizationStrategy
Sanitizes potentially dangerous input:

```python
# Security measures:
# - Remove code execution patterns
# - Redact PII (SSN, credit cards, emails)
# - Escape dangerous characters
```

#### SessionRestartStrategy
Restarts stalled sessions:

```python
# Convergence optimization:
# - Save critical context
# - Apply fresh parameters
# - Restart with tie-breaker agent
```

### Custom Strategies
Implement custom mitigation strategies:

```python
from agentnet.risk.mitigation import MitigationStrategy, MitigationResult

class CustomMitigationStrategy(MitigationStrategy):
    def __init__(self):
        super().__init__("custom_strategy")
    
    def can_handle(self, risk_event):
        return risk_event.risk_id == "custom_risk"
    
    def execute(self, risk_event, context):
        # Implement custom mitigation logic
        try:
            # Your mitigation code here
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message="Custom mitigation successful",
                details={"action": "custom_action_taken"},
                timestamp=datetime.now()
            )
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Custom mitigation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )

# Register custom strategy
mitigation_engine.register_strategy(CustomMitigationStrategy())
```

## Integration Examples

### With Cost Tracking
Risk management integrates with cost tracking:

```python
from agentnet.core.cost.recorder import CostRecorder
from agentnet.core.cost.analytics import SpendAlertEngine

# Cost-based risk detection
cost_recorder = CostRecorder()
spend_alerts = SpendAlertEngine(cost_recorder).check_spend_velocity(
    tenant_id="my-tenant"
)

# Convert cost alerts to risk events
for alert in spend_alerts:
    registry.register_risk_event(
        risk_id="cost_spike",
        description=f"Cost spike detected: {alert.message}",
        context=alert.details,
        severity=RiskLevel.HIGH,
        tenant_id=alert.tenant_id
    )
```

### With CI/CD Pipeline
Automated risk monitoring in deployments:

```yaml
# .github/workflows/risk-monitoring.yml
- name: Pre-deployment risk check
  run: |
    python -c "
    from agentnet.risk import RiskRegistry, RiskLevel
    from datetime import datetime, timedelta
    
    registry = RiskRegistry()
    
    # Check for critical unresolved risks
    critical_events = registry.get_risk_events(
        severity=RiskLevel.CRITICAL,
        resolved=False,
        start_date=datetime.now() - timedelta(hours=24)
    )
    
    if critical_events:
        print('DEPLOYMENT BLOCKED: Critical risks detected')
        for event in critical_events:
            print(f'- {event.risk_id}: {event.description}')
        exit(1)
    else:
        print('✓ No critical risks - deployment can proceed')
    "
```

### With Multi-tenant Systems
Tenant-specific risk management:

```python
# Per-tenant risk monitoring
tenants = ["tenant-a", "tenant-b", "tenant-c"]

for tenant_id in tenants:
    # Get tenant-specific risk summary
    summary = registry.get_risk_summary(
        tenant_id=tenant_id,
        days_back=7
    )
    
    # Check unresolved risks
    unresolved = registry.get_risk_events(
        tenant_id=tenant_id,
        resolved=False
    )
    
    print(f"Tenant {tenant_id}: {len(unresolved)} unresolved risks")
    
    # Apply tenant-specific mitigation policies
    for event in unresolved:
        if event.severity == RiskLevel.CRITICAL:
            results = mitigation_engine.mitigate_risk(
                risk_event=event,
                context=event.context,
                auto_execute=True
            )
```

## Reporting and Analytics

### Risk Summary
Get comprehensive risk overview:

```python
summary = registry.get_risk_summary(tenant_id="my-tenant", days_back=30)

print(f"Risk Summary (30 days):")
print(f"  Total events: {summary['total_events']}")
print(f"  Resolution rate: {summary['resolution_rate']:.1%}")
print(f"  By category:")
for category, count in summary['by_category'].items():
    print(f"    {category}: {count}")
print(f"  By severity:")
for severity, count in summary['by_severity'].items():
    print(f"    {severity}: {count}")
```

### Mitigation Effectiveness
Track mitigation performance:

```python
mitigation_summary = mitigation_engine.get_mitigation_summary(days_back=7)

print(f"Mitigation Summary (7 days):")
print(f"  Total mitigations: {mitigation_summary['total_mitigations']}")
print(f"  Success rate: {mitigation_summary['success_rate']:.1%}")
print(f"  Cost impact: ${mitigation_summary['total_cost_impact']:.2f}")
print(f"  Performance impact: {mitigation_summary['avg_performance_impact']:.1%}")

for strategy, stats in mitigation_summary['by_strategy'].items():
    print(f"  {strategy}: {stats['successful']}/{stats['total']} "
          f"({stats['success_rate']:.1%})")
```

### Risk Register Export
Export complete risk register for documentation:

```python
risk_register = registry.export_risk_register()

# Save to file for documentation
import json
with open('risk_register.json', 'w') as f:
    json.dump(risk_register, f, indent=2)

# Generate markdown documentation
def generate_risk_docs(risk_register):
    docs = "# Risk Register\\n\\n"
    
    for risk_id, risk_def in risk_register['risk_definitions'].items():
        docs += f"## {risk_def['name']}\\n\\n"
        docs += f"**ID:** {risk_id}\\n"
        docs += f"**Category:** {risk_def['category']}\\n"
        docs += f"**Impact:** {risk_def['impact']}\\n"
        docs += f"**Likelihood:** {risk_def['likelihood']}\\n\\n"
        docs += f"**Description:** {risk_def['description']}\\n\\n"
        docs += "**Mitigation Strategies:**\\n"
        for strategy in risk_def['mitigation_strategies']:
            docs += f"- {strategy}\\n"
        docs += "\\n"
    
    return docs

markdown_docs = generate_risk_docs(risk_register)
with open('RISK_REGISTER.md', 'w') as f:
    f.write(markdown_docs)
```

## Configuration

### Storage Configuration
Configure risk data storage:

```python
# Default storage in data/risk_logs/
registry = RiskRegistry(storage_dir="data/risk_logs")

# Custom storage location
registry = RiskRegistry(storage_dir="/app/persistent/risks")

# Risk events stored in daily files:
# risk_events_2024-01-15.jsonl
```

### Detection Rules
Customize risk detection rules:

```python
# Modify existing risk definitions
provider_risk = registry.risk_definitions["provider_outage"]
provider_risk.detection_rules["consecutive_failures"] = 5  # Increase threshold
provider_risk.detection_rules["error_rate_threshold"] = 0.7  # More tolerant

# Add custom detection rules
custom_detection_rules = {
    "response_time_threshold_ms": 10000,
    "queue_depth_threshold": 100,
    "custom_metric_threshold": 0.95
}

provider_risk.detection_rules.update(custom_detection_rules)
```

### Mitigation Configuration
Configure mitigation behavior:

```python
# Enable/disable auto-mitigation for specific risks
registry.risk_definitions["provider_outage"].auto_mitigation = True
registry.risk_definitions["policy_false_positive"].auto_mitigation = False

# Adjust escalation thresholds
registry.risk_definitions["cost_spike"].escalation_threshold = 5
registry.risk_definitions["tool_injection"].escalation_threshold = 1  # Immediate
```

## Best Practices

### Risk Definition
1. **Clear Naming**: Use descriptive, consistent risk IDs
2. **Comprehensive Detection**: Define multiple detection criteria
3. **Appropriate Severity**: Match severity to business impact
4. **Testable Rules**: Ensure detection rules can be validated

### Monitoring
1. **Real-time Critical**: Monitor security risks in real-time
2. **Periodic Non-critical**: Batch process performance risks
3. **Context Preservation**: Include relevant context in events
4. **Tenant Isolation**: Maintain tenant-specific monitoring

### Mitigation
1. **Graceful Degradation**: Prefer degraded service over failure
2. **Cost Awareness**: Consider mitigation costs vs. risk impact
3. **Rollback Capability**: Ensure mitigations can be reversed
4. **Documentation**: Document mitigation side effects

### Operations
1. **Regular Review**: Periodically review risk definitions
2. **Threshold Tuning**: Adjust detection thresholds based on data
3. **Mitigation Testing**: Test mitigation strategies regularly
4. **Incident Learning**: Update risks based on incidents

## Security Considerations

### Data Protection
- **Encryption**: Risk event data encrypted at rest
- **Access Control**: Role-based access to risk management
- **Audit Trail**: All risk operations logged
- **Retention Policy**: Configurable data retention periods

### Sensitive Information
- **Context Sanitization**: Remove sensitive data from risk contexts
- **PII Handling**: Proper handling of personally identifiable information
- **Secret Management**: Secure handling of API keys and tokens
- **Compliance**: Meet regulatory requirements (GDPR, HIPAA, etc.)

## Performance Optimization

### Query Optimization
```python
# Use date filters for better performance
from datetime import datetime, timedelta

recent_events = registry.get_risk_events(
    start_date=datetime.now() - timedelta(days=7),  # Last week only
    tenant_id="my-tenant"
)
```

### Batch Processing
```python
# Process multiple tenants efficiently
tenant_summaries = {}
for tenant_id in tenant_list:
    tenant_summaries[tenant_id] = registry.get_risk_summary(
        tenant_id=tenant_id,
        days_back=7
    )
```

### Caching
```python
import functools
from datetime import datetime

@functools.lru_cache(maxsize=100)
def cached_risk_summary(tenant_id, date_key):
    """Cache risk summaries for 1 hour."""
    return registry.get_risk_summary(tenant_id=tenant_id)

# Use with hourly cache key
date_key = datetime.now().strftime("%Y-%m-%d-%H")
summary = cached_risk_summary("my-tenant", date_key)
```

## Troubleshooting

### Common Issues

#### High False Positive Rate
```python
# Adjust detection thresholds
risk_def = registry.risk_definitions["policy_false_positive"]
risk_def.detection_rules["false_positive_rate"] = 0.15  # More tolerant

# Review historical events
events = registry.get_risk_events(
    risk_id="policy_false_positive",
    resolved=True
)
# Analyze patterns to improve detection
```

#### Mitigation Failures
```python
# Check mitigation results
results = mitigation_engine.get_mitigation_summary(days_back=1)
failed_strategies = [
    strategy for strategy, stats in results['by_strategy'].items()
    if stats['success_rate'] < 0.8
]

# Review failure patterns
for strategy in failed_strategies:
    print(f"Strategy {strategy} has low success rate")
    # Implement strategy improvements
```

#### Performance Issues
```python
# Monitor query performance
import time

start_time = time.time()
events = registry.get_risk_events(tenant_id="my-tenant")
duration = time.time() - start_time

if duration > 1.0:  # Slow query
    print(f"Slow risk query: {duration:.3f}s")
    # Consider adding date filters or indexing
```

### Debugging Tools
```python
# Enable debug logging
import logging
logging.getLogger("agentnet.risk").setLevel(logging.DEBUG)

# Health check function
def risk_system_health():
    health = {
        "registry": False,
        "monitor": False,
        "mitigation": False
    }
    
    try:
        # Test registry
        test_event = registry.register_risk_event(
            risk_id="health_check",
            description="System health check test",
            context={"test": True}
        )
        health["registry"] = test_event is not None
        
        # Test monitor
        alerts = monitor.get_all_active_alerts()
        health["monitor"] = isinstance(alerts, list)
        
        # Test mitigation
        summary = mitigation_engine.get_mitigation_summary(days_back=1)
        health["mitigation"] = "total_mitigations" in summary
        
    except Exception as e:
        logging.error(f"Risk system health check failed: {e}")
    
    return health
```