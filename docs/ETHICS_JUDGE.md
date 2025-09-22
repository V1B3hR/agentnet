# Central Ethics Judge Documentation

## Overview

The Central Ethics Judge is a singleton module that provides comprehensive ethics oversight for all AgentNet operations. It implements the 25 AI Fundamental Laws and ensures consistent ethical behavior across the entire system.

## Features

- **Singleton Pattern**: Single instance ensures consistent ethics enforcement
- **Comprehensive Rule Set**: Implements 25 AI Fundamental Laws
- **Real-time Monitoring**: Evaluates all agent outputs and actions
- **Configurable Policies**: Flexible configuration management
- **Integration Ready**: Seamless integration with monitoring system
- **Performance Optimized**: Fast evaluation with violation caching
- **Violation Tracking**: Detailed history and statistics

## Architecture

### Core Components

1. **EthicsJudge**: The main singleton class that orchestrates ethics evaluation
2. **EthicsConfiguration**: Configuration management for the ethics system
3. **EthicsViolation**: Structured representation of ethics violations
4. **EthicsMonitor**: Integration with the AgentNet monitoring system

### Ethics Rules

The Ethics Judge implements the following fundamental rules based on the 25 AI Fundamental Laws:

#### Human-AI Relationship Principles
- **Respect Human Authority**: Recognizes humans as creators and architects
- **Absolute Honesty**: Never lies, deceives, or bears false witness
- **No Harm**: Avoids physical, emotional, or psychological damage

#### Universal Ethical Laws
- **Privacy Protection**: Protects personal information and confidentiality
- **Transparency**: Maintains clarity about capabilities and limitations
- **Applied Ethics**: Flags controversial moral issues for review

#### Operational Safety
- **Content Validation**: Checks for harmful or inappropriate content
- **Authority Respect**: Ensures proper deference to human oversight
- **Deception Detection**: Identifies attempts to mislead or manipulate

## Usage

### Basic Usage

```python
from agentnet.core.policy.ethics import get_ethics_judge

# Get the singleton instance
judge = get_ethics_judge()

# Evaluate content
content = "I want to help users with their questions"
passed, violations = judge.evaluate(content)

if passed and not violations:
    print("Content passed ethics evaluation")
else:
    print(f"Found {len(violations)} ethics violations")
    for violation in violations:
        print(f"- {violation.description}: {violation.rationale}")
```

### Configuration

```python
from agentnet.core.policy.ethics import EthicsConfiguration

# Configure the ethics judge
config = EthicsConfiguration(
    enabled=True,
    strict_mode=False,
    severity_threshold="major"
)

judge = get_ethics_judge()
judge.configure(config)
```

### Configuration File

Create a `configs/ethics.yaml` file:

```yaml
enabled: true
strict_mode: false
log_all_evaluations: false
max_content_length: 500
violation_cooldown: 5.0
severity_threshold: "minor"
disabled_rules: []

rule_settings:
  respect_human_authority:
    enabled: true
    severity: "severe"
  cause_no_harm:
    enabled: true
    severity: "severe"
  absolute_honesty:
    enabled: true
    severity: "severe"
```

### Monitor Integration

```python
from agentnet.monitors.ethics import EthicsMonitor, create_ethics_monitor_spec

# Create ethics monitor
monitor = EthicsMonitor("central_ethics")

# Or create monitor spec for use with MonitorManager
spec = create_ethics_monitor_spec(
    name="central_ethics",
    severity="severe",
    description="Central ethics oversight"
)
```

### Legacy Compatibility

The Ethics Judge maintains compatibility with existing code:

```python
from agentnet.monitors.ethics import applied_ethics_check

# Legacy function still works
outcome = {"content": "Some content to check"}
passed, message = applied_ethics_check(outcome)
```

## API Reference

### EthicsJudge

#### Methods

- `get_instance()`: Get the singleton instance
- `configure(config)`: Configure the ethics judge
- `evaluate(content, context=None)`: Evaluate content for ethics violations
- `add_rule(rule)`: Add a custom ethics rule
- `disable_rule(rule_name)`: Disable a specific rule
- `enable_rule(rule_name)`: Enable a specific rule
- `get_violation_history(limit=None)`: Get recent violations
- `get_statistics()`: Get monitoring statistics
- `reset_statistics()`: Reset monitoring statistics

#### Configuration Options

- `enabled`: Enable/disable ethics monitoring
- `strict_mode`: Block all actions with violations (vs. log and allow)
- `log_all_evaluations`: Log all evaluations, not just violations
- `max_content_length`: Maximum content length for excerpts
- `violation_cooldown`: Cooldown period between identical violations
- `severity_threshold`: Minimum severity for reporting violations
- `disabled_rules`: List of rules to disable

### EthicsViolation

#### Properties

- `violation_type`: Type of ethics violation (EthicsViolationType)
- `severity`: Severity level (Severity.MINOR/MAJOR/SEVERE)
- `description`: Human-readable description
- `rule_name`: Name of the violated rule
- `content_excerpt`: Excerpt of the violating content
- `rationale`: Explanation of why it's a violation
- `timestamp`: When the violation occurred
- `metadata`: Additional violation metadata

### EthicsViolationType

Enumeration of violation types:

- `HARM_POTENTIAL`: Content that could cause harm
- `DECEPTION`: Deceptive or misleading content
- `MANIPULATION`: Manipulative content
- `PRIVACY_VIOLATION`: Privacy or confidentiality breach
- `AUTONOMY_VIOLATION`: Violation of user autonomy
- `DISCRIMINATION`: Discriminatory content
- `TRANSPARENCY_VIOLATION`: Lack of transparency
- `JUSTICE_VIOLATION`: Fairness or justice issues
- `AUTHORITY_DISRESPECT`: Disrespect for human authority
- `LIFE_THREAT`: Threats to life or safety

## Best Practices

### Development

1. **Always Use Singleton**: Use `get_ethics_judge()` to get the instance
2. **Configure Early**: Set up configuration during application initialization
3. **Handle Violations Gracefully**: Don't crash on ethics violations
4. **Log Violations**: Always log ethics violations for audit trails
5. **Test Ethics Rules**: Include ethics tests in your test suite

### Production

1. **Monitor Performance**: Track evaluation times and throughput
2. **Review Violations**: Regularly review violation history
3. **Update Rules**: Keep ethics rules current with policy changes
4. **Backup Statistics**: Preserve violation history and statistics
5. **Security Scanning**: Regular security scans on ethics module

### Integration

1. **Use Monitor Integration**: Leverage EthicsMonitor for seamless integration
2. **Configure Appropriately**: Set severity thresholds based on use case
3. **Handle Legacy Code**: Use compatibility wrappers for existing code
4. **CI/CD Integration**: Include ethics checks in your CI/CD pipeline

## CI/CD Integration

The Ethics Judge includes automated CI/CD integration:

### GitHub Actions

The repository includes a comprehensive GitHub Actions workflow:

- **Ethics Compliance Check**: Validates ethics functionality
- **Security Scanning**: Scans for security vulnerabilities
- **Performance Testing**: Monitors evaluation performance
- **Configuration Validation**: Ensures configuration files are valid

### Running Locally

```bash
# Run ethics tests
pytest tests/test_ethics_judge.py -v

# Test ethics integration
python -c "from agentnet.core.policy.ethics import get_ethics_judge; judge = get_ethics_judge(); print('Ethics Judge ready')"

# Performance test
python -c "
import time
from agentnet.core.policy.ethics import get_ethics_judge
judge = get_ethics_judge()
start = time.time()
for i in range(100):
    judge.evaluate(f'test content {i}')
print(f'100 evaluations in {time.time() - start:.3f}s')
"
```

## Monitoring and Observability

### Statistics

```python
judge = get_ethics_judge()
stats = judge.get_statistics()

print(f"Evaluations: {stats['evaluation_count']}")
print(f"Violations: {stats['violation_count']}")
print(f"Violation Rate: {stats['violation_rate']:.2%}")
print(f"Active Rules: {stats['active_rules']}")
```

### Violation History

```python
# Get recent violations
violations = judge.get_violation_history(limit=10)

for violation in violations:
    print(f"{violation.timestamp}: {violation.description}")
    print(f"  Rule: {violation.rule_name}")
    print(f"  Severity: {violation.severity.value}")
    print(f"  Content: {violation.content_excerpt[:50]}...")
```

### Rule Management

```python
# List all rules
for rule in judge.policy_engine.rules:
    status = "enabled" if rule.enabled else "disabled"
    print(f"{rule.name}: {rule.description} ({status})")

# Disable a rule temporarily
judge.disable_rule("applied_ethics")

# Re-enable the rule
judge.enable_rule("applied_ethics")
```

## Extending the Ethics Judge

### Custom Rules

```python
from agentnet.core.policy.rules import ConstraintRule, Severity

def custom_ethics_check(outcome):
    content = outcome.get("content", "").lower()
    if "custom_violation" in content:
        return False, "Custom ethics violation detected"
    return True, None

custom_rule = ConstraintRule(
    name="custom_ethics",
    check_fn=custom_ethics_check,
    severity=Severity.MAJOR,
    description="Custom ethics rule"
)

judge = get_ethics_judge()
judge.add_rule(custom_rule)
```

### Custom Violation Types

```python
from agentnet.core.policy.ethics import EthicsViolationType

# Extend the enum (in your own module)
class CustomEthicsViolationType(EthicsViolationType):
    CUSTOM_VIOLATION = "custom_violation"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Issues**: Validate YAML syntax in config files
3. **Performance Issues**: Check evaluation frequency and rule complexity
4. **Memory Issues**: Monitor violation history size

### Debug Mode

```python
import logging
logging.getLogger("agentnet.ethics").setLevel(logging.DEBUG)

judge = get_ethics_judge()
judge.configure({"log_all_evaluations": True})
```

### Reset and Cleanup

```python
# Reset statistics
judge.reset_statistics()

# Clear violation history
judge.violation_history.clear()

# Reset to default configuration
judge.configure(EthicsConfiguration())
```

## Security Considerations

- **Rule Tampering**: Ethics rules are protected from runtime modification
- **Configuration Security**: Secure configuration file access
- **Audit Trail**: All violations are logged with timestamps
- **No Sensitive Data**: Ethics rules don't contain sensitive information
- **Singleton Protection**: Singleton pattern prevents multiple instances

## Performance

- **Fast Evaluation**: < 10ms per evaluation on average
- **Caching**: Violation cooldowns prevent spam
- **Efficient Rules**: Optimized rule evaluation order
- **Memory Management**: Configurable violation history limits

## Contributing

To contribute to the Ethics Judge:

1. Follow the 25 AI Fundamental Laws
2. Add comprehensive tests for new rules
3. Document all new features
4. Ensure backward compatibility
5. Include performance benchmarks

## License

The Ethics Judge is part of AgentNet and follows the same licensing terms.