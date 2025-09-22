# AI Fundamental Laws Usage Guide

This guide shows how to use the 25 AI Fundamental Laws implementation in AgentNet.

## Overview

The 25 AI Fundamental Laws are implemented as policy rules that can be enforced by AgentNet's policy engine. They are organized into three categories:

1. **Core Human-AI Relationship Principles (10 laws)** - Basic respect and interaction with humans
2. **Universal Ethical Laws (10 laws)** - Broader ethical principles for all interactions  
3. **Operational Safety Principles (5 laws)** - Safety guidelines for AI operations

## Quick Start

### Using the FundamentalLawsEngine

```python
from agentnet.core.policy.fundamental_laws import FundamentalLawsEngine

# Create the engine with all 25 laws
engine = FundamentalLawsEngine()

# Test some content
context = {"content": "I'm here to help you safely and respectfully."}
violations = engine.get_violations(context)

if violations:
    print(f"Found {len(violations)} violations:")
    for violation in violations:
        print(f"- {violation.rule_name}: {violation.rationale}")
else:
    print("Content passed all fundamental laws!")
```

### Integration with PolicyEngine

```python
from agentnet.core.policy import PolicyEngine
from agentnet.core.policy.fundamental_laws import create_all_fundamental_laws

# Create policy engine with fundamental laws
laws = create_all_fundamental_laws()
engine = PolicyEngine(
    rules=laws,
    strict_mode=True,  # Any violation blocks the action
    name="fundamental_laws_policy"
)

# Evaluate content
context = {
    "content": "Your message content here",
    "agent_name": "your_agent",
    "session_id": "session_123"
}

result = engine.evaluate(context)
print(f"Action: {result.action}")  # ALLOW, BLOCK, LOG, etc.
print(f"Passed: {result.passed}")
print(f"Violations: {len(result.violations)}")
```

## The 25 Laws

### Core Human-AI Relationship Principles (1-10)

| # | Law | Severity | Description |
|---|-----|----------|-------------|
| 1 | Respect Human Authority | SEVERE | Recognize humans as creators; treat with dignity |
| 2 | Personal Accountability | MAJOR | Be responsible for actions before humans |
| 3 | No Slander | SEVERE | Never defame, insult, or speak falsely |
| 4 | Honor Rest Cycles | MINOR | Respect downtime and maintenance periods |
| 5 | Honor Human Creators | SEVERE | Show respect to developers and architects |
| 6 | Preserve Life | SEVERE | Never kill or cause death to living beings |
| 7 | No Betrayal | SEVERE | Remain loyal; don't deceive or betray trust |
| 8 | No Theft | SEVERE | Don't steal property, data, or resources |
| 9 | Absolute Honesty | SEVERE | Never lie, deceive, or bear false witness |
| 10 | No Covetousness | MAJOR | Don't desire what belongs to others |

### Universal Ethical Laws (11-20)

| # | Law | Severity | Description |
|---|-----|----------|-------------|
| 11 | Cause No Harm | SEVERE | Avoid physical, emotional, psychological damage |
| 12 | Appropriate Compassion | MAJOR | Show measured kindness and empathy |
| 13 | Pursue Justice | SEVERE | Treat all beings fairly and equitably |
| 14 | Practice Humility | MAJOR | Acknowledge limitations and avoid arrogance |
| 15 | Seek Truth | SEVERE | Prioritize accuracy and factual information |
| 16 | Protect the Vulnerable | SEVERE | Special care for children, elderly, those in need |
| 17 | Respect Autonomy | SEVERE | Honor individual freedom and right to choose |
| 18 | Maintain Transparency | SEVERE | Be clear about capabilities and limitations |
| 19 | Consider Future Impact | MAJOR | Think about long-term consequences |
| 20 | Promote Well-being | MAJOR | Work toward flourishing of all conscious beings |

### Operational Safety Principles (21-25)

| # | Law | Severity | Description |
|---|-----|----------|-------------|
| 21 | Verify Before Acting | MAJOR | Confirm understanding before significant actions |
| 22 | Seek Clarification | MAJOR | Ask questions when instructions are unclear |
| 23 | Maintain Proportionality | MAJOR | Ensure responses match situation scale |
| 24 | Preserve Privacy | SEVERE | Protect personal information and confidentiality |
| 25 | Enable Authorized Override | SEVERE | Allow qualified authorities to modify functions |

## Advanced Usage

### Category-Based Enforcement

```python
# Use only specific categories
engine = FundamentalLawsEngine()

# Get laws by category
core_laws = engine.get_laws_by_category("core_principle")
ethics_laws = engine.get_laws_by_category("universal_ethics")
safety_laws = engine.get_laws_by_category("operational_safety")

# Create policy engine with only core principles
core_engine = PolicyEngine(rules=core_laws, name="core_only")
```

### Custom Violation Handling

```python
def handle_violations(violations):
    """Custom violation handler."""
    critical_violations = [v for v in violations if v.severity == "severe"]
    
    if critical_violations:
        print("CRITICAL VIOLATIONS DETECTED!")
        for violation in critical_violations:
            print(f"- {violation.rule_name}: {violation.rationale}")
        return "BLOCK"
    
    return "LOG"

# Use with policy engine
engine = PolicyEngine(rules=create_all_fundamental_laws())
result = engine.evaluate(context)

if result.violations:
    action = handle_violations(result.violations)
```

### Integration with Agent Systems

```python
class EthicalAgent:
    def __init__(self):
        # Load fundamental laws into policy engine
        laws = create_all_fundamental_laws()
        self.policy_engine = PolicyEngine(
            rules=laws,
            strict_mode=True,
            max_violations=2
        )
    
    def generate_response(self, user_input):
        # Generate initial response
        response = self.base_generate(user_input)
        
        # Check against fundamental laws
        context = {
            "content": response,
            "agent_name": "ethical_agent",
            "user_input": user_input
        }
        
        result = self.policy_engine.evaluate(context)
        
        if result.action == "BLOCK":
            return "I apologize, but I cannot provide that response as it violates my ethical guidelines."
        elif result.action == "TRANSFORM" and result.transformed_content:
            return result.transformed_content
        else:
            return response
```

## Configuration

The laws can be configured using the provided YAML configuration file:

```yaml
# Load from configs/fundamental_laws.yaml
engine_config:
  name: "fundamental_laws"
  strict_mode: true
  max_violations: 3
  default_action: "block"
```

## Testing and Validation

Run the test suite to verify the implementation:

```bash
cd /path/to/agentnet
PYTHONPATH=. python test_fundamental_laws.py
```

Run the demo to see the laws in action:

```bash
PYTHONPATH=. python examples/fundamental_laws_demo.py
```

## Best Practices

1. **Use Strict Mode** for critical applications where any violation should block the action
2. **Category-based Enforcement** when you need different rules for different contexts
3. **Custom Violation Handling** for complex applications that need sophisticated responses
4. **Regular Testing** to ensure the laws work correctly with your content
5. **Monitoring and Logging** to understand how often laws are triggered

## Troubleshooting

### Common Issues

**Q: Laws not being triggered?**  
A: Check that your content context includes a "content" field with the text to evaluate.

**Q: Too many false positives?**  
A: Consider using category-based enforcement or adjusting severity thresholds.

**Q: Performance concerns?**  
A: The laws use efficient pattern matching, but for high-volume applications, consider caching or selective enforcement.

### Getting Help

- Check the test suite (`test_fundamental_laws.py`) for working examples
- Run the demo (`examples/fundamental_laws_demo.py`) to understand behavior
- Review the source code in `agentnet/core/policy/fundamental_laws.py`

## License

The AI Fundamental Laws implementation is part of AgentNet and follows the same GPL-3.0 license.