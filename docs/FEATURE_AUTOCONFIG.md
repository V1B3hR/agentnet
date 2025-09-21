# AutoConfig Feature Documentation

## Overview

AutoConfig is a dynamic parameter adaptation system that automatically adjusts scenario parameters based on task difficulty. It analyzes the complexity of tasks and configures appropriate settings for rounds, reasoning depth, and confidence thresholds to optimize performance for different types of workloads.

## Features

- **Automatic Task Difficulty Analysis**: Uses linguistic analysis to classify tasks as Simple, Medium, or Hard
- **Dynamic Parameter Adjustment**: Adapts rounds, depth hints, and confidence thresholds based on difficulty
- **Observability Integration**: Records auto-configuration decisions in session data
- **Backward Compatibility**: Can be disabled with `metadata.auto_config: false`
- **Confidence Threshold Preservation**: Never lowers user-specified confidence thresholds

## Configuration Matrix

| Task Difficulty | Rounds | Depth | Confidence Threshold | Confidence Adjustment |
|----------------|--------|-------|---------------------|----------------------|
| Simple         | 3      | 2     | 0.6                | -0.1 (lower)         |
| Medium         | 4      | 3     | 0.7                | 0.0 (keep)           |
| Hard           | 5      | 4     | 0.8                | +0.1 (raise)         |

## Task Difficulty Classification

### Simple Tasks
- Direct questions: "What is AI?", "Define machine learning"
- Simple commands: "List benefits", "Name three colors"
- Basic selections: "Choose yes or no", "Select the correct answer"

### Medium Tasks  
- Explanatory requests: "Explain microservices architecture"
- Comparative analysis: "Compare SQL and NoSQL databases"
- Process descriptions: "Describe CI/CD implementation"
- Planning tasks: "Outline a collaboration plan"

### Hard Tasks
- Comprehensive frameworks: "Develop ethical AI decision-making framework"
- Complex analysis: "Analyze distributed system architecture trade-offs"
- Research synthesis: "Synthesize findings into sophisticated methodology"
- Multi-stakeholder considerations: "Evaluate policy implications considering multiple perspectives"

## Usage Examples

### Basic Usage (Auto-enabled)

```python
from agentnet import AgentNet, ExampleEngine

# AutoConfig is enabled by default
agent = AgentNet("MyAgent", {"logic": 0.8}, ExampleEngine())

# Simple task - gets 3 rounds, depth 2, confidence 0.6
result = agent.generate_reasoning_tree("What is Python?")

# Hard task - gets 5 rounds, depth 4, confidence 0.8  
result = agent.generate_reasoning_tree(
    "Develop a comprehensive framework for ethical AI governance"
)
```

### Explicit Control

```python
# Enable AutoConfig explicitly
result = agent.generate_reasoning_tree(
    "Complex analysis task",
    metadata={"auto_config": True}
)

# Disable AutoConfig for manual control
result = agent.generate_reasoning_tree(
    "Complex analysis task", 
    metadata={"auto_config": False},
    max_depth=6,
    confidence_threshold=0.9
)
```

### Multi-Party Dialogue

```python
# AutoConfig applies to dialogue rounds
session = await agent.async_multi_party_dialogue(
    agents=[agent1, agent2],
    topic="Ethical AI decision-making framework",  # Hard task
    # Will automatically use 5 rounds instead of default
    metadata={"auto_config": True}
)
```

### Confidence Threshold Preservation

```python
# High user-specified threshold is preserved
result = agent.generate_reasoning_tree(
    "Simple task",
    confidence_threshold=0.95,  # User wants high confidence
    metadata={"auto_config": True}
)
# Result uses 0.95, not the auto-configured 0.6
```

## Observability

AutoConfig decisions are recorded in session data for analysis:

```python
result = agent.generate_reasoning_tree("Complex task")

# Check auto-configuration decisions
autoconfig_data = result["autoconfig"]
print(f"Difficulty: {autoconfig_data['difficulty']}")
print(f"Configured rounds: {autoconfig_data['configured_rounds']}")
print(f"Configured depth: {autoconfig_data['configured_max_depth']}")
print(f"Confidence threshold: {autoconfig_data['configured_confidence_threshold']}")
print(f"Reasoning: {autoconfig_data['reasoning']}")
```

Example output:
```json
{
  "autoconfig": {
    "difficulty": "hard",
    "configured_rounds": 5,
    "configured_max_depth": 4,
    "configured_confidence_threshold": 0.8,
    "reasoning": "Task classified as HARD due to complexity indicators. Using enhanced configuration: 5 rounds, depth 4, confidence 0.8",
    "confidence_adjustment": 0.1,
    "enabled": true
  }
}
```

## Advanced Configuration

### Custom Context

```python
# Provide additional context for difficulty analysis
result = agent.generate_reasoning_tree(
    "Implement solution",
    metadata={
        "auto_config": True,
        "domain": "technical research",  # Boosts difficulty
        "confidence": 0.3  # Low confidence suggests complexity
    }
)
```

### Global AutoConfig Management

```python
from agentnet.core.autoconfig import get_global_autoconfig, set_global_autoconfig, AutoConfig

# Get current global instance
autoconfig = get_global_autoconfig()

# Analyze task difficulty directly
difficulty = autoconfig.analyze_task_difficulty("Complex analysis task")
print(f"Task difficulty: {difficulty}")

# Configure scenario manually
params = autoconfig.configure_scenario("Complex task")
print(f"Recommended rounds: {params.rounds}")

# Replace global instance with custom configuration
custom_autoconfig = AutoConfig()
set_global_autoconfig(custom_autoconfig)
```

## Integration Points

### Core Agent Methods
- `generate_reasoning_tree()` - Applies depth and confidence adjustments
- `generate_reasoning_tree_enhanced()` - Enhanced version with memory integration
- `async_generate_reasoning_tree()` - Async wrapper with same functionality

### Legacy AgentNet
- `async_multi_party_dialogue()` - Applies round adjustments for dialogue sessions
- Session records include `autoconfig` observability data

### Observability Systems
- Dashboard data collectors receive autoconfig metrics
- Session metrics include difficulty classifications and parameter adjustments

## Algorithm Details

### Difficulty Scoring System

```python
difficulty_score = 0

# Complexity indicators (weight: 3 each)
difficulty_score += hard_indicator_count * 3

# Medium indicators (weight: 1 each)  
difficulty_score += medium_indicator_count * 1

# Simple indicators (weight: -1 each)
difficulty_score -= simple_indicator_count * 1

# Length complexity
if word_count > 50: difficulty_score += 2
elif word_count > 20: difficulty_score += 1

# Sentence complexity
if sentence_count > 3: difficulty_score += 1

# Punctuation complexity
if "?" in task: difficulty_score += 0.5
if comma_count > 2: difficulty_score += 0.5
if ";" or ":" in task: difficulty_score += 0.5

# Context adjustments
if confidence < 0.5: difficulty_score += 1
if domain in ["technical", "research", "policy"]: difficulty_score += 1

# Classification
if difficulty_score >= 5: return HARD
elif difficulty_score >= 2: return MEDIUM
else: return SIMPLE
```

### Parameter Selection Logic

1. **Rounds**: Use base rounds or auto-configured (whichever is higher)
2. **Max Depth**: Use base depth or auto-configured (whichever is higher)  
3. **Confidence Threshold**: 
   - If base is default (0.7): Use auto-configured value
   - If base is user-specified: Preserve or raise (never lower)

## Best Practices

### When to Use AutoConfig
- ✅ Variable task complexity in production
- ✅ Want automatic optimization without manual tuning
- ✅ Need observability into parameter decisions
- ✅ Prototyping and experimentation

### When to Disable AutoConfig  
- ❌ Need exact parameter control for benchmarking
- ❌ Performance-critical applications with strict requirements
- ❌ Tasks with domain-specific complexity not captured by linguistic analysis
- ❌ Custom parameter scheduling based on external factors

### Performance Considerations
- AutoConfig analysis adds minimal overhead (~1-2ms per task)
- Linguistic analysis is deterministic and cacheable
- No external dependencies or network calls
- Memory footprint is negligible

## Troubleshooting

### Common Issues

**Issue**: AutoConfig not applying parameters
```python
# Check if auto_config is disabled
result = agent.generate_reasoning_tree(task, metadata={"auto_config": True})
assert "autoconfig" in result
```

**Issue**: Confidence threshold not being lowered
```python
# Default threshold (0.7) prevents lowering - this is intended behavior
# Use explicit None to allow auto-config full control
result = agent.generate_reasoning_tree(task, confidence_threshold=None)
```

**Issue**: Task classified incorrectly
```python
# Provide additional context to influence classification
result = agent.generate_reasoning_tree(
    task,
    metadata={
        "domain": "technical",  # Boosts difficulty
        "confidence": 0.3       # Suggests complexity
    }
)
```

### Debugging

```python
from agentnet.core.autoconfig import get_global_autoconfig

autoconfig = get_global_autoconfig()

# Analyze task difficulty step by step
task = "Your task here"
difficulty = autoconfig.analyze_task_difficulty(task)
print(f"Classified as: {difficulty}")

# Get full configuration
params = autoconfig.configure_scenario(task)
print(f"Rounds: {params.rounds}")
print(f"Depth: {params.max_depth}")
print(f"Confidence: {params.confidence_threshold}")
print(f"Reasoning: {params.reasoning}")
```

## Version History

- **v0.5.0**: Initial AutoConfig implementation
  - Task difficulty analysis using linguistic indicators
  - Dynamic parameter adjustment for rounds, depth, confidence
  - Observability integration with session data
  - Backward compatibility with metadata.auto_config flag
  - Confidence threshold preservation logic

## Related Features

- [Reasoning Engine](./REASONING.md) - Uses auto-configured depth parameters
- [Dialogue System](./DIALOGUE.md) - Uses auto-configured round parameters  
- [Observability](./OBSERVABILITY.md) - Records autoconfig decisions
- [Session Management](./SESSIONS.md) - Stores autoconfig metadata