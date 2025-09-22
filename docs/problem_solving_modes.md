# Problem-Solving Modes in AgentNet

AgentNet provides integrated problem-solving modes, styles, and techniques that adapt the agent behavior and orchestration parameters to match different types of problems and solution approaches.

## Overview

The problem-solving framework consists of three main components:

1. **Modes** - High-level problem-solving approaches (brainstorm, debate, consensus, workflow, dialogue)
2. **Styles** - Agent behavioral styles (clarifier, ideator, developer, implementor)  
3. **Techniques** - Specific problem-solving techniques (simple, complex, troubleshooting, gap-from-standard, target-state, open-ended)

These components work together with AutoConfig to automatically adapt reasoning parameters, and with flow metrics to measure the effectiveness of the problem-solving process.

## Modes

### Brainstorm Mode
**Purpose**: Generate diverse, novel ideas without premature judgment

**Characteristics**:
- Lower confidence thresholds to encourage exploration
- Focus on idea generation and creative thinking
- Quantity over initial quality filtering
- Built-in creativity prompts

**Use Cases**: Product ideation, solution exploration, creative problem solving

```python
from agentnet.orchestrator.modes import BrainstormStrategy
from agentnet.core.enums import ProblemSolvingStyle, ProblemTechnique

strategy = BrainstormStrategy(
    style=ProblemSolvingStyle.IDEATOR,
    technique=ProblemTechnique.OPEN_ENDED
)

result = strategy.execute(
    agent=agent,
    task="Generate innovative features for mobile app"
)
```

### Debate Mode  
**Purpose**: Structured argumentation and critical analysis

**Characteristics**:
- Higher confidence thresholds for rigorous arguments
- Focus on evidence and logical reasoning
- Built-in counterargument consideration
- Critical evaluation framework

**Use Cases**: Technical reviews, policy analysis, decision evaluation

```python
from agentnet.orchestrator.modes import DebateStrategy

strategy = DebateStrategy(
    style=ProblemSolvingStyle.DEVELOPER,
    technique=ProblemTechnique.COMPLEX
)

result = strategy.execute(
    agent=agent,
    task="Evaluate microservices vs monolithic architecture"
)
```

### Consensus Mode
**Purpose**: Find common ground and shared agreement

**Characteristics**:
- Balanced confidence thresholds
- Focus on convergence and alignment  
- Built-in collaboration prompts
- Shared value identification

**Use Cases**: Team alignment, stakeholder agreement, collaborative planning

```python
from agentnet.orchestrator.modes import ConsensusStrategy

strategy = ConsensusStrategy(
    style=ProblemSolvingStyle.CLARIFIER,
    technique=ProblemTechnique.TARGET_STATE
)

result = strategy.execute(
    agent=agent,
    task="Align team on quarterly objectives"
)
```

### Workflow Mode
**Purpose**: Structured process execution and step-by-step problem solving

**Characteristics**:**
- High confidence thresholds for reliable steps
- Focus on process and sequence
- Built-in validation checkpoints
- Systematic approach emphasis

**Use Cases**: Process design, implementation planning, systematic analysis

```python
from agentnet.orchestrator.modes import WorkflowStrategy

strategy = WorkflowStrategy(
    style=ProblemSolvingStyle.IMPLEMENTOR,
    technique=ProblemTechnique.SIMPLE
)

result = strategy.execute(
    agent=agent,
    task="Design deployment process for new service"
)
```

### Dialogue Mode
**Purpose**: Conversational exploration and interactive problem solving

**Characteristics**:
- Moderate confidence thresholds for natural flow
- Focus on questions and exploration
- Built-in conversational prompts
- Interactive reasoning approach

**Use Cases**: Requirements gathering, exploratory analysis, learning sessions

```python
from agentnet.orchestrator.modes import DialogueStrategy

strategy = DialogueStrategy(
    style=ProblemSolvingStyle.CLARIFIER,
    technique=ProblemTechnique.OPEN_ENDED
)

result = strategy.execute(
    agent=agent,
    task="Explore user needs for new feature"
)
```

## Problem-Solving Styles

### Clarifier
- Focuses on understanding and defining problems clearly
- Asks probing questions and seeks context
- Emphasizes requirements and scope definition

### Ideator  
- Generates creative ideas and possibilities
- Thinks outside the box and explores alternatives
- Emphasizes innovation and novel approaches

### Developer
- Builds on ideas to create structured solutions
- Focuses on design and architecture
- Emphasizes systematic development

### Implementor
- Translates solutions into actionable plans
- Focuses on execution and practical delivery
- Emphasizes concrete steps and implementation

## Problem-Solving Techniques

### Simple
- Straightforward, direct problem-solving approach
- Minimal complexity and clear paths
- Suitable for well-defined problems

### Complex
- Multi-faceted analysis with deep reasoning
- Handles interconnected factors and dependencies
- Suitable for sophisticated problems

### Troubleshooting
- Diagnostic approach to identify and fix issues
- Root cause analysis and systematic debugging
- Suitable for problem resolution scenarios

### Gap-from-Standard
- Compare current state against expected standards
- Identify deviations and compliance issues
- Suitable for quality assurance and auditing

### Target-State
- Define desired end state and work backwards
- Goal-oriented planning and achievement
- Suitable for strategic planning and objectives

### Open-Ended
- Exploratory approach without predetermined outcomes
- Research and discovery orientation
- Suitable for investigation and learning

## Flow Metrics

AgentNet calculates electrical circuit-inspired metrics for each reasoning process:

### Current (I)
**Definition**: `tokens_output / runtime_seconds`
**Meaning**: Information processing rate

### Voltage (V)  
**Definition**: Task intensity/complexity (0-10 scale)
**Sources**: 
1. Explicit metadata voltage setting
2. Technique-based mapping (complex=8.0, simple=3.0, etc.)  
3. AutoConfig difficulty mapping (hard=8.0, medium=5.0, simple=3.0)

### Resistance (R)
**Definition**: `α*policy_hits + β*avg_tool_latency + γ*disagreement_score`
**Components**:
- Policy violations (α=0.4 weight)
- Tool execution latency (β=0.3 weight)  
- Reasoning disagreement/variance (γ=0.3 weight)

### Power (P)
**Definition**: `V × I`
**Meaning**: Overall reasoning effectiveness

```python
# Flow metrics are automatically calculated and attached to results
result = strategy.execute(agent, task)
flow_metrics = result['flow_metrics']

print(f"Current: {flow_metrics['current']:.2f} tokens/sec")
print(f"Voltage: {flow_metrics['voltage']:.2f}")  
print(f"Resistance: {flow_metrics['resistance']:.2f}")
print(f"Power: {flow_metrics['power']:.2f}")
```

## AutoConfig Integration

AutoConfig automatically recommends modes, styles, and techniques based on task analysis:

```python
from agentnet.core.autoconfig import get_global_autoconfig

autoconfig = get_global_autoconfig()
params = autoconfig.configure_scenario(
    task="Design a scalable microservices architecture",
    context={"domain": "technical"}
)

print(f"Recommended mode: {params.recommended_mode}")
print(f"Recommended style: {params.recommended_style}")  
print(f"Recommended technique: {params.recommended_technique}")
```

### Mode Recommendations
- **Brainstorm**: Tasks with "generate", "create", "ideas", "innovative"
- **Debate**: Tasks with "analyze", "evaluate", "compare", "argue"
- **Consensus**: Tasks with "align", "agree", "collaborate", "shared"
- **Workflow**: Tasks with "process", "implement", "steps", "execute"
- **Dialogue**: Tasks with "explore", "discuss", "understand", "clarify"

### Style Recommendations  
- **Clarifier**: Tasks requiring understanding and definition
- **Ideator**: Tasks requiring creativity and idea generation
- **Developer**: Tasks requiring solution design and architecture
- **Implementor**: Tasks requiring execution and delivery

### Technique Recommendations
- **Troubleshooting**: Tasks with "fix", "debug", "problem", "issue"
- **Gap-from-Standard**: Tasks with "compliance", "standard", "deviation"
- **Target-State**: Tasks with "goal", "objective", "achieve", "vision"
- **Open-Ended**: Tasks with "explore", "research", "investigate"
- **Complex/Simple**: Based on AutoConfig difficulty analysis

## Configuration Examples

### YAML Configuration
```yaml
# examples/problem_modes.yaml
modes:
  architecture_review:
    mode: "debate"
    style: "developer"  
    technique: "gap_from_standard"
    parameters:
      max_depth: 4
      confidence_threshold: 0.8
      rounds: 5
    metadata:
      voltage: 7.5
      domain: "technical"
```

### Python Configuration
```python
from agentnet import AgentNet
from agentnet.orchestrator.modes import DebateStrategy
from agentnet.core.enums import ProblemSolvingStyle, ProblemTechnique

# Create agent
agent = AgentNet("TechReviewer", {"logic": 0.8, "creativity": 0.6})

# Configure strategy
strategy = DebateStrategy(
    style=ProblemSolvingStyle.DEVELOPER,
    technique=ProblemTechnique.GAP_FROM_STANDARD
)

# Execute with flow metrics
result = strategy.execute(
    agent=agent,
    task="Review the proposed API design for RESTful best practices",
    max_depth=4,
    confidence_threshold=0.8
)

# Access results and metrics
print("Analysis:", result['result']['content'])
print("Flow metrics:", result['flow_metrics'])
print("Strategy info:", result['strategy'])
```

## Best Practices

### Choosing Modes
1. **Brainstorm** for divergent thinking and idea generation
2. **Debate** for rigorous analysis and evaluation  
3. **Consensus** for alignment and collaborative decisions
4. **Workflow** for structured implementation and processes
5. **Dialogue** for exploration and understanding

### Combining Styles and Techniques
- Match **styles** to the role you want the agent to play
- Match **techniques** to the type of problem structure
- Use **AutoConfig recommendations** as starting points
- Monitor **flow metrics** to optimize configurations

### Performance Optimization
- Higher **voltage** settings for more intensive reasoning
- Lower **resistance** through reduced policy violations and tool latency
- Monitor **current** (tokens/sec) for processing efficiency
- Optimize **power** (V×I) for overall effectiveness

## API Reference

### Core Enums
```python
from agentnet.core.enums import Mode, ProblemSolvingStyle, ProblemTechnique

# Available modes
Mode.BRAINSTORM, Mode.DEBATE, Mode.CONSENSUS, Mode.WORKFLOW, Mode.DIALOGUE

# Available styles  
ProblemSolvingStyle.CLARIFIER, ProblemSolvingStyle.IDEATOR
ProblemSolvingStyle.DEVELOPER, ProblemSolvingStyle.IMPLEMENTOR

# Available techniques
ProblemTechnique.SIMPLE, ProblemTechnique.COMPLEX
ProblemTechnique.TROUBLESHOOTING, ProblemTechnique.GAP_FROM_STANDARD
ProblemTechnique.TARGET_STATE, ProblemTechnique.OPEN_ENDED
```

### Strategy Classes
```python
from agentnet.orchestrator.modes import (
    BrainstormStrategy, DebateStrategy, ConsensusStrategy,
    WorkflowStrategy, DialogueStrategy
)
```

### Flow Metrics
```python
from agentnet.metrics.flow import FlowMetrics, calculate_flow_metrics
```

This integrated approach ensures that AgentNet adapts its reasoning behavior to match the specific requirements of different problem-solving contexts, while providing measurable feedback through flow metrics.