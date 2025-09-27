# AgentNet Message Schema Documentation

## Overview

The AgentNet Message Schema provides a comprehensive JSON contract for agent communications, implementing the specification defined in the AgentNet roadmap. This schema ensures standardized, validated, and traceable message exchange between agents, tools, and system components.

## Schema Components

### Core Message Structure

The `TurnMessage` class represents the primary message format with the following components:

```json
{
  "task_id": "unique-task-identifier",
  "agent": "agent-name", 
  "message_type": "turn|system|user|tool|error",
  "version": "schema-version",
  "input": { ... },
  "output": { ... },
  "monitors": [ ... ],
  "cost": { ... },
  "timing": { ... },
  "metadata": { ... }
}
```

### Input Structure

```json
"input": {
  "prompt": "Agent input prompt",
  "context": {
    "short_term": ["memory items"],
    "semantic_refs": [{"id": "ref1", "score": 0.8}],
    "episodic_refs": [{"id": "ep1", "timestamp": 123456}],
    "additional_context": {}
  },
  "metadata": {}
}
```

### Output Structure

```json
"output": {
  "content": "Agent response content",
  "confidence": 0.87,
  "style_insights": ["Applied logical reasoning"],
  "tokens": {
    "input": 324,
    "output": 512,
    "total": 836
  },
  "reasoning_type": "deductive",
  "metadata": {}
}
```

### Monitoring Structure

```json
"monitors": [
  {
    "name": "safety_check",
    "status": "pass|fail|error|skip",
    "elapsed_ms": 2.1,
    "details": {},
    "violations": []
  }
]
```

### Cost Tracking Structure

```json
"cost": {
  "provider": "openai|anthropic|azure|local|example",
  "model": "gpt-4o",
  "usd": 0.01234,
  "tokens_per_dollar": 500.0,
  "estimated": false
}
```

### Timing Structure

```json
"timing": {
  "started": 1736981000.123,
  "completed": 1736981001.001,
  "latency_ms": 878,
  "breakdown": {
    "inference": 500.0,
    "policy_check": 200.0,
    "memory_lookup": 178.0
  }
}
```

## Usage Examples

### Creating Messages

#### Using MessageFactory

```python
from agentnet.schemas import MessageFactory

# Create a standard turn message
message = MessageFactory.create_turn_message(
    agent_name="MyAgent",
    prompt="Analyze renewable energy benefits",
    content="Renewable energy provides environmental and economic advantages...",
    confidence=0.92,
    input_tokens=50,
    output_tokens=150
)
```

#### From AgentNet Results

```python
from agentnet import AgentNet, ExampleEngine
from agentnet.schemas import MessageFactory

agent = AgentNet("TestAgent", {"logic": 0.8}, engine=ExampleEngine())
result = agent.generate_reasoning_tree("Test prompt")

message = MessageFactory.create_from_agent_result(
    agent_name=agent.name,
    agent_result=result,
    prompt="Test prompt"
)
```

### Validation and Compliance

```python
from agentnet.schemas import MessageSchemaValidator

validator = MessageSchemaValidator()

# Validate message
is_valid = validator.validate_message(message)

# Get compliance report
report = validator.get_schema_compliance_report(message)
print(f"Valid: {report['valid']}")
print(f"Completeness: {report['completeness']:.1%}")
print(f"Warnings: {len(report['warnings'])}")
```

### JSON Serialization

```python
# Convert to JSON
json_string = message.to_json(indent=2)

# Create from JSON
reconstructed = TurnMessage.from_json(json_string)

# Convert to dictionary
message_dict = message.to_dict()

# Create from dictionary
reconstructed_dict = TurnMessage.from_dict(message_dict)
```

### Monitor Management

```python
# Add monitor results
message.add_monitor_result("content_filter", MonitorStatus.PASS, 3.5)
message.add_monitor_result("safety_check", MonitorStatus.FAIL, 1.2,
                          violations=[{"type": "unsafe_content"}])

# Check if successful (no failed monitors)
success = message.is_successful()
```

### Cost and Performance Analysis

```python
# Calculate total cost
total_cost = message.calculate_total_cost()

# Get latency breakdown
breakdown = message.get_latency_breakdown()
print(f"Total latency: {breakdown['total']}ms")
print(f"Inference: {breakdown.get('inference', 0)}ms")
```

## Validation Rules

### Required Fields
- `task_id`: Must be non-empty string (auto-generated if empty)
- `agent`: Agent identifier
- `input`: Input data with prompt
- `output`: Output data with content and confidence
- `timing`: Timing information with consistent timestamps

### Validation Features
- **Auto-correction**: Token totals and latency calculations
- **Range validation**: Confidence scores (0.0-1.0), token counts (≥0)
- **Consistency checks**: Timing relationships, semantic reference structure
- **Extensibility**: Additional fields allowed for custom extensions

## Error Handling

The schema provides robust error handling:

```python
try:
    message = TurnMessage.from_dict(data)
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Schema error: {e}")
```

## Best Practices

### 1. Use Factory Methods
Prefer `MessageFactory` methods for creating standardized messages:

```python
# Good
message = MessageFactory.create_turn_message(...)

# Avoid direct construction unless necessary
message = TurnMessage(task_id=..., agent=..., ...)
```

### 2. Validate Messages
Always validate messages, especially when receiving from external sources:

```python
if not validator.validate_message(message):
    raise ValueError("Invalid message format")
```

### 3. Include Monitoring
Add monitor results for governance and debugging:

```python
message.add_monitor_result("policy_check", MonitorStatus.PASS, elapsed_ms)
```

### 4. Track Costs
Include cost information for resource management:

```python
message.cost = CostModel(
    provider=CostProvider.OPENAI,
    model="gpt-4",
    usd=calculated_cost
)
```

### 5. Provide Context
Include relevant context for better traceability:

```python
message.input.context = ContextModel(
    short_term=recent_memories,
    semantic_refs=relevant_refs
)
```

## Integration Patterns

### With AgentNet Agents

```python
def create_traced_agent_call(agent, prompt):
    start_time = time.time()
    
    result = agent.generate_reasoning_tree(prompt)
    
    message = MessageFactory.create_from_agent_result(
        agent_name=agent.name,
        agent_result=result,
        prompt=prompt
    )
    
    # Add performance monitoring
    message.add_monitor_result("execution", MonitorStatus.PASS, 
                              (time.time() - start_time) * 1000)
    
    return message, result
```

### With Turn Engine

```python
class TracingTurnEngine(TurnEngine):
    def _execute_agent_turn(self, agent, prompt, ...):
        result = super()._execute_agent_turn(agent, prompt, ...)
        
        # Convert to schema message
        message = MessageFactory.create_turn_message(
            agent_name=agent.name,
            prompt=prompt,
            content=result.content,
            confidence=result.confidence
        )
        
        # Store for analysis
        self.store_message(message)
        
        return result
```

## Extensions and Customization

The schema supports extensions through:

1. **Additional metadata fields**
2. **Custom monitor types**
3. **Extended cost providers**
4. **Custom validation rules**

Example custom extension:

```python
class CustomTurnMessage(TurnMessage):
    custom_field: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields
```

## Performance Considerations

- **Memory**: Schema objects are lightweight and efficient
- **Serialization**: JSON operations are optimized for large messages
- **Validation**: Fast pydantic validation with minimal overhead
- **Storage**: Compact JSON representation suitable for logging/storage

## Compliance and Standards

The schema ensures compliance with:

- **AgentNet Roadmap**: Implements complete JSON contract specification
- **JSON Schema**: Valid JSON structure with type safety
- **OpenAPI**: Compatible with API documentation standards
- **Observability**: Full traceability and monitoring support

## Migration Guide

### From Legacy TurnResult

```python
# Legacy
turn_result = TurnResult(
    turn_id="123",
    agent_id="agent",
    content="response"
)

# New Schema
message = MessageFactory.create_turn_message(
    agent_name="agent",
    prompt="original_prompt", 
    content="response",
    confidence=0.8,
    task_id="123"
)
```

### Adding Schema to Existing Code

```python
# Wrap existing agent calls
def schema_aware_agent_call(agent, prompt):
    result = agent.generate_reasoning_tree(prompt)
    
    return MessageFactory.create_from_agent_result(
        agent_name=agent.name,
        agent_result=result,
        prompt=prompt
    )
```

This schema implementation provides a solid foundation for standardized agent communication while maintaining flexibility for future enhancements and integrations.