# Message/Turn Schema Testing Documentation

This document provides comprehensive documentation and examples for AgentNet's message/turn schema implementation and testing.

## Overview

The schema testing suite validates the complete JSON contract implementation for AgentNet's message/turn schema, including validation, serialization, and compliance checking. **Current Status: 15/15 tests passing (100%)**

## Schema Architecture

### Core Schema Components

```python
from agentnet.schemas import (
    TurnMessage,           # Main message container
    MessageType,           # Message type enumeration  
    ContextModel,          # Context information
    InputModel,            # Input data structure
    OutputModel,           # Output data structure
    TokensModel,           # Token usage tracking
    MonitorResultModel,    # Monitor execution results
    CostModel,             # Cost tracking
    TimingModel,           # Timing information
    MessageSchemaValidator, # Validation utilities
    MessageFactory,        # Message creation factory
    create_example_message # Example message generator
)
```

## Usage Examples

### Creating a Turn Message

```python
from agentnet.schemas import MessageFactory
import time

# Method 1: Using MessageFactory
message = MessageFactory.create_turn_message(
    agent_name="ExampleAgent",
    prompt="Analyze the benefits of renewable energy",
    content="Renewable energy offers environmental sustainability...",
    confidence=0.87,
    input_tokens=324,
    output_tokens=512,
    duration=0.878  # seconds
)

# Method 2: Direct instantiation
from agentnet.schemas import TurnMessage, InputModel, OutputModel, TokensModel, TimingModel

start_time = time.time()
end_time = start_time + 0.878

message = TurnMessage(
    task_id="example-task-123",
    agent="ExampleAgent",
    input=InputModel(
        prompt="Analyze the benefits of renewable energy",
        context=ContextModel(
            short_term=["Previous climate discussion"],
            semantic_refs=[{"id": "ref1", "score": 0.83}]
        )
    ),
    output=OutputModel(
        content="Renewable energy offers environmental sustainability...",
        confidence=0.87,
        tokens=TokensModel(input=324, output=512, total=836)
    ),
    timing=TimingModel(
        started=start_time,
        completed=end_time,
        latency_ms=878
    )
)
```

### Adding Monitoring Results

```python
from agentnet.schemas import MonitorStatus

# Add monitor results to existing message
message.add_monitor_result(
    name="keyword_guard",
    status=MonitorStatus.PASS,
    elapsed_ms=2.1,
    details={"keywords_found": 0}
)

message.add_monitor_result(
    name="cost_monitor", 
    status=MonitorStatus.PASS,
    elapsed_ms=1.5,
    violations=[]
)
```

### Adding Cost Information

```python
from agentnet.schemas import CostModel, CostProvider

message.cost = CostModel(
    provider=CostProvider.OPENAI,
    model="gpt-4o",
    usd=0.01234,
    tokens_per_dollar=68000,
    estimated=False
)
```

### Serialization and Deserialization

```python
# Convert to dictionary
message_dict = message.to_dict()

# Convert to JSON string
json_str = message.to_json(indent=2)

# Restore from dictionary
restored_msg = TurnMessage.from_dict(message_dict)

# Restore from JSON
restored_msg = TurnMessage.from_json(json_str)
```

### Schema Validation

```python
from agentnet.schemas import MessageSchemaValidator

validator = MessageSchemaValidator()

# Validate a message
is_valid = validator.validate_message(message)

# Validate JSON string
is_valid_json = validator.validate_json_schema(json_str)

# Get compliance report
report = validator.get_schema_compliance_report(message)
print(f"Compliance: {report['completeness']:.1%}")
print(f"Warnings: {len(report['warnings'])}")
print(f"Errors: {len(report['errors'])}")
```

## Test Coverage Details

### Schema Models ✅ 3/3 Tests

#### Context Model Validation
- **Test**: `test_context_model_validation`
- **Coverage**:
  - Short-term memory structure
  - Semantic reference validation (id + score)
  - Episodic reference handling
  - Additional context data
- **Validation Rules**:
  - Semantic refs must have 'id' and 'score' fields
  - Scores must be numbers between 0-1
  - Proper list/dictionary structures

#### Timing Model Validation  
- **Test**: `test_timing_model_validation`
- **Coverage**:
  - Start/completion timestamp validation
  - Latency calculation accuracy
  - Timing consistency checks
  - Breakdown component tracking
- **Auto-corrections**:
  - Latency adjusted to match timestamp difference
  - Completion time validation against start time

#### Token Model Validation
- **Test**: `test_tokens_model_validation`
- **Coverage**:
  - Input/output token counts
  - Total token auto-calculation
  - Negative value prevention
  - Token efficiency metrics
- **Auto-corrections**:
  - Total automatically calculated as input + output
  - Negative values converted to zero

### TurnMessage Implementation ✅ 5/5 Tests

#### Turn Message Creation
- **Test**: `test_turn_message_creation`
- **Coverage**:
  - Complete message instantiation
  - Field validation and defaults
  - UUID generation for missing task_id
  - Schema version management

#### Turn Message Serialization
- **Test**: `test_turn_message_serialization`
- **Coverage**:
  - JSON serialization/deserialization
  - Dictionary conversion
  - Round-trip data integrity
  - Enum value preservation

#### Monitor Result Management
- **Test**: `test_monitor_result_management`
- **Coverage**:
  - Adding monitor results
  - Monitor status tracking
  - Execution time recording
  - Violation logging

#### Cost Calculation
- **Test**: `test_cost_calculation`
- **Coverage**:
  - Total cost calculation
  - Multi-provider cost tracking
  - Estimated vs actual costs
  - Token efficiency metrics

#### Latency Breakdown
- **Test**: `test_latency_breakdown`
- **Coverage**:
  - Component-wise latency tracking
  - Total latency aggregation
  - Breakdown analysis
  - Performance bottleneck identification

### Message Factory ✅ 2/2 Tests

#### Create Turn Message
- **Test**: `test_create_turn_message`
- **Coverage**:
  - Factory method instantiation
  - Parameter mapping
  - Default value handling
  - Timing precision management
- **Features**:
  - Automatic task ID generation
  - Timing calculation from duration
  - Token total auto-calculation

#### Create from Agent Result
- **Test**: `test_create_from_agent_result`
- **Coverage**:
  - AgentNet result conversion
  - Result structure parsing
  - Token extraction
  - Metadata preservation

### Schema Validation ✅ 3/3 Tests

#### Message Validation
- **Test**: `test_message_validation`
- **Coverage**:
  - Message object validation
  - Dictionary format validation
  - Invalid structure detection
  - Type checking

#### JSON Schema Validation
- **Test**: `test_json_schema_validation`
- **Coverage**:
  - JSON string validation
  - Malformed JSON detection
  - Schema compliance checking
  - Parsing error handling

#### Compliance Reporting
- **Test**: `test_compliance_report`
- **Coverage**:
  - Completeness scoring
  - Warning generation
  - Error reporting
  - Field presence validation

### Schema Integration ✅ 2/2 Tests

#### Edge Case Handling
- **Test**: `test_edge_case_handling`
- **Coverage**:
  - Empty prompts
  - Maximum confidence values
  - Zero token scenarios
  - Boundary value testing

#### AgentNet Integration
- **Test**: `test_integration_with_agentnet`
- **Coverage**:
  - AgentNet result processing
  - Schema factory integration
  - End-to-end message creation
  - Real-world usage patterns

## JSON Contract Specification

### Complete Message Structure

```json
{
  "task_id": "example-task-123",
  "agent": "ExampleAgent", 
  "message_type": "turn",
  "version": "1.0.0",
  "input": {
    "prompt": "Analyze the benefits of renewable energy",
    "context": {
      "short_term": ["Previous climate discussion"],
      "semantic_refs": [{"id": "ref1", "score": 0.83}],
      "episodic_refs": [],
      "additional_context": {}
    },
    "metadata": {}
  },
  "output": {
    "content": "Renewable energy offers environmental sustainability...",
    "confidence": 0.87,
    "style_insights": ["Applying rigorous logical validation"],
    "tokens": {
      "input": 324,
      "output": 512, 
      "total": 836
    },
    "reasoning_type": "analytical",
    "metadata": {}
  },
  "monitors": [
    {
      "name": "keyword_guard",
      "status": "pass",
      "elapsed_ms": 2.1,
      "details": {"keywords_found": 0},
      "violations": []
    }
  ],
  "cost": {
    "provider": "openai",
    "model": "gpt-4o",
    "usd": 0.01234,
    "tokens_per_dollar": 68000,
    "estimated": false
  },
  "timing": {
    "started": 1736981000.123,
    "completed": 1736981001.001,
    "latency_ms": 878,
    "breakdown": {
      "inference": 650,
      "monitors": 15,
      "processing": 213
    }
  },
  "metadata": {}
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | string | ✅ | Unique identifier for the task |
| `agent` | string | ✅ | Agent name/identifier |
| `message_type` | enum | ✅ | Type of message (turn, system, user, tool, error) |
| `version` | string | ✅ | Schema version |
| `input` | object | ✅ | Input data structure |
| `output` | object | ✅ | Output data structure |
| `monitors` | array | ⚪ | Monitor execution results |
| `cost` | object | ⚪ | Cost tracking information |
| `timing` | object | ✅ | Timing information |
| `metadata` | object | ⚪ | Additional metadata |

## Running Schema Tests

```bash
# Run all schema tests
python tests/test_message_schema.py

# Run with pytest for detailed output
pytest tests/test_message_schema.py -v

# Run specific test class
pytest tests/test_message_schema.py::TestTurnMessage -v

# Test schema validation
python -c "from agentnet.schemas import create_example_message; print(create_example_message().to_json(indent=2))"
```

## Performance Characteristics

- **Validation Speed**: <1ms for typical messages
- **Serialization**: <5ms for complex messages with full monitoring data
- **Memory Usage**: ~2KB per message in memory
- **JSON Size**: Typical messages are 1-3KB serialized

## Best Practices

### Message Creation
1. Use `MessageFactory` for standard message creation
2. Set appropriate confidence scores (0.0-1.0)
3. Include context information for better reasoning
4. Add monitor results for governance compliance

### Validation
1. Always validate messages before transmission
2. Check compliance reports for completeness
3. Handle validation errors gracefully
4. Log validation failures for debugging

### Performance
1. Reuse validator instances when possible
2. Cache serialized messages if frequently accessed
3. Use streaming for large message collections
4. Monitor memory usage in long-running processes

## Future Enhancements

- **Schema Evolution**: Versioned schema migration support
- **Compression**: Optional message compression for large payloads
- **Encryption**: Built-in encryption for sensitive data
- **Streaming**: Support for streaming large messages