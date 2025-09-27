# MLops Workflow and Risk Management Integration Guide

This guide explains how to use the new MLops workflow and risk management capabilities in AgentNet.

## MLops Workflow

The MLops workflow system provides model lifecycle management with validation, versioning, and deployment tracking.

### Basic Usage

```python
from agentnet import MLopsWorkflow, ModelStage

# Initialize MLops workflow
mlops = MLopsWorkflow(storage_dir="my_mlops_data")

# Register a new model
model = mlops.register_model(
    model_id="my-model",
    version="1.0.0",
    provider="openai",
    model_name="gpt-4",
    stage=ModelStage.DEVELOPMENT,
    performance_metrics={"accuracy": 0.95, "latency_ms": 120},
    deployment_config={"max_tokens": 4096, "temperature": 0.7},
    metadata={"created_by": "data-science-team"}
)

# Validate the model
validation_results = mlops.validate_model(
    "my-model", 
    "1.0.0",
    validation_config={
        "performance_thresholds": {
            "accuracy": 0.9,
            "latency_ms": 200
        }
    }
)

# List models
models = mlops.list_models(stage=ModelStage.DEVELOPMENT)
```

## Risk Management System

The risk management system provides automated risk detection, logging, and mitigation tracking.

### Basic Usage

```python
from agentnet import RiskRegister, RiskLevel

# Initialize risk register
risk_register = RiskRegister(storage_dir="my_risk_data")

# Log cost spike risk
cost_risk = risk_register.log_cost_spike_risk(
    current_cost=25.50,
    threshold=20.00,
    agent_name="MyAgent",
    tenant_id="tenant-123"
)

# Log memory bloat risk
memory_risk = risk_register.log_memory_bloat_risk(
    memory_usage=2048,  # MB
    threshold=1024,     # MB
    agent_name="MyAgent",
    session_id="session-456"
)

# Apply mitigation
mitigation = risk_register.mitigate_risk(
    cost_risk.risk_id,
    "Applied rate limiting and cost alerts",
    automated=True,
    effectiveness=0.9
)

# Get risk summary
summary = risk_register.get_risk_summary(tenant_id="tenant-123")
print(f"Total risks: {summary['summary']['total']}")
```

## Cost Predictions

Enhanced cost tracking with predictive modeling and reporting.

### Basic Usage

```python
from agentnet import CostRecorder, CostPredictor

# Initialize cost tracking with predictions
cost_recorder = CostRecorder(storage_dir="cost_data")
cost_predictor = CostPredictor(cost_recorder)

# Generate cost predictions
prediction = cost_predictor.predict_monthly_cost(tenant_id="tenant-123")
print(f"Predicted monthly cost: ${prediction['predicted_monthly_cost']:.2f}")
print(f"Trend: {prediction['trend']}")
print(f"Confidence: {prediction['confidence']}")

# Analyze cost patterns
patterns = cost_predictor.analyze_cost_patterns(tenant_id="tenant-123")
print("Top cost drivers:")
for provider, cost in patterns["patterns"]["top_providers"]:
    print(f"  {provider}: ${cost:.2f}")

# Generate comprehensive report
report = cost_predictor.generate_cost_report(
    tenant_id="tenant-123",
    report_type="detailed"
)
```

## Integration with AgentNet

All these systems integrate seamlessly with existing AgentNet functionality:

```python
from agentnet import AgentNet, ExampleEngine, CostRecorder, RiskRegister, MLopsWorkflow

# Initialize all systems
cost_recorder = CostRecorder()
risk_register = RiskRegister()
mlops = MLopsWorkflow()

# Create agent with cost tracking
agent = AgentNet(
    name="ProductionAgent",
    style={"logic": 0.9},
    engine=ExampleEngine(),
    cost_recorder=cost_recorder,
    tenant_id="prod-tenant"
)

# Register the agent's model in MLops
model = mlops.register_model(
    model_id="production-reasoning-model",
    version="2.1.0",
    provider="example",
    model_name="reasoning-engine",
    stage=ModelStage.PRODUCTION,
    metadata={"agent_name": agent.name}
)

# Agent usage automatically generates cost records and can trigger risk events
result = agent.generate_reasoning_tree("Complex reasoning task")

# Check for any risks that may have been triggered
active_risks = risk_register.list_active_risks(tenant_id="prod-tenant")
if active_risks:
    print(f"⚠️ {len(active_risks)} active risks detected")
```

## Automated Risk Detection

The system includes several built-in risk types that are automatically detected:

- **Cost Spikes**: Token usage exceeding thresholds
- **Memory Bloat**: Memory usage exceeding limits  
- **Convergence Stalls**: Sessions running too long
- **Provider Outages**: High error rates from providers
- **Policy False Positives**: High false positive rates in governance policies

## Integration Tests

Comprehensive integration tests are available in `tests/test_mlops_integration.py` that demonstrate:

- End-to-end workflow validation
- Cost predictions with real data
- Risk management automation
- Error handling and resilience
- Data persistence and retrieval

Run the tests with:

```bash
python -m pytest tests/test_mlops_integration.py -v
```

## Storage and Persistence

All systems use file-based storage by default but can be extended for database integration:

- **MLops**: Stores model metadata, validation results, and deployment records
- **Risk Register**: Stores risk events, mitigations, and risk definitions  
- **Cost Predictions**: Uses existing cost recorder data for analysis

Data is stored in JSON format for easy integration with external systems.