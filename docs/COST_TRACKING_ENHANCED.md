# Enhanced Cost Tracking and Analytics

This document describes the enhanced cost tracking system in AgentNet, including predictive modeling, alerting, and comprehensive reporting capabilities.

## Overview

The enhanced cost tracking system extends the basic cost recording with advanced analytics, predictive modeling, and automated alerting to provide comprehensive cost management for multi-agent operations.

## Core Components

### CostPredictor
Predictive modeling engine for cost forecasting.

```python
from agentnet.core.cost.analytics import CostPredictor
from agentnet.core.cost.recorder import CostRecorder

recorder = CostRecorder()
predictor = CostPredictor(recorder)

# Predict monthly costs
prediction = predictor.predict_monthly_cost(tenant_id="my-tenant")
print(f"Predicted monthly cost: ${prediction.predicted_cost:.2f}")
print(f"Confidence interval: ${prediction.confidence_interval[0]:.2f} - ${prediction.confidence_interval[1]:.2f}")
print(f"Model accuracy: {prediction.model_accuracy:.2%}")

# Predict session costs
session_prediction = predictor.predict_session_cost(
    agent_name="my-agent",
    estimated_turns=20,
    provider="openai",
    model="gpt-4"
)
```

### SpendAlertEngine
Automated spend anomaly detection and alerting.

```python
from agentnet.core.cost.analytics import SpendAlertEngine

alert_engine = SpendAlertEngine(recorder)

# Check for spend velocity alerts
velocity_alerts = alert_engine.check_spend_velocity(
    tenant_id="my-tenant",
    lookback_hours=24
)

# Check for daily spend spikes
spike_alerts = alert_engine.check_daily_spend_spikes(
    tenant_id="my-tenant",
    days_to_check=7
)

# Check for agent anomalies
anomaly_alerts = alert_engine.check_agent_anomalies(
    tenant_id="my-tenant",
    lookback_hours=24
)

# Get all alerts
all_alerts = alert_engine.get_all_alerts(tenant_id="my-tenant")
```

### CostReportGenerator
Comprehensive cost analytics and executive reporting.

```python
from agentnet.core.cost.analytics import CostReportGenerator

report_generator = CostReportGenerator(recorder)

# Generate executive summary
summary = report_generator.generate_executive_summary(
    tenant_id="my-tenant",
    period_days=30
)

print(f"Total cost: ${summary['cost_summary']['total_cost']:.2f}")
print(f"Monthly forecast: ${summary['predictions']['monthly_forecast']:.2f}")
print(f"Critical alerts: {summary['alerts']['critical_alerts']}")
```

## Features

### Predictive Modeling

#### Monthly Cost Prediction
Predicts monthly costs based on historical usage patterns:

- **Trend Analysis**: Identifies spending trends and seasonal patterns
- **Confidence Intervals**: Provides uncertainty bounds for predictions
- **Contributing Factors**: Analysis of factors affecting cost predictions
- **Model Accuracy**: Self-assessed prediction accuracy based on historical performance

#### Session Cost Prediction
Estimates costs for upcoming agent sessions:

- **Agent-specific**: Uses historical data for specific agents
- **Turn-based Estimation**: Predicts cost based on expected conversation turns
- **Provider Optimization**: Recommends optimal provider/model combinations

### Spend Alerting

#### Velocity Alerts
Detects abnormal spending rate increases:

```python
# Alert when spending rate exceeds 3x baseline
velocity_multiplier = 3.0

# Example alert conditions:
# - Current hourly spend: $50
# - Baseline hourly spend: $15
# - Velocity ratio: 3.33x (triggers alert)
```

#### Spike Alerts
Identifies daily spending spikes:

```python
# Alert thresholds
daily_spike_threshold = 5.0    # $5 per day
hourly_spike_threshold = 1.0   # $1 per hour

# Severity levels:
# - Medium: Above threshold
# - High: 2x threshold
# - Critical: 4x threshold
```

#### Anomaly Detection
Identifies unusual agent spending patterns:

- **Statistical Analysis**: Uses standard deviation to detect outliers
- **Agent Comparison**: Compares agents within the same tenant
- **Cost per Inference**: Tracks efficiency metrics

### Comprehensive Reporting

#### Executive Summary
High-level cost overview for stakeholders:

```json
{
  "cost_summary": {
    "total_cost": 123.45,
    "average_daily_cost": 4.12,
    "cost_change_percentage": 15.2,
    "inference_count": 1250
  },
  "predictions": {
    "monthly_forecast": 156.78,
    "confidence_range": [120.45, 195.32],
    "model_accuracy": 0.85
  },
  "alerts": {
    "total_alerts": 3,
    "critical_alerts": 1
  },
  "top_providers": [
    {"provider": "openai", "cost": 98.76, "percentage": 80.1}
  ],
  "top_agents": [
    {"agent": "customer-support", "cost": 45.23, "inference_count": 456}
  ]
}
```

## Configuration

### Alert Thresholds
Customize alerting thresholds in the SpendAlertEngine:

```python
alert_engine = SpendAlertEngine(recorder)

# Modify default thresholds
alert_engine.alert_thresholds = {
    "velocity_multiplier": 2.5,        # Lower sensitivity
    "daily_spike_threshold": 10.0,     # Higher threshold
    "hourly_spike_threshold": 2.0,     # Higher threshold
}
```

### Prediction Parameters
Configure prediction model parameters:

```python
# Confidence level for predictions (0.90 = 90% confidence)
prediction = predictor.predict_monthly_cost(
    tenant_id="my-tenant",
    confidence_level=0.90
)

# Input/output token ratio for session predictions
session_prediction = predictor.predict_session_cost(
    agent_name="my-agent",
    estimated_turns=20,
    provider="openai",
    model="gpt-4"
)
```

## Integration Examples

### With Risk Management
Cost tracking integrates with the risk management system:

```python
from agentnet.risk.registry import RiskRegistry
from agentnet.risk.monitor import RiskMonitor

# Initialize risk monitoring
risk_registry = RiskRegistry()
risk_monitor = RiskMonitor(risk_registry)

# Cost spike detection with risk logging
cost_stats = {
    "current_hourly_cost": 50.0,
    "baseline_hourly_cost": 10.0,
    "daily_cost": 120.0
}

cost_alerts = risk_monitor.check_cost_risks(
    cost_stats=cost_stats,
    tenant_id="my-tenant"
)

# Automatically registers cost_spike risk events
```

### With CI/CD Pipeline
Automated cost monitoring in deployments:

```yaml
# In .github/workflows/deploy.yml
- name: Cost monitoring check
  run: |
    python -c "
    from agentnet.core.cost.recorder import CostRecorder
    from agentnet.core.cost.analytics import CostReportGenerator
    
    recorder = CostRecorder()
    report_gen = CostReportGenerator(recorder)
    
    summary = report_gen.generate_executive_summary(period_days=1)
    daily_cost = summary['cost_summary']['total_cost']
    
    if daily_cost > 100.0:  # Threshold
        print(f'HIGH COST ALERT: ${daily_cost:.2f} in last 24h')
        exit(1)
    "
```

### With Multi-tenant Systems
Tenant-specific cost tracking and alerting:

```python
# Per-tenant cost management
tenants = ["tenant-a", "tenant-b", "tenant-c"]

for tenant_id in tenants:
    # Generate tenant-specific reports
    summary = report_generator.generate_executive_summary(
        tenant_id=tenant_id,
        period_days=30
    )
    
    # Check tenant-specific alerts
    alerts = alert_engine.get_all_alerts(tenant_id=tenant_id)
    
    # Tenant-specific predictions
    prediction = predictor.predict_monthly_cost(tenant_id=tenant_id)
    
    print(f"Tenant {tenant_id}:")
    print(f"  Current cost: ${summary['cost_summary']['total_cost']:.2f}")
    print(f"  Predicted: ${prediction.predicted_cost:.2f}")
    print(f"  Alerts: {len(sum(alerts.values(), []))}")
```

## Advanced Analytics

### Cost Attribution
Track costs by various dimensions:

```python
from agentnet.core.cost.recorder import CostAggregator

aggregator = CostAggregator(recorder)

# Get cost breakdown by provider
summary = aggregator.get_cost_summary(tenant_id="my-tenant")
for provider, data in summary["provider_breakdown"].items():
    print(f"{provider}: ${data['cost']:.2f} ({data['tokens_input']} + {data['tokens_output']} tokens)")

# Get top cost-generating agents
top_agents = aggregator.get_top_cost_agents(limit=10)
for agent in top_agents:
    print(f"{agent['agent_name']}: ${agent['total_cost']:.2f} ({agent['inference_count']} inferences)")
```

### Trend Analysis
Analyze cost trends over time:

```python
# Get cost trends
trends = aggregator.get_cost_trends(days=30, tenant_id="my-tenant")

for day_data in trends["daily_trends"]:
    print(f"{day_data['date']}: ${day_data['total_cost']:.2f} ({day_data['total_tokens']:,} tokens)")
```

### Budget Management
Implement budget controls with the TenantCostTracker:

```python
from agentnet.core.cost.recorder import TenantCostTracker

tracker = TenantCostTracker(recorder)

# Set monthly budget
tracker.set_tenant_budget("my-tenant", 500.0)

# Set alert thresholds
tracker.set_tenant_alerts("my-tenant", {
    "warning": 0.75,    # 75% of budget
    "critical": 0.90    # 90% of budget
})

# Check budget status
status = tracker.check_tenant_budget("my-tenant")
print(f"Budget usage: {status['usage_percentage']:.1%}")
print(f"Remaining: ${status['remaining_budget']:.2f}")
print(f"Status: {status['status']}")
```

## Performance Considerations

### Data Retention
Configure data retention policies:

```python
# Cost data is stored in daily files
# Example: cost_logs/costs_2024-01-15.jsonl

# Cleanup old data (example script)
import os
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_old_cost_data(storage_dir, retention_days=90):
    storage_path = Path(storage_dir)
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    for file_path in storage_path.glob("costs_*.jsonl"):
        # Extract date from filename
        date_str = file_path.stem.split("_")[1]
        file_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        if file_date < cutoff_date:
            os.remove(file_path)
            print(f"Removed old cost data: {file_path}")
```

### Query Optimization
Optimize queries for large datasets:

```python
# Use date filters to limit data processing
from datetime import datetime, timedelta

# Query only recent data for better performance
end_date = datetime.now()
start_date = end_date - timedelta(days=7)  # Last week only

records = recorder.get_records(
    start_date=start_date,
    end_date=end_date,
    tenant_id="my-tenant"
)
```

### Caching
Implement caching for expensive operations:

```python
import functools
from datetime import datetime, timedelta

@functools.lru_cache(maxsize=100)
def cached_monthly_prediction(tenant_id, date_key):
    """Cache monthly predictions for 1 hour."""
    return predictor.predict_monthly_cost(tenant_id=tenant_id)

# Use with date-based cache key
date_key = datetime.now().strftime("%Y-%m-%d-%H")
prediction = cached_monthly_prediction("my-tenant", date_key)
```

## Monitoring and Observability

### Metrics Collection
Track cost analytics performance:

```python
import time
import logging

# Log prediction accuracy
logger = logging.getLogger("agentnet.cost.analytics")

def track_prediction_accuracy(predictor, tenant_id):
    start_time = time.time()
    prediction = predictor.predict_monthly_cost(tenant_id=tenant_id)
    duration = time.time() - start_time
    
    logger.info(f"Monthly prediction for {tenant_id}: "
                f"${prediction.predicted_cost:.2f} "
                f"(accuracy: {prediction.model_accuracy:.2%}, "
                f"duration: {duration:.3f}s)")
```

### Health Checks
Monitor system health:

```python
def cost_system_health_check():
    """Perform health check on cost tracking system."""
    health = {
        "recorder": False,
        "predictor": False,
        "alerting": False,
        "reporting": False
    }
    
    try:
        # Test recorder
        recorder = CostRecorder()
        test_record = recorder.record_inference_cost(
            provider="test", model="test", 
            result={"tokens_input": 1, "tokens_output": 1, "content": "test"},
            agent_name="health-check", task_id="health-check"
        )
        health["recorder"] = test_record is not None
        
        # Test predictor
        predictor = CostPredictor(recorder)
        prediction = predictor.predict_monthly_cost()
        health["predictor"] = prediction is not None
        
        # Test alerting
        alert_engine = SpendAlertEngine(recorder)
        alerts = alert_engine.get_all_alerts()
        health["alerting"] = isinstance(alerts, dict)
        
        # Test reporting
        report_gen = CostReportGenerator(recorder)
        summary = report_gen.generate_executive_summary(period_days=1)
        health["reporting"] = "cost_summary" in summary
        
    except Exception as e:
        logging.error(f"Cost system health check failed: {e}")
    
    return health
```

## Best Practices

### Cost Optimization
1. **Model Selection**: Use cost predictions to choose optimal models
2. **Rate Limiting**: Implement rate limits based on cost thresholds
3. **Budget Alerts**: Set up proactive budget monitoring
4. **Agent Efficiency**: Track and optimize cost per inference

### Data Quality
1. **Token Counting**: Ensure accurate token count reporting
2. **Provider Mapping**: Maintain accurate provider cost mappings
3. **Data Validation**: Validate cost records for consistency
4. **Error Handling**: Handle provider pricing changes gracefully

### Security
1. **Data Encryption**: Encrypt cost data at rest
2. **Access Controls**: Implement tenant isolation
3. **Audit Logging**: Log all cost-related operations
4. **PII Protection**: Avoid storing sensitive data in cost records

### Scalability
1. **Partitioning**: Partition cost data by date and tenant
2. **Aggregation**: Pre-compute common aggregations
3. **Archiving**: Archive old data to reduce query load
4. **Indexing**: Index frequently queried fields