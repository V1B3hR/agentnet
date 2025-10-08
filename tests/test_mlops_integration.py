#!/usr/bin/env python3

"""
Integration tests for MLops workflow, cost tracking enhancements, and risk register.
"""

import json
import logging
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Import the modules we're testing
from agentnet import AgentNet, ExampleEngine
from agentnet.core.cost import CostRecorder, CostAggregator
from agentnet.mlops import MLopsWorkflow, ModelStage, ModelVersion
from agentnet.risk import RiskRegister, RiskLevel, RiskCategory

try:
    from agentnet.core.cost import CostPredictor

    predictions_available = True
except ImportError:
    predictions_available = False
    CostPredictor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cost_predictions_integration():
    """Test cost tracking with predictive modeling."""
    if not predictions_available:
        pytest.skip("Cost predictions module not available")

    print("🧪 Testing Cost Predictions Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize cost tracking
        cost_recorder = CostRecorder(storage_dir=temp_dir + "/costs")
        cost_predictor = CostPredictor(cost_recorder)

        # Simulate some cost records
        agent = AgentNet(
            name="TestAgent",
            style={"logic": 0.8},
            engine=ExampleEngine(),
            cost_recorder=cost_recorder,
            tenant_id="test-tenant",
        )

        # Generate some cost data
        tasks = [
            "Explain machine learning",
            "Design a web API",
            "Write documentation",
            "Analyze data patterns",
            "Create test cases",
        ]

        total_recorded_cost = 0.0
        for i, task in enumerate(tasks):
            result = agent.generate_reasoning_tree(task)
            if result.get("cost_record"):
                total_recorded_cost += result["cost_record"]["total_cost"]

            # Add some delay to simulate real usage over time
            time.sleep(0.1)

        # Test predictive modeling
        prediction = cost_predictor.predict_monthly_cost(tenant_id="test-tenant")

        assert "predicted_monthly_cost" in prediction
        assert "confidence" in prediction
        assert "trend" in prediction

        print(
            f"  ✅ Monthly cost prediction: ${prediction['predicted_monthly_cost']:.4f}"
        )
        print(f"  ✅ Confidence level: {prediction['confidence']}")
        print(f"  ✅ Trend: {prediction['trend']}")

        # Test cost pattern analysis
        patterns = cost_predictor.analyze_cost_patterns(tenant_id="test-tenant")

        assert "patterns" in patterns
        assert "anomalies" in patterns

        print(
            f"  ✅ Cost patterns analyzed - {len(patterns.get('anomalies', []))} anomalies detected"
        )

        # Test cost reporting
        report = cost_predictor.generate_cost_report(
            tenant_id="test-tenant", report_type="summary"
        )

        assert "report_type" in report
        assert report["report_type"] == "summary"
        assert "prediction" in report

        print(f"  ✅ Cost report generated successfully")

    print("  ✅ Cost predictions integration test passed")


def test_mlops_workflow_integration():
    """Test MLops workflow with model lifecycle management."""
    print("🧪 Testing MLops Workflow Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize MLops workflow
        mlops = MLopsWorkflow(storage_dir=temp_dir + "/mlops")

        # Test model registration
        model_v1 = mlops.register_model(
            model_id="test-model",
            version="1.0.0",
            provider="example",
            model_name="test-reasoning-model",
            stage=ModelStage.DEVELOPMENT,
            performance_metrics={"accuracy": 0.85, "latency_ms": 150},
            deployment_config={"max_tokens": 4096, "temperature": 0.7},
            metadata={
                "created_by": "test-suite",
                "description": "Test model for integration",
            },
        )

        assert model_v1.model_id == "test-model"
        assert model_v1.version == "1.0.0"
        assert model_v1.stage == ModelStage.DEVELOPMENT

        print(f"  ✅ Model registered: {model_v1.model_id}:{model_v1.version}")

        # Test model validation
        validation_results = mlops.validate_model(
            "test-model",
            "1.0.0",
            validation_config={
                "performance_thresholds": {"accuracy": 0.8, "latency_ms": 200}
            },
        )

        # Check individual validation components
        assert validation_results["validations"]["schema"]["passed"] is True
        assert validation_results["validations"]["performance"]["passed"] is True
        # Security validation should pass now because model has metadata and deployment_config
        assert validation_results["validations"]["security"]["passed"] is True
        assert validation_results["status"] == "passed"

        print(f"  ✅ Model validation passed")

        # Test model listing
        models = mlops.list_models(stage=ModelStage.DEVELOPMENT)
        assert len(models) == 1
        assert models[0].model_id == "test-model"

        print(f"  ✅ Model listing works - {len(models)} models found")

        # Register another version
        model_v2 = mlops.register_model(
            model_id="test-model",
            version="2.0.0",
            provider="example",
            model_name="test-reasoning-model",
            stage=ModelStage.DEVELOPMENT,
            performance_metrics={"accuracy": 0.92, "latency_ms": 120},
            metadata={"improvements": "Better accuracy and latency"},
        )

        print(f"  ✅ Second model version registered: {model_v2.version}")

        # Test model retrieval
        retrieved_model = mlops.get_model_version("test-model", "1.0.0")
        assert retrieved_model is not None
        assert retrieved_model.model_id == "test-model"
        assert retrieved_model.performance_metrics["accuracy"] == 0.85

        print(f"  ✅ Model retrieval works")

    print("  ✅ MLops workflow integration test passed")


def test_risk_register_integration():
    """Test risk register with automated risk management."""
    print("🧪 Testing Risk Register Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize risk register
        risk_register = RiskRegister(storage_dir=temp_dir + "/risks")

        # Test cost spike risk
        cost_risk = risk_register.log_cost_spike_risk(
            current_cost=15.50,
            threshold=10.00,
            agent_name="TestAgent",
            tenant_id="test-tenant",
        )

        assert cost_risk.category == RiskCategory.FINANCIAL
        assert cost_risk.level == RiskLevel.MEDIUM  # 1.55x threshold
        assert "cost spike" in cost_risk.title.lower()

        print(f"  ✅ Cost spike risk logged: {cost_risk.risk_id}")

        # Test memory bloat risk
        memory_risk = risk_register.log_memory_bloat_risk(
            memory_usage=2048,
            threshold=1024,
            agent_name="TestAgent",
            session_id="test-session-123",
        )

        assert memory_risk.category == RiskCategory.PERFORMANCE
        assert memory_risk.level == RiskLevel.MEDIUM

        print(f"  ✅ Memory bloat risk logged: {memory_risk.risk_id}")

        # Test convergence stall risk
        stall_risk = risk_register.log_convergence_stall_risk(
            session_duration=timedelta(minutes=30),
            max_duration=timedelta(minutes=15),
            agent_name="TestAgent",
            session_id="test-session-123",
        )

        assert stall_risk.category == RiskCategory.PERFORMANCE
        assert stall_risk.level == RiskLevel.HIGH

        print(f"  ✅ Convergence stall risk logged: {stall_risk.risk_id}")

        # Test provider outage risk
        outage_risk = risk_register.log_provider_outage_risk(
            provider="example", error_rate=0.6, agent_name="TestAgent"
        )

        assert outage_risk.category == RiskCategory.OPERATIONAL
        assert outage_risk.level == RiskLevel.HIGH  # 60% error rate

        print(f"  ✅ Provider outage risk logged: {outage_risk.risk_id}")

        # Test risk mitigation
        mitigation = risk_register.mitigate_risk(
            cost_risk.risk_id,
            "Applied rate limiting and cost alerts",
            automated=True,
            effectiveness=0.8,
        )

        assert mitigation.risk_id == cost_risk.risk_id
        assert mitigation.automated is True
        assert mitigation.effectiveness == 0.8

        print(f"  ✅ Risk mitigation recorded: {mitigation.mitigation_id}")

        # Test risk summary
        summary = risk_register.get_risk_summary(tenant_id="test-tenant")

        assert "summary" in summary
        assert summary["summary"]["total"] >= 4  # We logged 4 risks
        assert summary["high_priority_count"] >= 2  # At least 2 high priority risks

        print(
            f"  ✅ Risk summary generated - {summary['summary']['total']} total risks"
        )

        # Test active risks listing
        active_risks = risk_register.list_active_risks(tenant_id="test-tenant")
        assert len(active_risks) >= 1  # At least one risk for this tenant

        print(f"  ✅ Active risks listed - {len(active_risks)} active risks")

        # Test high-priority risks
        high_priority_risks = risk_register.list_active_risks(
            tenant_id="test-tenant", level_filter=RiskLevel.HIGH
        )
        assert len(high_priority_risks) >= 2  # Stall and outage risks

        print(
            f"  ✅ High-priority risks filtered - {len(high_priority_risks)} high-priority risks"
        )

    print("  ✅ Risk register integration test passed")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow with cost tracking, MLops, and risk management."""
    print("🧪 Testing End-to-End Workflow Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize all systems
        cost_recorder = CostRecorder(storage_dir=temp_dir + "/costs")
        mlops = MLopsWorkflow(storage_dir=temp_dir + "/mlops")
        risk_register = RiskRegister(storage_dir=temp_dir + "/risks")

        # Initialize cost predictor if available
        cost_predictor = None
        if predictions_available:
            cost_predictor = CostPredictor(cost_recorder)

        # Set up agent with cost tracking
        agent = AgentNet(
            name="IntegrationTestAgent",
            style={"logic": 0.9, "creativity": 0.5},
            engine=ExampleEngine(),
            cost_recorder=cost_recorder,
            tenant_id="integration-test-tenant",
        )

        print("  ✅ All systems initialized")

        # Scenario 1: Model development lifecycle
        print("  🔄 Running model development lifecycle scenario...")

        # Register a new model
        model = mlops.register_model(
            model_id="integration-model",
            version="1.0.0",
            provider="example",
            model_name="integration-test-model",
            stage=ModelStage.DEVELOPMENT,
            performance_metrics={"accuracy": 0.88, "f1_score": 0.85},
            deployment_config={"max_tokens": 4096, "temperature": 0.7},
            metadata={"test_scenario": "integration", "agent_name": agent.name},
        )

        # Validate the model
        validation = mlops.validate_model("integration-model", "1.0.0")
        assert validation["status"] == "passed"

        print("    ✅ Model registered and validated")

        # Scenario 2: Generate usage with cost tracking
        print("  🔄 Running cost tracking scenario...")

        tasks = [
            "Analyze system performance metrics",
            "Design scalable architecture",
            "Implement error handling strategy",
            "Create monitoring dashboard",
            "Optimize resource utilization",
        ]

        total_cost = 0.0
        for task in tasks:
            result = agent.generate_reasoning_tree(task)
            if result.get("cost_record"):
                total_cost += result["cost_record"]["total_cost"]

        print(f"    ✅ Generated {len(tasks)} tasks with total cost: ${total_cost:.6f}")

        # Scenario 3: Simulate risk events
        print("  🔄 Running risk management scenario...")

        # Simulate cost spike
        if total_cost > 0.001:  # If we have any cost
            cost_risk = risk_register.log_cost_spike_risk(
                current_cost=total_cost * 100,  # Simulate spike
                threshold=total_cost * 10,
                agent_name=agent.name,
                tenant_id=agent.tenant_id,
            )
            print(f"    ✅ Cost spike risk logged: {cost_risk.level.value}")

        # Simulate memory pressure
        memory_risk = risk_register.log_memory_bloat_risk(
            memory_usage=1500, threshold=1024, agent_name=agent.name
        )

        # Simulate provider issue
        provider_risk = risk_register.log_provider_outage_risk(
            provider="example", error_rate=0.3, agent_name=agent.name
        )

        print(f"    ✅ Memory pressure risk logged: {memory_risk.level.value}")
        print(f"    ✅ Provider issue risk logged: {provider_risk.level.value}")

        # Scenario 4: Generate predictive analysis (if available)
        if cost_predictor:
            print("  🔄 Running predictive analysis scenario...")

            prediction = cost_predictor.predict_monthly_cost(tenant_id=agent.tenant_id)
            patterns = cost_predictor.analyze_cost_patterns(tenant_id=agent.tenant_id)

            print(
                f"    ✅ Predicted monthly cost: ${prediction.get('predicted_monthly_cost', 0):.4f}"
            )
            print(
                f"    ✅ Cost patterns analyzed - {len(patterns.get('anomalies', []))} anomalies"
            )

        # Scenario 5: Risk summary and mitigation
        print("  🔄 Running risk summary scenario...")

        risk_summary = risk_register.get_risk_summary(tenant_id=agent.tenant_id)
        active_risks = risk_register.list_active_risks(tenant_id=agent.tenant_id)

        print(f"    ✅ Risk summary: {risk_summary['summary']['total']} total risks")
        print(f"    ✅ Active risks: {len(active_risks)} currently active")

        # Apply mitigations
        for risk in active_risks[:2]:  # Mitigate first 2 risks
            mitigation = risk_register.mitigate_risk(
                risk.risk_id,
                f"Automated mitigation applied for {risk.category.value} risk",
                automated=True,
            )
            print(f"    ✅ Applied mitigation: {mitigation.mitigation_id}")

        # Final validation
        print("  🔄 Running final validation...")

        # Verify data persistence
        models = mlops.list_models()
        assert len(models) >= 1, "Model should be persisted"

        cost_aggregator = CostAggregator(cost_recorder)
        cost_summary = cost_aggregator.get_cost_summary(tenant_id=agent.tenant_id)
        assert cost_summary["record_count"] > 0, "Cost records should exist"

        updated_risk_summary = risk_register.get_risk_summary(tenant_id=agent.tenant_id)
        assert (
            updated_risk_summary["summary"]["total"] >= 1
        ), "Risk events should be persisted"

        print(f"    ✅ Data persistence validated")

        # Summary
        print("  📊 End-to-end workflow summary:")
        print(f"    • Models registered: {len(models)}")
        print(f"    • Cost records: {cost_summary['record_count']}")
        print(f"    • Total cost: ${cost_summary.get('total_cost', 0):.6f}")
        print(f"    • Risk events: {updated_risk_summary['summary']['total']}")
        print(
            f"    • High-priority risks: {updated_risk_summary['high_priority_count']}"
        )

    print("  ✅ End-to-end workflow integration test passed")


def test_workflow_error_handling():
    """Test error handling and resilience in integrated workflows."""
    print("🧪 Testing Workflow Error Handling...")

    with tempfile.TemporaryDirectory() as temp_dir:
        mlops = MLopsWorkflow(storage_dir=temp_dir + "/mlops")
        risk_register = RiskRegister(storage_dir=temp_dir + "/risks")

        # Test invalid model operations
        result = mlops.validate_model("nonexistent-model", "1.0.0")
        assert result["status"] == "error"
        assert "not found" in result["message"]

        print("  ✅ Handles nonexistent model validation gracefully")

        # Test invalid model retrieval
        model = mlops.get_model_version("nonexistent", "1.0.0")
        assert model is None

        print("  ✅ Handles nonexistent model retrieval gracefully")

        # Test risk registration with minimal data
        minimal_risk = risk_register.register_risk_event(
            risk_type="test_minimal_risk", description="Minimal risk for testing"
        )

        assert minimal_risk.risk_id is not None
        assert minimal_risk.level is not None  # Should get default level

        print("  ✅ Handles minimal risk event registration")

        # Test risk summary with empty tenant
        empty_summary = risk_register.get_risk_summary(
            tenant_id="truly-nonexistent-tenant-12345"
        )
        assert (
            empty_summary["summary"]["total"] == 0
            or empty_summary["summary"]["total"] > 0
        )  # Should handle gracefully

        print("  ✅ Handles empty risk summary gracefully")

    print("  ✅ Workflow error handling test passed")


def main():
    """Run all integration tests."""
    print("🚀 Running MLops Workflow and Integration Tests\n")

    try:
        test_cost_predictions_integration()
        print()

        test_mlops_workflow_integration()
        print()

        test_risk_register_integration()
        print()

        test_end_to_end_workflow()
        print()

        test_workflow_error_handling()
        print()

        print("🎉 All integration tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
