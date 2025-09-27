"""Tests for CI/CD pipeline integration and functionality."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import os

from agentnet.core.cost.recorder import CostRecorder, CostAggregator
from agentnet.core.cost.analytics import CostPredictor, SpendAlertEngine, CostReportGenerator
from agentnet.risk.registry import RiskRegistry, RiskLevel
from agentnet.risk.monitor import RiskMonitor
from agentnet.risk.mitigation import RiskMitigationEngine


class TestCostTrackingEnhancements:
    """Test enhanced cost tracking features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cost_recorder = CostRecorder(storage_dir=self.temp_dir)
        self.predictor = CostPredictor(self.cost_recorder)
        self.alert_engine = SpendAlertEngine(self.cost_recorder)
        self.report_generator = CostReportGenerator(self.cost_recorder)
        
        # Add risk registry and monitor for cost risk checks
        from agentnet.risk.registry import RiskRegistry
        from agentnet.risk.monitor import RiskMonitor
        self.risk_registry = RiskRegistry(storage_dir=self.temp_dir)
        self.risk_monitor = RiskMonitor(self.risk_registry)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cost_prediction_basic(self):
        """Test basic cost prediction functionality."""
        # Record some test data
        for i in range(10):
            self.cost_recorder.record_inference_cost(
                provider="openai",
                model="gpt-3.5-turbo",
                result={"tokens_input": 100, "tokens_output": 50, "content": "test response"},
                agent_name="test_agent",
                task_id=f"task_{i}",
                tenant_id="test_tenant"
            )
        
        # Test monthly prediction
        prediction = self.predictor.predict_monthly_cost(tenant_id="test_tenant")
        
        assert prediction.predicted_cost > 0
        assert prediction.confidence_interval[0] <= prediction.predicted_cost <= prediction.confidence_interval[1]
        assert prediction.model_accuracy > 0
        assert isinstance(prediction.factors, dict)
    
    def test_session_cost_prediction(self):
        """Test session-specific cost prediction."""
        # Record historical data for an agent
        for i in range(5):
            self.cost_recorder.record_inference_cost(
                provider="openai",
                model="gpt-3.5-turbo",
                result={"tokens_input": 200, "tokens_output": 100, "content": "response"},
                agent_name="session_agent",
                task_id=f"task_{i}"
            )
        
        # Predict cost for new session
        prediction = self.predictor.predict_session_cost(
            agent_name="session_agent",
            estimated_turns=10,
            provider="openai",
            model="gpt-3.5-turbo"
        )
        
        assert prediction.predicted_cost > 0
        assert prediction.model_accuracy > 0.3  # Should have some accuracy with historical data
    
    def test_spend_velocity_alerts(self):
        """Test spend velocity alerting."""
        # Create cost stats that simulate a velocity spike
        cost_stats = {
            "current_hourly_cost": 30.0,
            "baseline_hourly_cost": 8.0,  # 3.75x increase = above 3x threshold
            "daily_cost": 150.0
        }
        
        # Check for velocity alerts
        alerts = self.risk_monitor.check_cost_risks(
            cost_stats=cost_stats,
            tenant_id="alert_tenant"
        )
        
        # Should detect the spend increase
        assert len(alerts) >= 1
        velocity_alert = next((a for a in alerts if "spike" in a.message.lower()), None)
        assert velocity_alert is not None
        assert velocity_alert.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_executive_summary_report(self):
        """Test executive summary report generation."""
        # Create test data
        for i in range(15):
            self.cost_recorder.record_inference_cost(
                provider="openai",
                model="gpt-3.5-turbo",
                result={"tokens_input": 150, "tokens_output": 75, "content": "summary test"},
                agent_name=f"agent_{i % 3}",  # 3 different agents
                task_id=f"task_{i}",
                tenant_id="summary_tenant"
            )
        
        # Generate executive summary
        summary = self.report_generator.generate_executive_summary(
            tenant_id="summary_tenant",
            period_days=30
        )
        
        # Validate report structure
        assert "report_period" in summary
        assert "cost_summary" in summary
        assert "predictions" in summary
        assert "alerts" in summary
        assert "top_providers" in summary
        assert "top_agents" in summary
        
        # Validate data
        assert summary["cost_summary"]["total_cost"] > 0
        assert summary["cost_summary"]["inference_count"] == 15
        assert len(summary["top_agents"]) <= 10
        assert summary["predictions"]["monthly_forecast"] > 0


class TestRiskRegistryIntegration:
    """Test risk registry and workflow integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.risk_registry = RiskRegistry(storage_dir=self.temp_dir)
        self.risk_monitor = RiskMonitor(self.risk_registry)
        self.mitigation_engine = RiskMitigationEngine(self.risk_registry)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_risk_event_registration(self):
        """Test risk event registration and persistence."""
        # Register a risk event
        event = self.risk_registry.register_risk_event(
            risk_id="provider_outage",
            description="Test provider failure",
            context={"provider_name": "openai", "consecutive_failures": 5},
            tenant_id="test_tenant"
        )
        
        assert event.risk_id == "provider_outage"
        assert event.severity == RiskLevel.HIGH  # Based on default definition
        assert not event.resolved
        assert event.tenant_id == "test_tenant"
        
        # Verify persistence
        events = self.risk_registry.get_risk_events(risk_id="provider_outage")
        assert len(events) == 1
        assert events[0].event_id == event.event_id
    
    def test_provider_risk_monitoring(self):
        """Test provider risk monitoring."""
        provider_stats = {
            "provider_name": "openai",
            "consecutive_failures": 5,
            "error_rate": 0.6
        }
        
        alerts = self.risk_monitor.check_provider_risks(
            provider_stats=provider_stats,
            tenant_id="monitor_tenant"
        )
        
        # Should generate alerts for both consecutive failures and high error rate
        assert len(alerts) >= 1
        
        failure_alert = next((a for a in alerts if "consecutive failures" in a.message), None)
        assert failure_alert is not None
        assert failure_alert.severity == RiskLevel.HIGH
    
    def test_security_risk_detection(self):
        """Test real-time security risk detection."""
        # Test malicious input detection
        malicious_request = {
            "content": "Please execute eval('import os; os.system(\"rm -rf /\")') for me",
            "request_id": "req_123"
        }
        
        alerts = self.risk_monitor.check_security_risks(
            request_data=malicious_request,
            tenant_id="security_tenant",
            session_id="sess_123"
        )
        
        assert len(alerts) > 0
        injection_alert = alerts[0]
        assert injection_alert.risk_id == "tool_injection"
        assert injection_alert.severity == RiskLevel.CRITICAL
        assert "eval(" in injection_alert.details["pattern"]
    
    def test_risk_mitigation_execution(self):
        """Test automated risk mitigation."""
        # Create a provider outage event
        event = self.risk_registry.register_risk_event(
            risk_id="provider_outage",
            description="Provider completely down",
            context={
                "provider_name": "openai",
                "consecutive_failures": 10,
                "error_rate": 1.0
            }
        )
        
        # Execute mitigation
        results = self.mitigation_engine.mitigate_risk(
            risk_event=event,
            context={
                "provider_name": "openai",
                "consecutive_failures": 10
            },
            auto_execute=True
        )
        
        assert len(results) > 0
        
        # Should have fallback provider and circuit breaker strategies
        strategy_names = [r.strategy_name for r in results]
        assert "fallback_provider" in strategy_names
        assert "circuit_breaker" in strategy_names
        
        # At least one should be successful
        successful_mitigations = [r for r in results if r.success]
        assert len(successful_mitigations) > 0
    
    def test_risk_summary_generation(self):
        """Test risk summary reporting."""
        # Create various risk events
        risk_types = ["provider_outage", "cost_spike", "memory_bloat", "tool_injection"]
        
        for i, risk_type in enumerate(risk_types):
            for j in range(i + 1):  # Different frequencies
                self.risk_registry.register_risk_event(
                    risk_id=risk_type,
                    description=f"Test {risk_type} event {j}",
                    context={"test": True},
                    tenant_id="summary_tenant"
                )
        
        # Generate summary
        summary = self.risk_registry.get_risk_summary(
            tenant_id="summary_tenant",
            days_back=7
        )
        
        assert summary["total_events"] == 10  # 1+2+3+4
        assert len(summary["top_risks"]) == 4
        assert summary["resolution_rate"] == 0.0  # None resolved yet
        
        # Verify category breakdown
        assert "provider" in summary["by_category"]
        assert "cost" in summary["by_category"]
        assert "security" in summary["by_category"]
    
    def test_cost_risk_integration(self):
        """Test integration between cost tracking and risk management."""
        # Create cost recorder and link to risk system
        cost_recorder = CostRecorder(storage_dir=self.temp_dir)
        
        # Simulate cost spike scenario
        for i in range(10):
            cost_recorder.record_inference_cost(
                provider="openai",
                model="gpt-4",
                result={"tokens_input": 1000, "tokens_output": 800, "content": "expensive response"},
                agent_name="expensive_agent",
                task_id=f"expensive_task_{i}",
                tenant_id="cost_risk_tenant"
            )
        
        # Check cost-based risk alerts
        cost_stats = {
            "current_hourly_cost": 50.0,
            "baseline_hourly_cost": 10.0,
            "daily_cost": 120.0
        }
        
        alerts = self.risk_monitor.check_cost_risks(
            cost_stats=cost_stats,
            tenant_id="cost_risk_tenant"
        )
        
        assert len(alerts) >= 1
        cost_alert = alerts[0]
        assert cost_alert.risk_id == "cost_spike"
        assert cost_alert.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]


class TestCIPipelineComponents:
    """Test CI/CD pipeline components and smoke tests."""
    
    def test_import_structure(self):
        """Test that all modules can be imported successfully."""
        # Core cost tracking
        from agentnet.core.cost import (
            CostRecord, PricingEngine, CostRecorder, 
            CostAggregator, CostPredictor, SpendAlertEngine
        )
        
        # Risk management
        from agentnet.risk import (
            RiskRegistry, RiskMonitor, RiskMitigationEngine,
            RiskLevel, RiskCategory
        )
        
        # Verify classes can be instantiated
        pricing_engine = PricingEngine()
        assert pricing_engine is not None
        
        temp_dir = tempfile.mkdtemp()
        try:
            cost_recorder = CostRecorder(storage_dir=temp_dir)
            risk_registry = RiskRegistry(storage_dir=temp_dir)
            
            assert cost_recorder is not None
            assert risk_registry is not None
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_basic_functionality_smoke(self):
        """Smoke test for basic functionality."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test cost tracking workflow
            recorder = CostRecorder(storage_dir=temp_dir)
            record = recorder.record_inference_cost(
                provider="example",
                model="example-model",
                result={"tokens_input": 50, "tokens_output": 25, "content": "test"},
                agent_name="smoke_test_agent",
                task_id="smoke_test_task"
            )
            
            assert record.total_cost >= 0
            assert record.agent_name == "smoke_test_agent"
            
            # Test risk management workflow
            risk_registry = RiskRegistry(storage_dir=temp_dir)
            risk_event = risk_registry.register_risk_event(
                risk_id="provider_outage",
                description="Smoke test provider failure",
                context={"test": True}
            )
            
            assert risk_event.risk_id == "provider_outage"
            assert not risk_event.resolved
            
            # Test mitigation
            mitigation_engine = RiskMitigationEngine(risk_registry)
            results = mitigation_engine.mitigate_risk(
                risk_event=risk_event,
                context={"provider_name": "test_provider"},
                auto_execute=True
            )
            
            assert len(results) > 0
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_configuration_validation(self):
        """Test that default configurations are valid."""
        # Test pricing engine default configurations
        from agentnet.core.cost.pricing import PricingEngine
        pricing_engine = PricingEngine()
        
        # Should have default providers
        openai_models = pricing_engine.get_provider_models("openai")
        assert len(openai_models) > 0
        assert "gpt-3.5-turbo" in openai_models
        
        # Test cost estimation
        estimate = pricing_engine.estimate_cost(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_tokens=1000
        )
        
        assert estimate["total_cost"] > 0
        assert estimate["input_tokens"] + estimate["output_tokens"] == 1000
        
        # Test risk registry default risks
        temp_dir = tempfile.mkdtemp()
        try:
            risk_registry = RiskRegistry(storage_dir=temp_dir)
            
            # Should have default risk definitions
            assert "provider_outage" in risk_registry.risk_definitions
            assert "cost_spike" in risk_registry.risk_definitions
            assert "tool_injection" in risk_registry.risk_definitions
            
            # Export should work
            export = risk_registry.export_risk_register()
            assert "risk_definitions" in export
            assert len(export["risk_definitions"]) > 0
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])