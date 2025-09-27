#!/usr/bin/env python3
"""
Comprehensive tests for cost tracking enhancements and risk management system.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest


def test_cost_predictive_modeling():
    """Test cost prediction and optimization features."""
    print("🧪 Testing Cost Predictive Modeling...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.core.cost import CostRecorder, CostPredictor, CostOptimizer, CostReporter
        from agentnet import AgentNet, ExampleEngine
        
        # Set up cost tracking system
        recorder = CostRecorder(storage_dir=tmpdir)
        predictor = CostPredictor(recorder)
        optimizer = CostOptimizer(recorder, predictor)
        reporter = CostReporter(recorder, predictor, optimizer)
        
        # Create agent with cost tracking
        engine = ExampleEngine()
        agent = AgentNet(
            name="PredictiveTestAgent",
            style={"logic": 0.8},
            engine=engine,
            cost_recorder=recorder,
            tenant_id="predictive-test",
        )
        
        # Generate historical data
        tasks = [
            "Analyze system architecture",
            "Design security protocols",
            "Implement monitoring",
            "Optimize performance",
            "Create documentation",
        ]
        
        for task in tasks:
            result = agent.generate_reasoning_tree(task)
            time.sleep(0.1)  # Ensure timestamp differences
        
        print("  📊 Generated historical cost data")
        
        # Test cost prediction
        prediction = predictor.predict_monthly_cost(tenant_id="predictive-test")
        assert prediction.predicted_cost >= 0
        assert prediction.confidence_interval[0] <= prediction.predicted_cost <= prediction.confidence_interval[1]
        assert prediction.trend in ["increasing", "decreasing", "stable", "insufficient_data"]
        print(f"  📈 Monthly cost prediction: ${prediction.predicted_cost:.6f} ({prediction.trend})")
        
        # Test session cost prediction
        session_estimate = predictor.predict_session_cost(
            estimated_interactions=10,
            agent_name="PredictiveTestAgent",
        )
        assert "predicted_cost" in session_estimate
        assert "confidence" in session_estimate
        print(f"  📊 Session cost estimate: ${session_estimate['predicted_cost']:.6f}")
        
        # Test cost optimization
        recommendations = optimizer.generate_recommendations(tenant_id="predictive-test")
        assert isinstance(recommendations, list)
        print(f"  💡 Generated {len(recommendations)} optimization recommendations")
        
        # Test comprehensive reporting
        report = reporter.generate_comprehensive_report(tenant_id="predictive-test")
        assert "cost_summary" in report
        assert "predictions" in report
        assert "optimization_recommendations" in report
        assert "key_insights" in report
        
        insights_count = len(report["key_insights"])
        print(f"  📋 Comprehensive report: {insights_count} key insights")
        
    print("  ✅ Cost predictive modeling tests passed")


def test_risk_registry_system():
    """Test risk registry and core risk management."""
    print("🧪 Testing Risk Registry System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.risk import RiskRegistry, Risk, RiskLevel, RiskStatus, RiskCategory
        
        # Create risk registry
        registry_path = Path(tmpdir) / "test_risks.json"
        registry = RiskRegistry(storage_path=str(registry_path))
        
        # Test default risks are loaded
        default_risks = registry.get_active_risks()
        assert len(default_risks) > 0
        print(f"  📝 Loaded {len(default_risks)} default risks")
        
        # Test risk categorization
        provider_risks = registry.get_risks_by_category(RiskCategory.PROVIDER_OUTAGE)
        security_risks = registry.get_risks_by_category(RiskCategory.SECURITY)
        assert len(provider_risks) > 0
        print(f"  🔍 Found {len(provider_risks)} provider risks, {len(security_risks)} security risks")
        
        # Test high priority risk identification
        high_priority = registry.get_high_priority_risks()
        critical_risks = registry.get_risks_by_level(RiskLevel.CRITICAL)
        assert len(high_priority) >= len(critical_risks)
        print(f"  🚨 Identified {len(high_priority)} high priority risks")
        
        # Test risk statistics
        stats = registry.get_risk_statistics()
        assert stats["total_risks"] > 0
        assert "level_distribution" in stats
        assert "category_distribution" in stats
        print(f"  📊 Registry statistics: {stats['total_risks']} total risks")
        
        # Test custom risk creation
        custom_risk = Risk(
            risk_id="CUSTOM-001",
            title="Test Custom Risk",
            description="A risk created for testing purposes",
            category=RiskCategory.OPERATIONAL,
            level=RiskLevel.MEDIUM,
            status=RiskStatus.IDENTIFIED,
            probability=0.3,
            impact=0.5,
            risk_score=0.15,
            identified_date=datetime.now(),
            identified_by="test_system",
            last_updated=datetime.now(),
            updated_by="test_system",
        )
        
        registry.register_risk(custom_risk)
        retrieved_risk = registry.get_risk("CUSTOM-001")
        assert retrieved_risk is not None
        assert retrieved_risk.title == "Test Custom Risk"
        print("  ➕ Successfully created and retrieved custom risk")
        
        # Test risk persistence
        assert registry_path.exists()
        print("  💾 Risk registry persistence verified")
        
    print("  ✅ Risk registry system tests passed")


def test_risk_assessment_system():
    """Test automated risk assessment capabilities.""" 
    print("🧪 Testing Risk Assessment System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.risk import RiskRegistry, RiskAssessor
        
        registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        assessor = RiskAssessor()
        
        # Get a risk to assess
        risks = registry.get_active_risks()
        test_risk = risks[0]  # Use first available risk
        
        # Test basic risk assessment
        context = {
            "recent_provider_outages": 2,
            "active_providers": 1,
            "avg_memory_usage_mb": 128,
            "system_load": 0.6,
            "business_criticality": "high",
            "active_users": 500,
        }
        
        assessment = assessor.assess_risk(test_risk, context, assessor="test_system")
        
        assert assessment.risk_id == test_risk.risk_id
        assert 0.0 <= assessment.current_probability <= 1.0
        assert 0.0 <= assessment.current_impact <= 1.0
        assert 0.0 <= assessment.priority_score <= 100.0
        assert len(assessment.recommended_actions) > 0
        
        print(f"  📊 Assessed {test_risk.title}:")
        print(f"    Risk Score: {assessment.current_risk_score:.3f}")
        print(f"    Priority: {assessment.priority_score:.1f}/100")
        print(f"    Actions: {len(assessment.recommended_actions)}")
        
        # Test assessment history
        history = assessor.get_assessment_history(test_risk.risk_id)
        assert len(history) == 1
        assert history[0].assessment_id == assessment.assessment_id
        
        # Test second assessment with different context
        updated_context = context.copy()
        updated_context["recent_provider_outages"] = 0
        updated_context["active_providers"] = 3
        
        second_assessment = assessor.assess_risk(test_risk, updated_context)
        
        # Test assessment comparison
        comparison = assessor.compare_assessments(
            test_risk.risk_id,
            assessment.assessment_id,
            second_assessment.assessment_id,
        )
        
        assert "changes" in comparison
        assert "probability_change" in comparison["changes"]
        print(f"  🔄 Assessment comparison: {comparison['changes']['probability_change']:.3f} probability change")
        
    print("  ✅ Risk assessment system tests passed")


def test_risk_mitigation_system():
    """Test risk mitigation strategies and execution."""
    print("🧪 Testing Risk Mitigation System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.risk import RiskRegistry, RiskCategory
        from agentnet.risk.mitigation import MitigationMitigator, MitigationStatus
        
        registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        mitigator = MitigationMitigator()
        
        # Get a provider outage risk to mitigate
        provider_risks = registry.get_risks_by_category(RiskCategory.PROVIDER_OUTAGE)
        test_risk = provider_risks[0]
        
        # Test strategy creation
        strategy = mitigator.create_strategy_for_risk(
            test_risk, 
            created_by="test_system"
        )
        
        assert strategy.risk_id == test_risk.risk_id
        assert len(strategy.actions) > 0
        assert strategy.is_active
        print(f"  🎯 Created mitigation strategy with {len(strategy.actions)} actions")
        
        # Test action execution
        first_action = strategy.actions[0]
        assert first_action.status == MitigationStatus.PLANNED
        
        success = mitigator.execute_action(first_action.action_id, executor="test_system")
        assert success
        assert first_action.status == MitigationStatus.COMPLETED
        print(f"  ⚡ Executed action: {first_action.title}")
        
        # Test progress calculation
        progress = strategy.calculate_progress()
        assert 0.0 <= progress <= 1.0
        print(f"  📈 Strategy progress: {progress:.1%}")
        
        # Test risk reduction calculation
        risk_reduction = strategy.calculate_risk_reduction()
        assert 0.0 <= risk_reduction <= 1.0
        print(f"  📉 Risk reduction: {risk_reduction:.1%}")
        
        # Test mitigation summary
        summary = mitigator.get_mitigation_summary()
        assert summary["total_strategies"] >= 1
        assert summary["total_actions"] >= len(strategy.actions)
        print(f"  📋 Mitigation summary: {summary['total_strategies']} strategies, {summary['total_actions']} actions")
        
        # Test pending actions
        pending = mitigator.get_pending_actions()
        completed_count = len([a for a in strategy.actions if a.status == MitigationStatus.COMPLETED])
        expected_pending = len(strategy.actions) - completed_count
        assert len(pending) >= expected_pending
        print(f"  📋 Pending actions: {len(pending)}")
        
    print("  ✅ Risk mitigation system tests passed")


def test_risk_monitoring_system():
    """Test risk monitoring and alerting."""
    print("🧪 Testing Risk Monitoring System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.risk import RiskRegistry, RiskCategory, RiskLevel
        from agentnet.risk.monitoring import RiskMonitor, AlertSeverity, AlertChannel
        
        registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        monitor = RiskMonitor()
        
        # Test monitoring rule setup
        initial_rules = len(monitor.monitoring_rules)
        assert initial_rules > 0
        print(f"  📏 Loaded {initial_rules} default monitoring rules")
        
        # Test custom monitoring rule
        monitor.add_monitoring_rule(
            rule_id="test_custom_rule",
            description="Test custom monitoring rule",
            condition=lambda risk, ctx: risk.risk_score > 0.5,
            alert_type="test_alert",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        
        assert len(monitor.monitoring_rules) == initial_rules + 1
        print("  ➕ Added custom monitoring rule")
        
        # Test risk monitoring
        high_risk = None
        for risk in registry.get_active_risks():
            if risk.risk_score > 0.5:
                high_risk = risk
                break
        
        if high_risk:
            alerts = monitor.check_risk(high_risk, {"is_new_risk": False})
            print(f"  🚨 Generated {len(alerts)} alerts for high-risk scenario")
            
            if alerts:
                test_alert = alerts[0]
                assert test_alert.risk_id == high_risk.risk_id
                assert test_alert.severity in list(AlertSeverity)
                
                # Test alert acknowledgment
                ack_success = monitor.acknowledge_alert(test_alert.alert_id, "test_user")
                assert ack_success
                assert test_alert.is_acknowledged
                print("  ✅ Alert acknowledged successfully")
                
                # Test alert resolution
                resolve_success = monitor.resolve_alert(test_alert.alert_id)
                assert resolve_success
                assert test_alert.is_resolved
                print("  ✅ Alert resolved successfully")
        
        # Test monitoring statistics
        stats = monitor.get_alert_statistics()
        assert "total_alerts" in stats
        assert "monitoring_enabled" in stats
        print(f"  📊 Monitoring stats: {stats['total_alerts']} total alerts")
        
        # Test monitoring health
        health = monitor.get_monitoring_health()
        assert health["health_status"] in ["healthy", "warning", "unhealthy"]
        print(f"  🏥 Monitoring health: {health['health_status']}")
        
    print("  ✅ Risk monitoring system tests passed")


def test_risk_workflow_integration():
    """Test risk workflow integration with CI/CD processes."""
    print("🧪 Testing Risk Workflow Integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.risk import RiskRegistry, RiskAssessor
        from agentnet.risk.mitigation import MitigationMitigator
        from agentnet.risk.monitoring import RiskMonitor
        from agentnet.risk.workflow import (
            RiskWorkflow, WorkflowContext, WorkflowStage, RiskDecision
        )
        
        # Set up complete risk management system
        registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        assessor = RiskAssessor()
        mitigator = MitigationMitigator()
        monitor = RiskMonitor()
        
        workflow = RiskWorkflow(registry, assessor, mitigator, monitor)
        
        # Test pre-commit workflow stage
        pre_commit_context = WorkflowContext(
            stage=WorkflowStage.PRE_COMMIT,
            environment="production",
            change_type="feature",
            commit_sha="abc123",
            files_changed=["security/auth.py", "core/engine.py"],
            lines_changed=250,
            test_coverage=0.85,
            business_hours=True,
        )
        
        pre_commit_result = workflow.assess_workflow_stage(pre_commit_context)
        assert pre_commit_result.stage == WorkflowStage.PRE_COMMIT
        assert pre_commit_result.decision in list(RiskDecision)
        print(f"  🔄 Pre-commit assessment: {pre_commit_result.decision.value} (risk: {pre_commit_result.risk_score:.3f})")
        
        # Test security scan workflow stage  
        security_context = WorkflowContext(
            stage=WorkflowStage.SECURITY_SCAN,
            environment="production",
            change_type="feature",
            commit_sha="abc123",
            security_scan_results={
                "critical_vulnerabilities": 1,
                "high_vulnerabilities": 3,
            },
        )
        
        security_result = workflow.assess_workflow_stage(security_context)
        assert security_result.stage == WorkflowStage.SECURITY_SCAN
        print(f"  🔒 Security scan assessment: {security_result.decision.value}")
        
        # Should block due to critical vulnerability
        if security_context.security_scan_results["critical_vulnerabilities"] > 0:
            assert security_result.decision == RiskDecision.BLOCK
            print("  🛑 Correctly blocked deployment due to critical vulnerability")
        
        # Test production deployment workflow stage
        prod_context = WorkflowContext(
            stage=WorkflowStage.DEPLOY_PRODUCTION,
            environment="production", 
            change_type="feature",
            commit_sha="abc123",
            system_health={"cpu_usage": 0.4, "overall_health": "healthy"},
            active_incidents=[],
            business_hours=True,
        )
        
        prod_result = workflow.assess_workflow_stage(prod_context)
        assert prod_result.stage == WorkflowStage.DEPLOY_PRODUCTION
        print(f"  🚀 Production deployment assessment: {prod_result.decision.value}")
        
        # Test workflow statistics
        stats = workflow.get_workflow_statistics()
        assert stats["total_workflows"] >= 3  # We ran 3 assessments
        print(f"  📊 Workflow stats: {stats['total_workflows']} assessments")
        
        # Test approval request (if needed)
        if prod_result.decision == RiskDecision.REQUIRE_APPROVAL:
            # Mock approval
            prod_result.required_approvers = ["admin"]
            approval_success = workflow.request_approval(
                prod_result, 
                "admin",
                "Emergency hotfix deployment approved"
            )
            print(f"  ✅ Approval request: {'approved' if approval_success else 'denied'}")
        
    print("  ✅ Risk workflow integration tests passed")


def test_end_to_end_risk_cost_integration():
    """Test complete integration of cost tracking and risk management."""
    print("🧪 Testing End-to-End Risk-Cost Integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet import AgentNet, ExampleEngine
        from agentnet.core.cost import CostRecorder, CostPredictor, CostOptimizer
        from agentnet.risk import RiskRegistry, RiskCategory, RiskLevel
        from agentnet.risk.workflow import RiskWorkflow, WorkflowContext, WorkflowStage
        from agentnet.risk.assessment import RiskAssessor
        from agentnet.risk.mitigation import MitigationMitigator
        from agentnet.risk.monitoring import RiskMonitor
        
        # Set up integrated system
        cost_recorder = CostRecorder(storage_dir=str(Path(tmpdir) / "costs"))
        cost_predictor = CostPredictor(cost_recorder)
        cost_optimizer = CostOptimizer(cost_recorder, cost_predictor)
        
        risk_registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        risk_assessor = RiskAssessor() 
        mitigator = MitigationMitigator()
        monitor = RiskMonitor()
        
        workflow = RiskWorkflow(risk_registry, risk_assessor, mitigator, monitor)
        
        # Create agent with full integration
        engine = ExampleEngine()
        agent = AgentNet(
            name="IntegratedTestAgent",
            style={"logic": 0.8, "creativity": 0.6},
            engine=engine,
            cost_recorder=cost_recorder,
            tenant_id="integration-test",
        )
        
        # Simulate high-cost operations that trigger cost spike risk
        high_cost_tasks = [
            "Perform complex system analysis with detailed modeling",
            "Generate comprehensive architecture documentation with diagrams",
            "Conduct thorough security assessment with multiple scenarios",
            "Create detailed performance optimization recommendations",
            "Develop complete integration testing strategy with edge cases",
        ]
        
        total_cost = 0.0
        for task in high_cost_tasks:
            result = agent.generate_reasoning_tree(task)
            if "cost_record" in result and result["cost_record"]:
                total_cost += result["cost_record"]["total_cost"]
            time.sleep(0.1)
        
        print(f"  💰 Generated ${total_cost:.6f} in costs")
        
        # Test cost prediction integration with risk assessment
        monthly_prediction = cost_predictor.predict_monthly_cost(tenant_id="integration-test")
        print(f"  📈 Monthly cost prediction: ${monthly_prediction.predicted_cost:.6f}")
        
        # Default cost context for monitoring
        cost_context = {
            "cost_trend": "stable",
            "monthly_budget": 50.0,
            "potential_cost_overrun_percent": 5.0,
            "business_criticality": "medium",
        }
        
        # Create cost spike risk based on prediction
        if monthly_prediction.predicted_cost > 10.0:  # Arbitrary threshold
            cost_spike_risk = risk_registry.get_risks_by_category(RiskCategory.COST_SPIKE)[0]
            
            # Assess risk with cost context
            cost_context = {
                "cost_trend": "increasing" if monthly_prediction.trend == "increasing" else "stable",
                "monthly_budget": 50.0,
                "potential_cost_overrun_percent": (monthly_prediction.predicted_cost / 50.0) * 100,
                "business_criticality": "high",
            }
            
            assessment = risk_assessor.assess_risk(cost_spike_risk, cost_context)
            print(f"  🚨 Cost spike risk assessed: {assessment.current_risk_score:.3f} score")
            
            # Test workflow integration with cost-based decisions
            workflow_context = WorkflowContext(
                stage=WorkflowStage.DEPLOY_PRODUCTION,
                environment="production",
                change_type="feature",
                commit_sha="cost123",
                metadata={
                    "current_monthly_cost": monthly_prediction.predicted_cost,
                    "cost_trend": monthly_prediction.trend,
                    "budget_utilization": (monthly_prediction.predicted_cost / 50.0),
                },
            )
            
            workflow_result = workflow.assess_workflow_stage(workflow_context)
            print(f"  🔄 Workflow decision with cost context: {workflow_result.decision.value}")
            
            # Generate cost optimization recommendations
            recommendations = cost_optimizer.generate_recommendations(tenant_id="integration-test")
            print(f"  💡 Cost optimization recommendations: {len(recommendations)}")
            
            if recommendations:
                top_recommendation = recommendations[0]
                print(f"    Top recommendation: {top_recommendation.category} - ${top_recommendation.estimated_savings:.4f} savings")
        
        # Test monitoring integration
        cost_risks = risk_registry.get_risks_by_category(RiskCategory.COST_SPIKE)
        if cost_risks:
            alerts = monitor.check_risk(cost_risks[0], cost_context)
            print(f"  🚨 Generated {len(alerts)} cost-related alerts")
        
        # Verify data persistence across systems
        cost_summary = cost_recorder.get_records(tenant_id="integration-test")
        risk_stats = risk_registry.get_risk_statistics()
        
        assert len(cost_summary) >= len(high_cost_tasks)
        assert risk_stats["total_risks"] > 0
        
        print(f"  💾 Persisted {len(cost_summary)} cost records and {risk_stats['total_risks']} risks")
        
    print("  ✅ End-to-end risk-cost integration tests passed")


if __name__ == "__main__":
    print("🧪 Running Enhanced Cost Tracking and Risk Management Tests...")
    print("=" * 70)
    
    try:
        # Cost tracking enhancement tests
        test_cost_predictive_modeling()
        print()
        
        # Risk management system tests
        test_risk_registry_system()
        print()
        test_risk_assessment_system()
        print()
        test_risk_mitigation_system()
        print()
        test_risk_monitoring_system()
        print()
        test_risk_workflow_integration()
        print()
        
        # Integration tests
        test_end_to_end_risk_cost_integration()
        print()
        
        print("=" * 70)
        print("✅ All enhanced cost tracking and risk management tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)