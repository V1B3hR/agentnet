#!/usr/bin/env python3
"""
AgentNet Enhanced Features Demo

Demonstrates the new cost tracking enhancements and risk management system.
"""

import tempfile
import time
from pathlib import Path

def demo_cost_enhancements():
    """Demonstrate enhanced cost tracking with predictive modeling."""
    print("🔥 Cost Tracking Enhancements Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet import AgentNet, ExampleEngine
        from agentnet.core.cost import (
            CostRecorder, CostPredictor, CostOptimizer, CostReporter
        )
        
        # Setup enhanced cost system
        recorder = CostRecorder(storage_dir=tmpdir)
        predictor = CostPredictor(recorder)
        optimizer = CostOptimizer(recorder, predictor)
        reporter = CostReporter(recorder, predictor, optimizer)
        
        # Create agent with cost tracking
        engine = ExampleEngine()
        agent = AgentNet(
            name="DemoAgent",
            style={"logic": 0.8, "creativity": 0.7},
            engine=engine,
            cost_recorder=recorder,
            tenant_id="demo-tenant",
        )
        
        print("💰 Generating cost data...")
        tasks = [
            "Design system architecture",
            "Implement security protocols", 
            "Optimize database performance",
            "Create monitoring dashboards",
            "Document API interfaces",
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"  {i}/5: {task}")
            result = agent.generate_reasoning_tree(task)
            time.sleep(0.1)
        
        # Demonstrate predictive modeling
        print("\n📈 Cost Prediction:")
        prediction = predictor.predict_monthly_cost(tenant_id="demo-tenant")
        print(f"  Monthly forecast: ${prediction.predicted_cost:.6f}")
        print(f"  Trend: {prediction.trend}")
        print(f"  Confidence: {prediction.model_accuracy:.1%}")
        
        # Demonstrate optimization
        print("\n💡 Optimization Recommendations:")
        recommendations = optimizer.generate_recommendations(tenant_id="demo-tenant")
        if recommendations:
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec.category}: ${rec.estimated_savings:.4f} savings ({rec.priority} priority)")
        else:
            print("  No optimization opportunities identified (low usage)")
        
        # Generate comprehensive report
        print("\n📊 Comprehensive Report:")
        report = reporter.generate_comprehensive_report(tenant_id="demo-tenant")
        print(f"  Total cost: ${report['cost_summary']['total_cost']:.6f}")
        print(f"  Records: {report['cost_summary']['record_count']}")
        print(f"  Key insights: {len(report['key_insights'])}")
        for insight in report['key_insights']:
            print(f"    • {insight}")


def demo_risk_management():
    """Demonstrate comprehensive risk management system."""
    print("\n\n🛡️  Risk Management System Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.risk import (
            RiskRegistry, RiskAssessor, RiskCategory, RiskLevel
        )
        from agentnet.risk.mitigation import MitigationMitigator
        from agentnet.risk.monitoring import RiskMonitor, AlertChannel
        from agentnet.risk.workflow import (
            RiskWorkflow, WorkflowContext, WorkflowStage, RiskDecision
        )
        
        # Setup risk management system
        registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        assessor = RiskAssessor()
        mitigator = MitigationMitigator()
        monitor = RiskMonitor()
        workflow = RiskWorkflow(registry, assessor, mitigator, monitor)
        
        print("📋 Risk Registry:")
        stats = registry.get_risk_statistics()
        print(f"  Total risks: {stats['total_risks']}")
        print(f"  High priority: {stats.get('high_priority_risks', 0)}")
        
        # Demonstrate risk categories
        print("\n🔍 Risk Categories:")
        for category in RiskCategory:
            risks = registry.get_risks_by_category(category)
            if risks:
                print(f"  {category.value}: {len(risks)} risks")
        
        # Demonstrate risk assessment
        print("\n🎯 Risk Assessment Demo:")
        provider_risks = registry.get_risks_by_category(RiskCategory.PROVIDER_OUTAGE)
        if provider_risks:
            risk = provider_risks[0]
            context = {
                "recent_provider_outages": 1,
                "active_providers": 1,
                "business_criticality": "high",
                "active_users": 1000,
            }
            
            assessment = assessor.assess_risk(risk, context)
            print(f"  Risk: {risk.title}")
            print(f"  Current score: {assessment.current_risk_score:.3f}")
            print(f"  Level: {assessment.current_level.value}")
            print(f"  Priority: {assessment.priority_score:.0f}/100")
            print(f"  Recommended actions: {len(assessment.recommended_actions)}")
        
        # Demonstrate mitigation
        print("\n⚡ Mitigation Strategy:")
        if provider_risks:
            strategy = mitigator.create_strategy_for_risk(provider_risks[0])
            print(f"  Strategy: {strategy.name}")
            print(f"  Actions: {len(strategy.actions)}")
            
            # Execute first action
            if strategy.actions:
                action = strategy.actions[0]
                success = mitigator.execute_action(action.action_id)
                print(f"  Executed: {action.title} ({'✅' if success else '❌'})")
                print(f"  Progress: {strategy.calculate_progress():.1%}")
        
        # Demonstrate monitoring
        print("\n🚨 Risk Monitoring:")
        monitoring_stats = monitor.get_alert_statistics()
        print(f"  Monitoring rules: {monitoring_stats['monitoring_rules_count']}")
        print(f"  Total alerts: {monitoring_stats['total_alerts']}")
        
        # Demonstrate workflow integration
        print("\n🔄 CI/CD Workflow Integration:")
        workflow_context = WorkflowContext(
            stage=WorkflowStage.DEPLOY_PRODUCTION,
            environment="production",
            change_type="feature",
            commit_sha="demo123",
            files_changed=["security/auth.py"],
            lines_changed=50,
            business_hours=True,
        )
        
        result = workflow.assess_workflow_stage(workflow_context)
        print(f"  Stage: {result.stage.value}")
        print(f"  Decision: {result.decision.value}")
        print(f"  Risk score: {result.risk_score:.3f}")
        print(f"  Risks identified: {len(result.risks_identified)}")
        
        if result.recommendations:
            print("  Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"    • {rec}")


def demo_integration():
    """Demonstrate integration between cost tracking and risk management."""
    print("\n\n🔗 Cost-Risk Integration Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet import AgentNet, ExampleEngine
        from agentnet.core.cost import CostRecorder, CostPredictor
        from agentnet.risk import RiskRegistry, RiskCategory, RiskAssessor
        from agentnet.risk.workflow import WorkflowContext, WorkflowStage, RiskWorkflow
        from agentnet.risk.mitigation import MitigationMitigator
        from agentnet.risk.monitoring import RiskMonitor
        
        # Setup integrated systems
        cost_recorder = CostRecorder(storage_dir=str(Path(tmpdir) / "costs"))
        cost_predictor = CostPredictor(cost_recorder)
        
        risk_registry = RiskRegistry(storage_path=str(Path(tmpdir) / "risks.json"))
        risk_assessor = RiskAssessor()
        mitigator = MitigationMitigator()
        monitor = RiskMonitor()
        
        workflow = RiskWorkflow(risk_registry, risk_assessor, mitigator, monitor)
        
        # Create cost-aware agent
        engine = ExampleEngine()
        agent = AgentNet(
            name="IntegratedAgent",
            style={"logic": 0.9, "creativity": 0.5},
            engine=engine,
            cost_recorder=cost_recorder,
            tenant_id="integration-demo",
        )
        
        print("🔄 Running cost-generating operations...")
        expensive_tasks = [
            "Comprehensive security audit with detailed analysis",
            "Full system architecture review and optimization recommendations",
            "Complete API documentation with examples and test cases",
        ]
        
        for task in expensive_tasks:
            agent.generate_reasoning_tree(task)
            time.sleep(0.1)
        
        # Analyze cost impact on risk
        prediction = cost_predictor.predict_monthly_cost(tenant_id="integration-demo")
        cost_risks = risk_registry.get_risks_by_category(RiskCategory.COST_SPIKE)
        
        print(f"💰 Predicted monthly cost: ${prediction.predicted_cost:.6f}")
        
        if cost_risks and prediction.predicted_cost > 0:
            cost_risk = cost_risks[0]
            
            # Risk assessment with cost context
            cost_context = {
                "cost_trend": prediction.trend,
                "monthly_budget": 10.0,
                "business_criticality": "high",
                "potential_cost_overrun_percent": (prediction.predicted_cost / 10.0) * 100,
            }
            
            assessment = risk_assessor.assess_risk(cost_risk, cost_context)
            print(f"🚨 Cost spike risk score: {assessment.current_risk_score:.3f}")
            
            # Workflow decision with cost considerations
            workflow_context = WorkflowContext(
                stage=WorkflowStage.DEPLOY_PRODUCTION,
                environment="production",
                change_type="feature",
                commit_sha="cost456",
                metadata={
                    "predicted_cost": prediction.predicted_cost,
                    "cost_trend": prediction.trend,
                    "budget_utilization": prediction.predicted_cost / 10.0,
                },
            )
            
            workflow_result = workflow.assess_workflow_stage(workflow_context)
            print(f"🔄 Deployment decision: {workflow_result.decision.value}")
            
            if workflow_result.warnings:
                print("⚠️  Warnings:")
                for warning in workflow_result.warnings:
                    print(f"    • {warning}")


if __name__ == "__main__":
    print("🚀 AgentNet Enhanced Features Demo")
    print("Showcasing cost tracking enhancements and risk management integration")
    print()
    
    try:
        demo_cost_enhancements()
        demo_risk_management()
        demo_integration()
        
        print("\n\n✨ Demo Complete!")
        print("All enhanced features are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()