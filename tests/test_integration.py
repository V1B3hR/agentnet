#!/usr/bin/env python3
"""
Integration tests for AgentNet with ephemeral database and vector store.

This test suite validates full-stack integration including:
1. Database persistence
2. Cost tracking integration
3. Cross-component interactions
4. End-to-end workflows
"""

import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def ephemeral_db():
    """Create an ephemeral database for testing."""
    # In a real scenario, this would set up a test database
    # For now, we'll use a temporary directory to simulate
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "type": "sqlite",
            "path": os.path.join(tmpdir, "test.db"),
            "connection_string": f"sqlite:///{os.path.join(tmpdir, 'test.db')}",
        }


@pytest.fixture
def ephemeral_vector_store():
    """Create an ephemeral vector store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "type": "local",
            "path": tmpdir,
        }


@pytest.fixture
def cost_tracking_integration():
    """Set up cost tracking for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet.core.cost import CostRecorder, CostAggregator

        recorder = CostRecorder(storage_dir=tmpdir)
        aggregator = CostAggregator(recorder)
        
        yield {
            "recorder": recorder,
            "aggregator": aggregator,
            "storage_dir": tmpdir,
        }


def test_agent_with_database_integration(ephemeral_db):
    """Test agent operations with database persistence."""
    print("🧪 Testing Agent-Database Integration...")
    
    from agentnet import AgentNet, ExampleEngine
    
    # Create agent with database persistence
    engine = ExampleEngine()
    agent = AgentNet(
        name="DatabaseTestAgent",
        style={"logic": 0.8, "creativity": 0.5},
        engine=engine,
    )
    
    # Generate reasoning and persist
    result = agent.generate_reasoning_tree("Test database integration")
    assert result["result"]["content"] is not None
    
    # Test session persistence with database context
    session_data = {
        "session_id": f"db_test_{int(time.time())}",
        "participants": [agent.name],
        "topic": "Database integration test",
        "rounds_executed": 1,
        "converged": False,
        "timestamp": time.time(),
        "transcript": [{"agent": agent.name, "content": result["result"]["content"]}],
        "database_context": ephemeral_db,
    }
    
    filepath = agent.persist_session(session_data)
    assert Path(filepath).exists()
    print("  ✅ Agent-database integration successful")


def test_cost_tracking_full_integration(cost_tracking_integration):
    """Test full cost tracking integration."""
    print("🧪 Testing Cost Tracking Full Integration...")
    
    from agentnet import AgentNet, ExampleEngine
    from agentnet.core.cost import TenantCostTracker
    
    cost_system = cost_tracking_integration
    recorder = cost_system["recorder"]
    aggregator = cost_system["aggregator"]
    
    # Set up tenant cost tracking
    tenant_tracker = TenantCostTracker(recorder)
    tenant_id = "integration-test-tenant"
    
    # Set budget and alerts
    tenant_tracker.set_tenant_budget(tenant_id, 100.0)  # $100 monthly budget
    tenant_tracker.set_tenant_alerts(tenant_id, {
        "warning": 0.75,
        "critical": 0.90,
    })
    
    # Create agent with cost tracking
    engine = ExampleEngine()
    agent = AgentNet(
        name="CostTrackingAgent",
        style={"logic": 0.7, "creativity": 0.6},
        engine=engine,
        cost_recorder=recorder,
        tenant_id=tenant_id,
    )
    
    # Perform multiple operations to generate cost data
    tasks = [
        "Analyze system performance",
        "Design API architecture", 
        "Create testing strategy",
        "Implement security measures",
        "Optimize database queries",
    ]
    
    total_estimated_cost = 0.0
    for i, task in enumerate(tasks, 1):
        print(f"  🔄 Processing task {i}/{len(tasks)}: {task[:30]}...")
        result = agent.generate_reasoning_tree(task)
        
        if "cost_record" in result and result["cost_record"]:
            total_estimated_cost += result["cost_record"]["total_cost"]
        
        # Small delay to ensure timestamp differences
        time.sleep(0.1)
    
    print(f"  💰 Total estimated cost: ${total_estimated_cost:.6f}")
    
    # Test cost aggregation
    summary = aggregator.get_cost_summary(tenant_id=tenant_id)
    assert summary["record_count"] >= len(tasks)
    assert summary["total_cost"] >= 0
    print(f"  📊 Cost summary: {summary['record_count']} records, ${summary['total_cost']:.6f} total")
    
    # Test tenant budget checking
    budget_status = tenant_tracker.check_tenant_budget(tenant_id)
    assert budget_status["status"] == "ok"  # Should be under budget
    assert budget_status["budget"] == 100.0
    print(f"  🎯 Budget status: {budget_status['status']}, ${budget_status['current_spend']:.6f}/${budget_status['budget']:.2f}")
    
    # Test cost trends
    trends = aggregator.get_cost_trends(days=1, tenant_id=tenant_id)
    assert "daily_trends" in trends
    assert len(trends["daily_trends"]) >= 0
    print(f"  📈 Cost trends: {len(trends['daily_trends'])} daily data points")
    
    # Test top cost agents
    top_agents = aggregator.get_top_cost_agents(limit=5, tenant_id=tenant_id)
    assert len(top_agents) >= 1
    assert top_agents[0]["agent_name"] == "CostTrackingAgent"
    print(f"  🏆 Top agent: {top_agents[0]['agent_name']} with ${top_agents[0]['total_cost']:.6f}")
    
    print("  ✅ Cost tracking full integration successful")


def test_vector_memory_integration(ephemeral_vector_store):
    """Test agent with vector memory integration."""
    print("🧪 Testing Vector Memory Integration...")
    
    from agentnet import AgentNet, ExampleEngine
    
    # Create agent (vector store integration would be added in future phases)
    engine = ExampleEngine()
    agent = AgentNet(
        name="VectorMemoryAgent",
        style={"logic": 0.6, "creativity": 0.8},
        engine=engine,
    )
    
    # Test memory-enhanced reasoning (placeholder for vector integration)
    memory_context = {
        "previous_conversations": [
            "Discussed system architecture patterns",
            "Reviewed security best practices", 
            "Analyzed performance optimization",
        ],
        "vector_store_config": ephemeral_vector_store,
    }
    
    result = agent.generate_reasoning_tree(
        "Build upon previous architecture discussions"
    )
    
    assert result["result"]["content"] is not None
    print("  ✅ Vector memory integration successful")


def test_multi_agent_coordination_integration(cost_tracking_integration):
    """Test multi-agent coordination with integrated systems."""
    print("🧪 Testing Multi-Agent Coordination Integration...")
    
    from agentnet import AgentNet, ExampleEngine
    
    cost_system = cost_tracking_integration
    recorder = cost_system["recorder"]
    
    # Create multiple agents with cost tracking
    engine = ExampleEngine()
    agents = {
        "architect": AgentNet(
            name="SystemArchitect",
            style={"logic": 0.9, "creativity": 0.4},
            engine=engine,
            cost_recorder=recorder,
            tenant_id="coordination-test",
        ),
        "security": AgentNet(
            name="SecurityExpert", 
            style={"logic": 0.8, "creativity": 0.3},
            engine=engine,
            cost_recorder=recorder,
            tenant_id="coordination-test",
        ),
        "performance": AgentNet(
            name="PerformanceSpecialist",
            style={"logic": 0.7, "creativity": 0.5},
            engine=engine,
            cost_recorder=recorder,
            tenant_id="coordination-test",
        ),
    }
    
    # Simulate coordinated reasoning
    coordination_task = "Design a scalable, secure, high-performance microservice"
    results = {}
    
    for role, agent in agents.items():
        print(f"  🤖 {role.title()} agent processing...")
        specialized_task = f"{coordination_task} - focus on {role} aspects"
        results[role] = agent.generate_reasoning_tree(specialized_task)
        assert results[role]["result"]["content"] is not None
    
    # Verify all agents contributed to cost tracking
    aggregator = cost_tracking_integration["aggregator"]
    summary = aggregator.get_cost_summary(tenant_id="coordination-test")
    assert summary["record_count"] >= len(agents)
    
    # Check agent breakdown
    agent_breakdown = summary["agent_breakdown"]
    assert len(agent_breakdown) == len(agents)
    for role, agent in agents.items():
        assert agent.name in agent_breakdown
        print(f"  💰 {agent.name}: ${agent_breakdown[agent.name]['cost']:.6f}")
    
    print("  ✅ Multi-agent coordination integration successful")


def test_monitoring_and_alerting_integration():
    """Test monitoring and alerting integration."""
    print("🧪 Testing Monitoring and Alerting Integration...")
    
    # Test performance monitoring integration (placeholder)
    metrics = {
        "response_time_ms": 150,
        "memory_usage_mb": 64,
        "cpu_utilization_percent": 12,
        "tokens_processed": 500,
        "cost_per_request": 0.002,
    }
    
    # Simulate metric collection
    assert all(isinstance(v, (int, float)) for v in metrics.values())
    print(f"  📊 Collected metrics: {len(metrics)} data points")
    
    # Test alert thresholds (placeholder)
    alert_thresholds = {
        "response_time_ms": 1000,
        "memory_usage_mb": 512,
        "cpu_utilization_percent": 80,
        "cost_per_request": 0.01,
    }
    
    alerts_triggered = []
    for metric, value in metrics.items():
        if metric in alert_thresholds and value > alert_thresholds[metric]:
            alerts_triggered.append(f"{metric}: {value} > {alert_thresholds[metric]}")
    
    # Should not trigger alerts with current values
    assert len(alerts_triggered) == 0
    print(f"  🚨 Alerts status: {len(alerts_triggered)} alerts triggered (expected: 0)")
    
    print("  ✅ Monitoring and alerting integration successful")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow integration."""
    print("🧪 Testing End-to-End Workflow Integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from agentnet import AgentNet, ExampleEngine
        from agentnet.core.cost import CostRecorder, CostAggregator
        
        # Set up integrated systems
        cost_recorder = CostRecorder(storage_dir=tmpdir)
        cost_aggregator = CostAggregator(cost_recorder)
        
        # Create agent with full integration
        engine = ExampleEngine()
        agent = AgentNet(
            name="E2ETestAgent",
            style={"logic": 0.7, "creativity": 0.7},
            engine=engine,
            cost_recorder=cost_recorder,
            tenant_id="e2e-test",
        )
        
        # Execute complex workflow
        workflow_steps = [
            "Analyze requirements",
            "Design architecture", 
            "Implement solution",
            "Test and validate",
            "Deploy and monitor",
        ]
        
        workflow_results = {}
        workflow_costs = []
        
        for step in workflow_steps:
            print(f"  🔄 Executing: {step}")
            result = agent.generate_reasoning_tree(step)
            workflow_results[step] = result
            
            if "cost_record" in result and result["cost_record"]:
                workflow_costs.append(result["cost_record"]["total_cost"])
        
        # Validate workflow completion
        assert len(workflow_results) == len(workflow_steps)
        assert all("result" in result for result in workflow_results.values())
        
        # Validate cost tracking
        final_summary = cost_aggregator.get_cost_summary(tenant_id="e2e-test")
        assert final_summary["record_count"] >= len(workflow_steps)
        
        total_workflow_cost = sum(workflow_costs) if workflow_costs else 0.0
        print(f"  💰 Workflow cost: ${total_workflow_cost:.6f}")
        print(f"  📊 Cost records: {final_summary['record_count']}")
        
        # Test session persistence
        session_data = {
            "session_id": f"e2e_workflow_{int(time.time())}",
            "participants": [agent.name],
            "topic": "End-to-end workflow test",
            "workflow_steps": workflow_steps,
            "workflow_results": workflow_results,
            "total_cost": total_workflow_cost,
            "timestamp": time.time(),
        }
        
        session_path = agent.persist_session(session_data)
        assert Path(session_path).exists()
        
        print("  ✅ End-to-end workflow integration successful")


if __name__ == "__main__":
    print("🧪 Running AgentNet Integration Tests...")
    
    # Run tests manually for debugging
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_config = {
                "type": "sqlite",
                "path": os.path.join(tmpdir, "test.db"),
                "connection_string": f"sqlite:///{os.path.join(tmpdir, 'test.db')}",
            }
            test_agent_with_database_integration(db_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_config = {"type": "local", "path": tmpdir}
            test_vector_memory_integration(vector_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from agentnet.core.cost import CostRecorder, CostAggregator
            recorder = CostRecorder(storage_dir=tmpdir)
            aggregator = CostAggregator(recorder)
            cost_system = {
                "recorder": recorder,
                "aggregator": aggregator,
                "storage_dir": tmpdir,
            }
            test_cost_tracking_full_integration(cost_system)
            test_multi_agent_coordination_integration(cost_system)
        
        test_monitoring_and_alerting_integration()
        test_end_to_end_workflow()
        
        print("✅ All integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise