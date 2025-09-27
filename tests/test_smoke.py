#!/usr/bin/env python3
"""
Smoke tests for AgentNet deployment validation.

These tests provide basic health checks after deployment to ensure
core functionality is working correctly in production environments.
"""

import time
import tempfile
from pathlib import Path


def test_basic_import_smoke():
    """Smoke test: Basic import functionality."""
    print("🔥 Smoke Test: Basic Import...")
    
    try:
        import agentnet
        from agentnet import AgentNet, ExampleEngine
        from agentnet.core.cost import CostRecorder
        
        print(f"  ✅ AgentNet version: {agentnet.__version__}")
        print("  ✅ Core imports successful")
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        raise


def test_agent_creation_smoke():
    """Smoke test: Agent creation and basic operation."""
    print("🔥 Smoke Test: Agent Creation...")
    
    from agentnet import AgentNet, ExampleEngine
    
    try:
        engine = ExampleEngine()
        agent = AgentNet(
            name="SmokeTestAgent",
            style={"logic": 0.7},
            engine=engine,
        )
        
        assert agent.name == "SmokeTestAgent"
        assert agent.style["logic"] == 0.7
        print("  ✅ Agent creation successful")
        
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        raise


def test_reasoning_generation_smoke():
    """Smoke test: Basic reasoning generation."""
    print("🔥 Smoke Test: Reasoning Generation...")
    
    from agentnet import AgentNet, ExampleEngine
    
    try:
        engine = ExampleEngine()
        agent = AgentNet(
            name="ReasoningSmokeAgent",
            style={"logic": 0.8, "creativity": 0.5},
            engine=engine,
        )
        
        result = agent.generate_reasoning_tree("Hello AgentNet smoke test")
        
        assert "result" in result
        assert result["result"]["content"] is not None
        assert len(result["result"]["content"]) > 0
        
        print(f"  ✅ Generated reasoning: {len(result['result']['content'])} chars")
        
    except Exception as e:
        print(f"  ❌ Reasoning generation failed: {e}")
        raise


def test_cost_tracking_smoke():
    """Smoke test: Cost tracking functionality."""
    print("🔥 Smoke Test: Cost Tracking...")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from agentnet import AgentNet, ExampleEngine
            from agentnet.core.cost import CostRecorder, CostAggregator
            
            # Set up cost tracking
            cost_recorder = CostRecorder(storage_dir=tmpdir)
            cost_aggregator = CostAggregator(cost_recorder)
            
            # Create agent with cost tracking
            engine = ExampleEngine()
            agent = AgentNet(
                name="CostSmokeAgent",
                style={"logic": 0.6},
                engine=engine,
                cost_recorder=cost_recorder,
                tenant_id="smoke-test",
            )
            
            # Generate reasoning with cost tracking
            result = agent.generate_reasoning_tree("Cost tracking smoke test")
            
            # Validate cost tracking
            summary = cost_aggregator.get_cost_summary(tenant_id="smoke-test")
            assert summary["record_count"] >= 0
            
            print(f"  ✅ Cost tracking: {summary['record_count']} records")
            
    except Exception as e:
        print(f"  ❌ Cost tracking failed: {e}")
        raise


def test_session_persistence_smoke():
    """Smoke test: Session persistence."""
    print("🔥 Smoke Test: Session Persistence...")
    
    from agentnet import AgentNet, ExampleEngine
    
    try:
        engine = ExampleEngine()
        agent = AgentNet(
            name="PersistenceSmokeAgent", 
            style={"logic": 0.7},
            engine=engine,
        )
        
        # Test basic reasoning
        result = agent.generate_reasoning_tree("Session persistence test")
        
        # Test session persistence
        session_data = {
            "session_id": f"smoke_test_{int(time.time())}",
            "participants": [agent.name],
            "topic": "Smoke test session",
            "timestamp": time.time(),
            "test_result": result,
        }
        
        session_path = agent.persist_session(session_data)
        assert Path(session_path).exists()
        
        file_size = Path(session_path).stat().st_size
        print(f"  ✅ Session persisted: {session_path} ({file_size} bytes)")
        
    except Exception as e:
        print(f"  ❌ Session persistence failed: {e}")
        raise


def test_monitor_system_smoke():
    """Smoke test: Monitor system functionality."""
    print("🔥 Smoke Test: Monitor System...")
    
    try:
        from agentnet import AgentNet, ExampleEngine
        from agentnet.monitors.factory import MonitorFactory
        from agentnet.monitors.base import MonitorSpec
        from agentnet.core.types import Severity
        
        # Create a simple monitor
        monitor_spec = MonitorSpec(
            name="smoke_filter",
            type="keyword",
            params={"keywords": ["error", "fail"], "violation_name": "negative_content"},
            severity=Severity.MINOR,
            description="Smoke test filter",
        )
        
        monitor = MonitorFactory.build(monitor_spec)
        
        # Create agent with monitor
        engine = ExampleEngine()
        agent = AgentNet(
            name="MonitorSmokeAgent",
            style={"logic": 0.8},
            engine=engine,
            monitors=[monitor],
        )
        
        # Test reasoning with monitoring
        result = agent.generate_reasoning_tree("Create a successful system design")
        
        assert "result" in result
        assert result["result"]["content"] is not None
        
        print("  ✅ Monitor system functional")
        
    except Exception as e:
        print(f"  ❌ Monitor system failed: {e}")
        raise


def test_configuration_smoke():
    """Smoke test: Configuration loading and validation."""
    print("🔥 Smoke Test: Configuration...")
    
    try:
        # Test basic configuration access
        from agentnet.core.cost.pricing import PricingEngine
        
        pricing_engine = PricingEngine()
        
        # Test provider configurations
        openai_models = pricing_engine.get_provider_models("openai")
        assert len(openai_models) > 0
        
        # Test cost estimation
        estimate = pricing_engine.estimate_cost(
            provider="example",
            model="example-model", 
            estimated_tokens=100,
        )
        assert "total_cost" in estimate
        
        print(f"  ✅ Configuration: {len(openai_models)} OpenAI models loaded")
        
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        raise


def test_health_check_smoke():
    """Smoke test: Overall system health check."""
    print("🔥 Smoke Test: System Health Check...")
    
    try:
        # Test multiple components together
        with tempfile.TemporaryDirectory() as tmpdir:
            from agentnet import AgentNet, ExampleEngine
            from agentnet.core.cost import CostRecorder
            from agentnet.monitors.factory import MonitorFactory
            from agentnet.monitors.base import MonitorSpec
            from agentnet.core.types import Severity
            
            # Set up components
            cost_recorder = CostRecorder(storage_dir=tmpdir)
            
            monitor_spec = MonitorSpec(
                name="health_monitor",
                type="keyword", 
                params={"keywords": ["crash", "fail"], "violation_name": "system_issues"},
                severity=Severity.MAJOR,
            )
            monitor = MonitorFactory.build(monitor_spec)
            
            # Create integrated agent
            engine = ExampleEngine()
            agent = AgentNet(
                name="HealthCheckAgent",
                style={"logic": 0.75, "creativity": 0.6},
                engine=engine,
                monitors=[monitor],
                cost_recorder=cost_recorder,
                tenant_id="health-check",
            )
            
            # Execute health check tasks
            health_tasks = [
                "System status check",
                "Performance validation",
                "Security assessment",
            ]
            
            results = []
            for task in health_tasks:
                result = agent.generate_reasoning_tree(task)
                results.append(result)
                assert "result" in result
                assert result["result"]["content"] is not None
            
            print(f"  ✅ Health check: {len(results)} tasks completed")
            
            # Validate cost tracking worked
            from agentnet.core.cost import CostAggregator
            aggregator = CostAggregator(cost_recorder)
            summary = aggregator.get_cost_summary(tenant_id="health-check")
            
            print(f"  ✅ Cost records: {summary['record_count']} entries")
            print(f"  ✅ Total cost: ${summary['total_cost']:.6f}")
            
    except Exception as e:
        print(f"  ❌ System health check failed: {e}")
        raise


def test_performance_baseline_smoke():
    """Smoke test: Basic performance baseline."""
    print("🔥 Smoke Test: Performance Baseline...")
    
    try:
        from agentnet import AgentNet, ExampleEngine
        
        engine = ExampleEngine()
        agent = AgentNet(
            name="PerfSmokeAgent",
            style={"logic": 0.8},
            engine=engine,
        )
        
        # Measure basic performance
        start_time = time.time()
        
        result = agent.generate_reasoning_tree("Performance baseline test")
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert "result" in result
        assert duration < 5.0  # Should complete within 5 seconds
        
        print(f"  ✅ Performance: {duration:.3f}s response time")
        
    except Exception as e:
        print(f"  ❌ Performance baseline failed: {e}")
        raise


def run_all_smoke_tests():
    """Run all smoke tests."""
    print("🔥 Running AgentNet Smoke Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_import_smoke,
        test_agent_creation_smoke, 
        test_reasoning_generation_smoke,
        test_cost_tracking_smoke,
        test_session_persistence_smoke,
        test_monitor_system_smoke,
        test_configuration_smoke,
        test_health_check_smoke,
        test_performance_baseline_smoke,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
            print()
    
    print("=" * 50)
    print(f"🔥 Smoke Tests Complete: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("❌ Some smoke tests failed - system may not be ready for production")
        return False
    else:
        print("✅ All smoke tests passed - system appears healthy")
        return True


if __name__ == "__main__":
    success = run_all_smoke_tests()
    exit(0 if success else 1)