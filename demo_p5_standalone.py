#!/usr/bin/env python3
"""
Standalone P5 Observability Demo

Demonstrates P5 observability features without importing the full AgentNet package
to avoid dependency issues.
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add specific module paths to avoid importing main package
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import individual modules directly
def safe_import():
    """Safely import observability modules."""
    try:
        # Import metrics directly from file
        import importlib.util
        
        metrics_spec = importlib.util.spec_from_file_location(
            "metrics", current_dir / "agentnet/observability/metrics.py"
        )
        metrics_module = importlib.util.module_from_spec(metrics_spec)
        metrics_spec.loader.exec_module(metrics_module)
        
        tracing_spec = importlib.util.spec_from_file_location(
            "tracing", current_dir / "agentnet/observability/tracing.py"
        )
        tracing_module = importlib.util.module_from_spec(tracing_spec)
        tracing_spec.loader.exec_module(tracing_module)
        
        logging_spec = importlib.util.spec_from_file_location(
            "logging_obs", current_dir / "agentnet/observability/logging.py"
        )
        logging_module = importlib.util.spec_from_file_location(
            logging_spec.loader.exec_module(logging_module)
        )
        
        dashboard_spec = importlib.util.spec_from_file_location(
            "dashboard", current_dir / "agentnet/observability/dashboard.py"
        )
        dashboard_module = importlib.util.module_from_spec(dashboard_spec)
        dashboard_spec.loader.exec_module(dashboard_module)
        
        return metrics_module, tracing_module, logging_module, dashboard_module
    
    except Exception as e:
        print(f"Import failed: {e}")
        return None, None, None, None

def demo_standalone_metrics():
    """Demo metrics collection."""
    print("\n=== P5 Standalone Metrics Demo ===")
    
    try:
        # Import directly avoiding package init
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "metrics", current_dir / "agentnet/observability/metrics.py"
        )
        metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_module)
        
        # Test metrics
        metrics = metrics_module.AgentNetMetrics(enable_server=False)
        
        print("Recording sample metrics...")
        metrics.record_inference_latency(0.150, "gpt-3.5-turbo", "openai", "demo-agent")
        metrics.record_tokens_consumed(250, "gpt-3.5-turbo", "openai", "demo-tenant")
        metrics.record_violation("high", "token_limit_exceeded")
        metrics.record_cost(0.005, "openai", "gpt-3.5-turbo", "demo-tenant")
        
        local_metrics = metrics.get_local_metrics()
        print(f"✅ Collected {len(local_metrics)} metrics")
        
        for metric in local_metrics[-3:]:
            print(f"  {metric.name}: {metric.value} {metric.labels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics demo failed: {e}")
        return False

def demo_standalone_dashboard():
    """Demo dashboard functionality."""
    print("\n=== P5 Standalone Dashboard Demo ===")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dashboard", current_dir / "agentnet/observability/dashboard.py"
        )
        dashboard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dashboard_module)
        
        # Test dashboard
        dashboard = dashboard_module.DashboardDataCollector()
        
        print("Adding sample data...")
        base_time = datetime.now() - timedelta(hours=1)
        
        # Add sample events
        dashboard.add_cost_event("openai", "gpt-3.5-turbo", 0.003, 150, "demo-tenant", base_time)
        dashboard.add_performance_event("demo-agent", "gpt-3.5-turbo", "openai", 120.5, True, base_time)
        dashboard.add_violation_event("token_limit", "high", "demo-agent", "Token exceeded", base_time)
        dashboard.add_session_event("demo-session", "brainstorm", "start", 0, False, base_time)
        
        # Generate summaries
        cost_summary = dashboard.get_cost_summary()
        print(f"✅ Cost Summary: ${cost_summary.total_cost_usd:.4f}")
        
        performance_metrics = dashboard.get_performance_metrics()
        print(f"✅ Performance: {performance_metrics.total_requests} requests")
        
        # Generate HTML dashboard
        html_content = dashboard.generate_dashboard_html()
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        dashboard_file = output_dir / "p5_standalone_dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard saved to {dashboard_file}")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard demo failed: {e}")
        return False

def demo_standalone_logging():
    """Demo structured logging."""
    print("\n=== P5 Standalone Logging Demo ===")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "logging_obs", current_dir / "agentnet/observability/logging.py"
        )
        logging_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logging_module)
        
        # Test structured logging
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        log_file = output_dir / "p5_standalone_demo.jsonl"
        
        logging_module.setup_structured_logging(
            log_level="INFO",
            log_file=log_file,
            console_output=True,
            json_format=True
        )
        
        logger = logging_module.get_correlation_logger("standalone.demo")
        session_id = "standalone-demo-session"
        
        logging_module.set_correlation_id(session_id)
        logger.set_correlation_context(
            session_id=session_id,
            agent_name="demo-agent",
            operation="standalone_demo"
        )
        
        print("Generating structured logs...")
        logger.info("Standalone P5 demo started")
        logger.log_agent_inference("gpt-3.5-turbo", "openai", 150, 120.5)
        logger.log_cost_event("openai", "gpt-3.5-turbo", 0.003, 150)
        logger.log_session_event("start", 1, "brainstorm")
        
        # Verify log file
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"✅ Generated {len(lines)} log entries")
                
                # Show sample log
                if lines:
                    sample_log = json.loads(lines[0])
                    print(f"   Sample: {sample_log.get('message')} (correlation_id: {sample_log.get('correlation_id')})")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging demo failed: {e}")
        return False

def demo_integration():
    """Demo integrated observability."""
    print("\n=== P5 Integration Demo ===")
    
    try:
        # Import modules
        import importlib.util
        
        metrics_spec = importlib.util.spec_from_file_location(
            "metrics", current_dir / "agentnet/observability/metrics.py"
        )
        metrics_module = importlib.util.module_from_spec(metrics_spec)
        metrics_spec.loader.exec_module(metrics_module)
        
        dashboard_spec = importlib.util.spec_from_file_location(
            "dashboard", current_dir / "agentnet/observability/dashboard.py"
        )
        dashboard_module = importlib.util.module_from_spec(dashboard_spec)
        dashboard_spec.loader.exec_module(dashboard_module)
        
        # Initialize components
        metrics = metrics_module.AgentNetMetrics(enable_server=False)
        dashboard = dashboard_module.DashboardDataCollector()
        
        session_id = "integration-demo-session"
        
        print("Simulating integrated agent operation...")
        
        # Simulate agent operation with metrics
        start_time = time.time()
        
        # Simulate some processing
        time.sleep(0.05)
        
        duration = time.time() - start_time
        
        # Record metrics
        metrics.record_inference_latency(duration, "gpt-3.5-turbo", "openai", "integration-agent")
        metrics.record_tokens_consumed(175, "gpt-3.5-turbo", "openai", "integration-tenant")
        metrics.record_cost(0.004, "openai", "gpt-3.5-turbo", "integration-tenant")
        
        # Record in dashboard
        dashboard.add_performance_event("integration-agent", "gpt-3.5-turbo", "openai", duration * 1000, True)
        dashboard.add_cost_event("openai", "gpt-3.5-turbo", 0.004, 175, "integration-tenant")
        dashboard.add_session_event(session_id, "integration", "start", 0, False)
        
        # Show results
        local_metrics = metrics.get_local_metrics()
        print(f"✅ Recorded {len(local_metrics)} metrics")
        
        cost_summary = dashboard.get_cost_summary()
        print(f"✅ Total cost: ${cost_summary.total_cost_usd:.4f}")
        
        performance_metrics = dashboard.get_performance_metrics()
        print(f"✅ Avg latency: {performance_metrics.avg_inference_latency_ms:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False

def main():
    """Run standalone P5 demos."""
    print("🔍 AgentNet P5 Observability - Standalone Demo")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {current_dir}")
    
    demos = [
        ("Metrics Collection", demo_standalone_metrics),
        ("Dashboard Generation", demo_standalone_dashboard), 
        ("Structured Logging", demo_standalone_logging),
        ("Integration", demo_integration)
    ]
    
    passed = 0
    failed = 0
    
    for name, demo_func in demos:
        print(f"\n--- {name} ---")
        try:
            if demo_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🎉 Standalone Demo Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✅ All P5 Observability features working!")
        print("Key Features Demonstrated:")
        print("  • Prometheus metrics collection (with fallback)")
        print("  • Cost tracking and aggregation")
        print("  • Performance monitoring")
        print("  • Violation tracking")
        print("  • HTML dashboard generation")
        print("  • Structured JSON logging")
        print("  • Correlation ID tracking")
        print("  • Integrated observability stack")
        print("\nOutput files: demo_output/")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)