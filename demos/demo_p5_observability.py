#!/usr/bin/env python3
"""
Demo P5 Observability Features

Demonstrates metrics collection, tracing, structured logging, and dashboards
for AgentNet operations as implemented in P5.

Usage:
    python demo_p5_observability.py --mode [metrics|tracing|dashboard|full]
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Import AgentNet components
from agentnet import AgentNet
from agentnet.providers.example import ExampleEngine
from agentnet.providers.instrumented import instrument_provider

# Import P5 observability features
from agentnet.observability.metrics import AgentNetMetrics, MetricsCollector
from agentnet.observability.tracing import TracingManager, create_tracer
from agentnet.observability.logging import setup_structured_logging, get_correlation_logger
from agentnet.observability.dashboard import DashboardDataCollector, TimeRange

def setup_demo_logging():
    """Setup structured logging for the demo."""
    log_dir = Path("demo_output")
    log_dir.mkdir(exist_ok=True)
    
    setup_structured_logging(
        log_level="INFO",
        log_file=log_dir / "p5_observability_demo.jsonl",
        console_output=True,
        json_format=True
    )
    
    logger = get_correlation_logger("demo.p5.observability")
    logger.info("P5 Observability Demo started", demo_type="p5_observability")
    return logger

def demo_metrics_collection():
    """Demonstrate metrics collection."""
    print("\n=== P5 Metrics Collection Demo ===")
    
    # Initialize metrics
    metrics = AgentNetMetrics(enable_server=False)
    collector = MetricsCollector(metrics)
    
    # Simulate various operations
    print("Simulating inference operations...")
    
    # Record some inference latencies
    for i in range(5):
        latency = 0.1 + (i * 0.05)
        metrics.record_inference_latency(latency, "gpt-3.5-turbo", "openai", f"agent-{i}")
        print(f"  Recorded inference: {latency:.3f}s for agent-{i}")
    
    # Record token consumption
    for provider, model, tokens in [
        ("openai", "gpt-3.5-turbo", 150),
        ("openai", "gpt-4", 200),
        ("anthropic", "claude-3", 180)
    ]:
        metrics.record_tokens_consumed(tokens, model, provider, "tenant-demo")
        print(f"  Recorded {tokens} tokens for {model} on {provider}")
    
    # Record violations
    violations = [
        ("high", "resource_limit"),
        ("medium", "keyword_filter"),
        ("low", "style_constraint")
    ]
    
    for severity, rule in violations:
        metrics.record_violation(severity, rule)
        print(f"  Recorded {severity} violation: {rule}")
    
    # Record costs
    costs = [
        ("openai", "gpt-3.5-turbo", 0.003, "tenant-demo"),
        ("openai", "gpt-4", 0.012, "tenant-demo"),
        ("anthropic", "claude-3", 0.008, "tenant-premium")
    ]
    
    for provider, model, cost, tenant in costs:
        metrics.record_cost(cost, provider, model, tenant)
        print(f"  Recorded cost: ${cost:.4f} for {model} on {provider} (tenant: {tenant})")
    
    # Get local metrics if Prometheus not available
    if not metrics.enable_prometheus:
        local_metrics = metrics.get_local_metrics()
        print(f"\nCollected {len(local_metrics)} local metric events")
        
        # Show recent metrics
        print("\nRecent metrics:")
        for metric in local_metrics[-5:]:
            print(f"  {metric.name}: {metric.value} {metric.labels}")
    
    print("✅ Metrics collection demo complete")
    return metrics

def demo_tracing():
    """Demonstrate distributed tracing."""
    print("\n=== P5 Tracing Demo ===")
    
    # Initialize tracing
    tracer_manager = create_tracer(
        service_name="agentnet-demo",
        console_export=True  # Show traces in console
    )
    
    session_id = "demo-session-123"
    
    print(f"Starting traced session: {session_id}")
    
    # Simulate session round with tracing
    with tracer_manager.trace_session_round(session_id, 1, "brainstorm") as session_span:
        print("  Tracing session round 1...")
        
        # Simulate agent inference within the session
        with tracer_manager.trace_agent_inference("Athena", "gpt-3.5-turbo", "openai", session_id) as agent_span:
            print("    Tracing agent Athena inference...")
            time.sleep(0.1)  # Simulate processing
            agent_span.set_attribute("tokens_consumed", 150)
            agent_span.set_attribute("confidence", 0.85)
        
        # Simulate tool invocation
        with tracer_manager.trace_tool_invoke("web_search", session_id) as tool_span:
            print("    Tracing web search tool...")
            time.sleep(0.05)
            tool_span.set_attribute("search_results", 5)
        
        # Simulate monitor sequence
        with tracer_manager.trace_monitor_sequence("policy_check", 3, session_id) as monitor_span:
            print("    Tracing policy monitor...")
            time.sleep(0.02)
            monitor_span.set_attribute("violations_found", 0)
        
        session_span.set_attribute("agents_participated", 1)
        session_span.set_attribute("tools_used", 1)
    
    print("✅ Tracing demo complete")
    return tracer_manager

def demo_instrumented_provider():
    """Demonstrate instrumented provider adapter."""
    print("\n=== P5 Instrumented Provider Demo ===")
    
    # Create instrumented version of ExampleEngine
    InstrumentedExampleEngine = instrument_provider(ExampleEngine)
    provider = InstrumentedExampleEngine()
    
    session_id = "instrumented-demo-456"
    logger = get_correlation_logger("demo.instrumented.provider")
    
    print("Running instrumented provider operations...")
    
    # Test synchronous inference
    print("  Testing sync inference...")
    result = provider.infer(
        "What are the benefits of distributed tracing?",
        agent_name="ObservabilityAgent",
        session_id=session_id
    )
    print(f"    Result: {result['content'][:50]}...")
    
    # Test asynchronous inference
    print("  Testing async inference...")
    async def async_test():
        result = await provider.async_infer(
            "How do metrics help with system monitoring?",
            agent_name="MetricsAgent", 
            session_id=session_id
        )
        return result
    
    result = asyncio.run(async_test())
    print(f"    Result: {result['content'][:50]}...")
    
    print("✅ Instrumented provider demo complete")
    return provider

def demo_dashboard():
    """Demonstrate dashboard data collection and generation."""
    print("\n=== P5 Dashboard Demo ===")
    
    # Initialize dashboard collector
    dashboard = DashboardDataCollector()
    
    # Simulate historical data
    print("Generating sample dashboard data...")
    
    base_time = datetime.now() - timedelta(hours=2)
    
    # Add cost events
    for i in range(20):
        timestamp = base_time + timedelta(minutes=i*5)
        provider = "openai" if i % 2 == 0 else "anthropic"
        model = "gpt-3.5-turbo" if provider == "openai" else "claude-3"
        cost = 0.001 + (i * 0.0005)
        tokens = 100 + (i * 10)
        
        dashboard.add_cost_event(provider, model, cost, tokens, "demo-tenant", timestamp)
    
    # Add performance events
    for i in range(30):
        timestamp = base_time + timedelta(minutes=i*3)
        agent = f"agent-{i % 3}"
        model = ["gpt-3.5-turbo", "gpt-4", "claude-3"][i % 3]
        provider = ["openai", "openai", "anthropic"][i % 3]
        latency = 50 + (i * 5) + (i % 10) * 20  # Vary latency
        success = i % 10 != 7  # Simulate some failures
        
        dashboard.add_performance_event(agent, model, provider, latency, success, timestamp)
    
    # Add violation events
    violations = [
        ("high", "token_limit_exceeded"),
        ("medium", "inappropriate_content"),
        ("low", "style_deviation"),
        ("high", "resource_exhaustion"),
        ("medium", "policy_violation")
    ]
    
    for i, (severity, rule) in enumerate(violations):
        timestamp = base_time + timedelta(minutes=i*20)
        dashboard.add_violation_event(rule, severity, f"agent-{i}", f"Violation details {i}", timestamp)
    
    # Add session events
    for i in range(5):
        session_id = f"session-{i}"
        timestamp = base_time + timedelta(minutes=i*15)
        mode = ["brainstorm", "debate", "consensus"][i % 3]
        
        dashboard.add_session_event(session_id, mode, "start", 0, False, timestamp)
        
        # Add rounds
        for round_num in range(1, 4):
            round_timestamp = timestamp + timedelta(minutes=round_num*2)
            converged = round_num == 3 and i % 2 == 0
            dashboard.add_session_event(session_id, mode, "round", round_num, converged, round_timestamp)
    
    # Generate summaries
    print("\nGenerating dashboard summaries...")
    
    cost_summary = dashboard.get_cost_summary(TimeRange.LAST_24H)
    print(f"  Total cost (24h): ${cost_summary.total_cost_usd:.4f}")
    print(f"  Total tokens (24h): {cost_summary.token_count:,}")
    
    performance_metrics = dashboard.get_performance_metrics(TimeRange.LAST_24H)
    print(f"  Avg latency (24h): {performance_metrics.avg_inference_latency_ms:.1f}ms")
    print(f"  Total requests (24h): {performance_metrics.total_requests}")
    print(f"  Error rate (24h): {performance_metrics.error_rate_percent:.1f}%")
    
    violation_summary = dashboard.get_violation_summary(TimeRange.LAST_24H)
    print(f"  Total violations (24h): {violation_summary.total_violations}")
    
    session_metrics = dashboard.get_session_metrics(TimeRange.LAST_24H)
    print(f"  Total sessions (24h): {session_metrics.total_sessions}")
    print(f"  Convergence rate (24h): {session_metrics.convergence_rate_percent:.1f}%")
    
    # Generate HTML dashboard
    print("\nGenerating HTML dashboard...")
    html_content = dashboard.generate_dashboard_html(TimeRange.LAST_24H)
    
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    dashboard_file = output_dir / "p5_observability_dashboard.html"
    
    with open(dashboard_file, 'w') as f:
        f.write(html_content)
    
    print(f"  Dashboard saved to: {dashboard_file}")
    
    # Export JSON data
    json_data = dashboard.export_dashboard_data()
    json_file = output_dir / "p5_observability_data.json"
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"  Data exported to: {json_file}")
    
    print("✅ Dashboard demo complete")
    return dashboard

def demo_integrated_session():
    """Demonstrate integrated observability in a multi-agent session."""
    print("\n=== P5 Integrated Session Demo ===")
    
    # Setup observability
    metrics = AgentNetMetrics()
    tracer = create_tracer(service_name="agentnet-integrated-demo")
    dashboard = DashboardDataCollector()
    logger = get_correlation_logger("demo.integrated.session")
    
    session_id = "integrated-session-789"
    
    # Create instrumented agents
    InstrumentedEngine = instrument_provider(ExampleEngine)
    
    agents = {
        "Athena": AgentNet(
            agent_name="Athena",
            style_insights={"analytical": 0.8, "creative": 0.3},
            provider=InstrumentedEngine()
        ),
        "Apollo": AgentNet(
            agent_name="Apollo", 
            style_insights={"creative": 0.9, "analytical": 0.2},
            provider=InstrumentedEngine()
        )
    }
    
    topic = "Design a resilient distributed system"
    
    print(f"Starting integrated session: {session_id}")
    print(f"Topic: {topic}")
    
    # Trace the entire session
    with tracer.trace_session_round(session_id, 1, "brainstorm"):
        logger.set_correlation_context(session_id=session_id, operation="brainstorm_session")
        logger.log_session_event("start", 1, "brainstorm", topic=topic)
        
        # Record session start
        dashboard.add_session_event(session_id, "brainstorm", "start", 0)
        
        # Each agent contributes
        for agent_name, agent in agents.items():
            print(f"  {agent_name} contributing...")
            
            # Agent thinks about the topic
            try:
                result = agent.think(
                    f"Round 1: {topic}. Share your perspective.",
                    session_id=session_id
                )
                
                print(f"    {agent_name}: {result['content'][:80]}...")
                
                # Extract instrumentation data if available
                if '_instrumentation' in result:
                    instr = result['_instrumentation']
                    
                    # Add to dashboard
                    dashboard.add_performance_event(
                        agent_name, 
                        instr['model'], 
                        instr['provider'],
                        instr['duration_ms'],
                        True
                    )
                    
                    dashboard.add_cost_event(
                        instr['provider'],
                        instr['model'], 
                        0.001,  # Simulated cost
                        instr['tokens_input'] + instr['tokens_output'],
                        "demo-tenant"
                    )
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed", error=str(e))
                dashboard.add_performance_event(agent_name, "unknown", "unknown", 0, False)
        
        # Record session completion
        dashboard.add_session_event(session_id, "brainstorm", "round", 1, True)
        logger.log_session_event("complete", 1, "brainstorm", converged=True)
    
    print("✅ Integrated session demo complete")
    
    # Show final metrics summary
    print("\nSession Metrics Summary:")
    session_metrics = dashboard.get_session_metrics(TimeRange.LAST_HOUR)
    print(f"  Sessions completed: {session_metrics.total_sessions}")
    
    cost_summary = dashboard.get_cost_summary(TimeRange.LAST_HOUR)
    print(f"  Total cost: ${cost_summary.total_cost_usd:.4f}")
    
    performance_metrics = dashboard.get_performance_metrics(TimeRange.LAST_HOUR)
    print(f"  Average latency: {performance_metrics.avg_inference_latency_ms:.1f}ms")
    
    return {
        "metrics": metrics,
        "tracer": tracer,
        "dashboard": dashboard,
        "session_id": session_id
    }

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="P5 Observability Demo")
    parser.add_argument(
        "--mode",
        choices=["metrics", "tracing", "dashboard", "instrumented", "integrated", "full"],
        default="full",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_demo_logging()
    
    print("🔍 AgentNet P5 Observability Demo")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    try:
        if args.mode in ["metrics", "full"]:
            results["metrics"] = demo_metrics_collection()
        
        if args.mode in ["tracing", "full"]:
            results["tracing"] = demo_tracing()
        
        if args.mode in ["instrumented", "full"]:
            results["instrumented"] = demo_instrumented_provider()
        
        if args.mode in ["dashboard", "full"]:
            results["dashboard"] = demo_dashboard()
        
        if args.mode in ["integrated", "full"]:
            results["integrated"] = demo_integrated_session()
        
        print("\n" + "=" * 50)
        print("🎉 P5 Observability Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("✅ Prometheus metrics collection")
        print("✅ OpenTelemetry distributed tracing") 
        print("✅ Structured JSON logging with correlation")
        print("✅ Cost tracking and aggregation")
        print("✅ Performance monitoring dashboards")
        print("✅ Policy violation tracking")
        print("✅ Instrumented provider adapters")
        print("✅ Integrated multi-agent observability")
        
        print(f"\nOutput files saved to: demo_output/")
        
        logger.info("P5 Observability Demo completed successfully", 
                   mode=args.mode, 
                   features_demonstrated=8)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error("P5 Demo failed", error=str(e), mode=args.mode)
        raise

if __name__ == "__main__":
    main()