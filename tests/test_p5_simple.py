#!/usr/bin/env python3
"""
Simple P5 Observability Test

Tests P5 observability features without importing the full AgentNet package
to avoid networkx dependency issues.
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add agentnet to path to import modules directly
sys.path.insert(0, "/home/runner/work/agentnet/agentnet")


def test_metrics():
    """Test metrics collection."""
    print("Testing P5 Metrics...")

    from agentnet.observability.metrics import AgentNetMetrics, MetricsCollector

    metrics = AgentNetMetrics(enable_server=False)
    collector = MetricsCollector(metrics)

    # Test all metric types
    metrics.record_inference_latency(0.150, "gpt-3.5-turbo", "openai", "test-agent")
    metrics.record_tokens_consumed(250, "gpt-3.5-turbo", "openai", "test-tenant")
    metrics.record_violation("high", "token_limit")
    metrics.record_cost(0.005, "openai", "gpt-3.5-turbo", "test-tenant")
    metrics.record_session_round("brainstorm", True)
    metrics.record_tool_invocation("web_search", "success")
    metrics.record_dag_node_duration(0.075, "planner-agent", "planning")
    metrics.set_active_sessions(3)

    # Verify metrics are collected
    if not metrics.enable_prometheus:
        local_metrics = metrics.get_local_metrics()
        assert len(local_metrics) >= 8, f"Expected 8+ metrics, got {len(local_metrics)}"
        print(f"  ✓ Collected {len(local_metrics)} metrics")

    print("✅ Metrics test passed")
    return True


def test_tracing():
    """Test tracing functionality."""
    print("Testing P5 Tracing...")

    from agentnet.observability.tracing import TracingManager, create_tracer

    tracer = create_tracer("test-service", console_export=False)
    session_id = "test-session-123"

    # Test all trace types
    with tracer.trace_session_round(session_id, 1, "brainstorm") as span:
        time.sleep(0.01)
        span.set_attribute("test", "value")

    with tracer.trace_agent_inference(
        "test-agent", "gpt-3.5-turbo", "openai", session_id
    ) as span:
        time.sleep(0.01)
        span.set_attribute("tokens", 150)

    with tracer.trace_tool_invoke("web_search", session_id) as span:
        time.sleep(0.01)
        span.set_attribute("results", 5)

    with tracer.trace_monitor_sequence("policy_check", 3, session_id) as span:
        time.sleep(0.01)
        span.set_attribute("violations", 0)

    with tracer.trace_dag_node(
        "node-1", "planning", "planner-agent", session_id
    ) as span:
        time.sleep(0.01)

    print("✅ Tracing test passed")
    return True


def test_logging():
    """Test structured logging."""
    print("Testing P5 Logging...")

    from agentnet.observability.logging import (
        get_correlation_id,
        get_correlation_logger,
        set_correlation_id,
        setup_structured_logging,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.jsonl"

        setup_structured_logging(
            log_level="INFO", log_file=log_file, console_output=False, json_format=True
        )

        logger = get_correlation_logger("test.logger")
        session_id = "test-session-456"
        set_correlation_id(session_id)

        assert get_correlation_id() == session_id

        logger.set_correlation_context(session_id=session_id, agent_name="test-agent")

        logger.info("Test message")
        logger.log_agent_inference("gpt-3.5-turbo", "openai", 150, 100.5)

        # Verify log file
        if log_file.exists():
            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) >= 2

                first_log = json.loads(lines[0])
                assert "correlation_id" in first_log
                assert first_log["correlation_id"] == session_id
                print(f"  ✓ Generated {len(lines)} log entries")

    print("✅ Logging test passed")
    return True


def test_dashboard():
    """Test dashboard functionality."""
    print("Testing P5 Dashboard...")

    from agentnet.observability.dashboard import DashboardDataCollector, TimeRange

    dashboard = DashboardDataCollector()
    base_time = datetime.now() - timedelta(hours=1)

    # Add test data
    dashboard.add_cost_event(
        "openai", "gpt-3.5-turbo", 0.003, 150, "test-tenant", base_time
    )
    dashboard.add_performance_event(
        "agent-1", "gpt-3.5-turbo", "openai", 120.5, True, base_time
    )
    dashboard.add_violation_event(
        "token_limit", "high", "agent-1", "Token limit", base_time
    )
    dashboard.add_session_event("session-1", "brainstorm", "start", 0, False, base_time)

    # Test summaries
    cost_summary = dashboard.get_cost_summary(TimeRange.LAST_24H)
    assert cost_summary.total_cost_usd > 0
    print(f"  ✓ Cost summary: ${cost_summary.total_cost_usd:.4f}")

    performance_metrics = dashboard.get_performance_metrics(TimeRange.LAST_24H)
    assert performance_metrics.total_requests > 0
    print(f"  ✓ Performance: {performance_metrics.total_requests} requests")

    violation_summary = dashboard.get_violation_summary(TimeRange.LAST_24H)
    assert violation_summary.total_violations > 0
    print(f"  ✓ Violations: {violation_summary.total_violations} total")

    session_metrics = dashboard.get_session_metrics(TimeRange.LAST_24H)
    assert session_metrics.total_sessions > 0
    print(f"  ✓ Sessions: {session_metrics.total_sessions} total")

    # Test HTML generation
    html = dashboard.generate_dashboard_html()
    assert "AgentNet Observability Dashboard" in html
    print(f"  ✓ Generated HTML dashboard ({len(html)} chars)")

    print("✅ Dashboard test passed")
    return True


def test_instrumented_provider():
    """Test instrumented provider (simplified)."""
    print("Testing P5 Provider Instrumentation...")

    from agentnet.providers.base import ProviderAdapter
    from agentnet.providers.instrumented import InstrumentedProviderMixin

    # Create a simple test provider
    class TestProvider(InstrumentedProviderMixin, ProviderAdapter):
        def __init__(self):
            super().__init__()
            self.name = "TestProvider"
            self.model = "test-model"

        def infer(self, prompt, agent_name="Agent", **kwargs):
            time.sleep(0.01)
            return {
                "content": f"Test response to: {prompt[:30]}...",
                "confidence": 0.85,
            }

        async def async_infer(self, prompt, agent_name="Agent", **kwargs):
            return self.infer(prompt, agent_name, **kwargs)

    provider = TestProvider()

    # Test instrumented inference
    result = provider.infer(
        "Test prompt", agent_name="test-agent", session_id="test-session"
    )

    assert "_instrumentation" in result
    instr = result["_instrumentation"]
    assert "duration_ms" in instr
    assert "provider" in instr
    assert "model" in instr

    print(
        f"  ✓ Instrumentation data: {instr['duration_ms']:.1f}ms, {instr['provider']}"
    )

    print("✅ Provider instrumentation test passed")
    return True


def main():
    """Run simple P5 tests."""
    print("🧪 Simple P5 Observability Tests")
    print("=" * 40)

    tests = [
        test_metrics,
        test_tracing,
        test_logging,
        test_dashboard,
        test_instrumented_provider,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All simple P5 tests passed!")
        return True
    else:
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
