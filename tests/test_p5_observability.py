#!/usr/bin/env python3
"""
Test P5 Observability Implementation

Tests for metrics collection, tracing, structured logging, and dashboard functionality.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_metrics_collection():
    """Test metrics collection functionality."""
    print("Testing P5 Metrics Collection...")

    from agentnet.observability.metrics import AgentNetMetrics, MetricsCollector

    # Test metrics initialization
    metrics = AgentNetMetrics(enable_server=False)
    collector = MetricsCollector(metrics)

    assert metrics is not None
    assert collector is not None

    # Test inference latency recording
    metrics.record_inference_latency(0.150, "gpt-3.5-turbo", "openai", "test-agent")

    # Test token consumption recording
    metrics.record_tokens_consumed(250, "gpt-3.5-turbo", "openai", "test-tenant")

    # Test violation recording
    metrics.record_violation("high", "token_limit")

    # Test cost recording
    metrics.record_cost(0.005, "openai", "gpt-3.5-turbo", "test-tenant")

    # Test session round recording
    metrics.record_session_round("brainstorm", True)

    # Test tool invocation recording
    metrics.record_tool_invocation("web_search", "success")

    # Test DAG node duration recording
    metrics.record_dag_node_duration(0.075, "planner-agent", "planning")

    # Test active sessions gauge
    metrics.set_active_sessions(3)

    # Verify local metrics are collected when Prometheus not available
    if not metrics.enable_prometheus:
        local_metrics = metrics.get_local_metrics()
        assert len(local_metrics) > 0

        # Verify metric types
        metric_names = [m.name for m in local_metrics]
        expected_metrics = [
            "inference_latency_seconds",
            "tokens_consumed_total",
            "violations_total",
            "cost_usd_total",
            "session_rounds_total",
            "tool_invocations_total",
            "dag_node_duration_seconds",
            "active_sessions",
        ]

        for expected in expected_metrics:
            assert expected in metric_names, f"Missing metric: {expected}"

    print("✅ Metrics collection tests passed")
    return True


def test_tracing():
    """Test tracing functionality."""
    print("Testing P5 Tracing...")

    from agentnet.observability.tracing import TracingManager, create_tracer

    # Test tracer creation
    tracer = create_tracer("test-service", console_export=False)
    assert tracer is not None

    session_id = "test-session-123"

    # Test session round tracing
    with tracer.trace_session_round(session_id, 1, "brainstorm") as span:
        assert span is not None
        time.sleep(0.01)  # Simulate work

    # Test agent inference tracing
    with tracer.trace_agent_inference(
        "test-agent", "gpt-3.5-turbo", "openai", session_id
    ) as span:
        assert span is not None
        span.set_attribute("tokens", 150)
        time.sleep(0.01)

    # Test tool invocation tracing
    with tracer.trace_tool_invoke("web_search", session_id) as span:
        assert span is not None
        span.set_attribute("results_count", 5)
        time.sleep(0.01)

    # Test monitor sequence tracing
    with tracer.trace_monitor_sequence("policy_check", 3, session_id) as span:
        assert span is not None
        span.set_attribute("violations", 0)
        time.sleep(0.01)

    # Test DAG node tracing
    with tracer.trace_dag_node(
        "node-1", "planning", "planner-agent", session_id
    ) as span:
        assert span is not None
        time.sleep(0.01)

    print("✅ Tracing tests passed")
    return True


def test_structured_logging():
    """Test structured logging functionality."""
    print("Testing P5 Structured Logging...")

    from agentnet.observability.logging import (
        clear_correlation_context,
        get_correlation_id,
        get_correlation_logger,
        set_correlation_id,
        setup_structured_logging,
    )

    # Test structured logging setup
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.jsonl"

        setup_structured_logging(
            log_level="INFO", log_file=log_file, console_output=False, json_format=True
        )

        # Test correlation logger
        logger = get_correlation_logger("test.logger")

        # Test correlation context
        session_id = "test-session-456"
        set_correlation_id(session_id)

        assert get_correlation_id() == session_id

        # Test logger with context
        logger.set_correlation_context(
            session_id=session_id, agent_name="test-agent", operation="test-operation"
        )

        # Test various log methods
        logger.info("Test info message", extra_field="test_value")
        logger.log_agent_inference("gpt-3.5-turbo", "openai", 150, 100.5)
        logger.log_tool_invocation("web_search", "success", 50.0)
        logger.log_violation("token_limit", "high", "Token limit exceeded")
        logger.log_cost_event("openai", "gpt-3.5-turbo", 0.003, 150)
        logger.log_session_event("start", 1, "brainstorm")

        # Clear context
        clear_correlation_context()
        assert get_correlation_id() is None

        # Verify log file was created and contains JSON
        if log_file.exists():
            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) > 0

                # Verify first line is valid JSON
                first_log = json.loads(lines[0])
                assert "timestamp" in first_log
                assert "level" in first_log
                assert "message" in first_log
                assert "correlation_id" in first_log

    print("✅ Structured logging tests passed")
    return True


def test_dashboard():
    """Test dashboard functionality."""
    print("Testing P5 Dashboard...")

    from agentnet.observability.dashboard import (
        CostSummary,
        DashboardDataCollector,
        PerformanceMetrics,
        SessionMetrics,
        TimeRange,
        ViolationSummary,
    )

    dashboard = DashboardDataCollector()

    # Add test data
    base_time = datetime.now() - timedelta(hours=1)

    # Add cost events
    dashboard.add_cost_event(
        "openai", "gpt-3.5-turbo", 0.003, 150, "test-tenant", base_time
    )
    dashboard.add_cost_event(
        "anthropic",
        "claude-3",
        0.008,
        200,
        "test-tenant",
        base_time + timedelta(minutes=10),
    )

    # Add performance events
    dashboard.add_performance_event(
        "agent-1", "gpt-3.5-turbo", "openai", 120.5, True, base_time
    )
    dashboard.add_performance_event(
        "agent-2",
        "claude-3",
        "anthropic",
        150.0,
        True,
        base_time + timedelta(minutes=5),
    )
    dashboard.add_performance_event(
        "agent-1",
        "gpt-3.5-turbo",
        "openai",
        200.0,
        False,
        base_time + timedelta(minutes=15),
    )

    # Add violation events
    dashboard.add_violation_event(
        "token_limit", "high", "agent-1", "Token limit exceeded", base_time
    )
    dashboard.add_violation_event(
        "content_filter",
        "medium",
        "agent-2",
        "Content filtered",
        base_time + timedelta(minutes=20),
    )

    # Add session events
    dashboard.add_session_event("session-1", "brainstorm", "start", 0, False, base_time)
    dashboard.add_session_event(
        "session-1", "brainstorm", "round", 1, False, base_time + timedelta(minutes=5)
    )
    dashboard.add_session_event(
        "session-1", "brainstorm", "round", 2, True, base_time + timedelta(minutes=10)
    )

    # Test summaries
    cost_summary = dashboard.get_cost_summary(TimeRange.LAST_24H)
    assert isinstance(cost_summary, CostSummary)
    assert cost_summary.total_cost_usd > 0
    assert len(cost_summary.cost_by_provider) > 0

    performance_metrics = dashboard.get_performance_metrics(TimeRange.LAST_24H)
    assert isinstance(performance_metrics, PerformanceMetrics)
    assert performance_metrics.total_requests > 0
    assert performance_metrics.avg_inference_latency_ms > 0

    violation_summary = dashboard.get_violation_summary(TimeRange.LAST_24H)
    assert isinstance(violation_summary, ViolationSummary)
    assert violation_summary.total_violations > 0
    assert len(violation_summary.violations_by_severity) > 0

    session_metrics = dashboard.get_session_metrics(TimeRange.LAST_24H)
    assert isinstance(session_metrics, SessionMetrics)
    assert session_metrics.total_sessions > 0

    # Test HTML dashboard generation
    html_content = dashboard.generate_dashboard_html(TimeRange.LAST_24H)
    assert isinstance(html_content, str)
    assert "AgentNet Observability Dashboard" in html_content
    assert "Total Cost" in html_content

    # Test data export
    exported_data = dashboard.export_dashboard_data()
    assert isinstance(exported_data, dict)
    assert "cost_summary" in exported_data
    assert "performance_metrics" in exported_data
    assert "violation_summary" in exported_data
    assert "session_metrics" in exported_data

    print("✅ Dashboard tests passed")
    return True


def test_instrumented_provider():
    """Test instrumented provider functionality."""
    print("Testing P5 Instrumented Provider...")

    from agentnet.providers.example import ExampleEngine
    from agentnet.providers.instrumented import instrument_provider

    # Create instrumented provider
    InstrumentedEngine = instrument_provider(ExampleEngine)
    provider = InstrumentedEngine()

    session_id = "test-session-789"

    # Test sync inference with instrumentation
    result = provider.infer(
        "Test prompt for instrumented provider",
        agent_name="test-agent",
        session_id=session_id,
    )

    assert result is not None
    assert "content" in result
    assert "_instrumentation" in result

    instr_data = result["_instrumentation"]
    assert "duration_ms" in instr_data
    assert "tokens_input" in instr_data
    assert "tokens_output" in instr_data
    assert "provider" in instr_data
    assert "model" in instr_data
    assert "agent_name" in instr_data
    assert "session_id" in instr_data

    # Test async inference with instrumentation
    async def test_async():
        result = await provider.async_infer(
            "Test async prompt for instrumented provider",
            agent_name="test-async-agent",
            session_id=session_id,
        )

        assert result is not None
        assert "content" in result
        assert "_instrumentation" in result
        return result

    async_result = asyncio.run(test_async())
    assert async_result is not None

    print("✅ Instrumented provider tests passed")
    return True


def test_integration():
    """Test integration between observability components."""
    print("Testing P5 Integration...")

    from agentnet.observability.dashboard import DashboardDataCollector
    from agentnet.observability.logging import (
        get_correlation_logger,
        set_correlation_id,
    )
    from agentnet.observability.metrics import AgentNetMetrics
    from agentnet.observability.tracing import create_tracer
    from agentnet.providers.example import ExampleEngine
    from agentnet.providers.instrumented import instrument_provider

    # Setup observability stack
    metrics = AgentNetMetrics(enable_server=False)
    tracer = create_tracer("integration-test", console_export=False)
    dashboard = DashboardDataCollector()
    logger = get_correlation_logger("integration.test")

    session_id = "integration-test-session"
    set_correlation_id(session_id)

    # Create instrumented provider
    InstrumentedEngine = instrument_provider(ExampleEngine)
    provider = InstrumentedEngine()

    # Execute traced and instrumented operation
    with tracer.trace_session_round(session_id, 1, "test"):
        logger.set_correlation_context(
            session_id=session_id, operation="integration_test"
        )

        result = provider.infer(
            "Integration test prompt",
            agent_name="integration-agent",
            session_id=session_id,
        )

        # Verify instrumentation worked
        assert "_instrumentation" in result
        instr = result["_instrumentation"]

        # Manually add to dashboard (normally done by instrumentation)
        dashboard.add_performance_event(
            instr["agent_name"],
            instr["model"],
            instr["provider"],
            instr["duration_ms"],
            True,
        )

        dashboard.add_cost_event(
            instr["provider"],
            instr["model"],
            0.001,  # Simulated cost
            instr["tokens_input"] + instr["tokens_output"],
            "integration-tenant",
        )

        logger.log_agent_inference(
            instr["model"],
            instr["provider"],
            instr["tokens_input"] + instr["tokens_output"],
            instr["duration_ms"],
        )

    # Verify integration worked
    performance_metrics = dashboard.get_performance_metrics()
    assert performance_metrics.total_requests > 0

    cost_summary = dashboard.get_cost_summary()
    assert cost_summary.total_cost_usd > 0

    if not metrics.enable_prometheus:
        local_metrics = metrics.get_local_metrics()
        assert len(local_metrics) > 0

    print("✅ Integration tests passed")
    return True


def test_error_handling():
    """Test error handling in observability components."""
    print("Testing P5 Error Handling...")

    from agentnet.observability.dashboard import DashboardDataCollector
    from agentnet.observability.metrics import AgentNetMetrics
    from agentnet.observability.tracing import create_tracer

    # Test metrics error handling
    metrics = AgentNetMetrics(enable_server=False)

    # Should not crash with invalid inputs
    try:
        metrics.record_inference_latency(-1.0, "", "", "")
        metrics.record_tokens_consumed(-1, "", "", "")
        metrics.record_cost(-1.0, "", "", "")
    except Exception as e:
        print(f"Metrics error handling issue: {e}")
        return False

    # Test tracing error handling
    tracer = create_tracer("error-test", console_export=False)

    try:
        with tracer.trace_agent_inference("", "", "", "") as span:
            span.set_attribute("test", "value")
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected
    except Exception as e:
        print(f"Tracing error handling issue: {e}")
        return False

    # Test dashboard error handling
    dashboard = DashboardDataCollector()

    try:
        # Should handle empty data gracefully
        cost_summary = dashboard.get_cost_summary()
        assert cost_summary.total_cost_usd == 0.0

        performance_metrics = dashboard.get_performance_metrics()
        assert performance_metrics.total_requests == 0

        violation_summary = dashboard.get_violation_summary()
        assert violation_summary.total_violations == 0

        session_metrics = dashboard.get_session_metrics()
        assert session_metrics.total_sessions == 0
    except Exception as e:
        print(f"Dashboard error handling issue: {e}")
        return False

    print("✅ Error handling tests passed")
    return True


def main():
    """Run all P5 observability tests."""
    print("🧪 Running P5 Observability Tests")
    print("=" * 50)

    tests = [
        test_metrics_collection,
        test_tracing,
        test_structured_logging,
        test_dashboard,
        test_instrumented_provider,
        test_integration,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All P5 Observability tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
