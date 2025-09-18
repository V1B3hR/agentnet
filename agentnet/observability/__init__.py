"""
Observability package for AgentNet.

Provides metrics, tracing, and logging infrastructure for comprehensive
monitoring of agent operations, costs, and performance.
"""


# Use lazy imports to avoid pulling in networkx dependency from main package
def _get_metrics_classes():
    from .metrics import AgentNetMetrics, MetricsCollector

    return MetricsCollector, AgentNetMetrics


def _get_tracing_classes():
    from .tracing import TracingManager, create_tracer

    return TracingManager, create_tracer


def _get_logging_classes():
    from .logging import get_correlation_logger, setup_structured_logging

    return setup_structured_logging, get_correlation_logger


# Lazy loading attributes
def __getattr__(name):
    if name == "MetricsCollector":
        MetricsCollector, _ = _get_metrics_classes()
        return MetricsCollector
    elif name == "AgentNetMetrics":
        _, AgentNetMetrics = _get_metrics_classes()
        return AgentNetMetrics
    elif name == "TracingManager":
        TracingManager, _ = _get_tracing_classes()
        return TracingManager
    elif name == "create_tracer":
        _, create_tracer = _get_tracing_classes()
        return create_tracer
    elif name == "setup_structured_logging":
        setup_structured_logging, _ = _get_logging_classes()
        return setup_structured_logging
    elif name == "get_correlation_logger":
        _, get_correlation_logger = _get_logging_classes()
        return get_correlation_logger
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MetricsCollector",
    "AgentNetMetrics",
    "TracingManager",
    "create_tracer",
    "setup_structured_logging",
    "get_correlation_logger",
]
