"""
OpenTelemetry Tracing for AgentNet.

Implements distributed tracing with spans as specified in docs/RoadmapAgentNet.md:
- session.round.turn
- agent.inference
- monitor.sequence
- tool.invoke

Uses correlation_id = session_id for trace correlation.
"""

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, ParamSpec, TypeVar

logger = logging.getLogger("agentnet.observability.tracing")

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not available - tracing will be logged only")
    OPENTELEMETRY_AVAILABLE = False

    # Mock classes when OpenTelemetry is not available
    class MockSpan:
        def __init__(self, name: str):
            self.name = name
            self._attributes = {}
            self._status = None

        def set_attribute(self, key: str, value: Any):
            self._attributes[key] = value
            logger.debug(f"TRACE: {self.name} - {key}={value}")

        def set_status(self, status):
            self._status = status
            logger.debug(f"TRACE: {self.name} - status={status}")

        def record_exception(self, exception):
            logger.debug(f"TRACE: {self.name} - exception={exception}")

        def end(self):
            logger.debug(f"TRACE: {self.name} - span ended")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.record_exception(exc_val)
            self.end()

    class MockTracer:
        def start_span(self, name: str, **kwargs):
            return MockSpan(name)

        def start_as_current_span(self, name: str, **kwargs):
            return MockSpan(name)

    class Status:
        def __init__(self, status_code=None, description=None):
            self.status_code = status_code
            self.description = description

        def __str__(self):
            return f"Status({self.status_code})"

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    def get_tracer(name: str, version: str = None):
        return MockTracer()

    def extract(carrier):
        return {}

    def inject(carrier):
        pass


# Type hints for decorators
P = ParamSpec("P")
T = TypeVar("T")


class TracingManager:
    """
    Manages OpenTelemetry tracing configuration and provides instrumentation utilities.

    Handles tracer setup, span management, and correlation ID propagation across
    multi-agent operations.
    """

    def __init__(
        self,
        service_name: str = "agentnet",
        service_version: str = "0.4.0",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
    ):
        """
        Initialize tracing manager.

        Args:
            service_name: Name of the service for tracing
            service_version: Version of the service
            jaeger_endpoint: Jaeger collector endpoint (optional)
            otlp_endpoint: OTLP collector endpoint (optional)
            console_export: Whether to export traces to console
        """
        self.service_name = service_name
        self.service_version = service_version
        self.enable_tracing = OPENTELEMETRY_AVAILABLE

        if self.enable_tracing:
            self._setup_tracing(jaeger_endpoint, otlp_endpoint, console_export)
        else:
            logger.info("Using mock tracing (OpenTelemetry not available)")

        self.tracer = self._get_tracer()

    def _setup_tracing(
        self,
        jaeger_endpoint: Optional[str],
        otlp_endpoint: Optional[str],
        console_export: bool,
    ):
        """Setup OpenTelemetry tracing with configured exporters."""
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": self.service_version,
            }
        )

        provider = TracerProvider(resource=resource)

        # Add exporters
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
                collector_endpoint=jaeger_endpoint,
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            logger.info(f"Added Jaeger exporter: {jaeger_endpoint}")

        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"Added OTLP exporter: {otlp_endpoint}")

        if console_export:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
            logger.info("Added console exporter")

        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing initialized")

    def _get_tracer(self):
        """Get tracer instance."""
        if self.enable_tracing:
            return trace.get_tracer(self.service_name, self.service_version)
        else:
            return MockTracer()

    @contextmanager
    def trace_session_round(self, session_id: str, round_number: int, mode: str):
        """
        Trace a complete session round.

        Args:
            session_id: Unique session identifier (used as correlation_id)
            round_number: Round number in session
            mode: Session mode (debate, brainstorm, consensus, etc.)
        """
        with self.tracer.start_as_current_span(
            "session.round.turn",
            attributes={
                "session.id": session_id,
                "session.round": round_number,
                "session.mode": mode,
                "correlation_id": session_id,
            },
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_agent_inference(
        self,
        agent_name: str,
        model: str,
        provider: str,
        session_id: Optional[str] = None,
    ):
        """
        Trace agent inference operation.

        Args:
            agent_name: Name of the agent performing inference
            model: Model being used
            provider: Provider (OpenAI, Anthropic, etc.)
            session_id: Session ID for correlation
        """
        attributes = {
            "agent.name": agent_name,
            "agent.model": model,
            "agent.provider": provider,
        }

        if session_id:
            attributes["correlation_id"] = session_id
            attributes["session.id"] = session_id

        with self.tracer.start_as_current_span(
            "agent.inference", attributes=attributes
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_monitor_sequence(
        self, monitor_name: str, rule_count: int, session_id: Optional[str] = None
    ):
        """
        Trace monitor policy evaluation sequence.

        Args:
            monitor_name: Name of the monitor being executed
            rule_count: Number of rules being evaluated
            session_id: Session ID for correlation
        """
        attributes = {
            "monitor.name": monitor_name,
            "monitor.rule_count": rule_count,
        }

        if session_id:
            attributes["correlation_id"] = session_id
            attributes["session.id"] = session_id

        with self.tracer.start_as_current_span(
            "monitor.sequence", attributes=attributes
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_tool_invoke(self, tool_name: str, session_id: Optional[str] = None):
        """
        Trace tool invocation.

        Args:
            tool_name: Name of the tool being invoked
            session_id: Session ID for correlation
        """
        attributes = {
            "tool.name": tool_name,
        }

        if session_id:
            attributes["correlation_id"] = session_id
            attributes["session.id"] = session_id

        with self.tracer.start_as_current_span(
            "tool.invoke", attributes=attributes
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_dag_node(
        self,
        node_id: str,
        node_type: str,
        agent_name: str,
        session_id: Optional[str] = None,
    ):
        """
        Trace DAG node execution.

        Args:
            node_id: Unique node identifier
            node_type: Type of DAG node
            agent_name: Agent executing the node
            session_id: Session ID for correlation
        """
        attributes = {
            "dag.node.id": node_id,
            "dag.node.type": node_type,
            "dag.agent": agent_name,
        }

        if session_id:
            attributes["correlation_id"] = session_id
            attributes["session.id"] = session_id

        with self.tracer.start_as_current_span(
            "dag.node.execute", attributes=attributes
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def add_span_attributes(self, span, attributes: Dict[str, Any]):
        """Add multiple attributes to a span."""
        for key, value in attributes.items():
            span.set_attribute(key, value)

    def trace_function(
        self,
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Decorator to automatically trace function calls.

        Args:
            span_name: Custom span name (defaults to function name)
            attributes: Additional span attributes
        """

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                name = span_name or f"{func.__module__}.{func.__name__}"
                attrs = attributes or {}

                with self.tracer.start_as_current_span(name, attributes=attrs) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return wrapper

        return decorator


# Global tracing manager instance
_global_tracing_manager: Optional[TracingManager] = None


def create_tracer(
    service_name: str = "agentnet", service_version: str = "0.4.0", **kwargs
) -> TracingManager:
    """
    Create a new tracing manager instance.

    Args:
        service_name: Service name for tracing
        service_version: Service version
        **kwargs: Additional arguments for TracingManager

    Returns:
        TracingManager instance
    """
    return TracingManager(service_name, service_version, **kwargs)


def get_global_tracer() -> TracingManager:
    """Get or create global tracing manager."""
    global _global_tracing_manager
    if _global_tracing_manager is None:
        _global_tracing_manager = TracingManager()
    return _global_tracing_manager


def set_global_tracer(tracer: TracingManager):
    """Set global tracing manager."""
    global _global_tracing_manager
    _global_tracing_manager = tracer


def generate_correlation_id() -> str:
    """Generate a new correlation ID for session tracking."""
    return str(uuid.uuid4())


# Convenience functions for common tracing patterns
@contextmanager
def trace_agent_operation(
    agent_name: str, model: str, provider: str, session_id: Optional[str] = None
):
    """Convenience function for tracing agent operations."""
    tracer = get_global_tracer()
    with tracer.trace_agent_inference(agent_name, model, provider, session_id) as span:
        yield span


@contextmanager
def trace_tool_call(tool_name: str, session_id: Optional[str] = None):
    """Convenience function for tracing tool calls."""
    tracer = get_global_tracer()
    with tracer.trace_tool_invoke(tool_name, session_id) as span:
        yield span


@contextmanager
def trace_monitor_run(
    monitor_name: str, rule_count: int, session_id: Optional[str] = None
):
    """Convenience function for tracing monitor runs."""
    tracer = get_global_tracer()
    with tracer.trace_monitor_sequence(monitor_name, rule_count, session_id) as span:
        yield span
