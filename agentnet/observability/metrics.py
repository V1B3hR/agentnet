"""
Prometheus Metrics Collection for AgentNet.

Implements metrics specified in docs/RoadmapAgentNet.md section 18:
- inference_latency_ms (model, provider, agent)
- tokens_consumed_total (model, provider, tenant)  
- violations_total (severity, rule_name)
- cost_usd_total (provider, model, tenant)
- session_rounds (mode, converged)
- tool_invocations_total (tool_name, status)
- dag_node_duration_ms (agent, node_type)
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger("agentnet.observability.metrics")

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available - metrics will be logged only")
    PROMETHEUS_AVAILABLE = False
    
    # Mock classes for when Prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'mock')
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            logger.info(f"METRIC: {self.name} increment by {amount}")
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'mock')
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            logger.info(f"METRIC: {self.name} observe {value}")
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'mock')
        def labels(self, **kwargs):
            return self
        def set(self, value):
            logger.info(f"METRIC: {self.name} set to {value}")
        def inc(self, amount=1):
            logger.info(f"METRIC: {self.name} increment by {amount}")
    
    class CollectorRegistry:
        pass
    
    def start_http_server(port, registry=None):
        logger.info(f"Mock metrics server would start on port {port}")


@dataclass
class MetricValue:
    """Simple metric value holder for when Prometheus is not available."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    
    
class AgentNetMetrics:
    """
    Centralized metrics collection for AgentNet operations.
    
    Provides both Prometheus integration (when available) and structured logging
    for all key metrics defined in the roadmap.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, enable_server: bool = False, port: int = 8000):
        """
        Initialize metrics collection.
        
        Args:
            registry: Prometheus registry (optional)
            enable_server: Whether to start Prometheus HTTP server
            port: Port for Prometheus server
        """
        self.registry = registry
        self.enable_prometheus = PROMETHEUS_AVAILABLE
        self._local_metrics: List[MetricValue] = []
        
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
            if enable_server:
                start_http_server(port, registry=registry)
                logger.info(f"Prometheus metrics server started on port {port}")
        else:
            logger.info("Using local metrics collection (Prometheus not available)")
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        registry_kwargs = {"registry": self.registry} if self.registry else {}
        
        # Inference latency histogram
        self.inference_latency = Histogram(
            'agentnet_inference_latency_seconds',
            'Time spent on agent inference',
            labelnames=['model', 'provider', 'agent'],
            **registry_kwargs
        )
        
        # Token consumption counter
        self.tokens_consumed = Counter(
            'agentnet_tokens_consumed_total',
            'Total tokens consumed by provider',
            labelnames=['model', 'provider', 'tenant'],
            **registry_kwargs
        )
        
        # Violations counter
        self.violations_total = Counter(
            'agentnet_violations_total',
            'Total policy violations detected',
            labelnames=['severity', 'rule_name'],
            **registry_kwargs
        )
        
        # Cost tracking
        self.cost_usd_total = Counter(
            'agentnet_cost_usd_total',
            'Total cost in USD',
            labelnames=['provider', 'model', 'tenant'],
            **registry_kwargs
        )
        
        # Session rounds
        self.session_rounds = Counter(
            'agentnet_session_rounds_total',
            'Total session rounds completed',
            labelnames=['mode', 'converged'],
            **registry_kwargs
        )
        
        # Tool invocations
        self.tool_invocations = Counter(
            'agentnet_tool_invocations_total',
            'Total tool invocations',
            labelnames=['tool_name', 'status'],
            **registry_kwargs
        )
        
        # DAG node duration
        self.dag_node_duration = Histogram(
            'agentnet_dag_node_duration_seconds',
            'Time spent executing DAG nodes',
            labelnames=['agent', 'node_type'],
            **registry_kwargs
        )
        
        # System health metrics
        self.active_sessions = Gauge(
            'agentnet_active_sessions',
            'Number of currently active sessions',
            **registry_kwargs
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_inference_latency(self, duration_seconds: float, model: str, provider: str, agent: str):
        """Record inference latency metric."""
        if self.enable_prometheus:
            self.inference_latency.labels(model=model, provider=provider, agent=agent).observe(duration_seconds)
        else:
            self._record_local_metric("inference_latency_seconds", duration_seconds, {
                "model": model, "provider": provider, "agent": agent
            })
        
        logger.debug(f"Recorded inference latency: {duration_seconds:.3f}s for {agent} using {model} on {provider}")
    
    def record_tokens_consumed(self, token_count: int, model: str, provider: str, tenant: Optional[str] = None):
        """Record token consumption metric."""
        tenant = tenant or "default"
        
        if self.enable_prometheus:
            self.tokens_consumed.labels(model=model, provider=provider, tenant=tenant).inc(token_count)
        else:
            self._record_local_metric("tokens_consumed_total", token_count, {
                "model": model, "provider": provider, "tenant": tenant
            })
            
        logger.debug(f"Recorded {token_count} tokens consumed for {model} on {provider} (tenant: {tenant})")
    
    def record_violation(self, severity: str, rule_name: str):
        """Record policy violation metric."""
        if self.enable_prometheus:
            self.violations_total.labels(severity=severity, rule_name=rule_name).inc()
        else:
            self._record_local_metric("violations_total", 1, {
                "severity": severity, "rule_name": rule_name
            })
            
        logger.info(f"Recorded violation: {severity} severity for rule {rule_name}")
    
    def record_cost(self, cost_usd: float, provider: str, model: str, tenant: Optional[str] = None):
        """Record cost metric."""
        tenant = tenant or "default"
        
        if self.enable_prometheus:
            self.cost_usd_total.labels(provider=provider, model=model, tenant=tenant).inc(cost_usd)
        else:
            self._record_local_metric("cost_usd_total", cost_usd, {
                "provider": provider, "model": model, "tenant": tenant
            })
            
        logger.debug(f"Recorded cost: ${cost_usd:.4f} for {model} on {provider} (tenant: {tenant})")
    
    def record_session_round(self, mode: str, converged: bool):
        """Record session round completion."""
        converged_str = "true" if converged else "false"
        
        if self.enable_prometheus:
            self.session_rounds.labels(mode=mode, converged=converged_str).inc()
        else:
            self._record_local_metric("session_rounds_total", 1, {
                "mode": mode, "converged": converged_str
            })
            
        logger.debug(f"Recorded session round: mode={mode}, converged={converged}")
    
    def record_tool_invocation(self, tool_name: str, status: str):
        """Record tool invocation metric."""
        if self.enable_prometheus:
            self.tool_invocations.labels(tool_name=tool_name, status=status).inc()
        else:
            self._record_local_metric("tool_invocations_total", 1, {
                "tool_name": tool_name, "status": status
            })
            
        logger.debug(f"Recorded tool invocation: {tool_name} -> {status}")
    
    def record_dag_node_duration(self, duration_seconds: float, agent: str, node_type: str):
        """Record DAG node execution duration."""
        if self.enable_prometheus:
            self.dag_node_duration.labels(agent=agent, node_type=node_type).observe(duration_seconds)
        else:
            self._record_local_metric("dag_node_duration_seconds", duration_seconds, {
                "agent": agent, "node_type": node_type
            })
            
        logger.debug(f"Recorded DAG node duration: {duration_seconds:.3f}s for {agent} node {node_type}")
    
    def set_active_sessions(self, count: int):
        """Set current active session count."""
        if self.enable_prometheus:
            self.active_sessions.set(count)
        else:
            self._record_local_metric("active_sessions", count, {})
            
        logger.debug(f"Set active sessions count: {count}")
    
    def _record_local_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Record metric locally when Prometheus is not available."""
        metric = MetricValue(
            name=name,
            value=value,
            labels=labels,
            timestamp=datetime.now()
        )
        self._local_metrics.append(metric)
        
        # Keep only last 1000 metrics to prevent memory growth
        if len(self._local_metrics) > 1000:
            self._local_metrics = self._local_metrics[-1000:]
    
    def get_local_metrics(self) -> List[MetricValue]:
        """Get locally stored metrics (when Prometheus not available)."""
        return self._local_metrics.copy()
    
    def clear_local_metrics(self):
        """Clear locally stored metrics."""
        self._local_metrics.clear()


class MetricsCollector:
    """
    Context manager and decorator for automatic metrics collection.
    
    Provides easy instrumentation of agent operations with timing and metrics.
    """
    
    def __init__(self, metrics: AgentNetMetrics):
        self.metrics = metrics
    
    @contextmanager
    def measure_inference(self, model: str, provider: str, agent: str):
        """Context manager for measuring inference duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.record_inference_latency(duration, model, provider, agent)
    
    @contextmanager
    def measure_dag_node(self, agent: str, node_type: str):
        """Context manager for measuring DAG node execution."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.record_dag_node_duration(duration, agent, node_type)
    
    def record_operation_metrics(self, operation_result: Dict[str, Any]):
        """
        Record metrics from a completed operation result.
        
        Expected keys in operation_result:
        - tokens_input, tokens_output: int
        - model, provider: str
        - agent_name: str
        - cost_usd: float (optional)
        - tenant_id: str (optional)
        - violations: List[Dict] (optional)
        """
        model = operation_result.get("model", "unknown")
        provider = operation_result.get("provider", "unknown")
        agent = operation_result.get("agent_name", "unknown")
        tenant = operation_result.get("tenant_id")
        
        # Record token consumption
        tokens_input = operation_result.get("tokens_input", 0)
        tokens_output = operation_result.get("tokens_output", 0)
        total_tokens = tokens_input + tokens_output
        
        if total_tokens > 0:
            self.metrics.record_tokens_consumed(total_tokens, model, provider, tenant)
        
        # Record cost if provided
        cost_usd = operation_result.get("cost_usd")
        if cost_usd is not None:
            self.metrics.record_cost(cost_usd, provider, model, tenant)
        
        # Record violations if any
        violations = operation_result.get("violations", [])
        for violation in violations:
            severity = violation.get("severity", "unknown")
            rule_name = violation.get("rule_name", "unknown")
            self.metrics.record_violation(severity, rule_name)


# Global metrics instance (can be overridden)
_global_metrics: Optional[AgentNetMetrics] = None

def get_global_metrics() -> AgentNetMetrics:
    """Get or create global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = AgentNetMetrics()
    return _global_metrics

def set_global_metrics(metrics: AgentNetMetrics):
    """Set global metrics instance."""
    global _global_metrics
    _global_metrics = metrics