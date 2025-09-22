"""
Monitoring Stack Integrations

This module provides integrations with Grafana and Prometheus
for comprehensive monitoring, alerting, and observability of
AgentNet systems.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    help_text: Optional[str] = None


@dataclass
class Alert:
    """Represents an alert condition."""
    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str = "warning"  # warning, critical, info
    description: Optional[str] = None
    runbook_url: Optional[str] = None


class PrometheusIntegration:
    """
    Prometheus metrics integration for AgentNet.
    
    Features:
    - Custom metrics collection
    - Automatic AgentNet metrics
    - Push Gateway support
    - Service discovery integration
    """
    
    def __init__(
        self,
        registry: Optional[Any] = None,
        pushgateway_url: Optional[str] = None,
        job_name: str = "agentnet",
        instance_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Prometheus integration.
        
        Args:
            registry: Prometheus registry (creates default if None)
            pushgateway_url: Push Gateway URL for batch metrics
            job_name: Job name for metrics
            instance_name: Instance identifier
            **kwargs: Additional configuration
        """
        try:
            from prometheus_client import (
                CollectorRegistry, Counter, Histogram, Gauge, Summary,
                push_to_gateway, delete_from_gateway, REGISTRY
            )
            self.prometheus_client = __import__('prometheus_client')
            self.Counter = Counter
            self.Histogram = Histogram
            self.Gauge = Gauge
            self.Summary = Summary
            self.push_to_gateway = push_to_gateway
            self.delete_from_gateway = delete_from_gateway
            PROMETHEUS_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "Prometheus integration requires: pip install prometheus-client>=0.16.0"
            )
        
        # Configuration
        self.registry = registry or CollectorRegistry()
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.instance_name = instance_name or f"agentnet-{int(time.time())}"
        
        # Metrics storage
        self.metrics = {}
        self.custom_metrics = {}
        self.alerting_rules = []
        
        # Background processing
        self.metric_buffer = deque(maxlen=10000)
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        self._initialize_default_metrics()
        
        if kwargs.get("start_background_processing", True):
            self.start_background_processing()
    
    def _initialize_default_metrics(self):
        """Initialize default AgentNet metrics."""
        # Agent interaction metrics
        self.metrics['agent_inferences_total'] = self.Counter(
            'agentnet_agent_inferences_total',
            'Total number of agent inferences',
            ['agent_name', 'provider', 'model'],
            registry=self.registry
        )
        
        self.metrics['agent_inference_duration'] = self.Histogram(
            'agentnet_agent_inference_duration_seconds',
            'Agent inference duration in seconds',
            ['agent_name', 'provider', 'model'],
            registry=self.registry
        )
        
        self.metrics['agent_tokens_consumed'] = self.Counter(
            'agentnet_agent_tokens_consumed_total',
            'Total tokens consumed by agents',
            ['agent_name', 'provider', 'model', 'token_type'],
            registry=self.registry
        )
        
        self.metrics['agent_cost_usd'] = self.Counter(
            'agentnet_agent_cost_usd_total',
            'Total cost in USD',
            ['agent_name', 'provider', 'model'],
            registry=self.registry
        )
        
        # Monitor metrics
        self.metrics['monitor_violations'] = self.Counter(
            'agentnet_monitor_violations_total',
            'Total monitor violations',
            ['monitor_type', 'severity', 'agent_name'],
            registry=self.registry
        )
        
        # Session metrics
        self.metrics['session_rounds'] = self.Histogram(
            'agentnet_session_rounds',
            'Number of rounds per session',
            ['session_type', 'converged'],
            registry=self.registry
        )
        
        # Tool invocation metrics
        self.metrics['tool_invocations'] = self.Counter(
            'agentnet_tool_invocations_total',
            'Total tool invocations',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        # System metrics
        self.metrics['system_memory_usage'] = self.Gauge(
            'agentnet_system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.metrics['active_agents'] = self.Gauge(
            'agentnet_active_agents',
            'Number of active agents',
            registry=self.registry
        )
    
    def record_inference(
        self,
        agent_name: str,
        provider: str,
        model: str,
        duration: float,
        tokens_input: int,
        tokens_output: int,
        cost: float
    ):
        """Record agent inference metrics."""
        labels = [agent_name, provider, model]
        
        self.metrics['agent_inferences_total'].labels(*labels).inc()
        self.metrics['agent_inference_duration'].labels(*labels).observe(duration)
        self.metrics['agent_tokens_consumed'].labels(*labels, 'input').inc(tokens_input)
        self.metrics['agent_tokens_consumed'].labels(*labels, 'output').inc(tokens_output)
        self.metrics['agent_cost_usd'].labels(*labels).inc(cost)
    
    def record_violation(self, monitor_type: str, severity: str, agent_name: str):
        """Record monitor violation."""
        self.metrics['monitor_violations'].labels(
            monitor_type, severity, agent_name
        ).inc()
    
    def record_session(self, session_type: str, rounds: int, converged: bool):
        """Record session metrics."""
        self.metrics['session_rounds'].labels(
            session_type, str(converged).lower()
        ).observe(rounds)
    
    def record_tool_invocation(self, tool_name: str, status: str):
        """Record tool invocation."""
        self.metrics['tool_invocations'].labels(tool_name, status).inc()
    
    def update_system_metrics(self, memory_usage: float, active_agents: int):
        """Update system metrics."""
        self.metrics['system_memory_usage'].set(memory_usage)
        self.metrics['active_agents'].set(active_agents)
    
    def create_custom_metric(
        self,
        name: str,
        metric_type: str,
        description: str,
        labels: Optional[List[str]] = None
    ):
        """
        Create a custom metric.
        
        Args:
            name: Metric name
            metric_type: Type (counter, gauge, histogram, summary)
            description: Metric description
            labels: Label names
        """
        labels = labels or []
        
        if metric_type.lower() == "counter":
            metric = self.Counter(name, description, labels, registry=self.registry)
        elif metric_type.lower() == "gauge":
            metric = self.Gauge(name, description, labels, registry=self.registry)
        elif metric_type.lower() == "histogram":
            metric = self.Histogram(name, description, labels, registry=self.registry)
        elif metric_type.lower() == "summary":
            metric = self.Summary(name, description, labels, registry=self.registry)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self.custom_metrics[name] = metric
        return metric
    
    def push_metrics(self):
        """Push metrics to Prometheus Push Gateway."""
        if not self.pushgateway_url:
            logger.warning("No Push Gateway URL configured")
            return False
        
        try:
            self.push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key={'instance': self.instance_name}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus: {e}")
            return False
    
    def start_background_processing(self):
        """Start background thread for metric processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(
            target=self._background_processor,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Started Prometheus background processing")
    
    def stop_background_processing(self):
        """Stop background metric processing."""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Stopped Prometheus background processing")
    
    def _background_processor(self):
        """Background thread for processing metrics."""
        while not self.stop_processing.is_set():
            try:
                # Process buffered metrics
                while self.metric_buffer:
                    metric_data = self.metric_buffer.popleft()
                    self._process_metric(metric_data)
                
                # Push metrics periodically
                if self.pushgateway_url:
                    self.push_metrics()
                
                time.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in Prometheus background processing: {e}")
                time.sleep(5)
    
    def _process_metric(self, metric_data: Dict[str, Any]):
        """Process a single metric from buffer."""
        try:
            metric_name = metric_data['name']
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                labels = metric_data.get('labels', [])
                value = metric_data['value']
                
                if hasattr(metric, 'inc'):
                    metric.labels(*labels).inc(value)
                elif hasattr(metric, 'observe'):
                    metric.labels(*labels).observe(value)
                elif hasattr(metric, 'set'):
                    metric.labels(*labels).set(value)
        except Exception as e:
            logger.error(f"Failed to process metric {metric_data}: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode('utf-8')
    
    def create_alerting_rule(self, alert: Alert) -> Dict[str, Any]:
        """
        Create a Prometheus alerting rule.
        
        Args:
            alert: Alert configuration
            
        Returns:
            Prometheus alerting rule configuration
        """
        rule = {
            "alert": alert.name,
            "expr": alert.condition,
            "for": f"{alert.duration}s",
            "labels": {
                "severity": alert.severity,
            },
            "annotations": {
                "summary": alert.description or f"Alert: {alert.name}",
            }
        }
        
        if alert.runbook_url:
            rule["annotations"]["runbook_url"] = alert.runbook_url
        
        self.alerting_rules.append(rule)
        return rule
    
    def get_alerting_rules_yaml(self) -> str:
        """Get alerting rules in Prometheus YAML format."""
        import yaml
        
        rules_config = {
            "groups": [{
                "name": "agentnet_alerts",
                "rules": self.alerting_rules
            }]
        }
        
        return yaml.dump(rules_config, default_flow_style=False)


class GrafanaIntegration:
    """
    Grafana dashboard and alerting integration.
    
    Features:
    - Automatic dashboard creation
    - Custom dashboard templates
    - Alert notification channels
    - Data source management
    """
    
    def __init__(
        self,
        url: str,
        api_key: str,
        organization_id: int = 1,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize Grafana integration.
        
        Args:
            url: Grafana instance URL
            api_key: Grafana API key
            organization_id: Organization ID
            timeout: Request timeout
            **kwargs: Additional configuration
        """
        try:
            import requests
            self.requests = requests
            REQUESTS_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "Grafana integration requires: pip install requests"
            )
        
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.organization_id = organization_id
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Grafana-Org-Id": str(organization_id)
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Grafana."""
        try:
            response = self.requests.get(
                f"{self.url}/api/health",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("Connected to Grafana successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Grafana: {e}")
            raise
    
    def create_data_source(
        self,
        name: str,
        ds_type: str,
        url: str,
        access: str = "proxy",
        **config
    ) -> Dict[str, Any]:
        """
        Create a data source in Grafana.
        
        Args:
            name: Data source name
            ds_type: Data source type (prometheus, etc.)
            url: Data source URL
            access: Access mode (proxy, direct)
            **config: Additional configuration
            
        Returns:
            Created data source information
        """
        datasource_config = {
            "name": name,
            "type": ds_type,
            "url": url,
            "access": access,
            "isDefault": config.get("is_default", False),
            **config
        }
        
        try:
            response = self.requests.post(
                f"{self.url}/api/datasources",
                headers=self.headers,
                json=datasource_config,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created Grafana data source: {name}")
            return result
        except Exception as e:
            logger.error(f"Failed to create Grafana data source: {e}")
            raise
    
    def create_dashboard(
        self,
        title: str,
        panels: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
        folder_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a dashboard in Grafana.
        
        Args:
            title: Dashboard title
            panels: List of panel configurations
            tags: Dashboard tags
            folder_id: Folder ID for organization
            
        Returns:
            Created dashboard information
        """
        dashboard_config = {
            "dashboard": {
                "title": title,
                "tags": tags or [],
                "panels": panels,
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s",
                "schemaVersion": 30,
                "version": 1,
            },
            "folderId": folder_id,
            "overwrite": True
        }
        
        try:
            response = self.requests.post(
                f"{self.url}/api/dashboards/db",
                headers=self.headers,
                json=dashboard_config,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created Grafana dashboard: {title}")
            return result
        except Exception as e:
            logger.error(f"Failed to create Grafana dashboard: {e}")
            raise
    
    def create_agentnet_dashboard(
        self,
        prometheus_datasource: str = "Prometheus"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive AgentNet monitoring dashboard.
        
        Args:
            prometheus_datasource: Name of Prometheus data source
            
        Returns:
            Created dashboard information
        """
        panels = [
            # Agent Inferences Panel
            {
                "id": 1,
                "title": "Agent Inferences Rate",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [{
                    "expr": "rate(agentnet_agent_inferences_total[5m])",
                    "legendFormat": "{{agent_name}} - {{provider}}",
                    "datasource": prometheus_datasource
                }],
                "yAxes": [{
                    "label": "Requests/sec",
                    "min": 0
                }, {}],
                "legend": {"show": True}
            },
            
            # Response Time Panel
            {
                "id": 2,
                "title": "Agent Response Time",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(agentnet_agent_inference_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile",
                    "datasource": prometheus_datasource
                }, {
                    "expr": "histogram_quantile(0.50, rate(agentnet_agent_inference_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile",
                    "datasource": prometheus_datasource
                }],
                "yAxes": [{
                    "label": "Seconds",
                    "min": 0
                }, {}]
            },
            
            # Token Consumption Panel
            {
                "id": 3,
                "title": "Token Consumption",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [{
                    "expr": "rate(agentnet_agent_tokens_consumed_total[5m])",
                    "legendFormat": "{{token_type}} - {{agent_name}}",
                    "datasource": prometheus_datasource
                }],
                "yAxes": [{
                    "label": "Tokens/sec",
                    "min": 0
                }, {}]
            },
            
            # Cost Panel
            {
                "id": 4,
                "title": "Cost per Hour",
                "type": "singlestat",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [{
                    "expr": "rate(agentnet_agent_cost_usd_total[1h]) * 3600",
                    "datasource": prometheus_datasource
                }],
                "valueName": "current",
                "format": "currencyUSD",
                "colorBackground": True,
                "thresholds": "5,20"
            },
            
            # Monitor Violations Panel
            {
                "id": 5,
                "title": "Monitor Violations",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [{
                    "expr": "rate(agentnet_monitor_violations_total[5m])",
                    "legendFormat": "{{severity}} - {{monitor_type}}",
                    "datasource": prometheus_datasource
                }],
                "yAxes": [{
                    "label": "Violations/sec",
                    "min": 0
                }, {}]
            },
            
            # Active Agents Panel
            {
                "id": 6,
                "title": "Active Agents",
                "type": "singlestat",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                "targets": [{
                    "expr": "agentnet_active_agents",
                    "datasource": prometheus_datasource
                }],
                "valueName": "current",
                "colorBackground": True,
                "thresholds": "10,50"
            }
        ]
        
        return self.create_dashboard(
            title="AgentNet Overview",
            panels=panels,
            tags=["agentnet", "monitoring", "overview"]
        )
    
    def create_notification_channel(
        self,
        name: str,
        channel_type: str,
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a notification channel for alerts.
        
        Args:
            name: Channel name
            channel_type: Channel type (slack, email, webhook, etc.)
            settings: Channel-specific settings
            
        Returns:
            Created notification channel information
        """
        channel_config = {
            "name": name,
            "type": channel_type,
            "settings": settings
        }
        
        try:
            response = self.requests.post(
                f"{self.url}/api/alert-notifications",
                headers=self.headers,
                json=channel_config,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created Grafana notification channel: {name}")
            return result
        except Exception as e:
            logger.error(f"Failed to create notification channel: {e}")
            raise
    
    def create_alert_rule(
        self,
        dashboard_id: int,
        panel_id: int,
        name: str,
        condition: Dict[str, Any],
        notification_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Create an alert rule for a dashboard panel.
        
        Args:
            dashboard_id: Dashboard ID
            panel_id: Panel ID
            name: Alert name
            condition: Alert condition configuration
            notification_ids: List of notification channel IDs
            
        Returns:
            Created alert rule information
        """
        alert_config = {
            "dashboardId": dashboard_id,
            "panelId": panel_id,
            "name": name,
            "message": f"Alert: {name}",
            "frequency": "10s",
            "conditions": [condition],
            "executionErrorState": "alerting",
            "noDataState": "no_data",
            "for": "5m",
            "notifications": [{"id": nid} for nid in notification_ids]
        }
        
        try:
            response = self.requests.post(
                f"{self.url}/api/alerts",
                headers=self.headers,
                json=alert_config,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created Grafana alert rule: {name}")
            return result
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            raise
    
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of all dashboards."""
        try:
            response = self.requests.get(
                f"{self.url}/api/search?type=dash-db",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get dashboard list: {e}")
            return []
    
    def export_dashboard(self, dashboard_id: int) -> Dict[str, Any]:
        """Export dashboard configuration."""
        try:
            response = self.requests.get(
                f"{self.url}/api/dashboards/id/{dashboard_id}",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            return {}


# Utility functions
def setup_agentnet_monitoring(
    prometheus_config: Dict[str, Any],
    grafana_config: Dict[str, Any]
) -> Tuple[PrometheusIntegration, GrafanaIntegration]:
    """
    Set up complete AgentNet monitoring stack.
    
    Args:
        prometheus_config: Prometheus configuration
        grafana_config: Grafana configuration
        
    Returns:
        Tuple of (PrometheusIntegration, GrafanaIntegration)
    """
    # Initialize Prometheus
    prometheus = PrometheusIntegration(**prometheus_config)
    
    # Initialize Grafana
    grafana = GrafanaIntegration(**grafana_config)
    
    # Create Prometheus data source in Grafana
    if "prometheus_url" in prometheus_config:
        try:
            grafana.create_data_source(
                name="Prometheus",
                ds_type="prometheus",
                url=prometheus_config["prometheus_url"],
                is_default=True
            )
        except Exception as e:
            logger.warning(f"Failed to create Prometheus data source: {e}")
    
    # Create AgentNet dashboard
    try:
        grafana.create_agentnet_dashboard()
    except Exception as e:
        logger.warning(f"Failed to create AgentNet dashboard: {e}")
    
    return prometheus, grafana


def create_default_alerts() -> List[Alert]:
    """Create default AgentNet alert rules."""
    return [
        Alert(
            name="HighInferenceLatency",
            condition="histogram_quantile(0.95, rate(agentnet_agent_inference_duration_seconds_bucket[5m])) > 10",
            threshold=10.0,
            duration=300,
            severity="warning",
            description="Agent inference latency is high"
        ),
        Alert(
            name="HighCostRate",
            condition="rate(agentnet_agent_cost_usd_total[1h]) * 3600 > 50",
            threshold=50.0,
            duration=300,
            severity="critical",
            description="Cost rate is exceeding budget"
        ),
        Alert(
            name="MonitorViolationSpike",
            condition="rate(agentnet_monitor_violations_total[5m]) > 1",
            threshold=1.0,
            duration=60,
            severity="warning",
            description="Monitor violations are spiking"
        ),
        Alert(
            name="NoActiveAgents",
            condition="agentnet_active_agents == 0",
            threshold=0.0,
            duration=300,
            severity="critical",
            description="No active agents detected"
        ),
    ]