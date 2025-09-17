"""
Dashboard Components for AgentNet Observability.

Provides cost aggregation, performance metrics visualization, and violation tracking
dashboards as specified in RoadmapAgentNet.md P5 requirements.

Supports both web-based dashboards and programmatic access to aggregated data.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger("agentnet.observability.dashboard")


class TimeRange(str, Enum):
    """Time range options for dashboard data."""
    LAST_HOUR = "1h"
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"
    CUSTOM = "custom"


@dataclass
class CostSummary:
    """Cost summary data structure."""
    total_cost_usd: float
    cost_by_provider: Dict[str, float]
    cost_by_model: Dict[str, float]
    cost_by_tenant: Dict[str, float]
    token_count: int
    time_range: str
    start_time: datetime
    end_time: datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics summary."""
    avg_inference_latency_ms: float
    p95_inference_latency_ms: float
    p99_inference_latency_ms: float
    total_requests: int
    requests_per_minute: float
    error_rate_percent: float
    time_range: str
    start_time: datetime
    end_time: datetime


@dataclass
class ViolationSummary:
    """Policy violation summary."""
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_rule: Dict[str, int]
    recent_violations: List[Dict[str, Any]]
    time_range: str
    start_time: datetime
    end_time: datetime


@dataclass
class SessionMetrics:
    """Session-level metrics summary."""
    total_sessions: int
    active_sessions: int
    avg_rounds_per_session: float
    convergence_rate_percent: float
    mode_distribution: Dict[str, int]
    time_range: str
    start_time: datetime
    end_time: datetime


class DashboardDataCollector:
    """
    Collects and aggregates observability data for dashboard display.
    
    Integrates with metrics, tracing, and logging systems to provide
    comprehensive dashboard data.
    """
    
    def __init__(self, metrics_collector=None):
        """
        Initialize dashboard data collector.
        
        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self._cost_data: List[Dict[str, Any]] = []
        self._performance_data: List[Dict[str, Any]] = []
        self._violation_data: List[Dict[str, Any]] = []
        self._session_data: List[Dict[str, Any]] = []
    
    def add_cost_event(self, provider: str, model: str, cost_usd: float, 
                      tokens: int, tenant_id: Optional[str] = None,
                      timestamp: Optional[datetime] = None):
        """Add cost tracking event."""
        event = {
            "provider": provider,
            "model": model,
            "cost_usd": cost_usd,
            "tokens": tokens,
            "tenant_id": tenant_id or "default",
            "timestamp": timestamp or datetime.now()
        }
        self._cost_data.append(event)
        logger.debug(f"Added cost event: ${cost_usd:.4f} for {model} on {provider}")
    
    def add_performance_event(self, agent_name: str, model: str, provider: str,
                            latency_ms: float, success: bool,
                            timestamp: Optional[datetime] = None):
        """Add performance tracking event."""
        event = {
            "agent_name": agent_name,
            "model": model,
            "provider": provider,
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": timestamp or datetime.now()
        }
        self._performance_data.append(event)
        logger.debug(f"Added performance event: {latency_ms:.2f}ms for {agent_name}")
    
    def add_violation_event(self, rule_name: str, severity: str, 
                          agent_name: Optional[str] = None,
                          details: Optional[str] = None,
                          timestamp: Optional[datetime] = None):
        """Add policy violation event."""
        event = {
            "rule_name": rule_name,
            "severity": severity,
            "agent_name": agent_name,
            "details": details,
            "timestamp": timestamp or datetime.now()
        }
        self._violation_data.append(event)
        logger.debug(f"Added violation event: {severity} - {rule_name}")
    
    def add_session_event(self, session_id: str, mode: str, event_type: str,
                         round_number: Optional[int] = None,
                         converged: Optional[bool] = None,
                         timestamp: Optional[datetime] = None):
        """Add session lifecycle event."""
        event = {
            "session_id": session_id,
            "mode": mode,
            "event_type": event_type,  # start, round, end
            "round_number": round_number,
            "converged": converged,
            "timestamp": timestamp or datetime.now()
        }
        self._session_data.append(event)
        logger.debug(f"Added session event: {event_type} for {session_id}")
    
    def get_cost_summary(self, time_range: TimeRange = TimeRange.LAST_24H,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> CostSummary:
        """Get cost summary for specified time range."""
        start_time, end_time = self._resolve_time_range(time_range, start_time, end_time)
        
        filtered_data = [
            event for event in self._cost_data
            if start_time <= event["timestamp"] <= end_time
        ]
        
        if not filtered_data:
            return CostSummary(
                total_cost_usd=0.0,
                cost_by_provider={},
                cost_by_model={},
                cost_by_tenant={},
                token_count=0,
                time_range=time_range.value,
                start_time=start_time,
                end_time=end_time
            )
        
        total_cost = sum(event["cost_usd"] for event in filtered_data)
        total_tokens = sum(event["tokens"] for event in filtered_data)
        
        # Aggregate by provider
        cost_by_provider = defaultdict(float)
        for event in filtered_data:
            cost_by_provider[event["provider"]] += event["cost_usd"]
        
        # Aggregate by model
        cost_by_model = defaultdict(float)
        for event in filtered_data:
            cost_by_model[event["model"]] += event["cost_usd"]
        
        # Aggregate by tenant
        cost_by_tenant = defaultdict(float)
        for event in filtered_data:
            cost_by_tenant[event["tenant_id"]] += event["cost_usd"]
        
        return CostSummary(
            total_cost_usd=total_cost,
            cost_by_provider=dict(cost_by_provider),
            cost_by_model=dict(cost_by_model),
            cost_by_tenant=dict(cost_by_tenant),
            token_count=total_tokens,
            time_range=time_range.value,
            start_time=start_time,
            end_time=end_time
        )
    
    def get_performance_metrics(self, time_range: TimeRange = TimeRange.LAST_24H,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> PerformanceMetrics:
        """Get performance metrics for specified time range."""
        start_time, end_time = self._resolve_time_range(time_range, start_time, end_time)
        
        filtered_data = [
            event for event in self._performance_data
            if start_time <= event["timestamp"] <= end_time
        ]
        
        if not filtered_data:
            return PerformanceMetrics(
                avg_inference_latency_ms=0.0,
                p95_inference_latency_ms=0.0,
                p99_inference_latency_ms=0.0,
                total_requests=0,
                requests_per_minute=0.0,
                error_rate_percent=0.0,
                time_range=time_range.value,
                start_time=start_time,
                end_time=end_time
            )
        
        latencies = [event["latency_ms"] for event in filtered_data]
        latencies.sort()
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
        
        total_requests = len(filtered_data)
        successful_requests = sum(1 for event in filtered_data if event["success"])
        error_rate = ((total_requests - successful_requests) / total_requests * 100) if total_requests > 0 else 0.0
        
        duration_minutes = (end_time - start_time).total_seconds() / 60
        requests_per_minute = total_requests / duration_minutes if duration_minutes > 0 else 0.0
        
        return PerformanceMetrics(
            avg_inference_latency_ms=avg_latency,
            p95_inference_latency_ms=p95_latency,
            p99_inference_latency_ms=p99_latency,
            total_requests=total_requests,
            requests_per_minute=requests_per_minute,
            error_rate_percent=error_rate,
            time_range=time_range.value,
            start_time=start_time,
            end_time=end_time
        )
    
    def get_violation_summary(self, time_range: TimeRange = TimeRange.LAST_24H,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> ViolationSummary:
        """Get violation summary for specified time range."""
        start_time, end_time = self._resolve_time_range(time_range, start_time, end_time)
        
        filtered_data = [
            event for event in self._violation_data
            if start_time <= event["timestamp"] <= end_time
        ]
        
        if not filtered_data:
            return ViolationSummary(
                total_violations=0,
                violations_by_severity={},
                violations_by_rule={},
                recent_violations=[],
                time_range=time_range.value,
                start_time=start_time,
                end_time=end_time
            )
        
        # Aggregate by severity
        violations_by_severity = defaultdict(int)
        for event in filtered_data:
            violations_by_severity[event["severity"]] += 1
        
        # Aggregate by rule
        violations_by_rule = defaultdict(int)
        for event in filtered_data:
            violations_by_rule[event["rule_name"]] += 1
        
        # Get recent violations (last 10)
        recent_violations = sorted(filtered_data, key=lambda x: x["timestamp"], reverse=True)[:10]
        
        return ViolationSummary(
            total_violations=len(filtered_data),
            violations_by_severity=dict(violations_by_severity),
            violations_by_rule=dict(violations_by_rule),
            recent_violations=recent_violations,
            time_range=time_range.value,
            start_time=start_time,
            end_time=end_time
        )
    
    def get_session_metrics(self, time_range: TimeRange = TimeRange.LAST_24H,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> SessionMetrics:
        """Get session metrics for specified time range."""
        start_time, end_time = self._resolve_time_range(time_range, start_time, end_time)
        
        filtered_data = [
            event for event in self._session_data
            if start_time <= event["timestamp"] <= end_time
        ]
        
        if not filtered_data:
            return SessionMetrics(
                total_sessions=0,
                active_sessions=0,
                avg_rounds_per_session=0.0,
                convergence_rate_percent=0.0,
                mode_distribution={},
                time_range=time_range.value,
                start_time=start_time,
                end_time=end_time
            )
        
        # Group by session
        sessions = defaultdict(list)
        for event in filtered_data:
            sessions[event["session_id"]].append(event)
        
        total_sessions = len(sessions)
        
        # Count active sessions (sessions with recent activity)
        recent_cutoff = datetime.now() - timedelta(minutes=30)
        active_sessions = len([
            sid for sid, events in sessions.items()
            if any(event["timestamp"] > recent_cutoff for event in events)
        ])
        
        # Calculate average rounds per session
        total_rounds = 0
        converged_sessions = 0
        mode_distribution = defaultdict(int)
        
        for session_events in sessions.values():
            # Find max round number
            max_round = max((event.get("round_number", 0) or 0 for event in session_events), default=0)
            total_rounds += max_round
            
            # Check if session converged
            if any(event.get("converged") for event in session_events):
                converged_sessions += 1
            
            # Count mode distribution
            modes = [event["mode"] for event in session_events if event["mode"]]
            if modes:
                mode_distribution[modes[0]] += 1  # Use first mode found
        
        avg_rounds = total_rounds / total_sessions if total_sessions > 0 else 0.0
        convergence_rate = (converged_sessions / total_sessions * 100) if total_sessions > 0 else 0.0
        
        return SessionMetrics(
            total_sessions=total_sessions,
            active_sessions=active_sessions,
            avg_rounds_per_session=avg_rounds,
            convergence_rate_percent=convergence_rate,
            mode_distribution=dict(mode_distribution),
            time_range=time_range.value,
            start_time=start_time,
            end_time=end_time
        )
    
    def _resolve_time_range(self, time_range: TimeRange, 
                          start_time: Optional[datetime],
                          end_time: Optional[datetime]) -> Tuple[datetime, datetime]:
        """Resolve time range to start and end datetime objects."""
        if time_range == TimeRange.CUSTOM:
            if not start_time or not end_time:
                raise ValueError("Custom time range requires start_time and end_time")
            return start_time, end_time
        
        end_time = datetime.now()
        
        if time_range == TimeRange.LAST_HOUR:
            start_time = end_time - timedelta(hours=1)
        elif time_range == TimeRange.LAST_24H:
            start_time = end_time - timedelta(days=1)
        elif time_range == TimeRange.LAST_7D:
            start_time = end_time - timedelta(days=7)
        elif time_range == TimeRange.LAST_30D:
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)  # Default to 24h
        
        return start_time, end_time
    
    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export all dashboard data for external analysis."""
        return {
            "cost_summary": asdict(self.get_cost_summary()),
            "performance_metrics": asdict(self.get_performance_metrics()),
            "violation_summary": asdict(self.get_violation_summary()),
            "session_metrics": asdict(self.get_session_metrics()),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def generate_dashboard_html(self, time_range: TimeRange = TimeRange.LAST_24H) -> str:
        """Generate simple HTML dashboard."""
        cost_summary = self.get_cost_summary(time_range)
        performance_metrics = self.get_performance_metrics(time_range)
        violation_summary = self.get_violation_summary(time_range)
        session_metrics = self.get_session_metrics(time_range)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgentNet Observability Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #f9f9f9; }}
                .metric {{ margin: 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ color: #666; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #555; margin-top: 0; }}
                .error {{ color: #f44336; }}
                .success {{ color: #4CAF50; }}
                .warning {{ color: #ff9800; }}
            </style>
        </head>
        <body>
            <h1>AgentNet Observability Dashboard</h1>
            <p style="text-align: center; color: #666;">Time Range: {time_range.value} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="dashboard">
                <div class="card">
                    <h2>💰 Cost Summary</h2>
                    <div class="metric">
                        <div class="metric-value">${cost_summary.total_cost_usd:.4f}</div>
                        <div class="metric-label">Total Cost</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{cost_summary.token_count:,}</div>
                        <div class="metric-label">Total Tokens</div>
                    </div>
                    <h3>By Provider</h3>
                    {self._format_dict_as_html(cost_summary.cost_by_provider, "${:.4f}")}
                </div>
                
                <div class="card">
                    <h2>⚡ Performance</h2>
                    <div class="metric">
                        <div class="metric-value">{performance_metrics.avg_inference_latency_ms:.1f}ms</div>
                        <div class="metric-label">Avg Latency</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{performance_metrics.total_requests:,}</div>
                        <div class="metric-label">Total Requests</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {'error' if performance_metrics.error_rate_percent > 5 else 'success'}">{performance_metrics.error_rate_percent:.1f}%</div>
                        <div class="metric-label">Error Rate</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🚨 Violations</h2>
                    <div class="metric">
                        <div class="metric-value {'error' if violation_summary.total_violations > 0 else 'success'}">{violation_summary.total_violations}</div>
                        <div class="metric-label">Total Violations</div>
                    </div>
                    <h3>By Severity</h3>
                    {self._format_dict_as_html(violation_summary.violations_by_severity)}
                </div>
                
                <div class="card">
                    <h2>🎯 Sessions</h2>
                    <div class="metric">
                        <div class="metric-value">{session_metrics.total_sessions}</div>
                        <div class="metric-label">Total Sessions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value success">{session_metrics.active_sessions}</div>
                        <div class="metric-label">Active Sessions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{session_metrics.convergence_rate_percent:.1f}%</div>
                        <div class="metric-label">Convergence Rate</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_dict_as_html(self, data: Dict[str, Any], format_str: str = "{}") -> str:
        """Format dictionary as HTML list."""
        if not data:
            return "<p>No data</p>"
        
        items = []
        for key, value in data.items():
            formatted_value = format_str.format(value)
            items.append(f"<div>{key}: <strong>{formatted_value}</strong></div>")
        
        return "\n".join(items)


# Global dashboard collector instance
_global_dashboard_collector: Optional[DashboardDataCollector] = None

def get_global_dashboard_collector() -> DashboardDataCollector:
    """Get or create global dashboard collector."""
    global _global_dashboard_collector
    if _global_dashboard_collector is None:
        _global_dashboard_collector = DashboardDataCollector()
    return _global_dashboard_collector

def set_global_dashboard_collector(collector: DashboardDataCollector):
    """Set global dashboard collector."""
    global _global_dashboard_collector
    _global_dashboard_collector = collector