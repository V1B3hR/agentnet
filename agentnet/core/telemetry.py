"""Telemetry and observability features for AgentNet."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("agentnet.telemetry")


class EventType(str, Enum):
    """Types of telemetry events."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    TOOL_CALL = "tool_call"
    POLICY_VIOLATION = "policy_violation"
    COST_INCURRED = "cost_incurred"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RETENTION_EVICTION = "retention_eviction"
    SESSION_CHECKPOINT = "session_checkpoint"
    ERROR = "error"
    METRIC = "metric"


@dataclass
class TelemetryEvent:
    """A telemetry event with metadata."""

    event_type: EventType
    timestamp: float
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return json.dumps(self.to_dict())


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A metric measurement."""

    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    labels: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TelemetryCollector:
    """Collects and manages telemetry events and metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.buffer_size = self.config.get("buffer_size", 1000)

        # Event storage
        self.events: List[TelemetryEvent] = []
        self.metrics: Dict[str, List[Metric]] = {}

        # Export configuration
        self.export_dir = Path(self.config.get("export_dir", "telemetry"))
        self.export_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.active_timers: Dict[str, float] = {}

        logger.info(f"TelemetryCollector initialized (enabled={self.enabled})")

    def record_event(
        self,
        event_type: EventType,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Record a telemetry event."""
        if not self.enabled:
            return

        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            session_id=session_id,
            agent_name=agent_name,
            event_data=event_data,
            duration_ms=duration_ms,
            error=error,
            tags=tags,
        )

        self.events.append(event)

        # Trim buffer if too large
        if len(self.events) > self.buffer_size:
            self.events = self.events[-self.buffer_size :]

        logger.debug(f"Recorded event: {event_type}")

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric."""
        if not self.enabled:
            return

        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels,
        )

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(metric)

        # Trim metric history
        max_history = self.config.get("max_metric_history", 1000)
        if len(self.metrics[name]) > max_history:
            self.metrics[name] = self.metrics[name][-max_history:]

        logger.debug(f"Recorded metric: {name}={value}")

    def start_timer(self, timer_name: str) -> None:
        """Start a timer."""
        if not self.enabled:
            return

        self.active_timers[timer_name] = time.time()

    def end_timer(
        self, timer_name: str, event_type: Optional[EventType] = None, **event_kwargs
    ) -> Optional[float]:
        """End a timer and optionally record an event."""
        if not self.enabled or timer_name not in self.active_timers:
            return None

        start_time = self.active_timers.pop(timer_name)
        duration_ms = (time.time() - start_time) * 1000

        # Record timer metric
        self.record_metric(f"timer.{timer_name}", duration_ms, MetricType.TIMER)

        # Record event if requested
        if event_type:
            self.record_event(
                event_type=event_type, duration_ms=duration_ms, **event_kwargs
            )

        return duration_ms

    def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, labels)

    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, labels)

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        session_id: Optional[str] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[TelemetryEvent]:
        """Get filtered events."""
        events = self.events

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if session_id:
            events = [e for e in events if e.session_id == session_id]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return None

        values = [m.value for m in self.metrics[name]]
        if not values:
            return None

        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "latest_timestamp": self.metrics[name][-1].timestamp,
        }

    def export_events_jsonl(self, filename: Optional[str] = None) -> str:
        """Export events to JSONL format."""
        if filename is None:
            filename = f"events_{int(time.time())}.jsonl"

        filepath = self.export_dir / filename

        with open(filepath, "w") as f:
            for event in self.events:
                f.write(event.to_jsonl() + "\n")

        logger.info(f"Exported {len(self.events)} events to {filepath}")
        return str(filepath)

    def export_audit_bundle(self, session_id: str) -> str:
        """Export comprehensive audit bundle for a session."""
        bundle_dir = self.export_dir / f"audit_{session_id}_{int(time.time())}"
        bundle_dir.mkdir(exist_ok=True)

        # Export session events
        session_events = self.get_events(session_id=session_id)
        events_file = bundle_dir / "events.jsonl"
        with open(events_file, "w") as f:
            for event in session_events:
                f.write(event.to_jsonl() + "\n")

        # Export metrics summary
        metrics_summary = {}
        for name in self.metrics:
            summary = self.get_metric_summary(name)
            if summary:
                metrics_summary[name] = summary

        metrics_file = bundle_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # Export configuration (placeholder)
        config_file = bundle_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(
                {"session_id": session_id, "export_time": time.time()}, f, indent=2
            )

        logger.info(f"Exported audit bundle to {bundle_dir}")
        return str(bundle_dir)

    def clear_old_data(self, max_age_seconds: int = 24 * 3600) -> None:
        """Clear old telemetry data."""
        cutoff_time = time.time() - max_age_seconds

        # Clear old events
        original_event_count = len(self.events)
        self.events = [e for e in self.events if e.timestamp >= cutoff_time]

        # Clear old metrics
        original_metric_count = sum(len(metrics) for metrics in self.metrics.values())
        for name in self.metrics:
            self.metrics[name] = [
                m for m in self.metrics[name] if m.timestamp >= cutoff_time
            ]

        new_metric_count = sum(len(metrics) for metrics in self.metrics.values())

        logger.info(
            f"Cleared old telemetry data: "
            f"events {original_event_count} -> {len(self.events)}, "
            f"metrics {original_metric_count} -> {new_metric_count}"
        )


# Global telemetry instance
_telemetry: Optional[TelemetryCollector] = None


def init_telemetry(config: Optional[Dict[str, Any]] = None) -> TelemetryCollector:
    """Initialize global telemetry collector."""
    global _telemetry
    _telemetry = TelemetryCollector(config)
    return _telemetry


def get_telemetry() -> Optional[TelemetryCollector]:
    """Get global telemetry collector."""
    return _telemetry


def record_event(event_type: EventType, **kwargs) -> None:
    """Record event using global telemetry collector."""
    if _telemetry:
        _telemetry.record_event(event_type, **kwargs)


def record_metric(
    name: str, value: Union[int, float], metric_type: MetricType, **kwargs
) -> None:
    """Record metric using global telemetry collector."""
    if _telemetry:
        _telemetry.record_metric(name, value, metric_type, **kwargs)


def start_timer(timer_name: str) -> None:
    """Start timer using global telemetry collector."""
    if _telemetry:
        _telemetry.start_timer(timer_name)


def end_timer(timer_name: str, **kwargs) -> Optional[float]:
    """End timer using global telemetry collector."""
    if _telemetry:
        return _telemetry.end_timer(timer_name, **kwargs)
    return None
