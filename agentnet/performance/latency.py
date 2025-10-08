"""
High-Performance Turn Latency Measurement for AgentNet

Provides automated, context-aware latency tracking using context managers,
with advanced filtering, trend analysis, and production-ready features.
"""

import time
import logging
import statistics
from collections import deque
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Iterator, AsyncIterator

logger = logging.getLogger(__name__)


class LatencyComponent(str, Enum):
    """Components of turn latency that can be measured separately."""
    INFERENCE = "inference"
    POLICY_CHECK = "policy_check"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_ACCESS = "memory_access"
    RESPONSE_PROCESSING = "response_processing"
    ORCHESTRATION = "orchestration" # For strategy logic itself


@dataclass
class TurnLatencyMeasurement:
    """Detailed latency measurement for a single agent turn."""
    turn_id: str
    agent_id: str
    start_time: float
    end_time: float
    
    # Component latencies are stored in milliseconds
    component_latencies: Dict[LatencyComponent, float] = field(default_factory=dict)
    
    # Rich, queryable metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def latency_breakdown_percent(self) -> Dict[str, float]:
        """Breakdown of latency by component as percentages of the total."""
        total = self.total_latency_ms
        if total == 0:
            return {}
        return {
            comp.value: (lat / total) * 100
            for comp, lat in self.component_latencies.items()
        }


class LatencyTracker:
    """
    Tracks and analyzes turn latency using automated context managers.
    Implemented as a singleton for consistent, global access.
    """
    _instance: Optional["LatencyTracker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LatencyTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_history: int = 10000):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self._measurements: deque[TurnLatencyMeasurement] = deque(maxlen=max_history)
        self._active_measurements: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def start_turn_measurement(self, turn_id: str, agent_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Start measuring latency for a new turn with rich context."""
        if turn_id in self._active_measurements:
            logger.warning(f"Measurement for turn_id '{turn_id}' already started. Overwriting.")
        
        self._active_measurements[turn_id] = {
            "agent_id": agent_id,
            "start_time": time.monotonic(),
            "component_latencies": {},
            "metadata": metadata or {},
        }

    @asynccontextmanager
    async def measure(self, turn_id: str, component: LatencyComponent) -> AsyncIterator[None]:
        """Automated, async-compatible context manager for measuring a component's latency."""
        if turn_id not in self._active_measurements:
            logger.warning(f"Cannot measure component '{component.value}' for unknown turn '{turn_id}'.")
            yield
            return

        start_time = time.monotonic()
        try:
            yield
        finally:
            latency_ms = (time.monotonic() - start_time) * 1000
            # Use .setdefault to handle cases where a component is measured multiple times (e.g., tool calls)
            self._active_measurements[turn_id]["component_latencies"].setdefault(component, 0.0)
            self._active_measurements[turn_id]["component_latencies"][component] += latency_ms

    def end_turn_measurement(self, turn_id: str) -> Optional[TurnLatencyMeasurement]:
        """End turn measurement and create the final, immutable measurement record."""
        if turn_id not in self._active_measurements:
            logger.warning(f"No active measurement to end for turn '{turn_id}'.")
            return None

        data = self._active_measurements.pop(turn_id)
        end_time = time.monotonic()

        measurement = TurnLatencyMeasurement(
            turn_id=turn_id,
            agent_id=data["agent_id"],
            start_time=data["start_time"],
            end_time=end_time,
            component_latencies=data["component_latencies"],
            metadata=data["metadata"],
        )
        self._measurements.append(measurement)
        return measurement

    def get_measurements(
        self,
        agent_id: Optional[str] = None,
        min_latency_ms: Optional[float] = None,
        tool_used: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[TurnLatencyMeasurement]:
        """Get latency measurements with advanced filtering capabilities."""
        results = list(self._measurements)

        if agent_id:
            results = [m for m in results if m.agent_id == agent_id]
        if min_latency_ms:
            results = [m for m in results if m.total_latency_ms >= min_latency_ms]
        if tool_used:
            results = [m for m in results if tool_used in m.metadata.get("tools_used", [])]
        if metadata_filter:
            results = [
                m for m in results
                if all(item in m.metadata.items() for item in metadata_filter.items())
            ]
        return results

    def get_latency_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a statistical summary of latencies, including trend analysis."""
        measurements = self.get_measurements(agent_id=agent_id)
        if not measurements:
            return {"message": "No measurements found for the given filters."}

        latencies = [m.total_latency_ms for m in measurements]
        if not latencies:
            return {}

        stats = {
            "count": len(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        }
        
        # Trend analysis: compare last 10% of turns to overall p95
        trend_slice_index = -max(1, len(latencies) // 10)
        recent_latencies = latencies[trend_slice_index:]
        if recent_latencies:
            p95_recent = sorted(recent_latencies)[int(len(recent_latencies) * 0.95)]
            stats["p95_recent_ms"] = p95_recent
            stats["trend"] = "improving" if p95_recent < stats["p95_ms"] else "degrading"

        return stats

    def get_component_breakdown(self, agent_id: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get the average latency and percentage contribution of each component."""
        measurements = self.get_measurements(agent_id=agent_id)
        if not measurements:
            return {}

        component_totals: Dict[LatencyComponent, float] = {}
        component_counts: Dict[LatencyComponent, int] = {}
        for m in measurements:
            for comp, lat in m.component_latencies.items():
                component_totals.setdefault(comp, 0.0)
                component_counts.setdefault(comp, 0)
                component_totals[comp] += lat
                component_counts[comp] += 1

        avg_total_latency = statistics.mean(m.total_latency_ms for m in measurements)
        
        breakdown = {}
        for comp, total_lat in component_totals.items():
            avg_lat = total_lat / component_counts[comp]
            breakdown[comp.value] = {
                "avg_latency_ms": avg_lat,
                "percentage_of_total": (avg_lat / avg_total_latency) * 100 if avg_total_latency > 0 else 0,
                "count": component_counts[comp],
            }
        return breakdown

    def clear_measurements(self):
        """Clear all stored and active measurements."""
        self._measurements.clear()
        self._active_measurements.clear()
        logger.info("Cleared all latency measurements.")

# --- Global Singleton Management ---
_global_tracker_instance: Optional[LatencyTracker] = None

def get_latency_tracker(**kwargs) -> LatencyTracker:
    """
    Get the configured global latency tracker instance.
    Initializes it on first call.
    """
    global _global_tracker_instance
    if _global_tracker_instance is None:
        _global_tracker_instance = LatencyTracker(**kwargs)
    return _global_tracker_instance
