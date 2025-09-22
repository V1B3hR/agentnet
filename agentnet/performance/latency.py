"""
Turn Latency Measurement for AgentNet

Provides detailed latency tracking for individual turns, multi-agent interactions,
and turn-to-turn performance analysis as specified in Phase 5.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class LatencyComponent(str, Enum):
    """Components of turn latency that can be measured separately."""
    TOTAL = "total"
    INFERENCE = "inference" 
    POLICY_CHECK = "policy_check"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_ACCESS = "memory_access"
    RESPONSE_PROCESSING = "response_processing"


@dataclass
class TurnLatencyMeasurement:
    """Detailed latency measurement for a single turn."""
    
    turn_id: str
    agent_id: str
    start_time: float
    end_time: float
    
    # Component latencies (in milliseconds)
    component_latencies: Dict[LatencyComponent, float] = field(default_factory=dict)
    
    # Metadata
    prompt_length: int = 0
    response_length: int = 0
    tokens_processed: int = 0
    policy_violations: int = 0
    tools_used: List[str] = field(default_factory=list)
    
    @property 
    def total_latency_ms(self) -> float:
        """Total turn latency in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def latency_breakdown(self) -> Dict[str, float]:
        """Breakdown of latency by component as percentages."""
        total = self.total_latency_ms
        if total == 0:
            return {}
        
        return {
            component.value: (latency / total) * 100
            for component, latency in self.component_latencies.items()
        }


class LatencyTracker:
    """
    Tracks and analyzes turn latency across agent interactions.
    
    Provides detailed timing measurements for different components
    of agent processing and turn execution.
    """
    
    def __init__(self):
        self._measurements: List[TurnLatencyMeasurement] = []
        self._active_measurements: Dict[str, Dict[str, Any]] = {}
    
    def start_turn_measurement(
        self, 
        turn_id: str, 
        agent_id: str,
        prompt_length: int = 0
    ) -> None:
        """Start measuring latency for a turn."""
        start_time = time.time()
        
        self._active_measurements[turn_id] = {
            'agent_id': agent_id,
            'start_time': start_time,
            'prompt_length': prompt_length,
            'component_starts': {},
            'component_latencies': {},
            'tools_used': [],
            'policy_violations': 0
        }
        
        logger.debug(f"Started turn latency measurement: {turn_id} for {agent_id}")
    
    def start_component_measurement(
        self, 
        turn_id: str, 
        component: LatencyComponent
    ) -> None:
        """Start measuring a specific component within a turn."""
        if turn_id not in self._active_measurements:
            logger.warning(f"No active measurement for turn {turn_id}")
            return
        
        self._active_measurements[turn_id]['component_starts'][component] = time.time()
    
    def end_component_measurement(
        self, 
        turn_id: str, 
        component: LatencyComponent
    ) -> float:
        """End measuring a component and return its latency in ms."""
        if turn_id not in self._active_measurements:
            logger.warning(f"No active measurement for turn {turn_id}")
            return 0.0
        
        measurement = self._active_measurements[turn_id]
        if component not in measurement['component_starts']:
            logger.warning(f"Component {component} was not started for turn {turn_id}")
            return 0.0
        
        start_time = measurement['component_starts'][component]
        latency_ms = (time.time() - start_time) * 1000
        measurement['component_latencies'][component] = latency_ms
        
        return latency_ms
    
    def record_tool_usage(self, turn_id: str, tool_name: str) -> None:
        """Record tool usage for latency analysis."""
        if turn_id in self._active_measurements:
            self._active_measurements[turn_id]['tools_used'].append(tool_name)
    
    def record_policy_violation(self, turn_id: str) -> None:
        """Record policy violation for latency analysis."""
        if turn_id in self._active_measurements:
            self._active_measurements[turn_id]['policy_violations'] += 1
    
    def end_turn_measurement(
        self, 
        turn_id: str,
        response_length: int = 0,
        tokens_processed: int = 0
    ) -> TurnLatencyMeasurement:
        """End turn measurement and create final measurement record."""
        if turn_id not in self._active_measurements:
            logger.warning(f"No active measurement for turn {turn_id}")
            # Return dummy measurement
            return TurnLatencyMeasurement(
                turn_id=turn_id,
                agent_id="unknown",
                start_time=time.time(),
                end_time=time.time()
            )
        
        measurement_data = self._active_measurements.pop(turn_id)
        end_time = time.time()
        
        # Create measurement record
        measurement = TurnLatencyMeasurement(
            turn_id=turn_id,
            agent_id=measurement_data['agent_id'],
            start_time=measurement_data['start_time'],
            end_time=end_time,
            component_latencies=measurement_data['component_latencies'],
            prompt_length=measurement_data['prompt_length'],
            response_length=response_length,
            tokens_processed=tokens_processed,
            policy_violations=measurement_data['policy_violations'],
            tools_used=measurement_data['tools_used']
        )
        
        self._measurements.append(measurement)
        
        logger.debug(
            f"Completed turn latency measurement: {turn_id} - "
            f"{measurement.total_latency_ms:.2f}ms"
        )
        
        return measurement
    
    def get_measurements(
        self, 
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TurnLatencyMeasurement]:
        """Get latency measurements, optionally filtered by agent."""
        measurements = self._measurements
        
        if agent_id:
            measurements = [m for m in measurements if m.agent_id == agent_id]
        
        if limit:
            measurements = measurements[-limit:]
        
        return measurements
    
    def get_latency_statistics(
        self, 
        agent_id: Optional[str] = None,
        component: Optional[LatencyComponent] = None
    ) -> Dict[str, float]:
        """Get statistical summary of latency measurements."""
        measurements = self.get_measurements(agent_id=agent_id)
        
        if not measurements:
            return {}
        
        if component:
            # Statistics for specific component
            latencies = [
                m.component_latencies.get(component, 0.0) 
                for m in measurements
            ]
            latencies = [l for l in latencies if l > 0]  # Filter out missing components
        else:
            # Statistics for total latency
            latencies = [m.total_latency_ms for m in measurements]
        
        if not latencies:
            return {}
        
        return {
            'count': len(latencies),
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'p95': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
            'p99': sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0,
        }
    
    def get_component_breakdown(
        self, 
        agent_id: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get average latency breakdown by component."""
        measurements = self.get_measurements(agent_id=agent_id)
        
        if not measurements:
            return {}
        
        component_totals = {}
        component_counts = {}
        
        for measurement in measurements:
            for component, latency in measurement.component_latencies.items():
                if component not in component_totals:
                    component_totals[component] = 0.0
                    component_counts[component] = 0
                
                component_totals[component] += latency
                component_counts[component] += 1
        
        # Calculate averages and percentages
        total_avg_latency = sum(
            component_totals[comp] / component_counts[comp] 
            for comp in component_totals
        )
        
        result = {}
        for component in component_totals:
            avg_latency = component_totals[component] / component_counts[component]
            percentage = (avg_latency / total_avg_latency * 100) if total_avg_latency > 0 else 0
            
            result[component.value] = {
                'avg_latency_ms': avg_latency,
                'percentage': percentage,
                'count': component_counts[component]
            }
        
        return result
    
    def identify_performance_issues(
        self,
        latency_threshold_ms: float = 1000.0,
        agent_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Identify potential performance issues from latency measurements."""
        measurements = self.get_measurements(agent_id=agent_id)
        issues = {
            'high_latency_turns': [],
            'policy_heavy_turns': [],
            'tool_heavy_turns': [],
            'memory_intensive_turns': []
        }
        
        for measurement in measurements:
            # High latency turns
            if measurement.total_latency_ms > latency_threshold_ms:
                issues['high_latency_turns'].append(
                    f"{measurement.turn_id}: {measurement.total_latency_ms:.2f}ms"
                )
            
            # Policy-heavy turns
            if measurement.policy_violations > 2:
                issues['policy_heavy_turns'].append(
                    f"{measurement.turn_id}: {measurement.policy_violations} violations"
                )
            
            # Tool-heavy turns  
            if len(measurement.tools_used) > 3:
                issues['tool_heavy_turns'].append(
                    f"{measurement.turn_id}: {len(measurement.tools_used)} tools"
                )
            
            # Memory-intensive turns (high memory access latency)
            memory_latency = measurement.component_latencies.get(
                LatencyComponent.MEMORY_ACCESS, 0.0
            )
            if memory_latency > latency_threshold_ms * 0.3:  # 30% of threshold
                issues['memory_intensive_turns'].append(
                    f"{measurement.turn_id}: {memory_latency:.2f}ms memory access"
                )
        
        return issues
    
    def clear_measurements(self) -> None:
        """Clear all stored measurements."""
        self._measurements.clear()
        self._active_measurements.clear()
        logger.info("Cleared all latency measurements")