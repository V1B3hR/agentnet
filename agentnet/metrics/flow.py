"""
Flow metrics implementation for AgentNet reasoning trees.

Implements electrical circuit-inspired metrics:
- Current (I): tokens/sec using cost_record.tokens_output and runtime
- Voltage (V): from metadata.voltage (0-10), else mapped from technique or AutoConfig difficulty
- Resistance (R): α*policy_hits + β*avg_tool_latency_s + γ*disagreement_score
- Power (P): V × I

Design follows the specification from the problem statement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.enums import ProblemTechnique
from ..core.autoconfig import TaskDifficulty

logger = logging.getLogger("agentnet.metrics.flow")


@dataclass
class FlowMetrics:
    """Flow metrics calculated from reasoning tree outputs."""

    current: float  # I: tokens/sec
    voltage: float  # V: 0-10 scale
    resistance: float  # R: composite resistance score
    power: float  # P: V × I

    # Metadata for debugging and analysis
    runtime_seconds: float
    tokens_output: int
    metadata_voltage: Optional[float]
    technique_voltage: Optional[float]
    difficulty_voltage: Optional[float]
    policy_hits: int
    avg_tool_latency_s: float
    disagreement_score: float


def calculate_flow_metrics(
    reasoning_tree: Dict[str, Any],
    runtime_seconds: Optional[float] = None,
    technique: Optional[ProblemTechnique] = None,
    difficulty: Optional[TaskDifficulty] = None,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> FlowMetrics:
    """
    Calculate flow metrics from a reasoning tree output.

    Args:
        reasoning_tree: The reasoning tree dictionary from AgentNet
        runtime_seconds: Optional runtime override (uses tree runtime if not provided)
        technique: Optional technique for voltage calculation
        difficulty: Optional difficulty for voltage calculation
        alpha: Weight for policy_hits in resistance calculation
        beta: Weight for avg_tool_latency_s in resistance calculation
        gamma: Weight for disagreement_score in resistance calculation

    Returns:
        FlowMetrics instance with calculated values
    """
    # Extract runtime
    if runtime_seconds is None:
        runtime_seconds = reasoning_tree.get("runtime", 0.0)

    if runtime_seconds <= 0:
        runtime_seconds = 0.001  # Avoid division by zero

    # Extract cost and token information
    cost_record = reasoning_tree.get("cost_record", {})
    tokens_output = cost_record.get("tokens_output", 0)

    # Calculate Current (I): tokens/sec
    current = tokens_output / runtime_seconds

    # Calculate Voltage (V): from metadata, technique, or difficulty
    metadata = reasoning_tree.get("metadata", {})
    metadata_voltage = metadata.get("voltage")
    voltage = _calculate_voltage(metadata_voltage, technique, difficulty)

    # Extract resistance components (MVP: stub values since signals aren't fully emitted yet)
    policy_hits = _extract_policy_hits(reasoning_tree)
    avg_tool_latency_s = _extract_avg_tool_latency(reasoning_tree)
    disagreement_score = _extract_disagreement_score(reasoning_tree)

    # Calculate Resistance (R)
    resistance = (
        alpha * policy_hits + beta * avg_tool_latency_s + gamma * disagreement_score
    )

    # Ensure minimum resistance to avoid division issues
    resistance = max(resistance, 0.1)

    # Calculate Power (P): V × I
    power = voltage * current

    return FlowMetrics(
        current=current,
        voltage=voltage,
        resistance=resistance,
        power=power,
        runtime_seconds=runtime_seconds,
        tokens_output=tokens_output,
        metadata_voltage=metadata_voltage,
        technique_voltage=_technique_to_voltage(technique) if technique else None,
        difficulty_voltage=_difficulty_to_voltage(difficulty) if difficulty else None,
        policy_hits=policy_hits,
        avg_tool_latency_s=avg_tool_latency_s,
        disagreement_score=disagreement_score,
    )


def _calculate_voltage(
    metadata_voltage: Optional[float],
    technique: Optional[ProblemTechnique],
    difficulty: Optional[TaskDifficulty],
) -> float:
    """Calculate voltage from available sources."""
    # Priority 1: Use metadata.voltage if available
    if metadata_voltage is not None:
        return max(0.0, min(10.0, metadata_voltage))  # Clamp to 0-10 range

    # Priority 2: Map from technique
    if technique is not None:
        technique_voltage = _technique_to_voltage(technique)
        if technique_voltage is not None:
            return technique_voltage

    # Priority 3: Map from AutoConfig difficulty
    if difficulty is not None:
        difficulty_voltage = _difficulty_to_voltage(difficulty)
        if difficulty_voltage is not None:
            return difficulty_voltage

    # Default voltage
    return 5.0


def _technique_to_voltage(technique: ProblemTechnique) -> float:
    """Map problem technique to voltage level."""
    technique_voltage_map = {
        ProblemTechnique.SIMPLE: 3.0,
        ProblemTechnique.COMPLEX: 8.0,
        ProblemTechnique.TROUBLESHOOTING: 7.0,
        ProblemTechnique.GAP_FROM_STANDARD: 6.0,
        ProblemTechnique.TARGET_STATE: 7.5,
        ProblemTechnique.OPEN_ENDED: 5.5,
    }
    return technique_voltage_map.get(technique, 5.0)


def _difficulty_to_voltage(difficulty: TaskDifficulty) -> float:
    """Map AutoConfig difficulty to voltage level."""
    difficulty_voltage_map = {
        TaskDifficulty.SIMPLE: 3.0,
        TaskDifficulty.MEDIUM: 5.0,
        TaskDifficulty.HARD: 8.0,
    }
    return difficulty_voltage_map.get(difficulty, 5.0)


def _extract_policy_hits(reasoning_tree: Dict[str, Any]) -> int:
    """Extract policy violations count from reasoning tree (MVP: stub)."""
    # MVP implementation: check for violations in various places
    violations = reasoning_tree.get("violations", [])
    monitor_trace = reasoning_tree.get("monitor_trace", [])

    policy_hits = len(violations)

    # Check monitor trace for policy violations
    for trace_entry in monitor_trace:
        if isinstance(trace_entry, dict):
            trace_violations = trace_entry.get("violations", [])
            policy_hits += len(trace_violations)

    return policy_hits


def _extract_avg_tool_latency(reasoning_tree: Dict[str, Any]) -> float:
    """Extract average tool latency from reasoning tree (MVP: stub)."""
    # MVP implementation: look for tool execution data
    tool_calls = reasoning_tree.get("tool_calls", [])

    if not tool_calls:
        return 0.0

    total_latency = 0.0
    count = 0

    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            latency = tool_call.get("latency_s") or tool_call.get("duration", 0.0)
            if latency > 0:
                total_latency += latency
                count += 1

    return total_latency / count if count > 0 else 0.0


def _extract_disagreement_score(reasoning_tree: Dict[str, Any]) -> float:
    """Extract disagreement score from reasoning tree (MVP: stub)."""
    # MVP implementation: look for disagreement indicators
    # This is a placeholder until full disagreement tracking is implemented

    # Check for confidence variance as a proxy for disagreement
    nodes = reasoning_tree.get("nodes", [])
    confidences = []

    for node in nodes:
        if isinstance(node, dict):
            confidence = node.get("confidence")
            if confidence is not None:
                confidences.append(confidence)

    if len(confidences) < 2:
        return 0.0

    # Calculate coefficient of variation as disagreement proxy
    mean_confidence = sum(confidences) / len(confidences)
    variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
    std_dev = variance**0.5

    # Normalize to 0-1 scale (coefficient of variation)
    disagreement_score = std_dev / mean_confidence if mean_confidence > 0 else 0.0

    return min(disagreement_score, 1.0)  # Cap at 1.0
