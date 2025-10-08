"""
Advanced, stateful resource usage monitor for tracking cost, runtime, tokens,
and tool calls over a single turn or cumulatively across a task.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.resource")

# Global cache for storing cumulative resource usage
_cumulative_usage_cache: Dict[str, Dict[str, float]] = {}


def create_resource_monitor(spec: MonitorSpec) -> MonitorFn:
    """
    Create a comprehensive and stateful resource usage monitor.

    This monitor can track multiple resource types (cost, runtime, tokens, tool calls)
    and enforce budgets on a per-turn basis or cumulatively over a task or agent's lifetime.

    Args:
        spec: Monitor specification with parameters:
            - budgets (Dict[str, float]): A dictionary where keys are resource names
              (e.g., 'cost', 'runtime', 'tokens_total') and values are the
              budget limits. (Required)
            - scope (str): The scope for budget tracking. One of 'turn', 'task',
              or 'agent'. Defaults to 'turn'.
            - violation_name (str): Custom name for the violation.

    Returns:
        The configured monitor function.

    Raises:
        ValueError: If 'budgets' parameter is missing or empty.
    """
    # --- 1. Configuration and Validation ---
    budgets = spec.params.get("budgets")
    if not budgets or not isinstance(budgets, dict):
        raise ValueError("Resource monitor requires a non-empty 'budgets' dictionary in spec.params.")

    scope = spec.params.get("scope", "turn")
    if scope not in ["turn", "task", "agent"]:
        raise ValueError(f"Invalid scope '{scope}'. Must be 'turn', 'task', or 'agent'.")

    violation_name = spec.params.get("violation_name", f"{spec.name}_resource_exceeded")

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        from .factory import MonitorFactory
        if MonitorFactory._should_cooldown(spec, task):
            return

        # --- 2. Extract Resource Usage from the Current Turn ---
        current_usage = _extract_usage_from_result(result)

        # --- 3. Determine Values to Check Based on Scope ---
        if scope == "turn":
            usage_to_check = current_usage
            cache_key = None
        else:
            # Cumulative tracking
            cache_key = task if scope == "task" else agent.name
            if cache_key not in _cumulative_usage_cache:
                _cumulative_usage_cache[cache_key] = {key: 0.0 for key in budgets}

            # Update cumulative totals
            for resource, value in current_usage.items():
                if resource in _cumulative_usage_cache[cache_key]:
                    _cumulative_usage_cache[cache_key][resource] += value
            
            usage_to_check = _cumulative_usage_cache[cache_key]

        # --- 4. Check All Configured Budgets ---
        for resource, budget in budgets.items():
            if resource not in usage_to_check:
                continue  # Skip if resource data is not available

            measured_value = usage_to_check[resource]

            if measured_value > budget:
                _handle_violation(
                    spec=spec,
                    agent=agent,
                    task=task,
                    violation_name=violation_name,
                    resource=resource,
                    measured_value=measured_value,
                    budget=budget,
                    scope=scope,
                    turn_usage=current_usage
                )
                # Stop after the first violation to avoid spamming
                return

    return monitor


def _extract_usage_from_result(result: Dict[str, Any]) -> Dict[str, float]:
    """Safely extracts all available resource metrics from an agent's result dictionary."""
    if not isinstance(result, dict):
        return {}

    usage = {}
    
    # Extract runtime
    if "runtime" in result and isinstance(result["runtime"], (int, float)):
        usage["runtime"] = float(result["runtime"])

    # Extract cost and token info from cost_record
    cost_record = result.get("cost_record")
    if isinstance(cost_record, dict):
        if "total_cost" in cost_record: usage["cost"] = float(cost_record["total_cost"])
        if "tokens_input" in cost_record: usage["tokens_input"] = float(cost_record["tokens_input"])
        if "tokens_output" in cost_record: usage["tokens_output"] = float(cost_record["tokens_output"])
        if "tokens_input" in usage and "tokens_output" in usage:
            usage["tokens_total"] = usage["tokens_input"] + usage["tokens_output"]

    # Extract tool call count (assuming a standard key)
    if "tool_calls" in result and isinstance(result["tool_calls"], (int, float)):
        usage["tool_calls"] = float(result["tool_calls"])

    return usage


def _handle_violation(
    spec: MonitorSpec, agent: "AgentNet", task: str, violation_name: str,
    resource: str, measured_value: float, budget: float, scope: str,
    turn_usage: Dict[str, float]
) -> None:
    """Helper function to build and dispatch a resource violation."""
    from .factory import MonitorFactory

    scope_str = "Cumulative" if scope != "turn" else "Turn"
    rationale = (
        f"{scope_str} '{resource}' usage of {measured_value:.4f} exceeded the budget of {budget:.4f}."
    )

    meta = {
        "resource_violated": resource,
        "measured_value": measured_value,
        "budget_limit": budget,
        "budget_scope": scope,
        "current_turn_usage": turn_usage,
    }

    violation = MonitorFactory._build_violation(
        name=violation_name,
        vtype="resource_budget",
        severity=spec.severity,
        description=spec.description or f"Resource budget for '{resource}' was exceeded.",
        rationale=rationale,
        meta=meta,
    )

    detail = {"outcome": {"usage": turn_usage}, "violations": [violation]}
    MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
