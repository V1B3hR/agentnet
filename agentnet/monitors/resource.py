"""Resource usage monitor implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.resource")


def create_resource_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a resource usage monitor.

    Args:
        spec: Monitor specification with parameters:
            - budget_key: Key for budget in agent style (default: "resource_budget")
            - tolerance: Tolerance percentage for budget overrun (default: 0.2)
            - violation_name: Name for violation (optional)

    Returns:
        Monitor function
    """
    budget_key = spec.params.get("budget_key", "resource_budget")
    tolerance = float(spec.params.get("tolerance", 0.2))
    violation_name = spec.params.get("violation_name", f"{spec.name}_resource")

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        # Import here to avoid circular imports
        from .factory import MonitorFactory

        if MonitorFactory._should_cooldown(spec, task):
            return

        runtime = float(result.get("runtime", 0.0)) if isinstance(result, dict) else 0.0
        budget = float(agent.style.get(budget_key, 0.0))

        if budget <= 0:
            return

        leakage_fraction = (runtime - budget) / budget
        violated = leakage_fraction > tolerance

        if violated:
            pct_over = leakage_fraction * 100.0
            rationale = (
                f"Runtime {runtime:.4f}s exceeded budget {budget:.4f}s by "
                f"{pct_over:.1f}% (tolerance {tolerance*100:.0f}%)"
            )
            violations = [
                MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="resource",
                    severity=spec.severity,
                    description=spec.description or "Resource usage exceeded threshold",
                    rationale=rationale,
                    meta={
                        "runtime": runtime,
                        "budget": budget,
                        "over_pct": pct_over,
                        "tolerance_pct": tolerance * 100.0,
                    },
                )
            ]
            detail = {
                "outcome": {"runtime": runtime},
                "violations": violations,
            }
            MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)

    return monitor
