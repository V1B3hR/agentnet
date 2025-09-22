"""Monitor factory for creating monitor instances."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..core.types import CognitiveFault, Severity, _parse_severity
from .base import MonitorFn, MonitorSpec
from .custom import (
    create_custom_monitor,
    create_llm_classifier_monitor,
    create_numerical_threshold_monitor,
    register_custom_monitor_func,
)
from .keyword import create_keyword_monitor
from .regex import create_regex_monitor
from .resource import create_resource_monitor
from .semantic import create_semantic_similarity_monitor
from .ethics import EthicsMonitor

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors")


class MonitorFactory:
    """Factory for creating monitor functions from specifications."""

    _last_trigger: Dict[str, float] = {}

    @staticmethod
    def build(spec: MonitorSpec) -> MonitorFn:
        """Build a monitor function from a specification.

        Args:
            spec: Monitor specification

        Returns:
            Monitor function

        Raises:
            ValueError: If monitor type is unknown
        """
        t = spec.type.lower()
        if t == "keyword":
            return create_keyword_monitor(spec)
        if t == "regex":
            return create_regex_monitor(spec)
        if t == "resource":
            return create_resource_monitor(spec)
        if t == "custom":
            return create_custom_monitor(spec)
        if t == "semantic_similarity":
            return create_semantic_similarity_monitor(spec)
        if t == "llm_classifier":
            return create_llm_classifier_monitor(spec)
        if t == "numerical_threshold":
            return create_numerical_threshold_monitor(spec)
        if t == "ethics":
            return create_ethics_monitor(spec)
        raise ValueError(f"Unknown monitor type: {spec.type}")

    @staticmethod
    def _should_cooldown(spec: MonitorSpec, task: str) -> bool:
        """Check if monitor should be in cooldown period."""
        if not spec.cooldown_seconds:
            return False
        key = f"{spec.name}:{task}"
        now = time.time()
        last = MonitorFactory._last_trigger.get(key, 0.0)
        if now - last < float(spec.cooldown_seconds):
            return True
        MonitorFactory._last_trigger[key] = now
        return False

    @staticmethod
    def _handle(
        spec: MonitorSpec,
        agent: "AgentNet",
        task: str,
        passed: bool,
        detail: Dict[str, Any],
    ):
        """Handle monitor result."""
        if passed:
            return

        violations = detail.get("violations") or []
        if not violations:
            derived = spec.severity
        else:
            # Find highest severity from violations
            severities = [
                _parse_severity(v.get("severity", "minor")) for v in violations
            ]
            derived = max(
                severities, key=lambda s: ["minor", "major", "severe"].index(s.value)
            )

        is_severe = derived == Severity.SEVERE
        msg = f"Monitor[{spec.name}] violation: {spec.description or spec.type}"

        if is_severe:
            raise CognitiveFault(
                message=msg,
                severity=derived,
                violations=violations,
                context={"task": task, "monitor": spec.name, **detail},
            )
        else:
            logger.warning("[MONITOR:MINOR] %s", msg)
            agent.interaction_history.append(
                {
                    "type": "monitor_minor",
                    "monitor": spec.name,
                    "task": task,
                    "detail": detail,
                    "severity": derived.value,
                }
            )

    @staticmethod
    def _build_violation(
        name: str,
        vtype: str,
        severity: Severity,
        description: str,
        rationale: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a violation dictionary."""
        return {
            "name": name,
            "type": vtype,
            "severity": severity.value,
            "description": description,
            "rationale": rationale,
            "meta": meta or {},
        }


def create_ethics_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create an ethics monitor function from specification."""
    
    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        """Monitor function that uses EthicsMonitor."""
        if MonitorFactory._should_cooldown(spec, task):
            return
            
        # Create EthicsMonitor instance with config from spec params
        ethics_monitor = EthicsMonitor(
            name=spec.name,
            config=spec.params
        )
        
        # Evaluate the result
        outcome = result if isinstance(result, dict) else {"content": str(result)}
        passed, message, eval_time = ethics_monitor.evaluate(outcome)
        
        if not passed:
            # Handle violation using the factory's standard approach
            violations = [{
                "name": spec.name,
                "type": "ethics",
                "severity": spec.severity.value,
                "description": spec.description or "Ethics violation detected",
                "rationale": message or "Ethics rules violated",
                "meta": {"evaluation_time": eval_time}
            }]
            
            detail = {"outcome": outcome, "violations": violations}
            MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        else:
            logger.debug(f"Ethics monitor '{spec.name}' passed for task '{task}'")
    
    return monitor


# Re-export register function for backward compatibility
__all__ = ["MonitorFactory", "register_custom_monitor_func", "create_ethics_monitor"]
