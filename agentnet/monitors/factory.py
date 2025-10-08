"""
Extensible and dynamic monitor factory for creating and managing monitor instances.

This factory uses a registry pattern to allow for runtime registration of new
monitor types and provides advanced features like configurable cooldown scopes
and actionable violation handling.
"""

from __future- import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

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

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors")


class MonitorFactory:
    """
    Factory for creating monitor functions from specifications using a dynamic registry.
    """
    # --- Private, encapsulated state ---
    _last_trigger: Dict[str, float] = {}
    _monitor_registry: Dict[str, Callable[[MonitorSpec], MonitorFn]] = {
        "keyword": create_keyword_monitor,
        "regex": create_regex_monitor,
        "resource": create_resource_monitor,
        "custom": create_custom_monitor,
        "semantic_similarity": create_semantic_similarity_monitor,
        "llm_classifier": create_llm_classifier_monitor,
        "numerical_threshold": create_numerical_threshold_monitor,
    }

    @classmethod
    def register_monitor_type(cls, type_name: str, creation_func: Callable[[MonitorSpec], MonitorFn]):
        """

        Register a new custom monitor type at runtime.

        This allows for extending the factory with new monitor logic without
        modifying the core library code.

        Args:
            type_name: The unique name for the new monitor type (e.g., "pii_detector").
            creation_func: A function that takes a MonitorSpec and returns a MonitorFn.
        """
        type_name = type_name.lower()
        if type_name in cls._monitor_registry:
            logger.warning(f"Overwriting existing monitor registration for type: '{type_name}'")
        cls._monitor_registry[type_name] = creation_func
        logger.info(f"Successfully registered new monitor type: '{type_name}'")

    @classmethod
    def build(cls, spec: MonitorSpec) -> MonitorFn:
        """
        Build a monitor function from a specification using the registry.

        Args:
            spec: The monitor specification.

        Returns:
            The configured monitor function.

        Raises:
            ValueError: If the monitor type specified in the spec is not registered.
        """
        monitor_type = spec.type.lower()
        creation_func = cls._monitor_registry.get(monitor_type)

        if creation_func:
            return creation_func(spec)
        
        raise ValueError(
            f"Unknown monitor type: '{spec.type}'. "
            f"Available types are: {list(cls._monitor_registry.keys())}"
        )

    @classmethod
    def check_and_update_cooldown(cls, spec: MonitorSpec, agent: "AgentNet", task: str) -> bool:
        """
        Check if a monitor should be in a cooldown period and update its last trigger time.

        Args:
            spec: The monitor's specification, containing cooldown settings.
            agent: The agent instance, used for 'agent' scoped cooldowns.
            task: The task string, used for 'task' scoped cooldowns.

        Returns:
            True if the monitor is on cooldown, False otherwise.
        """
        if not spec.cooldown_seconds or spec.cooldown_seconds <= 0:
            return False

        scope = spec.cooldown_scope
        if scope == "task":
            key = f"{spec.name}:{task}"
        elif scope == "agent":
            key = f"{spec.name}:{agent.name}"
        else:  # "global"
            key = spec.name

        now = time.time()
        last_triggered = cls._last_trigger.get(key, 0.0)

        if (now - last_triggered) < float(spec.cooldown_seconds):
            return True  # On cooldown

        cls._last_trigger[key] = now
        return False

    @classmethod
    def _handle(
        cls,
        spec: MonitorSpec,
        agent: "AgentNet",
        task: str,
        passed: bool,
        detail: Dict[str, Any],
    ):
        """
        Handle the result of a monitor execution, taking configured actions.
        """
        if passed:
            return

        violations = detail.get("violations") or []
        if not violations:
            highest_severity = spec.severity
        else:
            # Determine the highest severity from all reported violations
            severities = [_parse_severity(v.get("severity", "minor")) for v in violations]
            highest_severity = max(severities, key=lambda s: Severity.level(s))

        action = spec.action
        msg = f"Monitor['{spec.name}'] violation on agent '{agent.name}': {spec.description or spec.type}"

        # Action 1: Always warn, never raise an exception.
        if action == "warn":
            logger.warning("[MONITOR:WARN] %s | Details: %s", msg, detail)
        
        # Action 2: Raise a CognitiveFault for severe violations.
        elif action == "raise" and highest_severity == Severity.SEVERE:
            raise CognitiveFault(
                message=msg,
                severity=highest_severity,
                violations=violations,
                context={"task": task, "monitor_name": spec.name, "agent_name": agent.name, "action_taken": "raise", **detail},
            )
        
        # Default behavior for non-severe violations when action is 'raise'
        else:
            logger.warning("[MONITOR:LOG] %s", msg)

        # Always record the violation in the agent's history
        agent.interaction_history.append(
            {
                "type": "monitor_violation",
                "monitor_name": spec.name,
                "task": task,
                "detail": detail,
                "severity": highest_severity.value,
                "action_taken": action,
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
        """
        Build a standardized violation dictionary with a timestamp.
        """
        return {
            "name": name,
            "type": vtype,
            "severity": severity.value,
            "description": description,
            "rationale": rationale,
            "timestamp": time.time(),
            "meta": meta or {},
        }


# Re-export register function for backward compatibility and ease of use
__all__ = ["MonitorFactory", "register_custom_monitor_func"]
