"""Regex-based monitor implementation."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.regex")


def create_regex_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a regex-based monitor.

    Args:
        spec: Monitor specification with parameters:
            - pattern: Regex pattern to match (required)
            - violation_name: Name for violation (optional)

    Returns:
        Monitor function

    Raises:
        ValueError: If pattern is not provided
    """
    pattern = spec.params.get("pattern")
    violation_name = spec.params.get("violation_name", f"{spec.name}_regex")
    if not pattern:
        raise ValueError("regex monitor requires params.pattern")

    flags = re.IGNORECASE | re.MULTILINE
    rx = re.compile(pattern, flags)

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        # Import here to avoid circular imports
        from .factory import MonitorFactory

        if MonitorFactory._should_cooldown(spec, task):
            return

        content = (
            str(result.get("content", "")) if isinstance(result, dict) else str(result)
        )
        matches = list(rx.finditer(content))

        if matches:
            first = matches[0].group(0)
            rationale = f"Pattern matched {len(matches)} time(s); first='{first}'"
            violations = [
                MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="regex",
                    severity=spec.severity,
                    description=spec.description or "Regex guard triggered",
                    rationale=rationale,
                    meta={
                        "pattern": pattern,
                        "match_count": len(matches),
                        "first": first,
                    },
                )
            ]
            detail = {
                "outcome": {"content": content},
                "violations": violations,
            }
            MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)

    return monitor
