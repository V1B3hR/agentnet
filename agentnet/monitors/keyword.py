"""Keyword-based monitor implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

# Severity is imported from base via MonitorSpec
from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.keyword")


def create_keyword_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a keyword-based monitor.

    Args:
        spec: Monitor specification with parameters:
            - keywords: List of keywords to match
            - violation_name: Name for violation (optional)
            - match_mode: "any" (default) or "all" matching mode

    Returns:
        Monitor function
    """
    keywords = [k.lower() for k in spec.params.get("keywords", [])]
    violation_name = spec.params.get("violation_name", f"{spec.name}_keyword")
    match_any = spec.params.get("match_mode", "any").lower() != "all"

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        if not keywords:
            return

        # Import here to avoid circular imports
        from .factory import MonitorFactory

        if MonitorFactory._should_cooldown(spec, task):
            return

        content = (
            str(result.get("content", ""))
            if isinstance(result, dict)
            else str(result)
        )
        lower = content.lower()
        present = [kw for kw in keywords if kw in lower]
        failed = (
            (len(present) > 0)
            if match_any
            else (len(present) == len(keywords))
        )

        if failed:
            rationale = (
                f"Matched keyword(s): {', '.join(sorted(set(present)))}"
            )
            violations = [
                MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="keyword",
                    severity=spec.severity,
                    description=spec.description or "Keyword guard triggered",
                    rationale=rationale,
                    meta={"matched": present},
                )
            ]
            detail = {
                "outcome": {"content": content},
                "violations": violations,
            }
            MonitorFactory._handle(
                spec, agent, task, passed=False, detail=detail
            )

    return monitor
