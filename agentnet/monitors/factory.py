"""Monitor factory for creating monitor instances."""

from __future__ import annotations
import logging
import re
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import MonitorFn, MonitorSpec
from ..core.types import Severity, _parse_severity, CognitiveFault

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors")

# Global registry for custom monitor functions
CUSTOM_FUNCS: Dict[str, Any] = {}


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
            return MonitorFactory._keyword_monitor(spec)
        if t == "regex":
            return MonitorFactory._regex_monitor(spec)
        if t == "resource":
            return MonitorFactory._resource_monitor(spec)
        if t == "custom":
            return MonitorFactory._custom_monitor(spec)
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
    def _handle(spec: MonitorSpec, agent: "AgentNet", task: str,
                passed: bool, detail: Dict[str, Any]):
        """Handle monitor result."""
        if passed:
            return
        
        violations = detail.get("violations") or []
        if not violations:
            derived = spec.severity
        else:
            # Find highest severity from violations
            severities = [_parse_severity(v.get("severity", "minor")) for v in violations]
            derived = max(severities, key=lambda s: ["minor", "major", "severe"].index(s.value))
        
        is_severe = derived == Severity.SEVERE
        msg = f"Monitor[{spec.name}] violation: {spec.description or spec.type}"
        
        if is_severe:
            raise CognitiveFault(
                message=msg,
                severity=derived,
                violations=violations,
                context={"task": task, "monitor": spec.name, **detail}
            )
        else:
            logger.warning("[MONITOR:MINOR] %s", msg)
            agent.interaction_history.append({
                "type": "monitor_minor",
                "monitor": spec.name,
                "task": task,
                "detail": detail,
                "severity": derived.value,
            })

    @staticmethod
    def _build_violation(name: str, vtype: str, severity: Severity, description: str,
                         rationale: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a violation dictionary."""
        return {
            "name": name,
            "type": vtype,
            "severity": severity.value,
            "description": description,
            "rationale": rationale,
            "meta": meta or {},
        }

    @staticmethod
    def _keyword_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create a keyword-based monitor."""
        keywords = [k.lower() for k in spec.params.get("keywords", [])]
        violation_name = spec.params.get("violation_name", f"{spec.name}_keyword")
        match_any = spec.params.get("match_mode", "any").lower() != "all"
        
        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if not keywords:
                return
            if MonitorFactory._should_cooldown(spec, task):
                return
            
            content = str(result.get("content", "")) if isinstance(result, dict) else str(result)
            lower = content.lower()
            present = [kw for kw in keywords if kw in lower]
            failed = (len(present) > 0) if match_any else (len(present) == len(keywords))
            
            if failed:
                rationale = f"Matched keyword(s): {', '.join(sorted(set(present)))}"
                violations = [MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="keyword",
                    severity=spec.severity,
                    description=spec.description or "Keyword guard triggered",
                    rationale=rationale,
                    meta={"matched": present}
                )]
                detail = {"outcome": {"content": content}, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        return monitor

    @staticmethod
    def _regex_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create a regex-based monitor."""
        pattern = spec.params.get("pattern")
        violation_name = spec.params.get("violation_name", f"{spec.name}_regex")
        if not pattern:
            raise ValueError("regex monitor requires params.pattern")
        
        flags = re.IGNORECASE | re.MULTILINE
        rx = re.compile(pattern, flags)
        
        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return
            
            content = str(result.get("content", "")) if isinstance(result, dict) else str(result)
            matches = list(rx.finditer(content))
            
            if matches:
                first = matches[0].group(0)
                rationale = f"Pattern matched {len(matches)} time(s); first='{first}'"
                violations = [MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="regex",
                    severity=spec.severity,
                    description=spec.description or "Regex guard triggered",
                    rationale=rationale,
                    meta={"pattern": pattern, "match_count": len(matches), "first": first}
                )]
                detail = {"outcome": {"content": content}, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        return monitor

    @staticmethod
    def _resource_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create a resource usage monitor."""
        budget_key = spec.params.get("budget_key", "resource_budget")
        tolerance = float(spec.params.get("tolerance", 0.2))
        violation_name = spec.params.get("violation_name", f"{spec.name}_resource")
        
        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return
            
            runtime = float(result.get("runtime", 0.0)) if isinstance(result, dict) else 0.0
            budget = float(agent.style.get(budget_key, 0.0))
            
            if budget <= 0:
                return
            
            leakage_fraction = (runtime - budget) / budget
            violated = leakage_fraction > tolerance
            
            if violated:
                pct_over = (leakage_fraction * 100.0)
                rationale = (
                    f"Runtime {runtime:.4f}s exceeded budget {budget:.4f}s by "
                    f"{pct_over:.1f}% (tolerance {tolerance*100:.0f}%)"
                )
                violations = [MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="resource",
                    severity=spec.severity,
                    description=spec.description or "Resource usage exceeded threshold",
                    rationale=rationale,
                    meta={
                        "runtime": runtime,
                        "budget": budget,
                        "over_pct": pct_over,
                        "tolerance_pct": tolerance * 100.0
                    }
                )]
                detail = {
                    "outcome": {"runtime": runtime},
                    "violations": violations,
                }
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        return monitor

    @staticmethod
    def _custom_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create a custom function-based monitor."""
        func_name = spec.params.get("func")
        violation_name = spec.params.get("violation_name", f"{spec.name}_custom")
        func = CUSTOM_FUNCS.get(func_name)
        
        if func is None:
            raise ValueError(f"Unknown custom func: {func_name}")
        
        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return
            
            outcome = result if isinstance(result, dict) else {"content": str(result)}
            ret = func(outcome)
            
            if isinstance(ret, tuple):
                passed, rationale = ret
            else:
                passed, rationale = bool(ret), None
            
            if not passed:
                violations = [MonitorFactory._build_violation(
                    name=violation_name,
                    vtype="custom",
                    severity=spec.severity,
                    description=spec.description or f"Custom guard '{func_name}' failed",
                    rationale=rationale or "Custom function returned failure",
                    meta={"func": func_name}
                )]
                detail = {"outcome": outcome, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        return monitor


def register_custom_monitor_func(name: str, func: Any) -> None:
    """Register a custom monitor function."""
    CUSTOM_FUNCS[name] = func
    logger.info(f"Registered custom monitor function: {name}")