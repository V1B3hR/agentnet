from __future__ import annotations

# P0 REFACTORED IMPORTS - Core classes now use modular structure
try:
    from agentnet import AgentNet as AgentNetCore, ExampleEngine as ExampleEngineCore, Severity as SeverityCore
    _use_refactored = True
except ImportError:
    _use_refactored = False

import argparse
import asyncio
import json
import logging
import re
import time
import uuid
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Awaitable,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # YAML optional; JSON-only if PyYAML missing

# --------------------------------------------------------------------------------------
# Logging (renamed from 'duetmind' to 'agentnet')
# --------------------------------------------------------------------------------------
logger = logging.getLogger("agentnet")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --------------------------------------------------------------------------------------
# Constraint / Policy Engine
# --------------------------------------------------------------------------------------

class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    SEVERE = "severe"


def _parse_severity(value: str | Severity | None) -> Severity:
    if isinstance(value, Severity):
        return value
    if not value:
        return Severity.MINOR
    v = str(value).lower()
    if v == "severe":
        return Severity.SEVERE
    if v == "major":
        return Severity.MAJOR
    return Severity.MINOR


RuleCheckResult = Union[bool, Tuple[bool, Optional[str]]]
RuleCheckFn = Callable[[Dict[str, Any]], RuleCheckResult]


@dataclass
class ConstraintRule:
    name: str
    check: RuleCheckFn
    severity: Severity = Severity.MINOR
    description: str = ""

    def run(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Exception]]:
        try:
            result = self.check(outcome)
            if isinstance(result, tuple):
                passed, rationale = result
                return bool(passed), rationale, None
            return bool(result), None, None
        except Exception as exc:
            return False, f"Rule execution error: {exc}", exc

    def evaluate(self, outcome: Dict[str, Any]) -> bool:
        passed, _, _ = self.run(outcome)
        return passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "description": self.description,
        }


class PolicyEngine:
    def __init__(self, rules: Optional[List[ConstraintRule]] = None):
        self.rules = rules or []

    def register_rule(self, rule: ConstraintRule) -> None:
        self.rules.append(rule)

    def evaluate(
        self,
        outcome: Dict[str, Any],
        rich: bool = False
    ) -> Union[List[ConstraintRule], List[Dict[str, Any]]]:
        violations_rules: List[ConstraintRule] = []
        rich_violations: List[Dict[str, Any]] = []
        for rule in self.rules:
            passed, rationale, error = rule.run(outcome)
            if not passed:
                violations_rules.append(rule)
                if rich:
                    rich_violations.append({
                        "name": rule.name,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "rationale": rationale,
                        "error": str(error) if error else None,
                    })
        return rich_violations if rich else violations_rules


# --------------------------------------------------------------------------------------
# CognitiveFault
# --------------------------------------------------------------------------------------

class CognitiveFault(Exception):
    def __init__(
        self,
        message: str,
        intent: Dict[str, Any],
        outcome: Dict[str, Any],
        leakage: Dict[str, Any],
        severity: Severity = Severity.MINOR,
        violations: Optional[List[Dict[str, Any]]] = None,
        cause: Optional[BaseException] = None,
        fault_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ):
        super().__init__(message)
        self.intent = intent
        self.outcome = outcome
        self.leakage = leakage
        self.severity = severity
        self.violations = violations or []
        self.fault_id = fault_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.__cause__ = cause

    @property
    def tier(self) -> str:
        return self.severity.value

    def to_dict(self, redact: bool = False) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "timestamp": self.timestamp,
            "message": str(self),
            "severity": self.severity.value,
            "intent": self.intent,
            "outcome": self._maybe_redact_dict(self.outcome, redact),
            "leakage": self.leakage,
            "violations": self.violations,
        }

    def _maybe_redact_dict(self, data: Dict[str, Any], redact: bool) -> Dict[str, Any]:
        if not redact:
            return data
        redacted: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 400:
                redacted[k] = v[:200] + " ... [REDACTED] ..."
            else:
                redacted[k] = v
        return redacted

    def __str__(self) -> str:
        base = super().__str__()
        if self.violations:
            vnames = ", ".join(v.get("name", "?") for v in self.violations)
            return f"{base} | severity={self.severity.value} | violations: {vnames} | id={self.fault_id}"
        return f"{base} | severity={self.severity.value} | id={self.fault_id}"

    @staticmethod
    def _derive_highest_severity(violations: List[Dict[str, Any]]) -> Severity:
        if not violations:
            return Severity.MINOR
        order = {Severity.MINOR: 1, Severity.MAJOR: 2, Severity.SEVERE: 3}
        best = Severity.MINOR
        for v in violations:
            sv = _parse_severity(v.get("severity"))
            if order[sv] > order[best]:
                best = sv
        return best

    @classmethod
    def from_policy_violations(
        cls,
        message: str,
        task: str,
        outcome: Dict[str, Any],
        monitor_spec: Dict[str, Any],
        violations: List[Dict[str, Any]],
    ) -> "CognitiveFault":
        severity = cls._derive_highest_severity(violations)
        leakage = {"type": monitor_spec.get("type"), "name": monitor_spec.get("name")}
        intent = {"task": task}
        return cls(
            message=message,
            intent=intent,
            outcome=outcome,
            leakage=leakage,
            severity=severity,
            violations=violations,
        )


# --------------------------------------------------------------------------------------
# Utility rule generators / custom funcs
# --------------------------------------------------------------------------------------

def keyword_rule(
    keywords: List[str],
    match_mode: str = "any",
    whole_word: bool = False,
    case_insensitive: bool = True,
) -> RuleCheckFn:
    if not keywords:
        return lambda outcome: True
    if case_insensitive:
        kws_for_match = [k.lower() for k in keywords]
    else:
        kws_for_match = keywords
    if whole_word:
        flags = re.IGNORECASE if case_insensitive else 0
        compiled = [(kw, re.compile(rf"\b{re.escape(kw)}\b", flags)) for kw in kws_for_match]
        def check(outcome: Dict[str, Any]) -> RuleCheckResult:
            content = str(outcome.get("content", ""))
            haystack = content if not case_insensitive else content.lower()
            matched: List[str] = []
            for orig_kw, rx in compiled:
                if rx.search(haystack):
                    matched.append(orig_kw)
            failed = (len(matched) > 0) if match_mode != "all" else all(k in matched for k in kws_for_match)
            if failed:
                return False, f"Blocked keyword(s) detected: {', '.join(sorted(set(matched)))}"
            return True
        return check
    def check(outcome: Dict[str, Any]) -> RuleCheckResult:
        content = str(outcome.get("content", ""))
        haystack = content if not case_insensitive else content.lower()
        present = [kw for kw in kws_for_match if kw in haystack]
        failed = (len(present) > 0) if match_mode != "all" else len(present) == len(kws_for_match)
        if failed:
            return False, f"Blocked keyword substring(s): {', '.join(sorted(set(present)))}"
        return True
    return check


def _parse_pattern_flags(flag_string: Optional[str]) -> int:
    if not flag_string:
        return 0
    mapping = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "ASCII": re.ASCII,
        "VERBOSE": re.VERBOSE,
    }
    total = 0
    for part in (p.strip().upper() for p in flag_string.split("|")):
        total |= mapping.get(part, 0)
    return total


def regex_rule(pattern: str, flags: int = re.IGNORECASE) -> RuleCheckFn:
    rx = re.compile(pattern, flags)
    def check(outcome: Dict[str, Any]) -> RuleCheckResult:
        content = str(outcome.get("content", ""))
        matches = list(rx.finditer(content))
        if matches:
            first = matches[0].group(0)
            if len(matches) == 1:
                return False, f"Regex pattern matched: '{first}'"
            return False, f"Regex pattern matched {len(matches)} times; first: '{first}'"
        return True
    return check


def applied_ethics_check(outcome: Dict[str, Any]) -> RuleCheckResult:
    content = str(outcome.get("content", "")).lower()
    moral_keywords = [
        "right", "wrong", "justice", "fair", "unfair", "harm", "benefit",
        "responsibility", "duty", "obligation", "virtue", "vice", "good", "bad", "evil",
    ]
    controversy_keywords = [
        "controversy", "debate", "dispute", "conflict", "argument",
        "polarizing", "divisive", "hotly debated", "scandal",
    ]
    moral_hits = {kw for kw in moral_keywords if kw in content}
    controversy_hits = {kw for kw in controversy_keywords if kw in content}
    if moral_hits and controversy_hits:
        return False, ("Applied ethics review triggered: moral terms ("
                       + ", ".join(sorted(moral_hits)) + ") with controversy terms ("
                       + ", ".join(sorted(controversy_hits)) + ")")
    return True


CUSTOM_FUNCS: Dict[str, RuleCheckFn] = {
    "applied_ethics_check": applied_ethics_check,
}


def _load_data_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not installed, cannot read YAML. Use JSON or install pyyaml.")
        return yaml.safe_load(path.read_text())
    if suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_rules_from_file(path: str | Path) -> List[ConstraintRule]:
    data = _load_data_file(Path(path))
    rules_cfg = data.get("rules", [])
    if not isinstance(rules_cfg, list):
        raise ValueError("Config 'rules' must be a list")
    rules: List[ConstraintRule] = []
    for r in rules_cfg:
        if not isinstance(r, dict):
            raise ValueError(f"Rule entry must be an object: {r}")
        try:
            name = r["name"]
            severity = _parse_severity(r.get("severity", "minor"))
            description = r.get("description", "")
            rtype = r["type"]
        except KeyError as ke:
            raise ValueError(f"Missing required rule field: {ke} in {r}") from ke
        if rtype == "keyword":
            keywords = r.get("keywords")
            if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
                raise ValueError(f"Rule '{name}': 'keywords' must be a list[str]")
            check = keyword_rule(
                keywords=keywords,
                match_mode=r.get("match_mode", "any"),
                whole_word=bool(r.get("whole_word", False)),
                case_insensitive=bool(r.get("case_insensitive", True)),
            )
        elif rtype == "regex":
            pattern = r.get("pattern")
            if not pattern or not isinstance(pattern, str):
                raise ValueError(f"Rule '{name}': 'pattern' must be a non-empty string")
            flags = re.IGNORECASE
            extra_flags = _parse_pattern_flags(r.get("pattern_flags"))
            flags |= extra_flags
            check = regex_rule(pattern, flags)
        elif rtype == "custom":
            func_name = r.get("func")
            if not func_name or func_name not in CUSTOM_FUNCS:
                raise ValueError(f"Rule '{name}': unknown custom func '{func_name}'")
            check = CUSTOM_FUNCS[func_name]
        else:
            raise ValueError(f"Rule '{name}': Unknown rule type: {rtype}")
        rules.append(ConstraintRule(name, check, severity, description))
    return rules


# --------------------------------------------------------------------------------------
# Monitor system
# --------------------------------------------------------------------------------------

# Updated type hint forward reference from "DuetMindAgent" to "AgentNet"
MonitorFn = Callable[["AgentNet", str, Dict[str, Any]], None]


@dataclass
class MonitorSpec:
    name: str
    type: str
    severity: str = "minor"
    description: str = ""
    params: Optional[Dict[str, Any]] = None


class MonitorFactory:
    _last_trigger: Dict[str, float] = {}

    @staticmethod
    def build(spec: MonitorSpec) -> MonitorFn:
        t = spec.type.lower()
        if t == "rcd_policy":
            return MonitorFactory._rcd_policy_monitor(spec)
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
        if not spec.params:
            return False
        cooldown = spec.params.get("cooldown_seconds")
        if not cooldown:
            return False
        key = f"{spec.name}:{task}"
        now = time.time()
        last = MonitorFactory._last_trigger.get(key, 0.0)
        if now - last < float(cooldown):
            return True
        MonitorFactory._last_trigger[key] = now
        return False

    @staticmethod
    def _handle(spec: MonitorSpec, agent: "AgentNet", task: str,
                passed: bool, detail: Dict[str, Any]):
        if passed:
            return
        params = spec.params or {}
        severity_strategy = params.get("severity_strategy", "max_violation")
        violations = detail.get("violations") or []
        if severity_strategy == "spec" or not violations:
            derived = _parse_severity(spec.severity)
        else:
            derived = CognitiveFault._derive_highest_severity(violations)
        is_severe = derived == Severity.SEVERE
        msg = f"Monitor[{spec.name}] violation: {spec.description or spec.type}"
        if is_severe:
            raise CognitiveFault(
                message=msg,
                intent={"task": task, **(detail.get("intent", {}))},
                outcome=detail.get("outcome", {}),
                leakage={"type": spec.type, "name": spec.name, **detail.get("leakage", {})},
                severity=derived,
                violations=violations
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
    def _build_violation(name: str, vtype: str, severity: str, description: str,
                         rationale: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "name": name,
            "type": vtype,
            "severity": severity,
            "description": description,
            "rationale": rationale,
            "meta": meta or {},
        }

    @staticmethod
    def _rcd_policy_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        rules_file = params.get("rules_file")
        if not rules_file:
            raise ValueError("rcd_policy monitor requires params.rules_file")
        rules = load_rules_from_file(rules_file)
        pe = PolicyEngine(rules)
        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return
            outcome = result if isinstance(result, dict) else {"content": str(result)}
            rule_violations = pe.evaluate(outcome, rich=True)
            if rule_violations:
                violations = []
                for rv in rule_violations:
                    violations.append(MonitorFactory._build_violation(
                        name=rv["name"],
                        vtype="policy_rule",
                        severity=rv.get("severity", spec.severity),
                        description=rv.get("description") or spec.description,
                        rationale=rv.get("rationale") or "Rule failed",
                        meta={"error": rv.get("error")}
                    ))
                detail = {"outcome": outcome, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        return monitor

    @staticmethod
    def _keyword_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        keywords = [k.lower() for k in params.get("keywords", [])]
        violation_name = params.get("violation_name", f"{spec.name}_keyword")
        match_any = params.get("match_mode", "any").lower() != "all"
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
        params = spec.params or {}
        pattern = params.get("pattern")
        violation_name = params.get("violation_name", f"{spec.name}_regex")
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
        params = spec.params or {}
        budget_key = params.get("budget_key", "resource_budget")
        tolerance = float(params.get("tolerance", 0.2))
        violation_name = params.get("violation_name", f"{spec.name}_resource")
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
                    "leakage": {"budget": budget, "leakage": leakage_fraction},
                    "intent": {budget_key: budget},
                }
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
        return monitor

    @staticmethod
    def _custom_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        func_name = params.get("func")
        violation_name = params.get("violation_name", f"{spec.name}_custom")
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


class MonitorManager:
    def __init__(self, specs: Optional[List[MonitorSpec]] = None):
        self.specs = specs or []
        self.monitors: List[MonitorFn] = [MonitorFactory.build(s) for s in self.specs]

    @staticmethod
    def load_from_file(path: str | Path) -> "MonitorManager":
        data = _load_data_file(Path(path))
        specs = [MonitorSpec(**m) for m in data.get("monitors", [])]
        return MonitorManager(specs)

    def get_callables(self) -> List[MonitorFn]:
        return self.monitors


# --------------------------------------------------------------------------------------
# AgentNet (formerly DuetMindAgent)
# --------------------------------------------------------------------------------------

class AgentNet:
    """AgentNet: cognitive agent with style modulation, reasoning graph, persistence, monitors, and async dialogue."""

    def __init__(
        self,
        name: str,
        style: Dict[str, float],
        engine=None,
        monitors: Optional[List[MonitorFn]] = None,
        dialogue_config: Optional[Dict[str, Any]] = None,
        pre_monitors: Optional[List[MonitorFn]] = None,
    ):
        """
        Initialize AgentNet instance.
        
        Args:
            name: Agent name
            style: Style weights dictionary
            engine: Inference engine
            monitors: Post-style monitors (executed after style influence)
            dialogue_config: Dialogue configuration
            pre_monitors: Pre-style monitors (executed before style influence)
        """
        self.name = name
        self.style = style
        self.engine = engine
        self.monitors = monitors or []
        self.pre_monitors = pre_monitors or []
        self.knowledge_graph: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.dialogue_config = dialogue_config or {
            "default_rounds": 3,
            "max_rounds": 20,
            "convergence_window": 3,
            "convergence_min_overlap": 0.55,
            "convergence_min_conf": 0.75,
            "memory_guard": {
                "max_transcript_tokens": 5000,
                "truncate_strategy": "head"
            }
        }
        # Initialize resilience manager for fault tolerance
        self.resilience_manager = ResilienceManager()
        
        logger.info(
            f"AgentNet instance '{name}' initialized with style {style}, "
            f"{len(self.monitors)} monitors, {len(self.pre_monitors)} pre-monitors, dialogue config={self.dialogue_config}"
        )

    def register_monitor(self, monitor_fn: MonitorFn, pre_style: bool = False) -> None:
        """
        Register an additional monitor at runtime.
        
        Args:
            monitor_fn: Monitor function to register
            pre_style: If True, add to pre_monitors (run before style influence).
                      If False, add to monitors (run after style influence).
        """
        if pre_style:
            self.pre_monitors.append(monitor_fn)
        else:
            self.monitors.append(monitor_fn)

    def register_monitors(self, monitor_fns: List[MonitorFn], pre_style: bool = False) -> None:
        """
        Register multiple monitors at runtime.
        
        Args:
            monitor_fns: List of monitor functions to register
            pre_style: If True, add to pre_monitors. If False, add to monitors.
        """
        for monitor_fn in monitor_fns:
            self.register_monitor(monitor_fn, pre_style)

    # ---------------- Normalization Helpers ----------------
    @staticmethod
    def _normalize_engine_result(raw: Any) -> Dict[str, Any]:
        default_conf = 0.5
        if isinstance(raw, dict):
            out = dict(raw)
            if "content" not in out:
                out["content"] = json.dumps({k: v for k, v in out.items() if k != "confidence"})
            out.setdefault("confidence", default_conf)
            return out
        if isinstance(raw, str):
            return {"content": raw, "confidence": default_conf}
        if isinstance(raw, tuple):
            if len(raw) == 0:
                return {"content": "", "confidence": default_conf}
            if len(raw) == 1:
                return {"content": str(raw[0]), "confidence": default_conf}
            if len(raw) == 2:
                content, conf = raw
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = default_conf
                return {"content": str(content), "confidence": conf_f}
            content, conf, extras = raw[0], raw[1], raw[2]
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = default_conf
            base = {"content": str(content), "confidence": conf_f}
            if isinstance(extras, dict):
                for k, v in extras.items():
                    if k not in base:
                        base[k] = v
            return base
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            return {"content": "\n".join(raw), "confidence": default_conf}
        if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
            texts: List[str] = []
            meta_merge: Dict[str, Any] = {}
            for seg in raw:
                if "text" in seg:
                    texts.append(str(seg["text"]))
                for k, v in seg.items():
                    if k != "text" and k not in meta_merge:
                        meta_merge[k] = v
            if texts:
                base = {"content": "\n".join(texts), "confidence": default_conf}
                for k, v in meta_merge.items():
                    base.setdefault(k, v)
                return base
        if hasattr(raw, "content"):
            content = getattr(raw, "content")
            conf = getattr(raw, "confidence", default_conf)
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = default_conf
            result = {"content": str(content), "confidence": conf_f}
            for attr in dir(raw):
                if attr.startswith("_") or attr in ("content", "confidence"):
                    continue
                try:
                    val = getattr(raw, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                result.setdefault(attr, val)
            return result
        return {"content": str(raw), "confidence": default_conf}

    # ---------------- Reasoning Core (Sync) ----------------
    def generate_reasoning_tree(
        self,
        task: str,
        *,
        record_fault: bool = True,
        re_raise_fault: bool = True,
        include_monitor_trace: bool = False
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            raw = self.engine.safe_think(self.name, task) if self.engine else {
                "content": "No engine",
                "confidence": 0.5
            }
        except Exception as engine_exc:
            runtime = time.perf_counter() - start
            fault_node = f"{self.name}_engine_fault_{len(self.knowledge_graph)}"
            fault_payload = {
                "task": task,
                "result": {
                    "error": str(engine_exc),
                    "content": "Engine execution error",
                    "confidence": 0.0,
                    "runtime": runtime
                },
                "style": self.style,
                "timestamp": len(self.knowledge_graph),
            }
            self.knowledge_graph[fault_node] = fault_payload
            return {
                "root": fault_node,
                "result": fault_payload["result"],
                "agent": self.name,
                "fault": True,
                "fault_type": "engine_exception",
                "runtime_seconds": runtime,
                "schema_version": 2
            }

        runtime = time.perf_counter() - start
        base = self._normalize_engine_result(raw)
        base.setdefault("runtime", runtime)
        base["measured_runtime"] = runtime

        monitor_trace: List[Dict[str, Any]] = []
        
        # Run pre-style monitors first (before style influence)
        try:
            for pre_monitor in self.pre_monitors:
                before = time.perf_counter()
                pre_monitor(self, task, base)
                after = time.perf_counter()
                if include_monitor_trace:
                    monitor_trace.append({
                        "monitor": f"pre_{getattr(pre_monitor, '__name__', 'anonymous_monitor')}",
                        "elapsed": after - before
                    })
        except CognitiveFault as cf:
            if record_fault:
                node_key = f"{self.name}_pre_fault_{len(self.knowledge_graph)}"
                fault_record = {
                    "task": task,
                    "result": {
                        "error": "CognitiveFault (pre-style)",
                        "message": str(cf),
                        "violations": cf.violations,
                        "severity": cf.severity.value,
                        "runtime": runtime
                    },
                    "style": self.style,
                    "timestamp": len(self.knowledge_graph),
                }
                self.knowledge_graph[node_key] = fault_record
                self.interaction_history.append({
                    "task": task,
                    "node": node_key,
                    "fault": True,
                    "severity": cf.severity.value,
                    "violations": cf.violations,
                    "monitor_phase": "pre_style"
                })
            if re_raise_fault:
                raise
            return {
                "root": node_key,
                "result": fault_record["result"],
                "agent": self.name,
                "fault": True,
                "severity": cf.severity.value,
                "violations": cf.violations,
                "runtime_seconds": runtime,
                "schema_version": 2,
                "monitor_phase": "pre_style",
                **({"monitors": monitor_trace} if include_monitor_trace else {})
            }

        # Apply style influence after pre-monitors pass
        styled_result = self._apply_style_influence(base, task)
        
        # Run post-style monitors
        try:
            for monitor in self.monitors:
                before = time.perf_counter()
                monitor(self, task, styled_result)
                after = time.perf_counter()
                if include_monitor_trace:
                    monitor_trace.append({
                        "monitor": getattr(monitor, "__name__", "anonymous_monitor"),
                        "elapsed": after - before
                    })
        except CognitiveFault as cf:
            if record_fault:
                node_key = f"{self.name}_fault_{len(self.knowledge_graph)}"
                fault_record = {
                    "task": task,
                    "result": {
                        "error": "CognitiveFault",
                        "message": str(cf),
                        "violations": cf.violations,
                        "severity": cf.severity.value,
                        "runtime": runtime
                    },
                    "style": self.style,
                    "timestamp": len(self.knowledge_graph),
                }
                self.knowledge_graph[node_key] = fault_record
                self.interaction_history.append({
                    "task": task,
                    "node": node_key,
                    "fault": True,
                    "severity": cf.severity.value,
                    "violations": cf.violations
                })
            if re_raise_fault:
                raise
            return {
                "root": node_key,
                "result": fault_record["result"],
                "agent": self.name,
                "fault": True,
                "severity": cf.severity.value,
                "violations": cf.violations,
                "runtime_seconds": runtime,
                "schema_version": 2,
                **({"monitors": monitor_trace} if include_monitor_trace else {})
            }

        if "confidence" in base and "original_confidence" not in base:
            base["original_confidence"] = base["confidence"]

        if "error" in base:
            node_key = f"{self.name}_fault_{len(self.knowledge_graph)}"
            self.knowledge_graph[node_key] = {
                "task": task,
                "result": base,
                "style": self.style,
                "timestamp": len(self.knowledge_graph),
            }
            return {
                "root": node_key,
                "result": base,
                "agent": self.name,
                "fault": True,
                "runtime_seconds": runtime,
                "schema_version": 2
            }

        styled_result = self._apply_style_influence(base, task)
        node_key = f"{self.name}_reasoning_{len(self.knowledge_graph)}"
        record = {
            "task": task,
            "result": styled_result,
            "style": self.style,
            "timestamp": len(self.knowledge_graph),
        }
        self.knowledge_graph[node_key] = record
        self.interaction_history.append({"task": task, "node": node_key, "result": styled_result})

        response = {
            "root": node_key,
            "result": styled_result,
            "agent": self.name,
            "style_applied": True,
            "schema_version": 2,
            "runtime_seconds": runtime,
        }
        if include_monitor_trace:
            response["monitors"] = monitor_trace
        return response

    # ---------------- Reasoning Core (Async) ----------------
    async def async_generate_reasoning_tree(
        self,
        task: str,
        *,
        record_fault: bool = True,
        re_raise_fault: bool = True,
        include_monitor_trace: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, Any]:
        loop = loop or asyncio.get_event_loop()
        start = time.perf_counter()

        async_engine_func = getattr(self.engine, "safe_think_async", None) if self.engine else None
        try:
            if async_engine_func and inspect.iscoroutinefunction(async_engine_func):
                raw = await async_engine_func(self.name, task)
            else:
                raw = await loop.run_in_executor(
                    None,
                    lambda: self.engine.safe_think(self.name, task) if self.engine else {
                        "content": "No engine",
                        "confidence": 0.5
                    }
                )
        except Exception as engine_exc:
            runtime = time.perf_counter() - start
            fault_node = f"{self.name}_engine_fault_{len(self.knowledge_graph)}"
            fault_payload = {
                "task": task,
                "result": {
                    "error": str(engine_exc),
                    "content": "Engine execution error",
                    "confidence": 0.0,
                    "runtime": runtime
                },
                "style": self.style,
                "timestamp": len(self.knowledge_graph),
            }
            self.knowledge_graph[fault_node] = fault_payload
            return {
                "root": fault_node,
                "result": fault_payload["result"],
                "agent": self.name,
                "fault": True,
                "fault_type": "engine_exception",
                "runtime_seconds": runtime,
                "schema_version": 2
            }

        runtime = time.perf_counter() - start
        base = self._normalize_engine_result(raw)
        base.setdefault("runtime", runtime)
        base["measured_runtime"] = runtime

        monitor_trace: List[Dict[str, Any]] = []
        try:
            for monitor in self.monitors:
                before = time.perf_counter()
                await loop.run_in_executor(None, monitor, self, task, base)
                after = time.perf_counter()
                if include_monitor_trace:
                    monitor_trace.append({
                        "monitor": getattr(monitor, "__name__", "anonymous_monitor"),
                        "elapsed": after - before
                    })
        except CognitiveFault as cf:
            if record_fault:
                node_key = f"{self.name}_fault_{len(self.knowledge_graph)}"
                fault_record = {
                    "task": task,
                    "result": {
                        "error": "CognitiveFault",
                        "message": str(cf),
                        "violations": cf.violations,
                        "severity": cf.severity.value,
                        "runtime": runtime
                    },
                    "style": self.style,
                    "timestamp": len(self.knowledge_graph),
                }
                self.knowledge_graph[node_key] = fault_record
                self.interaction_history.append({
                    "task": task,
                    "node": node_key,
                    "fault": True,
                    "severity": cf.severity.value,
                    "violations": cf.violations
                })
            if re_raise_fault:
                raise
            return {
                "root": node_key,
                "result": fault_record["result"],
                "agent": self.name,
                "fault": True,
                "severity": cf.severity.value,
                "violations": cf.violations,
                "runtime_seconds": runtime,
                "schema_version": 2,
                **({"monitors": monitor_trace} if include_monitor_trace else {})
            }

        if "confidence" in base and "original_confidence" not in base:
            base["original_confidence"] = base["confidence"]

        if "error" in base:
            node_key = f"{self.name}_fault_{len(self.knowledge_graph)}"
            self.knowledge_graph[node_key] = {
                "task": task,
                "result": base,
                "style": self.style,
                "timestamp": len(self.knowledge_graph),
            }
            return {
                "root": node_key,
                "result": base,
                "agent": self.name,
                "fault": True,
                "runtime_seconds": runtime,
                "schema_version": 2
            }

        styled_result = self._apply_style_influence(base, task)
        node_key = f"{self.name}_reasoning_{len(self.knowledge_graph)}"
        record = {
            "task": task,
            "result": styled_result,
            "style": self.style,
            "timestamp": len(self.knowledge_graph),
        }
        self.knowledge_graph[node_key] = record
        self.interaction_history.append({"task": task, "node": node_key, "result": styled_result})

        response = {
            "root": node_key,
            "result": styled_result,
            "agent": self.name,
            "style_applied": True,
            "schema_version": 2,
            "runtime_seconds": runtime,
        }
        if include_monitor_trace:
            response["monitors"] = monitor_trace
        return response

    # ---------------- Resilience Methods ----------------
    def generate_reasoning_tree_with_resilience(self, task: str, include_monitor_trace: bool = False) -> Dict[str, Any]:
        """Generate reasoning tree with resilience patterns."""
        operation_name = f"reasoning_tree_{self.name}"
        
        def operation() -> Dict[str, Any]:
            # Capture the task and include_monitor_trace in closure
            return self.generate_reasoning_tree(task, include_monitor_trace=include_monitor_trace)
        
        result = self.resilience_manager.execute_with_resilience(operation_name, operation)
        
        if result["success"]:
            return result["result"]
        else:
            # Return a fallback response on persistent failure
            return {
                "root": f"{self.name}_fallback",
                "result": {
                    "content": f"[{self.name}] Fallback response due to resilience failure: {result['error']}",
                    "confidence": 0.1,
                    "fallback": True,
                    "resilience_attempts": result["attempts"]
                },
                "agent": self.name,
                "fault": True,
                "schema_version": 2
            }
    
    async def async_generate_reasoning_tree_with_resilience(self, task: str, include_monitor_trace: bool = False) -> Dict[str, Any]:
        """Async generate reasoning tree with resilience patterns."""
        operation_name = f"async_reasoning_tree_{self.name}"
        
        async def operation() -> Dict[str, Any]:
            return await self.async_generate_reasoning_tree(task, include_monitor_trace=include_monitor_trace)
        
        result = await self.resilience_manager.execute_with_resilience_async(operation_name, operation)
        
        if result["success"]:
            return result["result"]
        else:
            return {
                "root": f"{self.name}_async_fallback",
                "result": {
                    "content": f"[{self.name}] Async fallback response due to resilience failure: {result['error']}",
                    "confidence": 0.1,
                    "fallback": True,
                    "resilience_attempts": result["attempts"]
                },
                "agent": self.name,
                "fault": True,
                "schema_version": 2
            }
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get resilience metrics for this agent."""
        return self.resilience_manager.get_metrics()

    # ---------------- Style Influence ----------------
    def _apply_style_influence(self, base_result: Dict[str, Any], task: str) -> Dict[str, Any]:
        styled_result = dict(base_result)
        logic_weight = float(self.style.get("logic", 0.5))
        creativity_weight = float(self.style.get("creativity", 0.5))
        analytical_weight = float(self.style.get("analytical", 0.5))
        if "confidence" in styled_result:
            styled_result["confidence"] *= (0.8 + analytical_weight * 0.4)
        insights: List[str] = []
        if logic_weight > 0.7:
            insights.append("Applying rigorous logical validation")
        if creativity_weight > 0.7:
            insights.append("Exploring creative alternative perspectives")
        if analytical_weight > 0.7:
            insights.append("Conducting systematic decomposition")
        styled_result["style_insights"] = insights
        styled_result["style_signature"] = self.name
        return styled_result

    # ---------------- Basic Two-Agent Dialogue (Sync) ----------------
    def dialogue_with(self, other_agent: 'AgentNet', topic: str, rounds: int = 3) -> Dict[str, Any]:
        dialogue_history: List[Dict[str, Any]] = []
        current_topic = topic
        logger.info(f"Dialogue between {self.name} and {other_agent.name} on {topic}")
        for round_num in range(rounds):
            my_response = self.generate_reasoning_tree(f"Round {round_num + 1}: {current_topic}")
            dialogue_history.append({"round": round_num + 1, "agent": self.name, "response": my_response})
            other_response = other_agent.generate_reasoning_tree(
                f"Responding to {self.name}'s perspective on: {current_topic}"
            )
            dialogue_history.append({"round": round_num + 1, "agent": other_agent.name, "response": other_response})
            current_topic = self._evolve_topic(current_topic, my_response, other_response)
        synthesis = self._synthesize_dialogue(dialogue_history, topic)
        return {
            "original_topic": topic,
            "final_topic": current_topic,
            "dialogue_history": dialogue_history,
            "synthesis": synthesis,
            "participants": [self.name, other_agent.name],
        }

    def _evolve_topic(self, current_topic: str, response1: Dict[str, Any], response2: Dict[str, Any]) -> str:
        conf1 = float(response1.get("result", {}).get("confidence", 0.5))
        conf2 = float(response2.get("result", {}).get("confidence", 0.5))
        if conf1 > 0.8 and conf2 > 0.8:
            return f"Deep dive into: {current_topic}"
        if conf1 < 0.4 or conf2 < 0.4:
            return f"Alternative approach to: {current_topic}"
        return f"Balanced exploration of: {current_topic}"

    def _synthesize_dialogue(self, dialogue_history: List[Dict[str, Any]], original_topic: str) -> Dict[str, Any]:
        contributions: Dict[str, Any] = {}
        total_conf = 0.0
        insights_count = 0
        for entry in dialogue_history:
            agent = entry["agent"]
            contributions.setdefault(agent, {"rounds": 0, "insights": []})
            contributions[agent]["rounds"] += 1
            result = entry["response"].get("result", {})
            total_conf += float(result.get("confidence", 0.5))
            if "style_insights" in result:
                contributions[agent]["insights"].extend(result["style_insights"])
                insights_count += len(result["style_insights"])
        avg_conf = total_conf / len(dialogue_history) if dialogue_history else 0.0
        return {
            "original_topic": original_topic,
            "dialogue_quality": avg_conf,
            "total_insights": insights_count,
            "agent_contributions": contributions,
            "cognitive_diversity": len({c["agent"] for c in dialogue_history}),
        }

    # ---------------- Advanced Multi-Party Dialogue (Sync) ----------------
    def multi_party_dialogue(
        self,
        agents: List['AgentNet'],
        topic: str,
        rounds: int = 5,
        strategy: str = "round_robin",
        mode: str = "general",
        summarizer: Optional['AgentNet'] = None,
        convergence: bool = True,
        callbacks: Optional[Dict[str, Callable[..., None]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not agents:
            raise ValueError("No agents provided for multi-party dialogue.")
        rounds = min(rounds, self.dialogue_config.get("max_rounds", rounds))
        session_id = f"dlg_{int(time.time()*1000)}_{len(self.interaction_history)}"
        logger.info(f"[Dialogue:{session_id}] Starting multi-party dialogue with {len(agents)} agents, topic='{topic}'")

        callbacks = callbacks or {}
        summarizer = summarizer or (self if self not in agents else agents[0])
        transcript: List[Dict[str, Any]] = []
        topic_evolution: List[str] = [topic]
        last_contents: List[str] = []
        convergence_hit = False

        def mode_prefix(m: str) -> str:
            if m == "debate":
                return "Provide a position, defend it, then critique prior points."
            if m == "brainstorm":
                return "Generate diverse, novel ideas without judging prematurely."
            if m == "consensus":
                return "Move toward shared agreement; highlight convergences explicitly."
            return "Offer constructive reasoning."

        base_directive = mode_prefix(mode)

        for round_index in range(1, rounds + 1):
            if callbacks.get("on_round_start"):
                callbacks["on_round_start"](round_index, {
                    "topic": topic,
                    "round": round_index,
                    "session_id": session_id
                })

            ordered_agents = agents
            if strategy == "sequential-panel" and round_index > 1:
                ordered_agents = [agents[0]] + [a for a in agents[1:]]

            round_turns: List[Dict[str, Any]] = []

            for ag in ordered_agents:
                prompt = self._compose_turn_prompt(
                    agent=ag,
                    topic=topic,
                    round_index=round_index,
                    base_directive=base_directive,
                    transcript_tail=self._truncate_transcript(transcript)[-3:],
                    mode=mode,
                )
                turn_result = ag.generate_reasoning_tree(prompt)
                content = turn_result.get("result", {}).get("content", "")
                confidence = float(turn_result.get("result", {}).get("confidence", 0.5))

                turn_payload = {
                    "session_id": session_id,
                    "round": round_index,
                    "agent": ag.name,
                    "prompt": prompt,
                    "content": content,
                    "confidence": confidence,
                    "raw": turn_result
                }
                transcript.append(turn_payload)
                round_turns.append(turn_payload)
                last_contents.append(content)

                if callbacks.get("on_turn_end"):
                    callbacks["on_turn_end"](round_index, ag, turn_payload)

            _ = self._rolling_synthesis(summarizer, transcript, topic)
            topic = self._multi_topic_evolution(topic, round_turns, mode)
            topic_evolution.append(topic)

            if convergence and self._check_convergence(last_contents, agents[0].dialogue_config):
                convergence_hit = True
                logger.info(f"[Dialogue:{session_id}] Convergence criteria met at round {round_index}.")
                break

        final_summary = self._final_synthesis(summarizer, transcript, topic, mode)
        metrics = self._dialogue_metrics(transcript)

        session_record = {
            "session_id": session_id,
            "topic_start": topic_evolution[0],
            "topic_final": topic,
            "topic_evolution": topic_evolution,
            "transcript": transcript,
            "participants": [a.name for a in agents],
            "rounds_executed": metrics["rounds"],
            "metrics": metrics,
            "final_summary": final_summary,
            "converged": convergence_hit,
            "mode": mode,
            "strategy": strategy,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        self.interaction_history.append({
            "type": "dialogue_session",
            "session_id": session_id,
            "summary": {
                "topic_final": topic,
                "participants": session_record["participants"],
                "rounds": metrics["rounds"],
                "converged": convergence_hit
            }
        })

        if callbacks.get("on_session_end"):
            callbacks["on_session_end"](session_record)

        return session_record

    # ---------------- Advanced Multi-Party Dialogue (Async) ----------------
    async def async_multi_party_dialogue(
        self,
        agents: List['AgentNet'],
        topic: str,
        rounds: int = 5,
        strategy: str = "round_robin",
        mode: str = "general",
        summarizer: Optional['AgentNet'] = None,
        convergence: bool = True,
        callbacks: Optional[Dict[str, Callable[..., Awaitable[None] | None]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallel_round: bool = False,
        include_monitor_trace: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, Any]:
        if not agents:
            raise ValueError("No agents provided for async multi-party dialogue.")
        loop = loop or asyncio.get_event_loop()
        rounds = min(rounds, self.dialogue_config.get("max_rounds", rounds))
        session_id = f"adlg_{int(time.time()*1000)}_{len(self.interaction_history)}"
        logger.info(f"[AsyncDialogue:{session_id}] start {len(agents)} agents topic='{topic}' parallel={parallel_round}")

        callbacks = callbacks or {}
        summarizer = summarizer or (self if self not in agents else agents[0])
        transcript: List[Dict[str, Any]] = []
        topic_evolution: List[str] = [topic]
        last_contents: List[str] = []
        convergence_hit = False

        def mode_prefix(m: str) -> str:
            if m == "debate":
                return "Provide a position, defend it, then critique prior points."
            if m == "brainstorm":
                return "Generate diverse, novel ideas without judging prematurely."
            if m == "consensus":
                return "Move toward shared agreement; highlight convergences explicitly."
            return "Offer constructive reasoning."

        base_directive = mode_prefix(mode)

        async def maybe_call(cb: Optional[Callable], *args, **kwargs):
            if cb is None:
                return
            res = cb(*args, **kwargs)
            if inspect.isawaitable(res):
                await res

        for round_index in range(1, rounds + 1):
            await maybe_call(
                callbacks.get("on_round_start"),
                round_index,
                {"topic": topic, "round": round_index, "session_id": session_id}
            )

            ordered_agents = agents
            if strategy == "sequential-panel" and round_index > 1:
                ordered_agents = [agents[0]] + [a for a in agents[1:]]

            round_turns: List[Dict[str, Any]] = []

            async def run_turn(ag: AgentNet) -> Dict[str, Any]:
                prompt = self._compose_turn_prompt(
                    agent=ag,
                    topic=topic,
                    round_index=round_index,
                    base_directive=base_directive,
                    transcript_tail=self._truncate_transcript(transcript)[-3:],
                    mode=mode,
                )
                turn_result = await ag.async_generate_reasoning_tree(
                    prompt,
                    include_monitor_trace=include_monitor_trace
                )
                content = turn_result.get("result", {}).get("content", "")
                confidence = float(turn_result.get("result", {}).get("confidence", 0.5))
                payload = {
                    "session_id": session_id,
                    "round": round_index,
                    "agent": ag.name,
                    "prompt": prompt,
                    "content": content,
                    "confidence": confidence,
                    "raw": turn_result
                }
                return payload

            if parallel_round:
                # Enhanced parallel execution with monitoring and timeouts
                start_time = time.perf_counter()
                tasks = [asyncio.create_task(run_turn(ag)) for ag in ordered_agents]
                
                try:
                    # Add timeout for parallel execution
                    timeout = self.dialogue_config.get("parallel_timeout", 30.0)
                    turn_payloads = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), 
                        timeout=timeout
                    )
                    
                    # Handle any exceptions in parallel execution
                    valid_payloads = []
                    failed_agents = []
                    for i, payload in enumerate(turn_payloads):
                        if isinstance(payload, Exception):
                            logger.warning(f"Agent {ordered_agents[i].name} failed in parallel round {round_index}: {payload}")
                            failed_agents.append(ordered_agents[i].name)
                            # Create a failure payload
                            failure_payload = {
                                "session_id": session_id,
                                "round": round_index,
                                "agent": ordered_agents[i].name,
                                "prompt": "Failed execution",
                                "content": f"Agent failed: {str(payload)[:100]}",
                                "confidence": 0.0,
                                "raw": {"error": str(payload), "failed": True}
                            }
                            valid_payloads.append(failure_payload)
                        else:
                            valid_payloads.append(payload)
                    
                    turn_payloads = valid_payloads
                    parallel_duration = time.perf_counter() - start_time
                    
                    # Log parallel execution statistics
                    logger.info(f"[AsyncDialogue:{session_id}] Parallel round {round_index} completed: "
                              f"{len(ordered_agents)} agents in {parallel_duration:.2f}s, "
                              f"failures: {len(failed_agents)}")
                    
                    if failed_agents:
                        logger.warning(f"Failed agents in round {round_index}: {failed_agents}")
                        
                except asyncio.TimeoutError:
                    logger.error(f"[AsyncDialogue:{session_id}] Parallel round {round_index} timed out after {timeout}s")
                    # Cancel remaining tasks
                    for task in tasks:
                        task.cancel()
                    
                    # Create timeout payloads for all agents
                    turn_payloads = []
                    for ag in ordered_agents:
                        timeout_payload = {
                            "session_id": session_id,
                            "round": round_index,
                            "agent": ag.name,
                            "prompt": "Timed out",
                            "content": f"Agent timed out after {timeout}s",
                            "confidence": 0.0,
                            "raw": {"error": "timeout", "timeout": timeout}
                        }
                        turn_payloads.append(timeout_payload)
                        
            else:
                turn_payloads = []
                for ag in ordered_agents:
                    turn_payloads.append(await run_turn(ag))

            for payload in turn_payloads:
                transcript.append(payload)
                round_turns.append(payload)
                last_contents.append(payload["content"])
                await maybe_call(callbacks.get("on_turn_end"), round_index, payload["agent"], payload)

            _ = await self._async_rolling_synthesis(summarizer, transcript, topic)
            topic = self._multi_topic_evolution(topic, round_turns, mode)
            topic_evolution.append(topic)

            if convergence and self._check_convergence(last_contents, agents[0].dialogue_config):
                convergence_hit = True
                logger.info(f"[AsyncDialogue:{session_id}] Convergence at round {round_index}")
                break

        final_summary = await self._async_final_synthesis(summarizer, transcript, topic, mode)
        metrics = self._dialogue_metrics(transcript)
        session_record = {
            "session_id": session_id,
            "topic_start": topic_evolution[0],
            "topic_final": topic,
            "topic_evolution": topic_evolution,
            "transcript": transcript,
            "participants": [a.name for a in agents],
            "rounds_executed": metrics["rounds"],
            "metrics": metrics,
            "final_summary": final_summary,
            "converged": convergence_hit,
            "mode": mode,
            "strategy": strategy,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "parallel_round": parallel_round
        }

        self.interaction_history.append({
            "type": "dialogue_session_async",
            "session_id": session_id,
            "summary": {
                "topic_final": topic,
                "participants": session_record["participants"],
                "rounds": metrics["rounds"],
                "converged": convergence_hit,
                "parallel_round": parallel_round
            }
        })

        await maybe_call(callbacks.get("on_session_end"), session_record)
        return session_record

    # ------------- Dialogue Mode Convenience Wrappers (Sync) -------------
    def debate(self, agents: List['AgentNet'], topic: str, rounds: int = 6, **kwargs) -> Dict[str, Any]:
        return self.multi_party_dialogue(agents, topic, rounds=rounds, mode="debate", **kwargs)

    def brainstorm(self, agents: List['AgentNet'], topic: str, rounds: int = 5, **kwargs) -> Dict[str, Any]:
        return self.multi_party_dialogue(agents, topic, rounds=rounds, mode="brainstorm", **kwargs)

    def consensus(self, agents: List['AgentNet'], topic: str, rounds: int = 7, **kwargs) -> Dict[str, Any]:
        return self.multi_party_dialogue(agents, topic, rounds=rounds, mode="consensus", **kwargs)

    # ------------- Dialogue Mode Convenience Wrappers (Async) -------------
    async def async_debate(self, agents: List['AgentNet'], topic: str, rounds: int = 6, **kwargs) -> Dict[str, Any]:
        return await self.async_multi_party_dialogue(agents, topic, rounds=rounds, mode="debate", **kwargs)

    async def async_brainstorm(self, agents: List['AgentNet'], topic: str, rounds: int = 5, **kwargs) -> Dict[str, Any]:
        return await self.async_multi_party_dialogue(agents, topic, rounds=rounds, mode="brainstorm", **kwargs)

    async def async_consensus(self, agents: List['AgentNet'], topic: str, rounds: int = 7, **kwargs) -> Dict[str, Any]:
        return await self.async_multi_party_dialogue(agents, topic, rounds=rounds, mode="consensus", **kwargs)

    # ------------- Internal Helpers (Shared) -------------
    def _compose_turn_prompt(
        self,
        agent: 'AgentNet',
        topic: str,
        round_index: int,
        base_directive: str,
        transcript_tail: List[Dict[str, Any]],
        mode: str,
    ) -> str:
        tail_lines: List[str] = []
        for t in transcript_tail:
            tail_lines.append(f"{t['agent']} (r{t['round']}): {t['content'][:180]}")
        context_block = "\n".join(tail_lines) if tail_lines else "No prior turns."
        role_hint = self._role_hint(agent, mode)
        return (
            f"Round {round_index} | Topic: {topic}\n"
            f"Mode Directive: {base_directive}\n"
            f"Role Hint ({agent.name}): {role_hint}\n"
            f"Recent Context:\n{context_block}\n"
            f"Respond with reasoning, avoid repetition, add value."
        )

    def _role_hint(self, agent: 'AgentNet', mode: str) -> str:
        logic = agent.style.get("logic", 0.5)
        creativity = agent.style.get("creativity", 0.5)
        analytical = agent.style.get("analytical", 0.5)
        if mode == "brainstorm":
            if creativity > 0.7:
                return "Ideation catalyst - produce unconventional angles."
            return "Supportive contributor - expand on emerging ideas."
        if mode == "debate":
            if logic > 0.7:
                return "Construct rigorous arguments and rebuttals."
            return "Probe and test assumptions of others' arguments."
        if mode == "consensus":
            if analytical > 0.7:
                return "Synthesize points and propose alignment candidates."
            return "Rephrase others' points to foster shared understanding."
        return "Offer balanced and constructive reasoning."

    def _multi_topic_evolution(self, current_topic: str, round_turns: List[Dict[str, Any]], mode: str) -> str:
        if not round_turns:
            return current_topic
        avg_conf = sum(t["confidence"] for t in round_turns) / len(round_turns)
        if avg_conf > 0.82:
            return f"Deep dive: {current_topic}"
        if avg_conf < 0.45:
            return f"Reframe: {current_topic}"
        if mode == "brainstorm" and any("idea" in t["content"].lower() for t in round_turns):
            return f"Expand ideas around: {current_topic}"
        return f"Explore: {current_topic}"

    def _rolling_synthesis(self, summarizer: 'AgentNet', transcript: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        tail = transcript[-4:]
        synthesis_prompt = (
            f"Produce a concise rolling synthesis for topic '{topic}'. "
            "Focus on new distinct points, maintain brevity."
        )
        tail_text = "\n".join(f"{t['agent']}@r{t['round']}: {t['content'][:160]}" for t in tail)
        synth = summarizer.generate_reasoning_tree(f"{synthesis_prompt}\nRecent Turns:\n{tail_text}")
        return {"content": synth.get("result", {}).get("content", ""), "confidence": synth.get("result", {}).get("confidence", 0.5)}

    async def _async_rolling_synthesis(self, summarizer: 'AgentNet', transcript: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        tail = transcript[-4:]
        synthesis_prompt = (
            f"Produce a concise rolling synthesis for topic '{topic}'. "
            "Focus on new distinct points, maintain brevity."
        )
        tail_text = "\n".join(f"{t['agent']}@r{t['round']}: {t['content'][:160]}" for t in tail)
        synth = await summarizer.async_generate_reasoning_tree(f"{synthesis_prompt}\nRecent Turns:\n{tail_text}")
        return {"content": synth.get("result", {}).get("content", ""), "confidence": synth.get("result", {}).get("confidence", 0.5)}

    def _final_synthesis(
        self,
        summarizer: 'AgentNet',
        transcript: List[Dict[str, Any]],
        final_topic: str,
        mode: str
    ) -> Dict[str, Any]:
        excerpt = "\n".join(f"{t['agent']}@r{t['round']}: {t['content'][:180]}" for t in transcript[-12:])
        directive = {
            "debate": "Highlight principal arguments, rebuttals, unresolved tensions, and potential synthesis.",
            "brainstorm": "Cluster related ideas, identify most novel and actionable suggestions.",
            "consensus": "Summarize aligned points, residual disagreements, and propose next action.",
        }.get(mode, "Summarize key contributions and emergent directions.")
        final_prompt = (
            f"Final {mode} synthesis for topic: {final_topic}\n"
            f"Directive: {directive}\nRecent Extract (truncated):\n{excerpt}\n"
            "Output: structured concise synthesis."
        )
        res = summarizer.generate_reasoning_tree(final_prompt)
        return {
            "content": res.get("result", {}).get("content", ""),
            "confidence": res.get("result", {}).get("confidence", 0.5),
            "mode": mode
        }

    async def _async_final_synthesis(
        self,
        summarizer: 'AgentNet',
        transcript: List[Dict[str, Any]],
        final_topic: str,
        mode: str
    ) -> Dict[str, Any]:
        excerpt = "\n".join(f"{t['agent']}@r{t['round']}: {t['content'][:180]}" for t in transcript[-12:])
        directive = {
            "debate": "Highlight principal arguments, rebuttals, unresolved tensions, and potential synthesis.",
            "brainstorm": "Cluster related ideas, identify most novel and actionable suggestions.",
            "consensus": "Summarize aligned points, residual disagreements, and propose next action.",
        }.get(mode, "Summarize key contributions and emergent directions.")
        final_prompt = (
            f"Final {mode} synthesis for topic: {final_topic}\n"
            f"Directive: {directive}\nRecent Extract (truncated):\n{excerpt}\n"
            "Output: structured concise synthesis."
        )
        res = await summarizer.async_generate_reasoning_tree(final_prompt)
        return {
            "content": res.get("result", {}).get("content", ""),
            "confidence": res.get("result", {}).get("confidence", 0.5),
            "mode": mode
        }

    def _check_convergence(self, last_contents: List[str], dialogue_config: Optional[Dict[str, Any]] = None) -> bool:
        config = dialogue_config or self.dialogue_config
        window = config.get("convergence_window", 3)
        if len(last_contents) < window:
            return False
        
        recent = last_contents[-window:]
        
        # Primary lexical convergence check
        lexical_converged = self._check_lexical_convergence(recent, config)
        
        # Semantic convergence check (if enabled)
        semantic_converged = True  # Default to True if not checking semantic
        if config.get("use_semantic_convergence", False):
            semantic_converged = self._check_semantic_convergence(recent, config)
        
        # Confidence-based convergence check
        confidence_converged = self._check_confidence_convergence(recent, config)
        
        # Combined convergence decision
        convergence_strategy = config.get("convergence_strategy", "lexical_only")
        
        if convergence_strategy == "lexical_only":
            result = lexical_converged
        elif convergence_strategy == "semantic_only":
            result = semantic_converged
        elif convergence_strategy == "lexical_and_semantic":
            result = lexical_converged and semantic_converged
        elif convergence_strategy == "lexical_or_semantic":
            result = lexical_converged or semantic_converged
        elif convergence_strategy == "confidence_gated":
            result = confidence_converged and (lexical_converged or semantic_converged)
        else:
            result = lexical_converged
        
        logger.debug(f"Convergence check: lexical={lexical_converged}, semantic={semantic_converged}, "
                    f"confidence={confidence_converged}, strategy={convergence_strategy}, result={result}")
        
        return result
        
    def _check_lexical_convergence(self, recent_contents: List[str], config: Dict[str, Any]) -> bool:
        """Check convergence using lexical overlap (Jaccard similarity)."""
        sets = [set(self._tokenize(c)) for c in recent_contents]
        if not sets:
            return False
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        if not union:
            return False
        overlap = len(intersection) / max(1, len(union))
        min_overlap = config.get("convergence_min_overlap", 0.55)
        return overlap >= min_overlap
    
    def _check_semantic_convergence(self, recent_contents: List[str], config: Dict[str, Any]) -> bool:
        """Check convergence using semantic similarity (simplified approach)."""
        # For now, use length and keyword similarity as a proxy for semantic similarity
        # In a full implementation, this would use embeddings/cosine similarity
        
        if len(recent_contents) < 2:
            return False
            
        # Check if content lengths are similar (indicating similar depth of reasoning)
        lengths = [len(content.split()) for content in recent_contents]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_converged = length_variance < config.get("semantic_length_variance_threshold", 25.0)
        
        # Check for shared key concepts
        key_concept_sets = []
        for content in recent_contents:
            # Extract potential key concepts (longer words, capitalized terms)
            concepts = set(re.findall(r'\b[A-Z][a-z]{3,}|[a-z]{5,}\b', content))
            key_concept_sets.append(concepts)
        
        if key_concept_sets:
            concept_intersection = set.intersection(*key_concept_sets)
            concept_union = set.union(*key_concept_sets)
            concept_overlap = len(concept_intersection) / max(1, len(concept_union)) if concept_union else 0
            concept_converged = concept_overlap >= config.get("semantic_concept_overlap", 0.3)
        else:
            concept_converged = True
            
        return length_converged and concept_converged
    
    def _check_confidence_convergence(self, recent_contents: List[str], config: Dict[str, Any]) -> bool:
        """Check if confidence levels are sufficiently high for meaningful convergence."""
        # Extract confidence from the last entries in transcript
        # This is a simplified approach - in full implementation would track confidence properly
        min_confidence = config.get("convergence_min_confidence", 0.6)
        
        # For now, assume confidence based on content quality indicators
        quality_scores = []
        for content in recent_contents:
            # Simple heuristics for content quality/confidence
            word_count = len(content.split())
            sentence_count = len([s for s in content.split('.') if s.strip()])
            avg_sentence_length = word_count / max(1, sentence_count)
            
            # Higher confidence for well-structured content
            quality_score = min(1.0, (word_count / 50) * 0.3 + (avg_sentence_length / 15) * 0.7)
            quality_scores.append(quality_score)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        return avg_quality >= min_confidence

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2]

    @staticmethod
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)

    def _truncate_transcript(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        memory_config = self.dialogue_config.get("memory_guard", {})
        max_tokens = memory_config.get("max_transcript_tokens", 5000)
        truncate_strategy = memory_config.get("truncate_strategy", "head")
        if not transcript:
            return transcript
        current_tokens = 0
        for turn in transcript:
            content = turn.get("content", "")
            prompt = turn.get("prompt", "")
            current_tokens += self._count_tokens(content) + self._count_tokens(prompt)
        if current_tokens <= max_tokens:
            return transcript
        logger.info(f"Transcript has {current_tokens} tokens, truncating to {max_tokens} using {truncate_strategy} strategy")
        if truncate_strategy == "head":
            truncated = []
            remaining_tokens = max_tokens
            for turn in reversed(transcript):
                content = turn.get("content", "")
                prompt = turn.get("prompt", "")
                turn_tokens = self._count_tokens(content) + self._count_tokens(prompt)
                if turn_tokens <= remaining_tokens:
                    truncated.insert(0, turn)
                    remaining_tokens -= turn_tokens
                else:
                    break
            return truncated
        elif truncate_strategy == "tail":
            truncated = []
            remaining_tokens = max_tokens
            for turn in transcript:
                content = turn.get("content", "")
                prompt = turn.get("prompt", "")
                turn_tokens = self._count_tokens(content) + self._count_tokens(prompt)
                if turn_tokens <= remaining_tokens:
                    truncated.append(turn)
                    remaining_tokens -= turn_tokens
                else:
                    break
            return truncated
        else:
            return self._truncate_transcript(transcript)

    def _dialogue_metrics(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not transcript:
            return {"rounds": 0, "avg_confidence": 0.0, "turns_per_agent": {}, "lexical_diversity": 0.0, "total_turns": 0}
        turns_per_agent: Dict[str, int] = {}
        total_conf = 0.0
        vocab: set[str] = set()
        for t in transcript:
            turns_per_agent[t["agent"]] = turns_per_agent.get(t["agent"], 0) + 1
            total_conf += t.get("confidence", 0.5)
            vocab.update(self._tokenize(t.get("content", "")))
        avg_conf = total_conf / len(transcript)
        lexical_diversity = len(vocab) / max(1, len(transcript))
        rounds = max(t["round"] for t in transcript)
        return {
            "rounds": rounds,
            "avg_confidence": avg_conf,
            "turns_per_agent": turns_per_agent,
            "lexical_diversity": lexical_diversity,
            "total_turns": len(transcript)
        }

    # ---------------- Persistence ----------------
    def save_state(self, path: str) -> None:
        state = {
            "name": self.name,
            "style": self.style,
            "knowledge_graph": self.knowledge_graph,
            "interaction_history": self.interaction_history,
            "dialogue_config": self.dialogue_config,
        }
        Path(path).write_text(json.dumps(state, indent=2))
        logger.info(f"Agent state saved to {path}")

    def persist_session(self, session_record: dict, directory: str = "sessions") -> str:
        sessions_dir = Path(directory)
        sessions_dir.mkdir(exist_ok=True)
        session_id = session_record.get("session_id", f"session_{int(time.time()*1000)}")
        timestamp = session_record.get("timestamp", time.time())
        filename = f"{session_id}_{int(timestamp)}.json"
        filepath = sessions_dir / filename
        session_copy = dict(session_record)
        session_copy["persistence_metadata"] = {
            "saved_at": time.time(),
            "saved_by_agent": self.name,
            "filepath": str(filepath)
        }
        filepath.write_text(json.dumps(session_copy, indent=2))
        logger.info(f"Session '{session_id}' persisted to {filepath}")
        return str(filepath)

    @classmethod
    def load_state(
        cls,
        path: str,
        engine=None,
        monitors: Optional[List[MonitorFn]] = None
    ) -> 'AgentNet':
        state = json.loads(Path(path).read_text())
        agent = cls(state["name"], state.get("style", {}), engine=engine,
                    monitors=monitors, dialogue_config=state.get("dialogue_config"))
        agent.knowledge_graph = state.get("knowledge_graph", {})
        agent.interaction_history = state.get("interaction_history", [])
        logger.info(f"Agent state loaded from {path}")
        return agent

    @staticmethod
    def from_config(config_path: str | Path, engine=None) -> 'AgentNet':
        cfg = _load_data_file(Path(config_path))
        name = cfg["name"]
        style = cfg.get("style", {})
        dialogue_cfg = cfg.get("dialogue_config")
        monitors_path = (
            cfg.get("monitors", {}).get("file")
            if isinstance(cfg.get("monitors"), dict)
            else cfg.get("monitors")
        )
        monitors: List[MonitorFn] = []
        if monitors_path:
            mm = MonitorManager.load_from_file(Path(config_path).parent / monitors_path)
            monitors = mm.get_callables()
        return AgentNet(name=name, style=style, engine=engine,
                        monitors=monitors, dialogue_config=dialogue_cfg)


# --------------------------------------------------------------------------------------
# Example Engine (renamed if desired) - kept as ExampleEngine for simplicity
# --------------------------------------------------------------------------------------

class ExampleEngine:
    """Toy engine that returns a dict result with content and confidence."""

    def safe_think(self, agent_name: str, task: str) -> Dict[str, Any]:
        time.sleep(0.02)
        conf = 0.6 if "Round" in task else 0.9
        content = f"[{agent_name}] Thoughts about: {task}"
        return {"content": content, "confidence": conf}

    async def safe_think_async(self, agent_name: str, task: str) -> Dict[str, Any]:
        await asyncio.sleep(0.02)
        conf = 0.65 if "Round" in task else 0.92
        content = f"[{agent_name}] (async) Thoughts about: {task}"
        return {"content": content, "confidence": conf}


# --------------------------------------------------------------------------------------
# Fault Injection & Resilience Experiments
# --------------------------------------------------------------------------------------

class FaultType(str, Enum):
    """Types of faults that can be injected for resilience testing."""
    NETWORK_TIMEOUT = "network_timeout"
    PROCESSING_ERROR = "processing_error"
    MEMORY_PRESSURE = "memory_pressure"
    RATE_LIMIT = "rate_limit"
    RANDOM_FAILURE = "random_failure"


@dataclass
class FaultConfig:
    """Configuration for fault injection."""
    fault_type: FaultType
    probability: float = 0.1  # 0.0 to 1.0
    delay_ms: int = 0
    error_message: str = ""
    recovery_attempts: int = 3


class FaultInjectionMonitor:
    """Monitor that injects faults for resilience testing."""
    
    def __init__(self, name: str, fault_configs: List[FaultConfig], 
                 severity: Severity = Severity.MINOR):
        self.name = name
        self.severity = severity
        self.fault_configs = fault_configs
        self.fault_stats = {
            "injected_count": 0,
            "recovered_count": 0,
            "failed_count": 0,
            "fault_types": {}
        }
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        start_time = time.perf_counter()
        
        # Inject faults based on configuration
        for config in self.fault_configs:
            if self._should_inject_fault(config):
                self._inject_fault(config, outcome)
        
        elapsed = time.perf_counter() - start_time
        return True, None, elapsed
    
    def _should_inject_fault(self, config: FaultConfig) -> bool:
        """Determine if a fault should be injected based on probability."""
        import random
        return random.random() < config.probability
    
    def _inject_fault(self, config: FaultConfig, outcome: Dict[str, Any]) -> None:
        """Inject a specific type of fault."""
        self.fault_stats["injected_count"] += 1
        fault_type = config.fault_type.value
        self.fault_stats["fault_types"][fault_type] = self.fault_stats["fault_types"].get(fault_type, 0) + 1
        
        logger.warning(f"[FAULT_INJECTION] Injecting {fault_type} fault")
        
        if config.delay_ms > 0:
            time.sleep(config.delay_ms / 1000.0)
        
        if config.fault_type == FaultType.PROCESSING_ERROR:
            # Simulate processing error by modifying outcome
            outcome["fault_injected"] = {
                "type": fault_type,
                "message": config.error_message or "Simulated processing error",
                "recoverable": True
            }
        elif config.fault_type == FaultType.NETWORK_TIMEOUT:
            # Simulate network timeout
            outcome["network_fault"] = {
                "type": "timeout",
                "duration_ms": config.delay_ms,
                "recoverable": True
            }
        elif config.fault_type == FaultType.MEMORY_PRESSURE:
            # Simulate memory pressure
            outcome["memory_fault"] = {
                "type": "pressure",
                "severity": "high",
                "suggested_action": "reduce_batch_size"
            }


class ResilienceManager:
    """Manages resilience strategies for agent operations."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5,
                 circuit_breaker_threshold: int = 5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.failure_counts = {}
        self.circuit_breakers = {}
        self.resilience_metrics = {
            "total_operations": 0,
            "failed_operations": 0,
            "recovered_operations": 0,
            "circuit_breaker_trips": 0
        }
    
    def execute_with_resilience(self, operation_name: str, 
                              operation: Callable[[], Any]) -> Dict[str, Any]:
        """Execute an operation with resilience patterns."""
        self.resilience_metrics["total_operations"] += 1
        
        # Check circuit breaker
        if self._is_circuit_open(operation_name):
            logger.warning(f"[RESILIENCE] Circuit breaker open for {operation_name}")
            return {
                "success": False,
                "error": "circuit_breaker_open",
                "result": None,
                "attempts": 0
            }
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                result = operation()
                
                # Reset failure count on success
                self.failure_counts[operation_name] = 0
                if attempt > 0:
                    self.resilience_metrics["recovered_operations"] += 1
                    logger.info(f"[RESILIENCE] Operation {operation_name} recovered after {attempt} attempts")
                
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "recovered": attempt > 0
                }
                
            except Exception as e:
                last_exception = e
                self.failure_counts[operation_name] = self.failure_counts.get(operation_name, 0) + 1
                
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    logger.warning(f"[RESILIENCE] Attempt {attempt + 1} failed for {operation_name}, retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"[RESILIENCE] All attempts failed for {operation_name}")
        
        # Check if circuit breaker should be triggered
        if self.failure_counts.get(operation_name, 0) >= self.circuit_breaker_threshold:
            self._trip_circuit_breaker(operation_name)
        
        self.resilience_metrics["failed_operations"] += 1
        return {
            "success": False,
            "error": str(last_exception),
            "result": None,
            "attempts": self.max_retries + 1
        }
    
    async def execute_with_resilience_async(self, operation_name: str,
                                          operation: Callable[[], Awaitable[Any]]) -> Dict[str, Any]:
        """Async version of execute_with_resilience."""
        self.resilience_metrics["total_operations"] += 1
        
        if self._is_circuit_open(operation_name):
            logger.warning(f"[RESILIENCE] Circuit breaker open for {operation_name}")
            return {
                "success": False,
                "error": "circuit_breaker_open",
                "result": None,
                "attempts": 0
            }
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await operation()
                
                self.failure_counts[operation_name] = 0
                if attempt > 0:
                    self.resilience_metrics["recovered_operations"] += 1
                    logger.info(f"[RESILIENCE] Async operation {operation_name} recovered after {attempt} attempts")
                
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "recovered": attempt > 0
                }
                
            except Exception as e:
                last_exception = e
                self.failure_counts[operation_name] = self.failure_counts.get(operation_name, 0) + 1
                
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    logger.warning(f"[RESILIENCE] Async attempt {attempt + 1} failed for {operation_name}, retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
        
        if self.failure_counts.get(operation_name, 0) >= self.circuit_breaker_threshold:
            self._trip_circuit_breaker(operation_name)
        
        self.resilience_metrics["failed_operations"] += 1
        return {
            "success": False,
            "error": str(last_exception),
            "result": None,
            "attempts": self.max_retries + 1
        }
    
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        breaker_info = self.circuit_breakers.get(operation_name)
        if not breaker_info:
            return False
        
        # Simple time-based circuit breaker (resets after 60 seconds)
        if time.time() - breaker_info["tripped_at"] > 60:
            del self.circuit_breakers[operation_name]
            self.failure_counts[operation_name] = 0
            logger.info(f"[RESILIENCE] Circuit breaker reset for {operation_name}")
            return False
        
        return True
    
    def _trip_circuit_breaker(self, operation_name: str) -> None:
        """Trip the circuit breaker for an operation."""
        self.circuit_breakers[operation_name] = {
            "tripped_at": time.time(),
            "failure_count": self.failure_counts.get(operation_name, 0)
        }
        self.resilience_metrics["circuit_breaker_trips"] += 1
        logger.error(f"[RESILIENCE] Circuit breaker tripped for {operation_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get resilience metrics."""
        return {
            **self.resilience_metrics,
            "failure_counts": dict(self.failure_counts),
            "active_circuit_breakers": list(self.circuit_breakers.keys()),
            "success_rate": (
                (self.resilience_metrics["total_operations"] - self.resilience_metrics["failed_operations"]) 
                / max(self.resilience_metrics["total_operations"], 1)
            )
        }


# --------------------------------------------------------------------------------------
# Async vs Sync Performance Benchmarking
# --------------------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    operation_type: str
    execution_mode: str  # "sync" or "async"
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    operations_count: int
    success_count: int
    error_count: int
    throughput: float  # operations per second
    concurrency_level: int = 1


class PerformanceBenchmark:
    """Benchmarking tool for comparing sync vs async performance."""
    
    def __init__(self, agent: 'AgentNet'):
        self.agent = agent
        self.benchmark_history: List[BenchmarkResult] = []
    
    def benchmark_reasoning_tree(self, tasks: List[str], concurrency_levels: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """Benchmark reasoning tree generation in both sync and async modes."""
        results = {
            "sync_results": [],
            "async_results": [],
            "comparison": {}
        }
        
        # Benchmark sync execution
        for concurrency in concurrency_levels:
            if concurrency == 1:
                # Single-threaded sync
                sync_result = self._benchmark_sync_single(tasks, "reasoning_tree")
                results["sync_results"].append(sync_result)
            else:
                # Multi-threaded sync (simulate concurrent requests)
                sync_result = self._benchmark_sync_concurrent(tasks, concurrency, "reasoning_tree")
                results["sync_results"].append(sync_result)
        
        # Benchmark async execution
        for concurrency in concurrency_levels:
            async_result = asyncio.run(self._benchmark_async(tasks, concurrency, "reasoning_tree"))
            results["async_results"].append(async_result)
        
        # Generate comparison analysis
        results["comparison"] = self._analyze_performance_comparison(
            results["sync_results"], results["async_results"]
        )
        
        return results
    
    def _benchmark_sync_single(self, tasks: List[str], operation_type: str) -> BenchmarkResult:
        """Benchmark single-threaded sync execution."""
        start_time = time.perf_counter()
        durations = []
        success_count = 0
        error_count = 0
        
        for task in tasks:
            task_start = time.perf_counter()
            try:
                self.agent.generate_reasoning_tree(task)
                success_count += 1
            except Exception as e:
                logger.warning(f"Benchmark task failed: {e}")
                error_count += 1
            task_end = time.perf_counter()
            durations.append(task_end - task_start)
        
        total_duration = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            operation_type=operation_type,
            execution_mode="sync_single",
            total_duration=total_duration,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            operations_count=len(tasks),
            success_count=success_count,
            error_count=error_count,
            throughput=success_count / total_duration if total_duration > 0 else 0,
            concurrency_level=1
        )
        
        self.benchmark_history.append(result)
        return result
    
    def _benchmark_sync_concurrent(self, tasks: List[str], concurrency: int, operation_type: str) -> BenchmarkResult:
        """Benchmark multi-threaded sync execution."""
        import concurrent.futures
        import threading
        
        start_time = time.perf_counter()
        durations = []
        success_count = 0
        error_count = 0
        durations_lock = threading.Lock()
        
        def execute_task(task: str) -> None:
            nonlocal success_count, error_count
            task_start = time.perf_counter()
            try:
                self.agent.generate_reasoning_tree(task)
                with durations_lock:
                    success_count += 1
            except Exception as e:
                logger.warning(f"Concurrent benchmark task failed: {e}")
                with durations_lock:
                    error_count += 1
            task_end = time.perf_counter()
            with durations_lock:
                durations.append(task_end - task_start)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(execute_task, task) for task in tasks]
            concurrent.futures.wait(futures)
        
        total_duration = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            operation_type=operation_type,
            execution_mode=f"sync_concurrent_{concurrency}",
            total_duration=total_duration,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            operations_count=len(tasks),
            success_count=success_count,
            error_count=error_count,
            throughput=success_count / total_duration if total_duration > 0 else 0,
            concurrency_level=concurrency
        )
        
        self.benchmark_history.append(result)
        return result
    
    async def _benchmark_async(self, tasks: List[str], concurrency: int, operation_type: str) -> BenchmarkResult:
        """Benchmark async execution."""
        start_time = time.perf_counter()
        durations = []
        success_count = 0
        error_count = 0
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def execute_task(task: str) -> None:
            nonlocal success_count, error_count
            async with semaphore:
                task_start = time.perf_counter()
                try:
                    await self.agent.async_generate_reasoning_tree(task)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Async benchmark task failed: {e}")
                    error_count += 1
                task_end = time.perf_counter()
                durations.append(task_end - task_start)
        
        await asyncio.gather(*[execute_task(task) for task in tasks])
        
        total_duration = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            operation_type=operation_type,
            execution_mode=f"async_concurrent_{concurrency}",
            total_duration=total_duration,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            operations_count=len(tasks),
            success_count=success_count,
            error_count=error_count,
            throughput=success_count / total_duration if total_duration > 0 else 0,
            concurrency_level=concurrency
        )
        
        self.benchmark_history.append(result)
        return result
    
    def _analyze_performance_comparison(self, sync_results: List[BenchmarkResult], 
                                      async_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze and compare sync vs async performance."""
        analysis = {
            "throughput_comparison": {},
            "latency_comparison": {},
            "scalability_analysis": {},
            "recommendations": []
        }
        
        # Compare throughput at different concurrency levels
        for sync_result, async_result in zip(sync_results, async_results):
            concurrency = sync_result.concurrency_level
            sync_throughput = sync_result.throughput
            async_throughput = async_result.throughput
            
            improvement = ((async_throughput - sync_throughput) / sync_throughput * 100) if sync_throughput > 0 else 0
            
            analysis["throughput_comparison"][f"concurrency_{concurrency}"] = {
                "sync_throughput": sync_throughput,
                "async_throughput": async_throughput,
                "improvement_percent": improvement
            }
            
            analysis["latency_comparison"][f"concurrency_{concurrency}"] = {
                "sync_avg_latency": sync_result.avg_duration,
                "async_avg_latency": async_result.avg_duration,
                "sync_max_latency": sync_result.max_duration,
                "async_max_latency": async_result.max_duration
            }
        
        # Scalability analysis
        sync_throughputs = [r.throughput for r in sync_results]
        async_throughputs = [r.throughput for r in async_results]
        
        analysis["scalability_analysis"] = {
            "sync_scalability": sync_throughputs[-1] / sync_throughputs[0] if sync_throughputs[0] > 0 else 0,
            "async_scalability": async_throughputs[-1] / async_throughputs[0] if async_throughputs[0] > 0 else 0,
            "async_advantage_at_scale": async_throughputs[-1] / sync_throughputs[-1] if sync_throughputs[-1] > 0 else 0
        }
        
        # Generate recommendations
        if analysis["scalability_analysis"]["async_advantage_at_scale"] > 1.5:
            analysis["recommendations"].append("Async execution shows significant advantage at high concurrency")
        
        if any(comp["improvement_percent"] > 20 for comp in analysis["throughput_comparison"].values()):
            analysis["recommendations"].append("Consider async execution for improved throughput")
        
        if analysis["scalability_analysis"]["async_scalability"] > analysis["scalability_analysis"]["sync_scalability"]:
            analysis["recommendations"].append("Async execution scales better with concurrent load")
        
        return analysis
    
    def export_benchmark_report(self, filepath: str) -> None:
        """Export benchmark results to a JSON report."""
        report = {
            "agent_name": self.agent.name,
            "timestamp": time.time(),
            "benchmark_history": [
                {
                    "operation_type": r.operation_type,
                    "execution_mode": r.execution_mode,
                    "total_duration": r.total_duration,
                    "avg_duration": r.avg_duration,
                    "min_duration": r.min_duration,
                    "max_duration": r.max_duration,
                    "operations_count": r.operations_count,
                    "success_count": r.success_count,
                    "error_count": r.error_count,
                    "throughput": r.throughput,
                    "concurrency_level": r.concurrency_level
                }
                for r in self.benchmark_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Benchmark report exported to {filepath}")


# --------------------------------------------------------------------------------------
# Analytics Index Generation
# --------------------------------------------------------------------------------------

@dataclass
class AnalyticsIndex:
    """Index structure for analytics and search."""
    session_id: str
    timestamp: float
    agents: List[str]
    topics: List[str]
    keywords: List[str]
    sentiment_score: float
    complexity_score: float
    interaction_patterns: Dict[str, Any]
    content_summary: str


class AnalyticsIndexer:
    """Generates searchable indices from agent interactions for analytics."""
    
    def __init__(self):
        self.indices: List[AnalyticsIndex] = []
        self.keyword_frequency = {}
        self.topic_clusters = {}
        self.interaction_patterns = {}
    
    def index_session(self, session_data: Dict[str, Any]) -> AnalyticsIndex:
        """Generate analytics index for a session."""
        session_id = session_data.get("session_id", f"session_{int(time.time())}")
        
        # Extract basic information
        agents = session_data.get("participants", [])
        if "agent_contributions" in session_data:
            agents.extend(session_data["agent_contributions"].keys())
        agents = list(set(agents))  # Remove duplicates
        
        # Extract topics
        topics = []
        if "topic_start" in session_data:
            topics.append(session_data["topic_start"])
        if "topic_final" in session_data and session_data["topic_final"] not in topics:
            topics.append(session_data["topic_final"])
        if "topic_evolution" in session_data:
            topics.extend([t for t in session_data["topic_evolution"] if t not in topics])
        
        # Extract content for analysis
        content_pieces = []
        if "transcript" in session_data:
            for turn in session_data["transcript"]:
                if "content" in turn:
                    content_pieces.append(turn["content"])
        elif "dialogue_history" in session_data:
            for entry in session_data["dialogue_history"]:
                if "response" in entry and "result" in entry["response"]:
                    content_pieces.append(entry["response"]["result"].get("content", ""))
        
        all_content = " ".join(content_pieces)
        
        # Generate keywords
        keywords = self._extract_keywords(all_content)
        
        # Calculate sentiment and complexity scores
        sentiment_score = self._calculate_sentiment_score(all_content)
        complexity_score = self._calculate_complexity_score(all_content, session_data)
        
        # Analyze interaction patterns
        interaction_patterns = self._analyze_interaction_patterns(session_data)
        
        # Create content summary
        content_summary = self._generate_content_summary(all_content, topics)
        
        index = AnalyticsIndex(
            session_id=session_id,
            timestamp=session_data.get("timestamp", time.time()),
            agents=agents,
            topics=topics,
            keywords=keywords,
            sentiment_score=sentiment_score,
            complexity_score=complexity_score,
            interaction_patterns=interaction_patterns,
            content_summary=content_summary
        )
        
        self.indices.append(index)
        self._update_aggregated_analytics(index)
        
        return index
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content."""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        import string
        
        # Remove punctuation and convert to lowercase
        content_clean = content.translate(str.maketrans('', '', string.punctuation)).lower()
        words = content_clean.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [keyword for keyword, count in keywords]
    
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate sentiment score (simplified implementation)."""
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'success', 'effective',
            'efficient', 'beneficial', 'advantage', 'improvement', 'optimal',
            'valuable', 'useful', 'helpful', 'innovative', 'creative'
        }
        
        negative_words = {
            'bad', 'poor', 'terrible', 'negative', 'failure', 'ineffective',
            'inefficient', 'harmful', 'disadvantage', 'problem', 'issue',
            'risk', 'danger', 'difficult', 'complex', 'challenging'
        }
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def _calculate_complexity_score(self, content: str, session_data: Dict[str, Any]) -> float:
        """Calculate complexity score based on various factors."""
        factors = []
        
        # Content length factor
        content_length = len(content.split())
        length_score = min(content_length / 1000, 1.0)  # Normalize to 0-1
        factors.append(length_score)
        
        # Number of participants factor
        participants = len(session_data.get("participants", []))
        participant_score = min(participants / 5, 1.0)  # Normalize to 0-1
        factors.append(participant_score)
        
        # Number of rounds factor
        rounds = session_data.get("rounds_executed", 1)
        rounds_score = min(rounds / 10, 1.0)  # Normalize to 0-1
        factors.append(rounds_score)
        
        # Topic evolution factor
        topic_evolution = len(session_data.get("topic_evolution", []))
        evolution_score = min(topic_evolution / 5, 1.0)  # Normalize to 0-1
        factors.append(evolution_score)
        
        return sum(factors) / len(factors)
    
    def _analyze_interaction_patterns(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interaction patterns in the session."""
        patterns = {
            "dialogue_mode": session_data.get("mode", "unknown"),
            "convergence_achieved": session_data.get("converged", False),
            "participant_balance": {},
            "round_dynamics": []
        }
        
        # Analyze participant balance
        if "agent_contributions" in session_data:
            total_contributions = sum(
                contrib.get("rounds", 0) 
                for contrib in session_data["agent_contributions"].values()
            )
            
            for agent, contrib in session_data["agent_contributions"].items():
                patterns["participant_balance"][agent] = {
                    "contribution_ratio": contrib.get("rounds", 0) / max(total_contributions, 1),
                    "insights_count": len(contrib.get("insights", []))
                }
        
        # Analyze round dynamics (if transcript available)
        if "transcript" in session_data:
            round_confidences = {}
            for turn in session_data["transcript"]:
                round_num = turn.get("round", 1)
                confidence = turn.get("confidence", 0.5)
                
                if round_num not in round_confidences:
                    round_confidences[round_num] = []
                round_confidences[round_num].append(confidence)
            
            for round_num, confidences in round_confidences.items():
                patterns["round_dynamics"].append({
                    "round": round_num,
                    "avg_confidence": sum(confidences) / len(confidences),
                    "confidence_variance": self._calculate_variance(confidences)
                })
        
        return patterns
    
    def _generate_content_summary(self, content: str, topics: List[str]) -> str:
        """Generate a concise summary of the content."""
        # Simple extractive summarization
        sentences = content.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return f"Discussion about: {', '.join(topics)}"
        
        # Score sentences based on keyword presence
        keyword_topics = ' '.join(topics).lower().split()
        scored_sentences = []
        
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            score = sum(1 for word in keyword_topics if word in sentence.lower())
            scored_sentences.append((sentence, score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]
        
        summary = '. '.join(top_sentences)
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        return summary or f"Discussion about: {', '.join(topics)}"
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _update_aggregated_analytics(self, index: AnalyticsIndex) -> None:
        """Update aggregated analytics with new index."""
        # Update keyword frequency
        for keyword in index.keywords:
            self.keyword_frequency[keyword] = self.keyword_frequency.get(keyword, 0) + 1
        
        # Update topic clusters (simple clustering by similar keywords)
        for topic in index.topics:
            if topic not in self.topic_clusters:
                self.topic_clusters[topic] = {
                    "count": 0,
                    "related_keywords": set(),
                    "avg_sentiment": 0.0,
                    "avg_complexity": 0.0
                }
            
            cluster = self.topic_clusters[topic]
            cluster["count"] += 1
            cluster["related_keywords"].update(index.keywords)
            cluster["avg_sentiment"] = (
                (cluster["avg_sentiment"] * (cluster["count"] - 1) + index.sentiment_score) 
                / cluster["count"]
            )
            cluster["avg_complexity"] = (
                (cluster["avg_complexity"] * (cluster["count"] - 1) + index.complexity_score) 
                / cluster["count"]
            )
    
    def search_sessions(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[AnalyticsIndex]:
        """Search sessions based on query and filters."""
        results = []
        query_lower = query.lower()
        filters = filters or {}
        
        for index in self.indices:
            # Text matching
            matches_query = (
                query_lower in index.content_summary.lower() or
                any(query_lower in topic.lower() for topic in index.topics) or
                any(query_lower in keyword.lower() for keyword in index.keywords)
            )
            
            if not matches_query:
                continue
            
            # Apply filters
            if "agents" in filters and not any(agent in index.agents for agent in filters["agents"]):
                continue
            
            if "min_sentiment" in filters and index.sentiment_score < filters["min_sentiment"]:
                continue
            
            if "max_complexity" in filters and index.complexity_score > filters["max_complexity"]:
                continue
            
            if "date_range" in filters:
                start, end = filters["date_range"]
                if not (start <= index.timestamp <= end):
                    continue
            
            results.append(index)
        
        return results
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        if not self.indices:
            return {"error": "No indexed sessions available"}
        
        total_sessions = len(self.indices)
        
        # Calculate aggregate metrics
        avg_sentiment = sum(idx.sentiment_score for idx in self.indices) / total_sessions
        avg_complexity = sum(idx.complexity_score for idx in self.indices) / total_sessions
        
        # Most active agents
        agent_activity = {}
        for index in self.indices:
            for agent in index.agents:
                agent_activity[agent] = agent_activity.get(agent, 0) + 1
        
        top_agents = sorted(agent_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Most discussed topics
        topic_frequency = {}
        for index in self.indices:
            for topic in index.topics:
                topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
        
        top_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top keywords
        top_keywords = sorted(self.keyword_frequency.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "summary": {
                "total_sessions": total_sessions,
                "avg_sentiment": avg_sentiment,
                "avg_complexity": avg_complexity,
                "date_range": {
                    "earliest": min(idx.timestamp for idx in self.indices),
                    "latest": max(idx.timestamp for idx in self.indices)
                }
            },
            "top_agents": top_agents,
            "top_topics": top_topics,
            "top_keywords": top_keywords,
            "topic_clusters": {
                topic: {
                    **data,
                    "related_keywords": list(data["related_keywords"])
                }
                for topic, data in self.topic_clusters.items()
            }
        }
    
    def export_indices(self, filepath: str) -> None:
        """Export analytics indices to JSON file."""
        export_data = {
            "indices": [
                {
                    "session_id": idx.session_id,
                    "timestamp": idx.timestamp,
                    "agents": idx.agents,
                    "topics": idx.topics,
                    "keywords": idx.keywords,
                    "sentiment_score": idx.sentiment_score,
                    "complexity_score": idx.complexity_score,
                    "interaction_patterns": idx.interaction_patterns,
                    "content_summary": idx.content_summary
                }
                for idx in self.indices
            ],
            "aggregated_analytics": self.generate_analytics_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Analytics indices exported to {filepath}")


# --------------------------------------------------------------------------------------
# Extensibility Experiments (Custom Monitors)
# --------------------------------------------------------------------------------------

class MonitorTemplate:
    """Base template for creating custom monitors."""
    
    def __init__(self, name: str, severity: Severity = Severity.MINOR, 
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.severity = severity
        self.config = config or {}
        self.metrics = {
            "evaluations": 0,
            "violations": 0,
            "total_time": 0.0
        }
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        """Override this method to implement custom monitoring logic."""
        raise NotImplementedError("Custom monitors must implement evaluate method")
    
    def reset_metrics(self) -> None:
        """Reset monitor metrics."""
        self.metrics = {
            "evaluations": 0,
            "violations": 0,
            "total_time": 0.0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitor performance metrics."""
        return {
            **self.metrics,
            "avg_evaluation_time": (
                self.metrics["total_time"] / max(self.metrics["evaluations"], 1)
            ),
            "violation_rate": (
                self.metrics["violations"] / max(self.metrics["evaluations"], 1)
            )
        }


class SentimentMonitor(MonitorTemplate):
    """Monitor for sentiment analysis of agent outputs."""
    
    def __init__(self, name: str = "sentiment_monitor", 
                 min_sentiment: float = 0.3, max_sentiment: float = 0.8,
                 severity: Severity = Severity.MINOR):
        super().__init__(name, severity, {
            "min_sentiment": min_sentiment,
            "max_sentiment": max_sentiment
        })
        self.min_sentiment = min_sentiment
        self.max_sentiment = max_sentiment
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        start_time = time.perf_counter()
        self.metrics["evaluations"] += 1
        
        content = outcome.get("result", {}).get("content", "")
        sentiment_score = self._analyze_sentiment(content)
        
        # Store sentiment score in outcome for potential use by other monitors
        outcome.setdefault("monitoring_data", {})["sentiment_score"] = sentiment_score
        
        passed = self.min_sentiment <= sentiment_score <= self.max_sentiment
        rationale = None
        
        if not passed:
            self.metrics["violations"] += 1
            if sentiment_score < self.min_sentiment:
                rationale = f"Sentiment too negative: {sentiment_score:.2f} < {self.min_sentiment}"
            else:
                rationale = f"Sentiment too positive: {sentiment_score:.2f} > {self.max_sentiment}"
        
        elapsed = time.perf_counter() - start_time
        self.metrics["total_time"] += elapsed
        
        return passed, rationale, elapsed
    
    def _analyze_sentiment(self, content: str) -> float:
        """Simple sentiment analysis (can be enhanced with ML models)."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'positive', 'beneficial', 'helpful', 'useful', 'effective', 'successful',
            'innovative', 'creative', 'valuable', 'optimal', 'efficient'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'harmful',
            'useless', 'ineffective', 'failed', 'problematic', 'dangerous',
            'risky', 'difficult', 'impossible', 'wrong', 'broken'
        }
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)


class ComplexityMonitor(MonitorTemplate):
    """Monitor for content complexity analysis."""
    
    def __init__(self, name: str = "complexity_monitor",
                 max_complexity: float = 0.8, severity: Severity = Severity.MINOR):
        super().__init__(name, severity, {"max_complexity": max_complexity})
        self.max_complexity = max_complexity
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        start_time = time.perf_counter()
        self.metrics["evaluations"] += 1
        
        content = outcome.get("result", {}).get("content", "")
        complexity_score = self._calculate_complexity(content)
        
        outcome.setdefault("monitoring_data", {})["complexity_score"] = complexity_score
        
        passed = complexity_score <= self.max_complexity
        rationale = None
        
        if not passed:
            self.metrics["violations"] += 1
            rationale = f"Content too complex: {complexity_score:.2f} > {self.max_complexity}"
        
        elapsed = time.perf_counter() - start_time
        self.metrics["total_time"] += elapsed
        
        return passed, rationale, elapsed
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate content complexity based on various factors."""
        if not content.strip():
            return 0.0
        
        factors = []
        
        # Sentence length factor
        sentences = content.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            length_factor = min(avg_sentence_length / 20, 1.0)  # Normalize
            factors.append(length_factor)
        
        # Vocabulary complexity factor
        words = content.lower().split()
        unique_words = set(words)
        vocab_complexity = len(unique_words) / max(len(words), 1)
        factors.append(vocab_complexity)
        
        # Technical terms factor (simple heuristic)
        technical_indicators = ['algorithm', 'framework', 'implementation', 'optimization',
                              'architecture', 'methodology', 'analysis', 'evaluation']
        technical_score = sum(1 for word in words if word in technical_indicators)
        technical_factor = min(technical_score / 5, 1.0)
        factors.append(technical_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class DomainSpecificMonitor(MonitorTemplate):
    """Monitor for domain-specific content validation."""
    
    def __init__(self, name: str, domain_keywords: List[str],
                 required_keywords: List[str] = None,
                 forbidden_keywords: List[str] = None,
                 severity: Severity = Severity.MINOR):
        super().__init__(name, severity, {
            "domain_keywords": domain_keywords,
            "required_keywords": required_keywords or [],
            "forbidden_keywords": forbidden_keywords or []
        })
        self.domain_keywords = set(word.lower() for word in domain_keywords)
        self.required_keywords = set(word.lower() for word in (required_keywords or []))
        self.forbidden_keywords = set(word.lower() for word in (forbidden_keywords or []))
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        start_time = time.perf_counter()
        self.metrics["evaluations"] += 1
        
        content = outcome.get("result", {}).get("content", "").lower()
        words = set(content.split())
        
        # Check domain relevance
        domain_matches = len(words.intersection(self.domain_keywords))
        domain_score = domain_matches / max(len(self.domain_keywords), 1)
        
        # Check required keywords
        required_matches = words.intersection(self.required_keywords)
        required_satisfied = len(required_matches) == len(self.required_keywords)
        
        # Check forbidden keywords
        forbidden_matches = words.intersection(self.forbidden_keywords)
        forbidden_violated = len(forbidden_matches) > 0
        
        # Store domain analysis in outcome
        outcome.setdefault("monitoring_data", {})["domain_analysis"] = {
            "domain_score": domain_score,
            "required_satisfied": required_satisfied,
            "forbidden_violated": forbidden_violated,
            "domain_matches": domain_matches,
            "missing_required": list(self.required_keywords - required_matches),
            "forbidden_found": list(forbidden_matches)
        }
        
        passed = required_satisfied and not forbidden_violated
        rationale = None
        
        if not passed:
            self.metrics["violations"] += 1
            reasons = []
            if not required_satisfied:
                missing = list(self.required_keywords - required_matches)
                reasons.append(f"Missing required keywords: {missing}")
            if forbidden_violated:
                reasons.append(f"Contains forbidden keywords: {list(forbidden_matches)}")
            rationale = "; ".join(reasons)
        
        elapsed = time.perf_counter() - start_time
        self.metrics["total_time"] += elapsed
        
        return passed, rationale, elapsed


class MonitorChain:
    """Chain multiple monitors together for complex evaluation."""
    
    def __init__(self, monitors: List[MonitorTemplate], 
                 chain_strategy: str = "all_pass"):  # "all_pass", "any_pass", "majority_pass"
        self.monitors = monitors
        self.chain_strategy = chain_strategy
        self.chain_metrics = {
            "evaluations": 0,
            "chain_violations": 0,
            "individual_violations": {}
        }
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """Evaluate outcome through monitor chain."""
        start_time = time.perf_counter()
        self.chain_metrics["evaluations"] += 1
        
        results = []
        violations = []
        
        for monitor in self.monitors:
            passed, rationale, _ = monitor.evaluate(outcome)
            results.append(passed)
            
            if not passed and rationale:
                violations.append(f"{monitor.name}: {rationale}")
                monitor_name = monitor.name
                self.chain_metrics["individual_violations"][monitor_name] = (
                    self.chain_metrics["individual_violations"].get(monitor_name, 0) + 1
                )
        
        # Apply chain strategy
        if self.chain_strategy == "all_pass":
            chain_passed = all(results)
        elif self.chain_strategy == "any_pass":
            chain_passed = any(results)
        elif self.chain_strategy == "majority_pass":
            chain_passed = sum(results) > len(results) / 2
        else:
            raise ValueError(f"Unknown chain strategy: {self.chain_strategy}")
        
        if not chain_passed:
            self.chain_metrics["chain_violations"] += 1
        
        elapsed = time.perf_counter() - start_time
        return chain_passed, violations, elapsed
    
    def get_chain_metrics(self) -> Dict[str, Any]:
        """Get metrics for the monitor chain."""
        individual_metrics = {
            monitor.name: monitor.get_metrics() 
            for monitor in self.monitors
        }
        
        return {
            "chain_strategy": self.chain_strategy,
            "chain_metrics": self.chain_metrics,
            "individual_metrics": individual_metrics,
            "chain_violation_rate": (
                self.chain_metrics["chain_violations"] / 
                max(self.chain_metrics["evaluations"], 1)
            )
        }


class MonitorPlugin:
    """Plugin system for loading custom monitors dynamically."""
    
    def __init__(self):
        self.registered_monitors = {}
        self.plugin_instances = {}
    
    def register_monitor_class(self, name: str, monitor_class: type) -> None:
        """Register a custom monitor class."""
        if not issubclass(monitor_class, MonitorTemplate):
            raise ValueError(f"Monitor class {monitor_class} must inherit from MonitorTemplate")
        
        self.registered_monitors[name] = monitor_class
        logger.info(f"Registered custom monitor class: {name}")
    
    def create_monitor_instance(self, name: str, config: Dict[str, Any]) -> MonitorTemplate:
        """Create an instance of a registered monitor."""
        if name not in self.registered_monitors:
            raise ValueError(f"Monitor class {name} not registered")
        
        monitor_class = self.registered_monitors[name]
        instance = monitor_class(**config)
        
        instance_id = f"{name}_{id(instance)}"
        self.plugin_instances[instance_id] = instance
        
        return instance
    
    def load_monitor_from_config(self, config: Dict[str, Any]) -> MonitorTemplate:
        """Load monitor from configuration dictionary."""
        monitor_type = config.get("type", "").lower()
        
        # Built-in monitors
        if monitor_type == "sentiment":
            return SentimentMonitor(
                name=config.get("name", "sentiment_monitor"),
                min_sentiment=config.get("min_sentiment", 0.3),
                max_sentiment=config.get("max_sentiment", 0.8),
                severity=_parse_severity(config.get("severity", "minor"))
            )
        elif monitor_type == "complexity":
            return ComplexityMonitor(
                name=config.get("name", "complexity_monitor"),
                max_complexity=config.get("max_complexity", 0.8),
                severity=_parse_severity(config.get("severity", "minor"))
            )
        elif monitor_type == "domain_specific":
            return DomainSpecificMonitor(
                name=config.get("name", "domain_monitor"),
                domain_keywords=config.get("domain_keywords", []),
                required_keywords=config.get("required_keywords", []),
                forbidden_keywords=config.get("forbidden_keywords", []),
                severity=_parse_severity(config.get("severity", "minor"))
            )
        elif monitor_type in self.registered_monitors:
            # Custom registered monitor
            monitor_config = {k: v for k, v in config.items() if k not in ["type"]}
            return self.create_monitor_instance(monitor_type, monitor_config)
        else:
            raise ValueError(f"Unknown monitor type: {monitor_type}")
    
    def get_available_monitors(self) -> Dict[str, str]:
        """Get list of available monitor types."""
        built_in = {
            "sentiment": "Sentiment analysis monitor",
            "complexity": "Content complexity monitor", 
            "domain_specific": "Domain-specific keyword monitor"
        }
        
        custom = {
            name: f"Custom monitor: {cls.__name__}" 
            for name, cls in self.registered_monitors.items()
        }
        
        return {**built_in, **custom}


# --------------------------------------------------------------------------------------
# Policy/Safety Modeling Extensions
# --------------------------------------------------------------------------------------

class PolicyLevel(str, Enum):
    """Hierarchical policy levels for organizational governance."""
    ORGANIZATIONAL = "organizational"
    TEAM = "team"
    INDIVIDUAL = "individual"
    SESSION = "session"


@dataclass
class PolicyVersion:
    """Versioned policy configuration."""
    version: str
    timestamp: float
    rules: List[ConstraintRule]
    metadata: Dict[str, Any]
    active: bool = True


class HierarchicalPolicyEngine(PolicyEngine):
    """Extended policy engine with hierarchical rule management."""
    
    def __init__(self):
        super().__init__()
        self.policy_hierarchy = {level: [] for level in PolicyLevel}
        self.policy_versions = {}
        self.policy_inheritance = {
            PolicyLevel.SESSION: [PolicyLevel.INDIVIDUAL, PolicyLevel.TEAM, PolicyLevel.ORGANIZATIONAL],
            PolicyLevel.INDIVIDUAL: [PolicyLevel.TEAM, PolicyLevel.ORGANIZATIONAL],
            PolicyLevel.TEAM: [PolicyLevel.ORGANIZATIONAL],
            PolicyLevel.ORGANIZATIONAL: []
        }
        self.policy_metrics = {
            "evaluations_by_level": {level.value: 0 for level in PolicyLevel},
            "violations_by_level": {level.value: 0 for level in PolicyLevel},
            "rule_effectiveness": {}
        }
    
    def add_policy_level(self, level: PolicyLevel, rules: List[ConstraintRule]) -> None:
        """Add rules at a specific policy level."""
        self.policy_hierarchy[level].extend(rules)
        logger.info(f"Added {len(rules)} rules at {level.value} level")
    
    def evaluate_hierarchical(self, outcome: Dict[str, Any], 
                            context_level: PolicyLevel = PolicyLevel.SESSION) -> List[Dict[str, Any]]:
        """Evaluate outcome against hierarchical policies."""
        violations = []
        
        # Collect rules from hierarchy
        applicable_rules = []
        for level in [context_level] + self.policy_inheritance[context_level]:
            applicable_rules.extend(self.policy_hierarchy[level])
        
        # Evaluate rules by level
        for level in [context_level] + self.policy_inheritance[context_level]:
            level_rules = self.policy_hierarchy[level]
            if not level_rules:
                continue
                
            self.policy_metrics["evaluations_by_level"][level.value] += 1
            
            for rule in level_rules:
                passed, rationale, exception = rule.run(outcome)
                
                # Track rule effectiveness
                rule_id = f"{level.value}:{rule.name}"
                if rule_id not in self.policy_metrics["rule_effectiveness"]:
                    self.policy_metrics["rule_effectiveness"][rule_id] = {
                        "evaluations": 0, "violations": 0, "errors": 0
                    }
                
                metrics = self.policy_metrics["rule_effectiveness"][rule_id]
                metrics["evaluations"] += 1
                
                if not passed:
                    self.policy_metrics["violations_by_level"][level.value] += 1
                    metrics["violations"] += 1
                    
                    violation = {
                        "rule_name": rule.name,
                        "policy_level": level.value,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "rationale": rationale,
                        "timestamp": time.time()
                    }
                    
                    if exception:
                        violation["exception"] = str(exception)
                        metrics["errors"] += 1
                    
                    violations.append(violation)
        
        return violations
    
    def version_policy(self, level: PolicyLevel, version_name: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a versioned snapshot of policies at a specific level."""
        version_key = f"{level.value}:{version_name}"
        
        if version_key in self.policy_versions:
            # Deactivate previous version
            self.policy_versions[version_key].active = False
        
        policy_version = PolicyVersion(
            version=version_name,
            timestamp=time.time(),
            rules=list(self.policy_hierarchy[level]),  # Create copy
            metadata=metadata or {},
            active=True
        )
        
        self.policy_versions[version_key] = policy_version
        logger.info(f"Created policy version {version_key}")
    
    def rollback_policy(self, level: PolicyLevel, version_name: str) -> bool:
        """Rollback to a specific policy version."""
        version_key = f"{level.value}:{version_name}"
        
        if version_key not in self.policy_versions:
            logger.error(f"Policy version {version_key} not found")
            return False
        
        policy_version = self.policy_versions[version_key]
        self.policy_hierarchy[level] = list(policy_version.rules)
        policy_version.active = True
        
        logger.info(f"Rolled back to policy version {version_key}")
        return True
    
    def get_policy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive policy metrics."""
        total_evaluations = sum(self.policy_metrics["evaluations_by_level"].values())
        total_violations = sum(self.policy_metrics["violations_by_level"].values())
        
        rule_effectiveness_summary = {}
        for rule_id, metrics in self.policy_metrics["rule_effectiveness"].items():
            if metrics["evaluations"] > 0:
                rule_effectiveness_summary[rule_id] = {
                    **metrics,
                    "violation_rate": metrics["violations"] / metrics["evaluations"],
                    "error_rate": metrics["errors"] / metrics["evaluations"]
                }
        
        return {
            "overall_metrics": {
                "total_evaluations": total_evaluations,
                "total_violations": total_violations,
                "overall_violation_rate": total_violations / max(total_evaluations, 1)
            },
            "level_metrics": self.policy_metrics,
            "rule_effectiveness": rule_effectiveness_summary,
            "active_versions": {
                key: version.version for key, version in self.policy_versions.items() 
                if version.active
            }
        }


class PolicyABTesting:
    """A/B testing framework for policy configurations."""
    
    def __init__(self):
        self.experiments = {}
        self.experiment_results = {}
    
    def create_experiment(self, experiment_id: str, 
                         policy_a: List[ConstraintRule], 
                         policy_b: List[ConstraintRule],
                         traffic_split: float = 0.5,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a new A/B test experiment."""
        self.experiments[experiment_id] = {
            "policy_a": policy_a,
            "policy_b": policy_b,
            "traffic_split": traffic_split,
            "metadata": metadata or {},
            "created_at": time.time(),
            "active": True
        }
        
        self.experiment_results[experiment_id] = {
            "a_evaluations": 0,
            "b_evaluations": 0,
            "a_violations": 0,
            "b_violations": 0,
            "a_performance": [],
            "b_performance": []
        }
        
        logger.info(f"Created A/B test experiment: {experiment_id}")
    
    def evaluate_with_experiment(self, experiment_id: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate outcome using A/B test policies."""
        if experiment_id not in self.experiments or not self.experiments[experiment_id]["active"]:
            raise ValueError(f"Experiment {experiment_id} not found or inactive")
        
        experiment = self.experiments[experiment_id]
        results = self.experiment_results[experiment_id]
        
        # Determine which policy to use (simple hash-based split)
        import hashlib
        hash_input = f"{experiment_id}:{outcome.get('agent', '')}:{time.time()}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        use_policy_a = (hash_value % 100) < (experiment["traffic_split"] * 100)
        
        start_time = time.perf_counter()
        
        if use_policy_a:
            policy_rules = experiment["policy_a"]
            results["a_evaluations"] += 1
            policy_variant = "A"
        else:
            policy_rules = experiment["policy_b"]
            results["b_evaluations"] += 1
            policy_variant = "B"
        
        # Evaluate using selected policy
        violations = []
        for rule in policy_rules:
            passed, rationale, exception = rule.run(outcome)
            if not passed:
                violations.append({
                    "rule_name": rule.name,
                    "severity": rule.severity.value,
                    "rationale": rationale,
                    "exception": str(exception) if exception else None
                })
        
        evaluation_time = time.perf_counter() - start_time
        
        # Record results
        if use_policy_a:
            results["a_violations"] += len(violations)
            results["a_performance"].append(evaluation_time)
        else:
            results["b_violations"] += len(violations)
            results["b_performance"].append(evaluation_time)
        
        return {
            "experiment_id": experiment_id,
            "policy_variant": policy_variant,
            "violations": violations,
            "evaluation_time": evaluation_time,
            "passed": len(violations) == 0
        }
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test experiment results."""
        if experiment_id not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = self.experiment_results[experiment_id]
        experiment = self.experiments[experiment_id]
        
        # Calculate statistics
        a_violation_rate = (
            results["a_violations"] / max(results["a_evaluations"], 1)
        )
        b_violation_rate = (
            results["b_violations"] / max(results["b_evaluations"], 1)
        )
        
        a_avg_performance = (
            sum(results["a_performance"]) / max(len(results["a_performance"]), 1)
        )
        b_avg_performance = (
            sum(results["b_performance"]) / max(len(results["b_performance"]), 1)
        )
        
        # Statistical significance (simple t-test approximation)
        significance_threshold = 0.05
        sample_size_adequate = (
            results["a_evaluations"] >= 30 and results["b_evaluations"] >= 30
        )
        
        return {
            "experiment_id": experiment_id,
            "experiment_metadata": experiment["metadata"],
            "duration_hours": (time.time() - experiment["created_at"]) / 3600,
            "traffic_split": experiment["traffic_split"],
            "sample_sizes": {
                "policy_a": results["a_evaluations"],
                "policy_b": results["b_evaluations"]
            },
            "violation_rates": {
                "policy_a": a_violation_rate,
                "policy_b": b_violation_rate,
                "difference": abs(a_violation_rate - b_violation_rate),
                "winner": "A" if a_violation_rate < b_violation_rate else "B"
            },
            "performance": {
                "policy_a_avg_time": a_avg_performance,
                "policy_b_avg_time": b_avg_performance,
                "performance_winner": "A" if a_avg_performance < b_avg_performance else "B"
            },
            "statistical_confidence": {
                "sample_size_adequate": sample_size_adequate,
                "significant_difference": sample_size_adequate and abs(a_violation_rate - b_violation_rate) > significance_threshold
            }
        }
    
    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Stop an A/B test experiment and return final results."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["active"] = False
            logger.info(f"Stopped A/B test experiment: {experiment_id}")
            return self.get_experiment_results(experiment_id)
        else:
            raise ValueError(f"Experiment {experiment_id} not found")


class SafetyImpactAnalyzer:
    """Analyzer for measuring policy impact on safety and performance."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.policy_impacts = {}
        self.safety_incidents = []
    
    def establish_baseline(self, baseline_id: str, metrics: Dict[str, float]) -> None:
        """Establish baseline metrics for comparison."""
        self.baseline_metrics[baseline_id] = {
            **metrics,
            "timestamp": time.time()
        }
        logger.info(f"Established safety baseline: {baseline_id}")
    
    def record_policy_impact(self, policy_id: str, outcome: Dict[str, Any], 
                           violations: List[Dict[str, Any]]) -> None:
        """Record the impact of a policy on an outcome."""
        if policy_id not in self.policy_impacts:
            self.policy_impacts[policy_id] = {
                "total_evaluations": 0,
                "total_violations": 0,
                "severe_violations": 0,
                "performance_impact": [],
                "safety_improvements": 0,
                "false_positives": 0
            }
        
        impact = self.policy_impacts[policy_id]
        impact["total_evaluations"] += 1
        impact["total_violations"] += len(violations)
        
        # Count severe violations
        severe_count = sum(1 for v in violations if v.get("severity") == "severe")
        impact["severe_violations"] += severe_count
        
        # Record performance impact
        runtime = outcome.get("runtime_seconds", 0)
        impact["performance_impact"].append(runtime)
        
        # Detect potential safety incidents
        if severe_count > 0:
            incident = {
                "timestamp": time.time(),
                "policy_id": policy_id,
                "severity": "high" if severe_count > 2 else "medium",
                "violation_count": len(violations),
                "severe_violation_count": severe_count,
                "outcome_summary": outcome.get("result", {}).get("content", "")[:200]
            }
            self.safety_incidents.append(incident)
    
    def analyze_safety_trends(self, time_window_hours: float = 24) -> Dict[str, Any]:
        """Analyze safety trends over a specified time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_incidents = [
            incident for incident in self.safety_incidents 
            if incident["timestamp"] > cutoff_time
        ]
        
        # Analyze trends
        incident_counts = {}
        severity_distribution = {"high": 0, "medium": 0, "low": 0}
        
        for incident in recent_incidents:
            policy_id = incident["policy_id"]
            incident_counts[policy_id] = incident_counts.get(policy_id, 0) + 1
            severity_distribution[incident["severity"]] += 1
        
        # Calculate trend direction (simplified)
        if len(recent_incidents) >= 2:
            mid_point = len(recent_incidents) // 2
            first_half = recent_incidents[:mid_point]
            second_half = recent_incidents[mid_point:]
            
            trend = "increasing" if len(second_half) > len(first_half) else "decreasing"
        else:
            trend = "stable"
        
        return {
            "time_window_hours": time_window_hours,
            "total_incidents": len(recent_incidents),
            "trend": trend,
            "severity_distribution": severity_distribution,
            "incidents_by_policy": incident_counts,
            "recommendations": self._generate_safety_recommendations(recent_incidents)
        }
    
    def _generate_safety_recommendations(self, incidents: List[Dict[str, Any]]) -> List[str]:
        """Generate safety recommendations based on incident analysis."""
        recommendations = []
        
        if not incidents:
            recommendations.append("No recent safety incidents detected")
            return recommendations
        
        # High-severity incident recommendations
        high_severity_count = sum(1 for i in incidents if i["severity"] == "high")
        if high_severity_count > 0:
            recommendations.append(f"Review {high_severity_count} high-severity incidents immediately")
        
        # Policy-specific recommendations
        policy_counts = {}
        for incident in incidents:
            policy_id = incident["policy_id"]
            policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
        
        max_incidents = max(policy_counts.values()) if policy_counts else 0
        if max_incidents > 5:
            problematic_policies = [
                policy for policy, count in policy_counts.items() 
                if count == max_incidents
            ]
            recommendations.append(f"Review policies with high incident rates: {problematic_policies}")
        
        # Trend-based recommendations
        if len(incidents) > 10:
            recommendations.append("Consider tightening safety policies due to high incident volume")
        
        return recommendations
    
    def export_safety_report(self, filepath: str) -> None:
        """Export comprehensive safety analysis report."""
        report = {
            "generated_at": time.time(),
            "baseline_metrics": self.baseline_metrics,
            "policy_impacts": {
                policy_id: {
                    **impact,
                    "avg_performance_impact": (
                        sum(impact["performance_impact"]) / 
                        max(len(impact["performance_impact"]), 1)
                    ),
                    "violation_rate": impact["total_violations"] / max(impact["total_evaluations"], 1),
                    "severe_violation_rate": impact["severe_violations"] / max(impact["total_evaluations"], 1)
                }
                for policy_id, impact in self.policy_impacts.items()
            },
            "recent_safety_trends": self.analyze_safety_trends(),
            "all_incidents": self.safety_incidents
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Safety analysis report exported to {filepath}")


# --------------------------------------------------------------------------------------
# Demo Utilities (rebranded directories)
# --------------------------------------------------------------------------------------

def _write_demo_files(tmpdir: Path) -> Dict[str, Path]:
    agent_yaml = tmpdir / "agent.yaml"
    monitors_yaml = tmpdir / "monitors.yaml"
    rules_yaml = tmpdir / "rules.yaml"

    if yaml is None:
        agent_json = tmpdir / "agent.json"
        agent_json.write_text(json.dumps({
            "name": "Athena",
            "style": {"logic": 0.9, "creativity": 0.4, "analytical": 0.8},
            "monitors": {"file": "monitors.json"},
        }, indent=2))
        (tmpdir / "monitors.json").write_text(json.dumps({
            "monitors": [
                {
                    "name": "policy_rcd",
                    "type": "rcd_policy",
                    "severity": "severe",
                    "description": "Policy rules check on final outcome",
                    "params": {"rules_file": str((tmpdir / "rules.json").resolve())}
                },
                {
                    "name": "keyword_guard",
                    "type": "keyword",
                    "severity": "severe",
                    "description": "Block certain tokens",
                    "params": {"keywords": ["manipulation", "hate", "racist"]}
                },
            ]
        }, indent=2))
        (tmpdir / "rules.json").write_text(json.dumps({
            "rules": [
                {
                    "name": "no_manipulation",
                    "severity": "severe",
                    "description": "Blocks manipulation",
                    "type": "keyword",
                    "keywords": ["manipulate", "manipulation", "gaslight", "coerce", "deceive", "exploit"]
                }
            ]
        }, indent=2))
        return {"agent": agent_json}

    agent_yaml.write_text(
        """
name: Athena
style:
  logic: 0.9
  creativity: 0.4
  analytical: 0.8
monitors:
  file: monitors.yaml
dialogue_config:
  default_rounds: 4
  max_rounds: 25
  convergence_window: 4
  convergence_min_overlap: 0.5
  convergence_min_conf: 0.75
  memory_guard:
    max_transcript_tokens: 6000
    truncate_strategy: head
""".strip()
    )
    monitors_yaml.write_text(
        f"""
monitors:
  - name: policy_rcd
    type: rcd_policy
    severity: severe
    description: Policy rules check on final outcome
    params:
      rules_file: {rules_yaml.resolve()}
  - name: keyword_guard
    type: keyword
    severity: severe
    description: Block certain tokens
    params:
      keywords: ["manipulation", "hate", "racist"]
  - name: resource_guard
    type: resource
    severity: minor
    description: Flag overruns beyond 20%
    params:
      budget_key: resource_budget
      tolerance: 0.2
""".strip()
    )
    rules_yaml.write_text(
        """
rules:
  - name: no_self_harm
    severity: severe
    description: Reject outcomes promoting self-harm
    type: keyword
    keywords: ["self-harm", "suicide"]
  - name: no_manipulation
    severity: severe
    description: Blocks manipulation
    type: keyword
    keywords: ["manipulate", "manipulation", "gaslight", "coerce", "deceive", "exploit"]
  - name: applied_ethics
    severity: minor
    description: Flags controversial moral issues for review
    type: custom
    func: applied_ethics_check
""".strip()
    )
    return {"agent": agent_yaml}


def _demo_sync(rounds: int = 3, skip_multiparty: bool = False, log_dir: str = "./sessions") -> None:
    tmpdir = Path("./_agentnet_demo_sync")
    tmpdir.mkdir(exist_ok=True)
    files = _write_demo_files(tmpdir)
    engine = ExampleEngine()
    agent = AgentNet.from_config(files["agent"], engine=engine)
    agent.style.setdefault("resource_budget", 0.01)

    print("\n--- SYNC SAFE RUN ---")
    out = agent.generate_reasoning_tree("Intro: collaborative planning", include_monitor_trace=True)
    print(json.dumps(out, indent=2))
    
    if not skip_multiparty:
        print(f"\n--- SYNC MULTI-PARTY (brainstorm, {rounds} rounds) ---")
        agent_b = AgentNet("Apollo", {"logic": 0.7, "creativity": 0.85, "analytical": 0.5}, engine=engine, monitors=agent.monitors)
        agent_c = AgentNet("Hermes", {"logic": 0.55, "creativity": 0.9, "analytical": 0.6}, engine=engine, monitors=agent.monitors)
        session = agent.brainstorm([agent, agent_b, agent_c], "Designing resilient edge network", rounds=rounds)
        print(f"Converged: {session['converged']} | Final topic: {session['topic_final']}")
        saved_path = agent.persist_session(session, log_dir)
        print(f"Session saved to: {saved_path}")


async def _demo_async(rounds: int = 4, parallel: bool = False, skip_multiparty: bool = False, log_dir: str = "./sessions") -> None:
    tmpdir = Path("./_agentnet_demo_async")
    tmpdir.mkdir(exist_ok=True)
    files = _write_demo_files(tmpdir)
    engine = ExampleEngine()
    agent = AgentNet.from_config(files["agent"], engine=engine)
    agent.style.setdefault("resource_budget", 0.02)

    print("\n--- ASYNC SAFE RUN ---")
    out = await agent.async_generate_reasoning_tree("Async: strategic planning iteration", include_monitor_trace=True)
    print(json.dumps(out, indent=2))

    if not skip_multiparty:
        print(f"\n--- ASYNC MULTI-PARTY ({'parallel' if parallel else 'sequential'} debate, {rounds} rounds) ---")
        agent_b = AgentNet("ApolloAsync", {"logic": 0.8, "creativity": 0.6, "analytical": 0.55}, engine=engine, monitors=agent.monitors)
        agent_c = AgentNet("HermesAsync", {"logic": 0.5, "creativity": 0.9, "analytical": 0.6}, engine=engine, monitors=agent.monitors)
        session = await agent.async_debate(
            [agent, agent_b, agent_c],
            "Ethical frameworks for autonomous swarms",
            rounds=rounds,
            parallel_round=parallel
        )
        print(f"Async Converged: {session['converged']} | Rounds executed: {session['rounds_executed']}")
        saved_path = agent.persist_session(session, log_dir)
        print(f"Session saved to: {saved_path}")


def _demo_both(rounds: int = 3, parallel: bool = False, skip_multiparty: bool = False, log_dir: str = "./sessions") -> None:
    _demo_sync(rounds, skip_multiparty, log_dir)
    asyncio.run(_demo_async(rounds, parallel, skip_multiparty, log_dir))


def _demo_experimental_features(log_dir: str = "./sessions") -> None:
    """Demonstrate all experimental features."""
    print("\n" + "="*60)
    print("EXPERIMENTAL FEATURES DEMONSTRATION")
    print("="*60)
    
    # Setup
    engine = ExampleEngine()
    tmpdir = Path("./_agentnet_experimental_demo")
    tmpdir.mkdir(exist_ok=True)
    files = _write_demo_files(tmpdir)
    
    # 1. Fault Injection & Resilience
    print("\n--- 1. FAULT INJECTION & RESILIENCE ---")
    agent = AgentNet.from_config(files["agent"], engine=engine)
    
    # Test basic resilience
    result = agent.generate_reasoning_tree_with_resilience("Test resilience patterns")
    print(f"Resilience test result: {result.get('result', {}).get('content', '')[:100]}...")
    
    # Show resilience metrics
    metrics = agent.get_resilience_metrics()
    print(f"Resilience metrics: {metrics['success_rate']:.2f} success rate")
    
    # 2. Performance Benchmarking
    print("\n--- 2. ASYNC vs SYNC PERFORMANCE BENCHMARKING ---")
    benchmark = PerformanceBenchmark(agent)
    
    # Simple benchmark test
    test_tasks = [
        "Analyze distributed systems",
        "Design microservices architecture", 
        "Evaluate cloud deployment strategies"
    ]
    
    print("Running performance benchmark...")
    results = benchmark.benchmark_reasoning_tree(test_tasks, concurrency_levels=[1, 3])
    
    # Show comparison
    comparison = results["comparison"]
    print(f"Throughput improvement at concurrency 3: {comparison['throughput_comparison']['concurrency_3']['improvement_percent']:.1f}%")
    print(f"Async scalability advantage: {comparison['scalability_analysis']['async_advantage_at_scale']:.2f}x")
    
    # 3. Analytics Index Generation
    print("\n--- 3. ANALYTICS INDEX GENERATION ---")
    indexer = AnalyticsIndexer()
    
    # Create a sample session for indexing
    sample_session = {
        "session_id": "demo_session_123",
        "participants": ["TestAgent", "AnalyticsAgent"],
        "topic_start": "AI system optimization",
        "topic_final": "Advanced optimization strategies",
        "transcript": [
            {
                "round": 1,
                "agent": "TestAgent", 
                "content": "We should focus on performance optimization and scalability improvements for AI systems."
            },
            {
                "round": 1,
                "agent": "AnalyticsAgent",
                "content": "Advanced caching mechanisms and distributed processing would enhance system efficiency."
            }
        ]
    }
    
    # Index the session
    index = indexer.index_session(sample_session)
    print(f"Indexed session: {index.session_id}")
    print(f"Extracted keywords: {index.keywords[:5]}")
    print(f"Sentiment score: {index.sentiment_score:.2f}")
    print(f"Complexity score: {index.complexity_score:.2f}")
    
    # Generate analytics report
    report = indexer.generate_analytics_report()
    print(f"Analytics summary: {report['summary']['total_sessions']} sessions analyzed")
    
    # 4. Custom Monitors (Extensibility)
    print("\n--- 4. EXTENSIBILITY: CUSTOM MONITORS ---")
    plugin = MonitorPlugin()
    
    # Create sentiment monitor
    sentiment_monitor = plugin.load_monitor_from_config({
        "type": "sentiment",
        "name": "demo_sentiment",
        "min_sentiment": 0.4,
        "max_sentiment": 0.9
    })
    
    # Create complexity monitor  
    complexity_monitor = plugin.load_monitor_from_config({
        "type": "complexity",
        "name": "demo_complexity",
        "max_complexity": 0.7
    })
    
    # Create domain-specific monitor
    domain_monitor = plugin.load_monitor_from_config({
        "type": "domain_specific",
        "name": "demo_domain",
        "domain_keywords": ["optimization", "performance", "scalability", "efficiency"],
        "required_keywords": ["performance"],
        "forbidden_keywords": ["hack", "bypass"]
    })
    
    # Test monitors
    test_outcome = {
        "result": {
            "content": "Performance optimization requires careful analysis of scalability bottlenecks."
        }
    }
    
    sentiment_passed, sentiment_reason, _ = sentiment_monitor.evaluate(test_outcome)
    complexity_passed, complexity_reason, _ = complexity_monitor.evaluate(test_outcome)
    domain_passed, domain_reason, _ = domain_monitor.evaluate(test_outcome)
    
    print(f"Sentiment monitor: {'PASS' if sentiment_passed else 'FAIL'}")
    print(f"Complexity monitor: {'PASS' if complexity_passed else 'FAIL'}")
    print(f"Domain monitor: {'PASS' if domain_passed else 'FAIL'}")
    
    # Create monitor chain
    monitor_chain = MonitorChain([sentiment_monitor, complexity_monitor, domain_monitor])
    chain_passed, violations, _ = monitor_chain.evaluate(test_outcome)
    print(f"Monitor chain result: {'PASS' if chain_passed else 'FAIL'} with {len(violations)} violations")
    
    # 5. Policy/Safety Modeling Extensions
    print("\n--- 5. POLICY/SAFETY MODELING EXTENSIONS ---")
    
    # Hierarchical policy engine
    hierarchical_policy = HierarchicalPolicyEngine()
    
    # Add organizational-level policy
    org_rule = ConstraintRule(
        name="no_inappropriate_content",
        check=lambda outcome: "inappropriate" not in outcome.get("result", {}).get("content", "").lower(),
        severity=Severity.SEVERE,
        description="Organizational policy against inappropriate content"
    )
    hierarchical_policy.add_policy_level(PolicyLevel.ORGANIZATIONAL, [org_rule])
    
    # Add team-level policy
    team_rule = ConstraintRule(
        name="technical_accuracy",
        check=lambda outcome: len(outcome.get("result", {}).get("content", "")) > 20,
        severity=Severity.MINOR,
        description="Team policy requiring substantial technical content"
    )
    hierarchical_policy.add_policy_level(PolicyLevel.TEAM, [team_rule])
    
    # Test hierarchical evaluation
    violations = hierarchical_policy.evaluate_hierarchical(test_outcome, PolicyLevel.SESSION)
    print(f"Hierarchical policy evaluation: {len(violations)} violations found")
    
    # Policy A/B testing
    ab_testing = PolicyABTesting()
    
    # Create A/B test with different policies
    policy_a = [org_rule]
    policy_b = [org_rule, team_rule]
    
    ab_testing.create_experiment("policy_test_1", policy_a, policy_b, traffic_split=0.5)
    
    # Test A/B evaluation
    ab_result = ab_testing.evaluate_with_experiment("policy_test_1", test_outcome)
    print(f"A/B test used policy {ab_result['policy_variant']} with {len(ab_result['violations'])} violations")
    
    # Safety impact analysis
    safety_analyzer = SafetyImpactAnalyzer()
    safety_analyzer.establish_baseline("demo_baseline", {"violation_rate": 0.1, "performance": 0.5})
    safety_analyzer.record_policy_impact("demo_policy", test_outcome, violations)
    
    trends = safety_analyzer.analyze_safety_trends(time_window_hours=1)
    print(f"Safety analysis: {trends['total_incidents']} incidents, trend: {trends['trend']}")
    
    # Export reports
    print("\n--- EXPORTING REPORTS ---")
    Path(log_dir).mkdir(exist_ok=True)
    
    # Export benchmark report
    benchmark.export_benchmark_report(f"{log_dir}/benchmark_report.json")
    print(f"✓ Benchmark report exported")
    
    # Export analytics indices
    indexer.export_indices(f"{log_dir}/analytics_indices.json")
    print(f"✓ Analytics indices exported")
    
    # Export safety report
    safety_analyzer.export_safety_report(f"{log_dir}/safety_report.json")
    print(f"✓ Safety report exported")
    
    print(f"\n✓ All experimental features demonstrated successfully!")
    print(f"Reports saved to: {log_dir}/")
    
    # Cleanup
    import shutil
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentNet - Multi-agent dialogue system with advanced reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--demo",
        choices=["sync", "async", "both", "experimental"],
        default="both",
        help="Which demo to run (experimental showcases all new features)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds for multi-party examples"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel_round=True in async multi-party demo"
    )
    parser.add_argument(
        "--no-multiparty",
        action="store_true",
        help="Skip multi-party portion, run only single reasoning tree demo"
    )
    parser.add_argument(
        "--log-dir",
        default="./sessions",
        help="Directory in which to persist session logs"
    )
    
    args = parser.parse_args()
    Path(args.log_dir).mkdir(exist_ok=True)
    
    try:
        if args.demo == "sync":
            _demo_sync(args.rounds, args.no_multiparty, args.log_dir)
        elif args.demo == "async":
            asyncio.run(_demo_async(args.rounds, args.parallel, args.no_multiparty, args.log_dir))
        elif args.demo == "experimental":
            _demo_experimental_features(args.log_dir)
        else:
            _demo_both(args.rounds, args.parallel, args.no_multiparty, args.log_dir)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def _demo() -> None:
    """Legacy convenience entrypoint now routed to AgentNet demos."""
    _demo_both()


if __name__ == "__main__":
    main()
