from __future__ import annotations

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
# Logging
# --------------------------------------------------------------------------------------
logger = logging.getLogger("duetmind")
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
# Monitor system (unified violations & rationale)
# --------------------------------------------------------------------------------------

MonitorFn = Callable[["DuetMindAgent", str, Dict[str, Any]], None]


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
    def _handle(spec: MonitorSpec, agent: "DuetMindAgent", task: str,
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
        violation_name = params.get("violation_name", spec.name)
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
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
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
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
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
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
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
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
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
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
# DuetMindAgent (with advanced & async dialogue)
# --------------------------------------------------------------------------------------

class DuetMindAgent:
    """A cognitive agent with style, reasoning, persistence, pluggable monitors, and async dialogue."""

    def __init__(
        self,
        name: str,
        style: Dict[str, float],
        engine=None,
        monitors: Optional[List[MonitorFn]] = None,
        dialogue_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.style = style
        self.engine = engine
        self.monitors = monitors or []
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
        logger.info(
            f"Agent '{name}' initialized with style {style}, "
            f"{len(self.monitors)} monitors, dialogue config={self.dialogue_config}"
        )

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
        try:
            for monitor in self.monitors:
                before = time.perf_counter()
                monitor(self, task, base)
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
        """
        Async variant:
          - Uses engine.safe_think_async if present
          - Else offloads engine.safe_think to a thread executor
          - Monitors run synchronously (can be upgraded later)
        """
        loop = loop or asyncio.get_event_loop()
        start = time.perf_counter()

        async_engine_func = getattr(self.engine, "safe_think_async", None) if self.engine else None
        try:
            if async_engine_func and inspect.iscoroutinefunction(async_engine_func):
                raw = await async_engine_func(self.name, task)
            else:
                # Offload sync think
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
            # Run monitors sequentially (could parallelize if needed)
            for monitor in self.monitors:
                before = time.perf_counter()
                # Offload monitor since they are sync
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
    def dialogue_with(self, other_agent: 'DuetMindAgent', topic: str, rounds: int = 3) -> Dict[str, Any]:
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
        agents: List['DuetMindAgent'],
        topic: str,
        rounds: int = 5,
        strategy: str = "round_robin",
        mode: str = "general",
        summarizer: Optional['DuetMindAgent'] = None,
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

            if convergence and self._check_convergence(last_contents):
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
        agents: List['DuetMindAgent'],
        topic: str,
        rounds: int = 5,
        strategy: str = "round_robin",
        mode: str = "general",
        summarizer: Optional['DuetMindAgent'] = None,
        convergence: bool = True,
        callbacks: Optional[Dict[str, Callable[..., Awaitable[None] | None]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallel_round: bool = False,
        include_monitor_trace: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, Any]:
        """
        Async multi-party dialogue.
          - parallel_round=True => each round's agent turns executed concurrently
          - summarizer may also be async
        """
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

            async def run_turn(ag: DuetMindAgent) -> Dict[str, Any]:
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
                tasks = [asyncio.create_task(run_turn(ag)) for ag in ordered_agents]
                turn_payloads = await asyncio.gather(*tasks)
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

            if convergence and self._check_convergence(last_contents):
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
    def debate(self, agents: List['DuetMindAgent'], topic: str, rounds: int = 6, **kwargs) -> Dict[str, Any]:
        return self.multi_party_dialogue(agents, topic, rounds=rounds, mode="debate", **kwargs)

    def brainstorm(self, agents: List['DuetMindAgent'], topic: str, rounds: int = 5, **kwargs) -> Dict[str, Any]:
        return self.multi_party_dialogue(agents, topic, rounds=rounds, mode="brainstorm", **kwargs)

    def consensus(self, agents: List['DuetMindAgent'], topic: str, rounds: int = 7, **kwargs) -> Dict[str, Any]:
        return self.multi_party_dialogue(agents, topic, rounds=rounds, mode="consensus", **kwargs)

    # ------------- Dialogue Mode Convenience Wrappers (Async) -------------
    async def async_debate(self, agents: List['DuetMindAgent'], topic: str, rounds: int = 6, **kwargs) -> Dict[str, Any]:
        return await self.async_multi_party_dialogue(agents, topic, rounds=rounds, mode="debate", **kwargs)

    async def async_brainstorm(self, agents: List['DuetMindAgent'], topic: str, rounds: int = 5, **kwargs) -> Dict[str, Any]:
        return await self.async_multi_party_dialogue(agents, topic, rounds=rounds, mode="brainstorm", **kwargs)

    async def async_consensus(self, agents: List['DuetMindAgent'], topic: str, rounds: int = 7, **kwargs) -> Dict[str, Any]:
        return await self.async_multi_party_dialogue(agents, topic, rounds=rounds, mode="consensus", **kwargs)

    # ------------- Internal Helpers (Shared) -------------
    def _compose_turn_prompt(
        self,
        agent: 'DuetMindAgent',
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

    def _role_hint(self, agent: 'DuetMindAgent', mode: str) -> str:
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

    def _rolling_synthesis(self, summarizer: 'DuetMindAgent', transcript: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        tail = transcript[-4:]
        synthesis_prompt = (
            f"Produce a concise rolling synthesis for topic '{topic}'. "
            "Focus on new distinct points, maintain brevity."
        )
        tail_text = "\n".join(f"{t['agent']}@r{t['round']}: {t['content'][:160]}" for t in tail)
        synth = summarizer.generate_reasoning_tree(f"{synthesis_prompt}\nRecent Turns:\n{tail_text}")
        return {"content": synth.get("result", {}).get("content", ""), "confidence": synth.get("result", {}).get("confidence", 0.5)}

    async def _async_rolling_synthesis(self, summarizer: 'DuetMindAgent', transcript: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
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
        summarizer: 'DuetMindAgent',
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
        summarizer: 'DuetMindAgent',
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

    def _check_convergence(self, last_contents: List[str]) -> bool:
        window = self.dialogue_config.get("convergence_window", 3)
        if len(last_contents) < window:
            return False
        recent = last_contents[-window:]
        sets = [set(self._tokenize(c)) for c in recent]
        if not sets:
            return False
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        if not union:
            return False
        overlap = len(intersection) / max(1, len(union))
        if overlap >= self.dialogue_config.get("convergence_min_overlap", 0.55):
            return True
        return False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2]

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Simple token counting based on whitespace and punctuation splitting.
        This approximates OpenAI-style tokenization for budget management.
        """
        if not text:
            return 0
        # Simple approximation: split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)

    def _truncate_transcript(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Truncate transcript based on memory_guard configuration to stay within token budget.
        
        Args:
            transcript: List of dialogue turns
            
        Returns:
            Truncated transcript
        """
        memory_config = self.dialogue_config.get("memory_guard", {})
        max_tokens = memory_config.get("max_transcript_tokens", 5000)
        truncate_strategy = memory_config.get("truncate_strategy", "head")
        
        if not transcript:
            return transcript
            
        # Calculate current token count
        current_tokens = 0
        for turn in transcript:
            content = turn.get("content", "")
            prompt = turn.get("prompt", "")
            current_tokens += self._count_tokens(content) + self._count_tokens(prompt)
        
        if current_tokens <= max_tokens:
            return transcript
            
        logger.info(f"Transcript has {current_tokens} tokens, truncating to {max_tokens} using {truncate_strategy} strategy")
        
        if truncate_strategy == "head":
            # Keep newer turns, remove older ones
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
            # Keep older turns, remove newer ones
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
            # Default to head strategy
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
        """
        Persists a dialogue session record to a JSON file in the specified directory.
        
        Args:
            session_record: Dictionary containing session data (from multi_party_dialogue)
            directory: Directory path where to save session logs (default: "sessions")
            
        Returns:
            str: Full path to the saved session file
        """
        sessions_dir = Path(directory)
        sessions_dir.mkdir(exist_ok=True)
        
        session_id = session_record.get("session_id", f"session_{int(time.time()*1000)}")
        timestamp = session_record.get("timestamp", time.time())
        
        # Create filename with timestamp for easy sorting
        filename = f"{session_id}_{int(timestamp)}.json"
        filepath = sessions_dir / filename
        
        # Add metadata about persistence
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
    ) -> 'DuetMindAgent':
        state = json.loads(Path(path).read_text())
        agent = cls(state["name"], state.get("style", {}), engine=engine,
                    monitors=monitors, dialogue_config=state.get("dialogue_config"))
        agent.knowledge_graph = state.get("knowledge_graph", {})
        agent.interaction_history = state.get("interaction_history", [])
        logger.info(f"Agent state loaded from {path}")
        return agent

    @staticmethod
    def from_config(config_path: str | Path, engine=None) -> 'DuetMindAgent':
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
        return DuetMindAgent(name=name, style=style, engine=engine,
                             monitors=monitors, dialogue_config=dialogue_cfg)


# --------------------------------------------------------------------------------------
# Minimal Example Engine (sync + async)
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
# Demo Utilities
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
    tmpdir = Path("./_duetmind_demo_sync")
    tmpdir.mkdir(exist_ok=True)
    files = _write_demo_files(tmpdir)
    engine = ExampleEngine()
    agent = DuetMindAgent.from_config(files["agent"], engine=engine)
    agent.style.setdefault("resource_budget", 0.01)

    print("\n--- SYNC SAFE RUN ---")
    out = agent.generate_reasoning_tree("Intro: collaborative planning", include_monitor_trace=True)
    print(json.dumps(out, indent=2))
    
    if not skip_multiparty:
        print(f"\n--- SYNC MULTI-PARTY (brainstorm, {rounds} rounds) ---")
        agent_b = DuetMindAgent("Apollo", {"logic": 0.7, "creativity": 0.85, "analytical": 0.5}, engine=engine, monitors=agent.monitors)
        agent_c = DuetMindAgent("Hermes", {"logic": 0.55, "creativity": 0.9, "analytical": 0.6}, engine=engine, monitors=agent.monitors)
        session = agent.brainstorm([agent, agent_b, agent_c], "Designing resilient edge network", rounds=rounds)
        print(f"Converged: {session['converged']} | Final topic: {session['topic_final']}")
        
        # Persist the session
        saved_path = agent.persist_session(session, log_dir)
        print(f"Session saved to: {saved_path}")


async def _demo_async(rounds: int = 4, parallel: bool = False, skip_multiparty: bool = False, log_dir: str = "./sessions") -> None:
    tmpdir = Path("./_duetmind_demo_async")
    tmpdir.mkdir(exist_ok=True)
    files = _write_demo_files(tmpdir)
    engine = ExampleEngine()
    agent = DuetMindAgent.from_config(files["agent"], engine=engine)
    agent.style.setdefault("resource_budget", 0.02)

    print("\n--- ASYNC SAFE RUN ---")
    out = await agent.async_generate_reasoning_tree("Async: strategic planning iteration", include_monitor_trace=True)
    print(json.dumps(out, indent=2))

    if not skip_multiparty:
        print(f"\n--- ASYNC MULTI-PARTY ({'parallel' if parallel else 'sequential'} debate, {rounds} rounds) ---")
        agent_b = DuetMindAgent("ApolloAsync", {"logic": 0.8, "creativity": 0.6, "analytical": 0.55}, engine=engine, monitors=agent.monitors)
        agent_c = DuetMindAgent("HermesAsync", {"logic": 0.5, "creativity": 0.9, "analytical": 0.6}, engine=engine, monitors=agent.monitors)

        session = await agent.async_debate(
            [agent, agent_b, agent_c],
            "Ethical frameworks for autonomous swarms",
            rounds=rounds,
            parallel_round=parallel
        )
        print(f"Async Converged: {session['converged']} | Rounds executed: {session['rounds_executed']}")
        
        # Persist the session
        saved_path = agent.persist_session(session, log_dir)
        print(f"Session saved to: {saved_path}")


def _demo_both(rounds: int = 3, parallel: bool = False, skip_multiparty: bool = False, log_dir: str = "./sessions") -> None:
    """Run both sync and async demos"""
    _demo_sync(rounds, skip_multiparty, log_dir)
    asyncio.run(_demo_async(rounds, parallel, skip_multiparty, log_dir))


def main() -> None:
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(
        description="DuetMindAgent - Multi-agent dialogue system with advanced reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--demo",
        choices=["sync", "async", "both"],
        default="both",
        help="Which demo to run"
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
    
    # Ensure log directory exists
    Path(args.log_dir).mkdir(exist_ok=True)
    
    try:
        if args.demo == "sync":
            _demo_sync(args.rounds, args.no_multiparty, args.log_dir)
        elif args.demo == "async":
            asyncio.run(_demo_async(args.rounds, args.parallel, args.no_multiparty, args.log_dir))
        else:  # both
            _demo_both(args.rounds, args.parallel, args.no_multiparty, args.log_dir)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


# Backward compatibility: maintain old _demo function
def _demo() -> None:
    """Legacy demo function for backward compatibility"""
    _demo_both()


if __name__ == "__main__":
    main()
