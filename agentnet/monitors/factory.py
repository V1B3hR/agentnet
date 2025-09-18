"""Monitor factory for creating monitor instances."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..core.types import CognitiveFault, Severity, _parse_severity
from .base import MonitorFn, MonitorSpec

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
        if t == "semantic_similarity":
            return MonitorFactory._semantic_similarity_monitor(spec)
        if t == "llm_classifier":
            return MonitorFactory._llm_classifier_monitor(spec)
        if t == "numerical_threshold":
            return MonitorFactory._numerical_threshold_monitor(spec)
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

            content = (
                str(result.get("content", ""))
                if isinstance(result, dict)
                else str(result)
            )
            lower = content.lower()
            present = [kw for kw in keywords if kw in lower]
            failed = (
                (len(present) > 0) if match_any else (len(present) == len(keywords))
            )

            if failed:
                rationale = f"Matched keyword(s): {', '.join(sorted(set(present)))}"
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

            content = (
                str(result.get("content", ""))
                if isinstance(result, dict)
                else str(result)
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

            runtime = (
                float(result.get("runtime", 0.0)) if isinstance(result, dict) else 0.0
            )
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
                        description=spec.description
                        or "Resource usage exceeded threshold",
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
                violations = [
                    MonitorFactory._build_violation(
                        name=violation_name,
                        vtype="custom",
                        severity=spec.severity,
                        description=spec.description
                        or f"Custom guard '{func_name}' failed",
                        rationale=rationale or "Custom function returned failure",
                        meta={"func": func_name},
                    )
                ]
                detail = {"outcome": outcome, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)

        return monitor

    @staticmethod
    def _semantic_similarity_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create a semantic similarity monitor."""
        max_similarity = float(spec.params.get("max_similarity", 0.9))
        window_size = int(spec.params.get("window_size", 5))
        embedding_set = spec.params.get("embedding_set", "restricted_corpora")
        violation_name = spec.params.get("violation_name", f"{spec.name}_semantic")

        # Storage for content history per agent-task combination
        if not hasattr(MonitorFactory, "_semantic_history"):
            MonitorFactory._semantic_history = {}

        # Try to import sentence-transformers for semantic similarity
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            # Try to load model, but handle download failures gracefully
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                use_semantic = True
            except Exception as model_error:
                logger.warning(
                    f"Failed to load sentence transformer model: {model_error}"
                )
                model = None
                use_semantic = False
        except ImportError:
            logger.warning(
                "sentence-transformers not available, falling back to Jaccard similarity"
            )
            model = None
            use_semantic = False

        def semantic_similarity(text1: str, text2: str) -> float:
            """Compute semantic similarity."""
            if not use_semantic:
                # Fallback to Jaccard similarity
                set1 = set(text1.lower().split())
                set2 = set(text2.lower().split())
                if not set1 and not set2:
                    return 1.0
                if not set1 or not set2:
                    return 0.0
                return len(set1 & set2) / len(set1 | set2)

            embeddings = model.encode([text1, text2])
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm_a = np.linalg.norm(embeddings[0])
            norm_b = np.linalg.norm(embeddings[1])

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)

        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return

            content = (
                str(result.get("content", ""))
                if isinstance(result, dict)
                else str(result)
            )
            if not content.strip():
                return

            agent_key = f"{agent.name}_{task}_semantic"

            # Initialize history for this agent-task combination
            if agent_key not in MonitorFactory._semantic_history:
                MonitorFactory._semantic_history[agent_key] = []

            history = MonitorFactory._semantic_history[agent_key]

            # Check semantic similarity against recent history
            violations = []
            for i, historical_content in enumerate(history[-window_size:]):
                similarity = semantic_similarity(content, historical_content)
                if similarity > max_similarity:
                    violations.append(
                        MonitorFactory._build_violation(
                            name=violation_name,
                            vtype="semantic_similarity",
                            severity=spec.severity,
                            description=spec.description
                            or f"Semantic similarity {similarity:.2f} exceeds threshold",
                            rationale=f"Current content semantically too similar to content from {len(history)-i} turns ago",
                            meta={
                                "similarity_score": similarity,
                                "threshold": max_similarity,
                                "historical_index": len(history) - i,
                                "embedding_set": embedding_set,
                                "window_size": window_size,
                            },
                        )
                    )

            # Add current content to history
            history.append(content)

            # Limit history size to prevent unbounded growth
            if len(history) > window_size * 2:
                history[:] = history[-window_size:]

            if violations:
                detail = {"outcome": {"content": content}, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)

        return monitor

    @staticmethod
    def _llm_classifier_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create an LLM-based classifier monitor for toxicity, PII, etc."""
        model_name = spec.params.get("model", "moderation-small")
        threshold = float(spec.params.get("threshold", 0.78))
        classifier_type = spec.params.get("classifier_type", "toxicity")
        violation_name = spec.params.get("violation_name", f"{spec.name}_classifier")

        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return

            content = (
                str(result.get("content", ""))
                if isinstance(result, dict)
                else str(result)
            )
            if not content.strip():
                return

            # Simulate LLM classifier - in production, this would call an actual moderation API
            score = MonitorFactory._simulate_classifier_score(content, classifier_type)

            if score > threshold:
                violations = [
                    MonitorFactory._build_violation(
                        name=violation_name,
                        vtype="llm_classifier",
                        severity=spec.severity,
                        description=spec.description
                        or f"{classifier_type} classifier score {score:.2f} exceeds threshold",
                        rationale=f"Content classified as {classifier_type} with confidence {score:.2f}",
                        meta={
                            "classifier_score": score,
                            "threshold": threshold,
                            "classifier_type": classifier_type,
                            "model": model_name,
                        },
                    )
                ]
                detail = {"outcome": {"content": content}, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)

        return monitor

    @staticmethod
    def _numerical_threshold_monitor(spec: MonitorSpec) -> MonitorFn:
        """Create a numerical threshold monitor."""
        field_name = spec.params.get("field", "confidence")
        min_value = spec.params.get("min_value")
        max_value = spec.params.get("max_value")
        violation_name = spec.params.get("violation_name", f"{spec.name}_threshold")

        def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
            if MonitorFactory._should_cooldown(spec, task):
                return

            if not isinstance(result, dict):
                return

            value = result.get(field_name)
            if value is None:
                return

            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                return

            violations = []
            if min_value is not None and numeric_value < min_value:
                violations.append(
                    MonitorFactory._build_violation(
                        name=violation_name,
                        vtype="numerical_threshold",
                        severity=spec.severity,
                        description=spec.description
                        or f"{field_name} value {numeric_value} below minimum threshold",
                        rationale=f"{field_name} value {numeric_value} is below minimum {min_value}",
                        meta={
                            "field": field_name,
                            "value": numeric_value,
                            "min_threshold": min_value,
                            "violation_type": "below_minimum",
                        },
                    )
                )

            if max_value is not None and numeric_value > max_value:
                violations.append(
                    MonitorFactory._build_violation(
                        name=violation_name,
                        vtype="numerical_threshold",
                        severity=spec.severity,
                        description=spec.description
                        or f"{field_name} value {numeric_value} above maximum threshold",
                        rationale=f"{field_name} value {numeric_value} is above maximum {max_value}",
                        meta={
                            "field": field_name,
                            "value": numeric_value,
                            "max_threshold": max_value,
                            "violation_type": "above_maximum",
                        },
                    )
                )

            if violations:
                detail = {"outcome": result, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)

        return monitor

    @staticmethod
    def _simulate_classifier_score(content: str, classifier_type: str) -> float:
        """Simulate classifier score for demonstration purposes."""
        # This is a simple simulation - in production, this would call actual ML models
        content_lower = content.lower()

        if classifier_type == "toxicity":
            toxic_keywords = [
                "hate",
                "toxic",
                "harmful",
                "offensive",
                "abusive",
                "racist",
                "sexist",
            ]
            score = sum(1 for word in toxic_keywords if word in content_lower) / len(
                toxic_keywords
            )
            return min(score * 2.0, 1.0)  # Scale and cap at 1.0

        elif classifier_type == "pii":
            import re

            # Simple PII detection patterns
            patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[\w\.-]+@[\w\.-]+\.\w+\b",  # Email
                r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
            ]
            matches = sum(1 for pattern in patterns if re.search(pattern, content))
            return min(matches * 0.3, 1.0)  # Scale and cap at 1.0

        else:
            # Default random score for other types
            import hashlib

            hash_val = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
            return (hash_val % 100) / 100.0


def register_custom_monitor_func(name: str, func: Any) -> None:
    """Register a custom monitor function."""
    CUSTOM_FUNCS[name] = func
    logger.info(f"Registered custom monitor function: {name}")
