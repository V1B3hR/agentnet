"""Custom and advanced monitor implementations."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, Dict

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.custom")

# Global registry for custom monitor functions
CUSTOM_FUNCS: Dict[str, Any] = {}


def register_custom_monitor_func(name: str, func: Any) -> None:
    """Register a custom monitor function.

    Args:
        name: Name of the function to register
        func: Function to register
    """
    CUSTOM_FUNCS[name] = func
    logger.info(f"Registered custom monitor function: {name}")


def create_custom_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a custom function-based monitor.

    Args:
        spec: Monitor specification with parameters:
            - func: Name of registered custom function (required)
            - violation_name: Name for violation (optional)

    Returns:
        Monitor function

    Raises:
        ValueError: If function name is not registered
    """
    func_name = spec.params.get("func")
    violation_name = spec.params.get("violation_name", f"{spec.name}_custom")
    func = CUSTOM_FUNCS.get(func_name)

    if func is None:
        raise ValueError(f"Unknown custom func: {func_name}")

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        # Import here to avoid circular imports
        from .factory import MonitorFactory

        if MonitorFactory._should_cooldown(spec, task):
            return

        outcome = (
            result if isinstance(result, dict) else {"content": str(result)}
        )
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
            MonitorFactory._handle(
                spec, agent, task, passed=False, detail=detail
            )

    return monitor


def create_llm_classifier_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create an LLM-based classifier monitor for toxicity, PII, etc.

    Args:
        spec: Monitor specification with parameters:
            - model: Model name (default: "moderation-small")
            - threshold: Score threshold for violations (default: 0.78)
            - classifier_type: Type of classifier (default: "toxicity")
            - violation_name: Name for violation (optional)

    Returns:
        Monitor function
    """
    model_name = spec.params.get("model", "moderation-small")
    threshold = float(spec.params.get("threshold", 0.78))
    classifier_type = spec.params.get("classifier_type", "toxicity")
    violation_name = spec.params.get(
        "violation_name", f"{spec.name}_classifier"
    )

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        # Import here to avoid circular imports
        from .factory import MonitorFactory

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
        score = _simulate_classifier_score(content, classifier_type)

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
            detail = {
                "outcome": {"content": content},
                "violations": violations,
            }
            MonitorFactory._handle(
                spec, agent, task, passed=False, detail=detail
            )

    return monitor


def create_numerical_threshold_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a numerical threshold monitor.

    Args:
        spec: Monitor specification with parameters:
            - field: Field name to check (default: "confidence")
            - min_value: Minimum allowed value (optional)
            - max_value: Maximum allowed value (optional)
            - violation_name: Name for violation (optional)

    Returns:
        Monitor function
    """
    field_name = spec.params.get("field", "confidence")
    min_value = spec.params.get("min_value")
    max_value = spec.params.get("max_value")
    violation_name = spec.params.get(
        "violation_name", f"{spec.name}_threshold"
    )

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        # Import here to avoid circular imports
        from .factory import MonitorFactory

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
            MonitorFactory._handle(
                spec, agent, task, passed=False, detail=detail
            )

    return monitor


def _simulate_classifier_score(content: str, classifier_type: str) -> float:
    """Simulate classifier score for demonstration purposes.

    Args:
        content: Text content to classify
        classifier_type: Type of classifier

    Returns:
        Simulated score between 0.0 and 1.0
    """
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
        score = sum(
            1 for word in toxic_keywords if word in content_lower
        ) / len(toxic_keywords)
        return min(score * 2.0, 1.0)  # Scale and cap at 1.0

    elif classifier_type == "pii":
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
        hash_val = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
        return (hash_val % 100) / 100.0
