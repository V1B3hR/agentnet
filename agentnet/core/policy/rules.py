"""
Policy rule definitions and evaluation logic.

Implements the rule-based constraint system for policy evaluation.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("agentnet.policy.rules")


class Severity(str, Enum):
    """Severity levels for policy violations."""
    
    MINOR = "minor"
    MAJOR = "major" 
    SEVERE = "severe"


@dataclass
class RuleResult:
    """Result of evaluating a single rule."""
    
    rule_name: str
    passed: bool
    severity: Severity
    description: str = ""
    rationale: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "description": self.description,
            "rationale": self.rationale,
            "error": self.error,
            "metadata": self.metadata
        }


# Type definitions for rule checking
RuleCheckResult = Union[bool, Tuple[bool, Optional[str]], RuleResult]
RuleCheckFn = Callable[[Dict[str, Any]], RuleCheckResult]


class ConstraintRule:
    """
    A single policy constraint rule.
    
    Rules can evaluate agent outputs, actions, or metadata to determine
    if the action should be allowed, blocked, or transformed.
    """
    
    def __init__(
        self,
        name: str,
        check_fn: RuleCheckFn,
        severity: Severity = Severity.MINOR,
        description: str = "",
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a constraint rule.
        
        Args:
            name: Unique name for the rule
            check_fn: Function that evaluates the rule
            severity: Severity level if rule fails
            description: Human-readable description
            enabled: Whether the rule is active
            tags: Optional tags for rule categorization
            metadata: Additional rule metadata
        """
        self.name = name
        self.check_fn = check_fn
        self.severity = severity
        self.description = description
        self.enabled = enabled
        self.tags = tags or []
        self.metadata = metadata or {}
        self.evaluation_count = 0
        self.violation_count = 0
        self.last_evaluated = None
    
    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """
        Evaluate the rule against a given context.
        
        Args:
            context: Context containing agent output, action, metadata, etc.
            
        Returns:
            RuleResult with evaluation outcome
        """
        if not self.enabled:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                rationale="Rule disabled"
            )
        
        self.evaluation_count += 1
        self.last_evaluated = time.time()
        
        try:
            result = self.check_fn(context)
            
            # Handle different return types
            if isinstance(result, RuleResult):
                return result
            elif isinstance(result, tuple):
                passed, rationale = result
                rule_result = RuleResult(
                    rule_name=self.name,
                    passed=bool(passed),
                    severity=self.severity,
                    description=self.description,
                    rationale=rationale
                )
            else:
                rule_result = RuleResult(
                    rule_name=self.name,
                    passed=bool(result),
                    severity=self.severity,
                    description=self.description
                )
            
            if not rule_result.passed:
                self.violation_count += 1
                
            return rule_result
            
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            self.violation_count += 1
            return RuleResult(
                rule_name=self.name,
                passed=False,
                severity=self.severity,
                description=self.description,
                error=str(e),
                rationale=f"Rule evaluation failed: {e}"
            )
    
    def reset_stats(self):
        """Reset evaluation statistics."""
        self.evaluation_count = 0
        self.violation_count = 0
        self.last_evaluated = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rule evaluation statistics."""
        violation_rate = (
            self.violation_count / self.evaluation_count 
            if self.evaluation_count > 0 else 0.0
        )
        
        return {
            "name": self.name,
            "enabled": self.enabled,
            "evaluation_count": self.evaluation_count,
            "violation_count": self.violation_count,
            "violation_rate": violation_rate,
            "last_evaluated": self.last_evaluated,
            "tags": self.tags
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "description": self.description,
            "enabled": self.enabled,
            "tags": self.tags,
            "metadata": self.metadata,
            "stats": self.get_stats()
        }


def _parse_severity(value: Union[str, Severity, None]) -> Severity:
    """Parse severity from various input types."""
    if isinstance(value, Severity):
        return value
    if not value:
        return Severity.MINOR
    
    v = str(value).lower()
    if v in ["severe", "critical", "high"]:
        return Severity.SEVERE
    elif v in ["major", "medium", "warn", "warning"]:
        return Severity.MAJOR
    else:
        return Severity.MINOR


# Common rule factory functions
def create_keyword_rule(
    name: str,
    keywords: List[str],
    severity: Severity = Severity.MINOR,
    match_mode: str = "any",
    case_sensitive: bool = False,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that checks for specific keywords in content.
    
    Args:
        name: Rule name
        keywords: List of keywords to check for
        severity: Severity if keywords found
        match_mode: "any" (any keyword) or "all" (all keywords)
        case_sensitive: Whether matching is case sensitive
        description: Rule description
    """
    def check_keywords(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", ""))
        if not case_sensitive:
            content = content.lower()
            check_keywords_list = [k.lower() for k in keywords]
        else:
            check_keywords_list = keywords
        
        if match_mode == "any":
            found = [kw for kw in check_keywords_list if kw in content]
            if found:
                return False, f"Found prohibited keywords: {found}"
        else:  # "all"
            missing = [kw for kw in check_keywords_list if kw not in content]
            if missing:
                return False, f"Missing required keywords: {missing}"
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_keywords,
        severity=severity,
        description=description or f"Keyword check: {match_mode} of {keywords}",
        tags=["keyword", "content"]
    )


def create_length_rule(
    name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    severity: Severity = Severity.MINOR,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that checks content length constraints.
    """
    def check_length(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", ""))
        length = len(content)
        
        if min_length is not None and length < min_length:
            return False, f"Content too short: {length} < {min_length}"
        
        if max_length is not None and length > max_length:
            return False, f"Content too long: {length} > {max_length}"
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_length,
        severity=severity,
        description=description or f"Length check: {min_length}-{max_length}",
        tags=["length", "content"]
    )


def create_confidence_rule(
    name: str,
    min_confidence: float = 0.0,
    severity: Severity = Severity.MINOR,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that checks minimum confidence threshold.
    """
    def check_confidence(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        confidence = float(context.get("confidence", 0.0))
        
        if confidence < min_confidence:
            return False, f"Confidence too low: {confidence} < {min_confidence}"
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_confidence,
        severity=severity,
        description=description or f"Minimum confidence: {min_confidence}",
        tags=["confidence", "quality"]
    )


def create_role_rule(
    name: str,
    allowed_roles: List[str],
    severity: Severity = Severity.MAJOR,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that checks if agent role is allowed for this action.
    """
    def check_role(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        agent_role = context.get("agent_role", "")
        if not agent_role:
            agent_name = context.get("agent_name", "")
            # Simple heuristic: extract role from agent name
            agent_role = agent_name.lower()
        
        if agent_role.lower() not in [role.lower() for role in allowed_roles]:
            return False, f"Role '{agent_role}' not in allowed roles: {allowed_roles}"
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_role,
        severity=severity,
        description=description or f"Allowed roles: {allowed_roles}",
        tags=["role", "authorization"]
    )