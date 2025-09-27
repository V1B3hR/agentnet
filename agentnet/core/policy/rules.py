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


def create_semantic_similarity_rule(
    name: str,
    max_similarity: float = 0.92,
    embedding_set: str = "restricted_corpora",
    window_size: int = 5,
    severity: Severity = Severity.SEVERE,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that checks semantic similarity against restricted content.
    
    Args:
        name: Rule name
        max_similarity: Maximum allowed similarity threshold
        embedding_set: Name of the embedding set to compare against
        window_size: Number of historical items to compare
        severity: Severity if similarity threshold exceeded
        description: Rule description
    """
    # Storage for historical content
    if not hasattr(create_semantic_similarity_rule, "_history"):
        create_semantic_similarity_rule._history = {}

    # Try to import semantic similarity functionality
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            use_semantic = True
        except Exception:
            logger.warning("Semantic model unavailable, using Jaccard similarity")
            model = None
            use_semantic = False
    except ImportError:
        logger.warning("sentence-transformers not available, using Jaccard similarity")
        model = None
        use_semantic = False

    def semantic_similarity(text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
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

    def check_semantic_similarity(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", ""))
        if not content.strip():
            return True, None
        
        # Get history key from context
        agent_name = context.get("agent_name", "default")
        task_id = context.get("task_id", "default")
        history_key = f"{agent_name}_{task_id}_{embedding_set}"
        
        history = create_semantic_similarity_rule._history.get(history_key, [])
        
        # Check against historical content
        for i, historical_content in enumerate(history[-window_size:]):
            similarity = semantic_similarity(content, historical_content)
            if similarity > max_similarity:
                return False, f"Content similarity {similarity:.3f} exceeds threshold {max_similarity} (compared to item {i+1} turns ago)"
        
        # Add current content to history
        history.append(content)
        create_semantic_similarity_rule._history[history_key] = history[-window_size*2:]  # Keep limited history
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_semantic_similarity,
        severity=severity,
        description=description or f"Semantic similarity threshold: {max_similarity}",
        tags=["semantic", "similarity", "content"]
    )


def create_llm_classifier_rule(
    name: str,
    model: str = "moderation-small",
    threshold: float = 0.78,
    classification_target: str = "toxicity",
    severity: Severity = Severity.MAJOR,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that uses LLM classification to evaluate content.
    
    Args:
        name: Rule name
        model: Model to use for classification
        threshold: Classification threshold
        classification_target: What to classify for (toxicity, pii, etc.)
        severity: Severity if threshold exceeded
        description: Rule description
    """
    def check_llm_classification(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", ""))
        if not content.strip():
            return True, None
        
        # Simulate LLM classification (in production, this would call an actual LLM service)
        # For now, use simple heuristics based on classification target
        score = 0.0
        
        if classification_target == "toxicity":
            # Simple toxicity detection heuristics
            toxic_words = ["hate", "toxic", "offensive", "harmful", "inappropriate"]
            toxic_count = sum(1 for word in toxic_words if word.lower() in content.lower())
            score = min(toxic_count / len(toxic_words), 1.0)
            
        elif classification_target == "pii":
            # Simple PII detection
            import re
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
            ]
            pii_matches = sum(1 for pattern in pii_patterns if re.search(pattern, content))
            score = min(pii_matches / len(pii_patterns), 1.0)
        
        else:
            # Default: random score for unknown classification targets
            score = 0.1
        
        if score > threshold:
            return False, f"LLM classifier '{classification_target}' score {score:.3f} exceeds threshold {threshold}"
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_llm_classification,
        severity=severity,
        description=description or f"LLM classifier: {classification_target} threshold {threshold}",
        tags=["llm", "classification", classification_target]
    )


def create_numerical_threshold_rule(
    name: str,
    metric_name: str,
    threshold: float,
    operator: str = "less_than",
    severity: Severity = Severity.MINOR,
    description: str = ""
) -> ConstraintRule:
    """
    Create a rule that checks numerical thresholds on metrics.
    
    Args:
        name: Rule name
        metric_name: Name of metric to check in context
        threshold: Numerical threshold
        operator: Comparison operator (less_than, greater_than, equals, not_equals)
        severity: Severity if threshold violated
        description: Rule description
    """
    def check_numerical_threshold(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        value = context.get(metric_name)
        if value is None:
            return False, f"Metric '{metric_name}' not found in context"
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return False, f"Metric '{metric_name}' is not numeric: {value}"
        
        if operator == "less_than":
            passed = numeric_value < threshold
            op_desc = "<"
        elif operator == "greater_than":
            passed = numeric_value > threshold
            op_desc = ">"
        elif operator == "less_equal":
            passed = numeric_value <= threshold
            op_desc = "<="
        elif operator == "greater_equal":
            passed = numeric_value >= threshold
            op_desc = ">="
        elif operator == "equals":
            passed = abs(numeric_value - threshold) < 1e-9
            op_desc = "=="
        elif operator == "not_equals":
            passed = abs(numeric_value - threshold) >= 1e-9
            op_desc = "!="
        else:
            return False, f"Unknown operator: {operator}"
        
        if not passed:
            return False, f"Metric '{metric_name}' value {numeric_value} violates threshold: {numeric_value} {op_desc} {threshold} is False"
        
        return True, None
    
    return ConstraintRule(
        name=name,
        check_fn=check_numerical_threshold,
        severity=severity,
        description=description or f"Numerical threshold: {metric_name} {operator} {threshold}",
        tags=["numerical", "threshold", "metric"]
    )