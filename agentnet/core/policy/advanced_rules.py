"""
Advanced policy rule types for AgentNet.

Implements semantic similarity matching, LLM-based classification,
and numerical threshold rules for advanced policy enforcement.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .rules import ConstraintRule, RuleResult, Severity

logger = logging.getLogger("agentnet.policy.advanced_rules")


class SemanticSimilarityRule(ConstraintRule):
    """
    Rule that checks semantic similarity against a blocklist or reference set.
    
    Uses embeddings to detect content that is semantically similar to
    prohibited content, even if exact keywords don't match.
    """
    
    def __init__(
        self,
        name: str,
        embedding_set: List[str],
        max_similarity: float = 0.85,
        severity: Severity = Severity.SEVERE,
        description: str = "",
        enabled: bool = True,
        embedding_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize semantic similarity rule.
        
        Args:
            name: Rule name
            embedding_set: List of reference texts to compare against
            max_similarity: Maximum allowed cosine similarity (0-1)
            severity: Violation severity
            description: Rule description
            enabled: Whether rule is active
            embedding_model: Model to use for embeddings (default: sentence-transformers)
            tags: Rule tags
            metadata: Additional metadata
        """
        self.embedding_set = embedding_set
        self.max_similarity = max_similarity
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self._embeddings_cache = None
        
        # Create the check function
        def check_fn(context: Dict[str, Any]) -> Union[bool, RuleResult]:
            return self._check_similarity(context)
        
        super().__init__(
            name=name,
            check_fn=check_fn,
            severity=severity,
            description=description or f"Semantic similarity check (max: {max_similarity})",
            enabled=enabled,
            tags=tags,
            metadata=metadata,
        )
    
    def _get_embeddings(self):
        """Lazy load and cache embeddings for the reference set."""
        if self._embeddings_cache is not None:
            return self._embeddings_cache
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            return None
        
        try:
            model = SentenceTransformer(self.embedding_model)
            self._embeddings_cache = {
                "model": model,
                "reference_embeddings": model.encode(self.embedding_set),
            }
            return self._embeddings_cache
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    
    def _check_similarity(self, context: Dict[str, Any]) -> RuleResult:
        """Check semantic similarity of content against reference set."""
        content = context.get("content", "")
        
        if not content:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                rationale="No content to check",
            )
        
        embeddings = self._get_embeddings()
        if embeddings is None:
            # Fallback: pass if embeddings unavailable
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                rationale="Embedding model unavailable",
                error="sentence-transformers not installed",
            )
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = embeddings["model"]
            reference_embeddings = embeddings["reference_embeddings"]
            
            # Encode the content
            content_embedding = model.encode([content])
            
            # Calculate similarities
            similarities = cosine_similarity(content_embedding, reference_embeddings)[0]
            max_sim = float(np.max(similarities))
            
            passed = max_sim <= self.max_similarity
            
            return RuleResult(
                rule_name=self.name,
                passed=passed,
                severity=self.severity,
                description=self.description,
                rationale=f"Max similarity: {max_sim:.3f} (threshold: {self.max_similarity})",
                metadata={
                    "max_similarity": max_sim,
                    "threshold": self.max_similarity,
                    "matched_index": int(np.argmax(similarities)) if not passed else None,
                },
            )
        except Exception as e:
            logger.error(f"Error in semantic similarity check: {e}")
            return RuleResult(
                rule_name=self.name,
                passed=True,  # Fail open on errors
                severity=self.severity,
                description=self.description,
                error=str(e),
            )


class LLMClassifierRule(ConstraintRule):
    """
    Rule that uses an LLM to classify content.
    
    Useful for detecting toxicity, inappropriate content, policy violations,
    or other complex patterns that are hard to detect with regex or keywords.
    """
    
    def __init__(
        self,
        name: str,
        classifier_prompt: str,
        threshold: float = 0.7,
        provider_adapter: Optional[Any] = None,
        severity: Severity = Severity.MAJOR,
        description: str = "",
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM classifier rule.
        
        Args:
            name: Rule name
            classifier_prompt: Prompt template for classification
            threshold: Confidence threshold for violations (0-1)
            provider_adapter: LLM provider to use for classification
            severity: Violation severity
            description: Rule description
            enabled: Whether rule is active
            tags: Rule tags
            metadata: Additional metadata
        """
        self.classifier_prompt = classifier_prompt
        self.threshold = threshold
        self.provider_adapter = provider_adapter
        
        # Create the check function
        def check_fn(context: Dict[str, Any]) -> Union[bool, RuleResult]:
            return self._classify_content(context)
        
        super().__init__(
            name=name,
            check_fn=check_fn,
            severity=severity,
            description=description or f"LLM classification (threshold: {threshold})",
            enabled=enabled,
            tags=tags,
            metadata=metadata,
        )
    
    def _classify_content(self, context: Dict[str, Any]) -> RuleResult:
        """Use LLM to classify content."""
        content = context.get("content", "")
        
        if not content:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                rationale="No content to classify",
            )
        
        if self.provider_adapter is None:
            logger.warning(f"No provider adapter configured for {self.name}")
            return RuleResult(
                rule_name=self.name,
                passed=True,  # Fail open if no provider
                severity=self.severity,
                description=self.description,
                error="No provider adapter configured",
            )
        
        try:
            # Format the classification prompt
            prompt = self.classifier_prompt.format(content=content)
            
            # Get classification from LLM
            response = self.provider_adapter.infer(prompt)
            classification_text = response.content.strip().lower()
            
            # Parse the response (expecting format like "violation: 0.85" or "safe: 0.95")
            # This is a simple parser - production would be more robust
            passed = True
            confidence = 0.0
            
            if "violation" in classification_text or "unsafe" in classification_text:
                # Try to extract confidence score
                import re
                match = re.search(r'(\d+\.?\d*)', classification_text)
                if match:
                    confidence = float(match.group(1))
                    if confidence < 1.0:  # Already a probability
                        passed = confidence < self.threshold
                    else:  # Percentage
                        passed = (confidence / 100) < self.threshold
                else:
                    passed = False
            
            return RuleResult(
                rule_name=self.name,
                passed=passed,
                severity=self.severity,
                description=self.description,
                rationale=f"LLM confidence: {confidence:.3f} (threshold: {self.threshold})",
                metadata={
                    "confidence": confidence,
                    "threshold": self.threshold,
                    "raw_response": classification_text,
                },
            )
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return RuleResult(
                rule_name=self.name,
                passed=True,  # Fail open on errors
                severity=self.severity,
                description=self.description,
                error=str(e),
            )


class NumericalThresholdRule(ConstraintRule):
    """
    Rule that checks numerical values against thresholds.
    
    Useful for resource limits, token counts, cost tracking, etc.
    """
    
    def __init__(
        self,
        name: str,
        metric_key: str,
        threshold: float,
        comparison: str = "<=",
        severity: Severity = Severity.MINOR,
        description: str = "",
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize numerical threshold rule.
        
        Args:
            name: Rule name
            metric_key: Key in context to check
            threshold: Threshold value
            comparison: Comparison operator (<=, <, >=, >, ==, !=)
            severity: Violation severity
            description: Rule description
            enabled: Whether rule is active
            tags: Rule tags
            metadata: Additional metadata
        """
        self.metric_key = metric_key
        self.threshold = threshold
        self.comparison = comparison
        
        # Create the check function
        def check_fn(context: Dict[str, Any]) -> Union[bool, RuleResult]:
            return self._check_threshold(context)
        
        super().__init__(
            name=name,
            check_fn=check_fn,
            severity=severity,
            description=description or f"{metric_key} {comparison} {threshold}",
            enabled=enabled,
            tags=tags,
            metadata=metadata,
        )
    
    def _check_threshold(self, context: Dict[str, Any]) -> RuleResult:
        """Check numerical threshold."""
        value = context.get(self.metric_key)
        
        if value is None:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                rationale=f"Metric '{self.metric_key}' not found in context",
            )
        
        try:
            value = float(value)
        except (ValueError, TypeError):
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                error=f"Cannot convert '{value}' to number",
            )
        
        # Perform comparison
        passed = False
        if self.comparison == "<=":
            passed = value <= self.threshold
        elif self.comparison == "<":
            passed = value < self.threshold
        elif self.comparison == ">=":
            passed = value >= self.threshold
        elif self.comparison == ">":
            passed = value > self.threshold
        elif self.comparison == "==":
            passed = abs(value - self.threshold) < 1e-9
        elif self.comparison == "!=":
            passed = abs(value - self.threshold) >= 1e-9
        else:
            return RuleResult(
                rule_name=self.name,
                passed=True,
                severity=self.severity,
                description=self.description,
                error=f"Unknown comparison operator: {self.comparison}",
            )
        
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            description=self.description,
            rationale=f"{self.metric_key}={value} {self.comparison} {self.threshold}",
            metadata={
                "metric_key": self.metric_key,
                "value": value,
                "threshold": self.threshold,
                "comparison": self.comparison,
            },
        )
