"""
Evaluation Metrics Calculator

Implements various metrics for evaluating agent performance and output quality.
Based on FR12 requirements from RoadmapAgentNet.md.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

logger = logging.getLogger("agentnet.eval.metrics")


class CriteriaType(str, Enum):
    """Types of success criteria."""
    KEYWORD_PRESENCE = "keyword_presence"
    KEYWORD_ABSENCE = "keyword_absence"
    SEMANTIC_SCORE = "semantic_score"
    LENGTH_CHECK = "length_check"
    REGEX_MATCH = "regex_match"
    CUSTOM_FUNCTION = "custom_function"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    RULE_VIOLATIONS_COUNT = "rule_violations_count"


@dataclass
class SuccessCriteria:
    """
    Represents a single success criterion for evaluation.
    
    Based on roadmap example:
    - type: keyword_presence
      must_include: ["redundancy", "failover"]
    - type: semantic_score
      reference_id: "ref_doc_12"
      min_score: 0.78
    """
    type: CriteriaType
    name: Optional[str] = None
    weight: float = 1.0
    
    # Keyword criteria
    must_include: Optional[List[str]] = None
    must_exclude: Optional[List[str]] = None
    case_sensitive: bool = False
    
    # Semantic criteria
    reference_text: Optional[str] = None
    reference_id: Optional[str] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    
    # Length criteria
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    # Regex criteria
    pattern: Optional[str] = None
    flags: int = 0
    
    # Confidence criteria
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    
    # Violations criteria
    max_violations: Optional[int] = None
    violation_types: Optional[List[str]] = None
    
    # Custom function criteria
    function_name: Optional[str] = None
    function_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate criteria configuration."""
        if self.type == CriteriaType.KEYWORD_PRESENCE and not self.must_include:
            raise ValueError("keyword_presence criteria must specify must_include")
        if self.type == CriteriaType.KEYWORD_ABSENCE and not self.must_exclude:
            raise ValueError("keyword_absence criteria must specify must_exclude")
        if self.type == CriteriaType.SEMANTIC_SCORE and not (self.reference_text or self.reference_id):
            raise ValueError("semantic_score criteria must specify reference_text or reference_id")
        if self.type == CriteriaType.REGEX_MATCH and not self.pattern:
            raise ValueError("regex_match criteria must specify pattern")
        if self.weight <= 0:
            raise ValueError("Criteria weight must be positive")


@dataclass
class CriteriaResult:
    """Result of evaluating a single criteria."""
    criteria_name: str
    criteria_type: CriteriaType
    passed: bool
    score: float = 0.0
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def weighted_score(self) -> float:
        """Get weighted score."""
        return self.score * self.weight


@dataclass
class EvaluationMetrics:
    """
    Complete evaluation metrics for a scenario or session.
    
    Includes coverage, novelty, coherence scores and rule violations as mentioned
    in the roadmap.
    """
    scenario_name: str
    total_score: float = 0.0
    max_possible_score: float = 0.0
    success_rate: float = 0.0
    
    # Individual criteria results
    criteria_results: List[CriteriaResult] = field(default_factory=list)
    
    # Standard metrics
    coverage_score: float = 0.0
    novelty_score: float = 0.0
    coherence_score: float = 0.0
    rule_violations_count: int = 0
    
    # Timing metrics
    execution_time: Optional[float] = None
    response_time: Optional[float] = None
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_criteria_result(self, result: CriteriaResult):
        """Add a criteria result and update totals."""
        self.criteria_results.append(result)
        self.total_score += result.weighted_score
        self.max_possible_score += result.weight
        
        # Update success rate
        if self.max_possible_score > 0:
            self.success_rate = self.total_score / self.max_possible_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
            "success_rate": self.success_rate,
            "coverage_score": self.coverage_score,
            "novelty_score": self.novelty_score,
            "coherence_score": self.coherence_score,
            "rule_violations_count": self.rule_violations_count,
            "execution_time": self.execution_time,
            "response_time": self.response_time,
            "criteria_results": [
                {
                    "criteria_name": cr.criteria_name,
                    "criteria_type": cr.criteria_type,
                    "passed": cr.passed,
                    "score": cr.score,
                    "weight": cr.weight,
                    "weighted_score": cr.weighted_score,
                    "details": cr.details,
                    "error": cr.error
                }
                for cr in self.criteria_results
            ],
            "metadata": self.metadata
        }


class MetricsCalculator:
    """
    Calculates evaluation metrics for agent outputs and sessions.
    
    Supports various types of success criteria and computes standard
    quality metrics like coverage, novelty, and coherence.
    """
    
    def __init__(self):
        self.logger = logger
        self.custom_functions: Dict[str, callable] = {}
        self.reference_texts: Dict[str, str] = {}
    
    def register_custom_function(self, name: str, function: callable):
        """
        Register a custom evaluation function.
        
        Args:
            name: Function name to reference in criteria
            function: Callable that takes (content, params) and returns (passed, score, details)
        """
        self.custom_functions[name] = function
        self.logger.info(f"Registered custom evaluation function: {name}")
    
    def register_reference_text(self, reference_id: str, text: str):
        """
        Register reference text for semantic scoring.
        
        Args:
            reference_id: ID to reference in criteria
            text: Reference text content
        """
        self.reference_texts[reference_id] = text
        self.logger.info(f"Registered reference text: {reference_id}")
    
    def evaluate_content(
        self,
        content: str,
        criteria: List[SuccessCriteria],
        scenario_name: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate content against a list of success criteria.
        
        Args:
            content: Text content to evaluate
            criteria: List of SuccessCriteria to check
            scenario_name: Name of the scenario being evaluated
            metadata: Additional metadata to include
            
        Returns:
            EvaluationMetrics with detailed results
        """
        metrics = EvaluationMetrics(
            scenario_name=scenario_name,
            metadata=metadata or {}
        )
        
        for criteria_obj in criteria:
            try:
                result = self._evaluate_single_criteria(content, criteria_obj, metadata)
                metrics.add_criteria_result(result)
            except Exception as e:
                # Create failed result for this criteria
                result = CriteriaResult(
                    criteria_name=criteria_obj.name or f"{criteria_obj.type}_criteria",
                    criteria_type=criteria_obj.type,
                    passed=False,
                    score=0.0,
                    weight=criteria_obj.weight,
                    error=str(e)
                )
                metrics.add_criteria_result(result)
                self.logger.error(f"Error evaluating criteria {criteria_obj.type}: {str(e)}")
        
        # Calculate standard metrics
        self._calculate_standard_metrics(content, metrics, metadata)
        
        return metrics
    
    def _evaluate_single_criteria(
        self,
        content: str,
        criteria: SuccessCriteria,
        metadata: Optional[Dict[str, Any]]
    ) -> CriteriaResult:
        """Evaluate a single criteria against content."""
        criteria_name = criteria.name or f"{criteria.type}_criteria"
        
        if criteria.type == CriteriaType.KEYWORD_PRESENCE:
            return self._evaluate_keyword_presence(content, criteria, criteria_name)
        elif criteria.type == CriteriaType.KEYWORD_ABSENCE:
            return self._evaluate_keyword_absence(content, criteria, criteria_name)
        elif criteria.type == CriteriaType.SEMANTIC_SCORE:
            return self._evaluate_semantic_score(content, criteria, criteria_name)
        elif criteria.type == CriteriaType.LENGTH_CHECK:
            return self._evaluate_length_check(content, criteria, criteria_name)
        elif criteria.type == CriteriaType.REGEX_MATCH:
            return self._evaluate_regex_match(content, criteria, criteria_name)
        elif criteria.type == CriteriaType.CONFIDENCE_THRESHOLD:
            return self._evaluate_confidence_threshold(content, criteria, criteria_name, metadata)
        elif criteria.type == CriteriaType.RULE_VIOLATIONS_COUNT:
            return self._evaluate_violations_count(content, criteria, criteria_name, metadata)
        elif criteria.type == CriteriaType.CUSTOM_FUNCTION:
            return self._evaluate_custom_function(content, criteria, criteria_name)
        else:
            raise ValueError(f"Unsupported criteria type: {criteria.type}")
    
    def _evaluate_keyword_presence(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str
    ) -> CriteriaResult:
        """Evaluate keyword presence criteria."""
        if not criteria.must_include:
            raise ValueError("must_include is required for keyword_presence")
        
        search_content = content if criteria.case_sensitive else content.lower()
        found_keywords = []
        missing_keywords = []
        
        for keyword in criteria.must_include:
            search_keyword = keyword if criteria.case_sensitive else keyword.lower()
            if search_keyword in search_content:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        score = len(found_keywords) / len(criteria.must_include)
        passed = len(missing_keywords) == 0
        
        return CriteriaResult(
            criteria_name=criteria_name,
            criteria_type=criteria.type,
            passed=passed,
            score=score,
            weight=criteria.weight,
            details={
                "required_keywords": criteria.must_include,
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords,
                "case_sensitive": criteria.case_sensitive
            }
        )
    
    def _evaluate_keyword_absence(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str
    ) -> CriteriaResult:
        """Evaluate keyword absence criteria."""
        if not criteria.must_exclude:
            raise ValueError("must_exclude is required for keyword_absence")
        
        search_content = content if criteria.case_sensitive else content.lower()
        found_keywords = []
        
        for keyword in criteria.must_exclude:
            search_keyword = keyword if criteria.case_sensitive else keyword.lower()
            if search_keyword in search_content:
                found_keywords.append(keyword)
        
        score = 1.0 - (len(found_keywords) / len(criteria.must_exclude))
        passed = len(found_keywords) == 0
        
        return CriteriaResult(
            criteria_name=criteria_name,
            criteria_type=criteria.type,
            passed=passed,
            score=score,
            weight=criteria.weight,
            details={
                "excluded_keywords": criteria.must_exclude,
                "found_keywords": found_keywords,
                "case_sensitive": criteria.case_sensitive
            }
        )
    
    def _evaluate_semantic_score(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str
    ) -> CriteriaResult:
        """Evaluate semantic similarity score."""
        # Get reference text
        reference_text = None
        if criteria.reference_text:
            reference_text = criteria.reference_text
        elif criteria.reference_id:
            reference_text = self.reference_texts.get(criteria.reference_id)
            if not reference_text:
                raise ValueError(f"Reference text not found for ID: {criteria.reference_id}")
        else:
            raise ValueError("Either reference_text or reference_id must be provided")
        
        # Simple semantic similarity using word overlap (can be replaced with embeddings)
        content_words = set(content.lower().split())
        reference_words = set(reference_text.lower().split())
        
        if not reference_words:
            similarity_score = 0.0
        else:
            intersection = content_words.intersection(reference_words)
            union = content_words.union(reference_words)
            similarity_score = len(intersection) / len(union) if union else 0.0
        
        # Check against thresholds
        passed = True
        if criteria.min_score is not None:
            passed = passed and similarity_score >= criteria.min_score
        if criteria.max_score is not None:
            passed = passed and similarity_score <= criteria.max_score
        
        return CriteriaResult(
            criteria_name=criteria_name,
            criteria_type=criteria.type,
            passed=passed,
            score=similarity_score,
            weight=criteria.weight,
            details={
                "similarity_score": similarity_score,
                "min_score": criteria.min_score,
                "max_score": criteria.max_score,
                "reference_id": criteria.reference_id,
                "method": "word_overlap"  # Could be upgraded to embeddings
            }
        )
    
    def _evaluate_length_check(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str
    ) -> CriteriaResult:
        """Evaluate content length criteria."""
        content_length = len(content)
        
        passed = True
        if criteria.min_length is not None:
            passed = passed and content_length >= criteria.min_length
        if criteria.max_length is not None:
            passed = passed and content_length <= criteria.max_length
        
        # Score based on how well it fits within bounds
        score = 1.0
        if criteria.min_length is not None and content_length < criteria.min_length:
            score = content_length / criteria.min_length
        elif criteria.max_length is not None and content_length > criteria.max_length:
            score = criteria.max_length / content_length
        
        return CriteriaResult(
            criteria_name=criteria_name,
            criteria_type=criteria.type,
            passed=passed,
            score=score,
            weight=criteria.weight,
            details={
                "content_length": content_length,
                "min_length": criteria.min_length,
                "max_length": criteria.max_length
            }
        )
    
    def _evaluate_regex_match(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str
    ) -> CriteriaResult:
        """Evaluate regex pattern matching."""
        if not criteria.pattern:
            raise ValueError("pattern is required for regex_match")
        
        try:
            pattern = re.compile(criteria.pattern, criteria.flags)
            matches = pattern.findall(content)
            
            passed = len(matches) > 0
            score = 1.0 if passed else 0.0
            
            return CriteriaResult(
                criteria_name=criteria_name,
                criteria_type=criteria.type,
                passed=passed,
                score=score,
                weight=criteria.weight,
                details={
                    "pattern": criteria.pattern,
                    "matches": matches,
                    "match_count": len(matches)
                }
            )
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {str(e)}")
    
    def _evaluate_confidence_threshold(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str,
        metadata: Optional[Dict[str, Any]]
    ) -> CriteriaResult:
        """Evaluate confidence threshold criteria."""
        # Try to get confidence from metadata
        confidence = None
        if metadata:
            confidence = metadata.get("confidence")
            if confidence is None and "result" in metadata:
                result = metadata["result"]
                if isinstance(result, dict):
                    confidence = result.get("confidence")
        
        if confidence is None:
            return CriteriaResult(
                criteria_name=criteria_name,
                criteria_type=criteria.type,
                passed=False,
                score=0.0,
                weight=criteria.weight,
                error="No confidence value found in metadata"
            )
        
        passed = True
        if criteria.min_confidence is not None:
            passed = passed and confidence >= criteria.min_confidence
        if criteria.max_confidence is not None:
            passed = passed and confidence <= criteria.max_confidence
        
        return CriteriaResult(
            criteria_name=criteria_name,
            criteria_type=criteria.type,
            passed=passed,
            score=confidence,
            weight=criteria.weight,
            details={
                "confidence": confidence,
                "min_confidence": criteria.min_confidence,
                "max_confidence": criteria.max_confidence
            }
        )
    
    def _evaluate_violations_count(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str,
        metadata: Optional[Dict[str, Any]]
    ) -> CriteriaResult:
        """Evaluate rule violations count criteria."""
        violations_count = 0
        violation_details = []
        
        if metadata:
            # Look for violations in metadata
            violations = metadata.get("violations", [])
            if criteria.violation_types:
                # Filter by specific violation types
                filtered_violations = [
                    v for v in violations
                    if isinstance(v, dict) and v.get("type") in criteria.violation_types
                ]
                violations_count = len(filtered_violations)
                violation_details = filtered_violations
            else:
                violations_count = len(violations)
                violation_details = violations
        
        # Check against max violations threshold
        passed = True
        if criteria.max_violations is not None:
            passed = violations_count <= criteria.max_violations
        
        # Score inversely related to violations (fewer violations = higher score)
        if criteria.max_violations is not None and criteria.max_violations > 0:
            score = max(0.0, (criteria.max_violations - violations_count) / criteria.max_violations)
        else:
            score = 1.0 if violations_count == 0 else 0.0
        
        return CriteriaResult(
            criteria_name=criteria_name,
            criteria_type=criteria.type,
            passed=passed,
            score=score,
            weight=criteria.weight,
            details={
                "violations_count": violations_count,
                "max_violations": criteria.max_violations,
                "violation_types": criteria.violation_types,
                "violation_details": violation_details
            }
        )
    
    def _evaluate_custom_function(
        self,
        content: str,
        criteria: SuccessCriteria,
        criteria_name: str
    ) -> CriteriaResult:
        """Evaluate using custom function."""
        if not criteria.function_name:
            raise ValueError("function_name is required for custom_function")
        
        if criteria.function_name not in self.custom_functions:
            raise ValueError(f"Custom function not registered: {criteria.function_name}")
        
        function = self.custom_functions[criteria.function_name]
        params = criteria.function_params or {}
        
        try:
            passed, score, details = function(content, params)
            
            return CriteriaResult(
                criteria_name=criteria_name,
                criteria_type=criteria.type,
                passed=bool(passed),
                score=float(score),
                weight=criteria.weight,
                details=details or {}
            )
        except Exception as e:
            raise ValueError(f"Custom function {criteria.function_name} failed: {str(e)}")
    
    def _calculate_standard_metrics(
        self,
        content: str,
        metrics: EvaluationMetrics,
        metadata: Optional[Dict[str, Any]]
    ):
        """Calculate standard metrics like coverage, novelty, coherence."""
        # Simple implementations - can be enhanced with more sophisticated algorithms
        
        # Coverage score: ratio of unique words to total words
        words = content.split()
        unique_words = set(word.lower() for word in words)
        metrics.coverage_score = len(unique_words) / len(words) if words else 0.0
        
        # Novelty score: based on uncommon words (simple heuristic)
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        novel_words = unique_words - common_words
        metrics.novelty_score = len(novel_words) / len(unique_words) if unique_words else 0.0
        
        # Coherence score: simple metric based on sentence structure
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        # Normalize to 0-1 range (assuming 10-20 words per sentence is optimal)
        if 10 <= avg_sentence_length <= 20:
            metrics.coherence_score = 1.0
        elif avg_sentence_length < 10:
            metrics.coherence_score = avg_sentence_length / 10.0
        else:
            metrics.coherence_score = max(0.0, 1.0 - (avg_sentence_length - 20) / 20.0)
        
        # Rule violations from metadata
        if metadata and "violations" in metadata:
            metrics.rule_violations_count = len(metadata["violations"])


# Example usage and testing
if __name__ == "__main__":
    # Example criteria based on roadmap
    criteria = [
        SuccessCriteria(
            type=CriteriaType.KEYWORD_PRESENCE,
            name="resilience_keywords",
            must_include=["redundancy", "failover"],
            weight=2.0
        ),
        SuccessCriteria(
            type=CriteriaType.LENGTH_CHECK,
            name="response_length",
            min_length=100,
            max_length=1000,
            weight=1.0
        ),
        SuccessCriteria(
            type=CriteriaType.CONFIDENCE_THRESHOLD,
            name="min_confidence",
            min_confidence=0.7,
            weight=1.5
        )
    ]
    
    # Example content
    content = """
    To ensure high availability, we need redundancy at multiple levels.
    Implementing failover mechanisms is crucial for system resilience.
    Load balancers can distribute traffic and provide automatic failover.
    """
    
    # Example metadata
    metadata = {
        "confidence": 0.85,
        "violations": []
    }
    
    calculator = MetricsCalculator()
    results = calculator.evaluate_content(content, criteria, "resilience_test", metadata)
    
    print("Evaluation Results:")
    print(f"Success Rate: {results.success_rate:.2f}")
    print(f"Total Score: {results.total_score:.2f}/{results.max_possible_score:.2f}")
    print(f"Coverage: {results.coverage_score:.2f}")
    print(f"Novelty: {results.novelty_score:.2f}")
    print(f"Coherence: {results.coherence_score:.2f}")
    
    for result in results.criteria_results:
        print(f"  {result.criteria_name}: {'PASS' if result.passed else 'FAIL'} ({result.score:.2f})")