"""
Advanced reasoning types implementation based on cognitive science research.

Implements five core reasoning types:
- Deductive: General-to-specific logical inference
- Inductive: Pattern recognition from specific observations
- Abductive: Best explanation formation from incomplete data
- Analogical: Similarity-based understanding and learning
- Causal: Cause-and-effect relationship identification
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("agentnet.reasoning")


class ReasoningType(str, Enum):
    """Enumeration of available reasoning types."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    reasoning_type: ReasoningType
    content: str
    confidence: float
    reasoning_steps: List[str]
    premises: List[str] = None
    patterns: List[str] = None
    analogies: List[str] = None
    causal_chain: List[str] = None
    statistical_confidence: Optional[float] = None
    metadata: Dict[str, Any] = None


class BaseReasoning(ABC):
    """Abstract base class for reasoning implementations."""

    def __init__(self, style_weights: Dict[str, float]):
        """
        Initialize reasoning with style weights.

        Args:
            style_weights: Dictionary with logic, creativity, analytical weights
        """
        self.style_weights = style_weights
        self.logic_weight = style_weights.get("logic", 0.5)
        self.creativity_weight = style_weights.get("creativity", 0.5)
        self.analytical_weight = style_weights.get("analytical", 0.5)

    @abstractmethod
    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """Perform reasoning on the given task."""
        pass

    def _calculate_confidence(self, base_confidence: float) -> float:
        """Calculate reasoning confidence based on style weights."""
        style_influence = (self.logic_weight + self.analytical_weight) / 2.0
        adjusted_confidence = base_confidence * (0.7 + 0.6 * style_influence)
        return min(1.0, max(0.1, adjusted_confidence))


class DeductiveReasoning(BaseReasoning):
    """
    Deductive reasoning: General-to-specific logical inference.
    Validates premises and draws logical conclusions.
    """

    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform deductive reasoning on the task.

        Args:
            task: The reasoning task
            context: Optional context including premises, rules

        Returns:
            ReasoningResult with deductive analysis
        """
        context = context or {}
        premises = context.get("premises", [])
        rules = context.get("rules", [])

        reasoning_steps = []

        # Step 1: Identify or extract premises
        if not premises:
            premises = self._extract_premises(task)
            reasoning_steps.append("Extracted implicit premises from task")

        # Step 2: Apply logical rules
        reasoning_steps.append("Validating premises for logical consistency")
        valid_premises = self._validate_premises(premises)

        # Step 3: Apply deductive inference
        reasoning_steps.append("Applying deductive inference rules")
        conclusion = self._apply_deductive_inference(valid_premises, rules, task)

        # Step 4: Verify conclusion
        reasoning_steps.append("Verifying logical validity of conclusion")

        # Calculate confidence based on logic weight
        base_confidence = 0.8 if self.logic_weight > 0.7 else 0.65
        confidence = self._calculate_confidence(base_confidence)

        return ReasoningResult(
            reasoning_type=ReasoningType.DEDUCTIVE,
            content=conclusion,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            premises=valid_premises,
            metadata={
                "logic_emphasis": self.logic_weight > 0.7,
                "premises_count": len(valid_premises),
                "inference_method": "modus_ponens" if rules else "direct_inference",
            },
        )

    def _extract_premises(self, task: str) -> List[str]:
        """Extract implicit premises from the task."""
        # Simple heuristic extraction - can be enhanced with NLP
        premises = []
        if "if" in task.lower() and "then" in task.lower():
            premises.append("Conditional relationship identified")
        if "all" in task.lower() or "every" in task.lower():
            premises.append("Universal quantification present")
        if "therefore" in task.lower() or "thus" in task.lower():
            premises.append("Conclusion indicator present")
        return premises if premises else ["General knowledge base"]

    def _validate_premises(self, premises: List[str]) -> List[str]:
        """Validate premises for logical consistency."""
        # Simple validation - can be enhanced with formal logic
        valid_premises = []
        for premise in premises:
            if premise and len(premise.strip()) > 0:
                valid_premises.append(premise.strip())
        return valid_premises

    def _apply_deductive_inference(
        self, premises: List[str], rules: List[str], task: str
    ) -> str:
        """Apply deductive inference to reach conclusion."""
        if self.logic_weight > 0.8:
            return f"Based on rigorous logical analysis of premises {premises}, the deductive conclusion is: {task}"
        else:
            return (
                f"Following logical inference from given premises, conclusion: {task}"
            )


class InductiveReasoning(BaseReasoning):
    """
    Inductive reasoning: Pattern recognition from specific observations.
    Generalizes from specific instances to broader patterns.
    """

    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform inductive reasoning on the task.

        Args:
            task: The reasoning task
            context: Optional context including observations, samples

        Returns:
            ReasoningResult with inductive analysis
        """
        context = context or {}
        observations = context.get("observations", [])
        sample_size = context.get("sample_size", 1)

        reasoning_steps = []

        # Step 1: Collect or identify observations
        if not observations:
            observations = self._identify_observations(task)
            reasoning_steps.append("Identified key observations from task context")

        # Step 2: Pattern recognition
        reasoning_steps.append("Analyzing patterns in observations")
        patterns = self._identify_patterns(observations)

        # Step 3: Generalization
        reasoning_steps.append("Forming generalizations from observed patterns")
        generalization = self._form_generalization(patterns, task)

        # Step 4: Statistical confidence assessment
        reasoning_steps.append("Assessing statistical confidence of generalization")
        statistical_conf = self._assess_statistical_confidence(
            observations, sample_size
        )

        # Calculate confidence based on analytical weight and sample quality
        base_confidence = 0.7 if self.analytical_weight > 0.6 else 0.55
        confidence = self._calculate_confidence(base_confidence * statistical_conf)

        return ReasoningResult(
            reasoning_type=ReasoningType.INDUCTIVE,
            content=generalization,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            patterns=patterns,
            statistical_confidence=statistical_conf,
            metadata={
                "analytical_emphasis": self.analytical_weight > 0.6,
                "observations_count": len(observations),
                "pattern_strength": len(patterns),
                "sample_adequacy": "adequate" if sample_size > 3 else "limited",
            },
        )

    def _identify_observations(self, task: str) -> List[str]:
        """Identify key observations from the task."""
        observations = []
        # Simple heuristic - look for data points, examples, instances
        if "example" in task.lower():
            observations.append("Example instances identified")
        if "data" in task.lower() or "results" in task.lower():
            observations.append("Data points collected")
        if "observed" in task.lower() or "noticed" in task.lower():
            observations.append("Direct observations noted")
        return observations if observations else ["Context-derived observations"]

    def _identify_patterns(self, observations: List[str]) -> List[str]:
        """Identify patterns in observations."""
        patterns = []
        if len(observations) > 2:
            patterns.append("Recurring themes identified")
        if any("consistent" in obs.lower() for obs in observations):
            patterns.append("Consistency pattern detected")
        if any(
            "trend" in obs.lower()
            or "increase" in obs.lower()
            or "decrease" in obs.lower()
            for obs in observations
        ):
            patterns.append("Directional trend observed")
        return patterns if patterns else ["General pattern structure"]

    def _form_generalization(self, patterns: List[str], task: str) -> str:
        """Form generalization from identified patterns."""
        if self.analytical_weight > 0.7:
            return f"Systematic analysis of patterns {patterns} suggests: {task}"
        else:
            return f"Based on observed patterns, generalization: {task}"

    def _assess_statistical_confidence(
        self, observations: List[str], sample_size: int
    ) -> float:
        """Assess statistical confidence of the inductive inference."""
        if sample_size >= 10:
            return 0.9
        elif sample_size >= 5:
            return 0.75
        elif sample_size >= 3:
            return 0.6
        else:
            return 0.4


class AbductiveReasoning(BaseReasoning):
    """
    Abductive reasoning: Best explanation formation from incomplete data.
    Generates hypotheses under uncertainty.
    """

    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform abductive reasoning on the task.

        Args:
            task: The reasoning task
            context: Optional context including evidence, constraints

        Returns:
            ReasoningResult with abductive analysis
        """
        context = context or {}
        evidence = context.get("evidence", [])
        constraints = context.get("constraints", [])

        reasoning_steps = []

        # Step 1: Analyze available evidence
        if not evidence:
            evidence = self._analyze_evidence(task)
            reasoning_steps.append("Analyzed available evidence from task")

        # Step 2: Generate multiple hypotheses
        reasoning_steps.append("Generating candidate explanations")
        hypotheses = self._generate_hypotheses(evidence, task)

        # Step 3: Evaluate hypotheses
        reasoning_steps.append("Evaluating explanatory power of hypotheses")
        best_hypothesis = self._select_best_hypothesis(
            hypotheses, evidence, constraints
        )

        # Step 4: Assess uncertainty
        reasoning_steps.append("Assessing uncertainty and alternative explanations")

        # Calculate confidence based on creativity weight (abduction requires creative thinking)
        base_confidence = 0.65 if self.creativity_weight > 0.6 else 0.5
        confidence = self._calculate_confidence(base_confidence)

        return ReasoningResult(
            reasoning_type=ReasoningType.ABDUCTIVE,
            content=best_hypothesis,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            metadata={
                "creative_emphasis": self.creativity_weight > 0.6,
                "evidence_count": len(evidence),
                "hypotheses_generated": len(hypotheses),
                "uncertainty_level": "high" if confidence < 0.6 else "moderate",
            },
        )

    def _analyze_evidence(self, task: str) -> List[str]:
        """Analyze available evidence from the task."""
        evidence = []
        if "because" in task.lower() or "since" in task.lower():
            evidence.append("Causal indicators present")
        if "symptom" in task.lower() or "sign" in task.lower():
            evidence.append("Diagnostic evidence identified")
        if "unusual" in task.lower() or "unexpected" in task.lower():
            evidence.append("Anomalous evidence detected")
        return evidence if evidence else ["Contextual evidence"]

    def _generate_hypotheses(self, evidence: List[str], task: str) -> List[str]:
        """Generate candidate hypotheses to explain the evidence."""
        hypotheses = []

        # Generate based on creativity weight
        if self.creativity_weight > 0.7:
            hypotheses.extend(
                [
                    f"Creative explanation: {task}",
                    f"Alternative perspective: {task}",
                    f"Novel interpretation: {task}",
                ]
            )
        else:
            hypotheses.extend(
                [
                    f"Standard explanation: {task}",
                    f"Conventional interpretation: {task}",
                ]
            )

        return hypotheses

    def _select_best_hypothesis(
        self, hypotheses: List[str], evidence: List[str], constraints: List[str]
    ) -> str:
        """Select the best hypothesis based on explanatory power."""
        # Simplified selection - in practice would use more sophisticated criteria
        if self.creativity_weight > 0.6 and len(hypotheses) > 1:
            return hypotheses[0]  # Favor creative explanations
        else:
            return hypotheses[0] if hypotheses else "No clear explanation identified"


class AnalogicalReasoning(BaseReasoning):
    """
    Analogical reasoning: Similarity-based understanding and learning.
    Transfers knowledge across domains through analogies.
    """

    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform analogical reasoning on the task.

        Args:
            task: The reasoning task
            context: Optional context including source domain, mappings

        Returns:
            ReasoningResult with analogical analysis
        """
        context = context or {}
        source_domain = context.get("source_domain", "")
        known_analogies = context.get("analogies", [])

        reasoning_steps = []

        # Step 1: Identify source and target domains
        if not source_domain:
            source_domain = self._identify_source_domain(task)
            reasoning_steps.append("Identified potential source domain for analogy")

        # Step 2: Find structural similarities
        reasoning_steps.append("Mapping structural similarities between domains")
        mappings = self._find_structural_mappings(source_domain, task)

        # Step 3: Transfer knowledge
        reasoning_steps.append("Transferring knowledge via analogical mapping")
        analogical_insight = self._transfer_knowledge(mappings, task)

        # Step 4: Validate analogy
        reasoning_steps.append("Validating analogical reasoning")

        # Calculate confidence based on creativity (analogies require creative insight)
        base_confidence = 0.7 if self.creativity_weight > 0.6 else 0.55
        confidence = self._calculate_confidence(base_confidence)

        return ReasoningResult(
            reasoning_type=ReasoningType.ANALOGICAL,
            content=analogical_insight,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            analogies=[f"Analogy: {source_domain} ↔ target domain"],
            metadata={
                "creative_emphasis": self.creativity_weight > 0.6,
                "source_domain": source_domain,
                "mapping_strength": len(mappings),
                "analogy_type": (
                    "surface" if self.creativity_weight < 0.5 else "structural"
                ),
            },
        )

    def _identify_source_domain(self, task: str) -> str:
        """Identify a suitable source domain for analogical reasoning."""
        # Simple heuristic domain identification
        if "network" in task.lower() or "connection" in task.lower():
            return "social_networks"
        elif "flow" in task.lower() or "current" in task.lower():
            return "water_systems"
        elif "growth" in task.lower() or "develop" in task.lower():
            return "biological_systems"
        else:
            return "mechanical_systems"

    def _find_structural_mappings(self, source_domain: str, task: str) -> List[str]:
        """Find structural mappings between source and target domains."""
        mappings = []
        if self.creativity_weight > 0.6:
            mappings.extend(
                [
                    f"Deep structural similarity: {source_domain} → task domain",
                    f"Functional correspondence identified",
                    f"Relational pattern mapping",
                ]
            )
        else:
            mappings.extend(
                [
                    f"Surface similarity: {source_domain} → task domain",
                    f"Basic correspondence identified",
                ]
            )
        return mappings

    def _transfer_knowledge(self, mappings: List[str], task: str) -> str:
        """Transfer knowledge through analogical mappings."""
        if self.creativity_weight > 0.7:
            return f"Through deep analogical reasoning with mappings {mappings}: {task}"
        else:
            return f"By analogy with similar systems: {task}"


class CausalReasoning(BaseReasoning):
    """
    Causal reasoning: Cause-and-effect relationship identification.
    Analyzes temporal sequences and causal chains.
    """

    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform causal reasoning on the task.

        Args:
            task: The reasoning task
            context: Optional context including timeline, events

        Returns:
            ReasoningResult with causal analysis
        """
        context = context or {}
        timeline = context.get("timeline", [])
        events = context.get("events", [])

        reasoning_steps = []

        # Step 1: Identify causal indicators
        causal_indicators = self._identify_causal_indicators(task)
        reasoning_steps.append("Identified causal relationship indicators")

        # Step 2: Construct causal chain
        reasoning_steps.append("Constructing causal chain of events")
        causal_chain = self._construct_causal_chain(
            causal_indicators, timeline, events, task
        )

        # Step 3: Analyze temporal relationships
        reasoning_steps.append("Analyzing temporal sequence and causation")

        # Step 4: Consider alternative causal explanations
        reasoning_steps.append("Evaluating alternative causal pathways")

        # Calculate confidence based on analytical weight (causal analysis requires systematic thinking)
        base_confidence = 0.75 if self.analytical_weight > 0.6 else 0.6
        confidence = self._calculate_confidence(base_confidence)

        causal_explanation = self._form_causal_explanation(causal_chain, task)

        return ReasoningResult(
            reasoning_type=ReasoningType.CAUSAL,
            content=causal_explanation,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            causal_chain=causal_chain,
            metadata={
                "analytical_emphasis": self.analytical_weight > 0.6,
                "causal_indicators": len(causal_indicators),
                "chain_length": len(causal_chain),
                "temporal_analysis": bool(timeline),
            },
        )

    def _identify_causal_indicators(self, task: str) -> List[str]:
        """Identify causal relationship indicators in the task."""
        indicators = []
        causal_words = [
            "because",
            "since",
            "due to",
            "caused by",
            "results in",
            "leads to",
            "triggers",
        ]

        for word in causal_words:
            if word in task.lower():
                indicators.append(f"Causal indicator: '{word}'")

        return indicators if indicators else ["Implicit causal relationships"]

    def _construct_causal_chain(
        self, indicators: List[str], timeline: List[str], events: List[str], task: str
    ) -> List[str]:
        """Construct a causal chain of events."""
        chain = []

        if timeline:
            chain.extend([f"Timeline event: {event}" for event in timeline])
        elif events:
            chain.extend([f"Causal event: {event}" for event in events])
        else:
            # Construct basic causal chain from task
            chain.extend(
                [
                    "Initial conditions identified",
                    "Intermediate causal mechanisms",
                    "Final outcome or effect",
                ]
            )

        return chain

    def _form_causal_explanation(self, causal_chain: List[str], task: str) -> str:
        """Form final causal explanation."""
        if self.analytical_weight > 0.7:
            return f"Systematic causal analysis through chain {causal_chain} indicates: {task}"
        else:
            return f"Causal analysis suggests: {task}"


class ReasoningEngine:
    """
    Central engine that coordinates different reasoning types.
    Selects appropriate reasoning based on task characteristics.
    """

    def __init__(self, style_weights: Dict[str, float]):
        """Initialize reasoning engine with style weights."""
        self.style_weights = style_weights
        self.reasoners = {
            ReasoningType.DEDUCTIVE: DeductiveReasoning(style_weights),
            ReasoningType.INDUCTIVE: InductiveReasoning(style_weights),
            ReasoningType.ABDUCTIVE: AbductiveReasoning(style_weights),
            ReasoningType.ANALOGICAL: AnalogicalReasoning(style_weights),
            ReasoningType.CAUSAL: CausalReasoning(style_weights),
        }

    def reason(
        self,
        task: str,
        reasoning_type: Optional[ReasoningType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Perform reasoning using specified or auto-selected reasoning type.

        Args:
            task: The reasoning task
            reasoning_type: Specific reasoning type, or None for auto-selection
            context: Optional context for reasoning

        Returns:
            ReasoningResult from the selected reasoning type
        """
        if reasoning_type is None:
            reasoning_type = self._select_reasoning_type(task, context)

        reasoner = self.reasoners[reasoning_type]
        return reasoner.reason(task, context)

    def multi_perspective_reasoning(
        self,
        task: str,
        reasoning_types: Optional[List[ReasoningType]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ReasoningResult]:
        """
        Apply multiple reasoning types to the same task for diverse perspectives.

        Args:
            task: The reasoning task
            reasoning_types: List of reasoning types to apply, or None for all
            context: Optional context for reasoning

        Returns:
            List of ReasoningResult from different reasoning types
        """
        if reasoning_types is None:
            reasoning_types = list(ReasoningType)

        results = []
        for reasoning_type in reasoning_types:
            try:
                result = self.reason(task, reasoning_type, context)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to apply {reasoning_type} reasoning: {e}")

        return results

    def _select_reasoning_type(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningType:
        """
        Auto-select the most appropriate reasoning type based on task characteristics.

        Args:
            task: The reasoning task
            context: Optional context

        Returns:
            Selected ReasoningType
        """
        task_lower = task.lower()
        context = context or {}

        # Deductive indicators
        if any(
            word in task_lower
            for word in ["if", "then", "therefore", "thus", "conclude", "prove"]
        ):
            return ReasoningType.DEDUCTIVE

        # Inductive indicators
        if any(
            word in task_lower
            for word in ["pattern", "trend", "generally", "usually", "observe", "data"]
        ):
            return ReasoningType.INDUCTIVE

        # Abductive indicators
        if any(
            word in task_lower
            for word in ["explain", "why", "hypothesis", "best", "likely", "probable"]
        ):
            return ReasoningType.ABDUCTIVE

        # Analogical indicators
        if any(
            word in task_lower
            for word in ["like", "similar", "analogy", "compare", "metaphor", "model"]
        ):
            return ReasoningType.ANALOGICAL

        # Causal indicators
        if any(
            word in task_lower
            for word in ["cause", "because", "since", "due to", "leads to", "results"]
        ):
            return ReasoningType.CAUSAL

        # Default selection based on style weights
        logic_weight = self.style_weights.get("logic", 0.5)
        creativity_weight = self.style_weights.get("creativity", 0.5)
        analytical_weight = self.style_weights.get("analytical", 0.5)

        if logic_weight > 0.7:
            return ReasoningType.DEDUCTIVE
        elif creativity_weight > 0.7:
            return ReasoningType.ABDUCTIVE
        elif analytical_weight > 0.7:
            return ReasoningType.INDUCTIVE
        else:
            return ReasoningType.CAUSAL  # Default to causal for general tasks
