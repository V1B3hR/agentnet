"""
Reasoning-aware style modulation for AgentNet.

Integrates reasoning types with existing style influence system to provide
more sophisticated cognitive modulation based on reasoning requirements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .types import ReasoningResult, ReasoningType

logger = logging.getLogger("agentnet.reasoning.modulation")


class ReasoningStyleModulator:
    """
    Modulates agent style weights based on reasoning type requirements.

    Different reasoning types benefit from different cognitive emphases:
    - Deductive: High logic, moderate analytical
    - Inductive: High analytical, moderate logic
    - Abductive: High creativity, moderate analytical
    - Analogical: High creativity, moderate logic
    - Causal: High analytical, moderate logic
    """

    def __init__(self):
        """Initialize the reasoning style modulator."""
        # Optimal style profiles for each reasoning type
        self.reasoning_profiles = {
            ReasoningType.DEDUCTIVE: {
                "logic": 0.85,
                "creativity": 0.3,
                "analytical": 0.7,
            },
            ReasoningType.INDUCTIVE: {
                "logic": 0.6,
                "creativity": 0.4,
                "analytical": 0.9,
            },
            ReasoningType.ABDUCTIVE: {
                "logic": 0.5,
                "creativity": 0.9,
                "analytical": 0.6,
            },
            ReasoningType.ANALOGICAL: {
                "logic": 0.4,
                "creativity": 0.85,
                "analytical": 0.5,
            },
            ReasoningType.CAUSAL: {"logic": 0.7, "creativity": 0.4, "analytical": 0.8},
        }

    def modulate_style_for_reasoning(
        self,
        base_style: Dict[str, float],
        reasoning_type: ReasoningType,
        modulation_strength: float = 0.5,
    ) -> Dict[str, float]:
        """
        Modulate base style weights for optimal reasoning performance.

        Args:
            base_style: Agent's base style weights
            reasoning_type: The reasoning type being applied
            modulation_strength: How much to adjust towards optimal (0.0-1.0)

        Returns:
            Modulated style weights dictionary
        """
        if reasoning_type not in self.reasoning_profiles:
            logger.warning(f"Unknown reasoning type: {reasoning_type}")
            return base_style.copy()

        optimal_profile = self.reasoning_profiles[reasoning_type]
        modulated_style = {}

        for dimension, base_value in base_style.items():
            if dimension in optimal_profile:
                optimal_value = optimal_profile[dimension]
                # Weighted average between base and optimal
                modulated_value = (
                    base_value * (1 - modulation_strength)
                    + optimal_value * modulation_strength
                )
                modulated_style[dimension] = min(1.0, max(0.0, modulated_value))
            else:
                modulated_style[dimension] = base_value

        logger.debug(
            f"Style modulated for {reasoning_type}: {base_style} → {modulated_style}"
        )
        return modulated_style

    def suggest_reasoning_types(self, style: Dict[str, float]) -> List[ReasoningType]:
        """
        Suggest optimal reasoning types based on agent's style profile.

        Args:
            style: Agent's style weights

        Returns:
            List of recommended reasoning types, ordered by fitness
        """
        logic_weight = style.get("logic", 0.5)
        creativity_weight = style.get("creativity", 0.5)
        analytical_weight = style.get("analytical", 0.5)

        suggestions = []

        # Score each reasoning type based on style alignment
        scores = {}
        for reasoning_type, profile in self.reasoning_profiles.items():
            score = (
                abs(logic_weight - profile["logic"]) * -1
                + abs(creativity_weight - profile["creativity"]) * -1
                + abs(analytical_weight - profile["analytical"]) * -1
            )
            scores[reasoning_type] = score

        # Sort by score (higher is better, less negative)
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [reasoning_type for reasoning_type, _ in sorted_types]


class ReasoningAwareStyleInfluence:
    """
    Enhanced style influence system that incorporates reasoning type awareness.

    Extends the existing AgentNet style influence mechanism to dynamically
    adjust cognitive emphasis based on the reasoning being performed.
    """

    def __init__(self, modulator: Optional[ReasoningStyleModulator] = None):
        """Initialize with optional custom modulator."""
        self.modulator = modulator or ReasoningStyleModulator()

    def apply_reasoning_aware_style_influence(
        self,
        base_result: Dict[str, Any],
        task: str,
        base_style: Dict[str, float],
        reasoning_result: Optional[ReasoningResult] = None,
        modulation_strength: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Apply style influence with reasoning type awareness.

        Args:
            base_result: Base result from inference engine
            task: The reasoning task
            base_style: Agent's base style weights
            reasoning_result: Optional reasoning result for enhanced modulation
            modulation_strength: Strength of reasoning-based modulation

        Returns:
            Enhanced result with reasoning-aware style influence
        """
        styled_result = dict(base_result)

        # If we have reasoning result, use it for targeted modulation
        if reasoning_result:
            modulated_style = self.modulator.modulate_style_for_reasoning(
                base_style, reasoning_result.reasoning_type, modulation_strength
            )

            # Apply reasoning-specific enhancements
            styled_result = self._apply_reasoning_specific_enhancements(
                styled_result, reasoning_result, modulated_style
            )
        else:
            # Fallback to base style application
            modulated_style = base_style

        # Apply general style influence with modulated weights
        styled_result = self._apply_general_style_influence(
            styled_result, modulated_style, task
        )

        # Add reasoning metadata
        if reasoning_result:
            styled_result["reasoning_metadata"] = {
                "reasoning_type": reasoning_result.reasoning_type.value,
                "reasoning_confidence": reasoning_result.confidence,
                "reasoning_steps_count": len(reasoning_result.reasoning_steps),
                "style_modulation_applied": True,
                "modulated_style": modulated_style,
            }

        return styled_result

    def _apply_reasoning_specific_enhancements(
        self,
        result: Dict[str, Any],
        reasoning_result: ReasoningResult,
        modulated_style: Dict[str, float],
    ) -> Dict[str, Any]:
        """Apply reasoning-type-specific enhancements to the result."""
        content = result.get("content", "")
        confidence = result.get("confidence", 0.5)

        # Enhance confidence based on reasoning type and style alignment
        reasoning_confidence = reasoning_result.confidence
        style_alignment_bonus = self._calculate_style_alignment_bonus(
            reasoning_result.reasoning_type, modulated_style
        )

        enhanced_confidence = (
            confidence + reasoning_confidence + style_alignment_bonus
        ) / 3.0
        enhanced_confidence = min(1.0, max(0.1, enhanced_confidence))

        # Add reasoning-specific insights
        insights = result.get("style_insights", [])

        if reasoning_result.reasoning_type == ReasoningType.DEDUCTIVE:
            if modulated_style.get("logic", 0.5) > 0.7:
                insights.append("Applied rigorous deductive logic validation")
        elif reasoning_result.reasoning_type == ReasoningType.INDUCTIVE:
            if modulated_style.get("analytical", 0.5) > 0.7:
                insights.append("Conducted systematic pattern analysis")
        elif reasoning_result.reasoning_type == ReasoningType.ABDUCTIVE:
            if modulated_style.get("creativity", 0.5) > 0.7:
                insights.append("Generated creative explanatory hypotheses")
        elif reasoning_result.reasoning_type == ReasoningType.ANALOGICAL:
            if modulated_style.get("creativity", 0.5) > 0.7:
                insights.append("Developed innovative analogical mappings")
        elif reasoning_result.reasoning_type == ReasoningType.CAUSAL:
            if modulated_style.get("analytical", 0.5) > 0.7:
                insights.append("Analyzed systematic causal relationships")

        # Integrate reasoning steps as insights
        if reasoning_result.reasoning_steps:
            insights.extend(
                [f"Reasoning: {step}" for step in reasoning_result.reasoning_steps[:2]]
            )

        result.update(
            {
                "content": content,
                "confidence": enhanced_confidence,
                "style_insights": insights,
                "reasoning_enhanced": True,
            }
        )

        return result

    def _apply_general_style_influence(
        self, result: Dict[str, Any], style: Dict[str, float], task: str
    ) -> Dict[str, Any]:
        """Apply general style influence using modulated weights."""
        logic_weight = style.get("logic", 0.5)
        creativity_weight = style.get("creativity", 0.5)
        analytical_weight = style.get("analytical", 0.5)

        # Adjust confidence based on style coherence
        confidence = result.get("confidence", 0.5)
        style_influence = (logic_weight + analytical_weight) / 2.0
        adjusted_confidence = confidence * (0.8 + 0.4 * style_influence)
        adjusted_confidence = min(1.0, max(0.1, adjusted_confidence))

        result["confidence"] = adjusted_confidence
        result["style_applied"] = True
        result["style_influence"] = style_influence

        return result

    def _calculate_style_alignment_bonus(
        self, reasoning_type: ReasoningType, style: Dict[str, float]
    ) -> float:
        """Calculate bonus based on how well style aligns with reasoning type."""
        optimal_profile = self.modulator.reasoning_profiles.get(reasoning_type, {})

        if not optimal_profile:
            return 0.0

        alignment_score = 0.0
        dimensions = ["logic", "creativity", "analytical"]

        for dimension in dimensions:
            optimal_value = optimal_profile.get(dimension, 0.5)
            actual_value = style.get(dimension, 0.5)
            # Alignment decreases with distance from optimal
            alignment = 1.0 - abs(optimal_value - actual_value)
            alignment_score += alignment

        # Average alignment across dimensions, scaled to reasonable bonus range
        average_alignment = alignment_score / len(dimensions)
        return average_alignment * 0.2  # Max bonus of 0.2

    def create_reasoning_context(
        self,
        task: str,
        agent_style: Dict[str, float],
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create enhanced context for reasoning that includes style information.

        Args:
            task: The reasoning task
            agent_style: Agent's style profile
            memory_context: Optional memory context

        Returns:
            Enhanced context dictionary for reasoning
        """
        context = {
            "agent_style": agent_style,
            "suggested_reasoning_types": self.modulator.suggest_reasoning_types(
                agent_style
            ),
            "style_based_preferences": self._extract_style_preferences(agent_style),
        }

        if memory_context:
            context["memory_context"] = memory_context

        return context

    def _extract_style_preferences(self, style: Dict[str, float]) -> Dict[str, Any]:
        """Extract reasoning preferences based on style profile."""
        logic_weight = style.get("logic", 0.5)
        creativity_weight = style.get("creativity", 0.5)
        analytical_weight = style.get("analytical", 0.5)

        preferences = {
            "prefers_systematic_analysis": analytical_weight > 0.7,
            "prefers_creative_solutions": creativity_weight > 0.7,
            "prefers_logical_rigor": logic_weight > 0.7,
            "reasoning_depth": (
                "deep" if max(logic_weight, analytical_weight) > 0.8 else "moderate"
            ),
            "uncertainty_tolerance": "high" if creativity_weight > 0.6 else "moderate",
        }

        return preferences
