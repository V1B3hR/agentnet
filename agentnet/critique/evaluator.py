"""
Critique and Revision Evaluators

Implements self and cross-agent critique capabilities with revision triggers.
Provides scoring mechanisms for content quality, truthiness, and risk assessment.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger("agentnet.critique.evaluator")


class CritiqueType(str, Enum):
    """Types of critique that can be performed."""

    SELF_CRITIQUE = "self_critique"
    PEER_CRITIQUE = "peer_critique"
    EXPERT_CRITIQUE = "expert_critique"
    AUTOMATED_CRITIQUE = "automated_critique"


class RevisionTrigger(str, Enum):
    """Triggers that can initiate a revision process."""

    LOW_CONFIDENCE = "low_confidence"
    CRITIQUE_FEEDBACK = "critique_feedback"
    POLICY_VIOLATION = "policy_violation"
    QUALITY_THRESHOLD = "quality_threshold"
    MANUAL_REQUEST = "manual_request"


@dataclass
class CritiqueResult:
    """Result of a critique evaluation."""

    critique_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    critique_type: CritiqueType = CritiqueType.AUTOMATED_CRITIQUE
    timestamp: float = field(default_factory=time.time)

    # Scores (0.0 to 1.0)
    quality_score: float = 0.0
    truthiness_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0

    # Overall assessment
    overall_score: float = 0.0
    needs_revision: bool = False
    revision_triggers: List[RevisionTrigger] = field(default_factory=list)

    # Feedback
    critique_text: str = ""
    suggestions: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    # Context
    original_content: str = ""
    critiqued_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "critique_id": self.critique_id,
            "critique_type": self.critique_type.value,
            "timestamp": self.timestamp,
            "scores": {
                "quality": self.quality_score,
                "truthiness": self.truthiness_score,
                "coherence": self.coherence_score,
                "completeness": self.completeness_score,
                "overall": self.overall_score,
            },
            "needs_revision": self.needs_revision,
            "revision_triggers": [t.value for t in self.revision_triggers],
            "feedback": {
                "critique_text": self.critique_text,
                "suggestions": self.suggestions,
                "strengths": self.strengths,
                "weaknesses": self.weaknesses,
            },
            "context": {
                "original_content": self.original_content,
                "critiqued_by": self.critiqued_by,
                "metadata": self.metadata,
            },
        }


class CritiqueEvaluator:
    """
    Evaluator for providing critique and feedback on agent outputs.

    Can perform self-critique, peer critique, or expert critique
    with configurable quality thresholds and revision triggers.
    """

    def __init__(
        self,
        name: str = "default_critic",
        quality_threshold: float = 0.7,
        truthiness_threshold: float = 0.6,
        coherence_threshold: float = 0.8,
        completeness_threshold: float = 0.7,
        enable_automated_scoring: bool = True,
        critique_agent: Optional[Any] = None,  # AgentNet instance for critique
    ):
        """
        Initialize the critique evaluator.

        Args:
            name: Name identifier for this evaluator
            quality_threshold: Minimum quality score to avoid revision
            truthiness_threshold: Minimum truthiness score
            coherence_threshold: Minimum coherence score
            completeness_threshold: Minimum completeness score
            enable_automated_scoring: Whether to use automated scoring
            critique_agent: Optional agent for generating critiques
        """
        self.name = name
        self.quality_threshold = quality_threshold
        self.truthiness_threshold = truthiness_threshold
        self.coherence_threshold = coherence_threshold
        self.completeness_threshold = completeness_threshold
        self.enable_automated_scoring = enable_automated_scoring
        self.critique_agent = critique_agent

        # Statistics
        self.critiques_performed = 0
        self.revisions_triggered = 0
        self.average_quality_score = 0.0
        self.created_time = time.time()

        logger.info(f"CritiqueEvaluator '{name}' initialized")

    async def evaluate_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        critique_type: CritiqueType = CritiqueType.AUTOMATED_CRITIQUE,
        critiqued_by: str = "system",
    ) -> CritiqueResult:
        """
        Evaluate content and provide critique.

        Args:
            content: Content to critique
            context: Additional context for evaluation
            critique_type: Type of critique to perform
            critiqued_by: Who is performing the critique

        Returns:
            CritiqueResult with scores and feedback
        """
        context = context or {}

        # Create critique result
        critique = CritiqueResult(
            critique_type=critique_type,
            original_content=content,
            critiqued_by=critiqued_by,
            metadata=context,
        )

        try:
            # Perform scoring
            if self.enable_automated_scoring:
                await self._automated_scoring(critique, content, context)

            # Generate critique text if agent available
            if self.critique_agent:
                await self._agent_critique(critique, content, context)
            else:
                await self._heuristic_critique(critique, content, context)

            # Determine if revision is needed
            self._determine_revision_needs(critique)

            # Update statistics
            self.critiques_performed += 1
            if critique.needs_revision:
                self.revisions_triggered += 1

            # Update average quality score
            total_score = (
                self.average_quality_score * (self.critiques_performed - 1)
                + critique.overall_score
            ) / self.critiques_performed
            self.average_quality_score = total_score

            logger.debug(
                f"Critique completed: {critique.overall_score:.2f} score, revision={critique.needs_revision}"
            )

        except Exception as e:
            logger.error(f"Error in critique evaluation: {e}")
            critique.critique_text = f"Critique evaluation failed: {str(e)}"
            critique.needs_revision = True
            critique.revision_triggers.append(RevisionTrigger.MANUAL_REQUEST)

        return critique

    async def _automated_scoring(
        self, critique: CritiqueResult, content: str, context: Dict[str, Any]
    ) -> None:
        """Perform automated scoring using heuristics."""
        # Quality score - based on length, structure, keywords
        critique.quality_score = await self._calculate_quality_score(content, context)

        # Truthiness score - basic heuristics for factual claims
        critique.truthiness_score = await self._calculate_truthiness_score(
            content, context
        )

        # Coherence score - logical flow and consistency
        critique.coherence_score = await self._calculate_coherence_score(
            content, context
        )

        # Completeness score - addresses the prompt/question
        critique.completeness_score = await self._calculate_completeness_score(
            content, context
        )

        # Overall score - weighted average
        critique.overall_score = (
            critique.quality_score * 0.3
            + critique.truthiness_score * 0.2
            + critique.coherence_score * 0.3
            + critique.completeness_score * 0.2
        )

    async def _calculate_quality_score(
        self, content: str, context: Dict[str, Any]
    ) -> float:
        """Calculate content quality score."""
        score = 0.5  # Base score

        # Length consideration
        if 50 <= len(content) <= 2000:
            score += 0.2
        elif len(content) < 20:
            score -= 0.3

        # Structure indicators
        if any(punct in content for punct in [".", "!", "?"]):
            score += 0.1

        # Complexity/sophistication
        sentences = content.split(".")
        if len(sentences) > 2:
            score += 0.1

        # Confidence from context
        confidence = context.get("confidence", 0.5)
        score += confidence * 0.1

        return min(1.0, max(0.0, score))

    async def _calculate_truthiness_score(
        self, content: str, context: Dict[str, Any]
    ) -> float:
        """Calculate truthiness/factual accuracy score."""
        score = 0.7  # Default moderate score

        # Look for uncertainty markers
        uncertainty_words = [
            "might",
            "could",
            "possibly",
            "perhaps",
            "maybe",
            "uncertain",
        ]
        if any(word in content.lower() for word in uncertainty_words):
            score += 0.1  # Honest uncertainty is good

        # Look for definitive claims without support
        definitive_words = ["definitely", "certainly", "absolutely", "always", "never"]
        if any(word in content.lower() for word in definitive_words):
            score -= 0.1  # Overconfident claims are risky

        # Look for citations or references
        if any(
            marker in content for marker in ["[", "]", "according to", "research shows"]
        ):
            score += 0.2

        return min(1.0, max(0.0, score))

    async def _calculate_coherence_score(
        self, content: str, context: Dict[str, Any]
    ) -> float:
        """Calculate logical coherence score."""
        score = 0.6  # Base score

        # Check for logical connectors
        connectors = [
            "therefore",
            "however",
            "furthermore",
            "consequently",
            "because",
            "since",
        ]
        connector_count = sum(1 for conn in connectors if conn in content.lower())
        score += min(0.2, connector_count * 0.05)

        # Check for contradictions (simple heuristic)
        contradiction_patterns = [
            ("yes", "no"),
            ("true", "false"),
            ("always", "never"),
            ("increase", "decrease"),
            ("more", "less"),
        ]

        content_lower = content.lower()
        for word1, word2 in contradiction_patterns:
            if word1 in content_lower and word2 in content_lower:
                score -= 0.1

        return min(1.0, max(0.0, score))

    async def _calculate_completeness_score(
        self, content: str, context: Dict[str, Any]
    ) -> float:
        """Calculate completeness score."""
        score = 0.5  # Base score

        # Check if prompt context is available
        prompt = context.get("prompt", "")
        if prompt:
            # Simple keyword overlap
            prompt_words = set(prompt.lower().split())
            content_words = set(content.lower().split())

            if prompt_words:
                overlap = len(prompt_words & content_words) / len(prompt_words)
                score += overlap * 0.3

        # Check for conclusion markers
        conclusion_words = ["in conclusion", "therefore", "to summarize", "overall"]
        if any(word in content.lower() for word in conclusion_words):
            score += 0.2

        return min(1.0, max(0.0, score))

    async def _agent_critique(
        self, critique: CritiqueResult, content: str, context: Dict[str, Any]
    ) -> None:
        """Generate critique using an agent."""
        try:
            critique_prompt = self._build_critique_prompt(content, context)

            if hasattr(self.critique_agent, "async_generate_reasoning_tree"):
                response = await self.critique_agent.async_generate_reasoning_tree(
                    critique_prompt
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.critique_agent.generate_reasoning_tree, critique_prompt
                )

            result = response.get("result", {})
            critique.critique_text = result.get("content", "No critique generated")

            # Extract structured feedback if possible
            await self._parse_agent_critique(critique, critique.critique_text)

        except Exception as e:
            logger.error(f"Error in agent critique: {e}")
            critique.critique_text = f"Agent critique failed: {str(e)}"

    async def _heuristic_critique(
        self, critique: CritiqueResult, content: str, context: Dict[str, Any]
    ) -> None:
        """Generate critique using heuristics."""
        feedback_parts = []

        # Analyze strengths
        if critique.quality_score > 0.7:
            critique.strengths.append("Well-structured and comprehensive response")
        if critique.coherence_score > 0.8:
            critique.strengths.append("Logically coherent and well-reasoned")
        if critique.truthiness_score > 0.7:
            critique.strengths.append(
                "Appears factually sound with appropriate caveats"
            )

        # Analyze weaknesses
        if critique.quality_score < 0.5:
            critique.weaknesses.append("Response lacks depth and detail")
            critique.suggestions.append(
                "Provide more comprehensive analysis and examples"
            )

        if critique.truthiness_score < 0.6:
            critique.weaknesses.append("Contains potentially unsupported claims")
            critique.suggestions.append("Add citations or qualify uncertain statements")

        if critique.coherence_score < 0.6:
            critique.weaknesses.append("Logic flow could be improved")
            critique.suggestions.append(
                "Use clearer transitions and logical connectors"
            )

        if critique.completeness_score < 0.6:
            critique.weaknesses.append("Does not fully address the prompt")
            critique.suggestions.append(
                "Ensure all aspects of the question are covered"
            )

        # Build critique text
        if critique.strengths:
            feedback_parts.append(f"Strengths: {'; '.join(critique.strengths)}")

        if critique.weaknesses:
            feedback_parts.append(
                f"Areas for improvement: {'; '.join(critique.weaknesses)}"
            )

        if critique.suggestions:
            feedback_parts.append(f"Suggestions: {'; '.join(critique.suggestions)}")

        critique.critique_text = (
            " | ".join(feedback_parts) if feedback_parts else "No specific feedback."
        )

    def _build_critique_prompt(self, content: str, context: Dict[str, Any]) -> str:
        """Build prompt for agent-based critique."""
        prompt_parts = [
            "Please provide a detailed critique of the following response:",
            f"\nOriginal Content: {content}",
            "\nEvaluate the response on:",
            "1. Quality and depth of analysis",
            "2. Factual accuracy and truthiness",
            "3. Logical coherence and flow",
            "4. Completeness in addressing the topic",
            "\nProvide specific strengths, weaknesses, and suggestions for improvement.",
        ]

        if "prompt" in context:
            prompt_parts.insert(-1, f"\nOriginal Prompt: {context['prompt']}")

        return "\n".join(prompt_parts)

    async def _parse_agent_critique(
        self, critique: CritiqueResult, critique_text: str
    ) -> None:
        """Parse structured feedback from agent critique text."""
        # Simple parsing for structured feedback
        lines = critique_text.lower().split("\n")

        current_section = None
        for line in lines:
            line = line.strip()

            if "strength" in line:
                current_section = "strengths"
            elif "weakness" in line or "improvement" in line:
                current_section = "weaknesses"
            elif "suggestion" in line or "recommend" in line:
                current_section = "suggestions"
            elif line.startswith("- ") or line.startswith("* "):
                item = line[2:].strip()
                if current_section == "strengths":
                    critique.strengths.append(item)
                elif current_section == "weaknesses":
                    critique.weaknesses.append(item)
                elif current_section == "suggestions":
                    critique.suggestions.append(item)

    def _determine_revision_needs(self, critique: CritiqueResult) -> None:
        """Determine if revision is needed based on scores and thresholds."""
        triggers = []

        # Check individual score thresholds
        if critique.quality_score < self.quality_threshold:
            triggers.append(RevisionTrigger.QUALITY_THRESHOLD)

        if critique.truthiness_score < self.truthiness_threshold:
            triggers.append(RevisionTrigger.QUALITY_THRESHOLD)

        if critique.coherence_score < self.coherence_threshold:
            triggers.append(RevisionTrigger.QUALITY_THRESHOLD)

        if critique.completeness_score < self.completeness_threshold:
            triggers.append(RevisionTrigger.QUALITY_THRESHOLD)

        # Check overall confidence
        confidence = critique.metadata.get("confidence", 1.0)
        if confidence < 0.5:
            triggers.append(RevisionTrigger.LOW_CONFIDENCE)

        # Check for multiple weaknesses
        if len(critique.weaknesses) >= 2:
            triggers.append(RevisionTrigger.CRITIQUE_FEEDBACK)

        critique.revision_triggers = triggers
        critique.needs_revision = len(triggers) > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        uptime = time.time() - self.created_time
        revision_rate = (
            self.revisions_triggered / self.critiques_performed
            if self.critiques_performed > 0
            else 0.0
        )

        return {
            "name": self.name,
            "uptime": uptime,
            "critiques_performed": self.critiques_performed,
            "revisions_triggered": self.revisions_triggered,
            "revision_rate": revision_rate,
            "average_quality_score": self.average_quality_score,
            "thresholds": {
                "quality": self.quality_threshold,
                "truthiness": self.truthiness_threshold,
                "coherence": self.coherence_threshold,
                "completeness": self.completeness_threshold,
            },
        }


class RevisionEvaluator:
    """
    Evaluator for managing the revision process.

    Coordinates between original content, critique feedback, and revised content
    to determine if revisions successfully address identified issues.
    """

    def __init__(
        self,
        name: str = "revision_evaluator",
        max_revision_cycles: int = 3,
        improvement_threshold: float = 0.1,
    ):
        """
        Initialize revision evaluator.

        Args:
            name: Name identifier
            max_revision_cycles: Maximum number of revision cycles
            improvement_threshold: Minimum improvement needed to continue
        """
        self.name = name
        self.max_revision_cycles = max_revision_cycles
        self.improvement_threshold = improvement_threshold

        self.revisions_evaluated = 0
        self.successful_revisions = 0
        self.created_time = time.time()

        logger.info(f"RevisionEvaluator '{name}' initialized")

    async def evaluate_revision(
        self,
        original_critique: CritiqueResult,
        revised_content: str,
        revision_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, CritiqueResult]:
        """
        Evaluate if a revision successfully addresses critique feedback.

        Args:
            original_critique: Original critique that triggered revision
            revised_content: The revised content
            revision_context: Additional context for revision

        Returns:
            Tuple of (revision_successful, new_critique_result)
        """
        revision_context = revision_context or {}

        # Create a new evaluator for the revision
        temp_evaluator = CritiqueEvaluator(
            name=f"{self.name}_temp",
            quality_threshold=0.0,  # Don't trigger new revisions automatically
            enable_automated_scoring=True,
        )

        # Evaluate revised content
        new_critique = await temp_evaluator.evaluate_content(
            content=revised_content,
            context=revision_context,
            critique_type=CritiqueType.AUTOMATED_CRITIQUE,
            critiqued_by=f"{self.name}_revision_check",
        )

        # Compare scores
        improvement = new_critique.overall_score - original_critique.overall_score
        revision_successful = improvement >= self.improvement_threshold

        # Update statistics
        self.revisions_evaluated += 1
        if revision_successful:
            self.successful_revisions += 1

        # Add revision context to new critique
        new_critique.metadata.update(
            {
                "revision_cycle": revision_context.get("revision_cycle", 1),
                "improvement": improvement,
                "original_score": original_critique.overall_score,
                "revision_successful": revision_successful,
            }
        )

        logger.debug(
            f"Revision evaluation: {improvement:.2f} improvement, successful={revision_successful}"
        )

        return revision_successful, new_critique

    def get_stats(self) -> Dict[str, Any]:
        """Get revision evaluator statistics."""
        success_rate = (
            self.successful_revisions / self.revisions_evaluated
            if self.revisions_evaluated > 0
            else 0.0
        )

        return {
            "name": self.name,
            "revisions_evaluated": self.revisions_evaluated,
            "successful_revisions": self.successful_revisions,
            "success_rate": success_rate,
            "max_revision_cycles": self.max_revision_cycles,
            "improvement_threshold": self.improvement_threshold,
        }
