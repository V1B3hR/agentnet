"""
Arbitration System for Multi-Agent Debates

Implements various arbitration strategies including score weighting,
majority vote, and expert judgment for resolving debates and conflicts.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

from .debate import DebateResult, DebatePosition, DebateExchange

logger = logging.getLogger("agentnet.critique.arbitrator")


class ArbitrationStrategy(str, Enum):
    """Different strategies for arbitrating debates."""

    SCORE_WEIGHTING = "score_weighting"
    MAJORITY_VOTE = "majority_vote"
    EXPERT_JUDGMENT = "expert_judgment"
    CONFIDENCE_BASED = "confidence_based"
    EVIDENCE_BASED = "evidence_based"
    HYBRID = "hybrid"


@dataclass
class ArbitrationResult:
    """Result of an arbitration process."""

    arbitration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: ArbitrationStrategy = ArbitrationStrategy.SCORE_WEIGHTING
    timestamp: float = field(default_factory=time.time)

    # Outcome
    winning_position: Optional[str] = None
    winning_agent: Optional[str] = None
    confidence: float = 0.0

    # Scores and reasoning
    position_scores: Dict[str, float] = field(default_factory=dict)
    agent_scores: Dict[str, float] = field(default_factory=dict)
    arbitration_reasoning: str = ""

    # Details
    total_positions_evaluated: int = 0
    consensus_achieved: bool = False
    arbitration_quality: float = 0.0

    # Metadata
    debate_id: str = ""
    arbitrator_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "arbitration_id": self.arbitration_id,
            "strategy": self.strategy.value,
            "timestamp": self.timestamp,
            "winning_position": self.winning_position,
            "winning_agent": self.winning_agent,
            "confidence": self.confidence,
            "position_scores": self.position_scores,
            "agent_scores": self.agent_scores,
            "arbitration_reasoning": self.arbitration_reasoning,
            "total_positions_evaluated": self.total_positions_evaluated,
            "consensus_achieved": self.consensus_achieved,
            "arbitration_quality": self.arbitration_quality,
            "debate_id": self.debate_id,
            "arbitrator_name": self.arbitrator_name,
            "metadata": self.metadata,
        }


class Arbitrator:
    """
    Arbitrator for resolving multi-agent debates and conflicts.

    Implements various arbitration strategies to determine winning positions,
    assess debate quality, and provide final judgment on contentious issues.
    """

    def __init__(
        self,
        name: str = "default_arbitrator",
        default_strategy: ArbitrationStrategy = ArbitrationStrategy.HYBRID,
        score_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.7,
        expert_agent: Optional[Any] = None,
    ):
        """
        Initialize arbitrator.

        Args:
            name: Name identifier for this arbitrator
            default_strategy: Default arbitration strategy
            score_weights: Weights for different scoring criteria
            confidence_threshold: Minimum confidence for decisions
            expert_agent: Optional expert agent for expert judgment
        """
        self.name = name
        self.default_strategy = default_strategy
        self.confidence_threshold = confidence_threshold
        self.expert_agent = expert_agent

        # Default score weights
        self.score_weights = score_weights or {
            "confidence": 0.3,
            "coherence": 0.25,
            "evidence": 0.2,
            "completeness": 0.15,
            "originality": 0.1,
        }

        # Statistics
        self.arbitrations_performed = 0
        self.consensus_achieved_count = 0
        self.high_confidence_decisions = 0
        self.created_time = time.time()

        logger.info(
            f"Arbitrator '{name}' initialized with strategy {default_strategy.value}"
        )

    async def arbitrate_debate(
        self,
        debate: DebateResult,
        strategy: Optional[ArbitrationStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ArbitrationResult:
        """
        Arbitrate a completed debate to determine the outcome.

        Args:
            debate: Completed debate result
            strategy: Arbitration strategy to use (defaults to default_strategy)
            context: Additional context for arbitration

        Returns:
            ArbitrationResult with decision and reasoning
        """
        strategy = strategy or self.default_strategy
        context = context or {}

        # Create arbitration result
        result = ArbitrationResult(
            strategy=strategy,
            debate_id=debate.debate_id,
            arbitrator_name=self.name,
            total_positions_evaluated=len(debate.positions),
            metadata=context,
        )

        try:
            if strategy == ArbitrationStrategy.SCORE_WEIGHTING:
                await self._score_weighting_arbitration(debate, result)

            elif strategy == ArbitrationStrategy.MAJORITY_VOTE:
                await self._majority_vote_arbitration(debate, result)

            elif strategy == ArbitrationStrategy.EXPERT_JUDGMENT:
                await self._expert_judgment_arbitration(debate, result)

            elif strategy == ArbitrationStrategy.CONFIDENCE_BASED:
                await self._confidence_based_arbitration(debate, result)

            elif strategy == ArbitrationStrategy.EVIDENCE_BASED:
                await self._evidence_based_arbitration(debate, result)

            elif strategy == ArbitrationStrategy.HYBRID:
                await self._hybrid_arbitration(debate, result)

            # Update statistics
            self.arbitrations_performed += 1

            if result.consensus_achieved:
                self.consensus_achieved_count += 1

            if result.confidence >= self.confidence_threshold:
                self.high_confidence_decisions += 1

            logger.info(
                f"Arbitration completed: {result.winning_agent} wins with {result.confidence:.2f} confidence"
            )

        except Exception as e:
            logger.error(f"Error in arbitration: {e}")
            result.arbitration_reasoning = f"Arbitration failed: {str(e)}"
            result.confidence = 0.0

        return result

    async def _score_weighting_arbitration(
        self, debate: DebateResult, result: ArbitrationResult
    ) -> None:
        """Arbitrate based on weighted scoring of positions."""
        position_scores = {}

        for position_id, position in debate.positions.items():
            # Calculate weighted score for this position
            score = 0.0

            # Confidence component
            score += position.confidence * self.score_weights.get("confidence", 0.3)

            # Evidence component (based on evidence count)
            evidence_score = min(1.0, len(position.evidence) / 3.0)  # Max at 3 pieces
            score += evidence_score * self.score_weights.get("evidence", 0.2)

            # Completeness component (based on argument count)
            completeness_score = min(
                1.0, len(position.supporting_arguments) / 5.0
            )  # Max at 5 args
            score += completeness_score * self.score_weights.get("completeness", 0.15)

            # Evolution bonus (positions that evolved show engagement)
            if position.revised_times > 0:
                evolution_bonus = min(0.1, position.revised_times * 0.03)
                score += evolution_bonus

            position_scores[position_id] = score

        # Find winning position
        if position_scores:
            winning_position_id = max(position_scores, key=position_scores.get)
            winning_position = debate.positions[winning_position_id]

            result.winning_position = winning_position_id
            result.winning_agent = winning_position.agent_id
            result.position_scores = position_scores
            result.confidence = position_scores[winning_position_id]

            # Build reasoning
            score_details = [
                f"{pos_id}: {score:.2f}" for pos_id, score in position_scores.items()
            ]
            result.arbitration_reasoning = (
                f"Score-weighted arbitration: {', '.join(score_details)}. "
                f"Winner: {winning_position.agent_id} with score {result.confidence:.2f}"
            )
        else:
            result.arbitration_reasoning = "No positions found to evaluate"

    async def _majority_vote_arbitration(
        self, debate: DebateResult, result: ArbitrationResult
    ) -> None:
        """Arbitrate based on majority consensus indicators."""
        # Count exchanges per agent as "votes"
        agent_exchange_counts = {}

        for exchange in debate.exchanges:
            agent_exchange_counts[exchange.speaker_id] = (
                agent_exchange_counts.get(exchange.speaker_id, 0) + 1
            )

        # Look for agreement patterns in exchanges
        agreement_patterns = ["agree", "correct", "yes", "support", "endorse"]
        disagreement_patterns = ["disagree", "wrong", "no", "oppose", "reject"]

        agent_agreement_scores = {}

        for exchange in debate.exchanges:
            content_lower = exchange.content.lower()

            agreement_count = sum(
                1 for pattern in agreement_patterns if pattern in content_lower
            )
            disagreement_count = sum(
                1 for pattern in disagreement_patterns if pattern in content_lower
            )

            # Net agreement score
            net_score = agreement_count - disagreement_count
            agent_agreement_scores[exchange.speaker_id] = (
                agent_agreement_scores.get(exchange.speaker_id, 0) + net_score
            )

        # Determine majority
        if agent_agreement_scores:
            winning_agent = max(agent_agreement_scores, key=agent_agreement_scores.get)
            max_score = agent_agreement_scores[winning_agent]

            result.winning_agent = winning_agent
            result.agent_scores = agent_agreement_scores
            result.confidence = min(1.0, abs(max_score) / len(debate.exchanges))

            result.arbitration_reasoning = (
                f"Majority vote based on agreement patterns. "
                f"Agent scores: {agent_agreement_scores}. Winner: {winning_agent}"
            )
        else:
            result.arbitration_reasoning = "No clear majority found"

    async def _expert_judgment_arbitration(
        self, debate: DebateResult, result: ArbitrationResult
    ) -> None:
        """Arbitrate using expert agent judgment."""
        if not self.expert_agent:
            result.arbitration_reasoning = "No expert agent available for judgment"
            return

        try:
            # Build expert judgment prompt
            prompt = self._build_expert_judgment_prompt(debate)

            # Get expert opinion
            if hasattr(self.expert_agent, "async_generate_reasoning_tree"):
                response = await self.expert_agent.async_generate_reasoning_tree(prompt)
            else:
                import asyncio

                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.expert_agent.generate_reasoning_tree, prompt
                )

            expert_result = response.get("result", {})
            expert_content = expert_result.get("content", "")
            expert_confidence = float(expert_result.get("confidence", 0.0))

            # Parse expert judgment
            await self._parse_expert_judgment(result, expert_content, expert_confidence)

            result.arbitration_reasoning = f"Expert judgment: {expert_content[:200]}..."

        except Exception as e:
            logger.error(f"Error in expert judgment: {e}")
            result.arbitration_reasoning = f"Expert judgment failed: {str(e)}"

    async def _confidence_based_arbitration(
        self, debate: DebateResult, result: ArbitrationResult
    ) -> None:
        """Arbitrate based on agent confidence levels."""
        agent_avg_confidences = {}

        # Calculate average confidence per agent
        for exchange in debate.exchanges:
            agent_id = exchange.speaker_id
            if agent_id not in agent_avg_confidences:
                agent_avg_confidences[agent_id] = []
            agent_avg_confidences[agent_id].append(exchange.confidence)

        # Calculate averages
        for agent_id, confidences in agent_avg_confidences.items():
            agent_avg_confidences[agent_id] = sum(confidences) / len(confidences)

        if agent_avg_confidences:
            winning_agent = max(agent_avg_confidences, key=agent_avg_confidences.get)

            result.winning_agent = winning_agent
            result.agent_scores = agent_avg_confidences
            result.confidence = agent_avg_confidences[winning_agent]

            result.arbitration_reasoning = (
                f"Confidence-based arbitration. Average confidences: {agent_avg_confidences}. "
                f"Winner: {winning_agent} with {result.confidence:.2f} confidence"
            )
        else:
            result.arbitration_reasoning = "No confidence data available"

    async def _evidence_based_arbitration(
        self, debate: DebateResult, result: ArbitrationResult
    ) -> None:
        """Arbitrate based on evidence quality and quantity."""
        agent_evidence_scores = {}

        # Score based on position evidence
        for position in debate.positions.values():
            agent_id = position.agent_id

            # Evidence quantity score
            evidence_quantity = len(position.evidence)

            # Argument quality score (based on supporting arguments)
            argument_quality = len(position.supporting_arguments)

            # Combined evidence score
            evidence_score = (evidence_quantity * 0.6) + (argument_quality * 0.4)

            agent_evidence_scores[agent_id] = (
                agent_evidence_scores.get(agent_id, 0) + evidence_score
            )

        if agent_evidence_scores:
            winning_agent = max(agent_evidence_scores, key=agent_evidence_scores.get)
            max_score = agent_evidence_scores[winning_agent]

            # Normalize confidence
            total_score = sum(agent_evidence_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.0

            result.winning_agent = winning_agent
            result.agent_scores = agent_evidence_scores
            result.confidence = confidence

            result.arbitration_reasoning = (
                f"Evidence-based arbitration. Evidence scores: {agent_evidence_scores}. "
                f"Winner: {winning_agent}"
            )
        else:
            result.arbitration_reasoning = "No evidence data available"

    async def _hybrid_arbitration(
        self, debate: DebateResult, result: ArbitrationResult
    ) -> None:
        """Arbitrate using hybrid approach combining multiple strategies."""
        # Collect results from multiple strategies
        strategies_results = {}

        # Score weighting
        score_result = ArbitrationResult(strategy=ArbitrationStrategy.SCORE_WEIGHTING)
        await self._score_weighting_arbitration(debate, score_result)
        strategies_results["score"] = score_result

        # Confidence based
        confidence_result = ArbitrationResult(
            strategy=ArbitrationStrategy.CONFIDENCE_BASED
        )
        await self._confidence_based_arbitration(debate, confidence_result)
        strategies_results["confidence"] = confidence_result

        # Evidence based
        evidence_result = ArbitrationResult(strategy=ArbitrationStrategy.EVIDENCE_BASED)
        await self._evidence_based_arbitration(debate, evidence_result)
        strategies_results["evidence"] = evidence_result

        # Aggregate results
        agent_votes = {}
        total_confidence = 0.0

        for strategy_name, strategy_result in strategies_results.items():
            if strategy_result.winning_agent:
                agent_votes[strategy_result.winning_agent] = (
                    agent_votes.get(strategy_result.winning_agent, 0) + 1
                )
                total_confidence += strategy_result.confidence

        # Determine winner
        if agent_votes:
            winning_agent = max(agent_votes, key=agent_votes.get)
            vote_count = agent_votes[winning_agent]

            result.winning_agent = winning_agent
            result.confidence = total_confidence / len(strategies_results)

            # Check for consensus
            if vote_count == len(strategies_results):
                result.consensus_achieved = True

            result.arbitration_reasoning = (
                f"Hybrid arbitration combining score weighting, confidence, and evidence analysis. "
                f"Agent votes: {agent_votes}. Winner: {winning_agent} with {vote_count}/{len(strategies_results)} votes"
            )

            # Store sub-results in metadata
            result.metadata["sub_results"] = {
                name: res.to_dict() for name, res in strategies_results.items()
            }
        else:
            result.arbitration_reasoning = "No clear winner from hybrid analysis"

    def _build_expert_judgment_prompt(self, debate: DebateResult) -> str:
        """Build prompt for expert judgment."""
        # Summarize debate
        summary_parts = [
            f"Topic: {debate.topic}",
            f"Participants: {list(debate.participants.keys())}",
            f"Total exchanges: {len(debate.exchanges)}",
        ]

        # Add key positions
        if debate.positions:
            summary_parts.append("Positions:")
            for pos_id, position in debate.positions.items():
                summary_parts.append(
                    f"- {position.agent_id}: {position.statement[:200]}..."
                )

        # Add recent exchanges
        recent_exchanges = (
            debate.exchanges[-4:] if len(debate.exchanges) > 4 else debate.exchanges
        )
        if recent_exchanges:
            summary_parts.append("Recent exchanges:")
            for exchange in recent_exchanges:
                summary_parts.append(
                    f"- {exchange.speaker_id}: {exchange.content[:100]}..."
                )

        prompt = (
            "As an expert judge, evaluate this debate and determine the winner.\n\n"
            + "\n".join(summary_parts)
            + "\n\nProvide your judgment on:\n"
            "1. Which participant presented the strongest case\n"
            "2. Quality of arguments and evidence\n"
            "3. Your confidence in this decision (0-1)\n"
            "4. Brief reasoning for your decision"
        )

        return prompt

    async def _parse_expert_judgment(
        self, result: ArbitrationResult, expert_content: str, expert_confidence: float
    ) -> None:
        """Parse expert judgment to extract decision."""
        content_lower = expert_content.lower()

        # Look for winner indicators
        agent_names = [name.lower() for name in result.metadata.get("agent_names", [])]

        winner_found = False
        for agent_name in agent_names:
            if (
                f"winner: {agent_name}" in content_lower
                or f"{agent_name} wins" in content_lower
            ):
                result.winning_agent = agent_name
                winner_found = True
                break

        # If no explicit winner, use confidence as tie-breaker
        if not winner_found and agent_names:
            result.winning_agent = agent_names[0]  # Default to first agent

        result.confidence = expert_confidence

    def get_stats(self) -> Dict[str, Any]:
        """Get arbitrator statistics."""
        consensus_rate = (
            self.consensus_achieved_count / self.arbitrations_performed
            if self.arbitrations_performed > 0
            else 0.0
        )

        high_confidence_rate = (
            self.high_confidence_decisions / self.arbitrations_performed
            if self.arbitrations_performed > 0
            else 0.0
        )

        return {
            "name": self.name,
            "arbitrations_performed": self.arbitrations_performed,
            "consensus_achieved_count": self.consensus_achieved_count,
            "consensus_rate": consensus_rate,
            "high_confidence_decisions": self.high_confidence_decisions,
            "high_confidence_rate": high_confidence_rate,
            "default_strategy": self.default_strategy.value,
            "confidence_threshold": self.confidence_threshold,
            "score_weights": self.score_weights,
        }
