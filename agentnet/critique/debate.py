"""
Debate Management System

Implements structured debate mechanisms between agents including
analyst vs critic debates, structured argumentation, and position evolution.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentnet.critique.debate")


class DebateRole(str, Enum):
    """Roles agents can take in a debate."""

    PROPONENT = "proponent"  # Argues for the position
    OPPONENT = "opponent"  # Argues against the position
    ANALYST = "analyst"  # Provides analytical perspective
    CRITIC = "critic"  # Provides critical evaluation
    MODERATOR = "moderator"  # Facilitates the debate
    OBSERVER = "observer"  # Observes and summarizes


class DebatePhase(str, Enum):
    """Phases of a structured debate."""

    OPENING_STATEMENTS = "opening_statements"
    ARGUMENTATION = "argumentation"
    CROSS_EXAMINATION = "cross_examination"
    REBUTTAL = "rebuttal"
    CLOSING_STATEMENTS = "closing_statements"
    DELIBERATION = "deliberation"
    RESOLUTION = "resolution"


@dataclass
class DebatePosition:
    """A position taken in a debate."""

    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    role: DebateRole = DebateRole.PROPONENT
    stance: str = ""  # "for", "against", "neutral"

    # Position content
    statement: str = ""
    supporting_arguments: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0

    # Evolution tracking
    created_time: float = field(default_factory=time.time)
    revised_times: int = 0
    evolution_history: List[str] = field(default_factory=list)

    def add_argument(self, argument: str) -> None:
        """Add a supporting argument."""
        self.supporting_arguments.append(argument)

    def add_evidence(self, evidence: str) -> None:
        """Add supporting evidence."""
        self.evidence.append(evidence)

    def revise_position(self, new_statement: str, reason: str = "") -> None:
        """Revise the position statement."""
        self.evolution_history.append(f"[{time.time()}] {self.statement}")
        self.statement = new_statement
        self.revised_times += 1
        if reason:
            self.evolution_history.append(f"[{time.time()}] Revision reason: {reason}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "position_id": self.position_id,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "stance": self.stance,
            "statement": self.statement,
            "supporting_arguments": self.supporting_arguments,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "created_time": self.created_time,
            "revised_times": self.revised_times,
            "evolution_history": self.evolution_history,
        }


@dataclass
class DebateExchange:
    """A single exchange in a debate."""

    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: DebatePhase = DebatePhase.ARGUMENTATION
    speaker_id: str = ""
    target_id: Optional[str] = None  # Who this exchange is directed at

    content: str = ""
    exchange_type: str = "statement"  # statement, question, challenge, rebuttal
    timestamp: float = field(default_factory=time.time)

    # References
    references_position: Optional[str] = None
    references_exchange: Optional[str] = None

    # Metadata
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "exchange_id": self.exchange_id,
            "phase": self.phase.value,
            "speaker_id": self.speaker_id,
            "target_id": self.target_id,
            "content": self.content,
            "exchange_type": self.exchange_type,
            "timestamp": self.timestamp,
            "references_position": self.references_position,
            "references_exchange": self.references_exchange,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DebateResult:
    """Result of a complete debate session."""

    debate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    participants: Dict[str, DebateRole] = field(default_factory=dict)

    # Session tracking
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    current_phase: DebatePhase = DebatePhase.OPENING_STATEMENTS
    completed_phases: List[DebatePhase] = field(default_factory=list)

    # Content
    positions: Dict[str, DebatePosition] = field(default_factory=dict)
    exchanges: List[DebateExchange] = field(default_factory=list)

    # Resolution
    consensus_reached: bool = False
    winning_position: Optional[str] = None
    final_synthesis: str = ""

    # Metadata
    total_exchanges: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Duration of the debate."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def get_participant_exchanges(self, agent_id: str) -> List[DebateExchange]:
        """Get all exchanges for a specific participant."""
        return [ex for ex in self.exchanges if ex.speaker_id == agent_id]

    def get_phase_exchanges(self, phase: DebatePhase) -> List[DebateExchange]:
        """Get all exchanges for a specific phase."""
        return [ex for ex in self.exchanges if ex.phase == phase]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "participants": {
                agent_id: role.value for agent_id, role in self.participants.items()
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "current_phase": self.current_phase.value,
            "completed_phases": [p.value for p in self.completed_phases],
            "positions": {
                pos_id: pos.to_dict() for pos_id, pos in self.positions.items()
            },
            "exchanges": [ex.to_dict() for ex in self.exchanges],
            "consensus_reached": self.consensus_reached,
            "winning_position": self.winning_position,
            "final_synthesis": self.final_synthesis,
            "total_exchanges": self.total_exchanges,
            "metadata": self.metadata,
        }


class DebateManager:
    """
    Manager for structured debates between agents.

    Coordinates debate phases, manages participant roles, tracks positions,
    and facilitates structured argumentation.
    """

    def __init__(
        self,
        name: str = "debate_manager",
        max_exchanges_per_phase: int = 10,
        phase_timeout: float = 300.0,
        require_consensus: bool = False,
        enable_position_evolution: bool = True,
    ):
        """
        Initialize debate manager.

        Args:
            name: Name identifier for this manager
            max_exchanges_per_phase: Maximum exchanges allowed per phase
            phase_timeout: Timeout for each phase (seconds)
            require_consensus: Whether consensus is required for completion
            enable_position_evolution: Allow positions to evolve during debate
        """
        self.name = name
        self.max_exchanges_per_phase = max_exchanges_per_phase
        self.phase_timeout = phase_timeout
        self.require_consensus = require_consensus
        self.enable_position_evolution = enable_position_evolution

        # Active debates
        self.active_debates: Dict[str, DebateResult] = {}

        # Statistics
        self.debates_conducted = 0
        self.consensus_reached_count = 0
        self.created_time = time.time()

        logger.info(f"DebateManager '{name}' initialized")

    async def conduct_analyst_critic_debate(
        self,
        topic: str,
        analyst_agent: Any,
        critic_agent: Any,
        context: Optional[Dict[str, Any]] = None,
        rounds: int = 3,
    ) -> DebateResult:
        """
        Conduct a structured analyst vs critic debate.

        Args:
            topic: Topic for debate
            analyst_agent: Agent taking analyst role
            critic_agent: Agent taking critic role
            context: Additional context
            rounds: Number of debate rounds

        Returns:
            DebateResult with complete debate record
        """
        context = context or {}

        # Create debate session
        debate = DebateResult(
            topic=topic,
            participants={
                analyst_agent.name: DebateRole.ANALYST,
                critic_agent.name: DebateRole.CRITIC,
            },
            metadata=context,
        )

        self.active_debates[debate.debate_id] = debate

        try:
            # Phase 1: Opening Statements
            await self._conduct_opening_statements(
                debate, [analyst_agent, critic_agent]
            )

            # Phase 2: Structured Argumentation (multiple rounds)
            for round_num in range(rounds):
                await self._conduct_argumentation_round(
                    debate, analyst_agent, critic_agent, round_num + 1
                )

            # Phase 3: Cross Examination
            await self._conduct_cross_examination(debate, analyst_agent, critic_agent)

            # Phase 4: Closing Statements
            await self._conduct_closing_statements(
                debate, [analyst_agent, critic_agent]
            )

            # Phase 5: Resolution
            await self._conduct_resolution(debate)

            debate.end_time = time.time()
            self.debates_conducted += 1

            if debate.consensus_reached:
                self.consensus_reached_count += 1

            logger.info(f"Analyst-Critic debate completed: {debate.debate_id}")

        except Exception as e:
            logger.error(f"Error in analyst-critic debate: {e}")
            debate.metadata["error"] = str(e)

        finally:
            if debate.debate_id in self.active_debates:
                del self.active_debates[debate.debate_id]

        return debate

    async def _conduct_opening_statements(
        self, debate: DebateResult, agents: List[Any]
    ) -> None:
        """Conduct opening statements phase."""
        debate.current_phase = DebatePhase.OPENING_STATEMENTS

        for agent in agents:
            role = debate.participants[agent.name]

            # Generate opening statement prompt
            prompt = self._build_opening_statement_prompt(debate.topic, role)

            # Get agent response
            exchange = await self._get_agent_exchange(
                agent, prompt, debate.current_phase, debate.debate_id
            )

            debate.exchanges.append(exchange)

            # Extract position if this is a position-taking role
            if role in [DebateRole.ANALYST, DebateRole.PROPONENT]:
                position = await self._extract_position(
                    agent.name, role, exchange.content
                )
                position.stance = (
                    "for" if role == DebateRole.PROPONENT else "analytical"
                )
                debate.positions[position.position_id] = position

        debate.completed_phases.append(DebatePhase.OPENING_STATEMENTS)
        logger.debug(f"Completed opening statements for debate {debate.debate_id}")

    async def _conduct_argumentation_round(
        self, debate: DebateResult, analyst: Any, critic: Any, round_num: int
    ) -> None:
        """Conduct one round of argumentation."""
        debate.current_phase = DebatePhase.ARGUMENTATION

        # Analyst presents analysis
        analyst_prompt = self._build_argumentation_prompt(
            debate.topic, DebateRole.ANALYST, debate.exchanges, round_num
        )

        analyst_exchange = await self._get_agent_exchange(
            analyst, analyst_prompt, debate.current_phase, debate.debate_id
        )
        debate.exchanges.append(analyst_exchange)

        # Critic responds with critique
        critic_prompt = self._build_argumentation_prompt(
            debate.topic, DebateRole.CRITIC, debate.exchanges, round_num
        )

        critic_exchange = await self._get_agent_exchange(
            critic, critic_prompt, debate.current_phase, debate.debate_id
        )
        critic_exchange.references_exchange = analyst_exchange.exchange_id
        debate.exchanges.append(critic_exchange)

        # Update positions if position evolution is enabled
        if self.enable_position_evolution:
            await self._update_positions_from_exchange(debate, analyst_exchange)
            await self._update_positions_from_exchange(debate, critic_exchange)

        logger.debug(
            f"Completed argumentation round {round_num} for debate {debate.debate_id}"
        )

    async def _conduct_cross_examination(
        self, debate: DebateResult, analyst: Any, critic: Any
    ) -> None:
        """Conduct cross examination phase."""
        debate.current_phase = DebatePhase.CROSS_EXAMINATION

        # Critic questions analyst
        question_prompt = self._build_cross_examination_prompt(
            debate.topic,
            DebateRole.CRITIC,
            debate.exchanges,
            target_role=DebateRole.ANALYST,
        )

        question_exchange = await self._get_agent_exchange(
            critic, question_prompt, debate.current_phase, debate.debate_id
        )
        question_exchange.exchange_type = "question"
        question_exchange.target_id = analyst.name
        debate.exchanges.append(question_exchange)

        # Analyst responds
        response_prompt = self._build_response_prompt(
            debate.topic, DebateRole.ANALYST, question_exchange.content
        )

        response_exchange = await self._get_agent_exchange(
            analyst, response_prompt, debate.current_phase, debate.debate_id
        )
        response_exchange.exchange_type = "response"
        response_exchange.references_exchange = question_exchange.exchange_id
        debate.exchanges.append(response_exchange)

        debate.completed_phases.append(DebatePhase.CROSS_EXAMINATION)
        logger.debug(f"Completed cross examination for debate {debate.debate_id}")

    async def _conduct_closing_statements(
        self, debate: DebateResult, agents: List[Any]
    ) -> None:
        """Conduct closing statements phase."""
        debate.current_phase = DebatePhase.CLOSING_STATEMENTS

        for agent in agents:
            role = debate.participants[agent.name]

            # Generate closing statement prompt
            prompt = self._build_closing_statement_prompt(
                debate.topic, role, debate.exchanges
            )

            # Get agent response
            exchange = await self._get_agent_exchange(
                agent, prompt, debate.current_phase, debate.debate_id
            )
            exchange.exchange_type = "closing_statement"

            debate.exchanges.append(exchange)

        debate.completed_phases.append(DebatePhase.CLOSING_STATEMENTS)
        logger.debug(f"Completed closing statements for debate {debate.debate_id}")

    async def _conduct_resolution(self, debate: DebateResult) -> None:
        """Conduct resolution phase to determine outcome."""
        debate.current_phase = DebatePhase.RESOLUTION

        # Analyze the debate for consensus or winning position
        await self._analyze_debate_outcome(debate)

        # Generate final synthesis
        debate.final_synthesis = await self._generate_final_synthesis(debate)

        debate.completed_phases.append(DebatePhase.RESOLUTION)
        logger.debug(f"Completed resolution for debate {debate.debate_id}")

    async def _get_agent_exchange(
        self, agent: Any, prompt: str, phase: DebatePhase, debate_id: str
    ) -> DebateExchange:
        """Get an exchange from an agent."""
        try:
            if hasattr(agent, "async_generate_reasoning_tree"):
                response = await agent.async_generate_reasoning_tree(prompt)
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, agent.generate_reasoning_tree, prompt
                )

            result = response.get("result", {})
            content = result.get("content", "")
            confidence = float(result.get("confidence", 0.0))

            return DebateExchange(
                phase=phase,
                speaker_id=agent.name,
                content=content,
                confidence=confidence,
                metadata={"debate_id": debate_id, "raw_response": response},
            )

        except Exception as e:
            logger.error(f"Error getting exchange from agent {agent.name}: {e}")
            return DebateExchange(
                phase=phase,
                speaker_id=agent.name,
                content=f"[ERROR] Failed to generate response: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _build_opening_statement_prompt(self, topic: str, role: DebateRole) -> str:
        """Build opening statement prompt based on role."""
        if role == DebateRole.ANALYST:
            return (
                f"As an analyst, provide your opening analysis of: {topic}. "
                f"Present a thorough, objective analysis with key insights, "
                f"evidence, and implications. Be analytical and comprehensive."
            )

        elif role == DebateRole.CRITIC:
            return (
                f"As a critic, provide your opening critical perspective on: {topic}. "
                f"Identify potential issues, limitations, risks, and areas that need "
                f"scrutiny. Be thorough in your critical evaluation."
            )

        else:
            return f"Provide your opening statement on: {topic}."

    def _build_argumentation_prompt(
        self,
        topic: str,
        role: DebateRole,
        exchanges: List[DebateExchange],
        round_num: int,
    ) -> str:
        """Build argumentation prompt based on role and previous exchanges."""
        # Get recent context
        recent_exchanges = exchanges[-4:] if len(exchanges) > 4 else exchanges
        context = "\n".join(
            [f"{ex.speaker_id}: {ex.content[:200]}..." for ex in recent_exchanges]
        )

        if role == DebateRole.ANALYST:
            return (
                f"Round {round_num} - Continue your analysis of: {topic}.\n\n"
                f"Previous discussion:\n{context}\n\n"
                f"Provide deeper analysis, address any points raised, and present "
                f"additional evidence or insights. Build on the discussion."
            )

        elif role == DebateRole.CRITIC:
            return (
                f"Round {round_num} - Continue your critique of: {topic}.\n\n"
                f"Previous discussion:\n{context}\n\n"
                f"Address the analyst's points, identify weaknesses or gaps, "
                f"and provide critical evaluation. Challenge assumptions."
            )

        else:
            return f"Round {round_num} - Continue the discussion on: {topic}.\n\nContext:\n{context}"

    def _build_cross_examination_prompt(
        self,
        topic: str,
        role: DebateRole,
        exchanges: List[DebateExchange],
        target_role: DebateRole,
    ) -> str:
        """Build cross-examination prompt."""
        # Get target's recent statements
        target_exchanges = [
            ex for ex in exchanges if ex.speaker_id != exchanges[-1].speaker_id
        ][-2:]
        target_context = "\n".join(
            [f"{ex.content[:200]}..." for ex in target_exchanges]
        )

        return (
            f"Cross-examination phase for topic: {topic}.\n\n"
            f"The {target_role.value} has stated:\n{target_context}\n\n"
            f"As the {role.value}, ask probing questions to challenge their position, "
            f"identify weaknesses, or seek clarification on key points."
        )

    def _build_response_prompt(
        self, topic: str, role: DebateRole, question: str
    ) -> str:
        """Build response prompt for cross-examination."""
        return (
            f"You are being cross-examined about: {topic}.\n\n"
            f"Question: {question}\n\n"
            f"As the {role.value}, provide a clear, direct response that addresses "
            f"the question while defending your position."
        )

    def _build_closing_statement_prompt(
        self, topic: str, role: DebateRole, exchanges: List[DebateExchange]
    ) -> str:
        """Build closing statement prompt."""
        return (
            f"Provide your closing statement for the debate on: {topic}.\n\n"
            f"Summarize your key points, address counterarguments, and present "
            f"your final position. This is your last opportunity to make your case."
        )

    async def _extract_position(
        self, agent_id: str, role: DebateRole, content: str
    ) -> DebatePosition:
        """Extract a debate position from content."""
        position = DebatePosition(
            agent_id=agent_id,
            role=role,
            statement=content[:500] + "..." if len(content) > 500 else content,
        )

        # Simple extraction of arguments (look for bullet points, numbered lists)
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(("- ", "* ", "1. ", "2. ", "3. ")):
                position.add_argument(line[2:].strip())

        return position

    async def _update_positions_from_exchange(
        self, debate: DebateResult, exchange: DebateExchange
    ) -> None:
        """Update positions based on new exchange."""
        # Find relevant position
        agent_positions = [
            pos
            for pos in debate.positions.values()
            if pos.agent_id == exchange.speaker_id
        ]

        if agent_positions:
            position = agent_positions[0]  # Take the first/main position

            # Check if position has evolved significantly
            if len(exchange.content) > 100 and exchange.confidence > 0.7:
                # Simple heuristic: if content is substantially different, update position
                if exchange.content.lower() not in position.statement.lower():
                    position.revise_position(
                        exchange.content[:300] + "...",
                        "Updated based on ongoing debate",
                    )

    async def _analyze_debate_outcome(self, debate: DebateResult) -> None:
        """Analyze debate to determine outcome."""
        # Simple heuristic-based analysis
        analyst_exchanges = [
            ex
            for ex in debate.exchanges
            if ex.speaker_id
            in [k for k, v in debate.participants.items() if v == DebateRole.ANALYST]
        ]
        critic_exchanges = [
            ex
            for ex in debate.exchanges
            if ex.speaker_id
            in [k for k, v in debate.participants.items() if v == DebateRole.CRITIC]
        ]

        # Calculate average confidence scores
        analyst_avg_confidence = (
            sum(ex.confidence for ex in analyst_exchanges) / len(analyst_exchanges)
            if analyst_exchanges
            else 0.0
        )

        critic_avg_confidence = (
            sum(ex.confidence for ex in critic_exchanges) / len(critic_exchanges)
            if critic_exchanges
            else 0.0
        )

        # Determine outcome based on confidence and quality
        confidence_diff = abs(analyst_avg_confidence - critic_avg_confidence)

        if confidence_diff < 0.1:
            debate.consensus_reached = True
            debate.final_synthesis = (
                "Consensus reached through balanced analysis and critique"
            )
        elif analyst_avg_confidence > critic_avg_confidence:
            debate.winning_position = "analyst"
        else:
            debate.winning_position = "critic"

        # Store analysis metadata
        debate.metadata.update(
            {
                "analyst_avg_confidence": analyst_avg_confidence,
                "critic_avg_confidence": critic_avg_confidence,
                "confidence_diff": confidence_diff,
            }
        )

    async def _generate_final_synthesis(self, debate: DebateResult) -> str:
        """Generate final synthesis of the debate."""
        if debate.final_synthesis:
            return debate.final_synthesis

        # Simple synthesis based on key exchanges
        key_points = []

        # Get closing statements
        closing_exchanges = [
            ex for ex in debate.exchanges if ex.phase == DebatePhase.CLOSING_STATEMENTS
        ]

        for exchange in closing_exchanges:
            speaker_role = debate.participants.get(exchange.speaker_id, "participant")
            key_points.append(f"{speaker_role.title()}: {exchange.content[:200]}...")

        if key_points:
            return "Final synthesis: " + " | ".join(key_points)
        else:
            return (
                "Debate completed with structured argumentation and cross-examination."
            )

    def get_active_debates(self) -> Dict[str, DebateResult]:
        """Get all currently active debates."""
        return self.active_debates.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get debate manager statistics."""
        consensus_rate = (
            self.consensus_reached_count / self.debates_conducted
            if self.debates_conducted > 0
            else 0.0
        )

        return {
            "name": self.name,
            "debates_conducted": self.debates_conducted,
            "consensus_reached_count": self.consensus_reached_count,
            "consensus_rate": consensus_rate,
            "active_debates": len(self.active_debates),
            "config": {
                "max_exchanges_per_phase": self.max_exchanges_per_phase,
                "phase_timeout": self.phase_timeout,
                "require_consensus": self.require_consensus,
                "enable_position_evolution": self.enable_position_evolution,
            },
        }
