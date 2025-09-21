"""
Turn-based orchestration engine for single and multi-agent interactions.

Implements the MVP Phase 1 turn engine requirements:
- Single and multi-agent synchronous turn-taking
- Round-robin and policy-based termination conditions
- Integration with policy engine for action evaluation
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger("agentnet.orchestration.turn_engine")


class TerminationReason(str, Enum):
    """Reasons why a turn-based session can terminate."""
    
    COMPLETED = "completed"
    MAX_TURNS_REACHED = "max_turns_reached"
    CONSENSUS_REACHED = "consensus_reached"
    POLICY_VIOLATION = "policy_violation"
    TIMEOUT = "timeout"
    ERROR = "error"
    USER_REQUESTED = "user_requested"


class TurnMode(str, Enum):
    """Different modes for turn-based interactions."""
    
    SINGLE_AGENT = "single_agent"
    ROUND_ROBIN = "round_robin"
    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    CONSENSUS = "consensus"


@dataclass
class TurnResult:
    """Result of a single turn in the conversation."""
    
    turn_id: str
    agent_id: str
    round_number: int
    turn_number: int
    content: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    policy_violations: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class SessionResult:
    """Result of a complete turn-based session."""
    
    session_id: str
    mode: TurnMode
    status: TerminationReason
    turns: List[TurnResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_turns: int = 0
    total_rounds: int = 0
    agents_involved: List[str] = field(default_factory=list)
    final_consensus: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Duration of the session in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def get_agent_turns(self, agent_id: str) -> List[TurnResult]:
        """Get all turns for a specific agent."""
        return [turn for turn in self.turns if turn.agent_id == agent_id]
    
    def get_round_turns(self, round_number: int) -> List[TurnResult]:
        """Get all turns for a specific round."""
        return [turn for turn in self.turns if turn.round_number == round_number]


class TurnEngine:
    """
    Core turn-based orchestration engine.
    
    Manages synchronous turn-taking between agents with configurable
    termination conditions and policy enforcement.
    """
    
    def __init__(
        self,
        max_turns: int = 50,
        max_rounds: int = 10,
        turn_timeout: float = 30.0,
        session_timeout: float = 300.0,
        consensus_threshold: float = 0.8,
        policy_engine: Optional[Any] = None,  # PolicyEngine from legacy code
        event_callbacks: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the turn engine.
        
        Args:
            max_turns: Maximum total turns before termination
            max_rounds: Maximum rounds before termination
            turn_timeout: Timeout for individual turns (seconds)
            session_timeout: Timeout for entire session (seconds)
            consensus_threshold: Confidence threshold for consensus detection
            policy_engine: Optional policy engine for turn validation
            event_callbacks: Optional callbacks for turn events
        """
        self.max_turns = max_turns
        self.max_rounds = max_rounds
        self.turn_timeout = turn_timeout
        self.session_timeout = session_timeout
        self.consensus_threshold = consensus_threshold
        self.policy_engine = policy_engine
        self.event_callbacks = event_callbacks or {}
        
        # Active sessions tracking
        self.active_sessions: Dict[str, SessionResult] = {}
    
    async def execute_single_agent_session(
        self,
        agent: Any,  # AgentNet instance
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
        termination_conditions: Optional[List[str]] = None,
    ) -> SessionResult:
        """
        Execute a single-agent session with multiple turns.
        
        Args:
            agent: The agent to interact with
            initial_prompt: Initial prompt to start the conversation
            context: Optional context for the session
            termination_conditions: Custom termination conditions
            
        Returns:
            SessionResult with the complete interaction
        """
        session_id = f"single_{uuid.uuid4().hex[:8]}"
        session = SessionResult(
            session_id=session_id,
            mode=TurnMode.SINGLE_AGENT,
            status=TerminationReason.COMPLETED,
            agents_involved=[agent.name],
            metadata=context or {}
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Starting single-agent session {session_id} with {agent.name}")
        
        try:
            current_prompt = initial_prompt
            round_number = 1
            turn_number = 1
            
            # Session timeout
            session_start = time.time()
            
            while (turn_number <= self.max_turns and 
                   round_number <= self.max_rounds):
                
                # Check session timeout
                if time.time() - session_start > self.session_timeout:
                    session.status = TerminationReason.TIMEOUT
                    break
                
                # Execute turn
                turn_result = await self._execute_agent_turn(
                    agent=agent,
                    prompt=current_prompt,
                    session_id=session_id,
                    round_number=round_number,
                    turn_number=turn_number,
                    context=context
                )
                
                session.turns.append(turn_result)
                
                # Check for policy violations
                if turn_result.policy_violations:
                    session.status = TerminationReason.POLICY_VIOLATION
                    break
                
                # Check termination conditions
                if await self._should_terminate_single_agent(
                    session, turn_result, termination_conditions
                ):
                    break
                
                # Prepare next turn prompt based on previous response
                current_prompt = self._generate_followup_prompt(turn_result)
                turn_number += 1
                
                # Simple round progression for single agent
                if turn_number % 3 == 1:  # Every 3 turns = new round
                    round_number += 1
            
            # Check why we exited
            if turn_number > self.max_turns:
                session.status = TerminationReason.MAX_TURNS_REACHED
            elif round_number > self.max_rounds:
                session.status = TerminationReason.MAX_TURNS_REACHED
            
            session.total_turns = len(session.turns)
            session.total_rounds = round_number
            
        except Exception as e:
            logger.error(f"Error in single-agent session {session_id}: {e}")
            session.status = TerminationReason.ERROR
            session.metadata["error"] = str(e)
        
        finally:
            session.end_time = time.time()
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        logger.info(f"Single-agent session {session_id} completed: {session.status}")
        return session
    
    async def execute_multi_agent_session(
        self,
        agents: List[Any],  # List of AgentNet instances
        topic: str,
        mode: TurnMode = TurnMode.ROUND_ROBIN,
        context: Optional[Dict[str, Any]] = None,
        termination_conditions: Optional[List[str]] = None,
    ) -> SessionResult:
        """
        Execute a multi-agent session with configurable interaction mode.
        
        Args:
            agents: List of agents to participate
            topic: Topic for discussion
            mode: Interaction mode (round_robin, debate, brainstorm, consensus)
            context: Optional context for the session
            termination_conditions: Custom termination conditions
            
        Returns:
            SessionResult with the complete interaction
        """
        if not agents or len(agents) < 2:
            raise ValueError("Multi-agent session requires at least 2 agents")
        
        session_id = f"multi_{uuid.uuid4().hex[:8]}"
        session = SessionResult(
            session_id=session_id,
            mode=mode,
            status=TerminationReason.COMPLETED,
            agents_involved=[agent.name for agent in agents],
            metadata=context or {}
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Starting multi-agent session {session_id} with {len(agents)} agents, mode={mode}")
        
        try:
            round_number = 1
            turn_number = 1
            session_start = time.time()
            
            while (round_number <= self.max_rounds and 
                   turn_number <= self.max_turns):
                
                # Check session timeout
                if time.time() - session_start > self.session_timeout:
                    session.status = TerminationReason.TIMEOUT
                    break
                
                # Execute round based on mode
                round_turns = await self._execute_round(
                    agents=agents,
                    topic=topic,
                    mode=mode,
                    session=session,
                    round_number=round_number,
                    turn_number=turn_number,
                    context=context
                )
                
                session.turns.extend(round_turns)
                turn_number += len(round_turns)
                
                # Check for policy violations in any turn
                if any(turn.policy_violations for turn in round_turns):
                    session.status = TerminationReason.POLICY_VIOLATION
                    break
                
                # Check termination conditions
                if await self._should_terminate_multi_agent(
                    session, round_turns, termination_conditions
                ):
                    break
                
                round_number += 1
            
            # Check why we exited
            if round_number > self.max_rounds:
                session.status = TerminationReason.MAX_TURNS_REACHED
            elif turn_number > self.max_turns:
                session.status = TerminationReason.MAX_TURNS_REACHED
            
            session.total_turns = len(session.turns)
            session.total_rounds = round_number - 1
            
            # Try to detect consensus for appropriate modes
            if mode in [TurnMode.CONSENSUS, TurnMode.DEBATE]:
                consensus = await self._detect_consensus(session)
                if consensus:
                    session.final_consensus = consensus
                    if session.status == TerminationReason.COMPLETED:
                        session.status = TerminationReason.CONSENSUS_REACHED
        
        except Exception as e:
            logger.error(f"Error in multi-agent session {session_id}: {e}")
            session.status = TerminationReason.ERROR
            session.metadata["error"] = str(e)
        
        finally:
            session.end_time = time.time()
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        logger.info(f"Multi-agent session {session_id} completed: {session.status}")
        return session
    
    async def _execute_agent_turn(
        self,
        agent: Any,
        prompt: str,
        session_id: str,
        round_number: int,
        turn_number: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Execute a single agent turn."""
        turn_id = f"{session_id}_t{turn_number}"
        
        try:
            # Call event callback if available
            if "on_turn_start" in self.event_callbacks:
                await self._maybe_call_async(
                    self.event_callbacks["on_turn_start"],
                    session_id, agent.name, turn_number, round_number
                )
            
            # Execute the turn with timeout
            start_time = time.time()
            
            if hasattr(agent, 'async_generate_reasoning_tree'):
                raw_response = await asyncio.wait_for(
                    agent.async_generate_reasoning_tree(prompt),
                    timeout=self.turn_timeout
                )
            else:
                # Fallback to sync method
                raw_response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, agent.generate_reasoning_tree, prompt
                    ),
                    timeout=self.turn_timeout
                )
            
            # Extract content and confidence
            result = raw_response.get("result", {})
            content = result.get("content", "")
            confidence = float(result.get("confidence", 0.0))
            
            # Create turn result
            turn_result = TurnResult(
                turn_id=turn_id,
                agent_id=agent.name,
                round_number=round_number,
                turn_number=turn_number,
                content=content,
                confidence=confidence,
                timestamp=start_time,
                raw_response=raw_response
            )
            
            # Apply policy engine if available
            if self.policy_engine:
                violations = self.policy_engine.evaluate(raw_response)
                if isinstance(violations, list) and violations:
                    # Convert to rich format if needed
                    if hasattr(violations[0], 'to_dict'):
                        turn_result.policy_violations = [v.to_dict() for v in violations]
                    else:
                        turn_result.policy_violations = [{"name": str(v), "severity": "unknown"} for v in violations]
                    logger.warning(f"Policy violations in turn {turn_id}: {[str(v) for v in violations]}")
            
            # Call event callback if available
            if "on_turn_end" in self.event_callbacks:
                await self._maybe_call_async(
                    self.event_callbacks["on_turn_end"],
                    session_id, agent.name, turn_result
                )
            
            return turn_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Turn {turn_id} timed out after {self.turn_timeout}s")
            return TurnResult(
                turn_id=turn_id,
                agent_id=agent.name,
                round_number=round_number,
                turn_number=turn_number,
                content=f"[TIMEOUT] Agent {agent.name} timed out",
                confidence=0.0,
                metadata={"timeout": True}
            )
        
        except Exception as e:
            logger.error(f"Error in turn {turn_id}: {e}")
            return TurnResult(
                turn_id=turn_id,
                agent_id=agent.name,
                round_number=round_number,
                turn_number=turn_number,
                content=f"[ERROR] {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def _execute_round(
        self,
        agents: List[Any],
        topic: str,
        mode: TurnMode,
        session: SessionResult,
        round_number: int,
        turn_number: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TurnResult]:
        """Execute a complete round based on the interaction mode."""
        round_turns = []
        
        if mode == TurnMode.ROUND_ROBIN:
            # Simple round-robin: each agent takes one turn
            for i, agent in enumerate(agents):
                prompt = self._generate_round_robin_prompt(
                    topic, agent, session.turns, round_number
                )
                
                turn_result = await self._execute_agent_turn(
                    agent=agent,
                    prompt=prompt,
                    session_id=session.session_id,
                    round_number=round_number,
                    turn_number=turn_number + i,
                    context=context
                )
                round_turns.append(turn_result)
        
        elif mode == TurnMode.DEBATE:
            # Debate mode: agents take positions and counter-argue
            round_turns = await self._execute_debate_round(
                agents, topic, session, round_number, turn_number, context
            )
        
        elif mode == TurnMode.BRAINSTORM:
            # Brainstorm mode: agents generate diverse ideas
            round_turns = await self._execute_brainstorm_round(
                agents, topic, session, round_number, turn_number, context
            )
        
        elif mode == TurnMode.CONSENSUS:
            # Consensus mode: agents work toward agreement
            round_turns = await self._execute_consensus_round(
                agents, topic, session, round_number, turn_number, context
            )
        
        return round_turns
    
    async def _execute_debate_round(
        self, agents: List[Any], topic: str, session: SessionResult,
        round_number: int, turn_number: int, context: Optional[Dict[str, Any]]
    ) -> List[TurnResult]:
        """Execute a debate round where agents argue positions."""
        round_turns = []
        
        for i, agent in enumerate(agents):
            # Generate debate prompt based on previous turns
            if round_number == 1:
                if i == 0:
                    prompt = f"Take a strong position on: {topic}. Present your arguments clearly and defend your stance."
                else:
                    # Get previous agent's position
                    prev_turn = session.turns[-1] if session.turns else None
                    prev_content = prev_turn.content if prev_turn else "the previous position"
                    prompt = f"Counter-argue the position: '{prev_content}' regarding {topic}. Present alternative viewpoints and critique the previous arguments."
            else:
                # Later rounds: respond to recent arguments
                recent_turns = session.turns[-len(agents):] if session.turns else []
                recent_arguments = "; ".join([t.content[:100] + "..." for t in recent_turns])
                prompt = f"Continue the debate on {topic}. Recent arguments: {recent_arguments}. Strengthen your position or address counterarguments."
            
            turn_result = await self._execute_agent_turn(
                agent=agent,
                prompt=prompt,
                session_id=session.session_id,
                round_number=round_number,
                turn_number=turn_number + i,
                context=context
            )
            round_turns.append(turn_result)
        
        return round_turns
    
    async def _execute_brainstorm_round(
        self, agents: List[Any], topic: str, session: SessionResult,
        round_number: int, turn_number: int, context: Optional[Dict[str, Any]]
    ) -> List[TurnResult]:
        """Execute a brainstorm round where agents generate diverse ideas."""
        round_turns = []
        
        for i, agent in enumerate(agents):
            # Generate brainstorm prompt
            if round_number == 1:
                prompt = f"Brainstorm creative and diverse ideas about: {topic}. Think outside the box and don't judge ideas prematurely."
            else:
                # Build on previous ideas
                prev_ideas = [turn.content[:50] + "..." for turn in session.turns[-3:]]
                prompt = f"Continue brainstorming on {topic}. Previous ideas: {'; '.join(prev_ideas)}. Generate new, different approaches."
            
            turn_result = await self._execute_agent_turn(
                agent=agent,
                prompt=prompt,
                session_id=session.session_id,
                round_number=round_number,
                turn_number=turn_number + i,
                context=context
            )
            round_turns.append(turn_result)
        
        return round_turns
    
    async def _execute_consensus_round(
        self, agents: List[Any], topic: str, session: SessionResult,
        round_number: int, turn_number: int, context: Optional[Dict[str, Any]]
    ) -> List[TurnResult]:
        """Execute a consensus round where agents work toward agreement."""
        round_turns = []
        
        for i, agent in enumerate(agents):
            # Generate consensus-building prompt
            if round_number == 1:
                prompt = f"Propose a solution or approach to: {topic}. Focus on finding common ground and practical solutions."
            else:
                # Look for convergence
                recent_turns = session.turns[-len(agents):] if session.turns else []
                convergence_points = self._extract_convergence_points(recent_turns)
                prompt = f"Work toward consensus on {topic}. Common points so far: {convergence_points}. Build on agreements and resolve differences."
            
            turn_result = await self._execute_agent_turn(
                agent=agent,
                prompt=prompt,
                session_id=session.session_id,
                round_number=round_number,
                turn_number=turn_number + i,
                context=context
            )
            round_turns.append(turn_result)
        
        return round_turns
    
    def _generate_round_robin_prompt(
        self, topic: str, agent: Any, previous_turns: List[TurnResult], round_number: int
    ) -> str:
        """Generate a prompt for round-robin mode."""
        if round_number == 1:
            return f"Discuss and analyze: {topic}. Provide your perspective and insights."
        else:
            # Include context from previous turns
            recent_context = ""
            if previous_turns:
                last_few = previous_turns[-3:]
                recent_context = f" Recent discussion: {'; '.join([t.content[:100] + '...' for t in last_few])}"
            
            return f"Continue the discussion on: {topic}.{recent_context} Build on previous points or introduce new aspects."
    
    def _generate_followup_prompt(self, turn_result: TurnResult) -> str:
        """Generate a followup prompt based on the previous turn."""
        content_snippet = turn_result.content[:200] + "..." if len(turn_result.content) > 200 else turn_result.content
        return f"Continue your reasoning. Your previous response: '{content_snippet}'. Elaborate, provide examples, or explore implications."
    
    def _extract_convergence_points(self, turns: List[TurnResult]) -> str:
        """Extract common themes or agreement points from recent turns."""
        if not turns:
            return "None identified yet"
        
        # Simple heuristic: look for repeated keywords/phrases
        all_content = " ".join([turn.content.lower() for turn in turns])
        common_words = ["agree", "consensus", "common", "shared", "together", "both", "all"]
        
        found_convergence = []
        for word in common_words:
            if word in all_content:
                found_convergence.append(f"shared perspective on {word}")
        
        return "; ".join(found_convergence) if found_convergence else "Working toward alignment"
    
    async def _should_terminate_single_agent(
        self, session: SessionResult, turn_result: TurnResult, 
        termination_conditions: Optional[List[str]]
    ) -> bool:
        """Check if single-agent session should terminate."""
        # Check confidence threshold
        if turn_result.confidence >= self.consensus_threshold:
            return True
        
        # Check for completion keywords
        completion_keywords = ["complete", "finished", "done", "concluded", "final"]
        content_lower = turn_result.content.lower()
        
        if any(keyword in content_lower for keyword in completion_keywords):
            return True
        
        # Check custom termination conditions
        if termination_conditions:
            for condition in termination_conditions:
                if condition.lower() in content_lower:
                    return True
        
        return False
    
    async def _should_terminate_multi_agent(
        self, session: SessionResult, round_turns: List[TurnResult],
        termination_conditions: Optional[List[str]]
    ) -> bool:
        """Check if multi-agent session should terminate."""
        # Check if all agents have high confidence in recent turns
        recent_confidences = [turn.confidence for turn in round_turns]
        if all(conf >= self.consensus_threshold for conf in recent_confidences):
            return True
        
        # Check for convergence in content
        if len(session.turns) >= 4:  # Need some history
            if await self._detect_convergence(session):
                return True
        
        # Check custom termination conditions
        if termination_conditions:
            all_content = " ".join([turn.content.lower() for turn in round_turns])
            for condition in termination_conditions:
                if condition.lower() in all_content:
                    return True
        
        return False
    
    async def _detect_consensus(self, session: SessionResult) -> Optional[str]:
        """Detect if consensus has been reached and return the consensus statement."""
        if len(session.turns) < 4:  # Need minimum turns
            return None
        
        # Simple consensus detection: look for agreement keywords in recent turns
        recent_turns = session.turns[-4:]
        agreement_keywords = ["agree", "consensus", "aligned", "shared view", "common ground"]
        
        consensus_indicators = []
        for turn in recent_turns:
            content_lower = turn.content.lower()
            for keyword in agreement_keywords:
                if keyword in content_lower:
                    # Extract the sentence with the consensus
                    sentences = turn.content.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            consensus_indicators.append(sentence.strip())
                            break
        
        if len(consensus_indicators) >= 2:  # Multiple agents showing agreement
            return " | ".join(consensus_indicators[:2])
        
        return None
    
    async def _detect_convergence(self, session: SessionResult) -> bool:
        """Detect if agents are converging in their responses."""
        if len(session.turns) < 6:  # Need sufficient history
            return False
        
        # Compare recent turns for similarity (simple heuristic)
        recent_turns = session.turns[-4:]
        contents = [turn.content.lower() for turn in recent_turns]
        
        # Look for repeated themes or phrases
        word_counts = {}
        for content in contents:
            words = content.split()
            for word in words:
                if len(word) > 4:  # Only meaningful words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # If many words appear in multiple recent turns, suggest convergence
        repeated_words = sum(1 for count in word_counts.values() if count >= 2)
        convergence_ratio = repeated_words / len(word_counts) if word_counts else 0
        
        return convergence_ratio > 0.3  # 30% of words are repeated across turns
    
    async def _maybe_call_async(self, callback: Callable, *args, **kwargs):
        """Call a callback function, handling both sync and async versions."""
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def get_active_sessions(self) -> Dict[str, SessionResult]:
        """Get all currently active sessions."""
        return self.active_sessions.copy()
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate an active session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = TerminationReason.USER_REQUESTED
            session.end_time = time.time()
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} terminated by user request")
            return True
        return False