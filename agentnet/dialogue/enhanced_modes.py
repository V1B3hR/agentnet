"""
Enhanced dialogue modes implementation based on research from managebetter.com and PMC articles.

Implements five advanced dialogue types:
- Outer Dialogue: Standard agent-to-agent communication (enhanced)
- Modulated Conversation: Tension-building dialogue with dynamic intensity
- Interpolation Conversation: Context-filling dialogue with knowledge gaps
- Inner (Internal) Dialogue: Agent self-reflection capabilities
- Dialogue Mapping: Collaborative decision-making visualization
"""

from __future__ import annotations
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.agent import AgentNet

logger = logging.getLogger("agentnet.dialogue")


class DialogueMode(str, Enum):
    """Enumeration of available dialogue modes."""
    OUTER = "outer" 
    MODULATED = "modulated"
    INTERPOLATION = "interpolation"
    INNER = "inner"
    MAPPING = "mapping"


@dataclass
class DialogueTurn:
    """Represents a single turn in dialogue."""
    agent_name: str
    content: str
    confidence: float
    mode: DialogueMode
    round_number: int
    timestamp: float
    reasoning_type: Optional[str] = None
    internal_state: Optional[Dict[str, Any]] = None
    intensity_level: Optional[float] = None
    context_additions: Optional[List[str]] = None


@dataclass  
class DialogueState:
    """Tracks the state of an ongoing dialogue."""
    session_id: str
    mode: DialogueMode
    participants: List[str]
    topic: str
    current_round: int
    transcript: List[DialogueTurn]
    intensity_level: float = 0.5
    context_gaps: List[str] = None
    decision_map: Dict[str, Any] = None
    convergence_indicators: List[str] = None


class BaseDialogueMode(ABC):
    """Abstract base class for dialogue mode implementations."""
    
    def __init__(self, mode: DialogueMode):
        """Initialize with dialogue mode type."""
        self.mode = mode
        self.state_history: List[DialogueState] = []
    
    @abstractmethod
    def conduct_dialogue(self,
                        agents: List['AgentNet'],
                        topic: str,
                        rounds: int = 5,
                        **kwargs) -> Dict[str, Any]:
        """Conduct dialogue in this mode."""
        pass
    
    @abstractmethod
    async def conduct_async_dialogue(self,
                                   agents: List['AgentNet'],
                                   topic: str,
                                   rounds: int = 5,
                                   **kwargs) -> Dict[str, Any]:
        """Conduct asynchronous dialogue in this mode."""
        pass
    
    def _create_dialogue_state(self,
                             agents: List['AgentNet'],
                             topic: str,
                             **kwargs) -> DialogueState:
        """Create initial dialogue state."""
        session_id = f"{self.mode.value}_{int(time.time()*1000)}"
        return DialogueState(
            session_id=session_id,
            mode=self.mode,
            participants=[agent.name for agent in agents],
            topic=topic,
            current_round=0,
            transcript=[],
            context_gaps=kwargs.get("context_gaps", []),
            decision_map=kwargs.get("decision_map", {}),
            intensity_level=kwargs.get("initial_intensity", 0.5)
        )
    
    def _turn_to_dict(self, turn: DialogueTurn) -> Dict[str, Any]:
        """Convert DialogueTurn to dictionary for JSON serialization."""
        return {
            "agent": turn.agent_name,
            "content": turn.content,
            "confidence": turn.confidence,
            "round": turn.round_number,
            "timestamp": turn.timestamp,
            "reasoning_type": turn.reasoning_type,
            "intensity": turn.intensity_level,
            "context_additions": turn.context_additions,
            "internal_state": turn.internal_state
        }


class OuterDialogue(BaseDialogueMode):
    """
    Enhanced outer dialogue: Standard agent-to-agent communication.
    Enhanced with structured conversation patterns and style-aware modulation.
    """
    
    def __init__(self):
        """Initialize outer dialogue mode."""
        super().__init__(DialogueMode.OUTER)
    
    def conduct_dialogue(self,
                        agents: List['AgentNet'],
                        topic: str,
                        rounds: int = 5,
                        conversation_pattern: str = "collaborative",
                        **kwargs) -> Dict[str, Any]:
        """
        Conduct structured outer dialogue.
        
        Args:
            agents: List of participating agents
            topic: Dialogue topic
            rounds: Number of dialogue rounds
            conversation_pattern: Pattern type (collaborative, adversarial, exploratory)
            
        Returns:
            Dialogue results with transcript and analysis
        """
        dialogue_state = self._create_dialogue_state(agents, topic, **kwargs)
        pattern_behavior = self._get_pattern_behavior(conversation_pattern)
        
        logger.info(f"Starting outer dialogue: {dialogue_state.session_id}")
        
        for round_num in range(rounds):
            dialogue_state.current_round = round_num + 1
            
            for agent in agents:
                # Apply conversation pattern to prompt
                enhanced_prompt = self._apply_conversation_pattern(
                    topic, pattern_behavior, dialogue_state, agent
                )
                
                # Generate response using agent's reasoning capabilities
                if hasattr(agent, 'generate_reasoning_tree_enhanced'):
                    response = agent.generate_reasoning_tree_enhanced(enhanced_prompt)
                else:
                    response = agent.generate_reasoning_tree(enhanced_prompt)
                
                # Create dialogue turn
                turn = DialogueTurn(
                    agent_name=agent.name,
                    content=response.get("result", {}).get("content", ""),
                    confidence=response.get("result", {}).get("confidence", 0.5),
                    mode=DialogueMode.OUTER,
                    round_number=round_num + 1,
                    timestamp=time.time(),
                    reasoning_type=response.get("reasoning_metadata", {}).get("reasoning_type")
                )
                
                dialogue_state.transcript.append(turn)
                
                # Update topic based on conversation flow
                topic = self._evolve_topic_with_pattern(topic, turn, pattern_behavior)
        
        # Generate final analysis
        analysis = self._analyze_outer_dialogue(dialogue_state)
        
        return {
            "session_id": dialogue_state.session_id,
            "mode": "outer",
            "conversation_pattern": conversation_pattern,
            "participants": dialogue_state.participants,
            "original_topic": dialogue_state.topic,
            "final_topic": topic,
            "transcript": [self._turn_to_dict(turn) for turn in dialogue_state.transcript],
            "analysis": analysis,
            "rounds_completed": rounds
        }
    
    async def conduct_async_dialogue(self,
                                   agents: List['AgentNet'],
                                   topic: str,
                                   rounds: int = 5,
                                   conversation_pattern: str = "collaborative",
                                   **kwargs) -> Dict[str, Any]:
        """Async version of outer dialogue."""
        # For now, run sync version - can be enhanced for true async
        return self.conduct_dialogue(agents, topic, rounds, conversation_pattern, **kwargs)
    
    def _get_pattern_behavior(self, pattern: str) -> Dict[str, Any]:
        """Get behavior configuration for conversation pattern."""
        patterns = {
            "collaborative": {
                "directive": "Build on others' ideas and seek common ground",
                "tone": "supportive",
                "focus": "synthesis"
            },
            "adversarial": {
                "directive": "Challenge ideas and present counterarguments",
                "tone": "critical", 
                "focus": "debate"
            },
            "exploratory": {
                "directive": "Ask questions and explore different perspectives",
                "tone": "curious",
                "focus": "discovery"
            }
        }
        return patterns.get(pattern, patterns["collaborative"])
    
    def _apply_conversation_pattern(self,
                                  topic: str,
                                  pattern_behavior: Dict[str, Any],
                                  state: DialogueState,
                                  agent: 'AgentNet') -> str:
        """Apply conversation pattern to create enhanced prompt."""
        directive = pattern_behavior["directive"]
        tone = pattern_behavior["tone"]
        
        # Include recent context from transcript
        context_summary = ""
        if state.transcript:
            recent_turns = state.transcript[-2:]  # Last 2 turns
            context_summary = f"\n\nRecent discussion:\n"
            for turn in recent_turns:
                context_summary += f"- {turn.agent_name}: {turn.content[:100]}...\n"
        
        enhanced_prompt = f"""
{directive}

Topic: {topic}
Conversation tone: {tone}
Round: {state.current_round}

{context_summary}

Please respond with {tone} engagement to advance the {pattern_behavior["focus"]}.
"""
        return enhanced_prompt.strip()
    
    def _evolve_topic_with_pattern(self,
                                 current_topic: str,
                                 turn: DialogueTurn,
                                 pattern_behavior: Dict[str, Any]) -> str:
        """Evolve topic based on conversation pattern and latest turn."""
        focus = pattern_behavior["focus"]
        
        if focus == "synthesis" and turn.confidence > 0.7:
            return f"Synthesized view of: {current_topic}"
        elif focus == "debate" and turn.confidence > 0.8:
            return f"Debated aspects of: {current_topic}"
        elif focus == "discovery" and turn.confidence > 0.6:
            return f"Explored dimensions of: {current_topic}"
        else:
            return current_topic
    
    def _analyze_outer_dialogue(self, state: DialogueState) -> Dict[str, Any]:
        """Analyze outer dialogue for patterns and insights."""
        if not state.transcript:
            return {"error": "No dialogue content to analyze"}
        
        # Calculate engagement metrics
        avg_confidence = sum(turn.confidence for turn in state.transcript) / len(state.transcript)
        
        # Identify participation patterns
        participation = {}
        for turn in state.transcript:
            participation[turn.agent_name] = participation.get(turn.agent_name, 0) + 1
        
        # Analyze reasoning diversity
        reasoning_types = [turn.reasoning_type for turn in state.transcript if turn.reasoning_type]
        reasoning_diversity = len(set(reasoning_types)) if reasoning_types else 0
        
        return {
            "average_confidence": avg_confidence,
            "participation_balance": participation,
            "reasoning_diversity": reasoning_diversity,
            "total_turns": len(state.transcript),
            "dialogue_quality": self._assess_dialogue_quality(state.transcript)
        }
    
    def _assess_dialogue_quality(self, transcript: List[DialogueTurn]) -> str:
        """Assess overall dialogue quality."""
        if not transcript:
            return "insufficient_data"
        
        avg_confidence = sum(turn.confidence for turn in transcript) / len(transcript)
        
        if avg_confidence > 0.8:
            return "high_quality"
        elif avg_confidence > 0.6:
            return "moderate_quality"
        else:
            return "needs_improvement"
    



class ModulatedConversation(BaseDialogueMode):
    """
    Modulated conversation: Tension-building dialogue with dynamic intensity adjustment.
    Provides important details through tension modulation and context building.
    """
    
    def __init__(self):
        """Initialize modulated conversation mode."""
        super().__init__(DialogueMode.MODULATED)
    
    def conduct_dialogue(self,
                        agents: List['AgentNet'],
                        topic: str,
                        rounds: int = 5,
                        initial_intensity: float = 0.3,
                        max_intensity: float = 0.9,
                        tension_strategy: str = "gradual_build",
                        **kwargs) -> Dict[str, Any]:
        """
        Conduct modulated conversation with dynamic tension adjustment.
        
        Args:
            agents: List of participating agents
            topic: Dialogue topic
            rounds: Number of dialogue rounds
            initial_intensity: Starting intensity level (0.0-1.0)
            max_intensity: Maximum intensity level
            tension_strategy: How to build tension (gradual_build, spike, oscillate)
            
        Returns:
            Dialogue results with intensity tracking
        """
        dialogue_state = self._create_dialogue_state(
            agents, topic, initial_intensity=initial_intensity, **kwargs
        )
        
        logger.info(f"Starting modulated conversation: {dialogue_state.session_id}")
        
        for round_num in range(rounds):
            dialogue_state.current_round = round_num + 1
            
            # Calculate current intensity based on strategy
            current_intensity = self._calculate_intensity(
                round_num, rounds, initial_intensity, max_intensity, tension_strategy
            )
            dialogue_state.intensity_level = current_intensity
            
            for agent in agents:
                # Create intensity-modulated prompt
                modulated_prompt = self._create_intensity_prompt(
                    topic, current_intensity, dialogue_state, agent
                )
                
                # Generate response with intensity consideration
                if hasattr(agent, 'generate_reasoning_tree_enhanced'):
                    # Use enhanced version if available
                    style_override = self._get_intensity_style_override(current_intensity)
                    response = agent.generate_reasoning_tree_enhanced(
                        modulated_prompt, style_override=style_override
                    )
                else:
                    response = agent.generate_reasoning_tree(modulated_prompt)
                
                # Create dialogue turn with intensity tracking
                turn = DialogueTurn(
                    agent_name=agent.name,
                    content=response.get("result", {}).get("content", ""),
                    confidence=response.get("result", {}).get("confidence", 0.5),
                    mode=DialogueMode.MODULATED,
                    round_number=round_num + 1,
                    timestamp=time.time(),
                    intensity_level=current_intensity,
                    reasoning_type=response.get("reasoning_metadata", {}).get("reasoning_type")
                )
                
                dialogue_state.transcript.append(turn)
        
        # Analyze modulated conversation
        analysis = self._analyze_modulated_conversation(dialogue_state)
        
        return {
            "session_id": dialogue_state.session_id,
            "mode": "modulated",
            "tension_strategy": tension_strategy,
            "participants": dialogue_state.participants,
            "topic": dialogue_state.topic,
            "transcript": [self._turn_to_dict(turn) for turn in dialogue_state.transcript],
            "intensity_progression": self._extract_intensity_progression(dialogue_state),
            "analysis": analysis,
            "rounds_completed": rounds
        }
    
    async def conduct_async_dialogue(self,
                                   agents: List['AgentNet'],
                                   topic: str,
                                   rounds: int = 5,
                                   initial_intensity: float = 0.3,
                                   max_intensity: float = 0.9,
                                   tension_strategy: str = "gradual_build",
                                   **kwargs) -> Dict[str, Any]:
        """Async version of modulated conversation."""
        return self.conduct_dialogue(
            agents, topic, rounds, initial_intensity, max_intensity, tension_strategy, **kwargs
        )
    
    def _calculate_intensity(self,
                           round_num: int,
                           total_rounds: int,
                           initial: float,
                           maximum: float,
                           strategy: str) -> float:
        """Calculate current intensity based on strategy."""
        progress = round_num / max(1, total_rounds - 1)
        
        if strategy == "gradual_build":
            # Linear increase
            intensity = initial + (maximum - initial) * progress
        elif strategy == "spike":
            # Sharp increase at midpoint
            if progress < 0.5:
                intensity = initial + (maximum - initial) * (progress * 2)
            else:
                intensity = maximum
        elif strategy == "oscillate":
            # Oscillating intensity
            import math
            intensity = initial + (maximum - initial) * (math.sin(progress * math.pi * 2) + 1) / 2
        else:
            # Default to gradual build
            intensity = initial + (maximum - initial) * progress
        
        return min(maximum, max(initial, intensity))
    
    def _create_intensity_prompt(self,
                               topic: str,
                               intensity: float,
                               state: DialogueState,
                               agent: 'AgentNet') -> str:
        """Create prompt modulated by current intensity level."""
        # Adjust language and urgency based on intensity
        if intensity < 0.3:
            urgency = "Consider carefully"
            tone = "thoughtful"
        elif intensity < 0.6:
            urgency = "This is becoming important"
            tone = "engaged"
        elif intensity < 0.8:
            urgency = "This requires immediate attention"
            tone = "urgent"
        else:
            urgency = "This is critical"
            tone = "intense"
        
        # Include context from previous turns
        context_summary = ""
        if state.transcript:
            recent_turns = state.transcript[-3:]  # Last 3 turns for building context
            context_summary = f"\n\nEscalating discussion:\n"
            for turn in recent_turns:
                context_summary += f"- {turn.agent_name} (intensity {turn.intensity_level:.2f}): {turn.content[:80]}...\n"
        
        prompt = f"""
{urgency} - Topic: {topic}

Current intensity level: {intensity:.2f}
Conversation tone: {tone}
Round: {state.current_round}

{context_summary}

The conversation is building in intensity. Please respond with {tone} engagement, 
providing important details that advance the discussion while matching the current intensity level.
Focus on revealing crucial information or insights that may have been overlooked.
"""
        return prompt.strip()
    
    def _get_intensity_style_override(self, intensity: float) -> Dict[str, float]:
        """Get style override based on intensity level."""
        # Higher intensity increases creativity and reduces pure logic
        base_creativity = min(0.9, 0.3 + intensity * 0.6)
        base_logic = max(0.3, 0.8 - intensity * 0.3)
        base_analytical = 0.5 + intensity * 0.3
        
        return {
            "creativity": base_creativity,
            "logic": base_logic,
            "analytical": base_analytical
        }
    
    def _analyze_modulated_conversation(self, state: DialogueState) -> Dict[str, Any]:
        """Analyze modulated conversation for intensity patterns and effectiveness."""
        if not state.transcript:
            return {"error": "No dialogue content to analyze"}
        
        # Analyze intensity progression
        intensities = [turn.intensity_level for turn in state.transcript if turn.intensity_level]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0.0
        max_intensity_reached = max(intensities) if intensities else 0.0
        
        # Analyze content richness (more details revealed at higher intensities)
        content_lengths = [len(turn.content) for turn in state.transcript]
        detail_progression = []
        
        for i, turn in enumerate(state.transcript):
            if turn.intensity_level and i > 0:
                detail_change = len(turn.content) - len(state.transcript[i-1].content)
                detail_progression.append(detail_change)
        
        return {
            "average_intensity": avg_intensity,
            "max_intensity_reached": max_intensity_reached,
            "intensity_range": max_intensity_reached - min(intensities) if intensities else 0.0,
            "detail_progression_effective": sum(detail_progression) > 0 if detail_progression else False,
            "conversation_climax_round": self._find_climax_round(state.transcript),
            "tension_building_success": max_intensity_reached > 0.7
        }
    
    def _find_climax_round(self, transcript: List[DialogueTurn]) -> int:
        """Find the round with highest intensity (conversation climax)."""
        max_intensity = 0.0
        climax_round = 1
        
        for turn in transcript:
            if turn.intensity_level and turn.intensity_level > max_intensity:
                max_intensity = turn.intensity_level
                climax_round = turn.round_number
        
        return climax_round
    
    def _extract_intensity_progression(self, state: DialogueState) -> List[Dict[str, Any]]:
        """Extract intensity progression data for analysis."""
        progression = []
        
        for i, turn in enumerate(state.transcript):
            progression.append({
                "round": turn.round_number,
                "agent": turn.agent_name,
                "intensity": turn.intensity_level,
                "content_length": len(turn.content),
                "confidence": turn.confidence
            })
        
        return progression


class InterpolationConversation(BaseDialogueMode):
    """
    Interpolation conversation: Insert contextual information where direct details aren't provided.
    Fills knowledge gaps during agent interactions using memory and reasoning.
    """
    
    def __init__(self):
        """Initialize interpolation conversation mode."""
        super().__init__(DialogueMode.INTERPOLATION)
    
    def conduct_dialogue(self,
                        agents: List['AgentNet'],
                        topic: str,
                        rounds: int = 5,
                        context_gaps: Optional[List[str]] = None,
                        interpolation_strategy: str = "memory_guided",
                        **kwargs) -> Dict[str, Any]:
        """
        Conduct interpolation conversation that fills knowledge gaps.
        
        Args:
            agents: List of participating agents
            topic: Dialogue topic
            rounds: Number of dialogue rounds  
            context_gaps: Known gaps in context/knowledge
            interpolation_strategy: Strategy for gap filling (memory_guided, reasoning_based, collaborative)
            
        Returns:
            Dialogue results with gap analysis and interpolations
        """
        context_gaps = context_gaps or self._identify_initial_gaps(topic)
        dialogue_state = self._create_dialogue_state(
            agents, topic, context_gaps=context_gaps, **kwargs
        )
        
        logger.info(f"Starting interpolation conversation: {dialogue_state.session_id}")
        
        for round_num in range(rounds):
            dialogue_state.current_round = round_num + 1
            
            # Identify current knowledge gaps
            current_gaps = self._identify_current_gaps(dialogue_state, context_gaps)
            
            for agent in agents:
                # Create gap-filling prompt
                interpolation_prompt = self._create_interpolation_prompt(
                    topic, current_gaps, dialogue_state, agent, interpolation_strategy
                )
                
                # Generate response with memory context if available
                if hasattr(agent, 'generate_reasoning_tree_enhanced') and interpolation_strategy == "memory_guided":
                    memory_context = self._create_memory_context_for_gaps(current_gaps)
                    response = agent.generate_reasoning_tree_enhanced(
                        interpolation_prompt, 
                        use_memory=True,
                        memory_context=memory_context
                    )
                else:
                    response = agent.generate_reasoning_tree(interpolation_prompt)
                
                # Identify interpolated content
                interpolated_content = self._extract_interpolations(
                    response.get("result", {}).get("content", ""), current_gaps
                )
                
                # Create dialogue turn with interpolation tracking
                turn = DialogueTurn(
                    agent_name=agent.name,
                    content=response.get("result", {}).get("content", ""),
                    confidence=response.get("result", {}).get("confidence", 0.5),
                    mode=DialogueMode.INTERPOLATION,
                    round_number=round_num + 1,
                    timestamp=time.time(),
                    context_additions=interpolated_content,
                    reasoning_type=response.get("reasoning_metadata", {}).get("reasoning_type")
                )
                
                dialogue_state.transcript.append(turn)
                
                # Update gaps based on what was filled
                context_gaps = self._update_gaps_after_turn(context_gaps, interpolated_content)
        
        # Final gap analysis
        analysis = self._analyze_interpolation_conversation(dialogue_state, context_gaps)
        
        return {
            "session_id": dialogue_state.session_id,
            "mode": "interpolation",
            "interpolation_strategy": interpolation_strategy,
            "participants": dialogue_state.participants,
            "topic": dialogue_state.topic,
            "transcript": [self._turn_to_dict(turn) for turn in dialogue_state.transcript],
            "initial_gaps": dialogue_state.context_gaps,
            "remaining_gaps": context_gaps,
            "interpolations_made": self._extract_all_interpolations(dialogue_state),
            "analysis": analysis,
            "rounds_completed": rounds
        }
    
    async def conduct_async_dialogue(self,
                                   agents: List['AgentNet'],
                                   topic: str,
                                   rounds: int = 5,
                                   context_gaps: Optional[List[str]] = None,
                                   interpolation_strategy: str = "memory_guided",
                                   **kwargs) -> Dict[str, Any]:
        """Async version of interpolation conversation."""
        return self.conduct_dialogue(
            agents, topic, rounds, context_gaps, interpolation_strategy, **kwargs
        )
    
    def _identify_initial_gaps(self, topic: str) -> List[str]:
        """Identify initial knowledge gaps from the topic."""
        gaps = []
        
        # Simple heuristic gap identification
        if "?" in topic:
            gaps.append("Question requires specific information")
        if "unclear" in topic.lower() or "unknown" in topic.lower():
            gaps.append("Ambiguity needs clarification")
        if "context" in topic.lower():
            gaps.append("Additional context needed")
        
        return gaps if gaps else ["General context enhancement needed"]
    
    def _identify_current_gaps(self, state: DialogueState, original_gaps: List[str]) -> List[str]:
        """Identify current knowledge gaps based on dialogue progress."""
        # Start with original gaps
        current_gaps = original_gaps.copy()
        
        # Remove gaps that may have been addressed
        if state.transcript:
            last_turn = state.transcript[-1]
            if last_turn.context_additions:
                # Filter out gaps that were addressed
                current_gaps = [gap for gap in current_gaps 
                              if not any(addition.lower() in gap.lower() 
                                       for addition in last_turn.context_additions)]
        
        # Add new gaps identified from recent dialogue
        if len(state.transcript) > 1:
            recent_content = state.transcript[-1].content.lower()
            if "need more" in recent_content or "unclear" in recent_content:
                current_gaps.append("Dialogue generated new information need")
        
        return current_gaps
    
    def _create_interpolation_prompt(self,
                                   topic: str,
                                   gaps: List[str],
                                   state: DialogueState,
                                   agent: 'AgentNet',
                                   strategy: str) -> str:
        """Create prompt designed to elicit gap-filling information."""
        # Describe the gaps that need filling
        gaps_description = "\n".join([f"- {gap}" for gap in gaps])
        
        # Include relevant context from previous turns
        context_summary = ""
        if state.transcript:
            recent_turns = state.transcript[-2:]
            context_summary = f"\n\nPrevious context provided:\n"
            for turn in recent_turns:
                if turn.context_additions:
                    context_summary += f"- {turn.agent_name} added: {', '.join(turn.context_additions)}\n"
        
        strategy_instruction = self._get_strategy_instruction(strategy)
        
        prompt = f"""
Topic: {topic}

Knowledge gaps identified:
{gaps_description}

{context_summary}

{strategy_instruction}

Please provide information that fills these knowledge gaps, adding contextual details 
that weren't directly provided but are necessary for complete understanding.
Focus on interpolating missing information based on your knowledge and reasoning.
"""
        return prompt.strip()
    
    def _get_strategy_instruction(self, strategy: str) -> str:
        """Get strategy-specific instruction for interpolation."""
        strategies = {
            "memory_guided": "Use your memory and past experience to fill information gaps.",
            "reasoning_based": "Use logical reasoning to infer missing information.",
            "collaborative": "Build on others' contributions to complete the picture."
        }
        return strategies.get(strategy, strategies["memory_guided"])
    
    def _create_memory_context_for_gaps(self, gaps: List[str]) -> Dict[str, Any]:
        """Create memory context dictionary focused on addressing gaps."""
        return {
            "focus_areas": gaps,
            "retrieval_intent": "gap_filling",
            "context_type": "interpolation"
        }
    
    def _extract_interpolations(self, content: str, gaps: List[str]) -> List[str]:
        """Extract interpolated information from agent response."""
        interpolations = []
        
        # Simple heuristic extraction of new information
        content_lower = content.lower()
        
        # Look for interpolation indicators
        interpolation_phrases = [
            "it's likely that", "this suggests", "we can infer", 
            "based on this", "this implies", "additionally"
        ]
        
        for phrase in interpolation_phrases:
            if phrase in content_lower:
                # Extract the sentence containing the interpolation
                sentences = content.split('. ')
                for sentence in sentences:
                    if phrase in sentence.lower():
                        interpolations.append(sentence.strip())
        
        # If no specific interpolations found, check if content addresses gaps
        for gap in gaps:
            gap_keywords = gap.lower().split()
            if any(keyword in content_lower for keyword in gap_keywords):
                interpolations.append(f"Addressed: {gap}")
        
        return interpolations
    
    def _update_gaps_after_turn(self, gaps: List[str], interpolations: List[str]) -> List[str]:
        """Update remaining gaps after a turn with interpolations."""
        remaining_gaps = []
        
        for gap in gaps:
            gap_addressed = False
            for interpolation in interpolations:
                if any(word in interpolation.lower() for word in gap.lower().split()):
                    gap_addressed = True
                    break
            
            if not gap_addressed:
                remaining_gaps.append(gap)
        
        return remaining_gaps
    
    def _analyze_interpolation_conversation(self, state: DialogueState, remaining_gaps: List[str]) -> Dict[str, Any]:
        """Analyze interpolation conversation effectiveness."""
        if not state.transcript:
            return {"error": "No dialogue content to analyze"}
        
        initial_gap_count = len(state.context_gaps)
        remaining_gap_count = len(remaining_gaps)
        gaps_filled = initial_gap_count - remaining_gap_count
        
        # Count total interpolations made
        total_interpolations = sum(len(turn.context_additions) if turn.context_additions else 0 
                                 for turn in state.transcript)
        
        # Analyze interpolation quality
        interpolation_quality = self._assess_interpolation_quality(state.transcript)
        
        return {
            "initial_gaps": initial_gap_count,
            "gaps_filled": gaps_filled,
            "remaining_gaps": remaining_gap_count,
            "gap_filling_rate": gaps_filled / max(1, initial_gap_count),
            "total_interpolations": total_interpolations,
            "interpolation_quality": interpolation_quality,
            "effectiveness": "high" if gaps_filled / max(1, initial_gap_count) > 0.7 else "moderate"
        }
    
    def _assess_interpolation_quality(self, transcript: List[DialogueTurn]) -> str:
        """Assess the quality of interpolations made."""
        if not transcript:
            return "insufficient_data"
        
        # Count turns with successful interpolations
        turns_with_interpolations = sum(1 for turn in transcript if turn.context_additions)
        interpolation_rate = turns_with_interpolations / len(transcript)
        
        if interpolation_rate > 0.6:
            return "high_quality"
        elif interpolation_rate > 0.3:
            return "moderate_quality"
        else:
            return "needs_improvement"
    
    def _extract_all_interpolations(self, state: DialogueState) -> List[Dict[str, Any]]:
        """Extract all interpolations made during the conversation."""
        all_interpolations = []
        
        for turn in state.transcript:
            if turn.context_additions:
                for addition in turn.context_additions:
                    all_interpolations.append({
                        "agent": turn.agent_name,
                        "round": turn.round_number,
                        "interpolation": addition,
                        "confidence": turn.confidence
                    })
        
        return all_interpolations


class InnerDialogue(BaseDialogueMode):
    """
    Inner (Internal) dialogue: Agent self-reflection capabilities.
    Makes internal reasoning processes explicit and enables self-analysis.
    """
    
    def __init__(self):
        """Initialize inner dialogue mode."""
        super().__init__(DialogueMode.INNER)
    
    def conduct_dialogue(self,
                        agents: List['AgentNet'],
                        topic: str,
                        rounds: int = 3,
                        reflection_depth: str = "moderate",
                        focus_areas: Optional[List[str]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Conduct inner dialogue for agent self-reflection.
        
        Args:
            agents: List of agents (typically one for true inner dialogue)
            topic: Topic for self-reflection
            rounds: Number of reflection rounds
            reflection_depth: Depth of reflection (surface, moderate, deep)
            focus_areas: Specific areas for reflection (decision_making, reasoning, beliefs)
            
        Returns:
            Inner dialogue results with self-analysis
        """
        dialogue_state = self._create_dialogue_state(agents, topic, **kwargs)
        focus_areas = focus_areas or ["reasoning_process", "assumptions", "confidence_assessment"]
        
        logger.info(f"Starting inner dialogue: {dialogue_state.session_id}")
        
        # Inner dialogue typically involves a single agent reflecting
        primary_agent = agents[0] if agents else None
        if not primary_agent:
            raise ValueError("Inner dialogue requires at least one agent")
        
        internal_state_progression = []
        
        for round_num in range(rounds):
            dialogue_state.current_round = round_num + 1
            
            # Create self-reflection prompt for current focus area
            focus_area = focus_areas[round_num % len(focus_areas)] if focus_areas else "general_reflection"
            reflection_prompt = self._create_reflection_prompt(
                topic, focus_area, dialogue_state, reflection_depth
            )
            
            # Generate self-reflective response
            if hasattr(primary_agent, 'generate_reasoning_tree_enhanced'):
                response = primary_agent.generate_reasoning_tree_enhanced(
                    reflection_prompt,
                    include_monitor_trace=True  # Include monitoring for self-analysis
                )
            else:
                response = primary_agent.generate_reasoning_tree(reflection_prompt)
            
            # Extract internal state information
            internal_state = self._extract_internal_state(response, focus_area)
            internal_state_progression.append(internal_state)
            
            # Create inner dialogue turn
            turn = DialogueTurn(
                agent_name=primary_agent.name,
                content=response.get("result", {}).get("content", ""),
                confidence=response.get("result", {}).get("confidence", 0.5),
                mode=DialogueMode.INNER,
                round_number=round_num + 1,
                timestamp=time.time(),
                internal_state=internal_state,
                reasoning_type=response.get("reasoning_metadata", {}).get("reasoning_type")
            )
            
            dialogue_state.transcript.append(turn)
        
        # Analyze inner dialogue for insights
        analysis = self._analyze_inner_dialogue(dialogue_state, internal_state_progression)
        
        return {
            "session_id": dialogue_state.session_id,
            "mode": "inner",
            "reflection_depth": reflection_depth,
            "agent": primary_agent.name,
            "topic": dialogue_state.topic,
            "focus_areas": focus_areas,
            "transcript": [self._turn_to_dict(turn) for turn in dialogue_state.transcript],
            "internal_state_progression": internal_state_progression,
            "analysis": analysis,
            "rounds_completed": rounds
        }
    
    async def conduct_async_dialogue(self,
                                   agents: List['AgentNet'],
                                   topic: str,
                                   rounds: int = 3,
                                   reflection_depth: str = "moderate",
                                   focus_areas: Optional[List[str]] = None,
                                   **kwargs) -> Dict[str, Any]:
        """Async version of inner dialogue."""
        return self.conduct_dialogue(agents, topic, rounds, reflection_depth, focus_areas, **kwargs)
    
    def _create_reflection_prompt(self,
                                topic: str,
                                focus_area: str,
                                state: DialogueState,
                                depth: str) -> str:
        """Create prompt for self-reflection."""
        depth_instructions = {
            "surface": "Briefly reflect on your immediate thoughts and reactions.",
            "moderate": "Examine your reasoning process and underlying assumptions.",
            "deep": "Engage in thorough self-analysis of your cognitive processes, biases, and decision-making patterns."
        }
        
        focus_instructions = {
            "reasoning_process": "Focus on how you approached the reasoning and what steps you took.",
            "assumptions": "Examine what assumptions you made and whether they are justified.",
            "confidence_assessment": "Analyze your confidence levels and what factors influenced them.",
            "decision_making": "Reflect on how you made decisions and what alternatives you considered.",
            "beliefs": "Explore your underlying beliefs and how they shaped your perspective.",
            "biases": "Consider potential biases that may have influenced your thinking."
        }
        
        depth_instruction = depth_instructions.get(depth, depth_instructions["moderate"])
        focus_instruction = focus_instructions.get(focus_area, focus_instructions["reasoning_process"])
        
        # Include previous self-reflection context
        reflection_context = ""
        if state.transcript:
            reflection_context = f"\n\nPrevious self-reflection:\n"
            for turn in state.transcript[-2:]:  # Last 2 reflections
                reflection_context += f"Round {turn.round_number}: {turn.content[:100]}...\n"
        
        prompt = f"""
INNER DIALOGUE - Self-Reflection Session

Topic: {topic}
Focus Area: {focus_area}
Reflection Round: {state.current_round}

{depth_instruction}
{focus_instruction}

{reflection_context}

Please engage in honest self-reflection about your thinking process regarding this topic.
Consider your internal cognitive state, decision-making process, and any insights about your own reasoning.
This is an internal dialogue - be candid about your thought processes, uncertainties, and mental models.
"""
        return prompt.strip()
    
    def _extract_internal_state(self, response: Dict[str, Any], focus_area: str) -> Dict[str, Any]:
        """Extract internal state information from reflection response."""
        content = response.get("result", {}).get("content", "")
        confidence = response.get("result", {}).get("confidence", 0.5)
        
        # Extract different aspects of internal state
        internal_state = {
            "focus_area": focus_area,
            "confidence_level": confidence,
            "self_awareness_indicators": self._identify_self_awareness_indicators(content),
            "uncertainty_acknowledgment": "uncertain" in content.lower() or "unsure" in content.lower(),
            "assumption_recognition": "assume" in content.lower() or "assumption" in content.lower(),
            "bias_awareness": "bias" in content.lower() or "biased" in content.lower(),
            "metacognitive_statements": self._extract_metacognitive_statements(content),
            "emotional_indicators": self._identify_emotional_indicators(content)
        }
        
        # Add reasoning metadata if available
        if "reasoning_metadata" in response:
            internal_state["reasoning_type"] = response["reasoning_metadata"].get("reasoning_type")
            internal_state["reasoning_confidence"] = response["reasoning_metadata"].get("reasoning_confidence")
        
        return internal_state
    
    def _identify_self_awareness_indicators(self, content: str) -> List[str]:
        """Identify indicators of self-awareness in the reflection."""
        indicators = []
        content_lower = content.lower()
        
        self_awareness_phrases = [
            "i think", "i believe", "i feel", "i realize", "i notice",
            "my reasoning", "my approach", "my understanding", "my perspective",
            "i tend to", "i usually", "i often"
        ]
        
        for phrase in self_awareness_phrases:
            if phrase in content_lower:
                indicators.append(phrase)
        
        return indicators
    
    def _extract_metacognitive_statements(self, content: str) -> List[str]:
        """Extract statements that show thinking about thinking."""
        metacognitive_statements = []
        
        # Look for sentences that contain metacognitive indicators
        sentences = content.split('.')
        metacognitive_indicators = [
            "thinking about", "reasoning process", "my mind", "my thoughts",
            "how I think", "my mental", "cognitive", "my approach to"
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in metacognitive_indicators):
                metacognitive_statements.append(sentence.strip())
        
        return metacognitive_statements[:3]  # Limit to top 3
    
    def _identify_emotional_indicators(self, content: str) -> List[str]:
        """Identify emotional indicators in the reflection."""
        emotional_words = [
            "confident", "uncertain", "worried", "excited", "frustrated",
            "curious", "surprised", "confused", "satisfied", "concerned"
        ]
        
        content_lower = content.lower()
        found_emotions = [emotion for emotion in emotional_words if emotion in content_lower]
        
        return found_emotions
    
    def _analyze_inner_dialogue(self, state: DialogueState, internal_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze inner dialogue for self-awareness and reflection quality."""
        if not state.transcript or not internal_states:
            return {"error": "No inner dialogue content to analyze"}
        
        # Analyze self-awareness progression
        self_awareness_progression = []
        for i, internal_state in enumerate(internal_states):
            awareness_score = len(internal_state.get("self_awareness_indicators", []))
            self_awareness_progression.append(awareness_score)
        
        # Analyze metacognitive development
        metacognitive_depth = sum(len(state.get("metacognitive_statements", [])) for state in internal_states)
        
        # Check for uncertainty acknowledgment and bias awareness
        uncertainty_acknowledgments = sum(1 for state in internal_states if state.get("uncertainty_acknowledgment", False))
        bias_awareness_instances = sum(1 for state in internal_states if state.get("bias_awareness", False))
        
        # Calculate reflection quality metrics
        avg_confidence = sum(turn.confidence for turn in state.transcript) / len(state.transcript)
        emotional_range = len(set().union(*[state.get("emotional_indicators", []) for state in internal_states]))
        
        return {
            "self_awareness_progression": self_awareness_progression,
            "metacognitive_depth": metacognitive_depth,
            "uncertainty_acknowledgments": uncertainty_acknowledgments,
            "bias_awareness_instances": bias_awareness_instances,
            "average_reflection_confidence": avg_confidence,
            "emotional_range": emotional_range,
            "reflection_quality": self._assess_reflection_quality(internal_states),
            "cognitive_insights_count": len([state for state in internal_states if state.get("metacognitive_statements")])
        }
    
    def _assess_reflection_quality(self, internal_states: List[Dict[str, Any]]) -> str:
        """Assess overall quality of self-reflection."""
        if not internal_states:
            return "insufficient_data"
        
        # Quality indicators
        total_awareness_indicators = sum(len(state.get("self_awareness_indicators", [])) for state in internal_states)
        total_metacognitive_statements = sum(len(state.get("metacognitive_statements", [])) for state in internal_states)
        uncertainty_rate = sum(1 for state in internal_states if state.get("uncertainty_acknowledgment", False)) / len(internal_states)
        
        quality_score = (
            min(10, total_awareness_indicators) +
            min(10, total_metacognitive_statements) +
            uncertainty_rate * 10
        ) / 3
        
        if quality_score > 7:
            return "high_quality"
        elif quality_score > 4:
            return "moderate_quality"
        else:
            return "needs_improvement"


class DialogueMapping(BaseDialogueMode):
    """
    Dialogue mapping: Collaborative decision-making visualization.
    Tracks decision points, arguments, and solution-building processes.
    """
    
    def __init__(self):
        """Initialize dialogue mapping mode."""
        super().__init__(DialogueMode.MAPPING)
    
    def conduct_dialogue(self,
                        agents: List['AgentNet'],
                        topic: str,
                        rounds: int = 5,
                        mapping_focus: str = "decision_making",
                        track_arguments: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """
        Conduct dialogue with decision mapping and visualization.
        
        Args:
            agents: List of participating agents
            topic: Dialogue topic (typically a decision or problem)
            rounds: Number of dialogue rounds
            mapping_focus: Focus of mapping (decision_making, problem_solving, consensus_building)
            track_arguments: Whether to track arguments and counter-arguments
            
        Returns:
            Dialogue results with decision map and visualization data
        """
        dialogue_state = self._create_dialogue_state(agents, topic, **kwargs)
        dialogue_state.decision_map = self._initialize_decision_map(topic, mapping_focus)
        
        logger.info(f"Starting dialogue mapping: {dialogue_state.session_id}")
        
        for round_num in range(rounds):
            dialogue_state.current_round = round_num + 1
            
            for agent in agents:
                # Create mapping-focused prompt
                mapping_prompt = self._create_mapping_prompt(
                    topic, dialogue_state, agent, mapping_focus, track_arguments
                )
                
                # Generate response with mapping awareness
                if hasattr(agent, 'generate_reasoning_tree_enhanced'):
                    response = agent.generate_reasoning_tree_enhanced(mapping_prompt)
                else:
                    response = agent.generate_reasoning_tree(mapping_prompt)
                
                # Create dialogue turn
                turn = DialogueTurn(
                    agent_name=agent.name,
                    content=response.get("result", {}).get("content", ""),
                    confidence=response.get("result", {}).get("confidence", 0.5),
                    mode=DialogueMode.MAPPING,
                    round_number=round_num + 1,
                    timestamp=time.time(),
                    reasoning_type=response.get("reasoning_metadata", {}).get("reasoning_type")
                )
                
                dialogue_state.transcript.append(turn)
                
                # Update decision map with new information
                self._update_decision_map(dialogue_state.decision_map, turn, track_arguments)
        
        # Generate final decision map analysis
        mapping_analysis = self._analyze_decision_map(dialogue_state.decision_map, dialogue_state.transcript)
        
        return {
            "session_id": dialogue_state.session_id,
            "mode": "mapping",
            "mapping_focus": mapping_focus,
            "participants": dialogue_state.participants,
            "topic": dialogue_state.topic,
            "transcript": [self._turn_to_dict(turn) for turn in dialogue_state.transcript],
            "decision_map": dialogue_state.decision_map,
            "mapping_analysis": mapping_analysis,
            "visualization_data": self._generate_visualization_data(dialogue_state.decision_map),
            "rounds_completed": rounds
        }
    
    async def conduct_async_dialogue(self,
                                   agents: List['AgentNet'],
                                   topic: str,
                                   rounds: int = 5,
                                   mapping_focus: str = "decision_making",
                                   track_arguments: bool = True,
                                   **kwargs) -> Dict[str, Any]:
        """Async version of dialogue mapping."""
        return self.conduct_dialogue(agents, topic, rounds, mapping_focus, track_arguments, **kwargs)
    
    def _initialize_decision_map(self, topic: str, focus: str) -> Dict[str, Any]:
        """Initialize the decision map structure."""
        return {
            "topic": topic,
            "focus": focus,
            "decision_points": [],
            "arguments": {"pro": [], "con": [], "neutral": []},
            "alternatives": [],
            "constraints": [],
            "stakeholders": [],
            "criteria": [],
            "solution_elements": [],
            "convergence_points": [],
            "created_at": time.time(),
            "last_updated": time.time()
        }
    
    def _create_mapping_prompt(self,
                             topic: str,
                             state: DialogueState,
                             agent: 'AgentNet',
                             focus: str,
                             track_arguments: bool) -> str:
        """Create prompt designed for decision mapping."""
        focus_instructions = {
            "decision_making": "Focus on identifying decision points, alternatives, criteria, and trade-offs.",
            "problem_solving": "Focus on problem definition, solution alternatives, and implementation approaches.",
            "consensus_building": "Focus on areas of agreement, disagreement, and paths to consensus."
        }
        
        focus_instruction = focus_instructions.get(focus, focus_instructions["decision_making"])
        
        # Include current map state
        map_summary = self._summarize_current_map(state.decision_map)
        
        # Include recent dialogue context
        context_summary = ""
        if state.transcript:
            recent_turns = state.transcript[-2:]
            context_summary = f"\n\nRecent discussion:\n"
            for turn in recent_turns:
                context_summary += f"- {turn.agent_name}: {turn.content[:80]}...\n"
        
        argument_instruction = ""
        if track_arguments:
            argument_instruction = "\nClearly indicate if you are supporting, opposing, or providing neutral analysis of ideas."
        
        prompt = f"""
DIALOGUE MAPPING SESSION

Topic: {topic}
Mapping Focus: {focus}
Round: {state.current_round}

{focus_instruction}

Current decision map state:
{map_summary}

{context_summary}

Please contribute to the dialogue by:
1. Identifying key decision points or solution elements
2. Proposing alternatives or options
3. Highlighting important criteria or constraints
4. Building on others' contributions constructively

{argument_instruction}

Your response should help advance the {focus} process and contribute valuable elements to the decision map.
"""
        return prompt.strip()
    
    def _summarize_current_map(self, decision_map: Dict[str, Any]) -> str:
        """Summarize current state of the decision map."""
        summary_parts = []
        
        decision_points = decision_map.get("decision_points", [])
        if decision_points:
            summary_parts.append(f"Decision points identified: {len(decision_points)}")
        
        alternatives = decision_map.get("alternatives", [])
        if alternatives:
            summary_parts.append(f"Alternatives proposed: {len(alternatives)}")
        
        arguments = decision_map.get("arguments", {})
        total_args = sum(len(args) for args in arguments.values())
        if total_args > 0:
            summary_parts.append(f"Arguments made: {total_args}")
        
        criteria = decision_map.get("criteria", [])
        if criteria:
            summary_parts.append(f"Criteria established: {len(criteria)}")
        
        if not summary_parts:
            return "Decision map is being initialized"
        
        return "; ".join(summary_parts)
    
    def _update_decision_map(self, decision_map: Dict[str, Any], turn: DialogueTurn, track_arguments: bool) -> None:
        """Update decision map with information from the latest turn."""
        content = turn.content.lower()
        
        # Extract decision points
        decision_indicators = ["decide", "choice", "option", "alternative"]
        if any(indicator in content for indicator in decision_indicators):
            decision_map["decision_points"].append({
                "agent": turn.agent_name,
                "round": turn.round_number,
                "description": turn.content[:200] + "..." if len(turn.content) > 200 else turn.content,
                "confidence": turn.confidence
            })
        
        # Extract alternatives
        alternative_indicators = ["alternative", "option", "choice", "approach", "solution"]
        if any(indicator in content for indicator in alternative_indicators):
            decision_map["alternatives"].append({
                "agent": turn.agent_name,
                "round": turn.round_number,
                "alternative": turn.content[:150] + "..." if len(turn.content) > 150 else turn.content,
                "confidence": turn.confidence
            })
        
        # Track arguments if enabled
        if track_arguments:
            if any(word in content for word in ["support", "agree", "favor", "prefer"]):
                decision_map["arguments"]["pro"].append({
                    "agent": turn.agent_name,
                    "round": turn.round_number,
                    "argument": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                })
            elif any(word in content for word in ["disagree", "oppose", "against", "problem"]):
                decision_map["arguments"]["con"].append({
                    "agent": turn.agent_name,
                    "round": turn.round_number,
                    "argument": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                })
            else:
                decision_map["arguments"]["neutral"].append({
                    "agent": turn.agent_name,
                    "round": turn.round_number,
                    "argument": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                })
        
        # Extract criteria
        criteria_indicators = ["criteria", "requirement", "important", "consider", "factor"]
        if any(indicator in content for indicator in criteria_indicators):
            decision_map["criteria"].append({
                "agent": turn.agent_name,
                "round": turn.round_number,
                "criterion": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
            })
        
        # Extract constraints
        constraint_indicators = ["constraint", "limitation", "can't", "cannot", "must not", "limited"]
        if any(indicator in content for indicator in constraint_indicators):
            decision_map["constraints"].append({
                "agent": turn.agent_name,
                "round": turn.round_number,
                "constraint": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
            })
        
        decision_map["last_updated"] = time.time()
    
    def _analyze_decision_map(self, decision_map: Dict[str, Any], transcript: List[DialogueTurn]) -> Dict[str, Any]:
        """Analyze the decision map for insights and patterns."""
        if not transcript:
            return {"error": "No dialogue content to analyze"}
        
        # Analyze participation in mapping
        mapping_contributions = {}
        for point_type in ["decision_points", "alternatives", "criteria", "constraints"]:
            contributions = decision_map.get(point_type, [])
            for contrib in contributions:
                agent = contrib.get("agent", "unknown")
                mapping_contributions[agent] = mapping_contributions.get(agent, 0) + 1
        
        # Analyze argument balance if arguments were tracked
        argument_balance = {}
        arguments = decision_map.get("arguments", {})
        if arguments:
            argument_balance = {
                "pro_arguments": len(arguments.get("pro", [])),
                "con_arguments": len(arguments.get("con", [])),
                "neutral_arguments": len(arguments.get("neutral", [])),
                "total_arguments": sum(len(args) for args in arguments.values())
            }
        
        # Calculate mapping effectiveness
        mapping_elements_count = sum(len(decision_map.get(key, [])) for key in 
                                   ["decision_points", "alternatives", "criteria", "constraints"])
        
        mapping_effectiveness = "high" if mapping_elements_count > 10 else "moderate" if mapping_elements_count > 5 else "low"
        
        return {
            "mapping_contributions": mapping_contributions,
            "argument_balance": argument_balance,
            "total_mapping_elements": mapping_elements_count,
            "mapping_effectiveness": mapping_effectiveness,
            "decision_points_identified": len(decision_map.get("decision_points", [])),
            "alternatives_proposed": len(decision_map.get("alternatives", [])),
            "criteria_established": len(decision_map.get("criteria", [])),
            "constraints_identified": len(decision_map.get("constraints", []))
        }
    
    def _generate_visualization_data(self, decision_map: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data structure suitable for visualization."""
        # Create nodes and edges for decision map visualization
        nodes = []
        edges = []
        
        # Central topic node
        nodes.append({
            "id": "topic",
            "label": decision_map["topic"],
            "type": "topic",
            "size": 20,
            "color": "#4CAF50"
        })
        
        # Decision point nodes
        for i, decision_point in enumerate(decision_map.get("decision_points", [])):
            node_id = f"decision_{i}"
            nodes.append({
                "id": node_id,
                "label": f"Decision: {decision_point['description'][:30]}...",
                "type": "decision",
                "size": 15,
                "color": "#2196F3",
                "agent": decision_point["agent"],
                "confidence": decision_point["confidence"]
            })
            edges.append({"from": "topic", "to": node_id, "type": "decision"})
        
        # Alternative nodes
        for i, alternative in enumerate(decision_map.get("alternatives", [])):
            node_id = f"alt_{i}"
            nodes.append({
                "id": node_id,
                "label": f"Alternative: {alternative['alternative'][:30]}...",
                "type": "alternative",
                "size": 12,
                "color": "#FF9800",
                "agent": alternative["agent"],
                "confidence": alternative["confidence"]
            })
            edges.append({"from": "topic", "to": node_id, "type": "alternative"})
        
        # Criteria nodes
        for i, criterion in enumerate(decision_map.get("criteria", [])):
            node_id = f"criteria_{i}"
            nodes.append({
                "id": node_id,
                "label": f"Criterion: {criterion['criterion'][:30]}...",
                "type": "criterion",
                "size": 10,
                "color": "#9C27B0",
                "agent": criterion["agent"]
            })
            edges.append({"from": "topic", "to": node_id, "type": "criterion"})
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "hierarchical",
            "legend": {
                "topic": {"color": "#4CAF50", "description": "Main topic"},
                "decision": {"color": "#2196F3", "description": "Decision points"},
                "alternative": {"color": "#FF9800", "description": "Alternatives"},
                "criterion": {"color": "#9C27B0", "description": "Criteria"}
            }
        }