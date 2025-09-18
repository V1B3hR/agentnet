"""
Solution-focused conversation patterns based on managebetter.com research.

Implements problem-to-solution transition mechanisms, constructive questioning frameworks,
and collaborative problem-solving dialogue flows.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.agent import AgentNet

logger = logging.getLogger("agentnet.dialogue.solution_focused")


class ConversationPhase(str, Enum):
    """Phases of solution-focused conversation."""
    PROBLEM_IDENTIFICATION = "problem_identification"
    SOLUTION_EXPLORATION = "solution_exploration"
    SOLUTION_REFINEMENT = "solution_refinement"
    ACTION_PLANNING = "action_planning"


class QuestionType(str, Enum):
    """Types of constructive questions."""
    CLARIFYING = "clarifying"
    EXPLORING = "exploring"
    SCALING = "scaling"
    EXCEPTION_FINDING = "exception_finding"
    FUTURE_FOCUSED = "future_focused"
    RESOURCE_IDENTIFYING = "resource_identifying"


@dataclass
class SolutionElement:
    """Represents a solution element identified during conversation."""
    description: str
    proposed_by: str
    confidence: float
    feasibility: str
    phase_introduced: ConversationPhase
    supporting_agents: List[str] = None
    refinements: List[str] = None


@dataclass
class ProblemStatement:
    """Represents a problem statement."""
    description: str
    identified_by: str
    severity: str
    stakeholders: List[str] = None
    context: Dict[str, Any] = None


class ProblemSolutionTransition:
    """
    Manages the transition from problem identification to solution development.
    
    Based on solution-focused therapy and management research, this class helps
    guide conversations toward constructive outcomes.
    """
    
    def __init__(self):
        """Initialize problem-solution transition manager."""
        self.transition_patterns = {
            "direct": "Let's focus on what would need to change to solve this.",
            "scaling": "On a scale of 1-10, where would we like to be, and what would that look like?",
            "exception": "When has this problem been less severe or absent? What was different then?",
            "future_focused": "Imagine this problem is solved. What would you notice that's different?",
            "resource_based": "What resources and strengths can we leverage to address this?"
        }
    
    def identify_transition_opportunity(self, dialogue_content: str, current_phase: ConversationPhase) -> Optional[str]:
        """
        Identify when and how to transition from problem focus to solution focus.
        
        Args:
            dialogue_content: Recent dialogue content
            current_phase: Current conversation phase
            
        Returns:
            Transition pattern name if opportunity identified, None otherwise
        """
        content_lower = dialogue_content.lower()
        
        # Only transition from problem identification phase
        if current_phase != ConversationPhase.PROBLEM_IDENTIFICATION:
            return None
        
        # Look for transition opportunities
        problem_indicators = ["problem", "issue", "challenge", "difficulty", "concern"]
        solution_readiness_indicators = ["solve", "fix", "improve", "change", "better"]
        
        has_problem_focus = any(indicator in content_lower for indicator in problem_indicators)
        shows_solution_readiness = any(indicator in content_lower for indicator in solution_readiness_indicators)
        
        if has_problem_focus and shows_solution_readiness:
            return "direct"
        elif "scale" in content_lower or "level" in content_lower:
            return "scaling"
        elif "when" in content_lower and ("better" in content_lower or "worked" in content_lower):
            return "exception"
        elif "if" in content_lower or "imagine" in content_lower:
            return "future_focused"
        elif "resource" in content_lower or "strength" in content_lower:
            return "resource_based"
        
        # Default transition after sufficient problem exploration
        return "direct"
    
    def create_transition_prompt(self, pattern: str, problem_context: str) -> str:
        """
        Create a transition prompt using the specified pattern.
        
        Args:
            pattern: Transition pattern to use
            problem_context: Context about the problem being discussed
            
        Returns:
            Transition prompt string
        """
        base_transition = self.transition_patterns.get(pattern, self.transition_patterns["direct"])
        
        return f"""
Based on the problem context: {problem_context}

{base_transition}

Please shift focus toward constructive solutions and actionable approaches.
What specific steps or changes would move us in a positive direction?
"""


class ConstructiveQuestioning:
    """
    Framework for constructive questioning based on solution-focused principles.
    
    Provides different types of questions designed to guide conversations
    toward solutions and positive outcomes.
    """
    
    def __init__(self):
        """Initialize constructive questioning framework."""
        self.question_templates = {
            QuestionType.CLARIFYING: [
                "What specifically would need to change for this to improve?",
                "Can you help me understand what success would look like here?",
                "What's the most important aspect of this situation to address?"
            ],
            QuestionType.EXPLORING: [
                "What other ways might we approach this?",
                "What haven't we considered yet that might be relevant?",
                "What would happen if we tried a different approach?"
            ],
            QuestionType.SCALING: [
                "On a scale of 1-10, where are we now and where do we want to be?",
                "What would move us from a {current} to a {target} on this issue?",
                "If 10 is the ideal outcome, what would a 7 or 8 look like?"
            ],
            QuestionType.EXCEPTION_FINDING: [
                "When has this worked better in the past? What was different then?",
                "Are there times when this problem doesn't occur? What's happening instead?",
                "What's already working that we can build on?"
            ],
            QuestionType.FUTURE_FOCUSED: [
                "If we could solve this completely, what would that enable?",
                "What would others notice if this issue was resolved?",
                "How would success in this area impact other priorities?"
            ],
            QuestionType.RESOURCE_IDENTIFYING: [
                "What strengths and resources do we have available?",
                "Who or what could help us make progress on this?",
                "What's worked well in similar situations before?"
            ]
        }
    
    def generate_question(self, question_type: QuestionType, context: Dict[str, Any] = None) -> str:
        """
        Generate a constructive question of the specified type.
        
        Args:
            question_type: Type of question to generate
            context: Optional context for customizing the question
            
        Returns:
            Generated question string
        """
        templates = self.question_templates.get(question_type, [])
        if not templates:
            return "What would be most helpful to focus on next?"
        
        # Use the first template for now, could be enhanced with selection logic
        template = templates[0]
        
        # Simple template customization based on context
        if context and question_type == QuestionType.SCALING:
            current = context.get("current_level", "current position")
            target = context.get("target_level", "desired position")
            template = template.replace("{current}", str(current)).replace("{target}", str(target))
        
        return template
    
    def identify_question_opportunity(self, dialogue_content: str, conversation_phase: ConversationPhase) -> QuestionType:
        """
        Identify what type of constructive question would be most helpful.
        
        Args:
            dialogue_content: Recent dialogue content
            conversation_phase: Current phase of conversation
            
        Returns:
            Recommended question type
        """
        content_lower = dialogue_content.lower()
        
        # Phase-based question selection
        if conversation_phase == ConversationPhase.PROBLEM_IDENTIFICATION:
            if "unclear" in content_lower or "confusing" in content_lower:
                return QuestionType.CLARIFYING
            elif "scale" in content_lower or "level" in content_lower:
                return QuestionType.SCALING
            else:
                return QuestionType.EXPLORING
        
        elif conversation_phase == ConversationPhase.SOLUTION_EXPLORATION:
            if "worked" in content_lower or "successful" in content_lower:
                return QuestionType.EXCEPTION_FINDING
            elif "future" in content_lower or "outcome" in content_lower:
                return QuestionType.FUTURE_FOCUSED
            else:
                return QuestionType.RESOURCE_IDENTIFYING
        
        elif conversation_phase == ConversationPhase.SOLUTION_REFINEMENT:
            return QuestionType.SCALING
        
        else:  # ACTION_PLANNING
            return QuestionType.RESOURCE_IDENTIFYING


class SolutionFocusedDialogue:
    """
    Main class for conducting solution-focused dialogues.
    
    Integrates problem-solution transition and constructive questioning
    to guide conversations toward positive outcomes.
    """
    
    def __init__(self):
        """Initialize solution-focused dialogue manager."""
        self.transition_manager = ProblemSolutionTransition()
        self.questioning_framework = ConstructiveQuestioning()
        self.conversation_history: List[Dict[str, Any]] = []
    
    def conduct_solution_focused_dialogue(self,
                                        agents: List['AgentNet'],
                                        topic: str,
                                        rounds: int = 6,
                                        initial_phase: ConversationPhase = ConversationPhase.PROBLEM_IDENTIFICATION,
                                        transition_strategy: str = "adaptive",
                                        **kwargs) -> Dict[str, Any]:
        """
        Conduct a solution-focused dialogue session.
        
        Args:
            agents: List of participating agents
            topic: Dialogue topic (typically a problem or challenge)
            rounds: Number of dialogue rounds
            initial_phase: Starting conversation phase
            transition_strategy: How to manage phase transitions (adaptive, structured, agent_driven)
            
        Returns:
            Dialogue results with solution tracking and analysis
        """
        session_id = f"solution_focused_{int(time.time()*1000)}"
        current_phase = initial_phase
        phase_history = [initial_phase]
        
        # Track solution-focused elements
        problems_identified: List[ProblemStatement] = []
        solutions_proposed: List[SolutionElement] = []
        questions_asked: List[Dict[str, Any]] = []
        
        transcript = []
        
        logger.info(f"Starting solution-focused dialogue: {session_id}")
        
        for round_num in range(rounds):
            for agent in agents:
                # Create phase-appropriate prompt
                phase_prompt = self._create_phase_prompt(
                    topic, current_phase, transcript, round_num + 1
                )
                
                # Generate response
                if hasattr(agent, 'generate_reasoning_tree_enhanced'):
                    response = agent.generate_reasoning_tree_enhanced(phase_prompt)
                else:
                    response = agent.generate_reasoning_tree(phase_prompt)
                
                # Create turn record
                turn = {
                    "round": round_num + 1,
                    "agent": agent.name,
                    "content": response.get("result", {}).get("content", ""),
                    "confidence": response.get("result", {}).get("confidence", 0.5),
                    "phase": current_phase.value,
                    "timestamp": time.time()
                }
                transcript.append(turn)
                
                # Extract solution-focused elements
                self._extract_solution_elements(
                    turn, problems_identified, solutions_proposed, questions_asked, current_phase
                )
                
                # Check for phase transition opportunity
                if transition_strategy == "adaptive":
                    transition_opportunity = self.transition_manager.identify_transition_opportunity(
                        turn["content"], current_phase
                    )
                    if transition_opportunity and len(transcript) > 2:  # Allow some exploration first
                        current_phase = self._advance_phase(current_phase)
                        phase_history.append(current_phase)
                        logger.info(f"Phase transition to: {current_phase.value}")
        
        # Generate solution-focused analysis
        analysis = self._analyze_solution_focused_dialogue(
            transcript, problems_identified, solutions_proposed, questions_asked, phase_history
        )
        
        return {
            "session_id": session_id,
            "mode": "solution_focused",
            "participants": [agent.name for agent in agents],
            "topic": topic,
            "transcript": transcript,
            "phase_progression": [phase.value for phase in phase_history],
            "problems_identified": [self._problem_to_dict(p) for p in problems_identified],
            "solutions_proposed": [self._solution_to_dict(s) for s in solutions_proposed],
            "questions_generated": questions_asked,
            "analysis": analysis,
            "rounds_completed": rounds
        }
    
    def _create_phase_prompt(self,
                           topic: str,
                           phase: ConversationPhase,
                           transcript: List[Dict[str, Any]],
                           round_num: int) -> str:
        """Create a prompt appropriate for the current conversation phase."""
        phase_instructions = {
            ConversationPhase.PROBLEM_IDENTIFICATION: {
                "instruction": "Focus on clearly understanding and defining the problem or challenge.",
                "questions": "What exactly is the issue? Who is affected? What's the impact?"
            },
            ConversationPhase.SOLUTION_EXPLORATION: {
                "instruction": "Shift focus to exploring potential solutions and approaches.",
                "questions": "What are possible solutions? What has worked before? What resources are available?"
            },
            ConversationPhase.SOLUTION_REFINEMENT: {
                "instruction": "Refine and improve the most promising solutions.",
                "questions": "How can we make this solution better? What are the implementation details?"
            },
            ConversationPhase.ACTION_PLANNING: {
                "instruction": "Develop concrete action plans and next steps.",
                "questions": "What specific actions need to be taken? Who will do what? When?"
            }
        }
        
        phase_config = phase_instructions[phase]
        
        # Include recent context
        context_summary = ""
        if transcript:
            recent_turns = transcript[-2:]
            context_summary = f"\n\nRecent discussion:\n"
            for turn in recent_turns:
                context_summary += f"- {turn['agent']}: {turn['content'][:100]}...\n"
        
        # Generate a constructive question
        constructive_question = self.questioning_framework.generate_question(
            self.questioning_framework.identify_question_opportunity("", phase)
        )
        
        prompt = f"""
SOLUTION-FOCUSED DIALOGUE

Topic: {topic}
Current Phase: {phase.value.replace('_', ' ').title()}
Round: {round_num}

Phase Focus: {phase_config['instruction']}
Key Questions: {phase_config['questions']}

{context_summary}

Constructive Question to Consider: {constructive_question}

Please respond in a way that advances the {phase.value.replace('_', ' ')} process.
Focus on being constructive, solution-oriented, and collaborative.
"""
        return prompt.strip()
    
    def _extract_solution_elements(self,
                                 turn: Dict[str, Any],
                                 problems: List[ProblemStatement],
                                 solutions: List[SolutionElement],
                                 questions: List[Dict[str, Any]],
                                 phase: ConversationPhase) -> None:
        """Extract solution-focused elements from a dialogue turn."""
        content = turn["content"]
        content_lower = content.lower()
        agent = turn["agent"]
        
        # Extract problems (mainly in identification phase)
        if phase == ConversationPhase.PROBLEM_IDENTIFICATION:
            problem_indicators = ["problem", "issue", "challenge", "difficulty", "concern"]
            if any(indicator in content_lower for indicator in problem_indicators):
                problems.append(ProblemStatement(
                    description=content[:200] + "..." if len(content) > 200 else content,
                    identified_by=agent,
                    severity="moderate"  # Could be enhanced with sentiment analysis
                ))
        
        # Extract solutions (mainly in exploration and refinement phases)
        if phase in [ConversationPhase.SOLUTION_EXPLORATION, ConversationPhase.SOLUTION_REFINEMENT]:
            solution_indicators = ["solution", "approach", "way", "method", "strategy", "plan"]
            if any(indicator in content_lower for indicator in solution_indicators):
                solutions.append(SolutionElement(
                    description=content[:200] + "..." if len(content) > 200 else content,
                    proposed_by=agent,
                    confidence=turn["confidence"],
                    feasibility="moderate",  # Could be enhanced with analysis
                    phase_introduced=phase
                ))
        
        # Extract questions
        if "?" in content:
            question_sentences = [s.strip() for s in content.split('.') if '?' in s]
            for question in question_sentences:
                questions.append({
                    "question": question,
                    "asked_by": agent,
                    "round": turn["round"],
                    "phase": phase.value
                })
    
    def _advance_phase(self, current_phase: ConversationPhase) -> ConversationPhase:
        """Advance to the next conversation phase."""
        phase_sequence = [
            ConversationPhase.PROBLEM_IDENTIFICATION,
            ConversationPhase.SOLUTION_EXPLORATION,
            ConversationPhase.SOLUTION_REFINEMENT,
            ConversationPhase.ACTION_PLANNING
        ]
        
        try:
            current_index = phase_sequence.index(current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
            else:
                return current_phase  # Stay in final phase
        except ValueError:
            return ConversationPhase.SOLUTION_EXPLORATION  # Default fallback
    
    def _analyze_solution_focused_dialogue(self,
                                         transcript: List[Dict[str, Any]],
                                         problems: List[ProblemStatement],
                                         solutions: List[SolutionElement],
                                         questions: List[Dict[str, Any]],
                                         phase_history: List[ConversationPhase]) -> Dict[str, Any]:
        """Analyze the solution-focused dialogue for effectiveness."""
        if not transcript:
            return {"error": "No dialogue content to analyze"}
        
        # Calculate solution focus ratio
        total_turns = len(transcript)
        solution_focused_turns = sum(1 for turn in transcript 
                                   if any(word in turn["content"].lower() 
                                         for word in ["solution", "approach", "way", "plan", "strategy"]))
        solution_focus_ratio = solution_focused_turns / total_turns if total_turns > 0 else 0
        
        # Analyze phase progression
        phases_covered = len(set(phase_history))
        phase_transitions = len(phase_history) - 1
        
        # Calculate constructive question ratio
        questions_asked = len(questions)
        question_ratio = questions_asked / total_turns if total_turns > 0 else 0
        
        # Analyze problem-to-solution transition
        problem_turns = sum(1 for turn in transcript if turn["phase"] == ConversationPhase.PROBLEM_IDENTIFICATION.value)
        solution_turns = sum(1 for turn in transcript if turn["phase"] in [
            ConversationPhase.SOLUTION_EXPLORATION.value,
            ConversationPhase.SOLUTION_REFINEMENT.value,
            ConversationPhase.ACTION_PLANNING.value
        ])
        
        transition_effectiveness = "high" if solution_turns > problem_turns else "moderate" if solution_turns == problem_turns else "low"
        
        # Calculate overall effectiveness
        effectiveness_score = (
            solution_focus_ratio * 40 +
            min(phases_covered / 4, 1.0) * 30 +
            min(question_ratio * 2, 1.0) * 20 +
            (1.0 if transition_effectiveness == "high" else 0.5 if transition_effectiveness == "moderate" else 0.0) * 10
        )
        
        overall_effectiveness = "high" if effectiveness_score > 70 else "moderate" if effectiveness_score > 40 else "needs_improvement"
        
        return {
            "solution_focus_ratio": solution_focus_ratio,
            "phases_covered": phases_covered,
            "phase_transitions": phase_transitions,
            "problems_identified": len(problems),
            "solutions_proposed": len(solutions),
            "questions_asked": questions_asked,
            "question_ratio": question_ratio,
            "transition_effectiveness": transition_effectiveness,
            "overall_effectiveness": overall_effectiveness,
            "effectiveness_score": effectiveness_score,
            "constructive_elements": {
                "problem_statements": len(problems),
                "solution_elements": len(solutions),
                "constructive_questions": questions_asked
            }
        }
    
    def _problem_to_dict(self, problem: ProblemStatement) -> Dict[str, Any]:
        """Convert ProblemStatement to dictionary."""
        return {
            "description": problem.description,
            "identified_by": problem.identified_by,
            "severity": problem.severity,
            "stakeholders": problem.stakeholders or [],
            "context": problem.context or {}
        }
    
    def _solution_to_dict(self, solution: SolutionElement) -> Dict[str, Any]:
        """Convert SolutionElement to dictionary."""
        return {
            "description": solution.description,
            "proposed_by": solution.proposed_by,
            "confidence": solution.confidence,
            "feasibility": solution.feasibility,
            "phase_introduced": solution.phase_introduced.value,
            "supporting_agents": solution.supporting_agents or [],
            "refinements": solution.refinements or []
        }