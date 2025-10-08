"""
Enhanced, asynchronous Dialogue strategy implementation.

This strategy facilitates a structured, multi-turn conversational exchange
between two agents, guided by a specific goal and culminating in a
summary of the key insights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class DialogueStrategy(BaseStrategy):
    """
    An advanced strategy that orchestrates a multi-turn dialogue between two agents.

    The process involves:
    1. A turn-based conversation loop for a configured number of turns.
    2. Maintaining and passing a conversation history for context.
    3. A final summary step where the primary agent distills the dialogue's outcome.
    """

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = ProblemSolvingStyle.CLARIFIER,
        technique: Optional[ProblemTechnique] = None,
        **config: Any,
    ):
        """
        Initialize the dialogue strategy.

        Args:
            style: The default style is 'clarifier' to encourage thoughtful questions.
            technique: Optional problem-solving technique.
            **config: Configuration for the strategy, e.g.,
                - max_turns (int): The total number of conversational turns. Defaults to 4.
                - dialogue_goal (str): The objective of the conversation.
        """
        super().__init__(Mode.DIALOGUE, style, technique, **config)

    async def _execute(
        self,
        agent: "AgentNet",
        task: str,
        context: Dict[str, Any],
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the multi-turn dialogue and final summarization.
        """
        if not agents or len(agents) != 1:
            raise ValueError("DialogueStrategy requires exactly one other agent in the 'agents' list to converse with the primary 'agent'.")

        participant1 = agent
        participant2 = agents[0]
        participants = [participant1, participant2]
        
        max_turns = self.config.get("max_turns", 4)
        dialogue_goal = self.config.get("dialogue_goal", "to collaboratively explore and understand the topic in depth")

        self.logger.info(
            f"Starting {max_turns}-turn dialogue between '{participant1.name}' and "
            f"'{participant2.name}'. Goal: {dialogue_goal}"
        )

        conversation_history = []
        
        # --- Conversational Loop ---
        for i in range(max_turns):
            current_speaker_idx = i % 2
            current_speaker = participants[current_speaker_idx]
            
            self.logger.debug(f"Turn {i+1}: Speaker is '{current_speaker.name}'.")

            prompt = self._create_turn_prompt(
                task, dialogue_goal, conversation_history, current_speaker.name
            )

            response_result = await current_speaker.async_generate_reasoning_tree(
                task=prompt,
                confidence_threshold=0.65,
                metadata={"dialogue_turn": i + 1, "speaker": current_speaker.name},
            )
            
            response_content = self._extract_content(response_result, f"Turn {i+1} response")
            
            conversation_history.append({
                "speaker": current_speaker.name,
                "turn": i + 1,
                "message": response_content,
            })

        # --- Final Summary Step ---
        self.logger.info(f"Dialogue complete. Generating final summary with '{participant1.name}'.")
        
        summary_prompt = self._create_summary_prompt(task, dialogue_goal, conversation_history)
        
        summary_result = await participant1.async_generate_reasoning_tree(
            task=summary_prompt,
            confidence_threshold=0.75, # Higher confidence for a structured summary
            metadata={"dialogue_phase": "summary"},
        )

        return {
            "final_summary": summary_result.get("result", {}),
            "conversation_transcript": conversation_history,
        }

    def _create_turn_prompt(
        self, task: str, goal: str, history: List[Dict], speaker_name: str
    ) -> str:
        """Generates the prompt for the current speaker in the dialogue."""
        
        history_str = "\n".join(
            f"[{item['speaker']}]: {item['message']}" for item in history
        )

        if not history: # First turn
            return (
                f"**Topic:** {task}\n"
                f"**Dialogue Goal:** {goal}\n\n"
                f"You are '{speaker_name}'. Please begin the conversation with your opening thoughts or a clarifying question."
            )
        else:
            return (
                f"**Topic:** {task}\n"
                f"**Dialogue Goal:** {goal}\n\n"
                f"**Conversation History:**\n{history_str}\n\n"
                f"You are '{speaker_name}'. Based on the history, what is your response? "
                f"Continue the dialogue towards the stated goal."
            )

    def _create_summary_prompt(self, task: str, goal: str, history: List[Dict]) -> str:
        """Generates the prompt for the final summary."""
        
        history_str = "\n".join(
            f"[{item['speaker']}]: {item['message']}" for item in history
        )

        return (
            f"**Topic:** {task}\n"
            f"**Dialogue Goal:** {goal}\n\n"
            f"The following conversation took place. Your task is to synthesize it into a final summary.\n\n"
            f"Instructions:\n"
            f"1. **Identify Key Insights:** What were the most important points or discoveries?\n"
            f"2. **Note Areas of Agreement/Disagreement:** What did the participants agree or disagree on?\n"
            f"3. **Conclude on the Goal:** Was the dialogue goal achieved? What is the final conclusion?\n\n"
            f"**Full Conversation Transcript:**\n{history_str}"
        )

    def _extract_content(self, result: Dict[str, Any] | Exception, default_text: str) -> str:
        """Safely extracts content from a result or returns a default if it failed."""
        if isinstance(result, Exception):
            self.logger.error(f"A dialogue participant failed: {result}")
            return f"({default_text} could not be generated due to an error.)"
        return result.get("result", {}).get("content", f"({default_text} was not provided.)")
