"""
Enhanced, asynchronous Debate strategy implementation.

This strategy orchestrates a structured, multi-round debate between two agents,
with a third agent acting as a judge to determine the winner based on the
strength of the arguments presented.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class DebateStrategy(BaseStrategy):
    """
    An advanced strategy that facilitates a formal debate between two agents,
    concluding with a verdict from a third, neutral agent.

    The debate follows a structured, multi-round format:
    1. Opening Statements: Two agents take opposing stances and prepare their arguments.
    2. Rebuttals: Each agent critiques the other's opening statement.
    3. Adjudication: A judge reviews the full transcript and declares a winner.
    """

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = ProblemSolvingStyle.EVALUATOR,
        technique: Optional[ProblemTechnique] = None,
        **config: Any,
    ):
        """
        Initialize the debate strategy.

        Args:
            style: The default style is 'evaluator' to encourage critical analysis.
            technique: Optional problem-solving technique.
            **config: Configuration for the strategy.
        """
        super().__init__(Mode.DEBATE, style, technique, **config)

    async def _execute(
        self,
        agent: "AgentNet",
        task: str,
        context: Dict[str, Any],
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the multi-round debate and adjudication process.
        """
        if not agents or len(agents) != 2:
            raise ValueError("DebateStrategy requires exactly two participating agents (the debaters). The primary 'agent' will act as the judge.")
        
        judge = agent
        debater_for, debater_against = agents[0], agents[1]
        self.logger.info(f"Debate starting. Judge: {judge.name}, For: {debater_for.name}, Against: {debater_against.name}.")

        # --- Round 1: Concurrent Opening Statements ---
        self.logger.info("Round 1: Preparing opening statements.")
        prompt_for = self._create_prompt(task, "for", "opening_statement")
        prompt_against = self._create_prompt(task, "against", "opening_statement")

        opening_tasks = {
            "for": debater_for.async_generate_reasoning_tree(
                task=prompt_for, confidence_threshold=0.75, metadata={"debate_role": "for", "round": 1}
            ),
            "against": debater_against.async_generate_reasoning_tree(
                task=prompt_against, confidence_threshold=0.75, metadata={"debate_role": "against", "round": 1}
            ),
        }
        
        opening_results = await asyncio.gather(*opening_tasks.values(), return_exceptions=True)
        opening_statement_for = self._extract_content(opening_results[0], "Opening statement for 'for' side")
        opening_statement_against = self._extract_content(opening_results[1], "Opening statement for 'against' side")

        # --- Round 2: Concurrent Rebuttals ---
        self.logger.info("Round 2: Preparing rebuttals.")
        prompt_rebuttal_for = self._create_prompt(task, "for", "rebuttal", opponent_argument=opening_statement_against)
        prompt_rebuttal_against = self._create_prompt(task, "against", "rebuttal", opponent_argument=opening_statement_for)

        rebuttal_tasks = {
            "for": debater_for.async_generate_reasoning_tree(
                task=prompt_rebuttal_for, confidence_threshold=0.75, metadata={"debate_role": "for", "round": 2}
            ),
            "against": debater_against.async_generate_reasoning_tree(
                task=prompt_rebuttal_against, confidence_threshold=0.75, metadata={"debate_role": "against", "round": 2}
            ),
        }

        rebuttal_results = await asyncio.gather(*rebuttal_tasks.values(), return_exceptions=True)
        rebuttal_for = self._extract_content(rebuttal_results[0], "Rebuttal for 'for' side")
        rebuttal_against = self._extract_content(rebuttal_results[1], "Rebuttal for 'against' side")

        # --- Final Step: Adjudication ---
        self.logger.info(f"Final Step: Judge '{judge.name}' is adjudicating the debate.")
        debate_transcript = (
            f"**Debate Topic:** {task}\n\n"
            f"--- OPENING STATEMENT (FOR - Agent {debater_for.name}) ---\n{opening_statement_for}\n\n"
            f"--- OPENING STATEMENT (AGAINST - Agent {debater_against.name}) ---\n{opening_statement_against}\n\n"
            f"--- REBUTTAL (FOR - Agent {debater_for.name}) ---\n{rebuttal_for}\n\n"
            f"--- REBUTTAL (AGAINST - Agent {debater_against.name}) ---\n{rebuttal_against}\n\n"
        )

        adjudication_prompt = (
            f"You are the judge of the following debate. Your task is to determine a winner based on the "
            f"strength, logic, and evidence presented in the arguments.\n\n"
            f"Instructions:\n"
            f"1. **Review the entire transcript** objectively.\n"
            f"2. **Declare a winner:** State clearly whether the 'FOR' or 'AGAINST' side won.\n"
            f"3. **Provide a detailed justification:** Explain your decision by referencing specific points, "
            f"identifying logical fallacies, and comparing the quality of the rebuttals.\n\n"
            f"--- DEBATE TRANSCRIPT ---\n{debate_transcript}"
        )

        verdict_result = await judge.async_generate_reasoning_tree(
            task=adjudication_prompt,
            confidence_threshold=0.8, # High confidence for a final judgment
            metadata={"debate_role": "judge", "round": "final"},
        )

        return {
            "verdict": verdict_result.get("result", {}),
            "winner_declared": "for" if "for" in verdict_result.get("result", {}).get("content", "").lower()[:20] else "against",
            "debate_transcript": {
                "topic": task,
                "for_agent": debater_for.name,
                "against_agent": debater_against.name,
                "opening_statement_for": opening_statement_for,
                "opening_statement_against": opening_statement_against,
                "rebuttal_for": rebuttal_for,
                "rebuttal_against": rebuttal_against,
            },
        }

    def _create_prompt(self, task: str, stance: str, round_type: str, opponent_argument: Optional[str] = None) -> str:
        """Helper to generate prompts for different stages of the debate."""
        if round_type == "opening_statement":
            return (
                f"**Debate Topic:** {task}\n\n"
                f"**Your Stance:** You are arguing **FOR** the motion.\n\n" if stance == "for" else
                f"**Debate Topic:** {task}\n\n"
                f"**Your Stance:** You are arguing **AGAINST** the motion.\n\n"
            ) + (
                f"Please prepare a strong, well-reasoned opening statement. "
                f"Structure your argument with a clear position, supporting points, and evidence."
            )
        
        elif round_type == "rebuttal":
            return (
                f"**Debate Topic:** {task}\n\n"
                f"**Your Stance:** You are arguing **{'FOR' if stance == 'for' else 'AGAINST'}** the motion.\n\n"
                f"**Your Opponent's Argument:**\n---\n{opponent_argument}\n---\n\n"
                f"**Your Task:** Prepare a rebuttal. Directly address and critique your opponent's points. "
                f"Identify weaknesses in their argument, provide counter-evidence, and reaffirm your own position."
            )
        return ""

    def _extract_content(self, result: Dict[str, Any] | Exception, default_text: str) -> str:
        """Safely extracts content from a result or returns a default if it failed."""
        if isinstance(result, Exception):
            self.logger.error(f"A debate participant failed: {result}")
            return f"({default_text} could not be generated due to an error.)"
        return result.get("result", {}).get("content", f"({default_text} was not provided.)")
