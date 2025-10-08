"""
Enhanced, asynchronous Brainstorm strategy implementation.

Leverages concurrent, multi-agent idea generation and a final synthesis step
to produce structured, creative outputs. Supports multiple configurable
brainstorming techniques like role-playing and SCAMPER.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class BrainstormStrategy(BaseStrategy):
    """
    An advanced strategy for brainstorming that orchestrates multiple agents
    to generate and then synthesize ideas concurrently.
    """

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = ProblemSolvingStyle.IDEATOR,
        technique: Optional[ProblemTechnique] = None,
        **config: Any,
    ):
        """
        Initialize the brainstorm strategy.

        Args:
            style: The default style is 'ideator' to encourage creativity.
            technique: Optional problem-solving technique.
            **config: Configuration for the strategy, e.g.,
                - technique (str): 'divergent', 'role_playing', 'scamper'.
                - ideas_per_agent (int): How many ideas each agent should aim for.
                - roles (List[str]): A list of personas for the 'role_playing' technique.
        """
        super().__init__(Mode.BRAINSTORM, style, technique, **config)

    def _create_generation_prompt(
        self, task: str, technique: str, role: Optional[str] = None
    ) -> str:
        """Creates a tailored prompt based on the chosen brainstorming technique."""
        ideas_count = self.config.get("ideas_per_agent", 3)
        header = f"Brainstorming Task: {task}\nGenerate {ideas_count} diverse and novel ideas."

        if technique == "role_playing" and role:
            return (
                f"{header}\n\n**Your Persona: The {role.title()}**\n"
                f"From this perspective, what are your ideas? Focus on your persona's "
                f"unique priorities, concerns, and mindset. Be creative and bold."
            )
        elif technique == "scamper":
            return (
                f"{header}\n\nUse the SCAMPER method to guide your thinking:\n"
                f"- **Substitute:** What can be replaced?\n"
                f"- **Combine:** What ideas or components can be merged?\n"
                f"- **Adapt:** What can be adapted from another context?\n"

                f"- **Modify:** How can you magnify, minimize, or change an attribute?\n"
                f"- **Put to another use:** What are alternative uses?\n"
                f"- **Eliminate:** What can be removed or simplified?\n"
                f"- **Reverse:** How can you rearrange or reverse the process?"
            )
        # Default to divergent thinking
        return (
            f"{header}\n\nFocus on:\n"
            f"- Unconventional and creative approaches.\n"
            f"- A wide variety of perspectives.\n"
            f"- Quantity over initial quality; do not self-censor.\n"
            f"Think freely and explore all possibilities."
        )

    async def _execute(
        self,
        agent: "AgentNet",
        task: str,
        context: Dict[str, Any],
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the two-phase brainstorm strategy: concurrent generation followed by synthesis.
        """
        participating_agents = agents or [agent]
        technique = self.config.get("technique", "divergent")
        roles = self.config.get("roles", ["Skeptic", "Visionary", "Pragmatist", "Customer Advocate"])

        # --- Phase 1: Concurrent Idea Generation ---
        self.logger.info(
            f"Starting idea generation with {len(participating_agents)} agents "
            f"using '{technique}' technique."
        )
        generation_tasks = []
        for i, participant in enumerate(participating_agents):
            role = roles[i % len(roles)] if technique == "role_playing" else None
            prompt = self._create_generation_prompt(task, technique, role)
            
            # Use lower confidence for more creative, less filtered ideas
            generation_tasks.append(
                participant.async_generate_reasoning_tree(
                    task=prompt,
                    confidence_threshold=0.55,
                    metadata={"brainstorm_phase": "generation", "agent_role": role},
                )
            )

        # Run all generation tasks in parallel
        generation_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        # --- Process Raw Ideas ---
        raw_ideas = []
        for i, res in enumerate(generation_results):
            agent_name = participating_agents[i].name
            if isinstance(res, Exception):
                self.logger.warning(f"Agent '{agent_name}' failed during idea generation: {res}")
                continue
            
            idea_content = res.get("result", {}).get("content")
            if idea_content:
                raw_ideas.append(f"--- Idea from Agent '{agent_name}' ---\n{idea_content}\n")

        if not raw_ideas:
            raise RuntimeError("No ideas were generated by any participating agents.")

        # --- Phase 2: Synthesis ---
        self.logger.info(f"Synthesizing {len(raw_ideas)} raw ideas using primary agent '{agent.name}'.")
        synthesis_prompt = (
            f"Task: {task}\n\n"
            f"The following raw ideas were generated in a brainstorming session. "
            f"Your task is to synthesize them into a structured and actionable summary.\n\n"
            f"Instructions:\n"
            f"1. **Review and Group:** Read all ideas and group them into logical categories or themes.\n"
            f"2. **De-duplicate and Refine:** Merge similar ideas and rephrase them for clarity.\n"
            f"3. **Rank:** Identify the top 3-5 most promising or innovative ideas.\n"
            f"4. **Summarize:** Provide a brief summary for each ranked idea, explaining its potential.\n\n"
            f"Present the final output clearly.\n\n"
            f"--- RAW IDEAS ---\n{''.join(raw_ideas)}"
        )

        synthesis_result = await agent.async_generate_reasoning_tree(
            task=synthesis_prompt,
            confidence_threshold=0.7, # Higher confidence for structured output
            metadata={"brainstorm_phase": "synthesis"},
        )

        return {
            "synthesized_output": synthesis_result.get("result", {}),
            "raw_ideas": [res.get("result", {}) for res in generation_results if not isinstance(res, Exception)],
            "technique_used": technique,
        }
