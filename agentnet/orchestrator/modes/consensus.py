"""
Enhanced, asynchronous Consensus strategy implementation.

This strategy facilitates agreement among multiple agents through an iterative,
multi-round process of proposing, sharing, and refining solutions until a
shared understanding is reached.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class ConsensusStrategy(BaseStrategy):
    """
    An advanced strategy for achieving consensus among multiple agents.

    It operates in an iterative loop:
    1. Propose: All agents generate initial solutions concurrently.
    2. Share & Refine: All proposals are shared, and agents create new,
       refined proposals that incorporate feedback and common ground.
    3. Synthesize: A final consensus is extracted from the last round's proposals.
    """

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = ProblemSolvingStyle.SYNTHESIZER,
        technique: Optional[ProblemTechnique] = None,
        **config: Any,
    ):
        """
        Initialize the consensus strategy.

        Args:
            style: The default style is 'synthesizer' to encourage finding common ground.
            technique: Optional problem-solving technique.
            **config: Configuration for the strategy, e.g.,
                - max_rounds (int): The number of refinement rounds. Defaults to 2.
        """
        super().__init__(Mode.CONSENSUS, style, technique, **config)

    async def _execute(
        self,
        agent: "AgentNet",
        task: str,
        context: Dict[str, Any],
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the multi-round consensus-building process.
        """
        if not agents or len(agents) < 2:
            raise ValueError("ConsensusStrategy requires at least two agents to participate.")

        max_rounds = self.config.get("max_rounds", 2)
        proposals: Dict[str, str] = {}
        round_history = []

        # --- Round 1: Initial Proposal Generation ---
        self.logger.info(f"Starting Round 1: Initial proposal generation with {len(agents)} agents.")
        initial_prompt = (
            f"Task: {task}\n\n"
            f"Please provide your initial proposed solution or plan. "
            f"Clearly state your key assumptions and reasoning. This is the first step "
            f"in a collaborative process to reach a consensus."
        )

        proposal_tasks = [
            p.async_generate_reasoning_tree(
                task=initial_prompt,
                confidence_threshold=0.7,
                metadata={"consensus_round": 1, "phase": "propose"},
            )
            for p in agents
        ]
        initial_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)

        current_proposals = self._process_round_results(1, agents, initial_results)
        if not current_proposals:
            raise RuntimeError("No initial proposals were generated in Round 1.")
        round_history.append({"round": 1, "proposals": current_proposals})

        # --- Iterative Refinement Rounds ---
        for i in range(2, max_rounds + 1):
            self.logger.info(f"Starting Round {i}: Sharing and refining {len(current_proposals)} proposals.")
            
            # Create a shared context of all previous proposals
            shared_context = "\n\n".join(
                f"--- Proposal from Agent '{name}' ---\n{content}"
                for name, content in current_proposals.items()
            )
            
            refinement_prompt = (
                f"Task: {task}\n\n"
                f"Below are the proposals from the previous round. Your goal is to move towards consensus.\n\n"
                f"Instructions:\n"
                f"1. **Review all proposals:** Identify points of agreement and disagreement.\n"
                f"2. **Create a single, improved proposal:** Synthesize the best ideas and address conflicts.\n"
                f"3. **Justify your changes:** Briefly explain why your new proposal is a better compromise.\n\n"
                f"{shared_context}"
            )

            refinement_tasks = [
                p.async_generate_reasoning_tree(
                    task=refinement_prompt,
                    confidence_threshold=0.75, # Slightly higher confidence for refinement
                    metadata={"consensus_round": i, "phase": "refine"},
                )
                for p in agents
            ]
            refinement_results = await asyncio.gather(*refinement_tasks, return_exceptions=True)
            
            current_proposals = self._process_round_results(i, agents, refinement_results)
            if not current_proposals:
                self.logger.warning(f"No refined proposals were generated in Round {i}. Ending process.")
                break
            round_history.append({"round": i, "proposals": current_proposals})

        # --- Final Synthesis Step ---
        self.logger.info(f"Synthesizing the final consensus from {len(current_proposals)} proposals.")
        final_proposals_text = "\n\n".join(
            f"--- Final Refined Proposal from Agent '{name}' ---\n{content}"
            for name, content in current_proposals.items()
        )
        
        synthesis_prompt = (
            f"Task: {task}\n\n"
            f"The following refined proposals have been generated after {max_rounds} rounds of collaboration. "
            f"Your final task is to determine the consensus.\n\n"
            f"Instructions:\n"
            f"1. **Identify the core consensus:** State the final agreed-upon solution or plan clearly.\n"
            f"2. **List key supporting points:** Summarize the main reasons this consensus was reached.\n"
            f"3. **Note any minor remaining disagreements:** If any small points of divergence remain, list them briefly.\n\n"
            f"{final_proposals_text}"
        )

        synthesis_result = await agent.async_generate_reasoning_tree(
            task=synthesis_prompt,
            confidence_threshold=0.8,
            metadata={"consensus_round": "final", "phase": "synthesize"},
        )

        return {
            "final_consensus": synthesis_result.get("result", {}),
            "round_history": round_history,
            "total_rounds": len(round_history),
        }

    def _process_round_results(
        self, round_num: int, agents: List["AgentNet"], results: List[Dict[str, Any] | Exception]
    ) -> Dict[str, str]:
        """Helper to extract successful proposals from a round's results."""
        proposals = {}
        for i, res in enumerate(results):
            agent_name = agents[i].name
            if isinstance(res, Exception):
                self.logger.warning(f"Agent '{agent_name}' failed in round {round_num}: {res}")
                continue
            
            content = res.get("result", {}).get("content")
            if content:
                proposals[agent_name] = content
        return proposals
