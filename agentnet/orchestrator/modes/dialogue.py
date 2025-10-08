"""
Dialogue strategy implementation.

Focuses on conversational exploration and interactive problem solving with AgentNet.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class DialogueStrategy(BaseStrategy):
    """Strategy for dialogue mode - focuses on conversational exploration and interaction."""

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None,
    ):
        """Initialize dialogue strategy."""
        super().__init__(Mode.DIALOGUE, style, technique)

    def execute(
        self,
        agent: "AgentNet",
        task: str,
        agents: Optional[List["AgentNet"]] = None,
        max_depth: int = 3,
        confidence_threshold: float = 0.65,  # Moderate threshold for dialogue
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute dialogue strategy.

        Args:
            agent: Primary agent for execution
            task: Task description
            agents: Optional list of agents for multi-agent dialogue
            max_depth: Maximum reasoning depth
            confidence_threshold: Confidence threshold for dialogue
            **kwargs: Additional parameters

        Returns:
            Result dictionary with dialogue output
        """
        start_time = time.time()
        self._log_strategy_execution(task, **kwargs)

        # Prepare metadata with dialogue-specific tags
        metadata = self._prepare_metadata(**kwargs)
        metadata["dialogue_focus"] = "conversational_exploration"
        metadata["interactive_mode"] = True

        # Remove metadata from kwargs to avoid duplicate parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "metadata"}

        # Modify task prompt for dialogue context
        dialogue_task = f"""Engage in thoughtful dialogue about: {task}

Approach this conversationally:
- Ask probing questions to deepen understanding
- Build on ideas through natural conversation flow
- Explore different angles through dialogue
- Listen actively and respond thoughtfully
- Seek clarification and elaboration
- Connect ideas through interactive exploration

Engage as if in a meaningful conversation to explore this topic."""

        # Execute reasoning with dialogue-optimized parameters
        result = agent.generate_reasoning_tree(
            task=dialogue_task,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            metadata=metadata,
            **filtered_kwargs,
        )

        # Add dialogue-specific metadata to result
        result["strategy"] = {
            "mode": self.mode.value,
            "style": self.style.value if self.style else None,
            "technique": self.technique.value if self.technique else None,
            "focus": "conversational_exploration",
            "execution_time": time.time() - start_time,
        }

        # Calculate and attach flow metrics
        result = self._calculate_and_attach_flow_metrics(
            result, time.time() - start_time
        )

        return result
