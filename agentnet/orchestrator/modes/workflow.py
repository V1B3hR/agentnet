"""
Workflow strategy implementation.

Focuses on structured process execution and step-by-step problem solving with AgentNet.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class WorkflowStrategy(BaseStrategy):
    """Strategy for workflow mode - focuses on structured process execution."""

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None,
    ):
        """Initialize workflow strategy."""
        super().__init__(Mode.WORKFLOW, style, technique)

    def execute(
        self,
        agent: "AgentNet",
        task: str,
        agents: Optional[List["AgentNet"]] = None,
        max_depth: int = 4,  # Deeper for step-by-step analysis
        confidence_threshold: float = 0.8,  # Higher threshold for workflow steps
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute workflow strategy.

        Args:
            agent: Primary agent for execution
            task: Task description
            agents: Optional list of agents for workflow execution
            max_depth: Maximum reasoning depth
            confidence_threshold: Confidence threshold for workflow steps
            **kwargs: Additional parameters

        Returns:
            Result dictionary with workflow output
        """
        start_time = time.time()
        self._log_strategy_execution(task, **kwargs)

        # Prepare metadata with workflow-specific tags
        metadata = self._prepare_metadata(**kwargs)
        metadata["workflow_focus"] = "process_execution"
        metadata["structured_mode"] = True

        # Remove metadata from kwargs to avoid duplicate parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "metadata"}

        # Modify task prompt for workflow context
        workflow_task = f"""Execute a structured workflow for: {task}

Approach this systematically:
1. Break down into clear, manageable steps
2. Define prerequisites and dependencies
3. Identify required resources and capabilities
4. Plan execution sequence and timing
5. Consider quality checkpoints and validation
6. Plan for contingencies and error handling

Provide a detailed, actionable workflow with clear milestones."""

        # Execute reasoning with workflow-optimized parameters
        result = agent.generate_reasoning_tree(
            task=workflow_task,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            metadata=metadata,
            **filtered_kwargs,
        )

        # Add workflow-specific metadata to result
        result["strategy"] = {
            "mode": self.mode.value,
            "style": self.style.value if self.style else None,
            "technique": self.technique.value if self.technique else None,
            "focus": "process_execution",
            "execution_time": time.time() - start_time,
        }

        # Calculate and attach flow metrics
        result = self._calculate_and_attach_flow_metrics(
            result, time.time() - start_time
        )

        return result
