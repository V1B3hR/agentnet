"""
Base strategy class for problem-solving modes.

Provides common functionality and interface for all mode strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from ...metrics.flow import calculate_flow_metrics

if TYPE_CHECKING:
    from ...core.agent import AgentNet

logger = logging.getLogger("agentnet.orchestrator.modes.base")


class BaseStrategy(ABC):
    """Base class for all problem-solving mode strategies."""

    def __init__(
        self,
        mode: Mode,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None,
    ):
        """
        Initialize base strategy.

        Args:
            mode: The problem-solving mode this strategy implements
            style: Optional problem-solving style
            technique: Optional problem-solving technique
        """
        self.mode = mode
        self.style = style
        self.technique = technique
        self.logger = logging.getLogger(f"agentnet.orchestrator.modes.{mode.value}")

    @abstractmethod
    def execute(
        self,
        agent: "AgentNet",
        task: str,
        agents: Optional[List["AgentNet"]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the strategy for the given task.

        Args:
            agent: Primary agent for execution
            task: Task description
            agents: Optional list of agents for multi-agent modes
            **kwargs: Additional strategy-specific parameters

        Returns:
            Result dictionary with strategy output and metadata
        """
        pass

    def _prepare_metadata(self, **kwargs) -> Dict[str, Any]:
        """Prepare metadata dictionary with mode, style, and technique information."""
        metadata = kwargs.get("metadata", {})

        # Add mode information
        metadata["mode"] = self.mode.value
        if self.style:
            metadata["problem_solving_style"] = self.style.value
        if self.technique:
            metadata["problem_technique"] = self.technique.value

        return metadata

    def _calculate_and_attach_flow_metrics(
        self, result: Dict[str, Any], runtime_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate flow metrics and attach them to the result."""
        try:
            # Extract difficulty from AutoConfig if available
            difficulty = None
            autoconfig_data = result.get("autoconfig")
            if autoconfig_data:
                difficulty_str = autoconfig_data.get("difficulty")
                if difficulty_str:
                    from ...core.autoconfig import TaskDifficulty

                    try:
                        difficulty = TaskDifficulty(difficulty_str)
                    except ValueError:
                        pass

            flow_metrics = calculate_flow_metrics(
                reasoning_tree=result,
                runtime_seconds=runtime_seconds,
                technique=self.technique,
                difficulty=difficulty,
            )

            result["flow_metrics"] = {
                "current": flow_metrics.current,
                "voltage": flow_metrics.voltage,
                "resistance": flow_metrics.resistance,
                "power": flow_metrics.power,
                "runtime_seconds": flow_metrics.runtime_seconds,
                "tokens_output": flow_metrics.tokens_output,
            }

            self.logger.debug(
                f"Flow metrics calculated: I={flow_metrics.current:.2f}, "
                f"V={flow_metrics.voltage:.2f}, R={flow_metrics.resistance:.2f}, "
                f"P={flow_metrics.power:.2f}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to calculate flow metrics: {e}")
            # Don't fail the entire operation if metrics calculation fails

        return result

    def _log_strategy_execution(self, task: str, **kwargs):
        """Log strategy execution for observability."""
        self.logger.info(
            f"Executing {self.mode.value} strategy for task: {task[:100]}..."
            + (f" (style: {self.style.value})" if self.style else "")
            + (f" (technique: {self.technique.value})" if self.technique else "")
        )
