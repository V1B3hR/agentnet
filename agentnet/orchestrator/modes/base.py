"""
Enhanced, asynchronous base strategy class for problem-solving modes.

Provides a robust, observable, and extensible foundation for all mode strategies,
incorporating async execution, lifecycle hooks, state management, and integrated
metrics.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Assuming a structure where these modules exist.
# If they are in different locations, the import paths may need adjustment.
from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from ...metrics.flow import calculate_flow_metrics
from ...observability.metrics import get_global_metrics

if TYPE_CHECKING:
    from ...core.agent import AgentNet
    from ...core.autoconfig import TaskDifficulty

logger = logging.getLogger("agentnet.orchestrator.modes.base")


class BaseStrategy(ABC):
    """
    Abstract base class for all asynchronous problem-solving mode strategies.

    This class provides a template method pattern via the `run` method, which
    handles timing, error handling, metrics, logging, and lifecycle hooks,
    while subclasses must implement the core logic in the `_execute` method.
    """

    def __init__(
        self,
        mode: Mode,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None,
        **config: Any,
    ):
        """
        Initialize the base strategy with its configuration.

        Args:
            mode: The problem-solving mode this strategy implements.
            style: Optional problem-solving style to influence behavior.
            technique: Optional problem-solving technique to apply.
            **config: Additional key-value configuration for the strategy.
        """
        self.mode = mode
        self.style = style
        self.technique = technique
        self.config = config
        self.state: Dict[str, Any] = {}  # For stateful strategies
        self.logger = logging.getLogger(f"agentnet.orchestrator.modes.{mode.value}")
        self.metrics = get_global_metrics()

    @abstractmethod
    async def _execute(
        self,
        agent: "AgentNet",
        task: str,
        context: Dict[str, Any],
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Core execution logic for the strategy (must be implemented by subclasses).

        Args:
            agent: The primary agent for execution.
            task: The description of the task to be performed.
            context: A dictionary for passing shared, session-level information.
            agents: An optional list of agents for multi-agent modes.

        Returns:
            A dictionary containing the primary result data of the strategy.
        """
        pass

    async def run(
        self,
        agent: "AgentNet",
        task: str,
        context: Optional[Dict[str, Any]] = None,
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Public-facing method to run the strategy with full lifecycle and observability.

        This method orchestrates the execution, handling logging, timing, error
        handling, metrics, and hooks.

        Args:
            agent: The primary agent for execution.
            task: The description of the task.
            context: Shared, session-level information.
            agents: Optional list of agents for multi-agent modes.

        Returns:
            A standardized result dictionary, indicating success or failure.
        """
        start_time = time.monotonic()
        context = context or {}
        self._log_strategy_execution(task)

        try:
            # --- Pre-Execution Hook ---
            await self.pre_run(agent=agent, task=task, context=context)

            # --- Core Execution ---
            result_data = await self._execute(
                agent=agent, task=task, context=context, agents=agents
            )

            # --- Format Success Result ---
            result = self._format_success(result_data, start_time)

        except Exception as e:
            self.logger.error(
                f"Strategy '{self.mode.value}' failed for task '{task[:100]}...': {e}",
                exc_info=True,
            )
            # --- Record Exception Metric ---
            self.metrics.record_exception(e, location=f"strategy:{self.mode.value}")
            # --- Format Error Result ---
            result = self._format_error(e, start_time)

        # --- Post-Execution Hook (runs for both success and failure) ---
        final_result = await self.post_run(result)

        # --- Record Metrics ---
        duration_s = final_result["metadata"]["runtime_seconds"]
        status = final_result["status"]
        # Assuming a metric for strategy execution exists
        # self.metrics.record_strategy_execution(duration_s, self.mode.value, status)

        return final_result

    # --- Lifecycle Hooks for Subclasses ---

    async def pre_run(self, agent: "AgentNet", task: str, context: Dict[str, Any]) -> None:
        """Hook for setup or validation before the main execution logic."""
        self.logger.debug(f"Pre-run checks for {self.mode.value} strategy.")
        pass

    async def post_run(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for cleanup, finalization, or result modification after execution."""
        if result["status"] == "success":
            result = self._calculate_and_attach_flow_metrics(result)
        self.logger.debug(f"Post-run finalization for {self.mode.value} strategy.")
        return result

    # --- State Management ---

    def update_state(self, key: str, value: Any):
        """Update a value in the strategy's internal state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the strategy's internal state."""
        return self.state.get(key, default)

    # --- Helper Methods ---

    def _format_success(self, data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Construct a standardized success result dictionary."""
        runtime = time.monotonic() - start_time
        return {
            "status": "success",
            "data": data,
            "metadata": self._prepare_metadata(runtime_seconds=runtime),
        }

    def _format_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Construct a standardized error result dictionary."""
        runtime = time.monotonic() - start_time
        return {
            "status": "error",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "metadata": self._prepare_metadata(runtime_seconds=runtime),
        }

    def _prepare_metadata(self, **kwargs) -> Dict[str, Any]:
        """Prepare metadata with mode, style, technique, and other info."""
        metadata = {
            "mode": self.mode.value,
            "problem_solving_style": self.style.value if self.style else None,
            "problem_technique": self.technique.value if self.technique else None,
            **kwargs,
        }
        return metadata

    def _calculate_and_attach_flow_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate and attach flow metrics to the result metadata."""
        try:
            runtime_seconds = result.get("metadata", {}).get("runtime_seconds")
            reasoning_tree = result.get("data", {}) # Assuming result data is the tree

            # Extract difficulty from AutoConfig if available
            difficulty: Optional["TaskDifficulty"] = None
            autoconfig_data = reasoning_tree.get("autoconfig")
            if autoconfig_data and (difficulty_str := autoconfig_data.get("difficulty")):
                from ...core.autoconfig import TaskDifficulty
                try:
                    difficulty = TaskDifficulty(difficulty_str)
                except ValueError:
                    self.logger.warning(f"Invalid difficulty string '{difficulty_str}' in autoconfig.")

            flow_metrics = calculate_flow_metrics(
                reasoning_tree=reasoning_tree,
                runtime_seconds=runtime_seconds,
                technique=self.technique,
                difficulty=difficulty,
            )

            result["metadata"]["flow_metrics"] = {
                "current": flow_metrics.current,
                "voltage": flow_metrics.voltage,
                "resistance": flow_metrics.resistance,
                "power": flow_metrics.power,
            }
            self.logger.debug(f"Flow metrics calculated: {result['metadata']['flow_metrics']}")

        except Exception as e:
            self.logger.warning(f"Failed to calculate flow metrics: {e}", exc_info=False)
            # Do not fail the entire operation if metrics calculation fails

        return result

    def _log_strategy_execution(self, task: str):
        """Log the start of the strategy execution for observability."""
        style_str = f" (style: {self.style.value})" if self.style else ""
        tech_str = f" (technique: {self.technique.value})" if self.technique else ""
        self.logger.info(
            f"Executing '{self.mode.value}' strategy for task: {task[:100]}..."
            f"{style_str}{tech_str}"
        )
