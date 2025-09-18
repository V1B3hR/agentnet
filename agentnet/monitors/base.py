"""Base monitor interfaces and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.types import Severity

# Type alias for monitor functions
MonitorFn = Callable[[Any, str, Dict[str, Any]], None]


@dataclass
class MonitorSpec:
    """Specification for a monitor instance."""

    name: str
    type: str  # "keyword", "regex", "custom", "resource"
    params: Dict[str, Any]
    severity: Any  # Will be Severity enum
    description: Optional[str] = None
    cooldown_seconds: Optional[float] = None


class MonitorTemplate(ABC):
    """Base template for creating custom monitors."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def check(self, agent: Any, task: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check the result and return monitoring information.

        Args:
            agent: The agent instance
            task: The task being monitored
            result: The result to check

        Returns:
            Dictionary with monitoring results including violations if any
        """
        pass

    def to_monitor_fn(self) -> MonitorFn:
        """Convert to a monitor function."""

        def monitor_fn(agent: Any, task: str, result: Dict[str, Any]) -> None:
            if not self.enabled:
                return
            check_result = self.check(agent, task, result)
            if check_result.get("violations"):
                # Handle violations (could integrate with agent's violation handling)
                pass

        return monitor_fn
