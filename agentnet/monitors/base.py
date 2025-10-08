"""
Base interfaces, data classes, and type definitions for the AgentNet monitoring system.

This module defines the core data structure, MonitorSpec, which encapsulates the
configuration for all monitor types, and the MonitorFn type alias for monitor
functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..core.types import Severity, _parse_severity

if TYPE_CHECKING:
    from ..core.agent import AgentNet

# --- Type Alias for Monitor Functions ---

# A monitor function is a callable that inspects an agent's output.
# It takes the agent instance, the original task, and the result dictionary.
# It does not return anything; it handles violations via the MonitorFactory.
MonitorFn = Callable[["AgentNet", str, Dict[str, Any]], None]


# --- Core Monitor Configuration ---

@dataclass
class MonitorSpec:
    """
    A declarative specification for configuring a single monitor instance.

    This data class holds all the metadata needed by the MonitorFactory to
    build and operate a monitor, including its type, parameters, and behavior
    on violation.
    """

    # --- Core Identity ---
    name: str
    type: str

    # --- Behavior and Parameters ---
    params: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.MINOR

    # --- Metadata and Control ---
    description: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)
    enabled: bool = True

    # --- Action and Cooldown Policy ---
    action: str = "raise"
    cooldown_seconds: Optional[float] = None
    cooldown_scope: str = "task"

    def __post_init__(self):
        """
        Performs validation and type coercion after initialization.
        This makes the spec robust to string-based inputs from config files.
        """
        # Coerce severity from string (e.g., "severe") to Severity enum member.
        if not isinstance(self.severity, Severity):
            self.severity = _parse_severity(self.severity)

        # Validate the 'action' parameter.
        valid_actions = ["raise", "warn"]
        if self.action not in valid_actions:
            raise ValueError(
                f"Invalid action '{self.action}' for monitor '{self.name}'. "
                f"Must be one of {valid_actions}."
            )

        # Validate the 'cooldown_scope' parameter.
        valid_scopes = ["task", "agent", "global"]
        if self.cooldown_scope not in valid_scopes:
            raise ValueError(
                f"Invalid cooldown_scope '{self.cooldown_scope}' for monitor '{self.name}'. "
                f"Must be one of {valid_scopes}."
            )
