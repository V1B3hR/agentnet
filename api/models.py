"""
API data models for AgentNet.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class DialogueMode(str, Enum):
    GENERAL = "general"
    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    CONSENSUS = "consensus"


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    style: Dict[str, float]
    monitors: Optional[List[str]] = None


@dataclass
class SessionRequest:
    """Request to create a new multi-agent session."""
    topic: str
    agents: List[AgentConfig]
    mode: DialogueMode = DialogueMode.GENERAL
    max_rounds: int = 5
    convergence: bool = True
    parallel_round: bool = False
    convergence_config: Optional[Dict[str, Any]] = None


@dataclass
class SessionResponse:
    """Response containing session information."""
    session_id: str
    status: str  # "running", "completed", "failed"
    topic_start: Optional[str] = None
    topic_final: Optional[str] = None
    converged: Optional[bool] = None
    rounds_executed: Optional[int] = None
    participants: Optional[List[str]] = None
    transcript: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


@dataclass
class SessionStatus:
    """Current status of a session."""
    session_id: str
    status: str
    current_round: int
    total_rounds: int
    converged: bool
    participants: List[str]
    last_speaker: Optional[str] = None