"""
Mode-based orchestration strategies for AgentNet.

Provides base strategy classes for different problem-solving modes.
"""

from .base import BaseStrategy
from .brainstorm import BrainstormStrategy
from .debate import DebateStrategy
from .consensus import ConsensusStrategy
from .workflow import WorkflowStrategy
from .dialogue import DialogueStrategy

__all__ = [
    "BaseStrategy",
    "BrainstormStrategy",
    "DebateStrategy",
    "ConsensusStrategy",
    "WorkflowStrategy",
    "DialogueStrategy",
]
