"""
Core enums for AgentNet problem-solving modes, styles, and techniques.

This module defines the enums for problem-solving modes, styles, and techniques
that can be used to adapt AgentNet behavior and AutoConfig parameters.
"""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    """Problem-solving modes for AgentNet orchestration."""
    
    BRAINSTORM = "brainstorm"
    DEBATE = "debate"
    CONSENSUS = "consensus"
    WORKFLOW = "workflow"
    DIALOGUE = "dialogue"


class ProblemSolvingStyle(str, Enum):
    """Problem-solving styles that influence agent behavior."""
    
    CLARIFIER = "clarifier"
    IDEATOR = "ideator"
    DEVELOPER = "developer"
    IMPLEMENTOR = "implementor"


class ProblemTechnique(str, Enum):
    """Problem-solving techniques for different scenarios."""
    
    SIMPLE = "simple"
    COMPLEX = "complex"
    TROUBLESHOOTING = "troubleshooting"
    GAP_FROM_STANDARD = "gap_from_standard"
    TARGET_STATE = "target_state"
    OPEN_ENDED = "open_ended"