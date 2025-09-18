"""Enhanced dialogue modes for AgentNet."""

from .enhanced_modes import (
    DialogueMapping,
    DialogueMode,
    InnerDialogue,
    InterpolationConversation,
    ModulatedConversation,
    OuterDialogue,
)
from .solution_focused import (
    ConstructiveQuestioning,
    ProblemSolutionTransition,
    SolutionFocusedDialogue,
)

__all__ = [
    "DialogueMode",
    "OuterDialogue",
    "ModulatedConversation",
    "InterpolationConversation",
    "InnerDialogue",
    "DialogueMapping",
    "SolutionFocusedDialogue",
    "ProblemSolutionTransition",
    "ConstructiveQuestioning",
]
