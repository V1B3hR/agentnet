"""Enhanced dialogue modes for AgentNet."""

from .enhanced_modes import (
    DialogueMode,
    OuterDialogue,
    ModulatedConversation,
    InterpolationConversation,
    InnerDialogue,
    DialogueMapping
)

from .solution_focused import (
    SolutionFocusedDialogue,
    ProblemSolutionTransition,
    ConstructiveQuestioning
)

__all__ = [
    'DialogueMode',
    'OuterDialogue',
    'ModulatedConversation', 
    'InterpolationConversation',
    'InnerDialogue',
    'DialogueMapping',
    'SolutionFocusedDialogue',
    'ProblemSolutionTransition',
    'ConstructiveQuestioning'
]