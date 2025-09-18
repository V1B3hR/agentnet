"""Advanced reasoning types for AgentNet."""

from .types import (
    ReasoningType,
    DeductiveReasoning,
    InductiveReasoning,
    AbductiveReasoning,
    AnalogicalReasoning,
    CausalReasoning,
    ReasoningEngine
)

from .modulation import (
    ReasoningStyleModulator,
    ReasoningAwareStyleInfluence
)

__all__ = [
    'ReasoningType',
    'DeductiveReasoning',
    'InductiveReasoning', 
    'AbductiveReasoning',
    'AnalogicalReasoning',
    'CausalReasoning',
    'ReasoningEngine',
    'ReasoningStyleModulator',
    'ReasoningAwareStyleInfluence'
]