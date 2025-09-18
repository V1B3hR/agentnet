"""Advanced reasoning types for AgentNet."""

from .modulation import ReasoningAwareStyleInfluence, ReasoningStyleModulator
from .types import (
    AbductiveReasoning,
    AnalogicalReasoning,
    CausalReasoning,
    DeductiveReasoning,
    InductiveReasoning,
    ReasoningEngine,
    ReasoningType,
)

__all__ = [
    "ReasoningType",
    "DeductiveReasoning",
    "InductiveReasoning",
    "AbductiveReasoning",
    "AnalogicalReasoning",
    "CausalReasoning",
    "ReasoningEngine",
    "ReasoningStyleModulator",
    "ReasoningAwareStyleInfluence",
]
