"""
Advanced reasoning system for AgentNet.

Provides different reasoning types:
- Deductive: General-to-specific logical inference
- Inductive: Pattern recognition from specific observations
- Abductive: Best explanation formation from incomplete data
- Analogical: Similarity-based understanding and learning
- Causal: Cause-and-effect relationship identification

Phase 7 Advanced Reasoning:
- Chain-of-thought reasoning with step validation
- Multi-hop reasoning across knowledge graphs
- Enhanced counterfactual analysis
- Symbolic reasoning integration
- Temporal reasoning for episodic memory
"""

from .modulation import ReasoningAwareStyleInfluence, ReasoningStyleModulator
from .types import (
    AbductiveReasoning,
    AnalogicalReasoning,
    BaseReasoning,
    CausalReasoning,
    DeductiveReasoning,
    InductiveReasoning,
    ReasoningEngine,
    ReasoningResult,
    ReasoningType,
)

# Phase 7 Advanced Reasoning components
try:
    from .advanced import (
        ChainOfThoughtReasoning,
        MultiHopReasoning,
        CounterfactualReasoning,
        SymbolicReasoning,
        AdvancedReasoningEngine,
        StepValidation,
        KnowledgeGraph,
        ReasoningStep,
        ValidationResult,
    )
    from .temporal import (
        TemporalReasoning,
        TemporalPattern,
        TemporalSequence,
        TemporalRule,
        TemporalEvent,
        TemporalRelation,
    )
    _ADVANCED_REASONING_AVAILABLE = True
except ImportError:
    # Graceful fallback if Phase 7 components are not available
    ChainOfThoughtReasoning = None
    MultiHopReasoning = None
    CounterfactualReasoning = None
    SymbolicReasoning = None
    AdvancedReasoningEngine = None
    StepValidation = None
    KnowledgeGraph = None
    ReasoningStep = None
    ValidationResult = None
    TemporalReasoning = None
    TemporalPattern = None
    TemporalSequence = None
    TemporalRule = None
    TemporalEvent = None
    TemporalRelation = None
    _ADVANCED_REASONING_AVAILABLE = False

__all__ = [
    "ReasoningType",
    "ReasoningResult",
    "BaseReasoning",
    "DeductiveReasoning",
    "InductiveReasoning",
    "AbductiveReasoning",
    "AnalogicalReasoning",
    "CausalReasoning",
    "ReasoningEngine",
    "ReasoningStyleModulator",
    "ReasoningAwareStyleInfluence",
]

# Add Phase 7 exports if available
if _ADVANCED_REASONING_AVAILABLE:
    __all__.extend([
        "ChainOfThoughtReasoning",
        "MultiHopReasoning",
        "CounterfactualReasoning",
        "SymbolicReasoning",
        "AdvancedReasoningEngine",
        "StepValidation",
        "KnowledgeGraph",
        "ReasoningStep",
        "ValidationResult",
        "TemporalReasoning",
        "TemporalPattern",
        "TemporalSequence",
        "TemporalRule",
        "TemporalEvent",
        "TemporalRelation",
    ])
