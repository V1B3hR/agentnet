"""
Critique and Evaluation Module

Provides self-revision capabilities, debate mechanisms, and arbitration strategies
for multi-agent reasoning and quality improvement.
"""

from .evaluator import CritiqueEvaluator, RevisionEvaluator, CritiqueResult
from .arbitrator import Arbitrator, ArbitrationStrategy, ArbitrationResult
from .debate import DebateManager, DebateRole, DebateResult

__all__ = [
    "CritiqueEvaluator",
    "RevisionEvaluator",
    "CritiqueResult",
    "Arbitrator",
    "ArbitrationStrategy",
    "ArbitrationResult",
    "DebateManager",
    "DebateRole",
    "DebateResult",
]
