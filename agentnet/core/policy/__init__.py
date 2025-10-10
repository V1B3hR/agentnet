"""
Policy Engine Module

Provides rule-based policy evaluation for agent actions and outputs.
Implements matchers for regex, role, tool usage, and content classification.
Includes the 25 AI Fundamental Laws implementation.
"""

from .engine import PolicyEngine, PolicyAction, PolicyResult
from .rules import ConstraintRule, RuleResult, Severity
from .fundamental_laws import FundamentalLawsEngine, create_all_fundamental_laws

# Import advanced rule types
try:
    from .advanced_rules import (
        SemanticSimilarityRule,
        LLMClassifierRule,
        NumericalThresholdRule,
    )
    _advanced_rules_available = True
except ImportError:
    SemanticSimilarityRule = None
    LLMClassifierRule = None
    NumericalThresholdRule = None
    _advanced_rules_available = False

__all__ = [
    "PolicyEngine",
    "PolicyAction",
    "PolicyResult",
    "ConstraintRule",
    "RuleResult",
    "Severity",
    "FundamentalLawsEngine",
    "create_all_fundamental_laws",
    "SemanticSimilarityRule",
    "LLMClassifierRule",
    "NumericalThresholdRule",
]
