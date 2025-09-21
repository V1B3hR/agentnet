"""
AutoConfig module for dynamic scenario parameter adaptation based on task difficulty.

This module analyzes task complexity and automatically adjusts:
- Scenario rounds (more rounds for harder tasks)
- Reasoning depth hints (deeper reasoning for complex tasks)
- Confidence thresholds (stricter thresholds for harder tasks)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("agentnet.core.autoconfig")


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class AutoConfigParams:
    """Auto-configured parameters for scenarios."""
    rounds: int
    max_depth: int
    confidence_threshold: float
    difficulty: TaskDifficulty
    reasoning: str
    confidence_adjustment: float


class AutoConfig:
    """
    Automatic configuration system that adapts scenario parameters to task difficulty.
    
    Harder tasks receive:
    - More rounds (default 5)
    - Deeper reasoning (depth hint 4)
    - Stricter confidence thresholds (0.8)
    
    Medium and simple tasks receive proportionally lighter configuration.
    """

    def __init__(self):
        """Initialize AutoConfig with default parameters."""
        self.default_params = {
            TaskDifficulty.SIMPLE: {
                "rounds": 3,
                "max_depth": 2,
                "confidence_threshold": 0.6,
                "confidence_adjustment": -0.1,  # Lower threshold
            },
            TaskDifficulty.MEDIUM: {
                "rounds": 4,
                "max_depth": 3,
                "confidence_threshold": 0.7,
                "confidence_adjustment": 0.0,  # Keep existing
            },
            TaskDifficulty.HARD: {
                "rounds": 5,
                "max_depth": 4,
                "confidence_threshold": 0.8,
                "confidence_adjustment": 0.1,  # Raise threshold
            },
        }

    def analyze_task_difficulty(self, task: str, context: Optional[Dict[str, Any]] = None) -> TaskDifficulty:
        """
        Analyze task difficulty based on linguistic complexity and domain indicators.
        
        Args:
            task: The task description to analyze
            context: Optional context for analysis
            
        Returns:
            TaskDifficulty level
        """
        if not task or len(task.strip()) == 0:
            return TaskDifficulty.SIMPLE

        task_lower = task.lower()
        
        # Hard task indicators
        hard_indicators = [
            # Complexity words
            "complex", "sophisticated", "intricate", "comprehensive", "multifaceted",
            "nuanced", "elaborate", "advanced", "in-depth", "thorough",
            
            # Technical/domain-specific terms
            "algorithm", "architecture", "framework", "methodology", "strategy",
            "optimization", "implementation", "analysis", "evaluation", "synthesis",
            
            # Multi-step reasoning
            "compare and contrast", "analyze the relationship", "evaluate the impact",
            "synthesize information", "develop a comprehensive", "create a detailed plan",
            
            # Research/academic terms  
            "research", "investigate", "hypothesis", "evidence", "conclusion",
            "implications", "considerations", "trade-offs", "alternatives",
            
            # Decision-making complexity
            "ethical", "policy", "governance", "compliance", "risk assessment",
            "stakeholder", "multi-criteria", "prioritization"
        ]

        # Medium task indicators
        medium_indicators = [
            "explain", "describe", "outline", "summarize", "compare",
            "identify", "classify", "categorize", "organize", "plan",
            "design", "propose", "recommend", "suggest", "improve",
            "problem", "solution", "approach", "method", "process"
        ]

        # Simple task indicators  
        simple_indicators = [
            "list", "name", "what is", "define", "when", "where", "who",
            "yes", "no", "true", "false", "choose", "select", "pick"
        ]

        # Count indicators
        hard_count = sum(1 for indicator in hard_indicators if indicator in task_lower)
        medium_count = sum(1 for indicator in medium_indicators if indicator in task_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in task_lower)

        # Check task length (longer tasks tend to be more complex)
        word_count = len(task.split())
        sentence_count = len(re.split(r'[.!?]+', task))
        
        # Scoring system
        difficulty_score = 0
        
        # Add points for complexity indicators
        difficulty_score += hard_count * 3
        difficulty_score += medium_count * 1
        difficulty_score -= simple_count * 1
        
        # Add points for length complexity
        if word_count > 50:
            difficulty_score += 2
        elif word_count > 20:
            difficulty_score += 1
        
        if sentence_count > 3:
            difficulty_score += 1
            
        # Add points for punctuation complexity (questions, multiple clauses)
        if "?" in task:
            difficulty_score += 0.5
        if task.count(",") > 2:
            difficulty_score += 0.5
        if ";" in task or ":" in task:
            difficulty_score += 0.5

        # Context-based adjustments
        if context:
            confidence = context.get("confidence", 0.5)
            if confidence < 0.5:
                difficulty_score += 1  # Low confidence suggests complexity
            
            # Check for domain-specific context
            domain_indicators = context.get("domain", "").lower()
            if any(domain in domain_indicators for domain in ["technical", "research", "policy", "academic"]):
                difficulty_score += 1

        # Classify based on score
        if difficulty_score >= 5:
            return TaskDifficulty.HARD
        elif difficulty_score >= 2:
            return TaskDifficulty.MEDIUM
        else:
            return TaskDifficulty.SIMPLE

    def configure_scenario(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None,
        base_rounds: Optional[int] = None,
        base_max_depth: Optional[int] = None,
        base_confidence_threshold: Optional[float] = None
    ) -> AutoConfigParams:
        """
        Configure scenario parameters based on task difficulty.
        
        Args:
            task: Task description
            context: Optional context for analysis
            base_rounds: Base rounds to adjust from
            base_max_depth: Base max depth to adjust from  
            base_confidence_threshold: Base confidence threshold to adjust from
            
        Returns:
            AutoConfigParams with configured values
        """
        difficulty = self.analyze_task_difficulty(task, context)
        params = self.default_params[difficulty].copy()
        
        # Override with base values if provided
        if base_rounds is not None:
            params["rounds"] = max(base_rounds, params["rounds"])
        if base_max_depth is not None:
            params["max_depth"] = max(base_max_depth, params["max_depth"])
        if base_confidence_threshold is not None:
            # For base parameter override, use the higher of base or auto-configured threshold
            # But still apply positive adjustments for harder tasks
            if params["confidence_adjustment"] > 0:
                # For harder tasks, apply the positive adjustment
                adjusted_threshold = base_confidence_threshold + params["confidence_adjustment"]
                params["confidence_threshold"] = max(base_confidence_threshold, adjusted_threshold)
            else:
                # For simpler tasks, preserve the base threshold if it's higher
                params["confidence_threshold"] = max(base_confidence_threshold, params["confidence_threshold"])
        
        reasoning = self._generate_reasoning(difficulty, task)
        
        return AutoConfigParams(
            rounds=params["rounds"],
            max_depth=params["max_depth"],
            confidence_threshold=params["confidence_threshold"],
            difficulty=difficulty,
            reasoning=reasoning,
            confidence_adjustment=params["confidence_adjustment"]
        )

    def _generate_reasoning(self, difficulty: TaskDifficulty, task: str) -> str:
        """Generate reasoning explanation for the auto-configuration."""
        if difficulty == TaskDifficulty.HARD:
            return f"Task classified as HARD due to complexity indicators. Using enhanced configuration: 5 rounds, depth 4, confidence 0.8"
        elif difficulty == TaskDifficulty.MEDIUM:
            return f"Task classified as MEDIUM complexity. Using balanced configuration: 4 rounds, depth 3, confidence 0.7"
        else:
            return f"Task classified as SIMPLE. Using lightweight configuration: 3 rounds, depth 2, confidence 0.6"

    def should_auto_configure(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if auto-configuration should be applied based on metadata.
        
        Args:
            metadata: Metadata dictionary that may contain auto_config setting
            
        Returns:
            True if auto-configuration should be applied, False otherwise
        """
        if metadata is None:
            return True
        
        auto_config = metadata.get("auto_config")
        if auto_config is None:
            return True
        
        return bool(auto_config)

    def inject_autoconfig_data(self, session_data: Dict[str, Any], config_params: AutoConfigParams) -> None:
        """
        Inject auto-configuration data into session data for observability.
        
        Args:
            session_data: Session data dictionary to modify
            config_params: Auto-configuration parameters to inject
        """
        session_data["autoconfig"] = {
            "difficulty": config_params.difficulty.value,
            "configured_rounds": config_params.rounds,
            "configured_max_depth": config_params.max_depth,
            "configured_confidence_threshold": config_params.confidence_threshold,
            "reasoning": config_params.reasoning,
            "confidence_adjustment": config_params.confidence_adjustment,
            "enabled": True
        }

    def preserve_confidence_threshold(
        self, 
        original_threshold: float, 
        config_params: AutoConfigParams
    ) -> float:
        """
        Preserve or raise confidence threshold, never lower it below original.
        
        Args:
            original_threshold: Original confidence threshold
            config_params: Auto-configuration parameters
            
        Returns:
            Preserved or raised confidence threshold
        """
        # Only raise the threshold, never lower it
        return max(original_threshold, config_params.confidence_threshold)


# Global AutoConfig instance
_global_autoconfig: Optional[AutoConfig] = None


def get_global_autoconfig() -> AutoConfig:
    """Get or create global AutoConfig instance."""
    global _global_autoconfig
    if _global_autoconfig is None:
        _global_autoconfig = AutoConfig()
    return _global_autoconfig


def set_global_autoconfig(autoconfig: AutoConfig) -> None:
    """Set global AutoConfig instance."""
    global _global_autoconfig
    _global_autoconfig = autoconfig