"""
Debate strategy implementation.

Focuses on structured argumentation and critical analysis with AgentNet.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class DebateStrategy(BaseStrategy):
    """Strategy for debate mode - focuses on structured argumentation and critical analysis."""
    
    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None
    ):
        """Initialize debate strategy."""
        super().__init__(Mode.DEBATE, style, technique)
    
    def execute(
        self,
        agent: "AgentNet",
        task: str,
        agents: Optional[List["AgentNet"]] = None,
        max_depth: int = 4,  # Deeper analysis for debate
        confidence_threshold: float = 0.75,  # Higher threshold for arguments
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute debate strategy.
        
        Args:
            agent: Primary agent for execution
            task: Task description
            agents: Optional list of agents for multi-agent debate
            max_depth: Maximum reasoning depth
            confidence_threshold: Confidence threshold for arguments
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary with debate output
        """
        start_time = time.time()
        self._log_strategy_execution(task, **kwargs)
        
        # Prepare metadata with debate-specific tags
        metadata = self._prepare_metadata(**kwargs)
        metadata["debate_focus"] = "critical_analysis"
        metadata["argumentation_mode"] = True
        
        # Remove metadata from kwargs to avoid duplicate parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'metadata'}
        
        # Modify task prompt for debate context
        debate_task = f"""Analyze and debate the following topic: {task}

Provide structured argumentation including:
- Clear position statement
- Supporting evidence and reasoning
- Consideration of counterarguments
- Critical evaluation of different perspectives
- Logical reasoning and evidence-based conclusions

Defend your position while acknowledging valid opposing viewpoints."""
        
        # Execute reasoning with debate-optimized parameters
        result = agent.generate_reasoning_tree(
            task=debate_task,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            metadata=metadata,
            **filtered_kwargs
        )
        
        # Add debate-specific metadata to result
        result["strategy"] = {
            "mode": self.mode.value,
            "style": self.style.value if self.style else None,
            "technique": self.technique.value if self.technique else None,
            "focus": "critical_analysis",
            "execution_time": time.time() - start_time
        }
        
        # Calculate and attach flow metrics
        result = self._calculate_and_attach_flow_metrics(result, time.time() - start_time)
        
        return result