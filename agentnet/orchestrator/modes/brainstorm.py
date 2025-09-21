"""
Brainstorm strategy implementation.

Focuses on idea generation and creative exploration with AgentNet.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class BrainstormStrategy(BaseStrategy):
    """Strategy for brainstorming mode - focuses on idea generation and creative exploration."""
    
    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None
    ):
        """Initialize brainstorm strategy."""
        super().__init__(Mode.BRAINSTORM, style, technique)
    
    def execute(
        self,
        agent: "AgentNet",
        task: str,
        agents: Optional[List["AgentNet"]] = None,
        max_depth: int = 3,
        confidence_threshold: float = 0.6,  # Lower threshold for creative exploration
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute brainstorm strategy.
        
        Args:
            agent: Primary agent for execution
            task: Task description
            agents: Optional list of agents for multi-agent brainstorming
            max_depth: Maximum reasoning depth
            confidence_threshold: Confidence threshold for ideas
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary with brainstorming output
        """
        start_time = time.time()
        self._log_strategy_execution(task, **kwargs)
        
        # Prepare metadata with brainstorm-specific tags
        metadata = self._prepare_metadata(**kwargs)
        metadata["brainstorm_focus"] = "idea_generation"
        metadata["creativity_mode"] = True
        
        # Modify task prompt for brainstorming context
        brainstorm_task = f"""Generate diverse, novel ideas for: {task}

Focus on:
- Creative and unconventional approaches
- Multiple perspectives and alternatives  
- Quantity over initial quality judgment
- Building on and combining ideas

Think freely and explore various possibilities without premature evaluation."""
        
        # Execute reasoning with brainstorm-optimized parameters
        result = agent.generate_reasoning_tree(
            task=brainstorm_task,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            metadata=metadata,
            **kwargs
        )
        
        # Add brainstorm-specific metadata to result
        result["strategy"] = {
            "mode": self.mode.value,
            "style": self.style.value if self.style else None,
            "technique": self.technique.value if self.technique else None,
            "focus": "idea_generation",
            "execution_time": time.time() - start_time
        }
        
        # Calculate and attach flow metrics
        result = self._calculate_and_attach_flow_metrics(result, time.time() - start_time)
        
        return result