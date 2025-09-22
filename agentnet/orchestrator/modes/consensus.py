"""
Consensus strategy implementation.

Focuses on finding common ground and shared agreement with AgentNet.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class ConsensusStrategy(BaseStrategy):
    """Strategy for consensus mode - focuses on finding common ground and shared agreement."""
    
    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = None,
        technique: Optional[ProblemTechnique] = None
    ):
        """Initialize consensus strategy."""
        super().__init__(Mode.CONSENSUS, style, technique)
    
    def execute(
        self,
        agent: "AgentNet",
        task: str,
        agents: Optional[List["AgentNet"]] = None,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute consensus strategy.
        
        Args:
            agent: Primary agent for execution
            task: Task description
            agents: Optional list of agents for multi-agent consensus building
            max_depth: Maximum reasoning depth
            confidence_threshold: Confidence threshold for agreement
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary with consensus output
        """
        start_time = time.time()
        self._log_strategy_execution(task, **kwargs)
        
        # Prepare metadata with consensus-specific tags
        metadata = self._prepare_metadata(**kwargs)
        metadata["consensus_focus"] = "shared_agreement"
        metadata["collaboration_mode"] = True
        
        # Remove metadata from kwargs to avoid duplicate parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'metadata'}
        
        # Modify task prompt for consensus-building context
        consensus_task = f"""Work toward shared agreement on: {task}

Focus on:
- Identifying common ground and shared values
- Finding mutually acceptable solutions
- Highlighting areas of convergence
- Building bridges between different perspectives
- Seeking win-win outcomes that satisfy key concerns

Move toward collaborative understanding and shared commitment."""
        
        # Execute reasoning with consensus-optimized parameters
        result = agent.generate_reasoning_tree(
            task=consensus_task,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            metadata=metadata,
            **filtered_kwargs
        )
        
        # Add consensus-specific metadata to result
        result["strategy"] = {
            "mode": self.mode.value,
            "style": self.style.value if self.style else None,
            "technique": self.technique.value if self.technique else None,
            "focus": "shared_agreement",
            "execution_time": time.time() - start_time
        }
        
        # Calculate and attach flow metrics
        result = self._calculate_and_attach_flow_metrics(result, time.time() - start_time)
        
        return result