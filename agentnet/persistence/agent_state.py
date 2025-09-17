"""Agent state persistence management."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.persistence")


class AgentStateManager:
    """Manages agent state persistence operations."""
    
    @staticmethod
    def save_state(agent: "AgentNet", path: str) -> None:
        """Save agent state to file.
        
        Args:
            agent: Agent instance to save
            path: File path to save to
        """
        state = {
            "name": agent.name,
            "style": agent.style,
            "knowledge_graph": agent.knowledge_graph,
            "interaction_history": agent.interaction_history,
            "dialogue_config": agent.dialogue_config,
            "version": "1.0"
        }
        Path(path).write_text(json.dumps(state, indent=2))
        logger.info(f"Agent state saved to {path}")

    @staticmethod
    def load_state(
        path: str,
        agent_class: type,
        engine=None,
        monitors: Optional[list] = None
    ) -> "AgentNet":
        """Load agent state from file.
        
        Args:
            path: File path to load from
            agent_class: AgentNet class to instantiate
            engine: Engine instance for the agent
            monitors: Monitor functions for the agent
            
        Returns:
            Restored agent instance
        """
        state = json.loads(Path(path).read_text())
        agent = agent_class(
            state["name"], 
            state.get("style", {}), 
            engine=engine,
            monitors=monitors, 
            dialogue_config=state.get("dialogue_config")
        )
        agent.knowledge_graph = state.get("knowledge_graph", {})
        agent.interaction_history = state.get("interaction_history", [])
        logger.info(f"Agent state loaded from {path}")
        return agent