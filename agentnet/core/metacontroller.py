"""
Meta-Controller Agent Module

Implements dynamic agent graph reconfiguration capabilities for Phase 6.
The meta-controller can add/remove agents, change connections, and adjust roles
based on context or performance feedback.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Standard agent roles that can be dynamically assigned."""
    ANALYST = "analyst"
    CRITIC = "critic"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"


class ReconfigurationTrigger(str, Enum):
    """Triggers that can cause graph reconfiguration."""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    TASK_COMPLEXITY = "task_complexity"
    FAILURE_RECOVERY = "failure_recovery"
    LOAD_BALANCING = "load_balancing"
    MANUAL_REQUEST = "manual_request"
    CONTEXT_CHANGE = "context_change"


@dataclass
class AgentNode:
    """Represents an agent in the dynamic graph."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: AgentRole = AgentRole.ANALYST
    capabilities: Set[str] = field(default_factory=set)
    connections: Set[str] = field(default_factory=set)  # Connected agent IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    performance_score: float = 0.0


@dataclass
class ReconfigurationEvent:
    """Record of a graph reconfiguration event."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    trigger: ReconfigurationTrigger = ReconfigurationTrigger.MANUAL_REQUEST
    action: str = ""  # Description of what was done
    affected_agents: List[str] = field(default_factory=list)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetaController:
    """
    Meta-controller agent for dynamic agent graph reconfiguration.
    
    This controller can:
    - Add/remove agents dynamically
    - Modify agent connections and roles
    - Switch orchestration modes based on context
    - Optimize agent allocation based on performance
    """
    
    def __init__(
        self,
        max_agents: int = 10,
        performance_threshold: float = 0.5,
        reconfiguration_cooldown: float = 30.0
    ):
        self.max_agents = max_agents
        self.performance_threshold = performance_threshold
        self.reconfiguration_cooldown = reconfiguration_cooldown
        
        self.agents: Dict[str, AgentNode] = {}
        self.connections: Dict[str, Set[str]] = {}
        self.reconfiguration_history: List[ReconfigurationEvent] = []
        self.last_reconfiguration: Optional[datetime] = None
        
        # Callbacks for orchestrator integration
        self.on_agent_added: Optional[Callable[[AgentNode], None]] = None
        self.on_agent_removed: Optional[Callable[[str], None]] = None
        self.on_connections_changed: Optional[Callable[[str, Set[str]], None]] = None
        
        logger.info(f"MetaController initialized with max_agents={max_agents}")
    
    def add_agent(
        self,
        name: str,
        role: AgentRole,
        capabilities: Optional[Set[str]] = None,
        connections: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new agent to the graph."""
        
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Maximum agent limit ({self.max_agents}) reached")
        
        agent = AgentNode(
            name=name,
            role=role,
            capabilities=capabilities or set(),
            connections=connections or set(),
            metadata=metadata or {}
        )
        
        self.agents[agent.id] = agent
        self.connections[agent.id] = agent.connections.copy()
        
        # Update bidirectional connections
        for connected_id in agent.connections:
            if connected_id in self.connections:
                self.connections[connected_id].add(agent.id)
        
        event = ReconfigurationEvent(
            trigger=ReconfigurationTrigger.MANUAL_REQUEST,
            action=f"Added agent '{name}' with role {role}",
            affected_agents=[agent.id]
        )
        self.reconfiguration_history.append(event)
        
        if self.on_agent_added:
            self.on_agent_added(agent)
        
        logger.info(f"Added agent {agent.id} ({name}) with role {role}")
        return agent.id
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the graph."""
        
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Remove connections to this agent
        for other_id, other_connections in self.connections.items():
            other_connections.discard(agent_id)
        
        # Remove the agent
        del self.agents[agent_id]
        del self.connections[agent_id]
        
        event = ReconfigurationEvent(
            trigger=ReconfigurationTrigger.MANUAL_REQUEST,
            action=f"Removed agent '{agent.name}'",
            affected_agents=[agent_id]
        )
        self.reconfiguration_history.append(event)
        
        if self.on_agent_removed:
            self.on_agent_removed(agent_id)
        
        logger.info(f"Removed agent {agent_id} ({agent.name})")
        return True
    
    def connect_agents(self, agent1_id: str, agent2_id: str) -> bool:
        """Create a bidirectional connection between two agents."""
        
        if agent1_id not in self.agents or agent2_id not in self.agents:
            return False
        
        self.connections[agent1_id].add(agent2_id)
        self.connections[agent2_id].add(agent1_id)
        
        self.agents[agent1_id].connections.add(agent2_id)
        self.agents[agent2_id].connections.add(agent1_id)
        
        event = ReconfigurationEvent(
            trigger=ReconfigurationTrigger.MANUAL_REQUEST,
            action=f"Connected agents {agent1_id} and {agent2_id}",
            affected_agents=[agent1_id, agent2_id]
        )
        self.reconfiguration_history.append(event)
        
        if self.on_connections_changed:
            self.on_connections_changed(agent1_id, self.connections[agent1_id])
            self.on_connections_changed(agent2_id, self.connections[agent2_id])
        
        logger.info(f"Connected agents {agent1_id} and {agent2_id}")
        return True
    
    def disconnect_agents(self, agent1_id: str, agent2_id: str) -> bool:
        """Remove connection between two agents."""
        
        if agent1_id not in self.agents or agent2_id not in self.agents:
            return False
        
        self.connections[agent1_id].discard(agent2_id)
        self.connections[agent2_id].discard(agent1_id)
        
        self.agents[agent1_id].connections.discard(agent2_id)
        self.agents[agent2_id].connections.discard(agent1_id)
        
        event = ReconfigurationEvent(
            trigger=ReconfigurationTrigger.MANUAL_REQUEST,
            action=f"Disconnected agents {agent1_id} and {agent2_id}",
            affected_agents=[agent1_id, agent2_id]
        )
        self.reconfiguration_history.append(event)
        
        if self.on_connections_changed:
            self.on_connections_changed(agent1_id, self.connections[agent1_id])
            self.on_connections_changed(agent2_id, self.connections[agent2_id])
        
        logger.info(f"Disconnected agents {agent1_id} and {agent2_id}")
        return True
    
    def change_agent_role(self, agent_id: str, new_role: AgentRole) -> bool:
        """Change an agent's role."""
        
        if agent_id not in self.agents:
            return False
        
        old_role = self.agents[agent_id].role
        self.agents[agent_id].role = new_role
        
        event = ReconfigurationEvent(
            trigger=ReconfigurationTrigger.MANUAL_REQUEST,
            action=f"Changed agent {agent_id} role from {old_role} to {new_role}",
            affected_agents=[agent_id]
        )
        self.reconfiguration_history.append(event)
        
        logger.info(f"Changed agent {agent_id} role from {old_role} to {new_role}")
        return True
    
    def update_performance(self, agent_id: str, performance_score: float) -> None:
        """Update an agent's performance score."""
        
        if agent_id in self.agents:
            self.agents[agent_id].performance_score = performance_score
            logger.debug(f"Updated performance for agent {agent_id}: {performance_score}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall graph performance and suggest reconfigurations."""
        
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
        avg_performance = sum(agent.performance_score for agent in self.agents.values()) / max(total_agents, 1)
        
        underperforming = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.performance_score < self.performance_threshold
        ]
        
        analysis = {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "average_performance": avg_performance,
            "underperforming_agents": underperforming,
            "suggestions": []
        }
        
        # Generate suggestions
        if len(underperforming) > total_agents * 0.3:
            analysis["suggestions"].append("Consider role reassignment for underperforming agents")
        
        if avg_performance < self.performance_threshold:
            analysis["suggestions"].append("Consider adding specialist agents")
        
        if total_agents < 3 and avg_performance > 0.8:
            analysis["suggestions"].append("Current configuration is optimal")
        
        return analysis
    
    def auto_reconfigure(self, trigger: ReconfigurationTrigger, context: Dict[str, Any]) -> bool:
        """Automatically reconfigure based on performance or context."""
        
        # Check cooldown
        if (self.last_reconfiguration and 
            (datetime.now() - self.last_reconfiguration).total_seconds() < self.reconfiguration_cooldown):
            logger.debug("Reconfiguration on cooldown")
            return False
        
        logger.info(f"Auto-reconfiguration triggered by {trigger}")
        
        reconfigured = False
        
        if trigger == ReconfigurationTrigger.PERFORMANCE_THRESHOLD:
            analysis = self.analyze_performance()
            
            # Remove underperforming agents if we have enough
            if len(analysis["underperforming_agents"]) > 0 and len(self.agents) > 2:
                for agent_id in analysis["underperforming_agents"][:1]:  # Remove one at a time
                    self.remove_agent(agent_id)
                    reconfigured = True
                    break
            
            # Add specialist if average performance is low
            elif analysis["average_performance"] < self.performance_threshold and len(self.agents) < self.max_agents:
                specialist_id = self.add_agent(
                    name=f"Specialist-{len(self.agents)+1}",
                    role=AgentRole.SPECIALIST,
                    capabilities={"optimization", "analysis"}
                )
                reconfigured = True
        
        elif trigger == ReconfigurationTrigger.TASK_COMPLEXITY:
            complexity = context.get("complexity", 0.5)
            if complexity > 0.7 and len(self.agents) < self.max_agents:
                # Add a coordinator for complex tasks
                coordinator_id = self.add_agent(
                    name=f"Coordinator-{len(self.agents)+1}",
                    role=AgentRole.COORDINATOR,
                    capabilities={"coordination", "planning"}
                )
                # Connect coordinator to all existing agents
                for agent_id in list(self.agents.keys())[:-1]:  # Exclude the coordinator itself
                    self.connect_agents(coordinator_id, agent_id)
                reconfigured = True
        
        if reconfigured:
            self.last_reconfiguration = datetime.now()
            
            event = ReconfigurationEvent(
                trigger=trigger,
                action=f"Auto-reconfiguration completed",
                affected_agents=list(self.agents.keys()),
                metadata=context
            )
            self.reconfiguration_history.append(event)
        
        return reconfigured
    
    def get_graph_state(self) -> Dict[str, Any]:
        """Get current state of the agent graph."""
        
        return {
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "role": agent.role,
                    "capabilities": list(agent.capabilities),
                    "connections": list(agent.connections),
                    "performance_score": agent.performance_score,
                    "is_active": agent.is_active,
                    "created_at": agent.created_at.isoformat()
                }
                for agent_id, agent in self.agents.items()
            },
            "total_agents": len(self.agents),
            "total_connections": sum(len(connections) for connections in self.connections.values()) // 2,
            "last_reconfiguration": self.last_reconfiguration.isoformat() if self.last_reconfiguration else None
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for persistence."""
        
        return {
            "meta_controller_config": {
                "max_agents": self.max_agents,
                "performance_threshold": self.performance_threshold,
                "reconfiguration_cooldown": self.reconfiguration_cooldown
            },
            "graph_state": self.get_graph_state(),
            "reconfiguration_history": [
                {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "trigger": event.trigger,
                    "action": event.action,
                    "affected_agents": event.affected_agents,
                    "success": event.success,
                    "metadata": event.metadata
                }
                for event in self.reconfiguration_history
            ]
        }