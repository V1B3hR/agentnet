"""Risk mitigation strategies and implementation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .registry import Risk, RiskCategory, RiskStatus

logger = logging.getLogger("agentnet.risk.mitigation")


class MitigationStatus(Enum):
    """Status of mitigation actions."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MitigationType(Enum):
    """Types of mitigation strategies."""
    PREVENTIVE = "preventive"  # Prevent risk from occurring
    DETECTIVE = "detective"    # Detect when risk occurs
    CORRECTIVE = "corrective"  # Correct after risk occurs
    COMPENSATING = "compensating"  # Compensate for risk impact


@dataclass
class MitigationAction:
    """Individual mitigation action."""
    
    action_id: str
    title: str
    description: str
    mitigation_type: MitigationType
    priority: int  # 1-10, 1 being highest
    
    # Implementation details
    implementation_steps: List[str]
    required_resources: List[str]
    estimated_effort_hours: float
    estimated_cost: float
    
    # Tracking
    status: MitigationStatus
    assigned_to: str
    created_date: datetime
    
    # Effectiveness
    expected_risk_reduction: float  # 0.0 to 1.0
    
    # Optional fields with defaults
    target_completion_date: Optional[datetime] = None
    actual_completion_date: Optional[datetime] = None
    actual_risk_reduction: Optional[float] = None
    
    # Automation
    is_automated: bool = False
    automation_trigger: Optional[str] = None
    automation_script: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Other action IDs
    blocks: List[str] = field(default_factory=list)      # Actions blocked by this
    
    # Metadata
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MitigationStrategy:
    """Collection of related mitigation actions for a risk."""
    
    strategy_id: str
    risk_id: str
    name: str
    description: str
    
    # Actions in this strategy
    actions: List[MitigationAction] = field(default_factory=list)
    
    # Strategy metadata
    created_by: str = "system"
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Effectiveness tracking
    overall_risk_reduction: float = 0.0
    implementation_progress: float = 0.0  # 0.0 to 1.0
    
    # Status
    is_active: bool = True
    is_approved: bool = False
    approved_by: Optional[str] = None
    
    def calculate_progress(self) -> float:
        """Calculate overall implementation progress."""
        if not self.actions:
            return 0.0
        
        completed_actions = len([a for a in self.actions if a.status == MitigationStatus.COMPLETED])
        in_progress_actions = len([a for a in self.actions if a.status == MitigationStatus.IN_PROGRESS])
        
        # Weight in-progress actions as 50% complete
        progress = (completed_actions + (in_progress_actions * 0.5)) / len(self.actions)
        self.implementation_progress = progress
        return progress
    
    def calculate_risk_reduction(self) -> float:
        """Calculate overall expected risk reduction from completed actions."""
        if not self.actions:
            return 0.0
        
        # Sum risk reduction from completed actions
        total_reduction = 0.0
        for action in self.actions:
            if action.status == MitigationStatus.COMPLETED:
                reduction = action.actual_risk_reduction or action.expected_risk_reduction
                total_reduction += reduction
        
        # Cap at 100% reduction
        self.overall_risk_reduction = min(1.0, total_reduction)
        return self.overall_risk_reduction


class MitigationLibrary:
    """Library of predefined mitigation strategies for different risk types."""
    
    def __init__(self):
        self.strategies = self._build_default_strategies()
    
    def _build_default_strategies(self) -> Dict[RiskCategory, List[Dict[str, Any]]]:
        """Build default mitigation strategies for each risk category."""
        
        return {
            RiskCategory.PROVIDER_OUTAGE: [
                {
                    "name": "Multi-Provider Failover",
                    "description": "Implement multiple provider support with automatic failover",
                    "actions": [
                        {
                            "title": "Implement Provider Health Checks",
                            "description": "Add health monitoring for all providers",
                            "type": MitigationType.DETECTIVE,
                            "priority": 1,
                            "steps": [
                                "Add health check endpoints for each provider",
                                "Implement periodic health monitoring",
                                "Set up alerting for provider failures",
                            ],
                            "effort_hours": 16,
                            "cost": 2000,
                            "risk_reduction": 0.3,
                        },
                        {
                            "title": "Implement Circuit Breaker Pattern",
                            "description": "Add circuit breakers to prevent cascade failures",
                            "type": MitigationType.PREVENTIVE,
                            "priority": 2,
                            "steps": [
                                "Implement circuit breaker for provider calls",
                                "Configure failure thresholds and timeouts",
                                "Add circuit breaker state monitoring",
                            ],
                            "effort_hours": 20,
                            "cost": 2500,
                            "risk_reduction": 0.4,
                        },
                        {
                            "title": "Configure Secondary Provider",
                            "description": "Set up automatic failover to secondary provider",
                            "type": MitigationType.CORRECTIVE,
                            "priority": 1,
                            "steps": [
                                "Integrate secondary provider API",
                                "Implement provider selection logic",
                                "Test failover scenarios",
                            ],
                            "effort_hours": 32,
                            "cost": 5000,
                            "risk_reduction": 0.6,
                        },
                    ],
                },
                {
                    "name": "Graceful Degradation",
                    "description": "Maintain limited functionality during provider outages",
                    "actions": [
                        {
                            "title": "Implement Offline Mode",
                            "description": "Provide basic functionality when providers unavailable",
                            "type": MitigationType.COMPENSATING,
                            "priority": 3,
                            "steps": [
                                "Design offline response templates",
                                "Implement local processing fallback",
                                "Add user notification for degraded mode",
                            ],
                            "effort_hours": 24,
                            "cost": 3000,
                            "risk_reduction": 0.3,
                        },
                    ],
                },
            ],
            
            RiskCategory.COST_SPIKE: [
                {
                    "name": "Cost Monitoring and Alerts",
                    "description": "Proactive cost monitoring with automated alerts and controls",
                    "actions": [
                        {
                            "title": "Implement Budget Alerts",
                            "description": "Set up automated budget monitoring and alerts",
                            "type": MitigationType.DETECTIVE,
                            "priority": 1,
                            "steps": [
                                "Configure budget thresholds (75%, 90%, 100%)",
                                "Set up email and Slack notifications", 
                                "Create cost dashboard for monitoring",
                            ],
                            "effort_hours": 8,
                            "cost": 1000,
                            "risk_reduction": 0.5,
                        },
                        {
                            "title": "Implement Spend Rate Limiting",
                            "description": "Automatically limit spend when thresholds exceeded",
                            "type": MitigationType.PREVENTIVE,
                            "priority": 2,
                            "steps": [
                                "Implement per-tenant spend limits",
                                "Add automatic request throttling",
                                "Configure emergency stop mechanisms",
                            ],
                            "effort_hours": 16,
                            "cost": 2000,
                            "risk_reduction": 0.7,
                        },
                        {
                            "title": "Model Optimization Engine",
                            "description": "Automatically select most cost-effective models",
                            "type": MitigationType.CORRECTIVE,
                            "priority": 3,
                            "steps": [
                                "Implement model performance vs cost analysis",
                                "Create automatic model selection logic",
                                "Add cost optimization recommendations",
                            ],
                            "effort_hours": 40,
                            "cost": 6000,
                            "risk_reduction": 0.4,
                        },
                    ],
                },
            ],
            
            RiskCategory.TOOL_INJECTION: [
                {
                    "name": "Input Validation and Sandboxing",
                    "description": "Comprehensive security controls for tool execution",
                    "actions": [
                        {
                            "title": "Implement Input Schema Validation",
                            "description": "Validate all tool inputs against strict schemas",
                            "type": MitigationType.PREVENTIVE,
                            "priority": 1,
                            "steps": [
                                "Define JSON schemas for all tool inputs",
                                "Implement schema validation middleware",
                                "Add input sanitization functions",
                            ],
                            "effort_hours": 20,
                            "cost": 2500,
                            "risk_reduction": 0.6,
                        },
                        {
                            "title": "Deploy Tool Execution Sandbox",
                            "description": "Isolate tool execution in secure containers",
                            "type": MitigationType.PREVENTIVE,
                            "priority": 1,
                            "steps": [
                                "Set up Docker containers for tool execution",
                                "Configure resource limits and network isolation",
                                "Implement execution monitoring and logging",
                            ],
                            "effort_hours": 32,
                            "cost": 4000,
                            "risk_reduction": 0.8,
                        },
                        {
                            "title": "Implement Audit Logging",
                            "description": "Comprehensive logging of all tool executions",
                            "type": MitigationType.DETECTIVE,
                            "priority": 2,
                            "steps": [
                                "Log all tool execution requests and responses",
                                "Implement anomaly detection for suspicious patterns",
                                "Set up security alert triggers",
                            ],
                            "effort_hours": 12,
                            "cost": 1500,
                            "risk_reduction": 0.3,
                        },
                    ],
                },
            ],
            
            RiskCategory.MEMORY_BLOAT: [
                {
                    "name": "Memory Management Controls",
                    "description": "Proactive memory usage monitoring and control",
                    "actions": [
                        {
                            "title": "Implement Memory Monitoring",
                            "description": "Real-time memory usage tracking and alerts",
                            "type": MitigationType.DETECTIVE,
                            "priority": 1,
                            "steps": [
                                "Add memory usage metrics collection",
                                "Set up memory threshold alerts",
                                "Create memory usage dashboards",
                            ],
                            "effort_hours": 8,
                            "cost": 1000,
                            "risk_reduction": 0.3,
                        },
                        {
                            "title": "Implement Context Pruning",
                            "description": "Automatically trim conversation context when limits reached",
                            "type": MitigationType.CORRECTIVE,
                            "priority": 2,
                            "steps": [
                                "Implement context size monitoring",
                                "Add intelligent context summarization",
                                "Configure automatic pruning triggers",
                            ],
                            "effort_hours": 24,
                            "cost": 3000,
                            "risk_reduction": 0.6,
                        },
                        {
                            "title": "Session Rotation System",
                            "description": "Automatically rotate long-running sessions",
                            "type": MitigationType.PREVENTIVE,
                            "priority": 3,
                            "steps": [
                                "Implement session duration tracking",
                                "Add graceful session termination",
                                "Create session state persistence",
                            ],
                            "effort_hours": 20,
                            "cost": 2500,
                            "risk_reduction": 0.5,
                        },
                    ],
                },
            ],
            
            RiskCategory.CONVERGENCE_STALL: [
                {
                    "name": "Reasoning Control Systems", 
                    "description": "Prevent infinite reasoning loops and stagnation",
                    "actions": [
                        {
                            "title": "Implement Stagnation Detection",
                            "description": "Detect when reasoning is not making progress",
                            "type": MitigationType.DETECTIVE,
                            "priority": 1,
                            "steps": [
                                "Track reasoning progress metrics",
                                "Implement circular logic detection", 
                                "Add stagnation alert triggers",
                            ],
                            "effort_hours": 16,
                            "cost": 2000,
                            "risk_reduction": 0.5,
                        },
                        {
                            "title": "Configure Hard Reasoning Limits",
                            "description": "Enforce maximum iterations and time limits",
                            "type": MitigationType.PREVENTIVE,
                            "priority": 1,
                            "steps": [
                                "Set maximum reasoning iterations per task",
                                "Implement wall-clock time limits",
                                "Add graceful termination handling",
                            ],
                            "effort_hours": 12,
                            "cost": 1500,
                            "risk_reduction": 0.7,
                        },
                    ],
                },
            ],
        }
    
    def get_strategies_for_risk(self, risk_category: RiskCategory) -> List[Dict[str, Any]]:
        """Get predefined mitigation strategies for a risk category."""
        return self.strategies.get(risk_category, [])


class MitigationMitigator:
    """Main mitigation management system."""
    
    def __init__(self):
        self.library = MitigationLibrary()
        self.active_strategies: Dict[str, MitigationStrategy] = {}
        self.action_handlers: Dict[str, Callable] = {}
    
    def create_strategy_for_risk(
        self,
        risk: Risk,
        strategy_name: Optional[str] = None,
        custom_actions: Optional[List[Dict[str, Any]]] = None,
        created_by: str = "system",
    ) -> MitigationStrategy:
        """Create a mitigation strategy for a risk."""
        
        strategy_id = f"MITIGATION_{risk.risk_id}_{int(datetime.now().timestamp())}"
        
        if strategy_name is None:
            predefined_strategies = self.library.get_strategies_for_risk(risk.category)
            if predefined_strategies:
                # Use first predefined strategy as default
                template = predefined_strategies[0]
                strategy_name = template["name"]
                actions_data = template["actions"]
            else:
                strategy_name = f"Custom Strategy for {risk.title}"
                actions_data = custom_actions or []
        else:
            actions_data = custom_actions or []
        
        # Create mitigation actions
        actions = []
        for i, action_data in enumerate(actions_data):
            action_id = f"{strategy_id}_ACTION_{i+1}"
            
            action = MitigationAction(
                action_id=action_id,
                title=action_data["title"],
                description=action_data["description"],
                mitigation_type=action_data.get("type", MitigationType.CORRECTIVE),
                priority=action_data.get("priority", 5),
                implementation_steps=action_data.get("steps", []),
                required_resources=action_data.get("resources", []),
                estimated_effort_hours=action_data.get("effort_hours", 8),
                estimated_cost=action_data.get("cost", 1000),
                status=MitigationStatus.PLANNED,
                assigned_to=action_data.get("assigned_to", "unassigned"),
                created_date=datetime.now(),
                expected_risk_reduction=action_data.get("risk_reduction", 0.2),
                is_automated=action_data.get("is_automated", False),
                automation_trigger=action_data.get("automation_trigger"),
                automation_script=action_data.get("automation_script"),
                depends_on=action_data.get("depends_on", []),
                notes=action_data.get("notes", ""),
            )
            actions.append(action)
        
        # Create strategy
        strategy = MitigationStrategy(
            strategy_id=strategy_id,
            risk_id=risk.risk_id,
            name=strategy_name,
            description=f"Mitigation strategy for {risk.title}",
            actions=actions,
            created_by=created_by,
            created_date=datetime.now(),
            last_updated=datetime.now(),
        )
        
        self.active_strategies[strategy_id] = strategy
        logger.info(f"Created mitigation strategy {strategy_id} for risk {risk.risk_id}")
        
        return strategy
    
    def execute_action(self, action_id: str, executor: str = "system") -> bool:
        """Execute a mitigation action."""
        
        # Find the action
        action = None
        strategy = None
        for strat in self.active_strategies.values():
            for act in strat.actions:
                if act.action_id == action_id:
                    action = act
                    strategy = strat
                    break
            if action:
                break
        
        if not action:
            raise ValueError(f"Action {action_id} not found")
        
        # Check dependencies
        for dep_id in action.depends_on:
            dep_action = None
            for strat in self.active_strategies.values():
                for act in strat.actions:
                    if act.action_id == dep_id:
                        dep_action = act
                        break
                if dep_action:
                    break
            
            if not dep_action or dep_action.status != MitigationStatus.COMPLETED:
                logger.warning(f"Cannot execute {action_id} - dependency {dep_id} not completed")
                return False
        
        try:
            # Update status to in-progress
            action.status = MitigationStatus.IN_PROGRESS
            strategy.last_updated = datetime.now()
            
            # Execute the action
            if action.is_automated and action.automation_script:
                # Execute automated action
                success = self._execute_automated_action(action)
            else:
                # Manual action - just mark as completed for demo
                logger.info(f"Manual action {action_id} marked for execution by {executor}")
                success = True
            
            if success:
                action.status = MitigationStatus.COMPLETED
                action.actual_completion_date = datetime.now()
                action.actual_risk_reduction = action.expected_risk_reduction
                logger.info(f"Successfully executed action {action_id}")
            else:
                action.status = MitigationStatus.FAILED
                logger.error(f"Failed to execute action {action_id}")
            
            # Update strategy progress
            strategy.calculate_progress()
            strategy.calculate_risk_reduction()
            
            return success
            
        except Exception as e:
            action.status = MitigationStatus.FAILED
            logger.error(f"Error executing action {action_id}: {e}")
            return False
    
    def _execute_automated_action(self, action: MitigationAction) -> bool:
        """Execute an automated mitigation action."""
        
        # In a real implementation, this would execute the automation script
        # For now, simulate success
        logger.info(f"Simulating automated execution of {action.action_id}")
        
        # Check if we have a handler for this action type
        handler = self.action_handlers.get(action.automation_trigger)
        if handler:
            return handler(action)
        
        # Default simulation - assume success
        return True
    
    def register_action_handler(self, trigger: str, handler: Callable) -> None:
        """Register a handler for automated actions."""
        self.action_handlers[trigger] = handler
        logger.info(f"Registered handler for trigger: {trigger}")
    
    def get_strategy(self, strategy_id: str) -> Optional[MitigationStrategy]:
        """Get a mitigation strategy by ID."""
        return self.active_strategies.get(strategy_id)
    
    def get_strategies_for_risk(self, risk_id: str) -> List[MitigationStrategy]:
        """Get all mitigation strategies for a risk."""
        return [s for s in self.active_strategies.values() if s.risk_id == risk_id]
    
    def get_pending_actions(self, assigned_to: Optional[str] = None) -> List[MitigationAction]:
        """Get all pending actions, optionally filtered by assignee."""
        pending = []
        for strategy in self.active_strategies.values():
            for action in strategy.actions:
                if action.status in {MitigationStatus.PLANNED, MitigationStatus.IN_PROGRESS}:
                    if assigned_to is None or action.assigned_to == assigned_to:
                        pending.append(action)
        return pending
    
    def get_overdue_actions(self) -> List[MitigationAction]:
        """Get actions that are past their target completion date."""
        now = datetime.now()
        overdue = []
        
        for strategy in self.active_strategies.values():
            for action in strategy.actions:
                if (action.target_completion_date and 
                    action.target_completion_date < now and
                    action.status not in {MitigationStatus.COMPLETED, MitigationStatus.CANCELLED}):
                    overdue.append(action)
        
        return overdue
    
    def get_mitigation_summary(self) -> Dict[str, Any]:
        """Get summary of all mitigation activities."""
        
        total_strategies = len(self.active_strategies)
        total_actions = sum(len(s.actions) for s in self.active_strategies.values())
        
        # Count actions by status
        status_counts = {status.value: 0 for status in MitigationStatus}
        for strategy in self.active_strategies.values():
            for action in strategy.actions:
                status_counts[action.status.value] += 1
        
        # Calculate overall progress
        completed_actions = status_counts[MitigationStatus.COMPLETED.value]
        overall_progress = (completed_actions / total_actions * 100) if total_actions > 0 else 0
        
        # Calculate total investment
        total_estimated_cost = sum(
            sum(action.estimated_cost for action in strategy.actions)
            for strategy in self.active_strategies.values()
        )
        
        total_estimated_effort = sum(
            sum(action.estimated_effort_hours for action in strategy.actions)
            for strategy in self.active_strategies.values()
        )
        
        return {
            "total_strategies": total_strategies,
            "total_actions": total_actions,
            "overall_progress_percent": round(overall_progress, 1),
            "action_status_counts": status_counts,
            "pending_actions": len(self.get_pending_actions()),
            "overdue_actions": len(self.get_overdue_actions()),
            "total_estimated_cost": total_estimated_cost,
            "total_estimated_effort_hours": total_estimated_effort,
        }