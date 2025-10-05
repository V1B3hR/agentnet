"""Runtime enforcement and monitoring integration for risk register."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from . import RiskCategory, RiskEvent, RiskLevel, RiskRegister

logger = logging.getLogger("agentnet.risk.enforcement")


@dataclass
class EnforcementRule:
    """Runtime enforcement rule based on risk events."""
    rule_id: str
    risk_type: str
    trigger_threshold: int  # Number of events before enforcement
    time_window_minutes: int  # Time window to count events
    enforcement_action: str  # Action to take (e.g., 'block', 'throttle', 'alert')
    severity: RiskLevel
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnforcementAction:
    """Record of an enforcement action taken."""
    action_id: str
    rule_id: str
    risk_type: str
    action_type: str
    executed_at: datetime = field(default_factory=datetime.now)
    target: Optional[str] = None  # Agent, session, or tenant affected
    duration_minutes: Optional[int] = None  # How long the action is in effect
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskEnforcementEngine:
    """Engine for runtime enforcement based on risk events."""
    
    def __init__(self, risk_register: RiskRegister):
        """Initialize enforcement engine.
        
        Args:
            risk_register: Risk register to monitor
        """
        self.risk_register = risk_register
        self._rules: Dict[str, EnforcementRule] = {}
        self._enforcement_history: List[EnforcementAction] = []
        self._blocked_targets: Dict[str, datetime] = {}  # target -> blocked_until
        self._throttled_targets: Dict[str, Dict[str, Any]] = {}  # target -> config
        
        # Callbacks for enforcement actions
        self._action_callbacks: Dict[str, List[Callable]] = {
            "block": [],
            "throttle": [],
            "alert": [],
            "downgrade": [],
        }
        
        logger.info("RiskEnforcementEngine initialized")
    
    def add_enforcement_rule(self, rule: EnforcementRule) -> None:
        """Add an enforcement rule.
        
        Args:
            rule: Enforcement rule to add
        """
        self._rules[rule.rule_id] = rule
        logger.info(f"Added enforcement rule: {rule.rule_id} for risk type: {rule.risk_type}")
    
    def remove_enforcement_rule(self, rule_id: str) -> bool:
        """Remove an enforcement rule.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if removed successfully
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed enforcement rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable an enforcement rule.
        
        Args:
            rule_id: ID of rule to enable
            
        Returns:
            True if enabled successfully
        """
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            logger.info(f"Enabled enforcement rule: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable an enforcement rule.
        
        Args:
            rule_id: ID of rule to disable
            
        Returns:
            True if disabled successfully
        """
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            logger.info(f"Disabled enforcement rule: {rule_id}")
            return True
        return False
    
    def check_and_enforce(
        self,
        risk_event: RiskEvent,
        target: Optional[str] = None
    ) -> Optional[EnforcementAction]:
        """Check if enforcement action is needed for a risk event.
        
        Args:
            risk_event: Risk event to check
            target: Target identifier (agent, session, or tenant)
            
        Returns:
            Enforcement action if triggered, None otherwise
        """
        # Find matching rules
        matching_rules = [
            rule for rule in self._rules.values()
            if rule.enabled and self._matches_risk_event(rule, risk_event)
        ]
        
        if not matching_rules:
            return None
        
        # Check each rule
        for rule in matching_rules:
            if self._should_enforce(rule, risk_event):
                action = self._execute_enforcement(rule, risk_event, target)
                return action
        
        return None
    
    def _matches_risk_event(self, rule: EnforcementRule, event: RiskEvent) -> bool:
        """Check if a rule matches a risk event.
        
        Args:
            rule: Enforcement rule
            event: Risk event
            
        Returns:
            True if matches
        """
        # Check risk type match
        if rule.risk_type != "*" and rule.risk_type != event.metadata.get("risk_type", ""):
            return False
        
        # Check severity level
        if event.level.value != rule.severity.value:
            # Could implement level hierarchy here
            pass
        
        return True
    
    def _should_enforce(self, rule: EnforcementRule, event: RiskEvent) -> bool:
        """Check if enforcement should be triggered.
        
        Args:
            rule: Enforcement rule
            event: Risk event
            
        Returns:
            True if should enforce
        """
        # Get recent events of this type
        cutoff_time = datetime.now().timestamp() - (rule.time_window_minutes * 60)
        
        # Count recent events
        # In a real implementation, query the risk register
        # For now, trigger based on severity
        if event.level == RiskLevel.CRITICAL:
            return True
        
        # Could implement more sophisticated logic here
        return False
    
    def _execute_enforcement(
        self,
        rule: EnforcementRule,
        event: RiskEvent,
        target: Optional[str]
    ) -> EnforcementAction:
        """Execute an enforcement action.
        
        Args:
            rule: Enforcement rule that triggered
            event: Risk event that triggered the rule
            target: Target of the enforcement
            
        Returns:
            Enforcement action record
        """
        action_id = f"action_{datetime.now().timestamp()}"
        
        action = EnforcementAction(
            action_id=action_id,
            rule_id=rule.rule_id,
            risk_type=rule.risk_type,
            action_type=rule.enforcement_action,
            target=target or event.agent_name or event.session_id or event.tenant_id,
            metadata={
                "event_id": event.risk_id,
                "rule_metadata": rule.metadata,
                "event_metadata": event.metadata,
            }
        )
        
        # Execute the specific action
        if rule.enforcement_action == "block":
            self._block_target(action)
        elif rule.enforcement_action == "throttle":
            self._throttle_target(action)
        elif rule.enforcement_action == "alert":
            self._send_alert(action)
        elif rule.enforcement_action == "downgrade":
            self._downgrade_service(action)
        
        self._enforcement_history.append(action)
        logger.warning(
            f"Enforcement action executed: {action.action_type} for {action.target} "
            f"due to risk: {event.title}"
        )
        
        # Trigger callbacks
        for callback in self._action_callbacks.get(rule.enforcement_action, []):
            try:
                callback(action, event)
            except Exception as e:
                logger.error(f"Error in enforcement callback: {e}")
        
        return action
    
    def _block_target(self, action: EnforcementAction) -> None:
        """Block a target from using the system.
        
        Args:
            action: Enforcement action
        """
        if action.target:
            # Block for 1 hour by default
            duration = action.duration_minutes or 60
            blocked_until = datetime.now().timestamp() + (duration * 60)
            self._blocked_targets[action.target] = datetime.fromtimestamp(blocked_until)
            logger.warning(f"Blocked target: {action.target} for {duration} minutes")
    
    def _throttle_target(self, action: EnforcementAction) -> None:
        """Throttle a target's usage.
        
        Args:
            action: Enforcement action
        """
        if action.target:
            # Reduce rate limit by 50% by default
            self._throttled_targets[action.target] = {
                "rate_multiplier": 0.5,
                "until": datetime.now().timestamp() + (action.duration_minutes or 30) * 60
            }
            logger.warning(f"Throttled target: {action.target}")
    
    def _send_alert(self, action: EnforcementAction) -> None:
        """Send alert for a risk event.
        
        Args:
            action: Enforcement action
        """
        # In a real implementation, send to alerting system
        logger.error(f"ALERT: {action.risk_type} triggered for {action.target}")
    
    def _downgrade_service(self, action: EnforcementAction) -> None:
        """Downgrade service level for a target.
        
        Args:
            action: Enforcement action
        """
        # In a real implementation, downgrade to cheaper/simpler service
        logger.warning(f"Downgrading service for target: {action.target}")
    
    def is_blocked(self, target: str) -> bool:
        """Check if a target is currently blocked.
        
        Args:
            target: Target identifier
            
        Returns:
            True if blocked
        """
        if target not in self._blocked_targets:
            return False
        
        blocked_until = self._blocked_targets[target]
        if datetime.now() > blocked_until:
            # Block has expired
            del self._blocked_targets[target]
            return False
        
        return True
    
    def is_throttled(self, target: str) -> tuple[bool, float]:
        """Check if a target is currently throttled.
        
        Args:
            target: Target identifier
            
        Returns:
            Tuple of (is_throttled, rate_multiplier)
        """
        if target not in self._throttled_targets:
            return False, 1.0
        
        throttle_info = self._throttled_targets[target]
        if datetime.now().timestamp() > throttle_info["until"]:
            # Throttle has expired
            del self._throttled_targets[target]
            return False, 1.0
        
        return True, throttle_info["rate_multiplier"]
    
    def register_action_callback(
        self,
        action_type: str,
        callback: Callable[[EnforcementAction, RiskEvent], None]
    ) -> None:
        """Register a callback for enforcement actions.
        
        Args:
            action_type: Type of action (block, throttle, alert, downgrade)
            callback: Callback function
        """
        if action_type not in self._action_callbacks:
            self._action_callbacks[action_type] = []
        
        self._action_callbacks[action_type].append(callback)
        logger.info(f"Registered callback for action type: {action_type}")
    
    def get_enforcement_history(
        self,
        target: Optional[str] = None,
        limit: int = 100
    ) -> List[EnforcementAction]:
        """Get enforcement action history.
        
        Args:
            target: Filter by target
            limit: Maximum number of actions to return
            
        Returns:
            List of enforcement actions
        """
        history = self._enforcement_history
        
        if target:
            history = [a for a in history if a.target == target]
        
        return history[-limit:]
    
    def get_active_rules(self) -> List[EnforcementRule]:
        """Get all active enforcement rules.
        
        Returns:
            List of enabled rules
        """
        return [rule for rule in self._rules.values() if rule.enabled]
    
    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get enforcement statistics.
        
        Returns:
            Dictionary with enforcement stats
        """
        total_actions = len(self._enforcement_history)
        action_type_counts = {}
        
        for action in self._enforcement_history:
            action_type = action.action_type
            action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
        
        return {
            "total_enforcement_actions": total_actions,
            "actions_by_type": action_type_counts,
            "currently_blocked": len(self._blocked_targets),
            "currently_throttled": len(self._throttled_targets),
            "active_rules": len(self.get_active_rules()),
            "total_rules": len(self._rules),
        }


# Default enforcement rules factory
def create_default_enforcement_rules() -> List[EnforcementRule]:
    """Create default enforcement rules.
    
    Returns:
        List of default enforcement rules
    """
    return [
        EnforcementRule(
            rule_id="cost_spike_enforcement",
            risk_type="token_cost_spike",
            trigger_threshold=3,
            time_window_minutes=60,
            enforcement_action="downgrade",
            severity=RiskLevel.HIGH,
            metadata={"description": "Downgrade to cheaper model on cost spike"}
        ),
        EnforcementRule(
            rule_id="provider_outage_enforcement",
            risk_type="provider_outage",
            trigger_threshold=1,
            time_window_minutes=5,
            enforcement_action="alert",
            severity=RiskLevel.CRITICAL,
            metadata={"description": "Alert on provider outage"}
        ),
        EnforcementRule(
            rule_id="memory_bloat_enforcement",
            risk_type="memory_bloat",
            trigger_threshold=2,
            time_window_minutes=30,
            enforcement_action="throttle",
            severity=RiskLevel.MEDIUM,
            metadata={"description": "Throttle on memory bloat"}
        ),
        EnforcementRule(
            rule_id="policy_violation_enforcement",
            risk_type="policy_false_positives",
            trigger_threshold=5,
            time_window_minutes=60,
            enforcement_action="alert",
            severity=RiskLevel.MEDIUM,
            metadata={"description": "Alert on high policy violation rate"}
        ),
    ]
