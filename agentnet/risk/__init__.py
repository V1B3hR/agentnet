"""Risk register and management system for AgentNet."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("agentnet.risk")


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskStatus(Enum):
    """Risk status states."""
    OPEN = "open"
    MITIGATING = "mitigating"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    CLOSED = "closed"


class RiskCategory(Enum):
    """Risk categories."""
    OPERATIONAL = "operational"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    TECHNICAL = "technical"


@dataclass
class RiskEvent:
    """Individual risk event record."""
    risk_id: str
    category: RiskCategory
    level: RiskLevel
    title: str
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMitigation:
    """Risk mitigation action."""
    mitigation_id: str
    risk_id: str
    action: str
    implemented_at: datetime = field(default_factory=datetime.now)
    effectiveness: Optional[float] = None  # 0.0 to 1.0
    automated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDefinition:
    """Risk definition from the register."""
    risk_type: str
    category: RiskCategory
    default_level: RiskLevel
    description: str
    impact_description: str
    mitigation_strategy: str
    detection_rules: List[str] = field(default_factory=list)
    automated_mitigation: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


class RiskRegister:
    """Risk register and automated risk management system."""
    
    def __init__(self, storage_dir: str = "risk_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "events").mkdir(exist_ok=True)
        (self.storage_dir / "mitigations").mkdir(exist_ok=True)
        (self.storage_dir / "definitions").mkdir(exist_ok=True)
        
        logger.info(f"RiskRegister initialized with storage_dir: {storage_dir}")
        
        # Risk detection handlers
        self._detection_handlers: Dict[str, Callable] = {}
        self._mitigation_handlers: Dict[str, Callable] = {}
        
        # In-memory caches
        self._risk_cache: Dict[str, RiskEvent] = {}
        self._definitions_cache: Dict[str, RiskDefinition] = {}
        
        # Initialize with default risk definitions from the roadmap
        self._initialize_default_risks()
        
    def register_risk_event(
        self,
        risk_type: str,
        level: Optional[RiskLevel] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskEvent:
        """Register a new risk event."""
        
        # Get risk definition
        risk_def = self.get_risk_definition(risk_type)
        if not risk_def:
            # Create dynamic risk definition
            risk_def = RiskDefinition(
                risk_type=risk_type,
                category=RiskCategory.OPERATIONAL,
                default_level=level or RiskLevel.MEDIUM,
                description=description or f"Dynamic risk: {risk_type}",
                impact_description="Unknown impact",
                mitigation_strategy="Manual review required"
            )
            self._save_risk_definition(risk_def)
        
        # Create risk event
        risk_id = f"{risk_type}_{int(datetime.now().timestamp())}_{agent_name or 'unknown'}"
        
        risk_event = RiskEvent(
            risk_id=risk_id,
            category=risk_def.category,
            level=level or risk_def.default_level,
            title=title or f"{risk_def.description}",
            description=description or risk_def.description,
            agent_name=agent_name,
            session_id=session_id,
            tenant_id=tenant_id,
            metadata={
                "risk_type": risk_type,
                **(metadata or {})
            }
        )
        
        # Save risk event
        self._save_risk_event(risk_event)
        self._risk_cache[risk_id] = risk_event
        
        logger.warning(f"Risk event registered: {risk_id} - Level: {risk_event.level.value}")
        
        # Trigger automated mitigation if configured
        if risk_def.automated_mitigation:
            self._trigger_automated_mitigation(risk_event, risk_def)
        
        return risk_event
    
    def log_cost_spike_risk(
        self,
        current_cost: float,
        threshold: float,
        agent_name: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> RiskEvent:
        """Log a cost spike risk event."""
        return self.register_risk_event(
            risk_type="token_cost_spike",
            level=RiskLevel.HIGH if current_cost > threshold * 2 else RiskLevel.MEDIUM,
            title=f"Token cost spike detected",
            description=f"Cost ${current_cost:.4f} exceeds threshold ${threshold:.4f}",
            agent_name=agent_name,
            tenant_id=tenant_id,
            metadata={
                "current_cost": current_cost,
                "threshold": threshold,
                "spike_ratio": current_cost / threshold if threshold > 0 else 0
            }
        )
    
    def log_memory_bloat_risk(
        self,
        memory_usage: int,
        threshold: int,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> RiskEvent:
        """Log a memory bloat risk event.""" 
        return self.register_risk_event(
            risk_type="memory_bloat",
            level=RiskLevel.MEDIUM,
            title="Memory usage above threshold",
            description=f"Memory usage {memory_usage} MB exceeds threshold {threshold} MB",
            agent_name=agent_name,
            session_id=session_id,
            metadata={
                "memory_usage_mb": memory_usage,
                "threshold_mb": threshold,
                "usage_ratio": memory_usage / threshold if threshold > 0 else 0
            }
        )
    
    def log_convergence_stall_risk(
        self,
        session_duration: timedelta,
        max_duration: timedelta,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> RiskEvent:
        """Log a convergence stall risk event."""
        return self.register_risk_event(
            risk_type="convergence_stall",
            level=RiskLevel.HIGH,
            title="Session convergence stall detected",
            description=f"Session duration {session_duration} exceeds maximum {max_duration}",
            agent_name=agent_name,
            session_id=session_id,
            metadata={
                "session_duration_seconds": session_duration.total_seconds(),
                "max_duration_seconds": max_duration.total_seconds(),
                "stall_detected": True
            }
        )
    
    def log_provider_outage_risk(
        self,
        provider: str,
        error_rate: float,
        agent_name: Optional[str] = None
    ) -> RiskEvent:
        """Log a provider outage risk event."""
        level = RiskLevel.CRITICAL if error_rate > 0.8 else RiskLevel.HIGH if error_rate > 0.5 else RiskLevel.MEDIUM
        
        return self.register_risk_event(
            risk_type="provider_outage",
            level=level,
            title=f"{provider} provider degraded service",
            description=f"Provider {provider} showing {error_rate:.1%} error rate",
            agent_name=agent_name,
            metadata={
                "provider": provider,
                "error_rate": error_rate,
                "service_degraded": True
            }
        )
    
    def log_policy_false_positive_risk(
        self,
        policy_name: str,
        false_positive_rate: float,
        agent_name: Optional[str] = None
    ) -> RiskEvent:
        """Log a policy false positive risk event."""
        return self.register_risk_event(
            risk_type="policy_false_positives",
            level=RiskLevel.MEDIUM,
            title=f"High false positive rate in {policy_name}",
            description=f"Policy {policy_name} showing {false_positive_rate:.1%} false positive rate",
            agent_name=agent_name,
            metadata={
                "policy_name": policy_name,
                "false_positive_rate": false_positive_rate,
                "requires_tuning": True
            }
        )
    
    def mitigate_risk(
        self,
        risk_id: str,
        mitigation_action: str,
        automated: bool = False,
        effectiveness: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskMitigation:
        """Record a risk mitigation action."""
        
        mitigation_id = f"mitigation_{risk_id}_{int(datetime.now().timestamp())}"
        
        mitigation = RiskMitigation(
            mitigation_id=mitigation_id,
            risk_id=risk_id,
            action=mitigation_action,
            effectiveness=effectiveness,
            automated=automated,
            metadata=metadata or {}
        )
        
        # Save mitigation
        self._save_mitigation(mitigation)
        
        logger.info(f"Risk mitigation recorded: {mitigation_id} for risk {risk_id}")
        return mitigation
    
    def get_risk_summary(
        self,
        tenant_id: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get risk summary for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        events = self._load_recent_events(cutoff_date, tenant_id)
        
        # Categorize risks
        risk_counts = {
            "by_level": {level.value: 0 for level in RiskLevel},
            "by_category": {cat.value: 0 for cat in RiskCategory},
            "by_type": {},
            "total": len(events)
        }
        
        for event in events:
            risk_counts["by_level"][event.level.value] += 1
            risk_counts["by_category"][event.category.value] += 1
            
            risk_type = event.metadata.get("risk_type", "unknown")
            risk_counts["by_type"][risk_type] = risk_counts["by_type"].get(risk_type, 0) + 1
        
        # Get top risks
        top_risks = sorted(
            risk_counts["by_type"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "period": f"Last {days_back} days",
            "summary": risk_counts,
            "top_risks": top_risks,
            "high_priority_count": risk_counts["by_level"]["high"] + risk_counts["by_level"]["critical"],
            "recommendation": self._generate_risk_recommendations(events)
        }
    
    def get_risk_definition(self, risk_type: str) -> Optional[RiskDefinition]:
        """Get risk definition by type."""
        if risk_type in self._definitions_cache:
            return self._definitions_cache[risk_type]
        
        # Load from storage
        def_file = self.storage_dir / "definitions" / f"{risk_type}.json"
        if def_file.exists():
            try:
                with open(def_file) as f:
                    data = json.load(f)
                    risk_def = RiskDefinition(
                        risk_type=data["risk_type"],
                        category=RiskCategory(data["category"]),
                        default_level=RiskLevel(data["default_level"]),
                        description=data["description"],
                        impact_description=data["impact_description"],
                        mitigation_strategy=data["mitigation_strategy"],
                        detection_rules=data.get("detection_rules", []),
                        automated_mitigation=data.get("automated_mitigation", False),
                        mitigation_actions=data.get("mitigation_actions", [])
                    )
                    self._definitions_cache[risk_type] = risk_def
                    return risk_def
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to load risk definition {risk_type}: {e}")
        
        return None
    
    def list_active_risks(
        self, 
        tenant_id: Optional[str] = None,
        level_filter: Optional[RiskLevel] = None
    ) -> List[RiskEvent]:
        """List currently active risks."""
        events = []
        
        # Load recent events (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        events = self._load_recent_events(cutoff_date, tenant_id)
        
        # Filter by level if specified
        if level_filter:
            events = [e for e in events if e.level == level_filter]
        
        return sorted(events, key=lambda e: e.detected_at, reverse=True)
    
    # Private helper methods
    
    def _initialize_default_risks(self):
        """Initialize default risk definitions from the roadmap."""
        default_risks = [
            RiskDefinition(
                risk_type="provider_outage",
                category=RiskCategory.OPERATIONAL,
                default_level=RiskLevel.HIGH,
                description="Provider service outage or degradation",
                impact_description="Degraded service availability and user experience",
                mitigation_strategy="Fallback providers + circuit breaker pattern",
                automated_mitigation=True,
                mitigation_actions=["switch_to_fallback_provider", "enable_circuit_breaker"]
            ),
            RiskDefinition(
                risk_type="policy_false_positives",
                category=RiskCategory.OPERATIONAL,
                default_level=RiskLevel.MEDIUM,
                description="High rate of policy false positives causing user frustration",
                impact_description="User frustration and reduced system usability",
                mitigation_strategy="Severity tiers + override token mechanism",
                mitigation_actions=["adjust_policy_thresholds", "provide_override_mechanism"]
            ),
            RiskDefinition(
                risk_type="token_cost_spike",
                category=RiskCategory.FINANCIAL,
                default_level=RiskLevel.HIGH,
                description="Unexpected spike in token usage costs",
                impact_description="Budget overrun and unexpected charges",
                mitigation_strategy="Spend alerts + automatic downgrade to cheaper models",
                automated_mitigation=True,
                mitigation_actions=["send_cost_alert", "downgrade_model", "apply_rate_limits"]
            ),
            RiskDefinition(
                risk_type="memory_bloat",
                category=RiskCategory.PERFORMANCE,
                default_level=RiskLevel.MEDIUM,
                description="Excessive memory usage causing system latency",
                impact_description="Increased response times and potential system instability",
                mitigation_strategy="Memory summaries + conversation pruning",
                automated_mitigation=True,
                mitigation_actions=["prune_old_conversations", "create_memory_summaries"]
            ),
            RiskDefinition(
                risk_type="tool_injection",
                category=RiskCategory.SECURITY,
                default_level=RiskLevel.CRITICAL,
                description="Potential tool injection attack or data exfiltration attempt",
                impact_description="Data breach and security compromise",
                mitigation_strategy="Schema validation + sandboxed tool execution",
                automated_mitigation=True,
                mitigation_actions=["validate_tool_inputs", "isolate_tool_execution", "audit_tool_usage"]
            ),
            RiskDefinition(
                risk_type="convergence_stall",
                category=RiskCategory.PERFORMANCE,
                default_level=RiskLevel.HIGH,
                description="Agent conversation failing to converge within reasonable time",
                impact_description="Long sessions consuming excessive resources",
                mitigation_strategy="Hard caps + stagnation detection",
                automated_mitigation=True,
                mitigation_actions=["apply_session_timeout", "detect_conversation_loops"]
            ),
            RiskDefinition(
                risk_type="prompt_leakage",
                category=RiskCategory.SECURITY,
                default_level=RiskLevel.CRITICAL,
                description="Potential system prompt or sensitive information leakage",
                impact_description="Compliance breach and intellectual property exposure",
                mitigation_strategy="Secret scanning + content redaction",
                automated_mitigation=True,
                mitigation_actions=["scan_for_secrets", "redact_sensitive_content", "audit_outputs"]
            )
        ]
        
        for risk_def in default_risks:
            if not self.get_risk_definition(risk_def.risk_type):
                self._save_risk_definition(risk_def)
    
    def _save_risk_event(self, event: RiskEvent):
        """Save risk event to storage."""
        event_file = self.storage_dir / "events" / f"{event.risk_id}.json"
        data = {
            "risk_id": event.risk_id,
            "category": event.category.value,
            "level": event.level.value,
            "title": event.title,
            "description": event.description,
            "detected_at": event.detected_at.isoformat(),
            "agent_name": event.agent_name,
            "session_id": event.session_id,
            "tenant_id": event.tenant_id,
            "metadata": event.metadata
        }
        
        with open(event_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_mitigation(self, mitigation: RiskMitigation):
        """Save risk mitigation to storage."""
        mitigation_file = self.storage_dir / "mitigations" / f"{mitigation.mitigation_id}.json"
        data = {
            "mitigation_id": mitigation.mitigation_id,
            "risk_id": mitigation.risk_id,
            "action": mitigation.action,
            "implemented_at": mitigation.implemented_at.isoformat(),
            "effectiveness": mitigation.effectiveness,
            "automated": mitigation.automated,
            "metadata": mitigation.metadata
        }
        
        with open(mitigation_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_risk_definition(self, risk_def: RiskDefinition):
        """Save risk definition to storage."""
        def_file = self.storage_dir / "definitions" / f"{risk_def.risk_type}.json"
        data = {
            "risk_type": risk_def.risk_type,
            "category": risk_def.category.value,
            "default_level": risk_def.default_level.value,
            "description": risk_def.description,
            "impact_description": risk_def.impact_description,
            "mitigation_strategy": risk_def.mitigation_strategy,
            "detection_rules": risk_def.detection_rules,
            "automated_mitigation": risk_def.automated_mitigation,
            "mitigation_actions": risk_def.mitigation_actions
        }
        
        with open(def_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update cache
        self._definitions_cache[risk_def.risk_type] = risk_def
    
    def _load_recent_events(
        self, 
        cutoff_date: datetime,
        tenant_id: Optional[str] = None
    ) -> List[RiskEvent]:
        """Load recent risk events."""
        events = []
        events_dir = self.storage_dir / "events"
        
        for event_file in events_dir.glob("*.json"):
            try:
                with open(event_file) as f:
                    data = json.load(f)
                    detected_at = datetime.fromisoformat(data["detected_at"])
                    
                    if detected_at >= cutoff_date:
                        # Include events that match the tenant_id OR have no tenant_id if we're not filtering by tenant
                        if tenant_id is None or data.get("tenant_id") == tenant_id or data.get("tenant_id") is None:
                            event = RiskEvent(
                                risk_id=data["risk_id"],
                                category=RiskCategory(data["category"]),
                                level=RiskLevel(data["level"]),
                                title=data["title"],
                                description=data["description"],
                                detected_at=detected_at,
                                agent_name=data.get("agent_name"),
                                session_id=data.get("session_id"),
                                tenant_id=data.get("tenant_id"),
                                metadata=data.get("metadata", {})
                            )
                            events.append(event)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load risk event from {event_file}: {e}")
                continue
        
        return events
    
    def _trigger_automated_mitigation(self, risk_event: RiskEvent, risk_def: RiskDefinition):
        """Trigger automated mitigation for a risk event."""
        for action in risk_def.mitigation_actions:
            try:
                if action in self._mitigation_handlers:
                    success = self._mitigation_handlers[action](risk_event)
                    self.mitigate_risk(
                        risk_event.risk_id,
                        action,
                        automated=True,
                        effectiveness=1.0 if success else 0.0,
                        metadata={"handler_success": success}
                    )
                else:
                    # Log mitigation as planned but not executed
                    self.mitigate_risk(
                        risk_event.risk_id,
                        f"Planned: {action}",
                        automated=False,
                        metadata={"handler_missing": True}
                    )
            except Exception as e:
                logger.error(f"Error executing automated mitigation {action}: {e}")
    
    def _generate_risk_recommendations(self, events: List[RiskEvent]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if not events:
            return ["No recent risk events - continue monitoring"]
        
        # Count high priority risks
        high_priority = sum(1 for e in events if e.level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        
        if high_priority > 5:
            recommendations.append("High number of critical risks - review system configuration")
        
        # Check for recurring risk types
        risk_types = {}
        for event in events:
            risk_type = event.metadata.get("risk_type", "unknown")
            risk_types[risk_type] = risk_types.get(risk_type, 0) + 1
        
        for risk_type, count in risk_types.items():
            if count > 3:
                recommendations.append(f"Recurring {risk_type} risks - implement preventive measures")
        
        if not recommendations:
            recommendations.append("Risk levels appear normal - continue monitoring")
        
        return recommendations


__all__ = [
    "RiskRegister",
    "RiskEvent",
    "RiskMitigation",
    "RiskDefinition",
    "RiskLevel",
    "RiskStatus",
    "RiskCategory"
]