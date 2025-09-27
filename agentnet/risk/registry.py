"""Risk registry implementation - moving risk management from docs to code."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentnet.risk")


class RiskCategory(Enum):
    """Risk categories based on AgentNet architecture."""
    
    PROVIDER = "provider"
    POLICY = "policy"
    COST = "cost"
    MEMORY = "memory"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


class RiskLevel(Enum):
    """Risk impact levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskEvent:
    """Represents a specific risk occurrence."""
    
    risk_id: str
    event_id: str
    timestamp: datetime
    severity: RiskLevel
    description: str
    context: Dict[str, Any]
    mitigation_applied: Optional[str] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None


@dataclass
class RiskDefinition:
    """Defines a type of risk and its mitigation strategies."""
    
    risk_id: str
    name: str
    category: RiskCategory
    description: str
    impact: RiskLevel
    likelihood: RiskLevel
    mitigation_strategies: List[str]
    detection_rules: Dict[str, Any]
    escalation_threshold: int = 3  # Number of events before escalation
    auto_mitigation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskRegistry:
    """Central registry for risk definitions and events."""
    
    def __init__(self, storage_dir: str = "data/risk_logs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.risk_definitions: Dict[str, RiskDefinition] = {}
        self.risk_events: List[RiskEvent] = []
        
        self._load_default_risks()
        logger.info(f"RiskRegistry initialized with storage_dir: {storage_dir}")
    
    def _load_default_risks(self):
        """Load default risk definitions from AgentNet roadmap."""
        
        # Provider outage risk
        self.risk_definitions["provider_outage"] = RiskDefinition(
            risk_id="provider_outage",
            name="Provider Outage",
            category=RiskCategory.PROVIDER,
            description="LLM provider service becomes unavailable",
            impact=RiskLevel.HIGH,
            likelihood=RiskLevel.MEDIUM,
            mitigation_strategies=[
                "fallback_provider",
                "circuit_breaker", 
                "retry_with_backoff",
                "degraded_service_mode"
            ],
            detection_rules={
                "consecutive_failures": 3,
                "error_rate_threshold": 0.5,
                "timeout_threshold_ms": 30000
            },
            auto_mitigation=True
        )
        
        # Policy false positives
        self.risk_definitions["policy_false_positive"] = RiskDefinition(
            risk_id="policy_false_positive",
            name="Policy False Positives",
            category=RiskCategory.POLICY,
            description="Policy rules incorrectly flagging legitimate requests",
            impact=RiskLevel.MEDIUM,
            likelihood=RiskLevel.HIGH,
            mitigation_strategies=[
                "severity_tiers",
                "override_token",
                "policy_refinement",
                "human_review_queue"
            ],
            detection_rules={
                "false_positive_rate": 0.1,
                "user_override_frequency": 0.2
            },
            auto_mitigation=False
        )
        
        # Token cost spike
        self.risk_definitions["cost_spike"] = RiskDefinition(
            risk_id="cost_spike",
            name="Token Cost Spike",
            category=RiskCategory.COST,
            description="Sudden increase in token usage and costs",
            impact=RiskLevel.HIGH,
            likelihood=RiskLevel.MEDIUM,
            mitigation_strategies=[
                "spend_alerts",
                "rate_limiting",
                "model_downgrade",
                "budget_enforcement"
            ],
            detection_rules={
                "cost_velocity_multiplier": 3.0,
                "daily_cost_threshold": 100.0,
                "token_rate_threshold": 10000
            },
            auto_mitigation=True
        )
        
        # Memory bloat
        self.risk_definitions["memory_bloat"] = RiskDefinition(
            risk_id="memory_bloat",
            name="Memory Bloat",
            category=RiskCategory.MEMORY,
            description="Excessive memory usage causing performance degradation",
            impact=RiskLevel.MEDIUM,
            likelihood=RiskLevel.MEDIUM,
            mitigation_strategies=[
                "memory_pruning",
                "summarization",
                "context_compression",
                "session_restart"
            ],
            detection_rules={
                "memory_usage_mb": 1000,
                "context_length_threshold": 50000,
                "response_time_degradation": 0.5
            },
            auto_mitigation=True
        )
        
        # Tool injection
        self.risk_definitions["tool_injection"] = RiskDefinition(
            risk_id="tool_injection",
            name="Tool Injection Attack",
            category=RiskCategory.SECURITY,
            description="Malicious input attempting to manipulate tool execution",
            impact=RiskLevel.CRITICAL,
            likelihood=RiskLevel.LOW,
            mitigation_strategies=[
                "input_sanitization",
                "schema_validation",
                "sandbox_execution",
                "privilege_separation"
            ],
            detection_rules={
                "suspicious_patterns": [
                    "eval(",
                    "__import__",
                    "exec(",
                    "system(",
                    "subprocess"
                ],
                "anomaly_score_threshold": 0.8
            },
            auto_mitigation=True,
            escalation_threshold=1  # Immediate escalation for security
        )
        
        # Convergence stall
        self.risk_definitions["convergence_stall"] = RiskDefinition(
            risk_id="convergence_stall",
            name="Convergence Stall",
            category=RiskCategory.PERFORMANCE,
            description="Multi-agent session fails to converge or reach consensus",
            impact=RiskLevel.MEDIUM,
            likelihood=RiskLevel.MEDIUM,
            mitigation_strategies=[
                "hard_turn_limit",
                "stagnation_detection",
                "tie_breaker_agent",
                "session_timeout"
            ],
            detection_rules={
                "max_turns": 50,
                "similarity_threshold": 0.95,
                "time_limit_minutes": 30
            },
            auto_mitigation=True
        )
        
        # Prompt leakage
        self.risk_definitions["prompt_leakage"] = RiskDefinition(
            risk_id="prompt_leakage",
            name="Prompt Leakage",
            category=RiskCategory.COMPLIANCE,
            description="Sensitive information in prompts exposed in logs or responses",
            impact=RiskLevel.HIGH,
            likelihood=RiskLevel.LOW,
            mitigation_strategies=[
                "secret_scanning",
                "pii_redaction",
                "prompt_sanitization",
                "audit_logging"
            ],
            detection_rules={
                "pii_patterns": [
                    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
                ],
                "sensitive_keywords": ["password", "token", "secret", "key"]
            },
            auto_mitigation=True,
            escalation_threshold=1
        )
    
    def register_risk_event(
        self,
        risk_id: str,
        description: str,
        context: Dict[str, Any],
        severity: Optional[RiskLevel] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> RiskEvent:
        """Register a new risk event."""
        
        if risk_id not in self.risk_definitions:
            logger.warning(f"Unknown risk_id: {risk_id}")
            # Create a generic risk definition
            self.risk_definitions[risk_id] = RiskDefinition(
                risk_id=risk_id,
                name=f"Unknown Risk: {risk_id}",
                category=RiskCategory.PERFORMANCE,
                description="Automatically created for unknown risk",
                impact=RiskLevel.MEDIUM,
                likelihood=RiskLevel.MEDIUM,
                mitigation_strategies=[],
                detection_rules={}
            )
        
        risk_def = self.risk_definitions[risk_id]
        event_severity = severity or risk_def.impact
        
        event = RiskEvent(
            risk_id=risk_id,
            event_id=f"{risk_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            severity=event_severity,
            description=description,
            context=context,
            tenant_id=tenant_id,
            session_id=session_id,
            agent_name=agent_name
        )
        
        self.risk_events.append(event)
        self._persist_event(event)
        
        logger.warning(
            f"Risk event registered: {risk_id} - {description} "
            f"(severity: {event_severity.value})"
        )
        
        return event
    
    def _persist_event(self, event: RiskEvent):
        """Persist risk event to storage."""
        date_str = event.timestamp.strftime("%Y-%m-%d")
        filename = self.storage_dir / f"risk_events_{date_str}.jsonl"
        
        event_dict = {
            "risk_id": event.risk_id,
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "severity": event.severity.value,
            "description": event.description,
            "context": event.context,
            "mitigation_applied": event.mitigation_applied,
            "resolved": event.resolved,
            "resolution_timestamp": event.resolution_timestamp.isoformat() if event.resolution_timestamp else None,
            "tenant_id": event.tenant_id,
            "session_id": event.session_id,
            "agent_name": event.agent_name
        }
        
        with open(filename, "a") as f:
            f.write(json.dumps(event_dict) + "\n")
    
    def get_risk_events(
        self,
        risk_id: Optional[str] = None,
        severity: Optional[RiskLevel] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
        resolved: Optional[bool] = None
    ) -> List[RiskEvent]:
        """Query risk events with filters."""
        
        filtered_events = []
        
        for event in self.risk_events:
            if risk_id and event.risk_id != risk_id:
                continue
            if severity and event.severity != severity:
                continue
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            if tenant_id and event.tenant_id != tenant_id:
                continue
            if resolved is not None and event.resolved != resolved:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def resolve_risk_event(
        self,
        event_id: str,
        mitigation_applied: Optional[str] = None
    ) -> bool:
        """Mark a risk event as resolved."""
        
        for event in self.risk_events:
            if event.event_id == event_id:
                event.resolved = True
                event.resolution_timestamp = datetime.now()
                event.mitigation_applied = mitigation_applied
                
                # Re-persist the resolved event
                self._persist_event(event)
                
                logger.info(f"Risk event resolved: {event_id}")
                return True
        
        logger.warning(f"Risk event not found: {event_id}")
        return False
    
    def get_risk_summary(
        self,
        tenant_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get summary of risk events."""
        
        start_date = datetime.now() - timedelta(days=days_back)
        events = self.get_risk_events(
            start_date=start_date,
            tenant_id=tenant_id
        )
        
        # Group by category and severity
        by_category = {}
        by_severity = {}
        
        for event in events:
            risk_def = self.risk_definitions.get(event.risk_id)
            category = risk_def.category.value if risk_def else "unknown"
            severity = event.severity.value
            
            by_category[category] = by_category.get(category, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Calculate resolution rate
        total_events = len(events)
        resolved_events = len([e for e in events if e.resolved])
        resolution_rate = resolved_events / total_events if total_events > 0 else 0
        
        # Top risks
        risk_counts = {}
        for event in events:
            risk_counts[event.risk_id] = risk_counts.get(event.risk_id, 0) + 1
        
        top_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "summary_period_days": days_back,
            "total_events": total_events,
            "resolved_events": resolved_events,
            "resolution_rate": resolution_rate,
            "unresolved_events": total_events - resolved_events,
            "by_category": by_category,
            "by_severity": by_severity,
            "top_risks": [
                {"risk_id": risk_id, "count": count, "name": self.risk_definitions.get(risk_id).name if risk_id in self.risk_definitions else risk_id}
                for risk_id, count in top_risks
            ],
            "summary_generated": datetime.now().isoformat()
        }
    
    def export_risk_register(self) -> Dict[str, Any]:
        """Export the complete risk register for documentation."""
        
        return {
            "risk_definitions": {
                risk_id: {
                    "name": risk_def.name,
                    "category": risk_def.category.value,
                    "description": risk_def.description,
                    "impact": risk_def.impact.value,
                    "likelihood": risk_def.likelihood.value,
                    "mitigation_strategies": risk_def.mitigation_strategies,
                    "detection_rules": risk_def.detection_rules,
                    "auto_mitigation": risk_def.auto_mitigation,
                    "escalation_threshold": risk_def.escalation_threshold
                }
                for risk_id, risk_def in self.risk_definitions.items()
            },
            "export_timestamp": datetime.now().isoformat(),
            "total_risk_types": len(self.risk_definitions)
        }