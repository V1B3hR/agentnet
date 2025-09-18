"""Audit Workflow Implementation for P6 Enterprise Hardening.

This module provides comprehensive audit event logging and workflow management
for enterprise compliance and security monitoring.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events tracked by the system."""

    AGENT_CREATED = "agent_created"
    AGENT_INFERENCE = "agent_inference"
    POLICY_VIOLATION = "policy_violation"
    EXPORT_EVALUATION = "export_evaluation"
    DATA_ACCESS = "data_access"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    CONFIG_CHANGE = "config_change"
    TOOL_INVOCATION = "tool_invocation"
    COST_EVENT = "cost_event"
    SECURITY_ALERT = "security_alert"
    SYSTEM_ERROR = "system_error"
    COMPLIANCE_CHECK = "compliance_check"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event for comprehensive logging."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    severity: AuditSeverity = AuditSeverity.LOW
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)
    retention_class: str = "standard"  # standard, extended, permanent

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for storage/serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "compliance_tags": self.compliance_tags,
            "retention_class": self.retention_class,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create audit event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.utcnow()
            ),
            event_type=AuditEventType(data.get("event_type", "system_error")),
            severity=AuditSeverity(data.get("severity", "low")),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            resource_id=data.get("resource_id"),
            action=data.get("action", ""),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            compliance_tags=data.get("compliance_tags", []),
            retention_class=data.get("retention_class", "standard"),
        )


class AuditLogger:
    """Enterprise audit logger for comprehensive event tracking."""

    def __init__(self, storage_backend: Any = None):
        self.storage = storage_backend
        self._event_handlers = {}
        self._event_buffer = []

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        # Add automatic compliance tags based on event type
        event.compliance_tags.extend(self._get_automatic_compliance_tags(event))

        # Store event
        if self.storage:
            self.storage.store_event(event)
        else:
            self._event_buffer.append(event)

        # Trigger event handlers
        for handler in self._event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in audit event handler: {e}")

        logger.info(f"Audit event logged: {event.event_type.value} - {event.action}")

    def _get_automatic_compliance_tags(self, event: AuditEvent) -> List[str]:
        """Get automatic compliance tags based on event characteristics."""
        tags = []

        # SOC2 compliance tags
        if event.event_type in [
            AuditEventType.USER_LOGIN,
            AuditEventType.USER_LOGOUT,
            AuditEventType.PERMISSION_GRANT,
        ]:
            tags.append("SOC2_SECURITY")

        if event.event_type in [
            AuditEventType.DATA_ACCESS,
            AuditEventType.EXPORT_EVALUATION,
        ]:
            tags.append("SOC2_CONFIDENTIALITY")

        if event.event_type == AuditEventType.CONFIG_CHANGE:
            tags.append("SOC2_PROCESSING_INTEGRITY")

        # GDPR compliance tags
        if "personal_data" in event.details or "pii" in str(event.details).lower():
            tags.append("GDPR")

        # High-risk indicators
        if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
            tags.append("HIGH_RISK")

        if event.event_type == AuditEventType.SECURITY_ALERT:
            tags.append("SECURITY_INCIDENT")

        return tags

    def log_agent_inference(
        self, agent_id: str, session_id: str, user_id: Optional[str] = None, **kwargs
    ) -> None:
        """Log agent inference event."""
        event = AuditEvent(
            event_type=AuditEventType.AGENT_INFERENCE,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            action="agent_inference_executed",
            details=kwargs,
        )
        self.log_event(event)

    def log_policy_violation(
        self,
        violation_details: Dict[str, Any],
        session_id: str,
        agent_id: str,
        **kwargs,
    ) -> None:
        """Log policy violation event."""
        severity = (
            AuditSeverity.HIGH
            if violation_details.get("severity") == "severe"
            else AuditSeverity.MEDIUM
        )

        event = AuditEvent(
            event_type=AuditEventType.POLICY_VIOLATION,
            severity=severity,
            session_id=session_id,
            agent_id=agent_id,
            action="policy_violation_detected",
            details={"violation": violation_details, **kwargs},
        )
        self.log_event(event)

    def log_export_evaluation(
        self,
        classification_level: str,
        export_allowed: bool,
        destination: str,
        **kwargs,
    ) -> None:
        """Log export control evaluation event."""
        severity = AuditSeverity.HIGH if not export_allowed else AuditSeverity.LOW

        event = AuditEvent(
            event_type=AuditEventType.EXPORT_EVALUATION,
            severity=severity,
            action="export_evaluation_performed",
            details={
                "classification_level": classification_level,
                "export_allowed": export_allowed,
                "destination": destination,
                **kwargs,
            },
        )
        self.log_event(event)

    def log_user_action(
        self, user_id: str, action: str, details: Dict[str, Any] = None, **kwargs
    ) -> None:
        """Log user action event."""
        event_type = (
            AuditEventType.USER_LOGIN
            if "login" in action.lower()
            else AuditEventType.DATA_ACCESS
        )

        event = AuditEvent(
            event_type=event_type,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            action=action,
            details=details or {},
            **kwargs,
        )
        self.log_event(event)

    def log_security_alert(
        self,
        alert_type: str,
        details: Dict[str, Any],
        severity: AuditSeverity = AuditSeverity.HIGH,
    ) -> None:
        """Log security alert event."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=severity,
            action=f"security_alert_{alert_type}",
            details=details,
            compliance_tags=["SECURITY_INCIDENT"],
        )
        self.log_event(event)

    def register_event_handler(
        self, event_type: AuditEventType, handler: callable
    ) -> None:
        """Register an event handler for specific audit event types."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_events(self, limit: int = 100) -> List[AuditEvent]:
        """Get recent audit events."""
        if self.storage:
            return self.storage.get_events(limit=limit)
        else:
            return self._event_buffer[-limit:]


class AuditWorkflow:
    """Manages audit workflows and compliance processes."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self._workflow_handlers = {}
        self._compliance_rules = []

    def register_workflow(self, workflow_name: str, handler: callable) -> None:
        """Register a workflow handler."""
        self._workflow_handlers[workflow_name] = handler

    def trigger_workflow(
        self, workflow_name: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger an audit workflow."""
        if workflow_name not in self._workflow_handlers:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Log workflow trigger
        self.audit_logger.log_event(
            AuditEvent(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                severity=AuditSeverity.MEDIUM,
                action=f"workflow_triggered_{workflow_name}",
                details={"workflow": workflow_name, "context": context},
            )
        )

        try:
            result = self._workflow_handlers[workflow_name](context)

            # Log workflow completion
            self.audit_logger.log_event(
                AuditEvent(
                    event_type=AuditEventType.COMPLIANCE_CHECK,
                    severity=AuditSeverity.LOW,
                    action=f"workflow_completed_{workflow_name}",
                    details={"workflow": workflow_name, "result": result},
                )
            )

            return result

        except Exception as e:
            # Log workflow failure
            self.audit_logger.log_event(
                AuditEvent(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    severity=AuditSeverity.HIGH,
                    action=f"workflow_failed_{workflow_name}",
                    details={"workflow": workflow_name, "error": str(e)},
                )
            )
            raise

    def add_compliance_rule(
        self, rule_name: str, condition: callable, action: callable
    ) -> None:
        """Add a compliance rule that triggers on audit events."""
        self._compliance_rules.append(
            {"name": rule_name, "condition": condition, "action": action}
        )

    def evaluate_compliance_rules(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Evaluate compliance rules against an audit event."""
        triggered_rules = []

        for rule in self._compliance_rules:
            try:
                if rule["condition"](event):
                    result = rule["action"](event)
                    triggered_rules.append(
                        {"rule_name": rule["name"], "result": result}
                    )
            except Exception as e:
                logger.error(f"Error evaluating compliance rule {rule['name']}: {e}")

        return triggered_rules
