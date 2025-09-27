"""Risk monitoring and alerting system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .registry import Risk, RiskCategory, RiskLevel, RiskStatus
from .assessment import RiskAssessment

logger = logging.getLogger("agentnet.risk.monitoring")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Available alert channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    SMS = "sms"


@dataclass
class RiskAlert:
    """Risk monitoring alert."""
    
    alert_id: str
    risk_id: str
    alert_type: str  # "threshold_breach", "status_change", "new_risk", etc.
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    
    # Timing
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    
    # Targeting
    channels: List[AlertChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    
    # Status
    is_resolved: bool = False
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskMonitor:
    """Continuous risk monitoring and alerting system."""
    
    def __init__(self):
        self.monitoring_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.monitoring_enabled = True
        
        # Initialize default monitoring rules
        self._setup_default_monitoring_rules()
    
    def _setup_default_monitoring_rules(self):
        """Set up default monitoring rules for different risk scenarios."""
        
        # High-priority risk monitoring
        self.add_monitoring_rule(
            rule_id="high_priority_risks",
            description="Monitor high and critical priority risks",
            condition=lambda risk, context: risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL},
            alert_type="high_priority_risk",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            check_interval_minutes=30,
        )
        
        # Risk score threshold monitoring
        self.add_monitoring_rule(
            rule_id="risk_score_threshold",
            description="Alert when risk score exceeds threshold",
            condition=lambda risk, context: risk.risk_score > 0.8,
            alert_type="risk_score_breach",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
            check_interval_minutes=15,
        )
        
        # Overdue risk resolution monitoring
        self.add_monitoring_rule(
            rule_id="overdue_risks",
            description="Monitor risks past their resolution target date",
            condition=lambda risk, context: (
                risk.target_resolution_date and 
                risk.target_resolution_date < datetime.now() and
                risk.status not in {RiskStatus.CLOSED, RiskStatus.ACCEPTED}
            ),
            alert_type="overdue_risk",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL],
            check_interval_minutes=60,
        )
        
        # New critical risk monitoring
        self.add_monitoring_rule(
            rule_id="new_critical_risks",
            description="Immediate alert for new critical risks",
            condition=lambda risk, context: (
                risk.level == RiskLevel.CRITICAL and
                risk.status == RiskStatus.IDENTIFIED and
                context.get("is_new_risk", False)
            ),
            alert_type="new_critical_risk",
            severity=AlertSeverity.EMERGENCY,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS],
            check_interval_minutes=5,
        )
        
        # Risk status change monitoring
        self.add_monitoring_rule(
            rule_id="status_changes",
            description="Monitor significant risk status changes",
            condition=lambda risk, context: context.get("status_changed", False),
            alert_type="status_change",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            check_interval_minutes=10,
        )
    
    def add_monitoring_rule(
        self,
        rule_id: str,
        description: str,
        condition: Callable[[Risk, Dict[str, Any]], bool],
        alert_type: str,
        severity: AlertSeverity,
        channels: List[AlertChannel],
        check_interval_minutes: int = 30,
        recipients: Optional[List[str]] = None,
    ):
        """Add a new monitoring rule."""
        
        self.monitoring_rules[rule_id] = {
            "description": description,
            "condition": condition,
            "alert_type": alert_type,
            "severity": severity,
            "channels": channels,
            "check_interval_minutes": check_interval_minutes,
            "recipients": recipients or [],
            "last_checked": datetime.now(),
        }
        
        logger.info(f"Added monitoring rule: {rule_id}")
    
    def check_risk(self, risk: Risk, context: Dict[str, Any] = None) -> List[RiskAlert]:
        """Check a risk against all monitoring rules and generate alerts."""
        
        if not self.monitoring_enabled:
            return []
        
        context = context or {}
        alerts_generated = []
        
        for rule_id, rule_config in self.monitoring_rules.items():
            try:
                # Check if it's time to run this rule
                now = datetime.now()
                time_since_last_check = now - rule_config["last_checked"]
                check_interval = timedelta(minutes=rule_config["check_interval_minutes"])
                
                if time_since_last_check < check_interval:
                    continue
                
                # Update last checked time
                rule_config["last_checked"] = now
                
                # Evaluate the condition
                if rule_config["condition"](risk, context):
                    # Generate alert
                    alert = self._generate_alert(
                        risk=risk,
                        rule_id=rule_id,
                        alert_type=rule_config["alert_type"],
                        severity=rule_config["severity"],
                        channels=rule_config["channels"],
                        recipients=rule_config["recipients"],
                        context=context,
                    )
                    
                    if alert:
                        alerts_generated.append(alert)
            
            except Exception as e:
                logger.error(f"Error checking monitoring rule {rule_id}: {e}")
        
        return alerts_generated
    
    def _generate_alert(
        self,
        risk: Risk,
        rule_id: str,
        alert_type: str,
        severity: AlertSeverity,
        channels: List[AlertChannel],
        recipients: List[str],
        context: Dict[str, Any],
    ) -> Optional[RiskAlert]:
        """Generate a risk alert."""
        
        alert_id = f"ALERT_{risk.risk_id}_{alert_type}_{int(datetime.now().timestamp())}"
        
        # Check if similar alert already exists
        existing_alert = self._find_existing_alert(risk.risk_id, alert_type)
        if existing_alert and not existing_alert.is_resolved:
            logger.debug(f"Similar alert already exists for {risk.risk_id}: {existing_alert.alert_id}")
            return None
        
        # Generate alert message
        message = self._generate_alert_message(risk, alert_type, context)
        
        # Create alert details
        details = {
            "rule_id": rule_id,
            "risk_title": risk.title,
            "risk_category": risk.category.value,
            "risk_level": risk.level.value,
            "risk_score": risk.risk_score,
            "risk_status": risk.status.value,
            "context": context,
        }
        
        # Create alert
        alert = RiskAlert(
            alert_id=alert_id,
            risk_id=risk.risk_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
            triggered_at=datetime.now(),
            channels=channels,
            recipients=recipients,
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send alert through configured channels
        self._send_alert(alert)
        
        logger.info(f"Generated alert {alert_id} for risk {risk.risk_id}")
        return alert
    
    def _generate_alert_message(self, risk: Risk, alert_type: str, context: Dict[str, Any]) -> str:
        """Generate human-readable alert message."""
        
        base_message = f"Risk Alert: {risk.title} [{risk.risk_id}]"
        
        if alert_type == "high_priority_risk":
            return f"{base_message} - High priority risk requires attention (Level: {risk.level.value}, Score: {risk.risk_score:.2f})"
        
        elif alert_type == "risk_score_breach":
            return f"{base_message} - Risk score exceeded threshold: {risk.risk_score:.2f} > 0.8"
        
        elif alert_type == "overdue_risk":
            days_overdue = (datetime.now() - risk.target_resolution_date).days
            return f"{base_message} - Risk resolution is {days_overdue} days overdue"
        
        elif alert_type == "new_critical_risk":
            return f"{base_message} - NEW CRITICAL RISK IDENTIFIED - Immediate attention required!"
        
        elif alert_type == "status_change":
            old_status = context.get("old_status", "unknown")
            return f"{base_message} - Status changed from {old_status} to {risk.status.value}"
        
        else:
            return f"{base_message} - Alert type: {alert_type}"
    
    def _find_existing_alert(self, risk_id: str, alert_type: str) -> Optional[RiskAlert]:
        """Find existing unresolved alert for the same risk and type."""
        
        for alert in self.active_alerts.values():
            if (alert.risk_id == risk_id and 
                alert.alert_type == alert_type and 
                not alert.is_resolved):
                return alert
        return None
    
    def _send_alert(self, alert: RiskAlert):
        """Send alert through configured channels."""
        
        for channel in alert.channels:
            try:
                handler = self.alert_handlers.get(channel)
                if handler:
                    handler(alert)
                else:
                    # Default handling
                    if channel == AlertChannel.LOG:
                        logger.warning(f"RISK ALERT [{alert.severity.value.upper()}]: {alert.message}")
                    else:
                        logger.info(f"No handler configured for alert channel: {channel.value}")
            
            except Exception as e:
                logger.error(f"Error sending alert {alert.alert_id} via {channel.value}: {e}")
    
    def register_alert_handler(self, channel: AlertChannel, handler: Callable[[RiskAlert], None]):
        """Register a handler for a specific alert channel."""
        
        self.alert_handlers[channel] = handler
        logger.info(f"Registered alert handler for channel: {channel.value}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.is_acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.is_resolved = True
        alert.resolved_at = datetime.now()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def get_active_alerts(
        self,
        risk_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[str] = None,
    ) -> List[RiskAlert]:
        """Get active alerts with optional filtering."""
        
        alerts = list(self.active_alerts.values())
        
        if risk_id:
            alerts = [a for a in alerts if a.risk_id == risk_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get monitoring and alert statistics."""
        
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Alert counts by severity
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        # Alert counts by type
        type_counts = {}
        for alert in self.alert_history:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.triggered_at > yesterday]
        
        # Average resolution time
        resolved_alerts = [a for a in self.alert_history if a.resolved_at]
        if resolved_alerts:
            resolution_times = [(a.resolved_at - a.triggered_at).total_seconds() / 3600 
                              for a in resolved_alerts]
            avg_resolution_hours = sum(resolution_times) / len(resolution_times)
        else:
            avg_resolution_hours = 0
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "alerts_last_24h": len(recent_alerts),
            "severity_distribution": severity_counts,
            "alert_type_distribution": type_counts,
            "avg_resolution_time_hours": round(avg_resolution_hours, 2),
            "monitoring_rules_count": len(self.monitoring_rules),
            "monitoring_enabled": self.monitoring_enabled,
        }
    
    def enable_monitoring(self):
        """Enable risk monitoring."""
        self.monitoring_enabled = True
        logger.info("Risk monitoring enabled")
    
    def disable_monitoring(self):
        """Disable risk monitoring."""
        self.monitoring_enabled = False
        logger.info("Risk monitoring disabled")
    
    def get_monitoring_health(self) -> Dict[str, Any]:
        """Get monitoring system health status."""
        
        # Check rule health
        now = datetime.now()
        stale_rules = []
        for rule_id, rule_config in self.monitoring_rules.items():
            time_since_check = now - rule_config["last_checked"]
            expected_interval = timedelta(minutes=rule_config["check_interval_minutes"])
            
            if time_since_check > expected_interval * 2:  # 2x the expected interval
                stale_rules.append(rule_id)
        
        # Check alert handler coverage
        configured_channels = set()
        for rule_config in self.monitoring_rules.values():
            configured_channels.update(rule_config["channels"])
        
        missing_handlers = [
            channel.value for channel in configured_channels 
            if channel not in self.alert_handlers and channel != AlertChannel.LOG
        ]
        
        # Determine overall health
        health_issues = []
        if not self.monitoring_enabled:
            health_issues.append("Monitoring is disabled")
        if stale_rules:
            health_issues.append(f"Stale monitoring rules: {', '.join(stale_rules)}")
        if missing_handlers:
            health_issues.append(f"Missing alert handlers: {', '.join(missing_handlers)}")
        
        if not health_issues:
            health_status = "healthy"
        elif len(health_issues) == 1 and "Missing alert handlers" in health_issues[0]:
            health_status = "warning"
        else:
            health_status = "unhealthy"
        
        return {
            "health_status": health_status,
            "monitoring_enabled": self.monitoring_enabled,
            "total_rules": len(self.monitoring_rules),
            "stale_rules": stale_rules,
            "missing_handlers": missing_handlers,
            "health_issues": health_issues,
            "last_check": now.isoformat(),
        }