"""Risk monitoring and alerting system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .registry import RiskRegistry, RiskLevel, RiskCategory

logger = logging.getLogger("agentnet.risk.monitor")


@dataclass
class RiskAlert:
    """Risk alert notification."""
    
    alert_id: str
    risk_id: str
    alert_type: str  # 'threshold', 'pattern', 'escalation'
    severity: RiskLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None


class RiskMonitor:
    """Monitors system state and detects risk conditions."""
    
    def __init__(self, risk_registry: RiskRegistry):
        self.risk_registry = risk_registry
        self.monitoring_enabled = True
        self.check_intervals = {
            RiskCategory.PROVIDER: 60,      # seconds
            RiskCategory.COST: 300,         # 5 minutes
            RiskCategory.SECURITY: 10,      # 10 seconds - critical
            RiskCategory.PERFORMANCE: 120,  # 2 minutes
            RiskCategory.MEMORY: 60,        # 1 minute
            RiskCategory.POLICY: 180,       # 3 minutes
            RiskCategory.COMPLIANCE: 600,   # 10 minutes
        }
        logger.info("RiskMonitor initialized")
    
    def check_provider_risks(
        self,
        provider_stats: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Check for provider-related risks."""
        alerts = []
        
        risk_def = self.risk_registry.risk_definitions.get("provider_outage")
        if not risk_def:
            return alerts
        
        # Check consecutive failures
        consecutive_failures = provider_stats.get("consecutive_failures", 0)
        failure_threshold = risk_def.detection_rules.get("consecutive_failures", 3)
        
        if consecutive_failures >= failure_threshold:
            alert = RiskAlert(
                alert_id=f"provider_outage_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="provider_outage",
                alert_type="threshold",
                severity=RiskLevel.HIGH,
                message=f"Provider experiencing {consecutive_failures} consecutive failures",
                details={
                    "consecutive_failures": consecutive_failures,
                    "threshold": failure_threshold,
                    "provider": provider_stats.get("provider_name", "unknown")
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id
            )
            alerts.append(alert)
            
            # Register risk event
            self.risk_registry.register_risk_event(
                risk_id="provider_outage",
                description=f"Provider outage detected: {consecutive_failures} failures",
                context=provider_stats,
                tenant_id=tenant_id
            )
        
        # Check error rate
        error_rate = provider_stats.get("error_rate", 0.0)
        error_threshold = risk_def.detection_rules.get("error_rate_threshold", 0.5)
        
        if error_rate >= error_threshold:
            alert = RiskAlert(
                alert_id=f"provider_error_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="provider_outage",
                alert_type="threshold",
                severity=RiskLevel.MEDIUM,
                message=f"High provider error rate: {error_rate:.2%}",
                details={
                    "error_rate": error_rate,
                    "threshold": error_threshold,
                    "provider": provider_stats.get("provider_name", "unknown")
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id
            )
            alerts.append(alert)
        
        return alerts
    
    def check_cost_risks(
        self,
        cost_stats: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Check for cost-related risks."""
        alerts = []
        
        risk_def = self.risk_registry.risk_definitions.get("cost_spike")
        if not risk_def:
            return alerts
        
        # Check cost velocity
        current_cost = cost_stats.get("current_hourly_cost", 0.0)
        baseline_cost = cost_stats.get("baseline_hourly_cost", 0.0)
        velocity_multiplier = risk_def.detection_rules.get("cost_velocity_multiplier", 3.0)
        
        if baseline_cost > 0 and current_cost >= baseline_cost * velocity_multiplier:
            alert = RiskAlert(
                alert_id=f"cost_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="cost_spike",
                alert_type="threshold",
                severity=RiskLevel.HIGH,
                message=f"Cost spike detected: {current_cost/baseline_cost:.1f}x baseline",
                details={
                    "current_cost": current_cost,
                    "baseline_cost": baseline_cost,
                    "velocity_multiplier": current_cost / baseline_cost
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id
            )
            alerts.append(alert)
            
            # Register risk event
            self.risk_registry.register_risk_event(
                risk_id="cost_spike",
                description=f"Cost spike: ${current_cost:.6f}/hour vs ${baseline_cost:.6f} baseline",
                context=cost_stats,
                tenant_id=tenant_id
            )
        
        # Check daily cost threshold
        daily_cost = cost_stats.get("daily_cost", 0.0)
        daily_threshold = risk_def.detection_rules.get("daily_cost_threshold", 100.0)
        
        if daily_cost >= daily_threshold:
            severity = RiskLevel.CRITICAL if daily_cost >= daily_threshold * 2 else RiskLevel.HIGH
            
            alert = RiskAlert(
                alert_id=f"cost_threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="cost_spike",
                alert_type="threshold",
                severity=severity,
                message=f"Daily cost threshold exceeded: ${daily_cost:.2f}",
                details={
                    "daily_cost": daily_cost,
                    "threshold": daily_threshold
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id
            )
            alerts.append(alert)
        
        return alerts
    
    def check_security_risks(
        self,
        request_data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Check for security risks in real-time."""
        alerts = []
        
        # Check for tool injection patterns
        risk_def = self.risk_registry.risk_definitions.get("tool_injection")
        if risk_def:
            content = request_data.get("content", "")
            suspicious_patterns = risk_def.detection_rules.get("suspicious_patterns", [])
            
            for pattern in suspicious_patterns:
                if pattern in content:
                    alert = RiskAlert(
                        alert_id=f"tool_injection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        risk_id="tool_injection",
                        alert_type="pattern",
                        severity=RiskLevel.CRITICAL,
                        message=f"Suspicious pattern detected: {pattern}",
                        details={
                            "pattern": pattern,
                            "content_length": len(content),
                            "request_id": request_data.get("request_id")
                        },
                        timestamp=datetime.now(),
                        tenant_id=tenant_id,
                        session_id=session_id
                    )
                    alerts.append(alert)
                    
                    # Register critical risk event immediately
                    self.risk_registry.register_risk_event(
                        risk_id="tool_injection",
                        description=f"Tool injection attempt detected: {pattern}",
                        context=request_data,
                        severity=RiskLevel.CRITICAL,
                        tenant_id=tenant_id,
                        session_id=session_id
                    )
        
        # Check for prompt leakage patterns
        prompt_risk_def = self.risk_registry.risk_definitions.get("prompt_leakage")
        if prompt_risk_def:
            content = request_data.get("content", "")
            
            # Check PII patterns
            import re
            pii_patterns = prompt_risk_def.detection_rules.get("pii_patterns", [])
            
            for pattern in pii_patterns:
                if re.search(pattern, content):
                    alert = RiskAlert(
                        alert_id=f"prompt_leakage_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        risk_id="prompt_leakage",
                        alert_type="pattern",
                        severity=RiskLevel.HIGH,
                        message="Potential PII detected in prompt",
                        details={
                            "pattern_type": "pii",
                            "content_length": len(content)
                        },
                        timestamp=datetime.now(),
                        tenant_id=tenant_id,
                        session_id=session_id
                    )
                    alerts.append(alert)
                    
                    # Register risk event
                    self.risk_registry.register_risk_event(
                        risk_id="prompt_leakage",
                        description="PII pattern detected in prompt",
                        context={"pattern_matched": True, "content_length": len(content)},
                        severity=RiskLevel.HIGH,
                        tenant_id=tenant_id,
                        session_id=session_id
                    )
                    break  # Only report once per request
            
            # Check sensitive keywords
            sensitive_keywords = prompt_risk_def.detection_rules.get("sensitive_keywords", [])
            content_lower = content.lower()
            
            for keyword in sensitive_keywords:
                if keyword in content_lower:
                    alert = RiskAlert(
                        alert_id=f"sensitive_keyword_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        risk_id="prompt_leakage",
                        alert_type="pattern",
                        severity=RiskLevel.MEDIUM,
                        message=f"Sensitive keyword detected: {keyword}",
                        details={
                            "keyword": keyword,
                            "content_length": len(content)
                        },
                        timestamp=datetime.now(),
                        tenant_id=tenant_id,
                        session_id=session_id
                    )
                    alerts.append(alert)
        
        return alerts
    
    def check_memory_risks(
        self,
        memory_stats: Dict[str, Any],
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Check for memory-related risks."""
        alerts = []
        
        risk_def = self.risk_registry.risk_definitions.get("memory_bloat")
        if not risk_def:
            return alerts
        
        # Check memory usage
        memory_usage_mb = memory_stats.get("memory_usage_mb", 0)
        memory_threshold = risk_def.detection_rules.get("memory_usage_mb", 1000)
        
        if memory_usage_mb >= memory_threshold:
            severity = RiskLevel.HIGH if memory_usage_mb >= memory_threshold * 2 else RiskLevel.MEDIUM
            
            alert = RiskAlert(
                alert_id=f"memory_bloat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="memory_bloat",
                alert_type="threshold",
                severity=severity,
                message=f"High memory usage: {memory_usage_mb}MB",
                details={
                    "memory_usage_mb": memory_usage_mb,
                    "threshold": memory_threshold
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                session_id=session_id
            )
            alerts.append(alert)
            
            # Register risk event
            self.risk_registry.register_risk_event(
                risk_id="memory_bloat",
                description=f"High memory usage: {memory_usage_mb}MB",
                context=memory_stats,
                tenant_id=tenant_id,
                session_id=session_id
            )
        
        # Check context length
        context_length = memory_stats.get("context_length", 0)
        context_threshold = risk_def.detection_rules.get("context_length_threshold", 50000)
        
        if context_length >= context_threshold:
            alert = RiskAlert(
                alert_id=f"context_length_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="memory_bloat",
                alert_type="threshold",
                severity=RiskLevel.MEDIUM,
                message=f"Large context length: {context_length} tokens",
                details={
                    "context_length": context_length,
                    "threshold": context_threshold
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                session_id=session_id
            )
            alerts.append(alert)
        
        return alerts
    
    def check_convergence_risks(
        self,
        session_stats: Dict[str, Any],
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Check for convergence stall risks."""
        alerts = []
        
        risk_def = self.risk_registry.risk_definitions.get("convergence_stall")
        if not risk_def:
            return alerts
        
        # Check turn count
        turn_count = session_stats.get("turn_count", 0)
        max_turns = risk_def.detection_rules.get("max_turns", 50)
        
        if turn_count >= max_turns:
            alert = RiskAlert(
                alert_id=f"convergence_stall_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="convergence_stall",
                alert_type="threshold",
                severity=RiskLevel.MEDIUM,
                message=f"Session approaching turn limit: {turn_count}/{max_turns}",
                details={
                    "turn_count": turn_count,
                    "max_turns": max_turns,
                    "session_duration_minutes": session_stats.get("duration_minutes", 0)
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                session_id=session_id
            )
            alerts.append(alert)
            
            # Register risk event
            self.risk_registry.register_risk_event(
                risk_id="convergence_stall",
                description=f"Session approaching turn limit: {turn_count} turns",
                context=session_stats,
                tenant_id=tenant_id,
                session_id=session_id
            )
        
        # Check session duration
        duration_minutes = session_stats.get("duration_minutes", 0)
        time_limit = risk_def.detection_rules.get("time_limit_minutes", 30)
        
        if duration_minutes >= time_limit:
            alert = RiskAlert(
                alert_id=f"session_timeout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id="convergence_stall",
                alert_type="threshold",
                severity=RiskLevel.MEDIUM,
                message=f"Long-running session: {duration_minutes} minutes",
                details={
                    "duration_minutes": duration_minutes,
                    "time_limit": time_limit,
                    "turn_count": turn_count
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                session_id=session_id
            )
            alerts.append(alert)
        
        return alerts
    
    def check_escalation_needed(
        self,
        risk_id: str,
        tenant_id: Optional[str] = None
    ) -> Optional[RiskAlert]:
        """Check if a risk needs escalation based on event frequency."""
        
        risk_def = self.risk_registry.risk_definitions.get(risk_id)
        if not risk_def:
            return None
        
        # Get recent events for this risk
        lookback_time = datetime.now() - timedelta(hours=1)
        recent_events = self.risk_registry.get_risk_events(
            risk_id=risk_id,
            start_date=lookback_time,
            tenant_id=tenant_id,
            resolved=False
        )
        
        if len(recent_events) >= risk_def.escalation_threshold:
            return RiskAlert(
                alert_id=f"escalation_{risk_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id=risk_id,
                alert_type="escalation",
                severity=RiskLevel.CRITICAL,
                message=f"Risk escalation: {len(recent_events)} unresolved {risk_id} events",
                details={
                    "event_count": len(recent_events),
                    "escalation_threshold": risk_def.escalation_threshold,
                    "lookback_hours": 1
                },
                timestamp=datetime.now(),
                tenant_id=tenant_id
            )
        
        return None
    
    def get_all_active_alerts(
        self,
        tenant_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Get all currently active risk alerts."""
        alerts = []
        
        # This would typically query a persistent alert store
        # For now, return a placeholder indicating the monitoring system is active
        logger.info(f"Monitoring active for tenant: {tenant_id or 'global'}")
        
        return alerts