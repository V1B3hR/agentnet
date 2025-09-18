"""Compliance Reporting for P6 Enterprise Hardening.

This module provides comprehensive compliance reporting capabilities including
audit trail generation, SOC2 logging, and regulatory compliance dashboards.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ComplianceReport:
    """Comprehensive compliance report structure."""
    report_id: str
    generated_at: datetime
    report_type: str
    time_period: Dict[str, datetime]
    summary: Dict[str, Any] = field(default_factory=dict)
    export_controls: Dict[str, Any] = field(default_factory=dict)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceReporter:
    """Generates compliance reports for enterprise audit requirements."""
    
    def __init__(self, storage_dir: str = "./compliance_reports"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._audit_events = []
        self._compliance_metrics = {}
    
    def record_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record an audit event for compliance tracking."""
        audit_event = {
            "event_id": f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "compliance_flags": self._evaluate_compliance_flags(event_type, details)
        }
        
        self._audit_events.append(audit_event)
        logger.info(f"Audit event recorded: {event_type}")
    
    def _evaluate_compliance_flags(self, event_type: str, details: Dict[str, Any]) -> List[str]:
        """Evaluate compliance flags for an audit event."""
        flags = []
        
        # SOC2 Type II compliance flags
        if event_type in ["data_access", "export_evaluation", "policy_violation"]:
            flags.append("SOC2_SECURITY")
        
        if event_type in ["data_export", "content_redaction"]:
            flags.append("DATA_PRIVACY")
        
        if event_type == "export_denied":
            flags.append("EXPORT_CONTROL")
        
        # Check for high-risk indicators
        if details.get("classification_level") in ["restricted", "top_secret"]:
            flags.append("HIGH_RISK")
        
        if details.get("violations_count", 0) > 0:
            flags.append("POLICY_VIOLATION")
        
        return flags
    
    def generate_soc2_report(self, start_date: datetime, end_date: datetime) -> ComplianceReport:
        """Generate SOC2 Type II compliance report."""
        report_id = f"soc2_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Filter audit events for time period
        period_events = [
            event for event in self._audit_events
            if start_date <= datetime.fromisoformat(event["timestamp"]) <= end_date
        ]
        
        # Calculate SOC2 metrics
        soc2_metrics = self._calculate_soc2_metrics(period_events)
        
        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            report_type="SOC2_TYPE_II",
            time_period={"start": start_date, "end": end_date},
            summary={
                "total_audit_events": len(period_events),
                "security_events": soc2_metrics["security_events"],
                "availability_uptime": soc2_metrics["availability_uptime"],
                "processing_integrity_score": soc2_metrics["processing_integrity"],
                "confidentiality_incidents": soc2_metrics["confidentiality_incidents"],
                "privacy_compliance_rate": soc2_metrics["privacy_compliance_rate"]
            },
            audit_events=period_events,
            violations=[event for event in period_events if "POLICY_VIOLATION" in event.get("compliance_flags", [])],
            recommendations=self._generate_soc2_recommendations(soc2_metrics),
            metadata={
                "compliance_framework": "SOC2_TYPE_II",
                "reporting_standard": "AICPA_TSC",
                "assessment_period_days": (end_date - start_date).days
            }
        )
        
        return report
    
    def _calculate_soc2_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate SOC2 compliance metrics."""
        total_events = len(events)
        security_events = len([e for e in events if "SOC2_SECURITY" in e.get("compliance_flags", [])])
        high_risk_events = len([e for e in events if "HIGH_RISK" in e.get("compliance_flags", [])])
        violation_events = len([e for e in events if "POLICY_VIOLATION" in e.get("compliance_flags", [])])
        
        return {
            "security_events": security_events,
            "availability_uptime": 99.9,  # Would be calculated from actual system metrics
            "processing_integrity": max(0, 100 - (violation_events / max(total_events, 1)) * 100),
            "confidentiality_incidents": high_risk_events,
            "privacy_compliance_rate": max(0, 100 - (violation_events / max(total_events, 1)) * 100)
        }
    
    def _generate_soc2_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate SOC2 compliance recommendations."""
        recommendations = []
        
        if metrics["confidentiality_incidents"] > 5:
            recommendations.append("Review and strengthen data classification policies")
        
        if metrics["processing_integrity"] < 95:
            recommendations.append("Implement additional policy validation controls")
        
        if metrics["privacy_compliance_rate"] < 98:
            recommendations.append("Enhance privacy protection mechanisms")
        
        if metrics["security_events"] > metrics["confidentiality_incidents"] * 3:
            recommendations.append("Optimize security event filtering to reduce noise")
        
        return recommendations
    
    def generate_export_control_report(self, export_manager) -> ComplianceReport:
        """Generate export control compliance report."""
        report_id = f"export_control_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Get export audit trail
        export_trail = export_manager.get_export_audit_trail()
        
        # Calculate export control metrics
        total_evaluations = len(export_trail)
        approved_exports = sum(1 for eval in export_trail if eval.get("export_allowed", False))
        denied_exports = total_evaluations - approved_exports
        
        classification_breakdown = {}
        for eval in export_trail:
            level = eval.get("classification_level", "unknown")
            classification_breakdown[level] = classification_breakdown.get(level, 0) + 1
        
        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            report_type="EXPORT_CONTROL",
            time_period={
                "start": datetime.utcnow() - timedelta(days=30),
                "end": datetime.utcnow()
            },
            summary={
                "total_export_evaluations": total_evaluations,
                "approved_exports": approved_exports,
                "denied_exports": denied_exports,
                "approval_rate": (approved_exports / max(total_evaluations, 1)) * 100,
                "classification_breakdown": classification_breakdown
            },
            export_controls={
                "policy_name": export_manager.policy.name,
                "policy_version": export_manager.policy.version,
                "redaction_rules": len(export_manager.policy.redaction_rules),
                "classification_rules": len(export_manager.policy.classification_rules)
            },
            audit_events=export_trail,
            recommendations=self._generate_export_recommendations(export_trail),
            metadata={
                "compliance_framework": "EXPORT_CONTROL",
                "reporting_standard": "ITAR_EAR_COMPLIANCE"
            }
        )
        
        return report
    
    def _generate_export_recommendations(self, export_trail: List[Dict[str, Any]]) -> List[str]:
        """Generate export control recommendations."""
        recommendations = []
        
        denied_count = sum(1 for eval in export_trail if not eval.get("export_allowed", True))
        total_count = len(export_trail)
        
        if denied_count / max(total_count, 1) > 0.1:  # >10% denial rate
            recommendations.append("Review export control policies for potential over-restriction")
        
        restricted_count = sum(1 for eval in export_trail if eval.get("classification_level") == "restricted")
        if restricted_count / max(total_count, 1) > 0.2:  # >20% restricted content
            recommendations.append("Consider additional user training on content classification")
        
        redaction_count = sum(len(eval.get("redactions_applied", [])) for eval in export_trail)
        if redaction_count > total_count * 0.5:  # Average >0.5 redactions per evaluation
            recommendations.append("Evaluate redaction patterns for optimization")
        
        return recommendations
    
    def save_report(self, report: ComplianceReport, format: str = "json") -> str:
        """Save compliance report to file."""
        filename = f"{report.report_id}.{format}"
        filepath = self.storage_dir / filename
        
        if format == "json":
            # Convert datetime objects to ISO strings for JSON serialization
            report_dict = self._serialize_report(report)
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Compliance report saved: {filepath}")
        return str(filepath)
    
    def _serialize_report(self, report: ComplianceReport) -> Dict[str, Any]:
        """Serialize report for JSON output."""
        return {
            "report_id": report.report_id,
            "generated_at": report.generated_at.isoformat(),
            "report_type": report.report_type,
            "time_period": {
                "start": report.time_period["start"].isoformat(),
                "end": report.time_period["end"].isoformat()
            },
            "summary": report.summary,
            "export_controls": report.export_controls,
            "audit_events": report.audit_events,
            "violations": report.violations,
            "recommendations": report.recommendations,
            "metadata": report.metadata
        }
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard."""
        recent_events = [
            event for event in self._audit_events
            if datetime.fromisoformat(event["timestamp"]) > datetime.utcnow() - timedelta(days=7)
        ]
        
        return {
            "total_audit_events": len(self._audit_events),
            "recent_events": len(recent_events),
            "compliance_flags": {
                flag: sum(1 for event in recent_events if flag in event.get("compliance_flags", []))
                for flag in ["SOC2_SECURITY", "DATA_PRIVACY", "EXPORT_CONTROL", "HIGH_RISK", "POLICY_VIOLATION"]
            },
            "event_types": {
                event_type: sum(1 for event in recent_events if event["event_type"] == event_type)
                for event_type in set(event["event_type"] for event in recent_events)
            },
            "last_updated": datetime.utcnow().isoformat()
        }