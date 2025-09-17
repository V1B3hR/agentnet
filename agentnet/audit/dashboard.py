"""Audit Dashboard Implementation for P6 Enterprise Hardening.

This module provides audit trail visualization and compliance dashboards
for enterprise security monitoring and SOC2 compliance reporting.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .storage import AuditStorage, AuditQuery
from .workflow import AuditEventType, AuditSeverity


class AuditDashboard:
    """Enterprise audit dashboard for compliance visualization."""
    
    def __init__(self, audit_storage: AuditStorage):
        self.storage = audit_storage
    
    def generate_compliance_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive compliance dashboard data."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Get event statistics
        stats = self.storage.get_event_statistics(start_time, end_time)
        
        # Get high-risk events
        high_risk_query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            severity_levels=[AuditSeverity.HIGH, AuditSeverity.CRITICAL],
            limit=50
        )
        high_risk_events = self.storage.get_events(high_risk_query)
        
        # Get security incidents
        security_query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            event_types=[AuditEventType.SECURITY_ALERT, AuditEventType.POLICY_VIOLATION],
            limit=25
        )
        security_incidents = self.storage.get_events(security_query)
        
        # Calculate compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(stats, high_risk_events)
        
        # Generate trend data
        trend_data = self._generate_trend_data(start_time, end_time)
        
        dashboard_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "time_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "summary": {
                "total_events": stats["total_events"],
                "high_risk_events": len(high_risk_events),
                "security_incidents": len(security_incidents),
                "compliance_score": compliance_metrics["overall_score"]
            },
            "statistics": stats,
            "compliance_metrics": compliance_metrics,
            "high_risk_events": [event.to_dict() for event in high_risk_events[:10]],
            "security_incidents": [event.to_dict() for event in security_incidents[:10]],
            "trends": trend_data,
            "recommendations": self._generate_recommendations(stats, high_risk_events)
        }
        
        return dashboard_data
    
    def _calculate_compliance_metrics(self, stats: Dict[str, Any], high_risk_events: List) -> Dict[str, Any]:
        """Calculate compliance-specific metrics."""
        total_events = stats["total_events"]
        high_risk_count = len(high_risk_events)
        
        # SOC2 metrics
        soc2_events = stats["compliance_tags"].get("SOC2_SECURITY", 0) + \
                      stats["compliance_tags"].get("SOC2_CONFIDENTIALITY", 0) + \
                      stats["compliance_tags"].get("SOC2_PROCESSING_INTEGRITY", 0)
        
        security_incidents = stats["compliance_tags"].get("SECURITY_INCIDENT", 0)
        policy_violations = stats["events_by_type"].get("policy_violation", 0)
        
        # Calculate scores (0-100)
        security_score = max(0, 100 - (security_incidents * 10))
        availability_score = 99.9  # Would be calculated from actual uptime metrics
        processing_integrity_score = max(0, 100 - (policy_violations / max(total_events, 1) * 100))
        confidentiality_score = max(0, 100 - (high_risk_count * 5))
        privacy_score = max(0, 100 - (stats["compliance_tags"].get("GDPR", 0) * 2))
        
        overall_score = (security_score + availability_score + processing_integrity_score + 
                        confidentiality_score + privacy_score) / 5
        
        return {
            "overall_score": round(overall_score, 1),
            "soc2_metrics": {
                "security": round(security_score, 1),
                "availability": round(availability_score, 1),
                "processing_integrity": round(processing_integrity_score, 1),
                "confidentiality": round(confidentiality_score, 1),
                "privacy": round(privacy_score, 1)
            },
            "incident_counts": {
                "security_incidents": security_incidents,
                "policy_violations": policy_violations,
                "high_risk_events": high_risk_count
            },
            "compliance_events": {
                "soc2_total": soc2_events,
                "gdpr_events": stats["compliance_tags"].get("GDPR", 0),
                "export_control_events": stats["compliance_tags"].get("EXPORT_CONTROL", 0)
            }
        }
    
    def _generate_trend_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate trend data for dashboard visualization."""
        # Generate daily event counts
        daily_counts = {}
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            query = AuditQuery(start_time=day_start, end_time=day_end, limit=10000)
            events = self.storage.get_events(query)
            
            daily_counts[current_date.isoformat()] = {
                "total": len(events),
                "high_risk": len([e for e in events if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]]),
                "security": len([e for e in events if e.event_type == AuditEventType.SECURITY_ALERT]),
                "violations": len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION])
            }
            
            current_date += timedelta(days=1)
        
        return {
            "daily_events": daily_counts,
            "event_type_trends": self._calculate_event_type_trends(start_time, end_time),
            "severity_trends": self._calculate_severity_trends(start_time, end_time)
        }
    
    def _calculate_event_type_trends(self, start_time: datetime, end_time: datetime) -> Dict[str, List]:
        """Calculate event type trends over time."""
        # This would typically use more sophisticated time-series analysis
        # For now, return basic hourly aggregation
        trends = {}
        for event_type in AuditEventType:
            query = AuditQuery(
                start_time=start_time,
                end_time=end_time,
                event_types=[event_type],
                limit=1000
            )
            events = self.storage.get_events(query)
            trends[event_type.value] = len(events)
        
        return trends
    
    def _calculate_severity_trends(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Calculate severity level trends."""
        trends = {}
        for severity in AuditSeverity:
            query = AuditQuery(
                start_time=start_time,
                end_time=end_time,
                severity_levels=[severity],
                limit=1000
            )
            events = self.storage.get_events(query)
            trends[severity.value] = len(events)
        
        return trends
    
    def _generate_recommendations(self, stats: Dict[str, Any], high_risk_events: List) -> List[str]:
        """Generate compliance recommendations based on audit data."""
        recommendations = []
        
        total_events = stats["total_events"]
        high_risk_count = len(high_risk_events)
        security_incidents = stats["compliance_tags"].get("SECURITY_INCIDENT", 0)
        policy_violations = stats["events_by_type"].get("policy_violation", 0)
        
        # High-risk event recommendations
        if high_risk_count / max(total_events, 1) > 0.1:  # >10% high-risk
            recommendations.append("High proportion of high-risk events detected. Review security policies and incident response procedures.")
        
        # Security incident recommendations
        if security_incidents > 5:
            recommendations.append("Multiple security incidents detected. Consider implementing additional security controls.")
        
        # Policy violation recommendations
        if policy_violations > total_events * 0.05:  # >5% violations
            recommendations.append("Elevated policy violation rate. Review and update governance policies.")
        
        # User activity recommendations
        if len(stats["top_users"]) < 3 and total_events > 50:
            recommendations.append("Limited user activity diversity. Verify user authentication and access controls.")
        
        # Compliance tag recommendations
        if stats["compliance_tags"].get("GDPR", 0) > 0:
            recommendations.append("GDPR-related events detected. Ensure data privacy compliance procedures are followed.")
        
        if stats["compliance_tags"].get("EXPORT_CONTROL", 0) > 10:
            recommendations.append("Significant export control activity. Review export control policies and procedures.")
        
        # Default recommendation if no issues found
        if not recommendations:
            recommendations.append("Audit activity appears normal. Continue monitoring for compliance adherence.")
        
        return recommendations
    
    def generate_soc2_dashboard(self) -> Dict[str, Any]:
        """Generate SOC2-specific compliance dashboard."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)  # Monthly SOC2 view
        
        # Get SOC2-relevant events
        soc2_query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            compliance_tags=["SOC2_SECURITY", "SOC2_CONFIDENTIALITY", "SOC2_PROCESSING_INTEGRITY"],
            limit=1000
        )
        soc2_events = self.storage.get_events(soc2_query)
        
        # Calculate SOC2 Trust Service Criteria metrics
        trust_criteria = {
            "security": self._evaluate_security_criteria(soc2_events),
            "availability": {"score": 99.9, "incidents": 0},  # Would come from system monitoring
            "processing_integrity": self._evaluate_processing_integrity(soc2_events),
            "confidentiality": self._evaluate_confidentiality_criteria(soc2_events),
            "privacy": self._evaluate_privacy_criteria(soc2_events)
        }
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "assessment_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "soc2_events_count": len(soc2_events),
            "trust_service_criteria": trust_criteria,
            "overall_compliance_score": sum(criteria["score"] for criteria in trust_criteria.values()) / len(trust_criteria),
            "recommendations": self._generate_soc2_recommendations(trust_criteria)
        }
    
    def _evaluate_security_criteria(self, events: List) -> Dict[str, Any]:
        """Evaluate SOC2 security criteria."""
        security_events = [e for e in events if "SOC2_SECURITY" in e.compliance_tags]
        security_incidents = [e for e in events if e.event_type == AuditEventType.SECURITY_ALERT]
        
        score = max(0, 100 - len(security_incidents) * 5)
        
        return {
            "score": score,
            "events_count": len(security_events),
            "incidents_count": len(security_incidents),
            "status": "COMPLIANT" if score > 90 else "NEEDS_ATTENTION"
        }
    
    def _evaluate_processing_integrity(self, events: List) -> Dict[str, Any]:
        """Evaluate SOC2 processing integrity criteria."""
        integrity_events = [e for e in events if "SOC2_PROCESSING_INTEGRITY" in e.compliance_tags]
        policy_violations = [e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION]
        
        score = max(0, 100 - len(policy_violations) * 2)
        
        return {
            "score": score,
            "events_count": len(integrity_events),
            "violations_count": len(policy_violations),
            "status": "COMPLIANT" if score > 95 else "NEEDS_ATTENTION"
        }
    
    def _evaluate_confidentiality_criteria(self, events: List) -> Dict[str, Any]:
        """Evaluate SOC2 confidentiality criteria."""
        confidentiality_events = [e for e in events if "SOC2_CONFIDENTIALITY" in e.compliance_tags]
        high_risk_data_events = [e for e in events if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]]
        
        score = max(0, 100 - len(high_risk_data_events) * 3)
        
        return {
            "score": score,
            "events_count": len(confidentiality_events),
            "high_risk_events": len(high_risk_data_events),
            "status": "COMPLIANT" if score > 92 else "NEEDS_ATTENTION"
        }
    
    def _evaluate_privacy_criteria(self, events: List) -> Dict[str, Any]:
        """Evaluate SOC2 privacy criteria."""
        privacy_events = [e for e in events if "GDPR" in e.compliance_tags]
        data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
        
        score = max(0, 100 - len(privacy_events) * 1)
        
        return {
            "score": score,
            "privacy_events": len(privacy_events),
            "data_access_events": len(data_access_events),
            "status": "COMPLIANT" if score > 95 else "NEEDS_ATTENTION"
        }
    
    def _generate_soc2_recommendations(self, criteria: Dict[str, Any]) -> List[str]:
        """Generate SOC2-specific recommendations."""
        recommendations = []
        
        for criterion_name, criterion_data in criteria.items():
            if criterion_data["score"] < 90:
                recommendations.append(f"Improve {criterion_name} controls - current score: {criterion_data['score']}")
        
        return recommendations or ["SOC2 compliance criteria are being met satisfactorily."]
    
    def export_dashboard(self, filepath: str, dashboard_type: str = "compliance") -> None:
        """Export dashboard data to file."""
        if dashboard_type == "compliance":
            dashboard_data = self.generate_compliance_dashboard()
        elif dashboard_type == "soc2":
            dashboard_data = self.generate_soc2_dashboard()
        else:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
        
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
    
    def generate_html_dashboard(self, dashboard_type: str = "compliance") -> str:
        """Generate HTML dashboard for web viewing."""
        if dashboard_type == "compliance":
            data = self.generate_compliance_dashboard()
        elif dashboard_type == "soc2":
            data = self.generate_soc2_dashboard()
        else:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgentNet P6 {dashboard_type.title()} Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .recommendations {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .event-list {{ max-height: 300px; overflow-y: auto; }}
                .event-item {{ border-bottom: 1px solid #eee; padding: 5px 0; }}
                .high-risk {{ color: #f44336; }}
                .medium-risk {{ color: #ff9800; }}
                .low-risk {{ color: #4caf50; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>AgentNet P6 {dashboard_type.title()} Dashboard</h1>
                    <p>Generated: {data.get('generated_at', 'Unknown')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{data.get('summary', {}).get('total_events', 0)}</div>
                        <div class="metric-label">Total Events</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{data.get('summary', {}).get('high_risk_events', 0)}</div>
                        <div class="metric-label">High Risk Events</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{data.get('summary', {}).get('compliance_score', 0):.1f}%</div>
                        <div class="metric-label">Compliance Score</div>
                    </div>
                </div>
                
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in data.get('recommendations', []))}
                    </ul>
                </div>
                
                <div class="event-details">
                    <h3>Recent High-Risk Events</h3>
                    <div class="event-list">
                        {''.join(f'<div class="event-item high-risk">{event.get("action", "Unknown")} - {event.get("timestamp", "Unknown")}</div>' 
                                for event in data.get('high_risk_events', [])[:10])}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template