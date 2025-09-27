"""Risk assessment and analysis system."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .registry import Risk, RiskCategory, RiskLevel, RiskStatus

logger = logging.getLogger("agentnet.risk.assessment")


@dataclass
class RiskAssessment:
    """Result of a risk assessment."""
    
    assessment_id: str
    risk_id: str
    assessed_by: str
    assessment_date: datetime
    
    # Current assessment
    current_probability: float
    current_impact: float
    current_risk_score: float
    current_level: RiskLevel
    
    # Factors analysis
    probability_factors: List[str]
    impact_factors: List[str]
    environmental_factors: List[str]
    
    # Recommendations
    recommended_status: RiskStatus
    recommended_actions: List[str]
    priority_score: float  # 0-100
    
    # Context
    assessment_context: Dict[str, Any]
    notes: str = ""
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.current_risk_score = self.current_probability * self.current_impact
        
        # Determine level based on risk score
        if self.current_risk_score >= 0.8:
            self.current_level = RiskLevel.CRITICAL
        elif self.current_risk_score >= 0.6:
            self.current_level = RiskLevel.HIGH
        elif self.current_risk_score >= 0.3:
            self.current_level = RiskLevel.MEDIUM
        else:
            self.current_level = RiskLevel.LOW


class RiskAssessor:
    """Automated and manual risk assessment system."""
    
    def __init__(self):
        self.assessment_history: Dict[str, List[RiskAssessment]] = {}
    
    def assess_risk(
        self,
        risk: Risk,
        context: Dict[str, Any] = None,
        assessor: str = "system",
    ) -> RiskAssessment:
        """Perform comprehensive risk assessment."""
        
        context = context or {}
        assessment_id = f"ASSESS_{risk.risk_id}_{int(datetime.now().timestamp())}"
        
        # Analyze current state
        probability_analysis = self._assess_probability(risk, context)
        impact_analysis = self._assess_impact(risk, context) 
        environmental_factors = self._assess_environmental_factors(risk, context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk, probability_analysis, impact_analysis, context)
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(
            probability_analysis["probability"],
            impact_analysis["impact"],
            risk,
            environmental_factors,
        )
        
        assessment = RiskAssessment(
            assessment_id=assessment_id,
            risk_id=risk.risk_id,
            assessed_by=assessor,
            assessment_date=datetime.now(),
            current_probability=probability_analysis["probability"],
            current_impact=impact_analysis["impact"],
            current_risk_score=probability_analysis["probability"] * impact_analysis["impact"],
            current_level=RiskLevel.LOW,  # Will be set in __post_init__
            probability_factors=probability_analysis["factors"],
            impact_factors=impact_analysis["factors"],
            environmental_factors=environmental_factors,
            recommended_status=recommendations["status"],
            recommended_actions=recommendations["actions"],
            priority_score=priority_score,
            assessment_context=context,
            notes=recommendations["notes"],
        )
        
        # Store assessment
        if risk.risk_id not in self.assessment_history:
            self.assessment_history[risk.risk_id] = []
        self.assessment_history[risk.risk_id].append(assessment)
        
        logger.info(f"Completed assessment {assessment_id} for risk {risk.risk_id}")
        return assessment
    
    def _assess_probability(self, risk: Risk, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current probability of risk occurrence."""
        
        base_probability = risk.probability
        probability_factors = []
        adjustments = 0.0
        
        # Category-specific probability assessment
        if risk.category == RiskCategory.PROVIDER_OUTAGE:
            # Check recent outage patterns
            recent_outages = context.get("recent_provider_outages", 0)
            if recent_outages > 0:
                adjustments += 0.1 * recent_outages
                probability_factors.append(f"Recent outages: {recent_outages}")
            
            # Check provider diversity
            provider_count = context.get("active_providers", 1)
            if provider_count == 1:
                adjustments += 0.2
                probability_factors.append("Single provider dependency")
        
        elif risk.category == RiskCategory.COST_SPIKE:
            # Check cost trends
            cost_trend = context.get("cost_trend", "stable")
            if cost_trend == "increasing":
                adjustments += 0.3
                probability_factors.append("Increasing cost trend")
            
            # Check usage patterns
            usage_volatility = context.get("usage_volatility", 0.0)
            if usage_volatility > 0.5:
                adjustments += 0.2
                probability_factors.append("High usage volatility")
        
        elif risk.category == RiskCategory.MEMORY_BLOAT:
            # Check memory usage trends
            memory_usage = context.get("avg_memory_usage_mb", 0)
            if memory_usage > 256:
                adjustments += 0.3
                probability_factors.append(f"High memory usage: {memory_usage}MB")
            
            # Check session duration
            avg_session_duration = context.get("avg_session_duration_minutes", 0)
            if avg_session_duration > 30:
                adjustments += 0.2
                probability_factors.append(f"Long sessions: {avg_session_duration}min avg")
        
        elif risk.category == RiskCategory.TOOL_INJECTION:
            # Check input validation controls
            has_input_validation = context.get("has_input_validation", True)
            if not has_input_validation:
                adjustments += 0.4
                probability_factors.append("Missing input validation")
            
            # Check tool sandboxing
            has_sandboxing = context.get("has_tool_sandboxing", True) 
            if not has_sandboxing:
                adjustments += 0.3
                probability_factors.append("Missing tool sandboxing")
        
        elif risk.category == RiskCategory.CONVERGENCE_STALL:
            # Check reasoning complexity
            avg_reasoning_steps = context.get("avg_reasoning_steps", 5)
            if avg_reasoning_steps > 20:
                adjustments += 0.3
                probability_factors.append(f"High reasoning complexity: {avg_reasoning_steps} steps avg")
            
            # Check timeout enforcement
            has_timeouts = context.get("has_reasoning_timeouts", True)
            if not has_timeouts:
                adjustments += 0.2
                probability_factors.append("Missing timeout controls")
        
        # Environmental factors
        system_load = context.get("system_load", 0.5)
        if system_load > 0.8:
            adjustments += 0.1
            probability_factors.append("High system load")
        
        # Calculate final probability
        final_probability = max(0.0, min(1.0, base_probability + adjustments))
        
        return {
            "probability": final_probability,
            "base_probability": base_probability,
            "adjustments": adjustments,
            "factors": probability_factors,
        }
    
    def _assess_impact(self, risk: Risk, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current impact if risk materializes."""
        
        base_impact = risk.impact
        impact_factors = []
        adjustments = 0.0
        
        # Business context impact adjustments
        business_criticality = context.get("business_criticality", "medium")
        if business_criticality == "critical":
            adjustments += 0.2
            impact_factors.append("Critical business operation")
        elif business_criticality == "low":
            adjustments -= 0.1
            impact_factors.append("Low business criticality")
        
        # User base impact
        active_users = context.get("active_users", 0)
        if active_users > 1000:
            adjustments += 0.2
            impact_factors.append(f"Large user base: {active_users}")
        elif active_users > 100:
            adjustments += 0.1
            impact_factors.append(f"Medium user base: {active_users}")
        
        # Financial impact context
        monthly_budget = context.get("monthly_budget", 0)
        if risk.category == RiskCategory.COST_SPIKE and monthly_budget > 0:
            potential_overrun = context.get("potential_cost_overrun_percent", 0)
            if potential_overrun > 50:
                adjustments += 0.3
                impact_factors.append(f"Major budget overrun risk: {potential_overrun}%")
            elif potential_overrun > 20:
                adjustments += 0.1
                impact_factors.append(f"Moderate budget overrun risk: {potential_overrun}%")
        
        # Compliance and regulatory impact
        has_compliance_requirements = context.get("has_compliance_requirements", False)
        if has_compliance_requirements and risk.category in {RiskCategory.PROMPT_LEAKAGE, RiskCategory.TOOL_INJECTION}:
            adjustments += 0.3
            impact_factors.append("Regulatory compliance impact")
        
        # Recovery time impact
        estimated_recovery_hours = context.get("estimated_recovery_hours", 1)
        if estimated_recovery_hours > 24:
            adjustments += 0.2
            impact_factors.append(f"Long recovery time: {estimated_recovery_hours}h")
        elif estimated_recovery_hours > 4:
            adjustments += 0.1
            impact_factors.append(f"Moderate recovery time: {estimated_recovery_hours}h")
        
        # Calculate final impact
        final_impact = max(0.0, min(1.0, base_impact + adjustments))
        
        return {
            "impact": final_impact,
            "base_impact": base_impact,
            "adjustments": adjustments,
            "factors": impact_factors,
        }
    
    def _assess_environmental_factors(self, risk: Risk, context: Dict[str, Any]) -> List[str]:
        """Assess environmental factors that might influence risk."""
        
        factors = []
        
        # System environment
        deployment_environment = context.get("environment", "production")
        if deployment_environment == "production":
            factors.append("Production environment - higher impact potential")
        elif deployment_environment == "development":
            factors.append("Development environment - lower impact scope")
        
        # Monitoring capabilities
        has_monitoring = context.get("has_monitoring", True)
        if has_monitoring:
            factors.append("Active monitoring in place")
        else:
            factors.append("Limited monitoring - reduced visibility")
        
        # Backup and recovery
        has_backups = context.get("has_backups", True)
        if has_backups:
            factors.append("Backup systems available")
        else:
            factors.append("No backup systems - increased recovery risk")
        
        # Team readiness
        has_on_call = context.get("has_on_call_team", True)
        if has_on_call:
            factors.append("On-call team available for response")
        else:
            factors.append("No dedicated response team")
        
        # Documentation
        has_runbooks = context.get("has_incident_runbooks", False)
        if has_runbooks:
            factors.append("Incident response procedures documented")
        else:
            factors.append("Limited incident response documentation")
        
        return factors
    
    def _generate_recommendations(
        self,
        risk: Risk,
        probability_analysis: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate recommendations based on assessment."""
        
        current_probability = probability_analysis["probability"]
        current_impact = impact_analysis["impact"]
        risk_score = current_probability * current_impact
        
        # Determine recommended status
        if risk_score >= 0.8:
            recommended_status = RiskStatus.MITIGATING
        elif risk_score >= 0.6:
            recommended_status = RiskStatus.ASSESSING if risk.status == RiskStatus.IDENTIFIED else RiskStatus.MITIGATING
        elif risk_score >= 0.3:
            recommended_status = RiskStatus.MONITORING
        else:
            recommended_status = RiskStatus.ACCEPTED
        
        # Generate specific actions
        actions = []
        notes = []
        
        # High priority actions
        if risk_score >= 0.7:
            actions.append("Implement immediate mitigation measures")
            actions.append("Escalate to management for resource allocation")
            actions.append("Establish daily monitoring and reporting")
        
        # Category-specific recommendations
        if risk.category == RiskCategory.PROVIDER_OUTAGE:
            if "Single provider dependency" in probability_analysis["factors"]:
                actions.append("Implement multi-provider fallback system")
            actions.append("Enhance provider health monitoring")
            actions.append("Test failover procedures")
        
        elif risk.category == RiskCategory.COST_SPIKE:
            if "Increasing cost trend" in probability_analysis["factors"]:
                actions.append("Implement cost alerts at 75% and 90% budget thresholds")
                actions.append("Review and optimize high-cost operations")
            actions.append("Establish cost governance policies")
        
        elif risk.category == RiskCategory.TOOL_INJECTION:
            if "Missing input validation" in probability_analysis["factors"]:
                actions.append("Implement comprehensive input validation")
            if "Missing tool sandboxing" in probability_analysis["factors"]:
                actions.append("Deploy tool execution sandboxing")
            actions.append("Conduct security penetration testing")
        
        # Environmental improvement recommendations
        environmental_factors = self._assess_environmental_factors(risk, context)
        if "Limited monitoring - reduced visibility" in environmental_factors:
            actions.append("Implement comprehensive monitoring and alerting")
        if "No backup systems - increased recovery risk" in environmental_factors:
            actions.append("Establish backup and recovery procedures")
        
        # Generate summary notes
        if risk_score >= 0.8:
            notes.append("CRITICAL: Immediate action required due to high risk score")
        elif risk_score >= 0.6:
            notes.append("HIGH: Priority mitigation needed")
        elif current_probability > 0.7:
            notes.append("High likelihood of occurrence - monitoring recommended")
        elif current_impact > 0.7:
            notes.append("High potential impact - preventive measures recommended")
        
        return {
            "status": recommended_status,
            "actions": actions,
            "notes": ". ".join(notes),
        }
    
    def _calculate_priority_score(
        self,
        probability: float,
        impact: float, 
        risk: Risk,
        environmental_factors: List[str],
    ) -> float:
        """Calculate priority score (0-100) for risk handling."""
        
        # Base score from risk score (0-1 to 0-70)
        risk_score = probability * impact
        base_score = risk_score * 70
        
        # Category weighting (0-15 points)
        category_weights = {
            RiskCategory.TOOL_INJECTION: 15,
            RiskCategory.PROMPT_LEAKAGE: 12,
            RiskCategory.PROVIDER_OUTAGE: 10,
            RiskCategory.COST_SPIKE: 8,
            RiskCategory.MEMORY_BLOAT: 6,
            RiskCategory.CONVERGENCE_STALL: 5,
            RiskCategory.POLICY_FALSE_POSITIVE: 4,
        }
        category_score = category_weights.get(risk.category, 5)
        
        # Environmental factors (0-15 points)
        env_score = 0
        high_impact_factors = [
            "Production environment - higher impact potential",
            "Large user base",
            "Regulatory compliance impact",
            "No backup systems - increased recovery risk",
            "Limited monitoring - reduced visibility",
        ]
        
        for factor in environmental_factors:
            if any(high_factor in factor for high_factor in high_impact_factors):
                env_score += 3
                
        env_score = min(env_score, 15)
        
        # Current status penalty (reduce priority if already being handled)
        status_adjustments = {
            RiskStatus.IDENTIFIED: 0,
            RiskStatus.ASSESSING: -5,
            RiskStatus.MITIGATING: -10,
            RiskStatus.MONITORING: -15,
            RiskStatus.ACCEPTED: -30,
            RiskStatus.CLOSED: -50,
        }
        status_adjustment = status_adjustments.get(risk.status, 0)
        
        # Calculate final priority score
        priority_score = base_score + category_score + env_score + status_adjustment
        return max(0, min(100, priority_score))
    
    def get_assessment_history(self, risk_id: str) -> List[RiskAssessment]:
        """Get assessment history for a risk."""
        return self.assessment_history.get(risk_id, [])
    
    def get_latest_assessment(self, risk_id: str) -> Optional[RiskAssessment]:
        """Get the most recent assessment for a risk."""
        history = self.get_assessment_history(risk_id)
        return history[-1] if history else None
    
    def compare_assessments(
        self, 
        risk_id: str, 
        assessment1_id: str, 
        assessment2_id: str,
    ) -> Dict[str, Any]:
        """Compare two assessments of the same risk."""
        
        history = self.get_assessment_history(risk_id)
        
        assessment1 = next((a for a in history if a.assessment_id == assessment1_id), None)
        assessment2 = next((a for a in history if a.assessment_id == assessment2_id), None)
        
        if not assessment1 or not assessment2:
            raise ValueError("One or both assessments not found")
        
        return {
            "risk_id": risk_id,
            "assessment1": {
                "id": assessment1.assessment_id,
                "date": assessment1.assessment_date.isoformat(),
                "probability": assessment1.current_probability,
                "impact": assessment1.current_impact,
                "risk_score": assessment1.current_risk_score,
                "level": assessment1.current_level.value,
                "priority_score": assessment1.priority_score,
            },
            "assessment2": {
                "id": assessment2.assessment_id,
                "date": assessment2.assessment_date.isoformat(),
                "probability": assessment2.current_probability,
                "impact": assessment2.current_impact,
                "risk_score": assessment2.current_risk_score,
                "level": assessment2.current_level.value,
                "priority_score": assessment2.priority_score,
            },
            "changes": {
                "probability_change": assessment2.current_probability - assessment1.current_probability,
                "impact_change": assessment2.current_impact - assessment1.current_impact,
                "risk_score_change": assessment2.current_risk_score - assessment1.current_risk_score,
                "priority_change": assessment2.priority_score - assessment1.priority_score,
                "level_change": f"{assessment1.current_level.value} → {assessment2.current_level.value}",
            },
        }