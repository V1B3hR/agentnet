"""Risk management workflow integration with CI/CD and deployment processes."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .registry import Risk, RiskRegistry, RiskCategory, RiskLevel, RiskStatus
from .assessment import RiskAssessor, RiskAssessment
from .mitigation import MitigationMitigator, MitigationStrategy
from .monitoring import RiskMonitor, RiskAlert

logger = logging.getLogger("agentnet.risk.workflow")


class WorkflowStage(Enum):
    """CI/CD workflow stages where risk management integrates."""
    PRE_COMMIT = "pre_commit"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY_STAGING = "deploy_staging"
    DEPLOY_PRODUCTION = "deploy_production"
    POST_DEPLOY = "post_deploy"
    MONITORING = "monitoring"


class RiskDecision(Enum):
    """Risk-based decisions for workflow gates."""
    PROCEED = "proceed"
    PROCEED_WITH_WARNING = "proceed_with_warning"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    ROLLBACK = "rollback"


@dataclass
class WorkflowContext:
    """Context information for workflow risk assessment."""
    
    stage: WorkflowStage
    environment: str  # dev, staging, production
    change_type: str  # feature, bugfix, hotfix, security
    commit_sha: str
    pull_request_id: Optional[str] = None
    deployment_id: Optional[str] = None
    
    # Change analysis
    files_changed: List[str] = field(default_factory=list)
    lines_changed: int = 0
    test_coverage: float = 0.0
    security_scan_results: Dict[str, Any] = field(default_factory=dict)
    
    # Business context
    business_hours: bool = True
    maintenance_window: bool = False
    high_traffic_period: bool = False
    
    # System state
    system_health: Dict[str, Any] = field(default_factory=dict)
    active_incidents: List[str] = field(default_factory=list)
    
    # Metadata
    triggered_by: str = "system"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGate:
    """Risk-based gate in CI/CD workflow."""
    
    gate_id: str
    stage: WorkflowStage
    name: str
    description: str
    
    # Gate configuration
    risk_threshold: float = 0.6  # Block if risk score > threshold
    approval_threshold: float = 0.4  # Require approval if risk score > threshold
    required_mitigations: List[str] = field(default_factory=list)
    
    # Gate logic
    assessment_rules: List[Callable[[WorkflowContext], Dict[str, Any]]] = field(default_factory=list)
    blocking_conditions: List[Callable[[WorkflowContext, List[Risk]], bool]] = field(default_factory=list)
    
    # Configuration
    enabled: bool = True
    bypass_allowed: bool = False
    bypass_approvers: List[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result of workflow risk assessment."""
    
    workflow_id: str
    stage: WorkflowStage
    decision: RiskDecision
    risk_score: float
    
    # Assessment results
    risks_identified: List[str]  # Risk IDs
    mitigations_required: List[str]  # Mitigation strategy IDs
    approval_required: bool = False
    required_approvers: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    assessment_time: datetime = field(default_factory=datetime.now)
    assessed_by: str = "system"
    context: Optional[WorkflowContext] = None


class RiskWorkflow:
    """Main risk management workflow integration system."""
    
    def __init__(
        self,
        risk_registry: RiskRegistry,
        risk_assessor: RiskAssessor,
        mitigator: MitigationMitigator,
        monitor: RiskMonitor,
    ):
        self.risk_registry = risk_registry
        self.risk_assessor = risk_assessor
        self.mitigator = mitigator
        self.monitor = monitor
        
        self.workflow_gates: Dict[WorkflowStage, List[WorkflowGate]] = {}
        self.workflow_history: List[WorkflowResult] = []
        self.approval_handlers: Dict[str, Callable] = {}
        
        # Initialize default workflow gates
        self._setup_default_gates()
    
    def _setup_default_gates(self):
        """Set up default risk gates for CI/CD workflow."""
        
        # Pre-commit gate
        pre_commit_gate = WorkflowGate(
            gate_id="pre_commit_risk_check",
            stage=WorkflowStage.PRE_COMMIT,
            name="Pre-Commit Risk Check",
            description="Assess risks before code commit",
            risk_threshold=0.8,
            approval_threshold=0.6,
        )
        pre_commit_gate.assessment_rules.append(self._assess_code_change_risks)
        pre_commit_gate.blocking_conditions.append(self._check_security_file_changes)
        
        # Build gate
        build_gate = WorkflowGate(
            gate_id="build_risk_assessment",
            stage=WorkflowStage.BUILD,
            name="Build Risk Assessment",
            description="Assess build and dependency risks",
            risk_threshold=0.7,
            approval_threshold=0.5,
        )
        build_gate.assessment_rules.append(self._assess_build_risks)
        
        # Security scan gate
        security_gate = WorkflowGate(
            gate_id="security_scan_gate",
            stage=WorkflowStage.SECURITY_SCAN,
            name="Security Scan Gate",
            description="Block on critical security findings",
            risk_threshold=0.9,
            approval_threshold=0.7,
        )
        security_gate.assessment_rules.append(self._assess_security_scan_risks)
        security_gate.blocking_conditions.append(self._check_critical_vulnerabilities)
        
        # Production deployment gate
        prod_deploy_gate = WorkflowGate(
            gate_id="production_deployment_gate",
            stage=WorkflowStage.DEPLOY_PRODUCTION,
            name="Production Deployment Gate",
            description="Comprehensive risk assessment for production deployment",
            risk_threshold=0.6,
            approval_threshold=0.3,
            required_mitigations=["provider_fallback", "rollback_plan"],
        )
        prod_deploy_gate.assessment_rules.append(self._assess_deployment_risks)
        prod_deploy_gate.blocking_conditions.append(self._check_production_readiness)
        
        # Register gates
        self.workflow_gates[WorkflowStage.PRE_COMMIT] = [pre_commit_gate]
        self.workflow_gates[WorkflowStage.BUILD] = [build_gate]
        self.workflow_gates[WorkflowStage.SECURITY_SCAN] = [security_gate]
        self.workflow_gates[WorkflowStage.DEPLOY_PRODUCTION] = [prod_deploy_gate]
    
    def assess_workflow_stage(
        self,
        context: WorkflowContext,
        workflow_id: Optional[str] = None,
    ) -> WorkflowResult:
        """Assess risks for a specific workflow stage."""
        
        workflow_id = workflow_id or f"WF_{int(datetime.now().timestamp())}"
        
        # Get gates for this stage
        gates = self.workflow_gates.get(context.stage, [])
        if not gates:
            # No gates configured for this stage - proceed
            return WorkflowResult(
                workflow_id=workflow_id,
                stage=context.stage,
                decision=RiskDecision.PROCEED,
                risk_score=0.0,
                risks_identified=[],
                mitigations_required=[],
                context=context,
            )
        
        # Process each gate
        overall_risk_score = 0.0
        all_risks_identified = []
        all_mitigations_required = []
        all_recommendations = []
        all_warnings = []
        requires_approval = False
        required_approvers = []
        final_decision = RiskDecision.PROCEED
        
        for gate in gates:
            if not gate.enabled:
                continue
            
            logger.info(f"Processing workflow gate: {gate.gate_id}")
            
            # Run assessment rules
            gate_risks = []
            for rule in gate.assessment_rules:
                try:
                    rule_result = rule(context)
                    if "risks" in rule_result:
                        gate_risks.extend(rule_result["risks"])
                    if "recommendations" in rule_result:
                        all_recommendations.extend(rule_result["recommendations"])
                    if "warnings" in rule_result:
                        all_warnings.extend(rule_result["warnings"])
                except Exception as e:
                    logger.error(f"Error running assessment rule in gate {gate.gate_id}: {e}")
                    all_warnings.append(f"Assessment rule failed: {e}")
            
            # Calculate gate risk score
            if gate_risks:
                gate_risk_score = max(risk.risk_score for risk in gate_risks)
                overall_risk_score = max(overall_risk_score, gate_risk_score)
                all_risks_identified.extend([risk.risk_id for risk in gate_risks])
            
            # Check blocking conditions
            blocked = False
            for condition in gate.blocking_conditions:
                try:
                    if condition(context, gate_risks):
                        blocked = True
                        break
                except Exception as e:
                    logger.error(f"Error checking blocking condition in gate {gate.gate_id}: {e}")
            
            # Determine gate decision
            if blocked:
                final_decision = RiskDecision.BLOCK
                break
            elif overall_risk_score > gate.risk_threshold:
                final_decision = RiskDecision.REQUIRE_APPROVAL
                requires_approval = True
                required_approvers.extend(gate.bypass_approvers)
            elif overall_risk_score > gate.approval_threshold:
                if final_decision == RiskDecision.PROCEED:
                    final_decision = RiskDecision.PROCEED_WITH_WARNING
            
            # Check required mitigations
            if gate.required_mitigations:
                missing_mitigations = self._check_required_mitigations(
                    gate.required_mitigations, context
                )
                if missing_mitigations:
                    all_mitigations_required.extend(missing_mitigations)
                    if final_decision == RiskDecision.PROCEED:
                        final_decision = RiskDecision.REQUIRE_APPROVAL
                        requires_approval = True
        
        # Create result
        result = WorkflowResult(
            workflow_id=workflow_id,
            stage=context.stage,
            decision=final_decision,
            risk_score=overall_risk_score,
            risks_identified=list(set(all_risks_identified)),
            mitigations_required=list(set(all_mitigations_required)),
            approval_required=requires_approval,
            required_approvers=list(set(required_approvers)),
            recommendations=list(set(all_recommendations)),
            warnings=list(set(all_warnings)),
            context=context,
        )
        
        # Store result
        self.workflow_history.append(result)
        
        # Log result
        logger.info(f"Workflow assessment complete: {workflow_id} - {final_decision.value} (risk: {overall_risk_score:.2f})")
        
        return result
    
    def _assess_code_change_risks(self, context: WorkflowContext) -> Dict[str, Any]:
        """Assess risks from code changes."""
        
        risks = []
        recommendations = []
        warnings = []
        
        # Check for high-risk file changes
        high_risk_patterns = [
            "security", "auth", "encryption", "password", "key", "token",
            "config", "settings", "environment", "deploy", "migration"
        ]
        
        high_risk_files = []
        for file_path in context.files_changed:
            if any(pattern in file_path.lower() for pattern in high_risk_patterns):
                high_risk_files.append(file_path)
        
        if high_risk_files:
            # Create or update security change risk
            security_risk = self._get_or_create_workflow_risk(
                "SECURITY_FILE_CHANGES",
                "Security-related file changes detected",
                RiskCategory.SECURITY,
                RiskLevel.MEDIUM,
                probability=0.4,
                impact=0.7,
            )
            risks.append(security_risk)
            warnings.append(f"Security-related files changed: {', '.join(high_risk_files[:3])}")
        
        # Check change size
        if context.lines_changed > 1000:
            large_change_risk = self._get_or_create_workflow_risk(
                "LARGE_CODE_CHANGE",
                "Large code change increases deployment risk",
                RiskCategory.OPERATIONAL,
                RiskLevel.MEDIUM,
                probability=0.3 + (context.lines_changed / 10000),  # Scale with size
                impact=0.6,
            )
            risks.append(large_change_risk)
            recommendations.append("Consider breaking large changes into smaller deployments")
        
        # Check test coverage
        if context.test_coverage < 0.8:
            coverage_risk = self._get_or_create_workflow_risk(
                "LOW_TEST_COVERAGE",
                "Low test coverage increases defect risk",
                RiskCategory.OPERATIONAL,
                RiskLevel.MEDIUM,
                probability=0.5,
                impact=0.5,
            )
            risks.append(coverage_risk)
            recommendations.append(f"Increase test coverage (current: {context.test_coverage:.1%})")
        
        return {
            "risks": risks,
            "recommendations": recommendations,
            "warnings": warnings,
        }
    
    def _assess_build_risks(self, context: WorkflowContext) -> Dict[str, Any]:
        """Assess build-related risks."""
        
        risks = []
        recommendations = []
        warnings = []
        
        # Simulate build risk assessment
        # In reality, this would check dependency vulnerabilities, build complexity, etc.
        
        if "dependency" in context.metadata.get("build_warnings", []):
            dep_risk = self._get_or_create_workflow_risk(
                "DEPENDENCY_VULNERABILITIES",
                "Vulnerable dependencies detected in build",
                RiskCategory.SECURITY,
                RiskLevel.HIGH,
                probability=0.6,
                impact=0.8,
            )
            risks.append(dep_risk)
            recommendations.append("Update vulnerable dependencies before deployment")
        
        return {
            "risks": risks,
            "recommendations": recommendations,
            "warnings": warnings,
        }
    
    def _assess_security_scan_risks(self, context: WorkflowContext) -> Dict[str, Any]:
        """Assess security scan results."""
        
        risks = []
        recommendations = []
        warnings = []
        
        scan_results = context.security_scan_results
        
        # Check for critical vulnerabilities
        critical_vulns = scan_results.get("critical_vulnerabilities", 0)
        high_vulns = scan_results.get("high_vulnerabilities", 0)
        
        if critical_vulns > 0:
            vuln_risk = self._get_or_create_workflow_risk(
                "CRITICAL_VULNERABILITIES",
                f"{critical_vulns} critical security vulnerabilities found",
                RiskCategory.SECURITY,
                RiskLevel.CRITICAL,
                probability=0.9,
                impact=0.9,
            )
            risks.append(vuln_risk)
            recommendations.append("Fix all critical vulnerabilities before deployment")
        
        elif high_vulns > 2:
            vuln_risk = self._get_or_create_workflow_risk(
                "HIGH_VULNERABILITIES",
                f"{high_vulns} high-severity security vulnerabilities found",
                RiskCategory.SECURITY,
                RiskLevel.HIGH,
                probability=0.7,
                impact=0.7,
            )
            risks.append(vuln_risk)
            recommendations.append("Review and fix high-severity vulnerabilities")
        
        return {
            "risks": risks,
            "recommendations": recommendations,
            "warnings": warnings,
        }
    
    def _assess_deployment_risks(self, context: WorkflowContext) -> Dict[str, Any]:
        """Assess deployment-specific risks."""
        
        risks = []
        recommendations = []
        warnings = []
        
        # Check system health
        if context.system_health.get("cpu_usage", 0) > 0.8:
            perf_risk = self._get_or_create_workflow_risk(
                "HIGH_SYSTEM_LOAD",
                "High system load increases deployment risk",
                RiskCategory.PERFORMANCE,
                RiskLevel.MEDIUM,
                probability=0.6,
                impact=0.6,
            )
            risks.append(perf_risk)
            recommendations.append("Consider deploying during lower traffic period")
        
        # Check for active incidents
        if context.active_incidents:
            incident_risk = self._get_or_create_workflow_risk(
                "ACTIVE_INCIDENTS",
                f"Active incidents may compound deployment risks: {', '.join(context.active_incidents)}",
                RiskCategory.OPERATIONAL,
                RiskLevel.HIGH,
                probability=0.7,
                impact=0.8,
            )
            risks.append(incident_risk)
            recommendations.append("Consider postponing deployment until incidents are resolved")
        
        # Check deployment timing
        if not context.business_hours and not context.maintenance_window:
            timing_risk = self._get_or_create_workflow_risk(
                "OFF_HOURS_DEPLOYMENT",
                "Deployment outside business hours with limited support coverage",
                RiskCategory.OPERATIONAL,
                RiskLevel.LOW,
                probability=0.3,
                impact=0.4,
            )
            risks.append(timing_risk)
            warnings.append("Deploying outside business hours - ensure on-call coverage")
        
        return {
            "risks": risks,
            "recommendations": recommendations,
            "warnings": warnings,
        }
    
    def _check_security_file_changes(self, context: WorkflowContext, risks: List[Risk]) -> bool:
        """Check if security-related file changes require blocking."""
        
        # Block if critical security files are changed without proper review
        critical_security_files = [
            "security.py", "auth.py", "encryption.py", "secrets.yaml", 
            "deployment.yaml", "production.env"
        ]
        
        for file_path in context.files_changed:
            if any(critical_file in file_path for critical_file in critical_security_files):
                # In a real system, would check if proper security review was done
                if not context.metadata.get("security_review_approved", False):
                    logger.warning(f"Blocking deployment due to security file change without review: {file_path}")
                    return True
        
        return False
    
    def _check_critical_vulnerabilities(self, context: WorkflowContext, risks: List[Risk]) -> bool:
        """Check if critical vulnerabilities should block deployment."""
        
        critical_vulns = context.security_scan_results.get("critical_vulnerabilities", 0)
        return critical_vulns > 0
    
    def _check_production_readiness(self, context: WorkflowContext, risks: List[Risk]) -> bool:
        """Check if system is ready for production deployment."""
        
        # Block if there are active P1/P2 incidents
        if context.active_incidents:
            high_severity_incidents = [
                inc for inc in context.active_incidents 
                if "P1" in inc or "P2" in inc
            ]
            if high_severity_incidents:
                logger.warning(f"Blocking deployment due to high-severity incidents: {high_severity_incidents}")
                return True
        
        # Block if system health is poor
        if context.system_health.get("overall_health", "healthy") == "unhealthy":
            logger.warning("Blocking deployment due to poor system health")
            return True
        
        return False
    
    def _check_required_mitigations(
        self, 
        required_mitigations: List[str], 
        context: WorkflowContext
    ) -> List[str]:
        """Check if required mitigations are in place."""
        
        missing = []
        implemented = context.metadata.get("implemented_mitigations", [])
        
        for mitigation in required_mitigations:
            if mitigation not in implemented:
                missing.append(mitigation)
        
        return missing
    
    def _get_or_create_workflow_risk(
        self,
        risk_id: str,
        title: str,
        category: RiskCategory,
        level: RiskLevel,
        probability: float,
        impact: float,
    ) -> Risk:
        """Get existing risk or create new workflow-related risk."""
        
        existing_risk = self.risk_registry.get_risk(risk_id)
        if existing_risk:
            return existing_risk
        
        # Create new risk
        risk = Risk(
            risk_id=risk_id,
            title=title,
            description=f"Workflow-identified risk: {title}",
            category=category,
            level=level,
            status=RiskStatus.IDENTIFIED,
            probability=probability,
            impact=impact,
            risk_score=probability * impact,
            identified_date=datetime.now(),
            identified_by="workflow_system",
            last_updated=datetime.now(),
            updated_by="workflow_system",
            tags={"workflow", "automated"},
        )
        
        self.risk_registry.register_risk(risk)
        return risk
    
    def request_approval(
        self,
        workflow_result: WorkflowResult,
        approver: str,
        justification: str = "",
    ) -> bool:
        """Request approval for a blocked workflow."""
        
        if workflow_result.decision != RiskDecision.REQUIRE_APPROVAL:
            return False
        
        # In a real system, this would integrate with approval systems
        # For now, simulate approval process
        logger.info(f"Approval requested for workflow {workflow_result.workflow_id} from {approver}")
        logger.info(f"Justification: {justification}")
        
        # Check if approver is authorized
        if approver in workflow_result.required_approvers:
            logger.info(f"Approval granted by {approver}")
            return True
        else:
            logger.warning(f"Approval denied - {approver} not in required approvers: {workflow_result.required_approvers}")
            return False
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow risk management statistics."""
        
        total_workflows = len(self.workflow_history)
        if total_workflows == 0:
            return {"total_workflows": 0}
        
        # Decision distribution
        decision_counts = {}
        for decision in RiskDecision:
            decision_counts[decision.value] = len([
                w for w in self.workflow_history if w.decision == decision
            ])
        
        # Stage distribution
        stage_counts = {}
        for stage in WorkflowStage:
            stage_counts[stage.value] = len([
                w for w in self.workflow_history if w.stage == stage
            ])
        
        # Risk metrics
        risk_scores = [w.risk_score for w in self.workflow_history]
        avg_risk_score = sum(risk_scores) / len(risk_scores)
        max_risk_score = max(risk_scores)
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(hours=24)
        recent_workflows = [
            w for w in self.workflow_history 
            if w.assessment_time > yesterday
        ]
        
        blocked_workflows = [
            w for w in self.workflow_history 
            if w.decision == RiskDecision.BLOCK
        ]
        
        return {
            "total_workflows": total_workflows,
            "workflows_last_24h": len(recent_workflows),
            "avg_risk_score": round(avg_risk_score, 3),
            "max_risk_score": round(max_risk_score, 3),
            "decision_distribution": decision_counts,
            "stage_distribution": stage_counts,
            "blocked_workflows": len(blocked_workflows),
            "total_gates_configured": sum(len(gates) for gates in self.workflow_gates.values()),
        }