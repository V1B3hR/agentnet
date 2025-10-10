"""
Tool governance system with approval workflows.

Provides advanced governance features for tool execution including:
- Human-in-the-loop approval workflows
- Risk assessment for tool usage
- Audit logging and compliance tracking
- Policy-based tool restrictions
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path
import json

logger = logging.getLogger("agentnet.tools.governance")


class ApprovalStatus(str, Enum):
    """Status of approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ToolRiskLevel(str, Enum):
    """Risk level for tool execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    """Request for human approval of tool execution."""
    
    request_id: str
    tool_name: str
    parameters: Dict[str, Any]
    agent_name: str
    session_id: str
    risk_level: ToolRiskLevel
    risk_reasons: List[str] = field(default_factory=list)
    
    # Approval details
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Response details
    approver_id: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    
    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "risk_level": self.risk_level.value,
            "risk_reasons": self.risk_reasons,
            "status": self.status.value,
            "requested_at": self.requested_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "approver_id": self.approver_id,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


@dataclass
class ToolGovernancePolicy:
    """Policy for tool governance."""
    
    tool_pattern: str  # Regex pattern for tool names
    require_approval: bool = False
    max_risk_level: ToolRiskLevel = ToolRiskLevel.HIGH
    allowed_agents: Optional[Set[str]] = None
    allowed_users: Optional[Set[str]] = None
    allowed_parameters: Optional[Dict[str, Any]] = None
    blocked_parameters: Optional[Dict[str, Any]] = None
    rate_limit_per_hour: Optional[int] = None
    audit_required: bool = True
    custom_validators: List[Callable] = field(default_factory=list)
    
    def matches_tool(self, tool_name: str) -> bool:
        """Check if policy matches tool name."""
        import re
        return bool(re.match(self.tool_pattern, tool_name))
    
    def is_agent_allowed(self, agent_name: str) -> bool:
        """Check if agent is allowed to use tool."""
        if self.allowed_agents is None:
            return True
        return agent_name in self.allowed_agents
    
    def is_user_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to use tool."""
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users


class ToolGovernanceManager:
    """
    Manager for tool governance with approval workflows.
    
    Handles risk assessment, approval requests, policy enforcement,
    and audit logging for tool execution.
    """
    
    def __init__(
        self,
        storage_dir: str = "tool_governance",
        approval_timeout_minutes: int = 30,
        enable_auto_approval: bool = False,
    ):
        """
        Initialize governance manager.
        
        Args:
            storage_dir: Directory for storing approval requests and audits
            approval_timeout_minutes: Default timeout for approval requests
            enable_auto_approval: Enable automatic approval for low-risk tools
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "approvals").mkdir(exist_ok=True)
        (self.storage_dir / "audits").mkdir(exist_ok=True)
        (self.storage_dir / "policies").mkdir(exist_ok=True)
        
        self.approval_timeout = timedelta(minutes=approval_timeout_minutes)
        self.enable_auto_approval = enable_auto_approval
        
        # In-memory tracking
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        self._policies: List[ToolGovernancePolicy] = []
        self._approval_callbacks: List[Callable] = []
        
        # Risk assessment rules
        self._risk_assessors: Dict[str, Callable] = {}
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info(f"ToolGovernanceManager initialized at {storage_dir}")
    
    def _initialize_default_policies(self):
        """Initialize default governance policies."""
        # High-risk tools require approval
        self._policies.append(
            ToolGovernancePolicy(
                tool_pattern=r".*exec.*|.*system.*|.*shell.*",
                require_approval=True,
                max_risk_level=ToolRiskLevel.CRITICAL,
                audit_required=True,
            )
        )
        
        # Database operations require approval
        self._policies.append(
            ToolGovernancePolicy(
                tool_pattern=r".*database.*|.*sql.*|.*query.*",
                require_approval=True,
                max_risk_level=ToolRiskLevel.HIGH,
                audit_required=True,
            )
        )
        
        # API calls may require approval based on risk
        self._policies.append(
            ToolGovernancePolicy(
                tool_pattern=r".*api.*|.*http.*|.*request.*",
                require_approval=False,
                max_risk_level=ToolRiskLevel.MEDIUM,
                audit_required=True,
            )
        )
    
    def add_policy(self, policy: ToolGovernancePolicy):
        """Add a governance policy."""
        self._policies.append(policy)
        logger.info(f"Added governance policy for pattern: {policy.tool_pattern}")
    
    def register_risk_assessor(self, tool_pattern: str, assessor: Callable):
        """Register a custom risk assessor function."""
        self._risk_assessors[tool_pattern] = assessor
    
    def register_approval_callback(self, callback: Callable):
        """Register a callback for approval notifications."""
        self._approval_callbacks.append(callback)
    
    def assess_risk(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[ToolRiskLevel, List[str]]:
        """
        Assess risk level for tool execution.
        
        Returns:
            Tuple of (risk_level, list of risk reasons)
        """
        risk_reasons = []
        risk_level = ToolRiskLevel.LOW
        
        # Check for high-risk patterns in tool name
        if any(pattern in tool_name.lower() for pattern in ["exec", "system", "shell", "delete"]):
            risk_level = ToolRiskLevel.CRITICAL
            risk_reasons.append(f"Tool name '{tool_name}' contains high-risk keywords")
        
        # Check parameters for sensitive data
        param_str = json.dumps(parameters).lower()
        if any(keyword in param_str for keyword in ["password", "secret", "token", "key"]):
            if risk_level.value != "critical":
                risk_level = ToolRiskLevel.HIGH
            risk_reasons.append("Parameters contain sensitive data keywords")
        
        # Check for destructive operations
        if any(keyword in param_str for keyword in ["drop", "delete", "remove", "destroy"]):
            if risk_level.value != "critical":
                risk_level = ToolRiskLevel.HIGH
            risk_reasons.append("Parameters suggest destructive operation")
        
        # Run custom risk assessors
        for pattern, assessor in self._risk_assessors.items():
            import re
            if re.match(pattern, tool_name):
                try:
                    custom_risk, custom_reasons = assessor(tool_name, parameters, agent_name, context)
                    if custom_risk.value in ["high", "critical"]:
                        risk_level = custom_risk
                    risk_reasons.extend(custom_reasons)
                except Exception as e:
                    logger.error(f"Error in custom risk assessor: {e}")
        
        return risk_level, risk_reasons
    
    def requires_approval(
        self,
        tool_name: str,
        risk_level: ToolRiskLevel,
        agent_name: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Check if tool execution requires approval."""
        # Check if auto-approval is enabled and risk is low
        if self.enable_auto_approval and risk_level == ToolRiskLevel.LOW:
            return False
        
        # Check governance policies
        for policy in self._policies:
            if policy.matches_tool(tool_name):
                # Check agent and user restrictions
                if not policy.is_agent_allowed(agent_name):
                    return True
                if user_id and not policy.is_user_allowed(user_id):
                    return True
                
                # Check if policy requires approval
                if policy.require_approval:
                    return True
                
                # Check risk level threshold
                risk_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                if risk_order[risk_level.value] > risk_order[policy.max_risk_level.value]:
                    return True
        
        return False
    
    async def request_approval(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_name: str,
        session_id: str,
        risk_level: ToolRiskLevel,
        risk_reasons: List[str],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Create an approval request."""
        request_id = f"approval_{int(time.time() * 1000)}_{tool_name}"
        
        request = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_name,
            parameters=parameters,
            agent_name=agent_name,
            session_id=session_id,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            expires_at=datetime.now() + self.approval_timeout,
            metadata=metadata or {},
        )
        
        # Store request
        self._pending_approvals[request_id] = request
        self._save_approval_request(request)
        
        # Notify callbacks
        for callback in self._approval_callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Error in approval callback: {e}")
        
        logger.info(f"Created approval request: {request_id}")
        return request
    
    def approve_request(self, request_id: str, approver_id: str) -> bool:
        """Approve a pending request."""
        request = self._pending_approvals.get(request_id)
        if not request:
            logger.warning(f"Approval request not found: {request_id}")
            return False
        
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            self._save_approval_request(request)
            logger.warning(f"Approval request expired: {request_id}")
            return False
        
        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver_id
        request.approved_at = datetime.now()
        
        self._save_approval_request(request)
        logger.info(f"Approved request {request_id} by {approver_id}")
        return True
    
    def reject_request(self, request_id: str, approver_id: str, reason: str) -> bool:
        """Reject a pending request."""
        request = self._pending_approvals.get(request_id)
        if not request:
            logger.warning(f"Approval request not found: {request_id}")
            return False
        
        request.status = ApprovalStatus.REJECTED
        request.approver_id = approver_id
        request.approved_at = datetime.now()
        request.rejection_reason = reason
        
        self._save_approval_request(request)
        logger.info(f"Rejected request {request_id} by {approver_id}: {reason}")
        return True
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            req for req in self._pending_approvals.values()
            if req.status == ApprovalStatus.PENDING and not req.is_expired()
        ]
    
    def _save_approval_request(self, request: ApprovalRequest):
        """Save approval request to disk."""
        file_path = self.storage_dir / "approvals" / f"{request.request_id}.json"
        with open(file_path, "w") as f:
            json.dump(request.to_dict(), f, indent=2)
    
    def audit_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        agent_name: str,
        session_id: str,
        user_id: Optional[str] = None,
        approval_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log tool execution for audit trail."""
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "parameters": parameters,
            "result_status": getattr(result, "status", "unknown"),
            "agent_name": agent_name,
            "session_id": session_id,
            "user_id": user_id,
            "approval_id": approval_id,
            "metadata": metadata or {},
        }
        
        # Save audit record
        audit_file = self.storage_dir / "audits" / f"audit_{int(time.time() * 1000)}.json"
        with open(audit_file, "w") as f:
            json.dump(audit_record, f, indent=2)
        
        logger.debug(f"Audited tool execution: {tool_name}")
