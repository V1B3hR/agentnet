"""
Human-in-Loop Gating Module

Implements approval and escalation flow for high-risk or policy-violating actions.
The system pauses, notifies humans, and only continues with their input.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of approval requests."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


class EscalationLevel(str, Enum):
    """Escalation levels for approval chain."""

    L1_OPERATOR = "l1_operator"
    L2_SUPERVISOR = "l2_supervisor"
    L3_MANAGER = "l3_manager"
    L4_EXECUTIVE = "l4_executive"


class RiskLevel(str, Enum):
    """Risk levels for actions requiring approval."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    """A request for human approval."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    action_description: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    escalation_level: EscalationLevel = EscalationLevel.L1_OPERATOR
    requester_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver_id: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    timeout_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))

    # Escalation tracking
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    max_escalation_level: EscalationLevel = EscalationLevel.L3_MANAGER


@dataclass
class Approver:
    """A human approver in the system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    level: EscalationLevel = EscalationLevel.L1_OPERATOR
    active: bool = True
    notification_preferences: Dict[str, Any] = field(default_factory=dict)


class HumanApprovalGate:
    """
    Human-in-loop gating system for approval and escalation flow.

    Features:
    - Async approval requests with timeout
    - Escalation chain management
    - Policy integration for risk assessment
    - Notification system
    """

    def __init__(
        self,
        default_timeout: timedelta = timedelta(hours=1),
        auto_escalate_timeout: timedelta = timedelta(minutes=30),
        enable_notifications: bool = True,
    ):
        self.default_timeout = default_timeout
        self.auto_escalate_timeout = auto_escalate_timeout
        self.enable_notifications = enable_notifications

        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.approvers: Dict[str, Approver] = {}
        self.approval_history: List[ApprovalRequest] = []

        # Callbacks for integration
        self.on_approval_request: Optional[Callable[[ApprovalRequest], None]] = None
        self.on_approval_granted: Optional[Callable[[ApprovalRequest], None]] = None
        self.on_approval_rejected: Optional[Callable[[ApprovalRequest], None]] = None
        self.on_escalation: Optional[
            Callable[[ApprovalRequest, EscalationLevel], None]
        ] = None

        # Background task for monitoring timeouts
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info("HumanApprovalGate initialized")

    def add_approver(
        self,
        name: str,
        email: str,
        level: EscalationLevel,
        notification_preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an approver to the system."""

        approver = Approver(
            name=name,
            email=email,
            level=level,
            notification_preferences=notification_preferences or {},
        )

        self.approvers[approver.id] = approver
        logger.info(f"Added approver {name} ({level}) with ID {approver.id}")

        return approver.id

    def remove_approver(self, approver_id: str) -> bool:
        """Remove an approver from the system."""

        if approver_id in self.approvers:
            approver = self.approvers[approver_id]
            del self.approvers[approver_id]
            logger.info(f"Removed approver {approver.name}")
            return True

        return False

    def get_approvers_by_level(self, level: EscalationLevel) -> List[Approver]:
        """Get all active approvers at a specific escalation level."""

        return [
            approver
            for approver in self.approvers.values()
            if approver.level == level and approver.active
        ]

    async def request_approval(
        self,
        action_description: str,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        requester_id: str = "",
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[timedelta] = None,
    ) -> ApprovalRequest:
        """Request human approval for an action."""

        # Determine escalation level based on risk
        escalation_level = self._determine_escalation_level(risk_level)

        request = ApprovalRequest(
            action_description=action_description,
            risk_level=risk_level,
            escalation_level=escalation_level,
            requester_id=requester_id,
            context=context or {},
            timeout_duration=timeout or self._get_timeout_for_risk(risk_level),
        )

        self.pending_requests[request.id] = request

        # Notify approvers
        await self._notify_approvers(request)

        # Start monitoring task if not already running
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitor_timeouts())

        if self.on_approval_request:
            self.on_approval_request(request)

        logger.info(f"Approval request {request.id} created for: {action_description}")

        return request

    async def wait_for_approval(self, request_id: str) -> ApprovalStatus:
        """Wait for approval of a specific request."""

        if request_id not in self.pending_requests:
            return ApprovalStatus.REJECTED

        request = self.pending_requests[request_id]

        # Wait for approval or timeout
        timeout_time = request.timestamp + request.timeout_duration

        while (
            datetime.now() < timeout_time and request.status == ApprovalStatus.PENDING
        ):
            await asyncio.sleep(1)  # Check every second

        # Handle timeout
        if request.status == ApprovalStatus.PENDING:
            await self._handle_timeout(request)

        return request.status

    def approve_request(
        self, request_id: str, approver_id: str, comment: str = ""
    ) -> bool:
        """Approve a pending request."""

        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False

        # Verify approver has sufficient level
        if approver_id not in self.approvers:
            return False

        approver = self.approvers[approver_id]
        if not self._can_approve_at_level(approver.level, request.escalation_level):
            return False

        # Grant approval
        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver_id
        request.approval_timestamp = datetime.now()
        request.context["approval_comment"] = comment

        # Move to history
        self._move_to_history(request_id)

        if self.on_approval_granted:
            self.on_approval_granted(request)

        logger.info(f"Request {request_id} approved by {approver.name}")

        return True

    def reject_request(
        self, request_id: str, approver_id: str, reason: str = ""
    ) -> bool:
        """Reject a pending request."""

        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False

        # Verify approver
        if approver_id not in self.approvers:
            return False

        # Reject request
        request.status = ApprovalStatus.REJECTED
        request.approver_id = approver_id
        request.approval_timestamp = datetime.now()
        request.rejection_reason = reason

        # Move to history
        self._move_to_history(request_id)

        if self.on_approval_rejected:
            self.on_approval_rejected(request)

        logger.info(f"Request {request_id} rejected by {approver_id}: {reason}")

        return True

    async def escalate_request(self, request_id: str, reason: str = "") -> bool:
        """Escalate a request to the next level."""

        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]

        # Determine next escalation level
        next_level = self._get_next_escalation_level(request.escalation_level)

        if not next_level or next_level.value > request.max_escalation_level.value:
            # Cannot escalate further - reject
            request.status = ApprovalStatus.REJECTED
            request.rejection_reason = f"Maximum escalation level reached: {reason}"
            self._move_to_history(request_id)
            return False

        # Record escalation
        escalation_record = {
            "from_level": request.escalation_level,
            "to_level": next_level,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
        }
        request.escalation_history.append(escalation_record)

        # Update request
        request.escalation_level = next_level
        request.status = ApprovalStatus.ESCALATED

        # Notify new level approvers
        await self._notify_approvers(request)

        # Reset status to pending for new level
        request.status = ApprovalStatus.PENDING

        if self.on_escalation:
            self.on_escalation(request, next_level)

        logger.info(f"Request {request_id} escalated to {next_level}: {reason}")

        return True

    def get_pending_requests(
        self, escalation_level: Optional[EscalationLevel] = None
    ) -> List[ApprovalRequest]:
        """Get pending requests, optionally filtered by escalation level."""

        requests = [
            req
            for req in self.pending_requests.values()
            if req.status == ApprovalStatus.PENDING
        ]

        if escalation_level:
            requests = [
                req for req in requests if req.escalation_level == escalation_level
            ]

        return requests

    def get_approval_metrics(self) -> Dict[str, Any]:
        """Get metrics about approval system performance."""

        total_requests = len(self.approval_history) + len(self.pending_requests)
        approved = len(
            [
                req
                for req in self.approval_history
                if req.status == ApprovalStatus.APPROVED
            ]
        )
        rejected = len(
            [
                req
                for req in self.approval_history
                if req.status == ApprovalStatus.REJECTED
            ]
        )
        pending = len(self.pending_requests)

        # Average processing time for completed requests
        completed_requests = [
            req for req in self.approval_history if req.approval_timestamp
        ]
        if completed_requests:
            avg_processing_time = sum(
                (req.approval_timestamp - req.timestamp).total_seconds()
                for req in completed_requests
            ) / len(completed_requests)
        else:
            avg_processing_time = 0

        return {
            "total_requests": total_requests,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "approval_rate": approved / max(total_requests - pending, 1),
            "average_processing_time_seconds": avg_processing_time,
            "escalated_requests": len(
                [
                    req
                    for req in self.approval_history
                    + list(self.pending_requests.values())
                    if req.escalation_history
                ]
            ),
        }

    async def shutdown(self):
        """Shutdown the approval gate and cleanup resources."""

        self._shutdown = True

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("HumanApprovalGate shutdown complete")

    def _determine_escalation_level(self, risk_level: RiskLevel) -> EscalationLevel:
        """Determine initial escalation level based on risk."""

        mapping = {
            RiskLevel.LOW: EscalationLevel.L1_OPERATOR,
            RiskLevel.MEDIUM: EscalationLevel.L2_SUPERVISOR,
            RiskLevel.HIGH: EscalationLevel.L3_MANAGER,
            RiskLevel.CRITICAL: EscalationLevel.L4_EXECUTIVE,
        }

        return mapping.get(risk_level, EscalationLevel.L2_SUPERVISOR)

    def _get_timeout_for_risk(self, risk_level: RiskLevel) -> timedelta:
        """Get timeout duration based on risk level."""

        timeouts = {
            RiskLevel.LOW: timedelta(hours=4),
            RiskLevel.MEDIUM: timedelta(hours=1),
            RiskLevel.HIGH: timedelta(minutes=30),
            RiskLevel.CRITICAL: timedelta(minutes=15),
        }

        return timeouts.get(risk_level, self.default_timeout)

    def _can_approve_at_level(
        self, approver_level: EscalationLevel, required_level: EscalationLevel
    ) -> bool:
        """Check if approver can approve at required level."""

        level_hierarchy = {
            EscalationLevel.L1_OPERATOR: 1,
            EscalationLevel.L2_SUPERVISOR: 2,
            EscalationLevel.L3_MANAGER: 3,
            EscalationLevel.L4_EXECUTIVE: 4,
        }

        return level_hierarchy[approver_level] >= level_hierarchy[required_level]

    def _get_next_escalation_level(
        self, current_level: EscalationLevel
    ) -> Optional[EscalationLevel]:
        """Get the next escalation level."""

        escalation_chain = {
            EscalationLevel.L1_OPERATOR: EscalationLevel.L2_SUPERVISOR,
            EscalationLevel.L2_SUPERVISOR: EscalationLevel.L3_MANAGER,
            EscalationLevel.L3_MANAGER: EscalationLevel.L4_EXECUTIVE,
            EscalationLevel.L4_EXECUTIVE: None,
        }

        return escalation_chain.get(current_level)

    async def _notify_approvers(self, request: ApprovalRequest) -> None:
        """Notify appropriate approvers about a request."""

        if not self.enable_notifications:
            return

        approvers = self.get_approvers_by_level(request.escalation_level)

        for approver in approvers:
            try:
                await self._send_notification(approver, request)
            except Exception as e:
                logger.error(f"Failed to notify approver {approver.name}: {e}")

    async def _send_notification(
        self, approver: Approver, request: ApprovalRequest
    ) -> None:
        """Send notification to an approver (placeholder for actual implementation)."""

        # This would integrate with actual notification systems (email, Slack, etc.)
        logger.info(f"Notification sent to {approver.name} for request {request.id}")

    async def _monitor_timeouts(self) -> None:
        """Background task to monitor request timeouts."""

        while not self._shutdown:
            try:
                current_time = datetime.now()

                # Check for timeouts
                for request_id, request in list(self.pending_requests.items()):
                    if request.status == ApprovalStatus.PENDING:
                        timeout_time = request.timestamp + request.timeout_duration

                        if current_time >= timeout_time:
                            await self._handle_timeout(request)
                        elif (
                            current_time
                            >= request.timestamp + self.auto_escalate_timeout
                        ):
                            # Auto-escalate if configured
                            await self.escalate_request(
                                request_id, "Auto-escalation due to timeout"
                            )

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in timeout monitoring: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _handle_timeout(self, request: ApprovalRequest) -> None:
        """Handle timeout of an approval request."""

        request.status = ApprovalStatus.TIMEOUT
        request.approval_timestamp = datetime.now()
        request.rejection_reason = "Request timed out"

        self._move_to_history(request.id)

        logger.warning(f"Request {request.id} timed out")

    def _move_to_history(self, request_id: str) -> None:
        """Move a request from pending to history."""

        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            self.approval_history.append(request)
            del self.pending_requests[request_id]
