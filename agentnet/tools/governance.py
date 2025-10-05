"""Tool governance and lifecycle management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from .base import Tool, ToolSpec


class ToolStatus(Enum):
    """Tool lifecycle status."""
    DRAFT = "draft"  # Tool is being developed
    TESTING = "testing"  # Tool is in testing phase
    APPROVED = "approved"  # Tool is approved for use
    ACTIVE = "active"  # Tool is actively deployed
    DEPRECATED = "deprecated"  # Tool is deprecated but still available
    RETIRED = "retired"  # Tool is no longer available


class GovernanceLevel(Enum):
    """Governance level for tools."""
    PUBLIC = "public"  # No restrictions
    INTERNAL = "internal"  # Requires internal authentication
    RESTRICTED = "restricted"  # Requires specific permissions
    CONFIDENTIAL = "confidential"  # Highest level of restriction


@dataclass
class ToolMetadata:
    """Extended metadata for tool governance."""
    status: ToolStatus = ToolStatus.DRAFT
    governance_level: GovernanceLevel = GovernanceLevel.PUBLIC
    owner: Optional[str] = None
    team: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    version: str = "1.0.0"
    changelog: List[str] = field(default_factory=list)
    allowed_tenants: Set[str] = field(default_factory=set)
    allowed_roles: Set[str] = field(default_factory=set)
    usage_quota: Optional[int] = None  # Total usage limit
    daily_quota: Optional[int] = None  # Daily usage limit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "governance_level": self.governance_level.value,
            "owner": self.owner,
            "team": self.team,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "retirement_date": self.retirement_date.isoformat() if self.retirement_date else None,
            "version": self.version,
            "changelog": self.changelog,
            "allowed_tenants": list(self.allowed_tenants),
            "allowed_roles": list(self.allowed_roles),
            "usage_quota": self.usage_quota,
            "daily_quota": self.daily_quota,
        }


class ToolGovernanceManager:
    """Manager for tool governance and lifecycle."""
    
    def __init__(self):
        self._metadata: Dict[str, ToolMetadata] = {}
        self._usage_tracking: Dict[str, Dict[str, int]] = {}  # tool_name -> {date: count}
        self._audit_log: List[Dict[str, Any]] = []
    
    def register_tool_metadata(self, tool_name: str, metadata: ToolMetadata) -> None:
        """Register metadata for a tool.
        
        Args:
            tool_name: Name of the tool
            metadata: Tool metadata
        """
        self._metadata[tool_name] = metadata
        self._log_audit_event(
            "tool_registered",
            tool_name=tool_name,
            status=metadata.status.value,
            owner=metadata.owner
        )
    
    def update_tool_status(
        self,
        tool_name: str,
        new_status: ToolStatus,
        updated_by: Optional[str] = None
    ) -> bool:
        """Update tool status.
        
        Args:
            tool_name: Name of the tool
            new_status: New status
            updated_by: User making the update
            
        Returns:
            True if updated successfully
        """
        if tool_name not in self._metadata:
            return False
        
        old_status = self._metadata[tool_name].status
        self._metadata[tool_name].status = new_status
        self._metadata[tool_name].updated_at = datetime.now()
        
        # Add to changelog
        self._metadata[tool_name].changelog.append(
            f"Status changed from {old_status.value} to {new_status.value} "
            f"by {updated_by or 'system'} at {datetime.now().isoformat()}"
        )
        
        self._log_audit_event(
            "status_changed",
            tool_name=tool_name,
            old_status=old_status.value,
            new_status=new_status.value,
            updated_by=updated_by
        )
        
        return True
    
    def approve_tool(
        self,
        tool_name: str,
        approver: str,
        make_active: bool = True
    ) -> bool:
        """Approve a tool for use.
        
        Args:
            tool_name: Name of the tool
            approver: Who is approving the tool
            make_active: Whether to make the tool active immediately
            
        Returns:
            True if approved successfully
        """
        if tool_name not in self._metadata:
            return False
        
        metadata = self._metadata[tool_name]
        metadata.approved_by = approver
        metadata.approved_at = datetime.now()
        metadata.status = ToolStatus.ACTIVE if make_active else ToolStatus.APPROVED
        metadata.updated_at = datetime.now()
        
        metadata.changelog.append(
            f"Tool approved by {approver} at {datetime.now().isoformat()}"
        )
        
        self._log_audit_event(
            "tool_approved",
            tool_name=tool_name,
            approver=approver,
            make_active=make_active
        )
        
        return True
    
    def deprecate_tool(
        self,
        tool_name: str,
        deprecation_date: Optional[datetime] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Deprecate a tool.
        
        Args:
            tool_name: Name of the tool
            deprecation_date: When the tool was/will be deprecated
            reason: Reason for deprecation
            
        Returns:
            True if deprecated successfully
        """
        if tool_name not in self._metadata:
            return False
        
        metadata = self._metadata[tool_name]
        metadata.status = ToolStatus.DEPRECATED
        metadata.deprecation_date = deprecation_date or datetime.now()
        metadata.updated_at = datetime.now()
        
        changelog_msg = f"Tool deprecated at {metadata.deprecation_date.isoformat()}"
        if reason:
            changelog_msg += f": {reason}"
        metadata.changelog.append(changelog_msg)
        
        self._log_audit_event(
            "tool_deprecated",
            tool_name=tool_name,
            deprecation_date=metadata.deprecation_date.isoformat(),
            reason=reason
        )
        
        return True
    
    def retire_tool(
        self,
        tool_name: str,
        retirement_date: Optional[datetime] = None
    ) -> bool:
        """Retire a tool (no longer available).
        
        Args:
            tool_name: Name of the tool
            retirement_date: When the tool was/will be retired
            
        Returns:
            True if retired successfully
        """
        if tool_name not in self._metadata:
            return False
        
        metadata = self._metadata[tool_name]
        metadata.status = ToolStatus.RETIRED
        metadata.retirement_date = retirement_date or datetime.now()
        metadata.updated_at = datetime.now()
        
        metadata.changelog.append(
            f"Tool retired at {metadata.retirement_date.isoformat()}"
        )
        
        self._log_audit_event(
            "tool_retired",
            tool_name=tool_name,
            retirement_date=metadata.retirement_date.isoformat()
        )
        
        return True
    
    def can_use_tool(
        self,
        tool_name: str,
        tenant_id: Optional[str] = None,
        user_roles: Optional[Set[str]] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if a tool can be used based on governance rules.
        
        Args:
            tool_name: Name of the tool
            tenant_id: Tenant trying to use the tool
            user_roles: User roles
            
        Returns:
            Tuple of (can_use, reason_if_not)
        """
        if tool_name not in self._metadata:
            return False, "Tool not found"
        
        metadata = self._metadata[tool_name]
        
        # Check status
        if metadata.status == ToolStatus.RETIRED:
            return False, "Tool has been retired"
        
        if metadata.status == ToolStatus.DRAFT:
            return False, "Tool is still in draft status"
        
        if metadata.status == ToolStatus.TESTING:
            return False, "Tool is in testing phase"
        
        # Check tenant restrictions
        if metadata.allowed_tenants and tenant_id:
            if tenant_id not in metadata.allowed_tenants:
                return False, "Tool not available for this tenant"
        
        # Check role restrictions
        if metadata.allowed_roles and user_roles:
            if not metadata.allowed_roles.intersection(user_roles):
                return False, "User does not have required role"
        
        # Check usage quota
        if metadata.usage_quota is not None:
            total_usage = sum(self._usage_tracking.get(tool_name, {}).values())
            if total_usage >= metadata.usage_quota:
                return False, "Tool usage quota exceeded"
        
        # Check daily quota
        if metadata.daily_quota is not None:
            today = datetime.now().date().isoformat()
            daily_usage = self._usage_tracking.get(tool_name, {}).get(today, 0)
            if daily_usage >= metadata.daily_quota:
                return False, "Daily usage quota exceeded"
        
        return True, None
    
    def track_usage(self, tool_name: str) -> None:
        """Track tool usage.
        
        Args:
            tool_name: Name of the tool
        """
        today = datetime.now().date().isoformat()
        
        if tool_name not in self._usage_tracking:
            self._usage_tracking[tool_name] = {}
        
        if today not in self._usage_tracking[tool_name]:
            self._usage_tracking[tool_name][today] = 0
        
        self._usage_tracking[tool_name][today] += 1
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool metadata if found
        """
        return self._metadata.get(tool_name)
    
    def get_tools_by_status(self, status: ToolStatus) -> List[str]:
        """Get all tools with a specific status.
        
        Args:
            status: Tool status to filter by
            
        Returns:
            List of tool names
        """
        return [
            name for name, metadata in self._metadata.items()
            if metadata.status == status
        ]
    
    def get_usage_stats(self, tool_name: str) -> Dict[str, int]:
        """Get usage statistics for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of date -> usage count
        """
        return self._usage_tracking.get(tool_name, {})
    
    def get_audit_log(
        self,
        tool_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries.
        
        Args:
            tool_name: Filter by tool name
            event_type: Filter by event type
            limit: Maximum number of entries to return
            
        Returns:
            List of audit log entries
        """
        filtered_log = self._audit_log
        
        if tool_name:
            filtered_log = [
                entry for entry in filtered_log
                if entry.get("tool_name") == tool_name
            ]
        
        if event_type:
            filtered_log = [
                entry for entry in filtered_log
                if entry.get("event_type") == event_type
            ]
        
        return filtered_log[-limit:]
    
    def _log_audit_event(self, event_type: str, **kwargs) -> None:
        """Log an audit event.
        
        Args:
            event_type: Type of event
            **kwargs: Additional event data
        """
        self._audit_log.append({
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        })
    
    def export_governance_data(self) -> Dict[str, Any]:
        """Export all governance data.
        
        Returns:
            Dictionary with all governance data
        """
        return {
            "metadata": {
                name: metadata.to_dict()
                for name, metadata in self._metadata.items()
            },
            "usage_tracking": self._usage_tracking,
            "audit_log": self._audit_log
        }
