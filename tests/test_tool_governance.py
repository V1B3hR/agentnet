#!/usr/bin/env python3
"""Tests for tool governance and lifecycle management."""

import pytest
from datetime import datetime

from agentnet.tools.governance import (
    ToolGovernanceManager,
    ToolMetadata,
    ToolStatus,
    GovernanceLevel,
)


def test_tool_metadata_creation():
    """Test creating tool metadata."""
    metadata = ToolMetadata(
        status=ToolStatus.ACTIVE,
        governance_level=GovernanceLevel.INTERNAL,
        owner="test-team",
        version="1.0.0"
    )
    
    assert metadata.status == ToolStatus.ACTIVE
    assert metadata.governance_level == GovernanceLevel.INTERNAL
    assert metadata.owner == "test-team"
    assert metadata.version == "1.0.0"


def test_register_tool_metadata():
    """Test registering tool metadata."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(
        status=ToolStatus.DRAFT,
        owner="dev-team"
    )
    
    manager.register_tool_metadata("test_tool", metadata)
    
    retrieved = manager.get_tool_metadata("test_tool")
    assert retrieved is not None
    assert retrieved.status == ToolStatus.DRAFT
    assert retrieved.owner == "dev-team"


def test_update_tool_status():
    """Test updating tool status."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(status=ToolStatus.DRAFT)
    manager.register_tool_metadata("test_tool", metadata)
    
    # Update status
    result = manager.update_tool_status(
        "test_tool",
        ToolStatus.TESTING,
        updated_by="admin"
    )
    
    assert result is True
    updated = manager.get_tool_metadata("test_tool")
    assert updated.status == ToolStatus.TESTING
    assert len(updated.changelog) > 0


def test_approve_tool():
    """Test approving a tool."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(status=ToolStatus.TESTING)
    manager.register_tool_metadata("test_tool", metadata)
    
    # Approve the tool
    result = manager.approve_tool("test_tool", approver="admin", make_active=True)
    
    assert result is True
    approved = manager.get_tool_metadata("test_tool")
    assert approved.status == ToolStatus.ACTIVE
    assert approved.approved_by == "admin"
    assert approved.approved_at is not None


def test_deprecate_tool():
    """Test deprecating a tool."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(status=ToolStatus.ACTIVE)
    manager.register_tool_metadata("test_tool", metadata)
    
    # Deprecate the tool
    result = manager.deprecate_tool(
        "test_tool",
        reason="Replaced by new_tool"
    )
    
    assert result is True
    deprecated = manager.get_tool_metadata("test_tool")
    assert deprecated.status == ToolStatus.DEPRECATED
    assert deprecated.deprecation_date is not None


def test_retire_tool():
    """Test retiring a tool."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(status=ToolStatus.DEPRECATED)
    manager.register_tool_metadata("test_tool", metadata)
    
    # Retire the tool
    result = manager.retire_tool("test_tool")
    
    assert result is True
    retired = manager.get_tool_metadata("test_tool")
    assert retired.status == ToolStatus.RETIRED
    assert retired.retirement_date is not None


def test_can_use_tool_status_checks():
    """Test tool usage checks based on status."""
    manager = ToolGovernanceManager()
    
    # Draft tool - cannot use
    metadata_draft = ToolMetadata(status=ToolStatus.DRAFT)
    manager.register_tool_metadata("draft_tool", metadata_draft)
    can_use, reason = manager.can_use_tool("draft_tool")
    assert can_use is False
    assert "draft" in reason.lower()
    
    # Active tool - can use
    metadata_active = ToolMetadata(status=ToolStatus.ACTIVE)
    manager.register_tool_metadata("active_tool", metadata_active)
    can_use, reason = manager.can_use_tool("active_tool")
    assert can_use is True
    assert reason is None
    
    # Retired tool - cannot use
    metadata_retired = ToolMetadata(status=ToolStatus.RETIRED)
    manager.register_tool_metadata("retired_tool", metadata_retired)
    can_use, reason = manager.can_use_tool("retired_tool")
    assert can_use is False
    assert "retired" in reason.lower()


def test_can_use_tool_tenant_restrictions():
    """Test tool usage checks with tenant restrictions."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(
        status=ToolStatus.ACTIVE,
        allowed_tenants={"tenant1", "tenant2"}
    )
    manager.register_tool_metadata("restricted_tool", metadata)
    
    # Allowed tenant
    can_use, reason = manager.can_use_tool("restricted_tool", tenant_id="tenant1")
    assert can_use is True
    
    # Disallowed tenant
    can_use, reason = manager.can_use_tool("restricted_tool", tenant_id="tenant3")
    assert can_use is False
    assert "tenant" in reason.lower()


def test_can_use_tool_role_restrictions():
    """Test tool usage checks with role restrictions."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(
        status=ToolStatus.ACTIVE,
        allowed_roles={"admin", "operator"}
    )
    manager.register_tool_metadata("admin_tool", metadata)
    
    # Has required role
    can_use, reason = manager.can_use_tool("admin_tool", user_roles={"admin", "user"})
    assert can_use is True
    
    # Does not have required role
    can_use, reason = manager.can_use_tool("admin_tool", user_roles={"user"})
    assert can_use is False
    assert "role" in reason.lower()


def test_usage_tracking():
    """Test usage tracking."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(status=ToolStatus.ACTIVE)
    manager.register_tool_metadata("test_tool", metadata)
    
    # Track usage
    manager.track_usage("test_tool")
    manager.track_usage("test_tool")
    manager.track_usage("test_tool")
    
    # Get usage stats
    stats = manager.get_usage_stats("test_tool")
    today = datetime.now().date().isoformat()
    assert stats[today] == 3


def test_usage_quota():
    """Test usage quota enforcement."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(
        status=ToolStatus.ACTIVE,
        usage_quota=2
    )
    manager.register_tool_metadata("quota_tool", metadata)
    
    # Track usage up to quota
    manager.track_usage("quota_tool")
    manager.track_usage("quota_tool")
    
    # Should not be able to use anymore
    can_use, reason = manager.can_use_tool("quota_tool")
    assert can_use is False
    assert "quota" in reason.lower()


def test_get_tools_by_status():
    """Test getting tools by status."""
    manager = ToolGovernanceManager()
    
    manager.register_tool_metadata("draft1", ToolMetadata(status=ToolStatus.DRAFT))
    manager.register_tool_metadata("draft2", ToolMetadata(status=ToolStatus.DRAFT))
    manager.register_tool_metadata("active1", ToolMetadata(status=ToolStatus.ACTIVE))
    
    draft_tools = manager.get_tools_by_status(ToolStatus.DRAFT)
    assert len(draft_tools) == 2
    assert "draft1" in draft_tools
    assert "draft2" in draft_tools
    
    active_tools = manager.get_tools_by_status(ToolStatus.ACTIVE)
    assert len(active_tools) == 1
    assert "active1" in active_tools


def test_audit_log():
    """Test audit log functionality."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(status=ToolStatus.DRAFT)
    manager.register_tool_metadata("test_tool", metadata)
    
    # Perform some actions
    manager.update_tool_status("test_tool", ToolStatus.TESTING)
    manager.approve_tool("test_tool", approver="admin")
    
    # Get audit log
    log = manager.get_audit_log(tool_name="test_tool")
    assert len(log) >= 3  # registration + status change + approval
    
    # Check event types
    event_types = {entry["event_type"] for entry in log}
    assert "tool_registered" in event_types


def test_export_governance_data():
    """Test exporting governance data."""
    manager = ToolGovernanceManager()
    
    metadata = ToolMetadata(
        status=ToolStatus.ACTIVE,
        owner="team1"
    )
    manager.register_tool_metadata("test_tool", metadata)
    manager.track_usage("test_tool")
    
    # Export data
    data = manager.export_governance_data()
    
    assert "metadata" in data
    assert "test_tool" in data["metadata"]
    assert "usage_tracking" in data
    assert "audit_log" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
