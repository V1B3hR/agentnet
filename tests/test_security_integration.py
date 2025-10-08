#!/usr/bin/env python3
"""
Integration test for security and isolation features.

This test demonstrates that all security isolation features work together
in a realistic multi-tenant scenario.
"""

import pytest
from agentnet.core.auth import (
    RBACManager,
    Role,
    AuthMiddleware,
    SecurityIsolationManager,
)


class TestSecurityIntegration:
    """Integration tests for complete security workflows."""

    def setup_method(self):
        """Set up test environment."""
        self.rbac_manager = RBACManager()
        self.auth_middleware = AuthMiddleware(
            self.rbac_manager, secret_key="test-secret"
        )
        self.isolation_manager = SecurityIsolationManager()

    def test_multi_tenant_isolation_workflow(self):
        """Test complete multi-tenant isolation workflow."""
        # Create users in different tenants
        tenant1_admin = self.rbac_manager.create_user(
            "t1_admin", "Tenant1 Admin", "admin@tenant1.com", [Role.ADMIN], "tenant1"
        )
        tenant1_user = self.rbac_manager.create_user(
            "t1_user", "Tenant1 User", "user@tenant1.com", [Role.TENANT_USER], "tenant1"
        )
        tenant2_user = self.rbac_manager.create_user(
            "t2_user", "Tenant2 User", "user@tenant2.com", [Role.TENANT_USER], "tenant2"
        )

        # Create isolated sessions for each user
        t1_admin_session = self.isolation_manager.create_isolated_session(
            tenant1_admin, "t1_admin_session", "basic"
        )
        t1_user_session = self.isolation_manager.create_isolated_session(
            tenant1_user, "t1_user_session", "standard"
        )
        t2_user_session = self.isolation_manager.create_isolated_session(
            tenant2_user, "t2_user_session", "strict"
        )

        # Verify session isolation
        assert t1_admin_session["session_id"] == "t1_admin_session"
        assert t1_user_session["session_id"] == "t1_user_session"
        assert t2_user_session["session_id"] == "t2_user_session"

        # Verify tenant boundaries
        assert "tenant1" in self.isolation_manager._tenant_boundaries
        assert "tenant2" in self.isolation_manager._tenant_boundaries
        assert (
            "t1_admin_session" in self.isolation_manager._tenant_boundaries["tenant1"]
        )
        assert "t1_user_session" in self.isolation_manager._tenant_boundaries["tenant1"]
        assert "t2_user_session" in self.isolation_manager._tenant_boundaries["tenant2"]

        # Verify resource access differs by role and isolation level
        admin_resources = t1_admin_session["resource_access"]
        user_resources = t1_user_session["resource_access"]
        strict_resources = t2_user_session["resource_access"]

        # Admin should have more resources
        assert admin_resources["compute_quota"] > user_resources["compute_quota"]
        assert admin_resources["network_access"] == "full"
        assert user_resources["network_access"] == "restricted"

        # Strict isolation should have least resources
        assert strict_resources["compute_quota"] < user_resources["compute_quota"]
        assert strict_resources["network_access"] == "none"

        # Verify data access policies
        admin_policy = t1_admin_session["data_access_policy"]
        user_policy = t1_user_session["data_access_policy"]

        assert admin_policy["cross_tenant_access"] is True
        assert user_policy["cross_tenant_access"] is False
        assert admin_policy["audit_logs_access"] is True
        assert user_policy["audit_logs_access"] is False

    def test_resource_locking_coordination(self):
        """Test resource locking prevents conflicts between sessions."""
        # Create two users and sessions
        user1 = self.rbac_manager.create_user(
            "user1", "User 1", "user1@test.com", [Role.TENANT_USER], "tenant1"
        )
        user2 = self.rbac_manager.create_user(
            "user2", "User 2", "user2@test.com", [Role.TENANT_USER], "tenant1"
        )

        session1 = self.isolation_manager.create_isolated_session(
            user1, "session1", "standard"
        )
        session2 = self.isolation_manager.create_isolated_session(
            user2, "session2", "standard"
        )

        # Session 1 acquires a resource lock
        resource_id = "shared_compute_resource"
        lock1 = self.isolation_manager.acquire_resource_lock("session1", resource_id)
        assert lock1 is True

        # Session 2 should not be able to acquire the same lock
        lock2 = self.isolation_manager.acquire_resource_lock("session2", resource_id)
        assert lock2 is False

        # Verify resource access validation
        can_access_1 = self.isolation_manager.validate_resource_access(
            "session1", "compute", "gpu1"
        )
        can_access_2 = self.isolation_manager.validate_resource_access(
            "session2", "compute", "gpu1"
        )

        assert can_access_1 is True
        assert can_access_2 is True  # Different resource, should be accessible

        # Release the lock
        released = self.isolation_manager.release_resource_lock("session1", resource_id)
        assert released is True

        # Now session 2 should be able to acquire it
        lock2_retry = self.isolation_manager.acquire_resource_lock(
            "session2", resource_id
        )
        assert lock2_retry is True

    def test_session_cleanup_releases_resources(self):
        """Test that session cleanup properly releases all resources."""
        # Create user and session
        user = self.rbac_manager.create_user(
            "test_user", "Test User", "test@test.com", [Role.TENANT_USER], "test_tenant"
        )

        session_id = "cleanup_test_session"
        session = self.isolation_manager.create_isolated_session(
            user, session_id, "standard"
        )

        # Acquire multiple resource locks
        resources = ["resource1", "resource2", "resource3"]
        for resource_id in resources:
            lock = self.isolation_manager.acquire_resource_lock(session_id, resource_id)
            assert lock is True

        # Verify session and locks exist
        assert session_id in self.isolation_manager._active_sessions
        locked_count = sum(
            1
            for r, s in self.isolation_manager._resource_locks.items()
            if s == session_id
        )
        assert locked_count == 3

        # Clean up session
        self.isolation_manager.cleanup_session(session_id)

        # Verify all resources are released
        assert session_id not in self.isolation_manager._active_sessions
        locked_count_after = sum(
            1
            for r, s in self.isolation_manager._resource_locks.items()
            if s == session_id
        )
        assert locked_count_after == 0

        # Verify tenant boundary is updated
        if "test_tenant" in self.isolation_manager._tenant_boundaries:
            assert (
                session_id
                not in self.isolation_manager._tenant_boundaries["test_tenant"]
            )

    def test_authentication_with_isolation(self):
        """Test that authentication and isolation work together."""
        # Create user
        user = self.rbac_manager.create_user(
            "auth_user", "Auth User", "auth@test.com", [Role.TENANT_USER], "auth_tenant"
        )

        # Create token
        token = self.auth_middleware.create_token(user)
        assert token is not None

        # Verify token
        payload = self.auth_middleware.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == user.user_id
        assert payload["tenant_id"] == "auth_tenant"

        # Create isolated session using authenticated user
        session = self.isolation_manager.create_isolated_session(
            user, f"session_{user.user_id}", "standard"
        )

        # Verify session has correct user and tenant info
        assert session["user_id"] == user.user_id
        assert session["tenant_id"] == "auth_tenant"
        assert session["isolation_level"] == "standard"

        # Verify the session is in tenant boundaries
        assert "auth_tenant" in self.isolation_manager._tenant_boundaries
        assert (
            f"session_{user.user_id}"
            in self.isolation_manager._tenant_boundaries["auth_tenant"]
        )

    def test_isolation_stats_accuracy(self):
        """Test that isolation statistics are accurate."""
        # Start with clean state
        initial_stats = self.isolation_manager.get_isolation_stats()
        initial_sessions = initial_stats["active_sessions"]

        # Create multiple sessions
        users = []
        sessions = []
        for i in range(3):
            user = self.rbac_manager.create_user(
                f"stats_user_{i}",
                f"User {i}",
                f"user{i}@test.com",
                [Role.TENANT_USER],
                f"tenant_{i % 2}",  # 2 tenants
            )
            users.append(user)

            session = self.isolation_manager.create_isolated_session(
                user, f"stats_session_{i}", "standard"
            )
            sessions.append(session)

        # Acquire some resource locks
        self.isolation_manager.acquire_resource_lock("stats_session_0", "resource_A")
        self.isolation_manager.acquire_resource_lock("stats_session_1", "resource_B")

        # Check stats
        stats = self.isolation_manager.get_isolation_stats()
        assert stats["active_sessions"] == initial_sessions + 3
        assert stats["resource_locks"] >= 2
        assert stats["tenant_boundaries"] >= 2

        # Verify isolation features are enabled
        features = stats["isolation_features"]
        assert features["session_isolation"] is True
        assert features["tenant_isolation"] is True
        assert features["resource_isolation"] is True
        assert features["data_classification"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
