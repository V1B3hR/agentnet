#!/usr/bin/env python3
"""
Quick validation script for security & isolation features.
Run this to verify all security features are working correctly.

DEPRECATED: This script is deprecated and will be removed in a future release.
Please use the unified CLI instead:
    python -m cli.main validate-security
    or: python cli/main.py validate-security
"""

import sys


def main():
    """Run all validation tests."""
    print("⚠️  DEPRECATION WARNING: This script is deprecated.")
    print("    Please use: python -m cli.main validate-security")
    print()

    # Import and run the validation
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from devtools.validation import validate_security

    return validate_security()


def validate_imports():
    """Validate all security module imports work."""
    print("1. Validating imports...")
    try:
        from agentnet.core.auth import (
            RBACManager,
            Role,
            AuthMiddleware,
            SecurityIsolationManager,
        )

        print("   ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False


def validate_basic_functionality():
    """Validate basic security functionality."""
    print("\n2. Validating basic functionality...")
    try:
        from agentnet.core.auth import RBACManager, Role, SecurityIsolationManager

        # Create managers
        rbac = RBACManager()
        isolation = SecurityIsolationManager()

        # Create user
        user = rbac.create_user(
            "test_user",
            "Test User",
            "test@example.com",
            [Role.TENANT_USER],
            "test_tenant",
        )

        # Create session
        session = isolation.create_isolated_session(user, "test_session", "standard")

        # Verify session
        assert session["session_id"] == "test_session"
        assert session["isolation_level"] == "standard"
        assert "resource_access" in session
        assert "network_policy" in session

        print("   ✓ Basic functionality working")
        return True
    except Exception as e:
        print(f"   ✗ Functionality test failed: {e}")
        return False


def validate_isolation_levels():
    """Validate all isolation levels work correctly."""
    print("\n3. Validating isolation levels...")
    try:
        from agentnet.core.auth import RBACManager, Role, SecurityIsolationManager

        rbac = RBACManager()
        isolation = SecurityIsolationManager()

        user = rbac.create_user(
            "test_user2",
            "Test User 2",
            "test2@example.com",
            [Role.TENANT_USER],
            "test_tenant",
        )

        # Test all isolation levels
        for level in ["basic", "standard", "strict"]:
            session = isolation.create_isolated_session(
                user, f"test_session_{level}", level
            )
            assert session["isolation_level"] == level

        print("   ✓ All isolation levels working (basic, standard, strict)")
        return True
    except Exception as e:
        print(f"   ✗ Isolation levels test failed: {e}")
        return False


def validate_resource_locking():
    """Validate resource locking mechanism."""
    print("\n4. Validating resource locking...")
    try:
        from agentnet.core.auth import SecurityIsolationManager

        isolation = SecurityIsolationManager()

        # Test locking
        lock1 = isolation.acquire_resource_lock("session1", "resource1")
        assert lock1 is True, "First lock should succeed"

        lock2 = isolation.acquire_resource_lock("session2", "resource1")
        assert lock2 is False, "Second lock should fail"

        released = isolation.release_resource_lock("session1", "resource1")
        assert released is True, "Release should succeed"

        lock3 = isolation.acquire_resource_lock("session2", "resource1")
        assert lock3 is True, "Lock after release should succeed"

        print("   ✓ Resource locking working correctly")
        return True
    except Exception as e:
        print(f"   ✗ Resource locking test failed: {e}")
        return False


def validate_tenant_isolation():
    """Validate tenant isolation boundaries."""
    print("\n5. Validating tenant isolation...")
    try:
        from agentnet.core.auth import RBACManager, Role, SecurityIsolationManager

        rbac = RBACManager()
        isolation = SecurityIsolationManager()

        # Create users in different tenants
        user1 = rbac.create_user(
            "t1_user",
            "Tenant 1 User",
            "user1@tenant1.com",
            [Role.TENANT_USER],
            "tenant1",
        )
        user2 = rbac.create_user(
            "t2_user",
            "Tenant 2 User",
            "user2@tenant2.com",
            [Role.TENANT_USER],
            "tenant2",
        )

        # Create sessions
        session1 = isolation.create_isolated_session(user1, "t1_session", "standard")
        session2 = isolation.create_isolated_session(user2, "t2_session", "standard")

        # Verify tenant boundaries
        assert "tenant1" in isolation._tenant_boundaries
        assert "tenant2" in isolation._tenant_boundaries
        assert "t1_session" in isolation._tenant_boundaries["tenant1"]
        assert "t2_session" in isolation._tenant_boundaries["tenant2"]

        print("   ✓ Tenant isolation working correctly")
        return True
    except Exception as e:
        print(f"   ✗ Tenant isolation test failed: {e}")
        return False


def validate_authentication():
    """Validate JWT authentication."""
    print("\n6. Validating authentication...")
    try:
        from agentnet.core.auth import RBACManager, Role, AuthMiddleware

        rbac = RBACManager()
        auth = AuthMiddleware(rbac, secret_key="test-secret")

        # Create user
        user = rbac.create_user(
            "auth_user",
            "Auth User",
            "auth@example.com",
            [Role.TENANT_USER],
            "test_tenant",
        )

        # Create and verify token
        token = auth.create_token(user)
        assert token is not None, "Token should be created"

        payload = auth.verify_token(token)
        assert payload is not None, "Token should be valid"
        assert payload["user_id"] == user.user_id

        print("   ✓ JWT authentication working correctly")
        return True
    except Exception as e:
        print(f"   ✗ Authentication test failed: {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())
