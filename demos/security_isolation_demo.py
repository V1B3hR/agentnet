#!/usr/bin/env python3
"""
Comprehensive demonstration of AgentNet Security & Isolation features.

This script demonstrates all implemented security and isolation mechanisms:
1. Multi-tenant isolation
2. Session isolation with different levels
3. Resource access control
4. Resource locking and coordination
5. Authentication with JWT tokens
6. Role-based access control (RBAC)
7. Network and data access policies
"""

from agentnet.core.auth import (
    RBACManager,
    Role,
    AuthMiddleware,
    SecurityIsolationManager,
)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def main():
    print("\n🔒 AgentNet Security & Isolation Features Demonstration")
    print("=" * 80)
    
    # Initialize security components
    rbac_manager = RBACManager()
    auth_middleware = AuthMiddleware(rbac_manager, secret_key="demo-secret-key")
    isolation_manager = SecurityIsolationManager()
    
    # =========================================================================
    # 1. Multi-Tenant Setup
    # =========================================================================
    print_section("1. Multi-Tenant User Setup")
    
    # Create users in different tenants with different roles
    admin_user = rbac_manager.create_user(
        user_id="admin001",
        username="system_admin",
        email="admin@agentnet.com",
        roles=[Role.ADMIN],
        tenant_id=None  # Global admin
    )
    print(f"✓ Created global admin: {admin_user.username}")
    
    tenant1_admin = rbac_manager.create_user(
        user_id="t1admin001",
        username="tenant1_admin",
        email="admin@tenant1.com",
        roles=[Role.ADMIN, Role.TENANT_USER],
        tenant_id="tenant1"
    )
    print(f"✓ Created Tenant 1 admin: {tenant1_admin.username}")
    
    tenant1_user = rbac_manager.create_user(
        user_id="t1user001",
        username="alice",
        email="alice@tenant1.com",
        roles=[Role.TENANT_USER],
        tenant_id="tenant1"
    )
    print(f"✓ Created Tenant 1 user: {tenant1_user.username}")
    
    tenant2_user = rbac_manager.create_user(
        user_id="t2user001",
        username="bob",
        email="bob@tenant2.com",
        roles=[Role.TENANT_USER],
        tenant_id="tenant2"
    )
    print(f"✓ Created Tenant 2 user: {tenant2_user.username}")
    
    auditor = rbac_manager.create_user(
        user_id="audit001",
        username="auditor",
        email="auditor@agentnet.com",
        roles=[Role.AUDITOR],
        tenant_id=None  # Cross-tenant auditor
    )
    print(f"✓ Created auditor: {auditor.username}")
    
    # =========================================================================
    # 2. JWT Token Generation and Verification
    # =========================================================================
    print_section("2. JWT Authentication")
    
    # Create tokens for users
    admin_token = auth_middleware.create_token(admin_user)
    alice_token = auth_middleware.create_token(tenant1_user)
    
    print(f"✓ Generated JWT token for admin (truncated): {admin_token[:50]}...")
    print(f"✓ Generated JWT token for alice (truncated): {alice_token[:50]}...")
    
    # Verify tokens
    admin_payload = auth_middleware.verify_token(admin_token)
    alice_payload = auth_middleware.verify_token(alice_token)
    
    print(f"✓ Admin token verified: user_id={admin_payload['user_id']}, roles={admin_payload['roles']}")
    print(f"✓ Alice token verified: user_id={alice_payload['user_id']}, tenant={alice_payload['tenant_id']}")
    
    # =========================================================================
    # 3. Isolated Session Creation with Different Levels
    # =========================================================================
    print_section("3. Isolated Session Creation")
    
    # Create sessions with different isolation levels
    admin_session = isolation_manager.create_isolated_session(
        admin_user, "admin_session_001", isolation_level="basic"
    )
    print(f"✓ Created admin session with 'basic' isolation")
    print(f"  - Compute quota: {admin_session['resource_access']['compute_quota']}")
    print(f"  - Memory limit: {admin_session['resource_access']['memory_limit_mb']} MB")
    print(f"  - Network access: {admin_session['resource_access']['network_access']}")
    
    alice_session = isolation_manager.create_isolated_session(
        tenant1_user, "alice_session_001", isolation_level="standard"
    )
    print(f"\n✓ Created alice session with 'standard' isolation")
    print(f"  - Compute quota: {alice_session['resource_access']['compute_quota']}")
    print(f"  - Memory limit: {alice_session['resource_access']['memory_limit_mb']} MB")
    print(f"  - Network access: {alice_session['resource_access']['network_access']}")
    
    bob_session = isolation_manager.create_isolated_session(
        tenant2_user, "bob_session_001", isolation_level="strict"
    )
    print(f"\n✓ Created bob session with 'strict' isolation")
    print(f"  - Compute quota: {bob_session['resource_access']['compute_quota']}")
    print(f"  - Memory limit: {bob_session['resource_access']['memory_limit_mb']} MB")
    print(f"  - Network access: {bob_session['resource_access']['network_access']}")
    
    # =========================================================================
    # 4. Tenant Isolation Boundaries
    # =========================================================================
    print_section("4. Tenant Isolation Boundaries")
    
    print("Tenant boundaries:")
    for tenant_id, session_ids in isolation_manager._tenant_boundaries.items():
        print(f"  - {tenant_id}: {len(session_ids)} active session(s) -> {list(session_ids)}")
    
    # =========================================================================
    # 5. Network and Data Access Policies
    # =========================================================================
    print_section("5. Network and Data Access Policies")
    
    print("Admin network policy:")
    admin_network = admin_session['network_policy']
    print(f"  - Outbound allowed: {admin_network['outbound_allowed']}")
    print(f"  - Rate limit: {admin_network['rate_limits']['requests_per_minute']} req/min")
    print(f"  - Allowed domains: {admin_network['allowed_domains'][:3]}...")
    
    print("\nAlice network policy:")
    alice_network = alice_session['network_policy']
    print(f"  - Outbound allowed: {alice_network['outbound_allowed']}")
    print(f"  - Rate limit: {alice_network['rate_limits']['requests_per_minute']} req/min")
    
    print("\nBob network policy (strict):")
    bob_network = bob_session['network_policy']
    print(f"  - Outbound allowed: {bob_network['outbound_allowed']}")
    print(f"  - Blocked domains: {bob_network['blocked_domains']}")
    
    print("\nData access policies:")
    print(f"  - Admin: {admin_session['data_access_policy']['classification_levels']}")
    print(f"  - Alice: {alice_session['data_access_policy']['classification_levels']}")
    print(f"  - Admin cross-tenant access: {admin_session['data_access_policy']['cross_tenant_access']}")
    print(f"  - Alice cross-tenant access: {alice_session['data_access_policy']['cross_tenant_access']}")
    
    # =========================================================================
    # 6. Resource Access Validation
    # =========================================================================
    print_section("6. Resource Access Validation")
    
    # Test resource access for different sessions
    test_resources = [
        ("alice_session_001", "compute", "gpu_cluster_1"),
        ("alice_session_001", "network", "external_api"),
        ("bob_session_001", "compute", "cpu_pool"),
        ("bob_session_001", "network", "external_api"),
    ]
    
    for session_id, resource_type, resource_id in test_resources:
        can_access = isolation_manager.validate_resource_access(
            session_id, resource_type, resource_id
        )
        status = "✓ ALLOWED" if can_access else "✗ DENIED"
        print(f"{status}: {session_id} -> {resource_type}:{resource_id}")
    
    # =========================================================================
    # 7. Resource Locking and Coordination
    # =========================================================================
    print_section("7. Resource Locking and Coordination")
    
    # Alice acquires a shared resource
    shared_resource = "shared_compute_cluster"
    lock1 = isolation_manager.acquire_resource_lock("alice_session_001", shared_resource)
    print(f"✓ Alice acquired lock on '{shared_resource}': {lock1}")
    
    # Bob tries to acquire the same resource
    lock2 = isolation_manager.acquire_resource_lock("bob_session_001", shared_resource)
    print(f"✗ Bob tried to acquire lock on '{shared_resource}': {lock2}")
    
    # Alice releases the lock
    released = isolation_manager.release_resource_lock("alice_session_001", shared_resource)
    print(f"✓ Alice released lock: {released}")
    
    # Now Bob can acquire it
    lock3 = isolation_manager.acquire_resource_lock("bob_session_001", shared_resource)
    print(f"✓ Bob acquired lock on '{shared_resource}': {lock3}")
    
    # =========================================================================
    # 8. Isolation System Statistics
    # =========================================================================
    print_section("8. Isolation System Statistics")
    
    stats = isolation_manager.get_isolation_stats()
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"Tenant boundaries: {stats['tenant_boundaries']}")
    print(f"Active resource locks: {stats['resource_locks']}")
    print(f"\nIsolation features:")
    for feature, enabled in stats['isolation_features'].items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {status}: {feature}")
    
    # =========================================================================
    # 9. Session Cleanup
    # =========================================================================
    print_section("9. Session Cleanup")
    
    # Clean up alice's session
    print(f"Cleaning up alice_session_001...")
    isolation_manager.cleanup_session("alice_session_001")
    
    # Get updated stats
    stats_after = isolation_manager.get_isolation_stats()
    print(f"✓ Session cleaned up")
    print(f"Active sessions: {stats['active_sessions']} -> {stats_after['active_sessions']}")
    print(f"Active resource locks: {stats['resource_locks']} -> {stats_after['resource_locks']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")
    
    print("✅ All security and isolation features demonstrated successfully!")
    print("\nKey Features Verified:")
    print("  ✓ Multi-tenant user and session isolation")
    print("  ✓ Role-based access control (RBAC)")
    print("  ✓ JWT token generation and verification")
    print("  ✓ Multiple isolation levels (basic, standard, strict)")
    print("  ✓ Resource access control and validation")
    print("  ✓ Exclusive resource locking for coordination")
    print("  ✓ Network and data access policies")
    print("  ✓ Tenant boundary enforcement")
    print("  ✓ Session cleanup and resource release")
    print("  ✓ Real-time isolation statistics")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully! 🎉")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
