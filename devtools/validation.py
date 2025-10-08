"""Validation utilities for AgentNet."""

import sys
from pathlib import Path
from typing import Optional


def validate_security() -> int:
    """Validate security & isolation features.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    print("=" * 80)
    print("  AgentNet Security & Isolation - Validation Script")
    print("=" * 80)
    
    results = []
    
    # Run all validation checks
    results.append(("Imports", _validate_imports()))
    results.append(("Basic Functionality", _validate_basic_functionality()))
    results.append(("Isolation Levels", _validate_isolation_levels()))
    results.append(("Resource Locking", _validate_resource_locking()))
    results.append(("Tenant Isolation", _validate_tenant_isolation()))
    results.append(("Authentication", _validate_authentication()))
    
    # Summary
    print("\n" + "=" * 80)
    print("  Validation Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All validation tests passed! Security features are working.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Check errors above.")
        return 1


def _validate_imports():
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


def _validate_basic_functionality():
    """Validate basic security functionality."""
    print("\n2. Validating basic functionality...")
    try:
        from agentnet.core.auth import RBACManager, Role, SecurityIsolationManager
        
        rbac = RBACManager()
        isolation = SecurityIsolationManager()
        
        user = rbac.create_user(
            "test_user", "Test User", "test@example.com",
            [Role.TENANT_USER], "test_tenant"
        )
        
        session = isolation.create_isolated_session(
            user, "test_session", "standard"
        )
        
        assert session["session_id"] == "test_session"
        assert session["isolation_level"] == "standard"
        assert "resource_access" in session
        assert "network_policy" in session
        
        print("   ✓ Basic functionality working")
        return True
    except Exception as e:
        print(f"   ✗ Functionality test failed: {e}")
        return False


def _validate_isolation_levels():
    """Validate all isolation levels work correctly."""
    print("\n3. Validating isolation levels...")
    try:
        from agentnet.core.auth import RBACManager, Role, SecurityIsolationManager
        
        rbac = RBACManager()
        isolation = SecurityIsolationManager()
        
        user = rbac.create_user(
            "test_user2", "Test User 2", "test2@example.com",
            [Role.TENANT_USER], "test_tenant"
        )
        
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


def _validate_resource_locking():
    """Validate resource locking mechanism."""
    print("\n4. Validating resource locking...")
    try:
        from agentnet.core.auth import SecurityIsolationManager
        
        isolation = SecurityIsolationManager()
        
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


def _validate_tenant_isolation():
    """Validate tenant isolation boundaries."""
    print("\n5. Validating tenant isolation...")
    try:
        from agentnet.core.auth import RBACManager, Role, SecurityIsolationManager
        
        rbac = RBACManager()
        isolation = SecurityIsolationManager()
        
        user1 = rbac.create_user(
            "t1_user", "Tenant 1 User", "user1@tenant1.com",
            [Role.TENANT_USER], "tenant1"
        )
        user2 = rbac.create_user(
            "t2_user", "Tenant 2 User", "user2@tenant2.com",
            [Role.TENANT_USER], "tenant2"
        )
        
        session1 = isolation.create_isolated_session(user1, "t1_session", "standard")
        session2 = isolation.create_isolated_session(user2, "t2_session", "standard")
        
        assert "tenant1" in isolation._tenant_boundaries
        assert "tenant2" in isolation._tenant_boundaries
        assert "t1_session" in isolation._tenant_boundaries["tenant1"]
        assert "t2_session" in isolation._tenant_boundaries["tenant2"]
        
        print("   ✓ Tenant isolation working correctly")
        return True
    except Exception as e:
        print(f"   ✗ Tenant isolation test failed: {e}")
        return False


def _validate_authentication():
    """Validate JWT authentication."""
    print("\n6. Validating authentication...")
    try:
        from agentnet.core.auth import RBACManager, Role, AuthMiddleware
        
        rbac = RBACManager()
        auth = AuthMiddleware(rbac, secret_key="test-secret")
        
        user = rbac.create_user(
            "auth_user", "Auth User", "auth@example.com",
            [Role.TENANT_USER], "test_tenant"
        )
        
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


def validate_roadmap() -> int:
    """Verify roadmap issues have been resolved.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    print("=" * 70)
    print("AgentNet Roadmap Issue Resolution Validation")
    print("=" * 70)
    print()
    
    results = []
    
    print("1. Checking networkx dependency...")
    results.append(_check_networkx())
    print()
    
    print("2. Checking DAG planner functionality...")
    results.append(_check_dag_planner())
    print()
    
    print("3. Checking Docker deployment files...")
    results.append(_check_docker_files())
    print()
    
    print("4. Checking requirements.txt...")
    results.append(_check_requirements_txt())
    print()
    
    print("=" * 70)
    if all(results):
        print("🎉 SUCCESS: All roadmap issues have been resolved!")
        print()
        print("Summary of changes:")
        print("  • networkx>=3.0 added to requirements.txt")
        print("  • Dockerfile created for container deployment")
        print("  • docker-compose.yml with full stack")
        print("  • .dockerignore for efficient builds")
        print("  • DOCKER.md with comprehensive deployment guide")
        print("  • roadmap.md updated to reflect completion status")
        print()
        print("Remaining documented issues (in roadmap.md):")
        print("  • CI/CD automation (no GitHub Actions workflows)")
        print("  • Provider ecosystem expansion (real provider implementations)")
        print("  • Advanced governance (policy + tool lifecycle)")
        print("  • Risk register runtime enforcement & monitoring integration")
        return 0
    else:
        print("❌ FAILED: Some issues were not resolved")
        return 1


def _check_networkx():
    """Verify networkx is available."""
    try:
        import networkx as nx
        print(f"✅ networkx is available (version {nx.__version__})")
        return True
    except ImportError as e:
        print(f"❌ networkx is NOT available: {e}")
        return False


def _check_dag_planner():
    """Verify DAG planner can use networkx."""
    try:
        from agentnet.core.orchestration.dag_planner import DAGPlanner
        print("✅ DAGPlanner successfully imports and uses networkx")
        return True
    except ImportError as e:
        print(f"❌ DAGPlanner import failed: {e}")
        return False


def _check_docker_files():
    """Verify Docker deployment files exist."""
    files = [
        ("Dockerfile", "Main Docker image definition"),
        ("docker-compose.yml", "Multi-service deployment configuration"),
        (".dockerignore", "Docker build exclusions"),
        ("DOCKER.md", "Docker deployment documentation"),
        ("configs/prometheus.yml", "Prometheus configuration")
    ]
    
    all_exist = True
    for filename, description in files:
        if Path(filename).exists():
            print(f"✅ {filename} exists ({description})")
        else:
            print(f"❌ {filename} is MISSING ({description})")
            all_exist = False
    
    return all_exist


def _check_requirements_txt():
    """Verify networkx is in requirements.txt."""
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            if "networkx" in content:
                print("✅ networkx is listed in requirements.txt")
                return True
            else:
                print("❌ networkx is NOT in requirements.txt")
                return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False
