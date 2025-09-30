# Security & Isolation Implementation Guide

This document provides a comprehensive guide to AgentNet's security and isolation features.

## Overview

AgentNet implements a multi-layered security architecture designed for multi-tenant deployments with the following key components:

1. **Authentication & Authorization** - JWT-based authentication with Role-Based Access Control (RBAC)
2. **Session Isolation** - Isolated execution contexts with configurable isolation levels
3. **Multi-Tenant Isolation** - Strict boundaries between tenant data and resources
4. **Resource Access Control** - Fine-grained control over compute, network, and data resources
5. **Resource Locking** - Coordination mechanisms for shared resource access
6. **Network Policies** - Configurable network access rules per session
7. **Data Access Policies** - Classification-based data access controls

## Quick Start

```python
from agentnet.core.auth import RBACManager, SecurityIsolationManager, Role

# Initialize components
rbac_manager = RBACManager()
isolation_manager = SecurityIsolationManager()

# Create a user
user = rbac_manager.create_user(
    user_id="user123",
    username="alice",
    email="alice@company.com",
    roles=[Role.TENANT_USER],
    tenant_id="tenant1"
)

# Create an isolated session
session = isolation_manager.create_isolated_session(
    user=user,
    session_id="session_001",
    isolation_level="standard"  # Options: "basic", "standard", "strict"
)

print(f"Session created with isolation level: {session['isolation_level']}")
print(f"Resource access: {session['resource_access']}")
print(f"Network policy: {session['network_policy']}")
```

## Architecture

### SecurityIsolationManager

The `SecurityIsolationManager` class is the core component that manages all isolation features:

```python
class SecurityIsolationManager:
    def __init__(self):
        self.session_isolation = True
        self.tenant_isolation = True
        self.resource_isolation = True
        self.data_classification_enabled = True
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_isolated_session()` | Create a new isolated session with specified isolation level |
| `validate_resource_access()` | Check if a session can access a specific resource |
| `acquire_resource_lock()` | Acquire exclusive lock on a resource |
| `release_resource_lock()` | Release a resource lock |
| `cleanup_session()` | Clean up session and release all resources |
| `get_isolation_stats()` | Get current isolation system statistics |

## Isolation Levels

AgentNet supports three isolation levels that provide different trade-offs between security and resource access:

### Strict Isolation

**Use Case:** High-security environments, untrusted code execution

```python
session = isolation_manager.create_isolated_session(
    user=user,
    session_id="strict_session",
    isolation_level="strict"
)
```

**Resource Limits:**
- Compute quota: 50 units
- Memory limit: 256 MB
- Network access: None
- External API access: Disabled

### Standard Isolation (Default)

**Use Case:** Normal multi-tenant operations

```python
session = isolation_manager.create_isolated_session(
    user=user,
    session_id="standard_session",
    isolation_level="standard"
)
```

**Resource Limits:**
- Compute quota: 100 units
- Memory limit: 512 MB
- Network access: Restricted
- External API access: Disabled

### Basic Isolation

**Use Case:** Trusted users, internal services

```python
session = isolation_manager.create_isolated_session(
    user=user,
    session_id="basic_session",
    isolation_level="basic"
)
```

**Resource Limits:**
- Compute quota: 200 units
- Memory limit: 1024 MB
- Network access: Limited
- External API access: Enabled

## Role-Based Access Control

AgentNet uses RBAC to control user permissions:

```python
from agentnet.core.auth import Role

# Available roles
Role.ADMIN          # Full system access
Role.OPERATOR       # Operational tasks
Role.AUDITOR        # Read-only audit access
Role.TENANT_USER    # Regular tenant user
Role.GUEST          # Limited guest access
```

### Role Hierarchy

```
ADMIN (Highest privileges)
  ├── Full cross-tenant access
  ├── All resource access
  └── Audit log access
  
OPERATOR
  ├── Operational control
  └── Resource management
  
AUDITOR
  ├── Read-only access
  └── Audit log access
  
TENANT_USER (Tenant-scoped)
  ├── Tenant data access
  └── Limited resources
  
GUEST (Lowest privileges)
  └── Minimal access
```

## Multi-Tenant Isolation

Tenant isolation ensures complete data and resource separation:

```python
# Create users in different tenants
tenant1_user = rbac_manager.create_user(
    "user1", "Alice", "alice@tenant1.com",
    [Role.TENANT_USER], "tenant1"
)

tenant2_user = rbac_manager.create_user(
    "user2", "Bob", "bob@tenant2.com",
    [Role.TENANT_USER], "tenant2"
)

# Create isolated sessions
session1 = isolation_manager.create_isolated_session(
    tenant1_user, "session1", "standard"
)
session2 = isolation_manager.create_isolated_session(
    tenant2_user, "session2", "standard"
)

# Verify tenant boundaries
print(f"Tenant boundaries: {isolation_manager._tenant_boundaries}")
# Output: {'tenant1': {'session1'}, 'tenant2': {'session2'}}
```

### Cross-Tenant Access

Only users with `ADMIN` role can access resources across tenants:

```python
admin_session = isolation_manager.create_isolated_session(
    admin_user, "admin_session", "basic"
)

# Check data access policy
policy = admin_session['data_access_policy']
print(f"Cross-tenant access: {policy['cross_tenant_access']}")  # True for admin
```

## Resource Locking

Prevent race conditions with exclusive resource locks:

```python
# Session 1 acquires a lock
success = isolation_manager.acquire_resource_lock(
    session_id="session1",
    resource_id="shared_database"
)
print(f"Lock acquired: {success}")  # True

# Session 2 tries to acquire the same lock
blocked = isolation_manager.acquire_resource_lock(
    session_id="session2",
    resource_id="shared_database"
)
print(f"Lock acquired: {blocked}")  # False - already locked

# Session 1 releases the lock
isolation_manager.release_resource_lock(
    session_id="session1",
    resource_id="shared_database"
)

# Now session 2 can acquire it
success = isolation_manager.acquire_resource_lock(
    session_id="session2",
    resource_id="shared_database"
)
print(f"Lock acquired: {success}")  # True
```

## Network Policies

Control network access per session:

```python
session = isolation_manager.create_isolated_session(
    user, "session_001", "standard"
)

network_policy = session['network_policy']
print(f"Outbound allowed: {network_policy['outbound_allowed']}")
print(f"Allowed domains: {network_policy['allowed_domains']}")
print(f"Rate limit: {network_policy['rate_limits']['requests_per_minute']}")
```

### Default Network Policies

| Isolation Level | Outbound | Allowed Domains | Rate Limit |
|----------------|----------|-----------------|------------|
| Strict | No | None | 10 req/min |
| Standard | No | Internal APIs | 10 req/min |
| Basic | Yes | Selected APIs | 50 req/min |
| Admin | Yes | All | 100 req/min |

## Data Access Policies

Control data access based on classification:

```python
session = isolation_manager.create_isolated_session(
    user, "session_001", "standard"
)

data_policy = session['data_access_policy']
print(f"Classification levels: {data_policy['classification_levels']}")
print(f"PII access: {data_policy['pii_access']}")
print(f"Audit logs access: {data_policy['audit_logs_access']}")
```

### Data Classification Levels

- **Public**: Publicly accessible data
- **Internal**: Internal company data
- **Confidential**: Sensitive business data
- **Restricted**: Highly sensitive data (requires special permissions)

## Session Cleanup

Always clean up sessions to release resources:

```python
# Automatic cleanup releases all locks
isolation_manager.cleanup_session("session_001")

# Verify cleanup
stats = isolation_manager.get_isolation_stats()
print(f"Active sessions: {stats['active_sessions']}")
```

## Best Practices

### 1. Use Appropriate Isolation Levels

```python
# For untrusted code
session = isolation_manager.create_isolated_session(
    user, "untrusted_session", isolation_level="strict"
)

# For internal tools
session = isolation_manager.create_isolated_session(
    admin, "internal_session", isolation_level="basic"
)
```

### 2. Always Clean Up Sessions

```python
try:
    session = isolation_manager.create_isolated_session(
        user, "temp_session", "standard"
    )
    # Do work...
finally:
    isolation_manager.cleanup_session("temp_session")
```

### 3. Use Resource Locks for Coordination

```python
# Acquire lock before accessing shared resource
if isolation_manager.acquire_resource_lock(session_id, resource_id):
    try:
        # Access resource safely
        pass
    finally:
        isolation_manager.release_resource_lock(session_id, resource_id)
```

### 4. Validate Resource Access

```python
# Check before accessing resources
if isolation_manager.validate_resource_access(
    session_id, "compute", "gpu_cluster"
):
    # Access the resource
    pass
else:
    # Handle access denied
    pass
```

## Monitoring and Statistics

Monitor the isolation system in real-time:

```python
stats = isolation_manager.get_isolation_stats()
print(f"""
Isolation System Status:
  Active sessions: {stats['active_sessions']}
  Tenant boundaries: {stats['tenant_boundaries']}
  Resource locks: {stats['resource_locks']}
  
Features:
  Session isolation: {stats['isolation_features']['session_isolation']}
  Tenant isolation: {stats['isolation_features']['tenant_isolation']}
  Resource isolation: {stats['isolation_features']['resource_isolation']}
  Data classification: {stats['isolation_features']['data_classification']}
""")
```

## Testing

AgentNet includes comprehensive security tests:

```bash
# Run security unit tests
pytest tests/test_high_priority_features.py::TestSecurityIsolationEnhancements -v

# Run integration tests
pytest tests/test_security_integration.py -v

# Run all security tests
pytest tests/test_high_priority_features.py::TestSecurityIsolationEnhancements tests/test_security_integration.py -v
```

All 11 security tests pass successfully.

## Demo

Run the comprehensive demo to see all features in action:

```bash
cd /path/to/agentnet
PYTHONPATH=. python demos/security_isolation_demo.py
```

## API Reference

### SecurityIsolationManager

#### `create_isolated_session(user, session_id, isolation_level="standard")`

Create an isolated session with security boundaries.

**Parameters:**
- `user` (User): User object from RBAC manager
- `session_id` (str): Unique session identifier
- `isolation_level` (str): "basic", "standard", or "strict"

**Returns:**
- `dict`: Session context with isolation parameters

#### `validate_resource_access(session_id, resource_type, resource_id)`

Validate if a session can access a resource.

**Parameters:**
- `session_id` (str): Session identifier
- `resource_type` (str): Type of resource (compute, network, file_system)
- `resource_id` (str): Resource identifier

**Returns:**
- `bool`: True if access is allowed

#### `acquire_resource_lock(session_id, resource_id)`

Acquire exclusive lock on a resource.

**Parameters:**
- `session_id` (str): Session identifier
- `resource_id` (str): Resource identifier

**Returns:**
- `bool`: True if lock acquired successfully

#### `release_resource_lock(session_id, resource_id)`

Release a resource lock.

**Parameters:**
- `session_id` (str): Session identifier
- `resource_id` (str): Resource identifier

**Returns:**
- `bool`: True if lock released successfully

#### `cleanup_session(session_id)`

Clean up session and release all resources.

**Parameters:**
- `session_id` (str): Session identifier

**Returns:**
- `None`

#### `get_isolation_stats()`

Get current isolation system statistics.

**Returns:**
- `dict`: Statistics including active sessions, tenant boundaries, resource locks

## Security Considerations

1. **Secret Management**: Use proper secret management for JWT secret keys
2. **Token Expiry**: Tokens expire after 24 hours by default
3. **Resource Limits**: Enforce resource limits at the infrastructure level
4. **Audit Logging**: Enable audit logging for all security-sensitive operations
5. **Network Segmentation**: Use network policies with infrastructure-level enforcement
6. **Data Classification**: Classify data and enforce access controls consistently

## Troubleshooting

### Session Not Found

```python
# Ensure session exists before validation
if session_id in isolation_manager._active_sessions:
    # Perform operations
    pass
else:
    # Session doesn't exist or was cleaned up
    pass
```

### Lock Already Held

```python
# Check lock status before acquiring
if resource_id not in isolation_manager._resource_locks:
    # Resource is available
    isolation_manager.acquire_resource_lock(session_id, resource_id)
else:
    # Resource is locked by another session
    pass
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/V1B3hR/agentnet/issues
- Documentation: See `docs/security_governance_workflows.md`
- Tests: See `tests/test_security_integration.py` for examples

## License

See LICENSE file in the repository root.
