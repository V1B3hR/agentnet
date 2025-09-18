"""Role-Based Access Control system for AgentNet."""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("agentnet.auth")

# Context variables for current user and tenant
current_user_context: contextvars.ContextVar[Optional["User"]] = contextvars.ContextVar(
    "current_user", default=None
)
current_tenant_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_tenant", default=None
)


class Permission(Enum):
    """Available permissions in the system."""

    # Agent operations
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"

    # Session operations
    SESSION_CREATE = "session:create"
    SESSION_READ = "session:read"
    SESSION_UPDATE = "session:update"
    SESSION_DELETE = "session:delete"
    SESSION_LIST = "session:list"

    # Policy operations
    POLICY_CREATE = "policy:create"
    POLICY_READ = "policy:read"
    POLICY_UPDATE = "policy:update"
    POLICY_DELETE = "policy:delete"

    # Cost and monitoring
    COST_READ = "cost:read"
    COST_ADMIN = "cost:admin"
    MONITOR_READ = "monitor:read"
    MONITOR_ADMIN = "monitor:admin"

    # Evaluation operations
    EVAL_CREATE = "eval:create"
    EVAL_READ = "eval:read"
    EVAL_EXECUTE = "eval:execute"

    # DAG operations
    DAG_CREATE = "dag:create"
    DAG_READ = "dag:read"
    DAG_EXECUTE = "dag:execute"

    # Tool operations
    TOOL_CREATE = "tool:create"
    TOOL_READ = "tool:read"
    TOOL_EXECUTE = "tool:execute"

    # Administrative operations
    USER_ADMIN = "user:admin"
    TENANT_ADMIN = "tenant:admin"
    SYSTEM_ADMIN = "system:admin"

    # Audit operations
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


class Role(Enum):
    """Predefined roles with associated permissions."""

    ADMIN = "admin"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    TENANT_USER = "tenant_user"


@dataclass
class User:
    """User representation with roles and permissions."""

    user_id: str
    username: str
    email: str
    roles: Set[Role] = field(default_factory=set)
    tenant_id: Optional[str] = None
    custom_permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class RBACManager:
    """Manages roles, permissions, and access control."""

    def __init__(self):
        self.role_permissions = self._setup_default_role_permissions()
        self.users: Dict[str, User] = {}
        logger.info("RBACManager initialized")

    def _setup_default_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Setup default permissions for each role."""
        return {
            Role.ADMIN: {
                # Full system access
                Permission.AGENT_CREATE,
                Permission.AGENT_READ,
                Permission.AGENT_UPDATE,
                Permission.AGENT_DELETE,
                Permission.AGENT_EXECUTE,
                Permission.SESSION_CREATE,
                Permission.SESSION_READ,
                Permission.SESSION_UPDATE,
                Permission.SESSION_DELETE,
                Permission.SESSION_LIST,
                Permission.POLICY_CREATE,
                Permission.POLICY_READ,
                Permission.POLICY_UPDATE,
                Permission.POLICY_DELETE,
                Permission.COST_READ,
                Permission.COST_ADMIN,
                Permission.MONITOR_READ,
                Permission.MONITOR_ADMIN,
                Permission.EVAL_CREATE,
                Permission.EVAL_READ,
                Permission.EVAL_EXECUTE,
                Permission.DAG_CREATE,
                Permission.DAG_READ,
                Permission.DAG_EXECUTE,
                Permission.TOOL_CREATE,
                Permission.TOOL_READ,
                Permission.TOOL_EXECUTE,
                Permission.USER_ADMIN,
                Permission.TENANT_ADMIN,
                Permission.SYSTEM_ADMIN,
                Permission.AUDIT_READ,
                Permission.AUDIT_EXPORT,
            },
            Role.OPERATOR: {
                # Operations and execution
                Permission.AGENT_READ,
                Permission.AGENT_EXECUTE,
                Permission.SESSION_CREATE,
                Permission.SESSION_READ,
                Permission.SESSION_UPDATE,
                Permission.SESSION_LIST,
                Permission.POLICY_READ,
                Permission.COST_READ,
                Permission.MONITOR_READ,
                Permission.EVAL_READ,
                Permission.EVAL_EXECUTE,
                Permission.DAG_READ,
                Permission.DAG_EXECUTE,
                Permission.TOOL_READ,
                Permission.TOOL_EXECUTE,
            },
            Role.AUDITOR: {
                # Read-only access for auditing
                Permission.AGENT_READ,
                Permission.SESSION_READ,
                Permission.SESSION_LIST,
                Permission.POLICY_READ,
                Permission.COST_READ,
                Permission.MONITOR_READ,
                Permission.EVAL_READ,
                Permission.DAG_READ,
                Permission.TOOL_READ,
                Permission.AUDIT_READ,
                Permission.AUDIT_EXPORT,
            },
            Role.TENANT_USER: {
                # Limited access within tenant
                Permission.AGENT_READ,
                Permission.AGENT_EXECUTE,
                Permission.SESSION_CREATE,
                Permission.SESSION_READ,
                Permission.SESSION_UPDATE,
                Permission.COST_READ,
                Permission.MONITOR_READ,
                Permission.EVAL_READ,
                Permission.EVAL_EXECUTE,
                Permission.DAG_READ,
                Permission.DAG_EXECUTE,
                Permission.TOOL_READ,
                Permission.TOOL_EXECUTE,
            },
        }

    def create_user(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[Role],
        tenant_id: Optional[str] = None,
        custom_permissions: Optional[List[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Create a new user."""
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=set(roles),
            tenant_id=tenant_id,
            custom_permissions=set(custom_permissions or []),
            metadata=metadata or {},
        )

        self.users[user_id] = user
        logger.info(f"Created user {username} ({user_id}) with roles {roles}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user."""
        permissions = set(user.custom_permissions)

        for role in user.roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])

        return permissions

    def user_has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        if not user.is_active:
            return False

        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions

    def user_can_access_tenant(self, user: User, tenant_id: str) -> bool:
        """Check if user can access a specific tenant."""
        # Admin users can access any tenant
        if Role.ADMIN in user.roles:
            return True

        # Users can only access their own tenant
        return user.tenant_id == tenant_id

    def add_role_to_user(self, user_id: str, role: Role):
        """Add a role to a user."""
        if user_id in self.users:
            self.users[user_id].roles.add(role)
            logger.info(f"Added role {role.value} to user {user_id}")

    def remove_role_from_user(self, user_id: str, role: Role):
        """Remove a role from a user."""
        if user_id in self.users:
            self.users[user_id].roles.discard(role)
            logger.info(f"Removed role {role.value} from user {user_id}")

    def update_role_permissions(self, role: Role, permissions: Set[Permission]):
        """Update permissions for a role."""
        self.role_permissions[role] = permissions
        logger.info(f"Updated permissions for role {role.value}")

    def list_users_by_role(self, role: Role) -> List[User]:
        """List all users with a specific role."""
        return [user for user in self.users.values() if role in user.roles]

    def list_users_by_tenant(self, tenant_id: str) -> List[User]:
        """List all users in a specific tenant."""
        return [user for user in self.users.values() if user.tenant_id == tenant_id]


def require_permission(permission: Permission, tenant_check: bool = True):
    """Decorator to require a specific permission for a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = current_user_context.get()
            if not user:
                raise PermissionError("Authentication required")

            # Get RBAC manager from kwargs or create default
            rbac_manager = kwargs.get("rbac_manager")
            if not rbac_manager:
                # In production, this would come from dependency injection
                rbac_manager = RBACManager()

            # Check permission
            if not rbac_manager.user_has_permission(user, permission):
                raise PermissionError(
                    f"Insufficient permissions: {permission.value} required"
                )

            # Check tenant access if required
            if tenant_check:
                tenant_id = current_tenant_context.get()
                if tenant_id and not rbac_manager.user_can_access_tenant(
                    user, tenant_id
                ):
                    raise PermissionError(f"Access denied to tenant {tenant_id}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_current_user() -> Optional[User]:
    """Get the current user from context."""
    return current_user_context.get()


def get_current_tenant() -> Optional[str]:
    """Get the current tenant from context."""
    return current_tenant_context.get()


def set_current_user(user: User):
    """Set the current user in context."""
    current_user_context.set(user)


def set_current_tenant(tenant_id: str):
    """Set the current tenant in context."""
    current_tenant_context.set(tenant_id)
