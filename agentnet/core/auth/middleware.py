"""Authentication middleware for AgentNet APIs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set

import jwt
from .rbac import RBACManager, Role, User, set_current_tenant, set_current_user

logger = logging.getLogger("agentnet.auth")


class AuthMiddleware:
    """Authentication and authorization middleware."""

    def __init__(
        self, rbac_manager: RBACManager, secret_key: str = "default-secret-key"
    ):
        self.rbac_manager = rbac_manager
        self.secret_key = secret_key
        logger.info("AuthMiddleware initialized")

    def create_token(self, user: User, expires_hours: int = 24) -> str:
        """Create a JWT token for a user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "roles": [role.value for role in user.roles],
            "exp": datetime.now(timezone.utc) + timedelta(hours=expires_hours),
            "iat": datetime.now(timezone.utc),
        }

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        logger.info(f"Created token for user {user.username}")
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def authenticate_request(
        self, authorization_header: Optional[str]
    ) -> Optional[User]:
        """Authenticate a request using Authorization header."""
        if not authorization_header:
            return None

        # Expected format: "Bearer <token>"
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning("Invalid authorization header format")
            return None

        token = parts[1]
        payload = self.verify_token(token)
        if not payload:
            return None

        # Get user from RBAC manager
        user = self.rbac_manager.get_user(payload["user_id"])
        if not user or not user.is_active:
            logger.warning(f"User {payload['user_id']} not found or inactive")
            return None

        # Set context
        set_current_user(user)
        if user.tenant_id:
            set_current_tenant(user.tenant_id)

        return user

    def create_demo_users(self):
        """Create demo users for testing purposes."""
        # Admin user
        admin_user = self.rbac_manager.create_user(
            user_id="admin-001",
            username="admin",
            email="admin@agentnet.example",
            roles=[Role.ADMIN],
            metadata={"created_by": "demo_setup"},
        )

        # Operator user
        operator_user = self.rbac_manager.create_user(
            user_id="op-001",
            username="operator",
            email="operator@agentnet.example",
            roles=[Role.OPERATOR],
            tenant_id="tenant-001",
            metadata={"created_by": "demo_setup"},
        )

        # Auditor user
        auditor_user = self.rbac_manager.create_user(
            user_id="audit-001",
            username="auditor",
            email="auditor@agentnet.example",
            roles=[Role.AUDITOR],
            metadata={"created_by": "demo_setup"},
        )

        # Tenant user
        tenant_user = self.rbac_manager.create_user(
            user_id="user-001",
            username="tenant_user",
            email="user@tenant001.example",
            roles=[Role.TENANT_USER],
            tenant_id="tenant-001",
            metadata={"created_by": "demo_setup"},
        )

        logger.info("Created demo users: admin, operator, auditor, tenant_user")
        return {
            "admin": admin_user,
            "operator": operator_user,
            "auditor": auditor_user,
            "tenant_user": tenant_user,
        }

    def create_demo_tokens(self) -> Dict[str, str]:
        """Create demo tokens for testing."""
        demo_users = self.create_demo_users()
        tokens = {}

        for role_name, user in demo_users.items():
            token = self.create_token(user)
            tokens[role_name] = token

        return tokens


def get_current_user() -> Optional[User]:
    """Get the current authenticated user."""
    from .rbac import get_current_user

    return get_current_user()


def get_current_tenant() -> Optional[str]:
    """Get the current tenant ID."""
    from .rbac import get_current_tenant

    return get_current_tenant()


class SecurityIsolationManager:
    """Enhanced security isolation mechanisms for AgentNet."""

    def __init__(self):
        self.session_isolation = True
        self.tenant_isolation = True
        self.resource_isolation = True
        self.data_classification_enabled = True
        
        # Track active sessions and isolation contexts
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._tenant_boundaries: Dict[str, Set[str]] = {}
        self._resource_locks: Dict[str, str] = {}  # resource_id -> session_id
        
        logger.info("SecurityIsolationManager initialized")

    def create_isolated_session(
        self, 
        user: User, 
        session_id: str,
        isolation_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Create an isolated session with appropriate security boundaries.
        
        Args:
            user: User creating the session
            session_id: Unique session identifier
            isolation_level: Level of isolation (basic, standard, strict)
            
        Returns:
            Session context with isolation parameters
        """
        session_context = {
            "session_id": session_id,
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "isolation_level": isolation_level,
            "created_at": datetime.now(timezone.utc),
            "resource_access": self._determine_resource_access(user, isolation_level),
            "network_policy": self._create_network_policy(user, isolation_level),
            "data_access_policy": self._create_data_access_policy(user),
            "audit_enabled": True
        }
        
        # Apply tenant isolation
        if self.tenant_isolation and user.tenant_id:
            if user.tenant_id not in self._tenant_boundaries:
                self._tenant_boundaries[user.tenant_id] = set()
            self._tenant_boundaries[user.tenant_id].add(session_id)
        
        self._active_sessions[session_id] = session_context
        
        logger.info(f"Created isolated session {session_id} for user {user.username} with {isolation_level} isolation")
        return session_context

    def _determine_resource_access(self, user: User, isolation_level: str) -> Dict[str, Any]:
        """Determine resource access permissions based on user and isolation level."""
        base_access = {
            "compute_quota": 100,  # Basic compute units
            "memory_limit_mb": 512,
            "network_access": "restricted",
            "file_system_access": "sandbox_only",
            "external_api_access": False
        }
        
        if isolation_level == "strict":
            base_access.update({
                "compute_quota": 50,
                "memory_limit_mb": 256,
                "network_access": "none",
                "external_api_access": False
            })
        elif isolation_level == "basic":
            base_access.update({
                "compute_quota": 200,
                "memory_limit_mb": 1024,
                "network_access": "limited",
                "external_api_access": True
            })
        
        # Adjust based on user roles
        if Role.ADMIN in user.roles:
            base_access.update({
                "compute_quota": base_access["compute_quota"] * 5,
                "memory_limit_mb": base_access["memory_limit_mb"] * 4,
                "network_access": "full",
                "file_system_access": "controlled",
                "external_api_access": True
            })
        
        return base_access

    def _create_network_policy(self, user: User, isolation_level: str) -> Dict[str, Any]:
        """Create network access policy for session."""
        policy = {
            "outbound_allowed": False,
            "inbound_allowed": False,
            "allowed_domains": [],
            "blocked_domains": ["*"],
            "rate_limits": {"requests_per_minute": 10}
        }
        
        if isolation_level != "strict":
            policy["allowed_domains"] = [
                "api.openai.com",
                "api.anthropic.com", 
                "agentnet-internal.local"
            ]
            policy["blocked_domains"] = [
                "*.malicious.com",
                "*.phishing.net"
            ]
        
        if Role.ADMIN in user.roles:
            policy.update({
                "outbound_allowed": True,
                "rate_limits": {"requests_per_minute": 100}
            })
        
        return policy

    def _create_data_access_policy(self, user: User) -> Dict[str, Any]:
        """Create data access policy based on user permissions and tenant."""
        policy = {
            "tenant_isolation": True,
            "classification_levels": ["public"],
            "pii_access": False,
            "audit_logs_access": False,
            "cross_tenant_access": False
        }
        
        # Admin users get broader access
        if Role.ADMIN in user.roles:
            policy.update({
                "classification_levels": ["public", "internal", "confidential"],
                "audit_logs_access": True,
                "cross_tenant_access": True
            })
        
        # Auditors get specific access patterns
        if Role.AUDITOR in user.roles:
            policy.update({
                "classification_levels": ["public", "internal"],
                "audit_logs_access": True,
                "pii_access": False  # Auditors should see redacted data
            })
        
        return policy

    def validate_resource_access(
        self, 
        session_id: str, 
        resource_type: str, 
        resource_id: str
    ) -> bool:
        """Validate if session can access a specific resource."""
        if session_id not in self._active_sessions:
            logger.warning(f"Session {session_id} not found for resource access validation")
            return False
        
        session = self._active_sessions[session_id]
        
        # Check resource locks
        if resource_id in self._resource_locks:
            if self._resource_locks[resource_id] != session_id:
                logger.info(f"Resource {resource_id} locked by another session")
                return False
        
        # Validate based on resource type and session policies
        resource_access = session.get("resource_access", {})
        
        if resource_type == "compute" and resource_access.get("compute_quota", 0) <= 0:
            return False
        
        if resource_type == "file_system" and resource_access.get("file_system_access") == "none":
            return False
        
        if resource_type == "network" and resource_access.get("network_access") == "none":
            return False
        
        return True

    def acquire_resource_lock(self, session_id: str, resource_id: str) -> bool:
        """Acquire exclusive lock on a resource for a session."""
        if resource_id in self._resource_locks:
            if self._resource_locks[resource_id] != session_id:
                return False  # Already locked by another session
        
        self._resource_locks[resource_id] = session_id
        logger.info(f"Resource {resource_id} locked by session {session_id}")
        return True

    def release_resource_lock(self, session_id: str, resource_id: str) -> bool:
        """Release resource lock held by session."""
        if resource_id not in self._resource_locks:
            return False
        
        if self._resource_locks[resource_id] != session_id:
            return False  # Can't release lock held by another session
        
        del self._resource_locks[resource_id]
        logger.info(f"Resource {resource_id} lock released by session {session_id}")
        return True

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session and release all associated resources."""
        if session_id not in self._active_sessions:
            return
        
        session = self._active_sessions[session_id]
        
        # Release all resource locks for this session
        locked_resources = [
            resource_id for resource_id, lock_session_id in self._resource_locks.items()
            if lock_session_id == session_id
        ]
        
        for resource_id in locked_resources:
            del self._resource_locks[resource_id]
        
        # Remove from tenant boundaries
        tenant_id = session.get("tenant_id")
        if tenant_id and tenant_id in self._tenant_boundaries:
            self._tenant_boundaries[tenant_id].discard(session_id)
        
        # Remove session
        del self._active_sessions[session_id]
        
        logger.info(f"Session {session_id} cleaned up, released {len(locked_resources)} resource locks")

    def get_isolation_stats(self) -> Dict[str, Any]:
        """Get current isolation system statistics."""
        return {
            "active_sessions": len(self._active_sessions),
            "tenant_boundaries": len(self._tenant_boundaries),
            "resource_locks": len(self._resource_locks),
            "isolation_features": {
                "session_isolation": self.session_isolation,
                "tenant_isolation": self.tenant_isolation,
                "resource_isolation": self.resource_isolation,
                "data_classification": self.data_classification_enabled
            }
        }
