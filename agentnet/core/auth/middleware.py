"""Authentication middleware for AgentNet APIs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

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
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
            "iat": datetime.utcnow(),
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
