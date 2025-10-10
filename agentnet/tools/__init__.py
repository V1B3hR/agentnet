"""Tool system for AgentNet.

Provides tool registry, execution framework with JSON schema validation,
rate limiting, and authentication scoping.
"""

from .base import Tool, ToolError, ToolResult, ToolSpec
from .executor import ToolExecutor
from .rate_limiter import RateLimiter
from .registry import ToolRegistry

# Import governance features
try:
    from .governance import (
        ToolGovernanceManager,
        ApprovalRequest,
        ApprovalStatus,
        ToolRiskLevel,
        ToolGovernancePolicy,
    )
    _governance_available = True
except ImportError:
    ToolGovernanceManager = None
    ApprovalRequest = None
    ApprovalStatus = None
    ToolRiskLevel = None
    ToolGovernancePolicy = None
    _governance_available = False

__all__ = [
    "Tool",
    "ToolResult",
    "ToolError",
    "ToolSpec",
    "ToolRegistry",
    "ToolExecutor",
    "RateLimiter",
    "ToolGovernanceManager",
    "ApprovalRequest",
    "ApprovalStatus",
    "ToolRiskLevel",
    "ToolGovernancePolicy",
]
