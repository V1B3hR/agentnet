"""Base tool interfaces and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import jsonschema


class ToolError(Exception):
    """Base exception for tool-related errors."""

    def __init__(self, message: str, tool_name: str = None, error_code: str = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.error_code = error_code


class ToolStatus(str, Enum):
    """Tool execution status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"


@dataclass
class ToolResult:
    """Result of tool execution."""

    status: ToolStatus
    data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolSpec:
    """Tool specification with schema and metadata."""

    name: str
    description: str
    schema: Dict[str, Any]  # JSON schema for parameters
    rate_limit_per_min: Optional[int] = None
    auth_required: bool = False
    auth_scope: Optional[str] = None
    timeout_seconds: float = 30.0
    cached: bool = False
    tags: Optional[list] = None

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters against JSON schema."""
        try:
            jsonschema.validate(parameters, self.schema)
        except jsonschema.ValidationError as e:
            raise ToolError(
                f"Parameter validation failed: {e.message}",
                self.name,
                "VALIDATION_ERROR",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
            "rate_limit_per_min": self.rate_limit_per_min,
            "auth_required": self.auth_required,
            "auth_scope": self.auth_scope,
            "timeout_seconds": self.timeout_seconds,
            "cached": self.cached,
            "tags": self.tags or [],
        }


class Tool(ABC):
    """Abstract base class for tools."""

    def __init__(self, spec: ToolSpec):
        self.spec = spec

    @abstractmethod
    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters using the tool's schema."""
        self.spec.validate_parameters(parameters)

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.spec.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.spec.description
