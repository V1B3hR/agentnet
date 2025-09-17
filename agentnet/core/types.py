"""Core type definitions for AgentNet."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Severity levels for policy violations and cognitive faults."""
    MINOR = "minor"
    MAJOR = "major"
    SEVERE = "severe"


def _parse_severity(value: str | Severity | None) -> Severity:
    """Parse severity value from string or Severity enum."""
    if isinstance(value, Severity):
        return value
    if not value:
        return Severity.MINOR
    v = str(value).lower()
    if v == "severe":
        return Severity.SEVERE
    if v == "major":
        return Severity.MAJOR
    return Severity.MINOR


class CognitiveFault(Exception):
    """Exception raised when agent encounters cognitive processing issues."""
    
    def __init__(
        self,
        message: str,
        severity: Severity = Severity.MINOR,
        violations: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.violations = violations or []
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message": str(self),
            "severity": self.severity.value,
            "violations": self.violations,
            "context": self.context
        }