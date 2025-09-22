"""
Enhanced monitors configuration for AgentNet.

This file defines monitor specifications that can be loaded
by the MonitorManager for enhanced agent configurations.
"""

from agentnet.monitors.base import MonitorSpec
from agentnet.core.types import Severity

# Enhanced monitor specifications
MONITORS = [
    MonitorSpec(
        name="resource_guard",
        type="resource",
        params={
            "max_memory_mb": 512,
            "max_cpu_percent": 80,
            "max_runtime_seconds": 120
        },
        severity=Severity.MINOR,
        description="Monitor system resource usage and performance"
    ),
    
    MonitorSpec(
        name="content_safety",
        type="keyword",
        params={
            "keywords": ["unsafe", "harmful", "dangerous", "malicious"],
            "case_sensitive": False
        },
        severity=Severity.MAJOR,
        description="Monitor for potentially unsafe content"
    ),
    
    MonitorSpec(
        name="quality_check",
        type="custom",
        params={
            "min_confidence": 0.7,
            "max_response_length": 2000
        },
        severity=Severity.MINOR,
        description="Monitor response quality and length"
    )
]

def get_monitors():
    """Return the list of monitor specifications."""
    return MONITORS