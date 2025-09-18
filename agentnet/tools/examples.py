"""Example tool implementations."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolSpec, ToolStatus


class WebSearchTool(Tool):
    """Example web search tool (mock implementation)."""

    def __init__(self):
        spec = ToolSpec(
            name="web_search",
            description="Search the web for information",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            rate_limit_per_min=30,
            auth_required=True,
            auth_scope="web_search",
            timeout_seconds=10.0,
            cached=True,
            tags=["search", "web", "information"],
        )
        super().__init__(spec)

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        query = parameters["query"]
        max_results = parameters.get("max_results", 5)

        # Simulate web search delay
        await asyncio.sleep(0.1)

        # Mock search results
        mock_results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a mock search result {i+1} containing information about {query}.",
            }
            for i in range(min(max_results, 5))
        ]

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "query": query,
                "results": mock_results,
                "total_results": len(mock_results),
            },
            metadata={"source": "mock_search_engine", "cached": True},
        )


class CalculatorTool(Tool):
    """Example calculator tool."""

    def __init__(self):
        spec = ToolSpec(
            name="calculator",
            description="Perform mathematical calculations",
            schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            rate_limit_per_min=100,
            auth_required=False,
            timeout_seconds=5.0,
            cached=True,
            tags=["math", "calculation", "utility"],
        )
        super().__init__(spec)

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        expression = parameters["expression"]

        try:
            # Simple and safe expression evaluation
            # In production, would use a proper math parser
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Expression contains invalid characters")

            # Evaluate safely (very basic)
            result = eval(expression)

            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"expression": expression, "result": result},
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR, error_message=f"Calculation error: {str(e)}"
            )


class FileWriteTool(Tool):
    """Example file writing tool."""

    def __init__(self):
        spec = ToolSpec(
            name="file_write",
            description="Write content to a file",
            schema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                    "append": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to append to existing file",
                    },
                },
                "required": ["filename", "content"],
            },
            rate_limit_per_min=20,
            auth_required=True,
            auth_scope="file_system",
            timeout_seconds=15.0,
            cached=False,
            tags=["file", "write", "storage"],
        )
        super().__init__(spec)

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        filename = parameters["filename"]
        content = parameters["content"]
        append_mode = parameters.get("append", False)

        try:
            # For demo purposes, write to a demo directory
            from pathlib import Path

            demo_dir = Path("demo_output")
            demo_dir.mkdir(exist_ok=True)

            filepath = demo_dir / filename
            mode = "a" if append_mode else "w"

            with open(filepath, mode) as f:
                f.write(content)

            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "filename": filename,
                    "filepath": str(filepath),
                    "bytes_written": len(content.encode("utf-8")),
                    "append_mode": append_mode,
                },
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR, error_message=f"File write error: {str(e)}"
            )


class StatusCheckTool(Tool):
    """Example system status check tool."""

    def __init__(self):
        spec = ToolSpec(
            name="status_check",
            description="Check system status and health",
            schema={
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": ["memory", "disk", "network", "overall"],
                        "default": "overall",
                        "description": "System component to check",
                    }
                },
            },
            rate_limit_per_min=60,
            auth_required=False,
            timeout_seconds=5.0,
            cached=False,
            tags=["system", "monitoring", "health"],
        )
        super().__init__(spec)

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        component = parameters.get("component", "overall")

        # Mock status check
        status_data = {
            "timestamp": time.time(),
            "component": component,
            "status": "healthy",
            "details": {},
        }

        if component == "memory":
            status_data["details"] = {"usage_percent": 45.2, "available_gb": 8.7}
        elif component == "disk":
            status_data["details"] = {"usage_percent": 67.8, "free_gb": 245.3}
        elif component == "network":
            status_data["details"] = {"latency_ms": 12.4, "bandwidth_mbps": 100.0}
        else:  # overall
            status_data["details"] = {
                "uptime_hours": 72.5,
                "load_average": 0.8,
                "active_processes": 157,
            }

        return ToolResult(status=ToolStatus.SUCCESS, data=status_data)
