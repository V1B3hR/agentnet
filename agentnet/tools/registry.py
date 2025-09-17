"""Tool registry for managing available tools."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import Tool, ToolSpec, ToolError


class ToolRegistry:
    """Registry for managing and discovering available tools."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self._tools: Dict[str, Tool] = {}
        self._specs: Dict[str, ToolSpec] = {}
        self._tags: Dict[str, Set[str]] = {}  # tag -> set of tool names
        
        self.config_path = config_path or Path("configs/tools.json")
        
        # Load tools from config if exists
        if self.config_path.exists():
            self._load_from_config()
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool instance."""
        tool_name = tool.name
        
        if tool_name in self._tools:
            raise ToolError(f"Tool '{tool_name}' is already registered", tool_name, "DUPLICATE_TOOL")
        
        self._tools[tool_name] = tool
        self._specs[tool_name] = tool.spec
        
        # Index by tags
        if tool.spec.tags:
            for tag in tool.spec.tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(tool_name)
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name not in self._tools:
            return False
        
        # Remove from main registry
        tool = self._tools.pop(tool_name)
        self._specs.pop(tool_name)
        
        # Remove from tag index
        if tool.spec.tags:
            for tag in tool.spec.tags:
                if tag in self._tags:
                    self._tags[tag].discard(tool_name)
                    if not self._tags[tag]:
                        del self._tags[tag]
        
        return True
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool instance by name."""
        return self._tools.get(tool_name)
    
    def get_tool_spec(self, tool_name: str) -> Optional[ToolSpec]:
        """Get tool specification by name."""
        return self._specs.get(tool_name)
    
    def list_tools(self, tag: Optional[str] = None) -> List[str]:
        """List available tool names, optionally filtered by tag."""
        if tag is None:
            return list(self._tools.keys())
        
        return list(self._tags.get(tag, set()))
    
    def list_tool_specs(self, tag: Optional[str] = None) -> List[ToolSpec]:
        """List tool specifications, optionally filtered by tag."""
        tool_names = self.list_tools(tag)
        return [self._specs[name] for name in tool_names if name in self._specs]
    
    def search_tools(self, query: str, tags: Optional[List[str]] = None) -> List[ToolSpec]:
        """Search tools by name or description."""
        query_lower = query.lower()
        results = []
        
        for spec in self._specs.values():
            # Check if query matches name or description
            name_match = query_lower in spec.name.lower()
            desc_match = query_lower in spec.description.lower()
            
            if not (name_match or desc_match):
                continue
            
            # Check tag filter
            if tags:
                spec_tags = set(spec.tags or [])
                query_tags = set(tags)
                if not query_tags.intersection(spec_tags):
                    continue
            
            results.append(spec)
        
        return results
    
    def get_tools_by_tag(self, tag: str) -> List[Tool]:
        """Get all tools with a specific tag."""
        tool_names = self._tags.get(tag, set())
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def list_tags(self) -> List[str]:
        """List all available tags."""
        return list(self._tags.keys())
    
    def export_specs(self) -> Dict[str, Any]:
        """Export all tool specifications as dictionary."""
        return {
            "tools": [spec.to_dict() for spec in self._specs.values()],
            "tags": {tag: list(tools) for tag, tools in self._tags.items()}
        }
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Save tool specifications to config file."""
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.export_specs()
        
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _load_from_config(self) -> None:
        """Load tool specifications from config file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load tool specs (but not actual tool instances)
            for tool_data in config_data.get("tools", []):
                spec = ToolSpec(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    schema=tool_data["schema"],
                    rate_limit_per_min=tool_data.get("rate_limit_per_min"),
                    auth_required=tool_data.get("auth_required", False),
                    auth_scope=tool_data.get("auth_scope"),
                    timeout_seconds=tool_data.get("timeout_seconds", 30.0),
                    cached=tool_data.get("cached", False),
                    tags=tool_data.get("tags", [])
                )
                
                self._specs[spec.name] = spec
                
                # Index by tags
                if spec.tags:
                    for tag in spec.tags:
                        if tag not in self._tags:
                            self._tags[tag] = set()
                        self._tags[tag].add(spec.name)
        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Config file doesn't exist or is invalid - start with empty registry
            pass
    
    def validate_tool_spec(self, spec: ToolSpec) -> List[str]:
        """Validate tool specification and return list of errors."""
        errors = []
        
        if not spec.name:
            errors.append("Tool name is required")
        
        if not spec.description:
            errors.append("Tool description is required")
        
        if not spec.schema:
            errors.append("Tool schema is required")
        elif not isinstance(spec.schema, dict):
            errors.append("Tool schema must be a dictionary")
        
        if spec.rate_limit_per_min is not None and spec.rate_limit_per_min <= 0:
            errors.append("Rate limit must be positive")
        
        if spec.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        return errors
    
    @property
    def tool_count(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    @property
    def spec_count(self) -> int:
        """Get number of tool specifications."""
        return len(self._specs)