"""Plugin Framework Implementation for P6 Enterprise Hardening.

This module provides the core plugin system for AgentNet including plugin
interfaces, lifecycle management, and registry functionality.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable, Type
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import importlib.util
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """Plugin metadata and information."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    dependencies: List[str] = field(default_factory=list)
    agentnet_version: str = ">=0.4.0"
    plugin_type: str = "general"
    entry_point: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    sandboxed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin info to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "dependencies": self.dependencies,
            "agentnet_version": self.agentnet_version,
            "plugin_type": self.plugin_type,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "permissions": self.permissions,
            "sandboxed": self.sandboxed
        }


class Plugin(ABC):
    """Base class for all AgentNet plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.status = PluginStatus.DISCOVERED
        self.error_message = ""
        self._hooks = {}
        self._event_handlers = {}
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin information."""
        pass
    
    def initialize(self) -> bool:
        """Initialize the plugin. Override in subclasses."""
        try:
            self.status = PluginStatus.INITIALIZED
            logger.info(f"Plugin {self.info.name} initialized")
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Failed to initialize plugin {self.info.name}: {e}")
            return False
    
    def activate(self) -> bool:
        """Activate the plugin. Override in subclasses."""
        try:
            if self.status != PluginStatus.INITIALIZED:
                return False
            
            self.status = PluginStatus.ACTIVE
            logger.info(f"Plugin {self.info.name} activated")
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Failed to activate plugin {self.info.name}: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the plugin. Override in subclasses."""
        try:
            self.status = PluginStatus.INACTIVE
            logger.info(f"Plugin {self.info.name} deactivated")
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Failed to deactivate plugin {self.info.name}: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup plugin resources. Override in subclasses."""
        pass
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook."""
        results = []
        for callback in self._hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {hook_name} callback failed: {e}")
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "name": self.info.name,
            "version": self.info.version,
            "status": self.status.value,
            "error_message": self.error_message,
            "hooks": list(self._hooks.keys()),
            "config": self.config
        }


class PluginRegistry:
    """Registry for managing discovered and loaded plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_info: Dict[str, PluginInfo] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin instance."""
        try:
            info = plugin.info
            
            if info.name in self._plugins:
                logger.warning(f"Plugin {info.name} already registered, replacing")
            
            self._plugins[info.name] = plugin
            self._plugin_info[info.name] = info
            self._dependencies[info.name] = info.dependencies
            
            logger.info(f"Registered plugin: {info.name} v{info.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin."""
        if plugin_name not in self._plugins:
            return False
        
        try:
            plugin = self._plugins[plugin_name]
            plugin.deactivate()
            plugin.cleanup()
            
            del self._plugins[plugin_name]
            del self._plugin_info[plugin_name]
            del self._dependencies[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin instance by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information by name."""
        return self._plugin_info.get(plugin_name)
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[str]:
        """List plugin names, optionally filtered by status."""
        if status_filter is None:
            return list(self._plugins.keys())
        
        return [
            name for name, plugin in self._plugins.items()
            if plugin.status == status_filter
        ]
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        return {
            name: plugin.get_status()
            for name, plugin in self._plugins.items()
        }
    
    def resolve_dependencies(self, plugin_name: str) -> List[str]:
        """Resolve plugin dependencies in load order."""
        visited = set()
        resolved = []
        
        def resolve(name: str):
            if name in visited:
                return
            visited.add(name)
            
            for dependency in self._dependencies.get(name, []):
                if dependency not in self._plugins:
                    raise ValueError(f"Missing dependency: {dependency} for plugin {name}")
                resolve(dependency)
            
            resolved.append(name)
        
        resolve(plugin_name)
        return resolved


class PluginManager:
    """Main plugin manager for AgentNet plugin system."""
    
    def __init__(self, plugin_directory: str = "./plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.plugin_directory.mkdir(exist_ok=True)
        self.registry = PluginRegistry()
        self._global_hooks = {}
        self._security_policy = None
    
    def set_security_policy(self, policy: Any) -> None:
        """Set security policy for plugin operations."""
        self._security_policy = policy
    
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover available plugins in the plugin directory."""
        discovered = []
        
        for plugin_path in self.plugin_directory.iterdir():
            if plugin_path.is_dir():
                manifest_path = plugin_path / "plugin.json"
                if manifest_path.exists():
                    try:
                        info = self._load_plugin_manifest(manifest_path)
                        discovered.append(info)
                    except Exception as e:
                        logger.error(f"Failed to load plugin manifest {manifest_path}: {e}")
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def _load_plugin_manifest(self, manifest_path: Path) -> PluginInfo:
        """Load plugin manifest from JSON file."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        return PluginInfo(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
            homepage=data.get("homepage", ""),
            dependencies=data.get("dependencies", []),
            agentnet_version=data.get("agentnet_version", ">=0.4.0"),
            plugin_type=data.get("plugin_type", "general"),
            entry_point=data.get("entry_point", ""),
            config_schema=data.get("config_schema", {}),
            permissions=data.get("permissions", []),
            sandboxed=data.get("sandboxed", True)
        )
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin by name."""
        try:
            # Find plugin manifest
            plugin_path = self.plugin_directory / plugin_name
            manifest_path = plugin_path / "plugin.json"
            
            if not manifest_path.exists():
                raise FileNotFoundError(f"Plugin manifest not found: {manifest_path}")
            
            info = self._load_plugin_manifest(manifest_path)
            
            # Check security policy
            if self._security_policy and not self._security_policy.can_load_plugin(info):
                raise PermissionError(f"Security policy denies loading plugin: {plugin_name}")
            
            # Load plugin module
            entry_point = plugin_path / (info.entry_point or "main.py")
            if not entry_point.exists():
                raise FileNotFoundError(f"Plugin entry point not found: {entry_point}")
            
            spec = importlib.util.spec_from_file_location(f"plugin_{plugin_name}", entry_point)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                raise ValueError(f"No Plugin class found in {entry_point}")
            
            # Create plugin instance
            plugin = plugin_class(config)
            plugin.status = PluginStatus.LOADED
            
            # Register plugin
            self.registry.register_plugin(plugin)
            
            logger.info(f"Loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def initialize_plugin(self, plugin_name: str) -> bool:
        """Initialize a loaded plugin."""
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        if plugin.status != PluginStatus.LOADED:
            logger.error(f"Plugin {plugin_name} must be loaded before initialization")
            return False
        
        # Check dependencies
        try:
            dependencies = self.registry.resolve_dependencies(plugin_name)
            for dep in dependencies[:-1]:  # Exclude self
                dep_plugin = self.registry.get_plugin(dep)
                if not dep_plugin or dep_plugin.status not in [PluginStatus.ACTIVE, PluginStatus.INITIALIZED]:
                    logger.error(f"Dependency {dep} not available for plugin {plugin_name}")
                    return False
        except Exception as e:
            logger.error(f"Dependency resolution failed for {plugin_name}: {e}")
            return False
        
        return plugin.initialize()
    
    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate an initialized plugin."""
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        return plugin.activate()
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate an active plugin."""
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        return plugin.deactivate()
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin completely."""
        return self.registry.unregister_plugin(plugin_name)
    
    def register_global_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a global hook callback."""
        if hook_name not in self._global_hooks:
            self._global_hooks[hook_name] = []
        self._global_hooks[hook_name].append(callback)
    
    def execute_global_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all global hook callbacks."""
        results = []
        
        # Execute global hooks
        for callback in self._global_hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Global hook {hook_name} callback failed: {e}")
        
        # Execute plugin hooks
        for plugin in self.registry._plugins.values():
            if plugin.status == PluginStatus.ACTIVE:
                plugin_results = plugin.execute_hook(hook_name, *args, **kwargs)
                results.extend(plugin_results)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive plugin system status."""
        plugin_status = self.registry.get_plugin_status()
        
        status_counts = {}
        for status in PluginStatus:
            status_counts[status.value] = sum(
                1 for p in plugin_status.values() 
                if p["status"] == status.value
            )
        
        return {
            "total_plugins": len(plugin_status),
            "status_counts": status_counts,
            "plugins": plugin_status,
            "global_hooks": list(self._global_hooks.keys()),
            "plugin_directory": str(self.plugin_directory)
        }