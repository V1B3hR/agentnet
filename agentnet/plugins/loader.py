"""Plugin Loader Implementation for P6 Enterprise Hardening.

This module provides plugin discovery, loading, and dependency management
capabilities for the AgentNet plugin system.
"""

import hashlib
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .framework import Plugin, PluginInfo, PluginStatus
from .security import PluginSandbox, SecurityPolicy

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """Discovers and validates plugins in various locations."""

    def __init__(self, search_paths: Optional[List[str]] = None):
        self.search_paths = [
            Path(p) for p in (search_paths or ["./plugins", "~/.agentnet/plugins"])
        ]
        self._discovered_plugins = {}

    def scan_for_plugins(self) -> Dict[str, PluginInfo]:
        """Scan all search paths for valid plugins."""
        discovered = {}

        for search_path in self.search_paths:
            try:
                expanded_path = Path(search_path).expanduser()
                if expanded_path.exists() and expanded_path.is_dir():
                    plugins = self._scan_directory(expanded_path)
                    discovered.update(plugins)
            except Exception as e:
                logger.error(f"Error scanning plugin directory {search_path}: {e}")

        self._discovered_plugins = discovered
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def _scan_directory(self, directory: Path) -> Dict[str, PluginInfo]:
        """Scan a directory for plugins."""
        plugins = {}

        for item in directory.iterdir():
            if item.is_dir():
                plugin_info = self._validate_plugin_directory(item)
                if plugin_info:
                    plugins[plugin_info.name] = plugin_info

        return plugins

    def _validate_plugin_directory(self, plugin_dir: Path) -> Optional[PluginInfo]:
        """Validate a plugin directory and extract plugin info."""
        manifest_file = plugin_dir / "plugin.json"

        if not manifest_file.exists():
            return None

        try:
            with open(manifest_file, "r") as f:
                manifest_data = json.load(f)

            # Validate required fields
            required_fields = ["name", "version"]
            for field in required_fields:
                if field not in manifest_data:
                    logger.error(
                        f"Plugin manifest missing required field '{field}': {manifest_file}"
                    )
                    return None

            # Create plugin info
            plugin_info = PluginInfo(
                name=manifest_data["name"],
                version=manifest_data["version"],
                description=manifest_data.get("description", ""),
                author=manifest_data.get("author", ""),
                license=manifest_data.get("license", ""),
                homepage=manifest_data.get("homepage", ""),
                dependencies=manifest_data.get("dependencies", []),
                agentnet_version=manifest_data.get("agentnet_version", ">=0.4.0"),
                plugin_type=manifest_data.get("plugin_type", "general"),
                entry_point=manifest_data.get("entry_point", "main.py"),
                config_schema=manifest_data.get("config_schema", {}),
                permissions=manifest_data.get("permissions", []),
                sandboxed=manifest_data.get("sandboxed", True),
            )

            # Validate entry point exists
            entry_point_file = plugin_dir / plugin_info.entry_point
            if not entry_point_file.exists():
                logger.error(f"Plugin entry point not found: {entry_point_file}")
                return None

            # Additional validation
            if not self._validate_plugin_structure(plugin_dir, plugin_info):
                return None

            return plugin_info

        except Exception as e:
            logger.error(f"Error validating plugin directory {plugin_dir}: {e}")
            return None

    def _validate_plugin_structure(
        self, plugin_dir: Path, plugin_info: PluginInfo
    ) -> bool:
        """Validate plugin directory structure."""
        entry_point = plugin_dir / plugin_info.entry_point

        # Check if entry point is valid Python file
        if not entry_point.suffix == ".py":
            logger.error(f"Plugin entry point must be a Python file: {entry_point}")
            return False

        # Check for optional files
        optional_files = ["README.md", "requirements.txt", "config.yaml"]
        for optional_file in optional_files:
            file_path = plugin_dir / optional_file
            if file_path.exists():
                logger.debug(f"Found optional file: {file_path}")

        return True

    def get_plugin_checksum(self, plugin_name: str) -> Optional[str]:
        """Calculate checksum for plugin validation."""
        if plugin_name not in self._discovered_plugins:
            return None

        plugin_info = self._discovered_plugins[plugin_name]
        plugin_dir = None

        # Find plugin directory
        for search_path in self.search_paths:
            expanded_path = Path(search_path).expanduser()
            potential_dir = expanded_path / plugin_name
            if potential_dir.exists():
                plugin_dir = potential_dir
                break

        if not plugin_dir:
            return None

        # Calculate checksum of main files
        hasher = hashlib.sha256()

        files_to_hash = [
            plugin_dir / "plugin.json",
            plugin_dir / plugin_info.entry_point,
        ]

        for file_path in files_to_hash:
            if file_path.exists():
                with open(file_path, "rb") as f:
                    hasher.update(f.read())

        return hasher.hexdigest()


class PluginLoader:
    """Loads and manages plugin lifecycle."""

    def __init__(self, security_policy: Optional[SecurityPolicy] = None):
        self.security_policy = security_policy
        self.sandbox = PluginSandbox(security_policy) if security_policy else None
        self._loaded_modules = {}
        self._plugin_checksums = {}

    def load_plugin_from_path(
        self, plugin_path: Path, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """Load a plugin from a specific path."""
        try:
            # Validate plugin directory
            discovery = PluginDiscovery()
            plugin_info = discovery._validate_plugin_directory(plugin_path)

            if not plugin_info:
                logger.error(f"Invalid plugin directory: {plugin_path}")
                return None

            # Check security policy
            if self.security_policy and not self.security_policy.can_load_plugin(
                plugin_info
            ):
                logger.error(
                    f"Security policy denies loading plugin: {plugin_info.name}"
                )
                return None

            # Calculate and store checksum
            discovery._discovered_plugins[plugin_info.name] = plugin_info
            checksum = discovery.get_plugin_checksum(plugin_info.name)
            self._plugin_checksums[plugin_info.name] = checksum

            # Load plugin module
            plugin_instance = self._load_plugin_module(plugin_path, plugin_info, config)

            if plugin_instance:
                logger.info(
                    f"Successfully loaded plugin: {plugin_info.name} v{plugin_info.version}"
                )

            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None

    def _load_plugin_module(
        self,
        plugin_path: Path,
        plugin_info: PluginInfo,
        config: Optional[Dict[str, Any]],
    ) -> Optional[Plugin]:
        """Load the plugin module and create instance."""
        entry_point = plugin_path / plugin_info.entry_point
        module_name = f"agentnet_plugin_{plugin_info.name}"

        # Check if already loaded
        if module_name in self._loaded_modules:
            logger.warning(f"Plugin module {module_name} already loaded, reloading")
            del self._loaded_modules[module_name]

        try:
            # Load module spec
            spec = importlib.util.spec_from_file_location(module_name, entry_point)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load module spec for {entry_point}")

            # Create module
            module = importlib.util.module_from_spec(spec)

            # Validate imports if security policy exists
            if self.security_policy:
                self._validate_plugin_code(entry_point, plugin_info)

            # Execute module
            spec.loader.exec_module(module)
            self._loaded_modules[module_name] = module

            # Find Plugin class
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                raise ValueError(f"No Plugin class found in {entry_point}")

            # Create plugin instance
            plugin_instance = plugin_class(config)
            plugin_instance.status = PluginStatus.LOADED

            return plugin_instance

        except Exception as e:
            logger.error(f"Error loading plugin module {module_name}: {e}")
            return None

    def _find_plugin_class(self, module) -> Optional[type]:
        """Find the Plugin class in a module."""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                return obj
        return None

    def _validate_plugin_code(self, entry_point: Path, plugin_info: PluginInfo) -> None:
        """Validate plugin code against security policy."""
        if not self.sandbox:
            return

        with open(entry_point, "r") as f:
            plugin_code = f.read()

        restrictions = self.security_policy.get_execution_restrictions(plugin_info)
        violations = self.sandbox.validate_plugin_imports(plugin_code, restrictions)

        if violations:
            raise SecurityError(f"Plugin security violations: {', '.join(violations)}")

    def unload_plugin_module(self, plugin_name: str) -> bool:
        """Unload a plugin module."""
        module_name = f"agentnet_plugin_{plugin_name}"

        if module_name in self._loaded_modules:
            # Remove from loaded modules
            del self._loaded_modules[module_name]

            # Remove from sys.modules if present
            if module_name in sys.modules:
                del sys.modules[module_name]

            logger.info(f"Unloaded plugin module: {module_name}")
            return True

        return False

    def reload_plugin(
        self, plugin_path: Path, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """Reload a plugin (unload and load again)."""
        # Extract plugin name from path
        discovery = PluginDiscovery()
        plugin_info = discovery._validate_plugin_directory(plugin_path)

        if not plugin_info:
            return None

        # Unload existing module
        self.unload_plugin_module(plugin_info.name)

        # Load again
        return self.load_plugin_from_path(plugin_path, config)

    def verify_plugin_integrity(self, plugin_name: str) -> bool:
        """Verify plugin integrity using checksum."""
        if plugin_name not in self._plugin_checksums:
            logger.warning(f"No checksum available for plugin: {plugin_name}")
            return False

        # Recalculate checksum
        discovery = PluginDiscovery()
        current_checksum = discovery.get_plugin_checksum(plugin_name)
        stored_checksum = self._plugin_checksums[plugin_name]

        if current_checksum != stored_checksum:
            logger.error(f"Plugin integrity check failed for {plugin_name}")
            return False

        return True

    def get_loader_status(self) -> Dict[str, Any]:
        """Get plugin loader status information."""
        return {
            "loaded_modules": list(self._loaded_modules.keys()),
            "plugin_checksums": len(self._plugin_checksums),
            "security_policy_active": self.security_policy is not None,
            "sandbox_available": self.sandbox is not None,
        }


class SecurityError(Exception):
    """Exception raised for plugin security violations."""

    pass
