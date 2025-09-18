"""Plugin Security Implementation for P6 Enterprise Hardening.

This module provides security controls for plugin execution including
sandboxing, permission management, and security policy enforcement.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import subprocess
import tempfile
import os
import sys
from pathlib import Path

from .framework import PluginInfo

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for plugin execution."""
    UNRESTRICTED = "unrestricted"
    SANDBOXED = "sandboxed"
    RESTRICTED = "restricted"
    MINIMAL = "minimal"


@dataclass
class SecurityPolicy:
    """Security policy for plugin management."""
    name: str
    version: str = "1.0"
    default_level: SecurityLevel = SecurityLevel.SANDBOXED
    allowed_permissions: Set[str] = field(default_factory=set)
    blocked_permissions: Set[str] = field(default_factory=set)
    allowed_imports: Set[str] = field(default_factory=set)
    blocked_imports: Set[str] = field(default_factory=set)
    max_memory_mb: int = 512
    max_cpu_time_seconds: int = 60
    network_access: bool = False
    filesystem_access: bool = False
    allowed_directories: List[str] = field(default_factory=list)
    
    def can_load_plugin(self, plugin_info: PluginInfo) -> bool:
        """Check if a plugin can be loaded under this policy."""
        # Check blocked permissions
        for permission in plugin_info.permissions:
            if permission in self.blocked_permissions:
                logger.warning(f"Plugin {plugin_info.name} blocked by permission: {permission}")
                return False
        
        # Check required permissions are allowed
        if self.allowed_permissions:
            for permission in plugin_info.permissions:
                if permission not in self.allowed_permissions:
                    logger.warning(f"Plugin {plugin_info.name} requires unavailable permission: {permission}")
                    return False
        
        # Additional checks can be added here
        # - Code scanning for dangerous patterns
        # - Signature verification
        # - Reputation checks
        
        return True
    
    def get_execution_restrictions(self, plugin_info: PluginInfo) -> Dict[str, Any]:
        """Get execution restrictions for a plugin."""
        level = SecurityLevel.SANDBOXED if plugin_info.sandboxed else self.default_level
        
        restrictions = {
            "security_level": level.value,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_time_seconds": self.max_cpu_time_seconds,
            "network_access": self.network_access,
            "filesystem_access": self.filesystem_access,
            "allowed_directories": self.allowed_directories.copy(),
            "allowed_imports": self.allowed_imports.copy(),
            "blocked_imports": self.blocked_imports.copy()
        }
        
        # Adjust restrictions based on security level
        if level == SecurityLevel.MINIMAL:
            restrictions["max_memory_mb"] = min(restrictions["max_memory_mb"], 128)
            restrictions["max_cpu_time_seconds"] = min(restrictions["max_cpu_time_seconds"], 10)
            restrictions["network_access"] = False
            restrictions["filesystem_access"] = False
        
        elif level == SecurityLevel.RESTRICTED:
            restrictions["max_memory_mb"] = min(restrictions["max_memory_mb"], 256)
            restrictions["max_cpu_time_seconds"] = min(restrictions["max_cpu_time_seconds"], 30)
            restrictions["network_access"] = False
        
        return restrictions


class PluginSandbox:
    """Sandbox for secure plugin execution."""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.policy = security_policy
        self._active_processes = {}
    
    def create_sandbox_environment(self, plugin_name: str, plugin_info: PluginInfo) -> Dict[str, Any]:
        """Create a sandboxed environment for plugin execution."""
        restrictions = self.policy.get_execution_restrictions(plugin_info)
        
        # Create temporary directory for plugin execution
        sandbox_dir = tempfile.mkdtemp(prefix=f"agentnet_plugin_{plugin_name}_")
        
        # Prepare environment variables
        env = os.environ.copy()
        env["AGENTNET_PLUGIN_SANDBOX"] = "true"
        env["AGENTNET_PLUGIN_NAME"] = plugin_name
        env["AGENTNET_SANDBOX_DIR"] = sandbox_dir
        
        # Restrict environment based on security level
        if restrictions["security_level"] in ["sandboxed", "restricted", "minimal"]:
            # Remove potentially dangerous environment variables
            dangerous_vars = ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "HOME"]
            for var in dangerous_vars:
                if var in env:
                    del env[var]
            
            # Set restricted PATH
            env["PATH"] = "/usr/bin:/bin"
            env["PYTHONPATH"] = ""
        
        sandbox_env = {
            "sandbox_dir": sandbox_dir,
            "environment": env,
            "restrictions": restrictions,
            "created_at": os.time.time()
        }
        
        logger.info(f"Created sandbox for plugin {plugin_name} at {sandbox_dir}")
        return sandbox_env
    
    def execute_in_sandbox(self, plugin_name: str, sandbox_env: Dict[str, Any], 
                          command: List[str]) -> subprocess.CompletedProcess:
        """Execute a command in the plugin sandbox."""
        restrictions = sandbox_env["restrictions"]
        
        # Prepare execution environment
        exec_env = sandbox_env["environment"].copy()
        
        # Resource limits (platform dependent)
        preexec_fn = None
        if hasattr(os, 'setrlimit'):
            import resource
            
            def limit_resources():
                # Memory limit
                max_memory = restrictions["max_memory_mb"] * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
                
                # CPU time limit
                max_cpu = restrictions["max_cpu_time_seconds"]
                resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu))
            
            preexec_fn = limit_resources
        
        # Execute with restrictions
        try:
            process = subprocess.run(
                command,
                env=exec_env,
                cwd=sandbox_env["sandbox_dir"],
                preexec_fn=preexec_fn,
                timeout=restrictions["max_cpu_time_seconds"] + 10,  # Grace period
                capture_output=True,
                text=True
            )
            
            logger.info(f"Sandbox execution completed for plugin {plugin_name}")
            return process
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Plugin {plugin_name} execution timed out")
            raise
        except Exception as e:
            logger.error(f"Sandbox execution failed for plugin {plugin_name}: {e}")
            raise
    
    def cleanup_sandbox(self, sandbox_env: Dict[str, Any]) -> None:
        """Clean up sandbox environment."""
        sandbox_dir = Path(sandbox_env["sandbox_dir"])
        
        if sandbox_dir.exists():
            import shutil
            shutil.rmtree(sandbox_dir)
            logger.info(f"Cleaned up sandbox directory: {sandbox_dir}")
    
    def monitor_plugin_resources(self, plugin_name: str) -> Dict[str, Any]:
        """Monitor resource usage of a plugin."""
        # This would integrate with system monitoring tools
        # For now, return placeholder data
        return {
            "plugin_name": plugin_name,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "network_connections": 0,
            "file_operations": 0,
            "status": "monitoring"
        }
    
    def validate_plugin_imports(self, plugin_code: str, restrictions: Dict[str, Any]) -> List[str]:
        """Validate plugin imports against security policy."""
        import ast
        
        violations = []
        allowed_imports = restrictions.get("allowed_imports", set())
        blocked_imports = restrictions.get("blocked_imports", set())
        
        try:
            tree = ast.parse(plugin_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if self._is_import_blocked(module_name, allowed_imports, blocked_imports):
                            violations.append(f"Blocked import: {module_name}")
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if self._is_import_blocked(module_name, allowed_imports, blocked_imports):
                        violations.append(f"Blocked import from: {module_name}")
        
        except SyntaxError as e:
            violations.append(f"Syntax error in plugin code: {e}")
        
        return violations
    
    def _is_import_blocked(self, module_name: str, allowed: Set[str], blocked: Set[str]) -> bool:
        """Check if an import is blocked by security policy."""
        # Check explicit blocks
        if module_name in blocked:
            return True
        
        # Check if module starts with blocked prefix
        for blocked_prefix in blocked:
            if module_name.startswith(blocked_prefix + "."):
                return True
        
        # If allowlist is defined, check if module is allowed
        if allowed:
            if module_name not in allowed:
                # Check if module starts with allowed prefix
                allowed_prefixes = [prefix for prefix in allowed if module_name.startswith(prefix + ".")]
                if not allowed_prefixes:
                    return True
        
        return False
    
    def create_default_security_policy(self) -> SecurityPolicy:
        """Create a default security policy for plugins."""
        return SecurityPolicy(
            name="default_agentnet_policy",
            version="1.0",
            default_level=SecurityLevel.SANDBOXED,
            allowed_permissions={
                "agentnet.agent.create",
                "agentnet.monitor.register",
                "agentnet.tool.register",
                "agentnet.memory.read",
                "agentnet.memory.write"
            },
            blocked_permissions={
                "system.execute",
                "network.raw_socket",
                "filesystem.write_system"
            },
            allowed_imports={
                "json", "re", "datetime", "uuid", "logging",
                "typing", "dataclasses", "enum", "abc",
                "agentnet", "agentnet.core", "agentnet.monitors",
                "agentnet.tools", "agentnet.memory"
            },
            blocked_imports={
                "os", "sys", "subprocess", "socket", "urllib",
                "requests", "httpx", "aiohttp", "sqlite3",
                "pymongo", "psycopg2", "mysql"
            },
            max_memory_mb=256,
            max_cpu_time_seconds=30,
            network_access=False,
            filesystem_access=False,
            allowed_directories=["/tmp/agentnet_plugins"]
        )
    
    def audit_plugin_activity(self, plugin_name: str) -> Dict[str, Any]:
        """Generate audit report for plugin activity."""
        # This would integrate with the audit system
        return {
            "plugin_name": plugin_name,
            "audit_timestamp": os.time.time(),
            "resource_usage": self.monitor_plugin_resources(plugin_name),
            "security_violations": [],
            "execution_log": [],
            "status": "compliant"
        }