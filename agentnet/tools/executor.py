"""Tool execution engine with rate limiting, auth, and policy enforcement."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError, ToolResult, ToolStatus
from .rate_limiter import RateLimiter
from .registry import ToolRegistry

logger = logging.getLogger("agentnet.tools.executor")


class ToolExecutor:
    """Executes tools with rate limiting, auth, and policy enforcement."""

    def __init__(
        self,
        registry: ToolRegistry,
        rate_limiter: Optional[RateLimiter] = None,
        auth_provider: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter or RateLimiter()
        self.auth_provider = auth_provider
        self.policy_engine = policy_engine

        # Execution cache for deterministic tools
        self._cache: Dict[str, ToolResult] = {}

        # Security configuration
        self.enable_sandboxing = True
        self.security_checks = True

        # Tool governance settings
        self.governance_enabled = policy_engine is not None
        self.custom_validators: Dict[str, List[callable]] = {}

        # Setup rate limits from registry
        self._setup_rate_limits()

    def _setup_rate_limits(self) -> None:
        """Configure rate limits from tool specifications."""
        for spec in self.registry.list_tool_specs():
            if spec.rate_limit_per_min:
                self.rate_limiter.set_tool_limit(spec.name, spec.rate_limit_per_min)

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute a tool with full error handling and rate limiting."""
        start_time = time.time()

        try:
            # Get tool
            tool = self.registry.get_tool(tool_name)
            if not tool:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error_message=f"Tool '{tool_name}' not found",
                    execution_time=time.time() - start_time,
                )

            # Check cache for deterministic tools
            if tool.spec.cached:
                cache_key = self._generate_cache_key(tool_name, parameters)
                if cache_key in self._cache:
                    cached_result = self._cache[cache_key]
                    # Update execution time but keep other data
                    cached_result.execution_time = time.time() - start_time
                    return cached_result

            # Check rate limits
            rate_info = self.rate_limiter.check_rate_limit(tool_name, user_id)
            if not rate_info.allowed:
                return ToolResult(
                    status=ToolStatus.RATE_LIMITED,
                    error_message=f"Rate limit exceeded. Retry after {rate_info.retry_after:.1f}s",
                    execution_time=time.time() - start_time,
                    metadata={
                        "retry_after": rate_info.retry_after,
                        "requests_remaining": rate_info.requests_remaining,
                    },
                )

            # Check authentication
            if tool.spec.auth_required:
                auth_result = await self._check_auth(
                    tool.spec.auth_scope, user_id, context
                )
                if not auth_result:
                    return ToolResult(
                        status=ToolStatus.AUTH_FAILED,
                        error_message="Authentication failed",
                        execution_time=time.time() - start_time,
                    )

            # Validate parameters
            try:
                tool.validate_parameters(parameters)
            except ToolError as e:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                )

            # Record request for rate limiting
            self.rate_limiter.record_request(tool_name, user_id)

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    tool.execute(parameters, context), timeout=tool.spec.timeout_seconds
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    status=ToolStatus.TIMEOUT,
                    error_message=f"Tool execution timed out after {tool.spec.timeout_seconds}s",
                    execution_time=time.time() - start_time,
                )

            # Update execution time
            result.execution_time = time.time() - start_time

            # Cache result if tool is deterministic
            if tool.spec.cached and result.status == ToolStatus.SUCCESS:
                cache_key = self._generate_cache_key(tool_name, parameters)
                self._cache[cache_key] = result

            return result

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error_message=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start_time,
            )

    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        tasks = []

        for call in tool_calls:
            task = self.execute_tool(
                tool_name=call["tool"],
                parameters=call["parameters"],
                user_id=user_id,
                context=context,
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_auth(
        self,
        auth_scope: Optional[str],
        user_id: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """Check authentication for tool execution."""
        if not self.auth_provider:
            # No auth provider configured - allow execution
            return True

        # Custom auth logic would go here
        # For now, simple placeholder that always allows
        return True

    def _generate_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for deterministic tool results."""
        import hashlib
        import json

        # Create deterministic string from tool name and parameters
        cache_data = {"tool": tool_name, "params": parameters}

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear execution cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_results": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "available_tools": self.registry.tool_count,
            "cached_results": len(self._cache),
            "rate_limiter_active": bool(self.rate_limiter._tool_limits),
            "governance_enabled": self.governance_enabled,
            "security_checks": self.security_checks,
            "sandboxing_enabled": self.enable_sandboxing,
        }

    def add_custom_validator(self, tool_name: str, validator_fn: callable) -> None:
        """Add a custom validation function for a specific tool."""
        if tool_name not in self.custom_validators:
            self.custom_validators[tool_name] = []
        self.custom_validators[tool_name].append(validator_fn)
        logger.info(f"Added custom validator for tool: {tool_name}")

    async def validate_tool_governance(
        self,
        tool: Tool,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate tool usage against governance policies.

        Returns:
            Dict with validation result and any policy violations
        """
        validation_result = {
            "allowed": True,
            "violations": [],
            "recommendations": [],
            "risk_level": "low",
        }

        if not self.governance_enabled:
            return validation_result

        # Create governance context
        gov_context = {
            "tool_name": tool.spec.name,
            "tool_category": tool.spec.category,
            "parameters": parameters,
            "user_id": user_id,
            "agent_name": context.get("agent_name", "") if context else "",
            "agent_role": context.get("agent_role", "") if context else "",
            "action_type": "tool_execution",
        }

        # Evaluate with policy engine
        try:
            policy_result = self.policy_engine.evaluate_tool_usage_policy(
                tool.spec.name, parameters, gov_context
            )

            if policy_result.action.value in ["block", "require_approval"]:
                validation_result["allowed"] = False
                validation_result["risk_level"] = "high"
            elif policy_result.action.value == "log":
                validation_result["risk_level"] = "medium"

            validation_result["violations"] = [
                v.to_dict() for v in policy_result.violations
            ]

        except Exception as e:
            logger.error(f"Policy evaluation failed for tool {tool.spec.name}: {e}")
            validation_result["violations"].append(
                {"type": "policy_evaluation_error", "message": str(e)}
            )

        # Run custom validators
        if tool.spec.name in self.custom_validators:
            for validator in self.custom_validators[tool.spec.name]:
                try:
                    validator_result = await validator(tool, parameters, context)
                    if not validator_result.get("valid", True):
                        validation_result["violations"].append(validator_result)
                        if validator_result.get("blocking", False):
                            validation_result["allowed"] = False
                except Exception as e:
                    logger.error(f"Custom validator failed for {tool.spec.name}: {e}")

        return validation_result

    def validate_tool_security(
        self, tool: Tool, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate tool security constraints.

        Returns:
            Dict with security validation results
        """
        security_result = {
            "secure": True,
            "security_level": "standard",
            "warnings": [],
            "sandbox_required": False,
        }

        if not self.security_checks:
            return security_result

        # Check for high-risk operations
        high_risk_categories = ["system", "file", "network", "exec"]
        if tool.spec.category in high_risk_categories:
            security_result["security_level"] = "high_risk"
            security_result["sandbox_required"] = True
            security_result["warnings"].append(
                f"Tool category '{tool.spec.category}' requires enhanced security"
            )

        # Check for dangerous parameters
        dangerous_params = ["command", "exec", "eval", "path", "url"]
        for param_name, param_value in parameters.items():
            if param_name.lower() in dangerous_params:
                security_result["warnings"].append(
                    f"Parameter '{param_name}' flagged as potentially dangerous"
                )
                if isinstance(param_value, str) and any(
                    danger in param_value.lower()
                    for danger in ["rm ", "del ", "format", "sudo", "admin"]
                ):
                    security_result["secure"] = False
                    security_result["warnings"].append(
                        f"Parameter '{param_name}' contains dangerous content"
                    )

        return security_result

    async def execute_with_governance(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute tool with full governance and security validation."""

        # Get tool
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                status=ToolStatus.ERROR,
                error_message=f"Tool '{tool_name}' not found",
                execution_time=0.0,
            )

        # Security validation
        security_result = self.validate_tool_security(tool, parameters)
        if not security_result["secure"]:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                error_message="Tool execution blocked due to security concerns",
                execution_time=0.0,
                metadata={"security_validation": security_result},
            )

        # Governance validation
        governance_result = await self.validate_tool_governance(
            tool, parameters, user_id, context
        )
        if not governance_result["allowed"]:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                error_message="Tool execution blocked by governance policy",
                execution_time=0.0,
                metadata={"governance_validation": governance_result},
            )

        # Execute tool with standard flow if all validations pass
        result = await self.execute_tool(tool_name, parameters, user_id, context)

        # Add governance metadata to result
        result.metadata.update(
            {
                "governance_validation": governance_result,
                "security_validation": security_result,
                "governance_enabled": self.governance_enabled,
            }
        )

        return result
