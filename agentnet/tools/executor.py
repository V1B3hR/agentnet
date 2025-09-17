"""Tool execution engine with rate limiting and auth."""

from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolError, ToolStatus
from .registry import ToolRegistry
from .rate_limiter import RateLimiter


class ToolExecutor:
    """Executes tools with rate limiting, auth, and error handling."""
    
    def __init__(
        self, 
        registry: ToolRegistry,
        rate_limiter: Optional[RateLimiter] = None,
        auth_provider: Optional[Any] = None
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter or RateLimiter()
        self.auth_provider = auth_provider
        
        # Execution cache for deterministic tools
        self._cache: Dict[str, ToolResult] = {}
        
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
        context: Optional[Dict[str, Any]] = None
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
                    execution_time=time.time() - start_time
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
                        "requests_remaining": rate_info.requests_remaining
                    }
                )
            
            # Check authentication
            if tool.spec.auth_required:
                auth_result = await self._check_auth(tool.spec.auth_scope, user_id, context)
                if not auth_result:
                    return ToolResult(
                        status=ToolStatus.AUTH_FAILED,
                        error_message="Authentication failed",
                        execution_time=time.time() - start_time
                    )
            
            # Validate parameters
            try:
                tool.validate_parameters(parameters)
            except ToolError as e:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error_message=str(e),
                    execution_time=time.time() - start_time
                )
            
            # Record request for rate limiting
            self.rate_limiter.record_request(tool_name, user_id)
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    tool.execute(parameters, context),
                    timeout=tool.spec.timeout_seconds
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    status=ToolStatus.TIMEOUT,
                    error_message=f"Tool execution timed out after {tool.spec.timeout_seconds}s",
                    execution_time=time.time() - start_time
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
                execution_time=time.time() - start_time
            )
    
    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        tasks = []
        
        for call in tool_calls:
            task = self.execute_tool(
                tool_name=call["tool"],
                parameters=call["parameters"],
                user_id=user_id,
                context=context
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_auth(
        self, 
        auth_scope: Optional[str], 
        user_id: Optional[str],
        context: Optional[Dict[str, Any]]
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
        cache_data = {
            "tool": tool_name,
            "params": parameters
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear execution cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_results": len(self._cache),
            "cache_keys": list(self._cache.keys())
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "available_tools": self.registry.tool_count,
            "cached_results": len(self._cache),
            "rate_limiter_active": bool(self.rate_limiter._tool_limits)
        }