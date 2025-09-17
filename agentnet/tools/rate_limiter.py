"""Rate limiting for tool execution."""

from __future__ import annotations
import time
from collections import defaultdict, deque
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RateLimitInfo:
    """Information about rate limiting status."""
    allowed: bool
    requests_remaining: int
    reset_time: float
    retry_after: Optional[float] = None


class RateLimiter:
    """Token bucket rate limiter for tool execution."""
    
    def __init__(self):
        # Per-tool rate limit tracking
        self._tool_requests: Dict[str, deque] = defaultdict(deque)
        self._tool_limits: Dict[str, int] = {}
    
    def set_tool_limit(self, tool_name: str, requests_per_minute: int) -> None:
        """Set rate limit for a specific tool."""
        self._tool_limits[tool_name] = requests_per_minute
    
    def check_rate_limit(self, tool_name: str, user_id: Optional[str] = None) -> RateLimitInfo:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Get rate limit for tool
        limit = self._tool_limits.get(tool_name)
        if limit is None:
            # No rate limit configured
            return RateLimitInfo(
                allowed=True,
                requests_remaining=float('inf'),
                reset_time=current_time
            )
        
        # Create rate limit key (tool + user if provided)
        rate_key = f"{tool_name}:{user_id}" if user_id else tool_name
        
        # Clean old requests (older than 1 minute)
        request_times = self._tool_requests[rate_key]
        cutoff_time = current_time - 60  # 1 minute ago
        
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Check if within limit
        current_count = len(request_times)
        
        if current_count >= limit:
            # Rate limited
            oldest_request = request_times[0] if request_times else current_time
            reset_time = oldest_request + 60
            retry_after = reset_time - current_time
            
            return RateLimitInfo(
                allowed=False,
                requests_remaining=0,
                reset_time=reset_time,
                retry_after=max(0, retry_after)
            )
        
        # Within limits
        return RateLimitInfo(
            allowed=True,
            requests_remaining=limit - current_count,
            reset_time=current_time + 60
        )
    
    def record_request(self, tool_name: str, user_id: Optional[str] = None) -> None:
        """Record a request for rate limiting."""
        rate_key = f"{tool_name}:{user_id}" if user_id else tool_name
        self._tool_requests[rate_key].append(time.time())
    
    def get_rate_limit_status(self, tool_name: str, user_id: Optional[str] = None) -> Dict[str, any]:
        """Get current rate limit status."""
        rate_info = self.check_rate_limit(tool_name, user_id)
        limit = self._tool_limits.get(tool_name, 0)
        
        return {
            "tool_name": tool_name,
            "user_id": user_id,
            "limit_per_minute": limit,
            "requests_remaining": rate_info.requests_remaining,
            "reset_time": rate_info.reset_time,
            "allowed": rate_info.allowed
        }
    
    def clear_limits(self) -> None:
        """Clear all rate limiting data."""
        self._tool_requests.clear()
        self._tool_limits.clear()
    
    def clear_tool_limits(self, tool_name: str) -> None:
        """Clear rate limiting data for specific tool."""
        # Remove all entries for this tool
        keys_to_remove = [key for key in self._tool_requests.keys() if key.startswith(f"{tool_name}:") or key == tool_name]
        
        for key in keys_to_remove:
            del self._tool_requests[key]
        
        if tool_name in self._tool_limits:
            del self._tool_limits[tool_name]