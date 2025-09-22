"""
Stream Handlers for AgentNet Streaming Collaboration

Provides specialized handlers for different types of streaming operations,
error handling, and collaboration patterns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, AsyncIterator
from enum import Enum

from .collaboration import PartialResponse, CollaborationSession
from .parser import ParseResult

logger = logging.getLogger(__name__)


class HandlerType(str, Enum):
    """Types of stream handlers."""
    STREAM = "stream"
    COLLABORATION = "collaboration"
    ERROR = "error"
    FILTER = "filter"
    TRANSFORM = "transform"


class StreamHandler(ABC):
    """Base class for stream handlers."""
    
    def __init__(self, name: str, handler_type: HandlerType):
        self.name = name
        self.handler_type = handler_type
        self.is_active = True
    
    @abstractmethod
    async def handle(self, data: Any, context: Dict[str, Any]) -> Any:
        """Handle stream data."""
        pass
    
    def activate(self) -> None:
        """Activate the handler."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Deactivate the handler."""
        self.is_active = False


@dataclass
class StreamContext:
    """Context information for stream processing."""
    
    session_id: str
    agent_id: str
    stream_id: str
    
    # Processing state
    bytes_processed: int = 0
    chunks_processed: int = 0
    start_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborationHandler(StreamHandler):
    """
    Handler for collaborative streaming between agents.
    
    Manages coordination, turn-taking, and response aggregation
    in multi-agent streaming scenarios.
    """
    
    def __init__(self, 
                 name: str = "collaboration_handler",
                 max_concurrent_streams: int = 5,
                 turn_timeout_seconds: float = 30.0):
        super().__init__(name, HandlerType.COLLABORATION)
        self.max_concurrent_streams = max_concurrent_streams
        self.turn_timeout_seconds = turn_timeout_seconds
        
        self._active_streams: Dict[str, StreamContext] = {}
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._coordination_lock = asyncio.Lock()
    
    async def handle(self, data: Any, context: Dict[str, Any]) -> Any:
        """Handle collaborative streaming data."""
        
        if not self.is_active:
            return data
        
        # Extract stream information
        stream_id = context.get('stream_id', 'unknown')
        session_id = context.get('session_id', 'unknown')
        agent_id = context.get('agent_id', 'unknown')
        
        # Create or update stream context
        if stream_id not in self._active_streams:
            self._active_streams[stream_id] = StreamContext(
                session_id=session_id,
                agent_id=agent_id,
                stream_id=stream_id
            )
        
        stream_context = self._active_streams[stream_id]
        stream_context.chunks_processed += 1
        
        if isinstance(data, (str, bytes)):
            stream_context.bytes_processed += len(data)
        
        # Coordinate with other streams
        async with self._coordination_lock:
            # Check if we need to throttle or coordinate
            if len(self._active_streams) > self.max_concurrent_streams:
                logger.warning(f"Too many concurrent streams ({len(self._active_streams)}), throttling")
                await asyncio.sleep(0.1)
            
            # Process the data
            processed_data = await self._process_collaborative_data(data, stream_context, context)
        
        return processed_data
    
    async def _process_collaborative_data(self, 
                                        data: Any, 
                                        stream_context: StreamContext,
                                        context: Dict[str, Any]) -> Any:
        """Process data in collaborative context."""
        
        # Add collaboration metadata
        if isinstance(data, dict):
            data['collaboration_metadata'] = {
                'stream_id': stream_context.stream_id,
                'agent_id': stream_context.agent_id,
                'session_id': stream_context.session_id,
                'chunks_processed': stream_context.chunks_processed,
                'bytes_processed': stream_context.bytes_processed
            }
        
        # Queue for coordination if needed
        await self._response_queue.put({
            'stream_id': stream_context.stream_id,
            'data': data,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        return data
    
    async def coordinate_responses(self, session_id: str) -> List[Dict[str, Any]]:
        """Coordinate responses across multiple agents in a session."""
        
        responses = []
        timeout = asyncio.get_event_loop().time() + self.turn_timeout_seconds
        
        while asyncio.get_event_loop().time() < timeout:
            try:
                # Get response with timeout
                response = await asyncio.wait_for(
                    self._response_queue.get(),
                    timeout=1.0
                )
                
                # Filter by session
                stream_context = self._active_streams.get(response['stream_id'])
                if stream_context and stream_context.session_id == session_id:
                    responses.append(response)
                
                # Mark task as done
                self._response_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
        
        return responses
    
    def cleanup_stream(self, stream_id: str) -> None:
        """Clean up completed stream."""
        
        if stream_id in self._active_streams:
            del self._active_streams[stream_id]
            logger.debug(f"Cleaned up stream {stream_id}")
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about active streams."""
        
        return {
            'active_streams': len(self._active_streams),
            'max_concurrent': self.max_concurrent_streams,
            'queue_size': self._response_queue.qsize(),
            'streams': {
                stream_id: {
                    'chunks_processed': ctx.chunks_processed,
                    'bytes_processed': ctx.bytes_processed,
                    'error_count': ctx.error_count
                }
                for stream_id, ctx in self._active_streams.items()
            }
        }


class ErrorHandler(StreamHandler):
    """
    Handler for streaming errors and recovery.
    
    Provides robust error handling, retry logic, and graceful
    degradation for streaming operations.
    """
    
    def __init__(self, 
                 name: str = "error_handler",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_recovery: bool = True):
        super().__init__(name, HandlerType.ERROR)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_recovery = enable_recovery
        
        self._error_counts: Dict[str, int] = {}
        self._recovery_strategies: Dict[str, Callable] = {}
    
    async def handle(self, data: Any, context: Dict[str, Any]) -> Any:
        """Handle streaming errors and implement recovery."""
        
        if not self.is_active:
            return data
        
        stream_id = context.get('stream_id', 'unknown')
        
        try:
            # Check if this is an error case
            if isinstance(data, Exception) or context.get('is_error', False):
                return await self._handle_error(data, stream_id, context)
            
            # Process normal data
            return await self._process_safe_data(data, stream_id, context)
            
        except Exception as e:
            return await self._handle_error(e, stream_id, context)
    
    async def _handle_error(self, error: Any, stream_id: str, context: Dict[str, Any]) -> Any:
        """Handle streaming error with recovery strategies."""
        
        # Track error count
        self._error_counts.setdefault(stream_id, 0)
        self._error_counts[stream_id] += 1
        
        error_count = self._error_counts[stream_id]
        error_msg = str(error) if isinstance(error, Exception) else str(error)
        
        logger.warning(f"Stream error in {stream_id} (attempt {error_count}): {error_msg}")
        
        # Check if we should retry
        if error_count <= self.max_retries and self.enable_recovery:
            # Wait before retry
            await asyncio.sleep(self.retry_delay * error_count)
            
            # Try recovery strategy
            recovery_result = await self._attempt_recovery(error, stream_id, context)
            if recovery_result is not None:
                logger.info(f"Stream {stream_id} recovered successfully")
                return recovery_result
        
        # Create error response
        error_response = {
            'error': True,
            'error_type': type(error).__name__ if isinstance(error, Exception) else 'StreamError',
            'error_message': error_msg,
            'stream_id': stream_id,
            'error_count': error_count,
            'max_retries_exceeded': error_count > self.max_retries,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        logger.error(f"Stream {stream_id} failed after {error_count} attempts: {error_msg}")
        return error_response
    
    async def _process_safe_data(self, data: Any, stream_id: str, context: Dict[str, Any]) -> Any:
        """Process data with error protection."""
        
        try:
            # Reset error count on successful processing
            if stream_id in self._error_counts:
                self._error_counts[stream_id] = 0
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing data in stream {stream_id}: {e}")
            raise
    
    async def _attempt_recovery(self, error: Any, stream_id: str, context: Dict[str, Any]) -> Optional[Any]:
        """Attempt to recover from streaming error."""
        
        error_type = type(error).__name__ if isinstance(error, Exception) else 'Unknown'
        
        # Try registered recovery strategy
        if error_type in self._recovery_strategies:
            try:
                recovery_func = self._recovery_strategies[error_type]
                return await recovery_func(error, stream_id, context)
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Default recovery strategies
        if error_type == 'JSONDecodeError':
            # Try to recover partial JSON
            return await self._recover_partial_json(error, context)
        elif error_type == 'TimeoutError':
            # Extend timeout and retry
            return await self._recover_timeout(error, context)
        elif error_type == 'ConnectionError':
            # Reconnect and retry
            return await self._recover_connection(error, context)
        
        return None
    
    async def _recover_partial_json(self, error: Any, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from partial JSON parsing errors."""
        
        # Try to extract partial data
        partial_data = context.get('partial_data', {})
        if partial_data:
            logger.info("Recovered partial JSON data")
            return {
                'recovered': True,
                'partial_data': partial_data,
                'recovery_type': 'partial_json'
            }
        
        return None
    
    async def _recover_timeout(self, error: Any, context: Dict[str, Any]) -> Optional[Any]:
        """Recover from timeout errors."""
        
        # Extend timeout for next attempt
        current_timeout = context.get('timeout', 30.0)
        extended_timeout = min(current_timeout * 1.5, 120.0)  # Cap at 2 minutes
        
        context['timeout'] = extended_timeout
        logger.info(f"Extended timeout to {extended_timeout}s for recovery")
        
        return {
            'recovered': True,
            'extended_timeout': extended_timeout,
            'recovery_type': 'timeout_extension'
        }
    
    async def _recover_connection(self, error: Any, context: Dict[str, Any]) -> Optional[Any]:
        """Recover from connection errors."""
        
        # Simulate connection recovery
        await asyncio.sleep(self.retry_delay)
        
        logger.info("Attempted connection recovery")
        return {
            'recovered': True,
            'recovery_type': 'connection_reset'
        }
    
    def register_recovery_strategy(self, error_type: str, recovery_func: Callable) -> None:
        """Register a custom recovery strategy for an error type."""
        
        self._recovery_strategies[error_type] = recovery_func
        logger.info(f"Registered recovery strategy for {error_type}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics across all streams."""
        
        total_errors = sum(self._error_counts.values())
        error_streams = len(self._error_counts)
        
        return {
            'total_errors': total_errors,
            'error_streams': error_streams,
            'recovery_strategies': len(self._recovery_strategies),
            'max_retries': self.max_retries,
            'error_details': dict(self._error_counts)
        }


class FilterHandler(StreamHandler):
    """Handler for filtering streaming content based on policies or patterns."""
    
    def __init__(self, 
                 name: str = "filter_handler",
                 filters: Optional[List[Callable]] = None):
        super().__init__(name, HandlerType.FILTER)
        self.filters = filters or []
        self._filter_stats: Dict[str, int] = {}
    
    async def handle(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply filters to streaming data."""
        
        if not self.is_active or not self.filters:
            return data
        
        # Apply each filter
        filtered_data = data
        for i, filter_func in enumerate(self.filters):
            try:
                if asyncio.iscoroutinefunction(filter_func):
                    filtered_data = await filter_func(filtered_data, context)
                else:
                    filtered_data = filter_func(filtered_data, context)
                
                # Track filter usage
                filter_name = getattr(filter_func, '__name__', f'filter_{i}')
                self._filter_stats.setdefault(filter_name, 0)
                self._filter_stats[filter_name] += 1
                
            except Exception as e:
                logger.error(f"Error in filter {i}: {e}")
                # Continue with unfiltered data
        
        return filtered_data
    
    def add_filter(self, filter_func: Callable) -> None:
        """Add a new filter function."""
        self.filters.append(filter_func)
    
    def remove_filter(self, filter_func: Callable) -> None:
        """Remove a filter function."""
        if filter_func in self.filters:
            self.filters.remove(filter_func)
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filter usage statistics."""
        return {
            'active_filters': len(self.filters),
            'filter_usage': dict(self._filter_stats)
        }


class TransformHandler(StreamHandler):
    """Handler for transforming streaming data between formats or structures."""
    
    def __init__(self, 
                 name: str = "transform_handler",
                 transformers: Optional[List[Callable]] = None):
        super().__init__(name, HandlerType.TRANSFORM)
        self.transformers = transformers or []
        self._transform_stats: Dict[str, int] = {}
    
    async def handle(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply transformations to streaming data."""
        
        if not self.is_active or not self.transformers:
            return data
        
        # Apply transformations in sequence
        transformed_data = data
        for i, transform_func in enumerate(self.transformers):
            try:
                if asyncio.iscoroutinefunction(transform_func):
                    transformed_data = await transform_func(transformed_data, context)
                else:
                    transformed_data = transform_func(transformed_data, context)
                
                # Track transformer usage
                transformer_name = getattr(transform_func, '__name__', f'transformer_{i}')
                self._transform_stats.setdefault(transformer_name, 0)
                self._transform_stats[transformer_name] += 1
                
            except Exception as e:
                logger.error(f"Error in transformer {i}: {e}")
                # Continue with original data
        
        return transformed_data
    
    def add_transformer(self, transform_func: Callable) -> None:
        """Add a new transformer function."""
        self.transformers.append(transform_func)
    
    def remove_transformer(self, transform_func: Callable) -> None:
        """Remove a transformer function."""
        if transform_func in self.transformers:
            self.transformers.remove(transform_func)
    
    def get_transform_stats(self) -> Dict[str, Any]:
        """Get transformation usage statistics."""
        return {
            'active_transformers': len(self.transformers),
            'transform_usage': dict(self._transform_stats)
        }


# Utility functions for common handler patterns

def create_policy_filter(policy_rules: List[Callable]) -> Callable:
    """Create a filter that applies policy rules to streaming content."""
    
    def policy_filter(data: Any, context: Dict[str, Any]) -> Any:
        if isinstance(data, str):
            # Apply text-based policy rules
            for rule in policy_rules:
                if not rule(data):
                    logger.warning("Policy filter blocked content")
                    return {"filtered": True, "reason": "policy_violation"}
        
        return data
    
    return policy_filter


def create_json_transformer() -> Callable:
    """Create a transformer that converts text to JSON structure."""
    
    def json_transformer(data: Any, context: Dict[str, Any]) -> Any:
        if isinstance(data, str):
            try:
                import json
                parsed = json.loads(data)
                return parsed
            except json.JSONDecodeError:
                # Return partial JSON if available
                parse_result = context.get('parse_result')
                if parse_result and parse_result.partial_data:
                    return parse_result.partial_data
        
        return data
    
    return json_transformer


async def create_async_rate_limiter(requests_per_second: float) -> Callable:
    """Create an async rate limiter for streaming operations."""
    
    last_request_time = 0.0
    min_interval = 1.0 / requests_per_second
    
    async def rate_limiter(data: Any, context: Dict[str, Any]) -> Any:
        nonlocal last_request_time
        
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - last_request_time
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        last_request_time = asyncio.get_event_loop().time()
        return data
    
    return rate_limiter