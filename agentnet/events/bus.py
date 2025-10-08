"""
Event Bus implementation for AgentNet observability.

Provides a simple event system for capturing and routing events
to different sinks (console, file, etc.).
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger("agentnet.events.bus")


class EventType(str, Enum):
    """Standard event types for AgentNet."""

    # Turn-based orchestration events
    TURN_START = "turn.start"
    TURN_END = "turn.end"
    SESSION_START = "session.start"
    SESSION_END = "session.end"

    # Agent events
    AGENT_REQUEST = "agent.request"
    AGENT_RESPONSE = "agent.response"

    # Policy events
    POLICY_VIOLATION = "policy.violation"
    POLICY_EVALUATION = "policy.evaluation"

    # Tool events
    TOOL_INVOKE = "tool.invoke"
    TOOL_RESULT = "tool.result"

    # Memory events
    MEMORY_STORE = "memory.store"
    MEMORY_RETRIEVE = "memory.retrieve"

    # Evaluation events
    EVALUATOR_SCORE = "evaluator.score"

    # Session events
    SESSION_CHECKPOINT = "session.checkpoint"

    # Error events
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Event:
    """A single event in the AgentNet system."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.INFO
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        """Get event timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return f"Event({self.event_type.value}, {self.source}, {self.datetime})"


class EventBus:
    """
    Simple event bus for AgentNet system observability.

    Supports multiple sinks for event processing and filtering.
    """

    def __init__(
        self,
        name: str = "default",
        buffer_size: int = 1000,
        async_processing: bool = True,
    ):
        """
        Initialize the event bus.

        Args:
            name: Name identifier for this event bus
            buffer_size: Maximum events to buffer
            async_processing: Whether to process events asynchronously
        """
        self.name = name
        self.buffer_size = buffer_size
        self.async_processing = async_processing

        self.sinks: List[Any] = []  # EventSink instances
        self.event_buffer: List[Event] = []
        self.total_events = 0
        self.dropped_events = 0
        self.created_time = time.time()

        # Event filtering
        self.enabled_types: Optional[set] = None  # None means all types enabled
        self.disabled_types: set = set()

        # Async processing
        self._processing_queue: Optional[asyncio.Queue] = None
        self._processor_task: Optional[asyncio.Task] = None

        if self.async_processing:
            self._setup_async_processing()

        logger.info(f"EventBus '{name}' initialized")

    def add_sink(self, sink: Any) -> None:
        """Add an event sink for processing events."""
        self.sinks.append(sink)
        logger.info(f"Added sink {sink.__class__.__name__} to EventBus '{self.name}'")

    def remove_sink(self, sink: Any) -> None:
        """Remove an event sink."""
        if sink in self.sinks:
            self.sinks.remove(sink)
            logger.info(
                f"Removed sink {sink.__class__.__name__} from EventBus '{self.name}'"
            )

    def enable_event_types(self, event_types: List[EventType]) -> None:
        """Enable only specific event types."""
        self.enabled_types = set(event_types)
        logger.info(
            f"EventBus '{self.name}' enabled types: {[t.value for t in event_types]}"
        )

    def disable_event_types(self, event_types: List[EventType]) -> None:
        """Disable specific event types."""
        self.disabled_types.update(event_types)
        logger.info(
            f"EventBus '{self.name}' disabled types: {[t.value for t in event_types]}"
        )

    def emit(
        self,
        event_type: EventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Emit an event to the bus.

        Args:
            event_type: Type of event
            source: Source component/module
            data: Event data payload
            metadata: Additional metadata

        Returns:
            The created Event instance
        """
        # Check if event type is enabled
        if not self._is_event_enabled(event_type):
            return None

        # Create event
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {},
            metadata=metadata or {},
        )

        self.total_events += 1

        # Process event
        if self.async_processing and self._processing_queue:
            try:
                self._processing_queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped_events += 1
                logger.warning(f"EventBus '{self.name}' queue full, dropping event")
        else:
            self._process_event_sync(event)

        return event

    def emit_turn_start(
        self,
        session_id: str,
        agent_name: str,
        turn_number: int,
        round_number: int,
        **kwargs,
    ) -> Event:
        """Emit a turn start event."""
        return self.emit(
            EventType.TURN_START,
            "turn_engine",
            {
                "session_id": session_id,
                "agent_name": agent_name,
                "turn_number": turn_number,
                "round_number": round_number,
                **kwargs,
            },
        )

    def emit_turn_end(
        self, session_id: str, agent_name: str, turn_result: Dict[str, Any], **kwargs
    ) -> Event:
        """Emit a turn end event."""
        return self.emit(
            EventType.TURN_END,
            "turn_engine",
            {
                "session_id": session_id,
                "agent_name": agent_name,
                "turn_result": turn_result,
                **kwargs,
            },
        )

    def emit_policy_violation(
        self,
        rule_name: str,
        severity: str,
        agent_name: str,
        violation_details: Dict[str, Any],
        **kwargs,
    ) -> Event:
        """Emit a policy violation event."""
        return self.emit(
            EventType.POLICY_VIOLATION,
            "policy_engine",
            {
                "rule_name": rule_name,
                "severity": severity,
                "agent_name": agent_name,
                "violation_details": violation_details,
                **kwargs,
            },
        )

    def emit_session_start(
        self, session_id: str, session_type: str, agents: List[str], **kwargs
    ) -> Event:
        """Emit a session start event."""
        return self.emit(
            EventType.SESSION_START,
            "orchestrator",
            {
                "session_id": session_id,
                "session_type": session_type,
                "agents": agents,
                **kwargs,
            },
        )

    def emit_session_end(
        self, session_id: str, status: str, duration: float, **kwargs
    ) -> Event:
        """Emit a session end event."""
        return self.emit(
            EventType.SESSION_END,
            "orchestrator",
            {
                "session_id": session_id,
                "status": status,
                "duration": duration,
                **kwargs,
            },
        )

    def _is_event_enabled(self, event_type: EventType) -> bool:
        """Check if an event type is enabled."""
        if event_type in self.disabled_types:
            return False

        if self.enabled_types is not None:
            return event_type in self.enabled_types

        return True

    def _process_event_sync(self, event: Event) -> None:
        """Process event synchronously."""
        # Add to buffer
        self.event_buffer.append(event)

        # Trim buffer if needed
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)

        # Send to sinks
        for sink in self.sinks:
            try:
                sink.process_event(event)
            except Exception as e:
                logger.error(f"Error in sink {sink.__class__.__name__}: {e}")

    def _setup_async_processing(self) -> None:
        """Set up async event processing."""
        self._processing_queue = asyncio.Queue(maxsize=self.buffer_size)
        # Start processor task when event loop is available
        try:
            loop = asyncio.get_event_loop()
            self._processor_task = loop.create_task(self._process_events_async())
        except RuntimeError:
            # No event loop running yet, will start later
            pass

    async def _process_events_async(self) -> None:
        """Async event processor."""
        while True:
            try:
                event = await self._processing_queue.get()

                # Add to buffer
                self.event_buffer.append(event)

                # Trim buffer if needed
                if len(self.event_buffer) > self.buffer_size:
                    self.event_buffer.pop(0)

                # Send to sinks
                for sink in self.sinks:
                    try:
                        if hasattr(sink, "process_event_async"):
                            await sink.process_event_async(event)
                        else:
                            sink.process_event(event)
                    except Exception as e:
                        logger.error(f"Error in sink {sink.__class__.__name__}: {e}")

                self._processing_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in async event processor: {e}")

    async def start_async_processing(self) -> None:
        """Start async processing if not already started."""
        if self.async_processing and not self._processor_task:
            self._setup_async_processing()

    async def stop_async_processing(self) -> None:
        """Stop async processing."""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Get events from buffer with optional filtering."""
        events = self.event_buffer

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        if limit:
            events = events[-limit:]

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        uptime = time.time() - self.created_time

        return {
            "name": self.name,
            "uptime": uptime,
            "total_events": self.total_events,
            "dropped_events": self.dropped_events,
            "buffer_size": len(self.event_buffer),
            "max_buffer_size": self.buffer_size,
            "sinks_count": len(self.sinks),
            "async_processing": self.async_processing,
            "enabled_types": (
                [t.value for t in self.enabled_types] if self.enabled_types else "all"
            ),
            "disabled_types": [t.value for t in self.disabled_types],
        }

    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        cleared_count = len(self.event_buffer)
        self.event_buffer.clear()
        logger.info(
            f"Cleared {cleared_count} events from EventBus '{self.name}' buffer"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event bus to dictionary representation."""
        return {
            "name": self.name,
            "stats": self.get_stats(),
            "recent_events": [e.to_dict() for e in self.event_buffer[-10:]],
        }
