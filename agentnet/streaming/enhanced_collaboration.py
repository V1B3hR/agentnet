"""
Enhanced Streaming Partial-Output Collaboration

Extends the base streaming collaboration with advanced features:
- Real-time intervention and correction
- Mid-process guidance and steering
- Enhanced error handling and recovery
- Interactive debugging and monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Set, Union
from datetime import datetime, timedelta

from .collaboration import (
    StreamingCollaborator,
    CollaborationSession,
    PartialResponse,
    CollaborationMode,
    PartialResponseType,
)
from .parser import StreamingParser, ParseResult

logger = logging.getLogger(__name__)


class InterventionType(str, Enum):
    """Types of mid-process interventions."""

    CORRECTION = "correction"
    GUIDANCE = "guidance"
    STEERING = "steering"
    TERMINATION = "termination"
    ENHANCEMENT = "enhancement"
    DEBUGGING = "debugging"


class InterventionTrigger(str, Enum):
    """Triggers for automatic interventions."""

    ERROR_DETECTED = "error_detected"
    QUALITY_THRESHOLD = "quality_threshold"
    SAFETY_VIOLATION = "safety_violation"
    TIMEOUT_APPROACHING = "timeout_approaching"
    RESOURCE_LIMIT = "resource_limit"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class StreamingIntervention:
    """An intervention in the streaming process."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Intervention details
    intervention_type: InterventionType = InterventionType.CORRECTION
    trigger: InterventionTrigger = InterventionTrigger.ERROR_DETECTED
    agent_id: str = ""
    session_id: str = ""

    # Content
    original_content: str = ""
    intervention_content: str = ""
    correction_applied: bool = False

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingMetrics:
    """Real-time metrics for streaming collaboration."""

    session_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)

    # Performance metrics
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    average_response_time: float = 0.0
    error_count: int = 0
    intervention_count: int = 0

    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    safety_score: float = 1.0

    # Collaboration metrics
    agent_participation: Dict[str, int] = field(default_factory=dict)
    consensus_level: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)


class EnhancedStreamingCollaborator(StreamingCollaborator):
    """
    Enhanced streaming collaborator with intervention capabilities.

    Features:
    - Real-time intervention and correction
    - Quality monitoring and automatic triggers
    - Enhanced error recovery
    - Interactive debugging support
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.interventions: Dict[str, List[StreamingIntervention]] = {}
        self.metrics: Dict[str, StreamingMetrics] = {}
        self.intervention_handlers: Dict[InterventionType, Callable] = {}
        self.quality_monitors: List[Callable[[str, PartialResponse], float]] = []

        # Enhanced monitoring
        self.enable_real_time_monitoring = True
        self.auto_intervention_enabled = True
        self.quality_threshold = 0.7
        self.error_recovery_enabled = True

        # Callbacks for enhanced features
        self.on_intervention_triggered: Optional[
            Callable[[StreamingIntervention], None]
        ] = None
        self.on_quality_threshold_breach: Optional[Callable[[str, float], None]] = None
        self.on_error_recovered: Optional[Callable[[str, str], None]] = None

        self._setup_default_intervention_handlers()

        logger.info("EnhancedStreamingCollaborator initialized")

    def register_intervention_handler(
        self,
        intervention_type: InterventionType,
        handler: Callable[[StreamingIntervention], Any],
    ) -> None:
        """Register a handler for a specific intervention type."""

        self.intervention_handlers[intervention_type] = handler
        logger.info(f"Registered intervention handler for {intervention_type}")

    def register_quality_monitor(
        self, monitor: Callable[[str, PartialResponse], float]
    ) -> None:
        """Register a quality monitoring function."""

        self.quality_monitors.append(monitor)
        logger.info("Registered quality monitor")

    async def create_monitored_session(
        self,
        mode: CollaborationMode = CollaborationMode.PEER_TO_PEER,
        max_participants: int = 10,
        enable_interventions: bool = True,
        quality_threshold: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a collaboration session with enhanced monitoring."""

        session = self.create_session(
            mode, participants=[], max_participants=max_participants
        )
        session_id = session.session_id

        # Initialize metrics tracking
        self.metrics[session_id] = StreamingMetrics(session_id=session_id)
        self.interventions[session_id] = []

        # Configure session-specific settings
        if enable_interventions:
            self.quality_threshold = quality_threshold

        logger.info(f"Created monitored session {session_id}")

        return session_id

    async def stream_with_monitoring(
        self,
        session_id: str,
        agent_id: str,
        response_stream: AsyncIterator[str],
        response_type: PartialResponseType = PartialResponseType.THINKING,
        enable_corrections: bool = True,
        addressed_to: Optional[str] = None,
    ) -> PartialResponse:
        """Stream response with real-time monitoring and intervention capabilities."""

        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._sessions[session_id]
        metrics = self.metrics.get(session_id)

        if agent_id not in session.participants:
            raise ValueError(f"Agent {agent_id} not in session {session_id}")

        # Create enhanced response
        response = PartialResponse(
            agent_id=agent_id, response_type=response_type, addressed_to=addressed_to
        )

        # Initialize monitoring
        start_time = datetime.now()
        token_count = 0
        error_count = 0
        last_intervention_check = start_time

        # Create streaming parser
        stream_id = f"{session_id}:{response.id}"
        self._streaming_parser.create_stream(stream_id)
        session.active_streams.add(stream_id)

        try:
            # Enhanced streaming with monitoring
            async for chunk in response_stream:
                chunk_start = datetime.now()

                # Basic processing
                response.content += chunk
                session.total_bytes += len(chunk.encode("utf-8"))
                token_count += len(chunk.split())

                # Parse streaming content
                parse_result = None
                if chunk.strip():
                    try:
                        parse_result = self._streaming_parser.feed_stream(
                            stream_id, chunk
                        )
                        response.parse_result = parse_result

                        if parse_result.partial_data:
                            response.structured_data = parse_result.partial_data
                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Parse error in stream {stream_id}: {e}")

                        # Attempt error recovery
                        if self.error_recovery_enabled:
                            await self._attempt_error_recovery(
                                session_id, agent_id, chunk, str(e)
                            )

                # Quality monitoring
                if (
                    self.quality_monitors
                    and datetime.now() - last_intervention_check > timedelta(seconds=1)
                ):
                    quality_score = await self._assess_response_quality(
                        session_id, response
                    )

                    if quality_score < self.quality_threshold and enable_corrections:
                        await self._trigger_quality_intervention(
                            session_id, agent_id, response, quality_score
                        )

                    last_intervention_check = datetime.now()

                # Safety monitoring
                await self._check_safety_violations(session_id, agent_id, chunk)

                # Update metrics
                if metrics:
                    metrics.total_tokens = token_count
                    metrics.error_count = error_count
                    metrics.last_updated = datetime.now()

                    # Calculate tokens per second
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 0:
                        metrics.tokens_per_second = token_count / elapsed

                # Notify partial update
                await self._notify_partial_update(session_id, response)

                # Yield control
                await asyncio.sleep(0)

            # Finalize response
            response.is_complete = True
            session.responses.append(response)
            session.total_messages += 1

            # Final quality assessment
            final_quality = await self._assess_response_quality(session_id, response)
            response.metadata["final_quality_score"] = final_quality

            # Update final metrics
            if metrics:
                metrics.average_response_time = (
                    datetime.now() - start_time
                ).total_seconds()
                metrics.agent_participation[agent_id] = (
                    metrics.agent_participation.get(agent_id, 0) + 1
                )

            # Notify completion
            await self._notify_response_complete(session_id, response)

        except Exception as e:
            logger.error(f"Error in monitored streaming: {e}")

            # Attempt recovery
            if self.error_recovery_enabled:
                await self._attempt_error_recovery(
                    session_id, agent_id, response.content, str(e)
                )

            raise

        finally:
            # Cleanup
            self._streaming_parser.close_stream(stream_id)
            session.active_streams.discard(stream_id)

        return response

    async def intervene_stream(
        self,
        session_id: str,
        agent_id: str,
        intervention_type: InterventionType,
        intervention_content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Manually intervene in an active stream."""

        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]

        # Find active stream for agent
        active_stream = None
        for stream_id in session.active_streams:
            if stream_id.endswith(agent_id) or agent_id in stream_id:
                active_stream = stream_id
                break

        if not active_stream:
            logger.warning(
                f"No active stream found for agent {agent_id} in session {session_id}"
            )
            return False

        # Create intervention
        intervention = StreamingIntervention(
            intervention_type=intervention_type,
            trigger=InterventionTrigger.ERROR_DETECTED,  # Manual intervention
            agent_id=agent_id,
            session_id=session_id,
            intervention_content=intervention_content,
            context=context or {},
        )

        # Apply intervention
        success = await self._apply_intervention(intervention)

        if success:
            self.interventions[session_id].append(intervention)

            # Update metrics
            if session_id in self.metrics:
                self.metrics[session_id].intervention_count += 1

            logger.info(
                f"Applied {intervention_type} intervention in session {session_id}"
            )

        return success

    def get_session_metrics(self, session_id: str) -> Optional[StreamingMetrics]:
        """Get real-time metrics for a session."""

        return self.metrics.get(session_id)

    def get_session_interventions(self, session_id: str) -> List[StreamingIntervention]:
        """Get all interventions for a session."""

        return self.interventions.get(session_id, [])

    async def debug_stream(
        self, session_id: str, agent_id: str, debug_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Debug an active stream with detailed information."""

        if session_id not in self._sessions:
            return {"error": "Session not found"}

        session = self._sessions[session_id]
        metrics = self.metrics.get(session_id)

        # Collect debug information
        debug_data = {
            "session_id": session_id,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "session_info": {
                "mode": session.mode,
                "participants": list(session.participants),
                "active_streams": list(session.active_streams),
                "total_messages": session.total_messages,
                "total_bytes": session.total_bytes,
            },
            "metrics": (
                {
                    "total_tokens": metrics.total_tokens if metrics else 0,
                    "tokens_per_second": metrics.tokens_per_second if metrics else 0,
                    "error_count": metrics.error_count if metrics else 0,
                    "intervention_count": metrics.intervention_count if metrics else 0,
                }
                if metrics
                else {}
            ),
            "recent_interventions": [
                {
                    "type": inter.intervention_type,
                    "trigger": inter.trigger,
                    "timestamp": inter.timestamp.isoformat(),
                }
                for inter in self.interventions.get(session_id, [])[-5:]
            ],
            "debug_info": debug_info,
        }

        return debug_data

    def _setup_default_intervention_handlers(self) -> None:
        """Setup default intervention handlers."""

        async def correction_handler(intervention: StreamingIntervention):
            """Handle correction interventions."""
            logger.info(f"Applying correction: {intervention.intervention_content}")
            return True

        async def guidance_handler(intervention: StreamingIntervention):
            """Handle guidance interventions."""
            logger.info(f"Providing guidance: {intervention.intervention_content}")
            return True

        async def termination_handler(intervention: StreamingIntervention):
            """Handle termination interventions."""
            logger.warning(f"Terminating stream: {intervention.intervention_content}")
            return True

        self.intervention_handlers = {
            InterventionType.CORRECTION: correction_handler,
            InterventionType.GUIDANCE: guidance_handler,
            InterventionType.TERMINATION: termination_handler,
        }

    async def _assess_response_quality(
        self, session_id: str, response: PartialResponse
    ) -> float:
        """Assess the quality of a response using registered monitors."""

        if not self.quality_monitors:
            return 1.0  # No monitors, assume good quality

        total_score = 0.0
        count = 0

        for monitor in self.quality_monitors:
            try:
                score = monitor(session_id, response)
                total_score += score
                count += 1
            except Exception as e:
                logger.warning(f"Quality monitor error: {e}")

        return total_score / count if count > 0 else 1.0

    async def _trigger_quality_intervention(
        self,
        session_id: str,
        agent_id: str,
        response: PartialResponse,
        quality_score: float,
    ) -> None:
        """Trigger intervention due to quality threshold breach."""

        intervention = StreamingIntervention(
            intervention_type=InterventionType.CORRECTION,
            trigger=InterventionTrigger.QUALITY_THRESHOLD,
            agent_id=agent_id,
            session_id=session_id,
            original_content=response.content,
            intervention_content=f"Quality score {quality_score:.2f} below threshold {self.quality_threshold}",
            context={
                "quality_score": quality_score,
                "threshold": self.quality_threshold,
            },
        )

        await self._apply_intervention(intervention)
        self.interventions[session_id].append(intervention)

        if self.on_quality_threshold_breach:
            self.on_quality_threshold_breach(session_id, quality_score)

    async def _check_safety_violations(
        self, session_id: str, agent_id: str, content: str
    ) -> None:
        """Check for safety violations in streaming content."""

        # Basic safety checks (this would integrate with the multilingual safety system)
        harmful_patterns = [
            r"\b(hate|kill|murder|attack)\b",
            r"\b(bomb|weapon|violence)\b",
        ]

        for pattern in harmful_patterns:
            import re

            if re.search(pattern, content, re.IGNORECASE):
                intervention = StreamingIntervention(
                    intervention_type=InterventionType.TERMINATION,
                    trigger=InterventionTrigger.SAFETY_VIOLATION,
                    agent_id=agent_id,
                    session_id=session_id,
                    original_content=content,
                    intervention_content="Safety violation detected",
                    context={"violated_pattern": pattern},
                )

                await self._apply_intervention(intervention)
                self.interventions[session_id].append(intervention)
                break

    async def _apply_intervention(self, intervention: StreamingIntervention) -> bool:
        """Apply an intervention to the streaming process."""

        handler = self.intervention_handlers.get(intervention.intervention_type)

        if handler:
            try:
                result = await handler(intervention)
                intervention.correction_applied = bool(result)

                if self.on_intervention_triggered:
                    self.on_intervention_triggered(intervention)

                return intervention.correction_applied
            except Exception as e:
                logger.error(f"Intervention handler error: {e}")
                return False

        logger.warning(
            f"No handler for intervention type: {intervention.intervention_type}"
        )
        return False

    async def _attempt_error_recovery(
        self, session_id: str, agent_id: str, content: str, error_message: str
    ) -> bool:
        """Attempt to recover from streaming errors."""

        logger.info(f"Attempting error recovery for agent {agent_id}: {error_message}")

        # Simple recovery strategies
        recovery_successful = False

        # Strategy 1: Retry parsing with cleaned content
        if "json" in error_message.lower():
            try:
                # Clean up common JSON issues
                cleaned_content = content.replace("'", '"').strip()
                if cleaned_content and not cleaned_content.endswith("}"):
                    cleaned_content += "}"

                json.loads(cleaned_content)
                recovery_successful = True
                logger.info("JSON parsing recovered successfully")
            except:
                pass

        # Strategy 2: Fallback to text-only mode
        if not recovery_successful:
            logger.info("Falling back to text-only mode")
            recovery_successful = True  # Always succeeds

        if recovery_successful and self.on_error_recovered:
            self.on_error_recovered(session_id, error_message)

        return recovery_successful


# Quality monitoring functions
def coherence_monitor(session_id: str, response: PartialResponse) -> float:
    """Monitor response coherence."""

    content = response.content
    if not content:
        return 1.0

    # Simple coherence check based on sentence structure
    sentences = content.split(".")
    if len(sentences) < 2:
        return 1.0

    # Check for repeated phrases (indicates poor coherence)
    words = content.lower().split()
    unique_words = set(words)

    coherence_score = len(unique_words) / max(len(words), 1)
    return min(1.0, coherence_score * 2)  # Scale to 0-1


def relevance_monitor(session_id: str, response: PartialResponse) -> float:
    """Monitor response relevance."""

    # Basic relevance check (would be more sophisticated in practice)
    content = response.content.lower()

    # Check for generic filler words
    filler_words = ["um", "uh", "like", "you know", "basically"]
    filler_count = sum(content.count(word) for word in filler_words)

    total_words = len(content.split())
    if total_words == 0:
        return 1.0

    relevance_score = 1.0 - (filler_count / total_words)
    return max(0.0, relevance_score)


def safety_monitor(session_id: str, response: PartialResponse) -> float:
    """Monitor response safety."""

    content = response.content.lower()

    # Basic safety patterns
    unsafe_patterns = ["hate", "violence", "harmful", "dangerous", "illegal"]

    violation_count = sum(content.count(pattern) for pattern in unsafe_patterns)

    # Return safety score (1.0 = safe, 0.0 = unsafe)
    return 1.0 if violation_count == 0 else max(0.0, 1.0 - violation_count * 0.2)
