"""
Streaming Collaboration Framework for AgentNet

Enables real-time collaborative work between agents with streaming
partial outputs and incremental processing capabilities.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Set
import logging

from .parser import StreamingParser, ParseResult

logger = logging.getLogger(__name__)


class CollaborationMode(str, Enum):
    """Types of collaborative interactions."""
    PEER_TO_PEER = "peer_to_peer"
    LEADER_FOLLOWER = "leader_follower"
    ROUND_ROBIN = "round_robin"
    BROADCAST = "broadcast"
    PIPELINE = "pipeline"


class PartialResponseType(str, Enum):
    """Types of partial responses."""
    THINKING = "thinking"
    PARTIAL_ANSWER = "partial_answer"
    QUESTION = "question"
    SUGGESTION = "suggestion"
    CORRECTION = "correction"
    FINAL_ANSWER = "final_answer"


@dataclass
class PartialResponse:
    """A partial response in a streaming collaboration."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    response_type: PartialResponseType = PartialResponseType.THINKING
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Streaming information
    timestamp: float = field(default_factory=time.time)
    is_complete: bool = False
    sequence_number: int = 0
    parent_id: Optional[str] = None
    
    # Collaboration context
    addressed_to: Optional[str] = None  # Specific agent or None for all
    requires_response: bool = False
    confidence: float = 1.0
    
    # JSON data (if applicable)
    structured_data: Optional[Dict[str, Any]] = None
    parse_result: Optional[ParseResult] = None


@dataclass
class CollaborationSession:
    """A session for streaming multi-agent collaboration."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: CollaborationMode = CollaborationMode.PEER_TO_PEER
    participants: List[str] = field(default_factory=list)
    
    # Session state
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    current_turn: Optional[str] = None
    
    # Configuration
    max_participants: int = 10
    turn_timeout_seconds: float = 30.0
    allow_interruptions: bool = True
    require_consensus: bool = False
    
    # Session data
    responses: List[PartialResponse] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    total_messages: int = 0
    total_bytes: int = 0
    active_streams: Set[str] = field(default_factory=set)


class StreamingCollaborator:
    """
    Manages streaming collaboration between multiple agents.
    
    Handles real-time partial output sharing, incremental processing,
    and coordination of multi-agent streaming workflows.
    """
    
    def __init__(self):
        self._sessions: Dict[str, CollaborationSession] = {}
        self._streaming_parser = StreamingParser(
            on_partial_update=self._handle_partial_update,
            on_complete=self._handle_complete_response,
            on_error=self._handle_parse_error
        )
        self._response_handlers: Dict[str, Callable] = {}
        self._session_listeners: Dict[str, List[Callable]] = {}
    
    def create_session(self, 
                      mode: CollaborationMode = CollaborationMode.PEER_TO_PEER,
                      participants: Optional[List[str]] = None,
                      **kwargs) -> CollaborationSession:
        """Create a new collaboration session."""
        
        session = CollaborationSession(
            mode=mode,
            participants=participants or [],
            **kwargs
        )
        
        self._sessions[session.session_id] = session
        self._session_listeners[session.session_id] = []
        
        logger.info(f"Created collaboration session {session.session_id} with mode {mode.value}")
        return session
    
    def join_session(self, session_id: str, agent_id: str) -> bool:
        """Add an agent to a collaboration session."""
        
        if session_id not in self._sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self._sessions[session_id]
        
        if len(session.participants) >= session.max_participants:
            logger.error(f"Session {session_id} is full")
            return False
        
        if agent_id not in session.participants:
            session.participants.append(agent_id)
            logger.info(f"Agent {agent_id} joined session {session_id}")
        
        return True
    
    def leave_session(self, session_id: str, agent_id: str) -> bool:
        """Remove an agent from a collaboration session."""
        
        if session_id not in self._sessions:
            return False
        
        session = self._sessions[session_id]
        
        if agent_id in session.participants:
            session.participants.remove(agent_id)
            logger.info(f"Agent {agent_id} left session {session_id}")
        
        # Close session if no participants remain
        if not session.participants:
            self.close_session(session_id)
        
        return True
    
    async def stream_response(self, 
                             session_id: str,
                             agent_id: str,
                             response_stream: AsyncIterator[str],
                             response_type: PartialResponseType = PartialResponseType.THINKING,
                             addressed_to: Optional[str] = None) -> PartialResponse:
        """Stream a response from an agent to the collaboration session."""
        
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self._sessions[session_id]
        
        if agent_id not in session.participants:
            raise ValueError(f"Agent {agent_id} not in session {session_id}")
        
        # Create partial response
        response = PartialResponse(
            agent_id=agent_id,
            response_type=response_type,
            addressed_to=addressed_to
        )
        
        # Create streaming parser for this response
        stream_id = f"{session_id}:{response.id}"
        self._streaming_parser.create_stream(stream_id)
        session.active_streams.add(stream_id)
        
        try:
            # Stream the response
            async for chunk in response_stream:
                response.content += chunk
                session.total_bytes += len(chunk.encode('utf-8'))
                
                # Parse streaming JSON if applicable
                if chunk.strip():
                    parse_result = self._streaming_parser.feed_stream(stream_id, chunk)
                    response.parse_result = parse_result
                    
                    if parse_result.partial_data:
                        response.structured_data = parse_result.partial_data
                
                # Notify listeners of partial update
                await self._notify_partial_update(session_id, response)
                
                # Yield control for other coroutines
                await asyncio.sleep(0)
            
            # Mark response as complete
            response.is_complete = True
            session.responses.append(response)
            session.total_messages += 1
            
            # Notify completion
            await self._notify_response_complete(session_id, response)
            
        finally:
            # Cleanup streaming parser
            self._streaming_parser.close_stream(stream_id)
            session.active_streams.discard(stream_id)
        
        return response
    
    async def collaborative_generate(self,
                                   session_id: str,
                                   prompt: str,
                                   agent_generators: Dict[str, Callable],
                                   max_iterations: int = 10,
                                   consensus_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Generate collaborative response using streaming from multiple agents.
        
        Args:
            session_id: Collaboration session ID
            prompt: Initial prompt for generation
            agent_generators: Dict mapping agent_id to generator functions
            max_iterations: Maximum collaboration iterations
            consensus_threshold: Threshold for reaching consensus
            
        Returns:
            Collaborative result with all agent contributions
        """
        
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        results = {
            'prompt': prompt,
            'iterations': [],
            'final_consensus': None,
            'all_responses': [],
            'session_stats': {}
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Collaboration iteration {iteration + 1}")
            
            iteration_results = {
                'iteration': iteration + 1,
                'responses': [],
                'consensus_score': 0.0
            }
            
            # Collect responses from all agents
            tasks = []
            for agent_id in session.participants:
                if agent_id in agent_generators:
                    generator_func = agent_generators[agent_id]
                    
                    # Create async generator for streaming
                    async def agent_stream():
                        # Get agent's streaming response
                        response = await generator_func(prompt, session.shared_context)
                        
                        # Simulate streaming by yielding chunks
                        if isinstance(response, str):
                            # Break response into chunks for streaming
                            chunk_size = max(1, len(response) // 10)
                            for i in range(0, len(response), chunk_size):
                                yield response[i:i + chunk_size]
                                await asyncio.sleep(0.01)  # Simulate processing time
                        else:
                            yield str(response)
                    
                    # Start streaming response
                    task = self.stream_response(
                        session_id,
                        agent_id,
                        agent_stream(),
                        PartialResponseType.PARTIAL_ANSWER
                    )
                    tasks.append(task)
            
            # Wait for all agents to complete their responses
            agent_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            valid_responses = []
            for response in agent_responses:
                if isinstance(response, PartialResponse):
                    valid_responses.append(response)
                    iteration_results['responses'].append({
                        'agent_id': response.agent_id,
                        'content': response.content,
                        'confidence': response.confidence,
                        'structured_data': response.structured_data
                    })
            
            # Calculate consensus (simplified)
            if len(valid_responses) > 1:
                consensus_score = self._calculate_consensus(valid_responses)
                iteration_results['consensus_score'] = consensus_score
                
                if consensus_score >= consensus_threshold:
                    results['final_consensus'] = self._merge_responses(valid_responses)
                    results['iterations'].append(iteration_results)
                    break
            
            results['iterations'].append(iteration_results)
            results['all_responses'].extend(valid_responses)
            
            # Update shared context for next iteration
            session.shared_context['previous_responses'] = valid_responses
            session.shared_context['iteration'] = iteration + 1
        
        # Add session statistics
        results['session_stats'] = {
            'total_messages': session.total_messages,
            'total_bytes': session.total_bytes,
            'active_streams': len(session.active_streams),
            'participants': len(session.participants)
        }
        
        return results
    
    def _calculate_consensus(self, responses: List[PartialResponse]) -> float:
        """Calculate consensus score among responses (simplified implementation)."""
        
        if len(responses) < 2:
            return 1.0
        
        # Simple consensus based on response similarity and confidence
        total_confidence = sum(r.confidence for r in responses)
        avg_confidence = total_confidence / len(responses)
        
        # Check for similar keywords (very simplified)
        all_words = set()
        for response in responses:
            words = set(response.content.lower().split())
            all_words.update(words)
        
        common_words = set()
        for word in all_words:
            count = sum(1 for r in responses if word in r.content.lower())
            if count >= len(responses) * 0.6:  # 60% of responses contain the word
                common_words.add(word)
        
        word_similarity = len(common_words) / max(len(all_words), 1)
        
        # Combine confidence and similarity
        return (avg_confidence * 0.7 + word_similarity * 0.3)
    
    def _merge_responses(self, responses: List[PartialResponse]) -> Dict[str, Any]:
        """Merge multiple responses into consensus result."""
        
        merged = {
            'content': "",
            'contributors': [],
            'confidence': 0.0,
            'structured_data': {}
        }
        
        # Combine content and metadata
        contents = []
        total_confidence = 0.0
        
        for response in responses:
            contents.append(f"[{response.agent_id}]: {response.content}")
            merged['contributors'].append(response.agent_id)
            total_confidence += response.confidence
            
            # Merge structured data
            if response.structured_data:
                for key, value in response.structured_data.items():
                    if key not in merged['structured_data']:
                        merged['structured_data'][key] = []
                    merged['structured_data'][key].append(value)
        
        merged['content'] = "\n\n".join(contents)
        merged['confidence'] = total_confidence / len(responses)
        
        return merged
    
    async def _notify_partial_update(self, session_id: str, response: PartialResponse) -> None:
        """Notify listeners of partial response updates."""
        
        if session_id in self._session_listeners:
            for listener in self._session_listeners[session_id]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener('partial_update', response)
                    else:
                        listener('partial_update', response)
                except Exception as e:
                    logger.error(f"Error in session listener: {e}")
    
    async def _notify_response_complete(self, session_id: str, response: PartialResponse) -> None:
        """Notify listeners of complete responses."""
        
        if session_id in self._session_listeners:
            for listener in self._session_listeners[session_id]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener('response_complete', response)
                    else:
                        listener('response_complete', response)
                except Exception as e:
                    logger.error(f"Error in session listener: {e}")
    
    def _handle_partial_update(self, stream_id: str, partial_data: Dict[str, Any], result: ParseResult) -> None:
        """Handle partial JSON parsing updates."""
        
        # Extract session info from stream_id
        if ':' in stream_id:
            session_id, response_id = stream_id.split(':', 1)
            
            if session_id in self._sessions:
                session = self._sessions[session_id]
                
                # Find the corresponding response
                for response in session.responses:
                    if response.id == response_id:
                        response.structured_data = partial_data
                        response.parse_result = result
                        break
    
    def _handle_complete_response(self, stream_id: str, parsed_data: Dict[str, Any], result: ParseResult) -> None:
        """Handle complete JSON parsing."""
        
        if ':' in stream_id:
            session_id, response_id = stream_id.split(':', 1)
            
            if session_id in self._sessions:
                session = self._sessions[session_id]
                
                for response in session.responses:
                    if response.id == response_id:
                        response.structured_data = parsed_data
                        response.parse_result = result
                        response.is_complete = True
                        break
    
    def _handle_parse_error(self, stream_id: str, error: str, result: ParseResult) -> None:
        """Handle JSON parsing errors."""
        
        logger.warning(f"Parsing error in stream {stream_id}: {error}")
        
        # Continue processing with partial data if available
        if result.partial_data:
            self._handle_partial_update(stream_id, result.partial_data, result)
    
    def add_session_listener(self, session_id: str, listener: Callable) -> None:
        """Add a listener for session events."""
        
        if session_id not in self._session_listeners:
            self._session_listeners[session_id] = []
        
        self._session_listeners[session_id].append(listener)
    
    def remove_session_listener(self, session_id: str, listener: Callable) -> None:
        """Remove a session listener."""
        
        if session_id in self._session_listeners:
            try:
                self._session_listeners[session_id].remove(listener)
            except ValueError:
                pass
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session by ID."""
        return self._sessions.get(session_id)
    
    def get_active_sessions(self) -> List[CollaborationSession]:
        """Get all active collaboration sessions."""
        return [s for s in self._sessions.values() if s.is_active]
    
    def close_session(self, session_id: str) -> bool:
        """Close a collaboration session."""
        
        if session_id not in self._sessions:
            return False
        
        session = self._sessions[session_id]
        session.is_active = False
        
        # Cleanup streaming parsers
        for stream_id in list(session.active_streams):
            self._streaming_parser.close_stream(stream_id)
        
        session.active_streams.clear()
        
        # Remove from active sessions
        del self._sessions[session_id]
        if session_id in self._session_listeners:
            del self._session_listeners[session_id]
        
        logger.info(f"Closed collaboration session {session_id}")
        return True
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a collaboration session."""
        
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'mode': session.mode.value,
            'participants': len(session.participants),
            'total_messages': session.total_messages,
            'total_bytes': session.total_bytes,
            'active_streams': len(session.active_streams),
            'created_at': session.created_at,
            'duration_seconds': time.time() - session.created_at,
            'responses_count': len(session.responses),
            'avg_response_length': sum(len(r.content) for r in session.responses) / max(len(session.responses), 1)
        }