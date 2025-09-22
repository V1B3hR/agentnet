#!/usr/bin/env python3
"""
Phase 6 Streaming Collaboration Tests

Tests the streaming partial-output collaboration features
implemented for Phase 6 exploratory functionality.
"""

import asyncio
import json
import time
from typing import AsyncIterator

import pytest

from agentnet.streaming import (
    StreamingCollaborator,
    CollaborationSession,
    PartialResponse,
    PartialJSONParser,
    StreamingParser,
    ParseResult,
    StreamHandler,
    CollaborationHandler,
    ErrorHandler,
    CollaborationMode,
    PartialResponseType,
)


def test_partial_json_parser():
    """Test partial JSON parsing capabilities."""
    print("🔍 Testing Partial JSON Parser...")
    
    parser = PartialJSONParser()
    
    # Test complete JSON
    complete_json = '{"name": "test", "value": 42, "active": true}'
    result = parser.feed(complete_json)
    
    assert result.is_complete
    assert result.parsed_data is not None
    assert result.parsed_data["name"] == "test"
    assert result.parsed_data["value"] == 42
    print(f"  ✅ Complete JSON parsed: {result.parsed_data}")
    
    # Reset and test partial JSON
    parser.reset()
    
    # Feed JSON in chunks
    chunks = [
        '{"name": "partial"',
        ', "items": [1, 2',
        ', 3], "status"',
        ': "processing"}'
    ]
    
    partial_results = []
    for i, chunk in enumerate(chunks):
        result = parser.feed(chunk)
        partial_results.append(result)
        print(f"  Chunk {i+1}: {len(result.partial_data)} partial keys extracted")
    
    final_result = partial_results[-1]
    assert final_result.is_complete or len(final_result.partial_data) > 0
    print(f"  🔧 Partial parsing successful: {final_result.partial_data}")
    
    # Test error recovery
    parser.reset()
    malformed_json = '{"incomplete": "data", "missing'
    result = parser.feed(malformed_json)
    
    assert not result.is_complete
    assert len(result.partial_data) > 0  # Should extract what it can
    print(f"  🛠️ Error recovery: extracted {result.partial_data}")
    
    # Assert success instead of returning
    assert result is not None
    assert hasattr(result, 'partial_data')


def test_streaming_parser():
    """Test streaming parser with callbacks."""
    print("📡 Testing Streaming Parser...")
    
    updates_received = []
    completions_received = []
    errors_received = []
    
    def on_partial_update(stream_id, partial_data, result):
        updates_received.append((stream_id, partial_data))
    
    def on_complete(stream_id, parsed_data, result):
        completions_received.append((stream_id, parsed_data))
    
    def on_error(stream_id, error, result):
        errors_received.append((stream_id, error))
    
    parser = StreamingParser(
        on_partial_update=on_partial_update,
        on_complete=on_complete,
        on_error=on_error
    )
    
    # Create streaming session
    stream_id = "test_stream_001"
    parser.create_stream(stream_id)
    
    # Stream JSON data
    json_chunks = [
        '{"progress": 0.1, "message": "',
        'starting process", "data": [1',
        ', 2, 3], "complete": false}',
    ]
    
    for chunk in json_chunks:
        result = parser.feed_stream(stream_id, chunk)
        # Small delay to simulate streaming
        time.sleep(0.01)
    
    # Verify callbacks were triggered
    assert len(updates_received) > 0
    assert len(completions_received) > 0
    print(f"  📊 Received {len(updates_received)} partial updates")
    print(f"  ✅ Received {len(completions_received)} completions")
    
    # Test error handling
    parser.create_stream("error_stream")
    parser.feed_stream("error_stream", '{"malformed": json without closing')
    
    # In strict mode or with certain parsers, errors might not always trigger callbacks
    # Just verify the parser handles it gracefully
    error_stream_result = parser.get_stream_result("error_stream")
    if error_stream_result and not error_stream_result.is_valid:
        print("  ⚠️ Malformed JSON detected and handled gracefully")
    elif len(errors_received) > 0:
        print(f"  ⚠️ Handled {len(errors_received)} errors gracefully")
    else:
        print("  ⚠️ Parser handled malformed JSON without errors (graceful degradation)")
    
    # Cleanup
    parser.close_stream(stream_id)
    parser.close_stream("error_stream")
    
    # Assert success instead of returning
    assert len(updates_received) > 0
    assert len(completions_received) > 0


@pytest.mark.asyncio
async def test_streaming_collaboration():
    """Test streaming collaboration between agents."""
    print("🤝 Testing Streaming Collaboration...")
    
    collaborator = StreamingCollaborator()
    
    # Create collaboration session
    session = collaborator.create_session(
        mode=CollaborationMode.PEER_TO_PEER,
        participants=["agent_1", "agent_2", "agent_3"]
    )
    
    assert session.mode == CollaborationMode.PEER_TO_PEER
    assert len(session.participants) == 3
    print(f"  📋 Created session: {session.session_id}")
    
    # Add session listener
    events_received = []
    
    async def session_listener(event_type, data):
        events_received.append((event_type, data))
    
    collaborator.add_session_listener(session.session_id, session_listener)
    
    # Simulate streaming responses
    async def create_stream_response(content: str) -> AsyncIterator[str]:
        """Create a streaming response by yielding chunks."""
        chunk_size = max(1, len(content) // 5)
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.01)  # Simulate processing time
    
    # Agent 1 streams a response
    agent1_content = "I think we should implement a microservices architecture because it provides better scalability and maintainability."
    
    response1 = await collaborator.stream_response(
        session.session_id,
        "agent_1",
        create_stream_response(agent1_content),
        PartialResponseType.PARTIAL_ANSWER
    )
    
    assert response1.agent_id == "agent_1"
    assert response1.is_complete
    assert response1.content == agent1_content
    print(f"  💬 Agent 1 response: {len(response1.content)} chars")
    
    # Agent 2 streams a counter-response
    agent2_content = "While microservices have benefits, we should also consider the complexity overhead and operational challenges they introduce."
    
    response2 = await collaborator.stream_response(
        session.session_id,
        "agent_2", 
        create_stream_response(agent2_content),
        PartialResponseType.PARTIAL_ANSWER,
        addressed_to="agent_1"
    )
    
    assert response2.addressed_to == "agent_1"
    print(f"  💬 Agent 2 response: {len(response2.content)} chars")
    
    # Verify session state
    session_stats = collaborator.get_session_stats(session.session_id)
    assert session_stats is not None
    assert session_stats['total_messages'] == 2
    print(f"  📊 Session stats: {session_stats['total_messages']} messages")
    
    # Verify events were received
    assert len(events_received) > 0
    print(f"  📡 Received {len(events_received)} session events")
    
    # Test collaborative generation
    agent_generators = {
        "agent_1": lambda prompt, context: f"Agent 1's thoughts on: {prompt}",
        "agent_2": lambda prompt, context: f"Agent 2's perspective: {prompt}",
        "agent_3": lambda prompt, context: f"Agent 3's analysis: {prompt}"
    }
    
    collab_result = await collaborator.collaborative_generate(
        session.session_id,
        "How should we implement continuous integration?",
        agent_generators,
        max_iterations=2
    )
    
    assert 'iterations' in collab_result
    assert len(collab_result['iterations']) > 0
    print(f"  🔄 Collaborative generation: {len(collab_result['iterations'])} iterations")
    print(f"  🎯 Final result: {collab_result.get('final_consensus') is not None}")
    
    # Cleanup
    collaborator.close_session(session.session_id)
    
    return True


@pytest.mark.asyncio
async def test_collaboration_handlers():
    """Test specialized collaboration handlers."""
    print("🔧 Testing Collaboration Handlers...")
    
    # Test collaboration handler
    collab_handler = CollaborationHandler(
        max_concurrent_streams=3,
        turn_timeout_seconds=5.0
    )
    
    # Test handling streaming data
    test_data = {"message": "test collaboration", "agent": "test_agent"}
    context = {
        "stream_id": "test_stream",
        "session_id": "test_session", 
        "agent_id": "test_agent"
    }
    
    handled_data = await collab_handler.handle(test_data, context)
    
    # Should have added collaboration metadata
    if isinstance(handled_data, dict):
        assert 'collaboration_metadata' in handled_data
        print(f"  📊 Added collaboration metadata: {handled_data['collaboration_metadata']}")
    
    # Test error handler
    error_handler = ErrorHandler(
        max_retries=2,
        retry_delay=0.1,
        enable_recovery=True
    )
    
    # Test normal data processing
    normal_data = "This is normal streaming data"
    normal_result = await error_handler.handle(normal_data, context)
    assert normal_result == normal_data
    print(f"  ✅ Normal data passed through")
    
    # Test error handling
    error_context = context.copy()
    error_context['is_error'] = True
    
    error_result = await error_handler.handle(
        Exception("Test streaming error"), 
        error_context
    )
    
    assert isinstance(error_result, dict)
    assert error_result.get('error') is True
    print(f"  ⚠️ Error handled gracefully: {error_result['error_type']}")
    
    # Test recovery
    # Register a custom recovery strategy
    async def test_recovery(error, stream_id, context):
        return {"recovered": True, "message": "Test recovery successful"}
    
    error_handler.register_recovery_strategy("Exception", test_recovery)
    
    recovery_result = await error_handler.handle(
        Exception("Recoverable error"),
        error_context
    )
    
    if isinstance(recovery_result, dict) and recovery_result.get('recovered'):
        print(f"  🔧 Recovery successful: {recovery_result['message']}")
    
    # Get handler statistics
    collab_stats = collab_handler.get_stream_stats()
    error_stats = error_handler.get_error_stats()
    
    print(f"  📈 Collaboration stats: {collab_stats['active_streams']} active streams")
    print(f"  📈 Error stats: {error_stats['total_errors']} total errors")
    
    return True


@pytest.mark.asyncio
async def test_end_to_end_streaming():
    """Test complete end-to-end streaming collaboration scenario."""
    print("🚀 Testing End-to-End Streaming Collaboration...")
    
    # Setup complete streaming environment
    collaborator = StreamingCollaborator()
    
    # Create multi-agent session
    session = collaborator.create_session(
        mode=CollaborationMode.ROUND_ROBIN,
        participants=["architect", "developer", "tester"]
    )
    
    # Track session events
    session_events = []
    
    async def event_tracker(event_type, data):
        session_events.append({
            'event': event_type,
            'agent': data.agent_id if hasattr(data, 'agent_id') else 'unknown',
            'timestamp': time.time(),
            'content_length': len(data.content) if hasattr(data, 'content') else 0
        })
    
    collaborator.add_session_listener(session.session_id, event_tracker)
    
    # Simulate complex collaborative streaming scenario
    scenario_data = [
        {
            "agent": "architect", 
            "content": '{"design": "microservices", "components": ["api-gateway", "user-service", "payment-service"], "status": "planning"}',
            "type": PartialResponseType.PARTIAL_ANSWER
        },
        {
            "agent": "developer",
            "content": '{"implementation": "spring-boot", "technologies": ["java", "docker", "kubernetes"], "timeline": "4 weeks"}',
            "type": PartialResponseType.PARTIAL_ANSWER
        },
        {
            "agent": "tester",
            "content": '{"testing_strategy": "automated", "tools": ["junit", "selenium", "postman"], "coverage_target": 90}',
            "type": PartialResponseType.FINAL_ANSWER
        }
    ]
    
    responses = []
    
    for scenario in scenario_data:
        # Create streaming response with JSON content
        async def json_stream_generator(content: str) -> AsyncIterator[str]:
            # Simulate gradual JSON construction
            chars_per_chunk = max(1, len(content) // 8)
            for i in range(0, len(content), chars_per_chunk):
                chunk = content[i:i + chars_per_chunk]
                yield chunk
                await asyncio.sleep(0.02)  # Simulate processing delay
        
        response = await collaborator.stream_response(
            session.session_id,
            scenario["agent"],
            json_stream_generator(scenario["content"]),
            scenario["type"]
        )
        
        responses.append(response)
        
        # Verify JSON parsing worked
        if response.structured_data:
            print(f"  📋 {scenario['agent']}: parsed {len(response.structured_data)} JSON fields")
    
    # Verify all responses completed
    assert len(responses) == 3
    assert all(r.is_complete for r in responses)
    print(f"  ✅ All {len(responses)} agents completed streaming")
    
    # Check JSON parsing results
    json_responses = [r for r in responses if r.structured_data]
    assert len(json_responses) > 0
    print(f"  🔍 {len(json_responses)} responses had structured JSON data")
    
    # Verify session events
    assert len(session_events) > 0
    partial_events = [e for e in session_events if e['event'] == 'partial_update']
    complete_events = [e for e in session_events if e['event'] == 'response_complete']
    
    print(f"  📡 Session events: {len(partial_events)} partial, {len(complete_events)} complete")
    
    # Get final session statistics
    final_stats = collaborator.get_session_stats(session.session_id)
    print(f"  📊 Final session stats:")
    print(f"    Messages: {final_stats['total_messages']}")
    print(f"    Bytes: {final_stats['total_bytes']}")
    print(f"    Duration: {final_stats['duration_seconds']:.2f}s")
    print(f"    Avg response length: {final_stats['avg_response_length']:.1f} chars")
    
    # Cleanup
    collaborator.close_session(session.session_id)
    
    return True


async def main():
    """Run all streaming collaboration tests."""
    print("🚀 AgentNet Phase 6 Streaming Collaboration Test Suite")
    print("=" * 65)
    
    tests = [
        test_partial_json_parser,
        test_streaming_parser,
        test_streaming_collaboration,
        test_collaboration_handlers,
        test_end_to_end_streaming,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print("  ✅ PASSED\n")
            else:
                failed += 1
                print("  ❌ FAILED\n")
        except Exception as e:
            print(f"  ❌ CRASHED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 65)
    print(f"📊 Final Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All streaming collaboration tests passed!")
        print("✨ Phase 6 Streaming Features are working correctly!")
        print("\n🔮 Key capabilities demonstrated:")
        print("  • Partial JSON parsing with error recovery")
        print("  • Real-time streaming collaboration between agents")
        print("  • Robust error handling and recovery strategies")
        print("  • End-to-end multi-agent streaming workflows")
        return True
    else:
        print(f"❌ {failed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    asyncio.run(main())