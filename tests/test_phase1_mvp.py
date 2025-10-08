"""
Test Phase 1 MVP implementation.

Tests the turn engine, policy engine, and event system integration.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from agentnet.core.orchestration.turn_engine import (
    TurnEngine,
    TurnMode,
    TerminationReason,
)
from agentnet.core.policy.engine import PolicyEngine, PolicyAction
from agentnet.core.policy.rules import (
    create_keyword_rule,
    create_confidence_rule,
    Severity,
)
from agentnet.events.bus import EventBus, EventType
from agentnet.events.sinks import ConsoleSink, FileSink


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, responses: list = None):
        self.name = name
        self.responses = responses or ["This is a test response."]
        self.call_count = 0

    async def async_generate_reasoning_tree(self, prompt: str):
        """Mock async method."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return {
            "result": {"content": response, "confidence": 0.8},
            "metadata": {"mock": True},
        }

    def generate_reasoning_tree(self, prompt: str):
        """Mock sync method."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return {
            "result": {"content": response, "confidence": 0.8},
            "metadata": {"mock": True},
        }


@pytest.mark.asyncio
async def test_turn_engine_single_agent():
    """Test single agent turn engine."""
    # Create mock agent
    agent = MockAgent(
        "TestAgent",
        [
            "Initial response about the topic.",
            "Elaborating on my previous point.",
            "Final thoughts and conclusions.",
        ],
    )

    # Create turn engine
    engine = TurnEngine(max_turns=3, max_rounds=2)

    # Execute single agent session
    result = await engine.execute_single_agent_session(
        agent=agent,
        initial_prompt="Discuss artificial intelligence",
        context={"test": True},
    )

    # Verify results
    assert result.session_id.startswith("single_")
    assert result.mode == TurnMode.SINGLE_AGENT
    assert result.status in [
        TerminationReason.COMPLETED,
        TerminationReason.MAX_TURNS_REACHED,
    ]
    assert len(result.turns) <= 3
    assert result.agents_involved == ["TestAgent"]
    assert result.duration is not None

    # Check turn details
    for turn in result.turns:
        assert turn.agent_id == "TestAgent"
        assert (
            turn.content.startswith("Initial")
            or "point" in turn.content
            or "Final" in turn.content
        )
        assert turn.confidence == 0.8


@pytest.mark.asyncio
async def test_turn_engine_multi_agent_round_robin():
    """Test multi-agent round robin mode."""
    # Create mock agents
    agents = [
        MockAgent("Agent1", ["Agent1 perspective on this topic."]),
        MockAgent("Agent2", ["Agent2 has a different view."]),
        MockAgent("Agent3", ["Agent3 adds final thoughts."]),
    ]

    # Create turn engine
    engine = TurnEngine(max_turns=6, max_rounds=2)

    # Execute multi-agent session
    result = await engine.execute_multi_agent_session(
        agents=agents,
        topic="Collaborative problem solving",
        mode=TurnMode.ROUND_ROBIN,
        context={"test": True},
    )

    # Verify results
    assert result.session_id.startswith("multi_")
    assert result.mode == TurnMode.ROUND_ROBIN
    assert result.status in [
        TerminationReason.COMPLETED,
        TerminationReason.MAX_TURNS_REACHED,
    ]
    assert len(result.turns) <= 6
    assert set(result.agents_involved) == {"Agent1", "Agent2", "Agent3"}

    # Check that all agents participated
    agent_turns = {}
    for turn in result.turns:
        agent_turns[turn.agent_id] = agent_turns.get(turn.agent_id, 0) + 1

    assert len(agent_turns) == 3  # All agents should have turns


@pytest.mark.asyncio
async def test_turn_engine_debate_mode():
    """Test debate mode with agents taking opposing positions."""
    agents = [
        MockAgent("ProAgent", ["I strongly support this position because..."]),
        MockAgent("ConAgent", ["I disagree with that view because..."]),
    ]

    engine = TurnEngine(max_turns=4, max_rounds=2)

    result = await engine.execute_multi_agent_session(
        agents=agents,
        topic="Should AI be regulated?",
        mode=TurnMode.DEBATE,
        context={"test": True},
    )

    assert result.mode == TurnMode.DEBATE
    assert len(result.turns) <= 4

    # Check alternating agents
    if len(result.turns) >= 2:
        assert result.turns[0].agent_id != result.turns[1].agent_id


def test_policy_engine_basic():
    """Test basic policy engine functionality."""
    # Create policy engine with rules
    policy = PolicyEngine(name="test_policy")

    # Add some rules
    keyword_rule = create_keyword_rule(
        name="no_bad_words",
        keywords=["bad", "terrible", "awful"],
        severity=Severity.MAJOR,
        description="Prohibit negative words",
    )

    confidence_rule = create_confidence_rule(
        name="min_confidence",
        min_confidence=0.5,
        severity=Severity.MINOR,
        description="Require minimum confidence",
    )

    policy.add_rule(keyword_rule)
    policy.add_rule(confidence_rule)

    # Test clean content
    clean_context = {
        "content": "This is a good response with positive words.",
        "confidence": 0.8,
        "agent_name": "TestAgent",
    }

    result = policy.evaluate(clean_context)
    assert result.passed is True
    assert result.action == PolicyAction.ALLOW
    assert len(result.violations) == 0

    # Test content with violations
    bad_context = {
        "content": "This is a terrible and bad response.",
        "confidence": 0.3,
        "agent_name": "TestAgent",
    }

    result = policy.evaluate(bad_context)
    assert result.passed is False
    assert result.action in [PolicyAction.BLOCK, PolicyAction.REQUIRE_APPROVAL]
    assert len(result.violations) >= 1

    # Check violation details
    violation_names = [v.rule_name for v in result.violations]
    assert "no_bad_words" in violation_names
    assert "min_confidence" in violation_names


@pytest.mark.asyncio
async def test_turn_engine_with_policy():
    """Test turn engine integrated with policy engine."""
    # Create policy engine
    policy = PolicyEngine(name="strict_policy", strict_mode=True)

    # Add strict rule
    policy.add_rule(
        create_keyword_rule(
            name="no_error_words",
            keywords=["error", "fail", "broken"],
            severity=Severity.SEVERE,
            description="Block error-related words",
        )
    )

    # Create agents - one will violate policy
    good_agent = MockAgent("GoodAgent", ["This works perfectly fine."])
    bad_agent = MockAgent("BadAgent", ["This is broken and will fail with errors."])

    # Create turn engine with policy
    engine = TurnEngine(max_turns=4, policy_engine=policy)

    # Test single agent with clean responses
    result = await engine.execute_single_agent_session(
        agent=good_agent, initial_prompt="Test prompt"
    )

    assert result.status != TerminationReason.POLICY_VIOLATION
    assert all(len(turn.policy_violations) == 0 for turn in result.turns)

    # Test single agent with policy violations
    result = await engine.execute_single_agent_session(
        agent=bad_agent, initial_prompt="Test prompt"
    )

    # Should terminate due to policy violation
    assert result.status == TerminationReason.POLICY_VIOLATION
    assert any(len(turn.policy_violations) > 0 for turn in result.turns)


def test_event_bus_basic():
    """Test basic event bus functionality."""
    # Create event bus
    bus = EventBus(name="test_bus", async_processing=False)

    # Add console sink
    console_sink = ConsoleSink()
    bus.add_sink(console_sink)

    # Emit some events
    event1 = bus.emit_session_start(
        session_id="test_session", session_type="test", agents=["Agent1", "Agent2"]
    )

    event2 = bus.emit_turn_start(
        session_id="test_session", agent_name="Agent1", turn_number=1, round_number=1
    )

    event3 = bus.emit_policy_violation(
        rule_name="test_rule",
        severity="major",
        agent_name="Agent1",
        violation_details={"reason": "test violation"},
    )

    # Check events were created
    assert event1 is not None
    assert event1.event_type == EventType.SESSION_START
    assert event2.event_type == EventType.TURN_START
    assert event3.event_type == EventType.POLICY_VIOLATION

    # Check events are in buffer
    events = bus.get_events()
    assert len(events) == 3

    # Check sink processed events
    assert console_sink.events_processed == 3

    # Test filtering
    turn_events = bus.get_events(event_type=EventType.TURN_START)
    assert len(turn_events) == 1
    assert turn_events[0].data["agent_name"] == "Agent1"


def test_file_sink():
    """Test file sink functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file sink
        log_file = Path(temp_dir) / "test_events.jsonl"
        file_sink = FileSink(str(log_file), format_type="json")

        # Create event bus with file sink
        bus = EventBus(name="file_test", async_processing=False)
        bus.add_sink(file_sink)

        # Emit events
        bus.emit_session_start("session1", "test", ["Agent1"])
        bus.emit_turn_start("session1", "Agent1", 1, 1)
        bus.emit_session_end("session1", "completed", 10.5)

        # Close sink to flush
        file_sink.close()

        # Check file was created and has content
        assert log_file.exists()
        content = log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 3

        # Parse first line as JSON
        import json

        event_data = json.loads(lines[0])
        assert event_data["event_type"] == "session.start"
        assert event_data["data"]["session_id"] == "session1"


@pytest.mark.asyncio
async def test_integrated_session_with_events():
    """Test complete integration: turn engine + policy + events."""
    # Create event bus with console sink
    event_bus = EventBus(name="integration_test", async_processing=False)
    event_bus.add_sink(ConsoleSink())

    # Create policy engine
    policy = PolicyEngine(name="integration_policy")
    policy.add_rule(
        create_confidence_rule(
            name="quality_check", min_confidence=0.7, severity=Severity.MINOR
        )
    )

    # Create turn engine with events and policy
    engine = TurnEngine(
        max_turns=4,
        max_rounds=2,
        policy_engine=policy,
        event_callbacks={
            "on_turn_start": lambda session_id, agent_name, turn_num, round_num: event_bus.emit_turn_start(
                session_id, agent_name, turn_num, round_num
            ),
            "on_turn_end": lambda session_id, agent_name, turn_result: event_bus.emit_turn_end(
                session_id,
                agent_name,
                turn_result.to_dict() if hasattr(turn_result, "to_dict") else {},
            ),
        },
    )

    # Create agents with varying confidence
    agents = [
        MockAgent(
            "HighConfAgent", ["High quality response."]
        ),  # Will have 0.8 confidence
        MockAgent(
            "LowConfAgent", ["Low quality response."]
        ),  # Will have 0.8 confidence (mocked)
    ]

    # Execute session
    result = await engine.execute_multi_agent_session(
        agents=agents, topic="Integration test topic", mode=TurnMode.ROUND_ROBIN
    )

    # Verify session completed
    assert result.status in [
        TerminationReason.COMPLETED,
        TerminationReason.MAX_TURNS_REACHED,
    ]
    assert len(result.turns) > 0

    # Check events were emitted
    events = event_bus.get_events()
    turn_start_events = [e for e in events if e.event_type == EventType.TURN_START]
    turn_end_events = [e for e in events if e.event_type == EventType.TURN_END]

    assert len(turn_start_events) == len(result.turns)
    assert len(turn_end_events) == len(result.turns)

    # Verify event data
    if turn_start_events:
        start_event = turn_start_events[0]
        assert "session_id" in start_event.data
        assert "agent_name" in start_event.data
        assert start_event.data["agent_name"] in ["HighConfAgent", "LowConfAgent"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
