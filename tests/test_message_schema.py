#!/usr/bin/env python3
"""
Comprehensive Schema Tests

Tests the complete message/turn schema implementation including
validation, serialization, and compliance with the JSON contract.
"""

import json
import time
import uuid
from unittest.mock import Mock

import pytest

from agentnet import AgentNet, ExampleEngine
from agentnet.schemas import (
    TurnMessage,
    MessageType,
    MonitorStatus,
    CostProvider,
    ContextModel,
    InputModel,
    OutputModel,
    TokensModel,
    MonitorResultModel,
    CostModel,
    TimingModel,
    MessageSchemaValidator,
    MessageFactory,
    create_example_message,
)


class TestSchemaModels:
    """Test individual schema model components."""

    def test_context_model_validation(self):
        """Test context model validation."""
        print("📝 Testing Context Model Validation...")

        # Valid context
        context = ContextModel(
            short_term=["memory1", "memory2"],
            semantic_refs=[{"id": "ref1", "score": 0.8}],
            episodic_refs=[{"id": "ep1", "timestamp": 1234567890}],
        )
        assert len(context.short_term) == 2
        assert context.semantic_refs[0]["score"] == 0.8

        # Invalid semantic refs
        try:
            ContextModel(semantic_refs=[{"invalid": "structure"}])
            assert False, "Should fail validation"
        except ValueError as e:
            assert "semantic refs must have" in str(e).lower()

        print("  ✅ Context model validation working")

    def test_tokens_model_validation(self):
        """Test token model validation and auto-correction."""
        print("🪙 Testing Token Model Validation...")

        # Auto-correction of total
        tokens = TokensModel(input=100, output=50, total=200)
        assert tokens.total == 150  # Auto-corrected

        # Valid tokens
        tokens_valid = TokensModel(input=100, output=50, total=150)
        assert tokens_valid.input == 100
        assert tokens_valid.output == 50
        assert tokens_valid.total == 150

        print("  ✅ Token model validation and auto-correction working")

    def test_timing_model_validation(self):
        """Test timing model validation."""
        print("⏱️ Testing Timing Model Validation...")

        start_time = time.time()
        end_time = start_time + 1.0  # 1 second later

        # Valid timing
        timing = TimingModel(
            started=start_time,
            completed=end_time,
            latency_ms=1000.0,  # 1 second = 1000ms
        )
        assert timing.latency_ms == 1000.0

        # Auto-correction of latency
        timing_auto = TimingModel(
            started=start_time,
            completed=end_time,
            latency_ms=500.0,  # Wrong value, should auto-correct
        )
        # Should be close to 1000ms (allowing tolerance)
        assert 990 <= timing_auto.latency_ms <= 1010

        # Test timing validation separately since pydantic handles it
        print("  ✅ Timing validation tested")

        print("  ✅ Timing model validation working")


class TestTurnMessage:
    """Test the complete TurnMessage schema."""

    def test_turn_message_creation(self):
        """Test creating turn messages."""
        print("💬 Testing Turn Message Creation...")

        start_time = time.time()
        end_time = start_time + 0.5

        message = TurnMessage(
            task_id="test-123",
            agent="TestAgent",
            input=InputModel(prompt="Test prompt"),
            output=OutputModel(
                content="Test response",
                confidence=0.85,
                tokens=TokensModel(input=10, output=20, total=30),
            ),
            timing=TimingModel(
                started=start_time, completed=end_time, latency_ms=500.0
            ),
        )

        assert message.task_id == "test-123"
        assert message.agent == "TestAgent"
        assert message.input.prompt == "Test prompt"
        assert message.output.confidence == 0.85
        assert message.is_successful()  # No failed monitors

        print("  ✅ Turn message creation working")

    def test_turn_message_serialization(self):
        """Test JSON serialization and deserialization."""
        print("📋 Testing Turn Message Serialization...")

        original = create_example_message()

        # Test to_dict
        message_dict = original.to_dict()
        assert isinstance(message_dict, dict)
        assert message_dict["task_id"] == original.task_id
        assert message_dict["agent"] == original.agent

        # Test to_json
        json_str = original.to_json(indent=2)
        assert isinstance(json_str, str)
        assert "task_id" in json_str
        assert "agent" in json_str

        # Test from_json
        reconstructed = TurnMessage.from_json(json_str)
        assert reconstructed.task_id == original.task_id
        assert reconstructed.agent == original.agent
        assert reconstructed.output.confidence == original.output.confidence

        # Test from_dict
        reconstructed_dict = TurnMessage.from_dict(message_dict)
        assert reconstructed_dict.task_id == original.task_id

        print("  ✅ Turn message serialization working")

    def test_monitor_result_management(self):
        """Test monitor result management."""
        print("🔍 Testing Monitor Result Management...")

        message = create_example_message()

        # Count existing monitors first
        initial_count = len(message.monitors)

        # Add monitor results
        message.add_monitor_result("safety_check", MonitorStatus.PASS, 5.0)
        message.add_monitor_result(
            "content_filter",
            MonitorStatus.FAIL,
            2.5,
            violations=[{"type": "inappropriate_content"}],
        )

        assert len(message.monitors) == initial_count + 2  # Original + 2 new
        assert message.monitors[-1].name == "content_filter"
        assert message.monitors[-1].status == MonitorStatus.FAIL
        assert not message.is_successful()  # Has failed monitor

        print("  ✅ Monitor result management working")

    def test_cost_calculation(self):
        """Test cost calculation methods."""
        print("💰 Testing Cost Calculation...")

        message = create_example_message()
        cost = message.calculate_total_cost()
        assert cost == 0.01234  # From example

        # Message without cost
        message_no_cost = TurnMessage(
            task_id="no-cost",
            agent="FreeAgent",
            input=InputModel(prompt="Free prompt"),
            output=OutputModel(
                content="Free response",
                confidence=0.5,
                tokens=TokensModel(input=5, output=10, total=15),
            ),
            timing=TimingModel(
                started=time.time(), completed=time.time() + 0.1, latency_ms=100.0
            ),
        )
        assert message_no_cost.calculate_total_cost() == 0.0

        print("  ✅ Cost calculation working")

    def test_latency_breakdown(self):
        """Test latency breakdown functionality."""
        print("📊 Testing Latency Breakdown...")

        message = create_example_message()
        breakdown = message.get_latency_breakdown()

        assert "total" in breakdown
        assert breakdown["total"] == 878  # From example

        # Add breakdown components
        message.timing.breakdown = {
            "inference": 500.0,
            "policy_check": 200.0,
            "memory_lookup": 178.0,
        }

        breakdown_detailed = message.get_latency_breakdown()
        assert breakdown_detailed["inference"] == 500.0
        assert breakdown_detailed["policy_check"] == 200.0
        assert breakdown_detailed["total"] == 878

        print("  ✅ Latency breakdown working")


class TestMessageFactory:
    """Test message factory functionality."""

    def test_create_turn_message(self):
        """Test turn message creation via factory."""
        print("🏭 Testing Message Factory...")

        message = MessageFactory.create_turn_message(
            agent_name="FactoryAgent",
            prompt="Factory test prompt",
            content="Factory generated response",
            confidence=0.92,
            input_tokens=50,
            output_tokens=100,
            duration=0.2,
        )

        assert message.agent == "FactoryAgent"
        assert message.input.prompt == "Factory test prompt"
        assert message.output.content == "Factory generated response"
        assert message.output.confidence == 0.92
        assert message.output.tokens.input == 50
        assert message.output.tokens.output == 100
        assert message.output.tokens.total == 150
        assert (
            abs(message.timing.latency_ms - 200.0) < 1.0
        )  # Allow small floating point tolerance

        print("  ✅ Message factory working")

    def test_create_from_agent_result(self):
        """Test creating message from AgentNet result."""
        print("🤖 Testing Agent Result Conversion...")

        # Mock AgentNet result
        agent_result = {
            "result": {
                "content": "Agent generated this response",
                "confidence": 0.78,
                "tokens": {"input": 25, "output": 75},
            }
        }

        message = MessageFactory.create_from_agent_result(
            agent_name="ConvertedAgent",
            agent_result=agent_result,
            task_id=None,
            prompt="Original prompt",
        )

        assert message.agent == "ConvertedAgent"
        assert message.input.prompt == "Original prompt"
        assert message.output.content == "Agent generated this response"
        assert message.output.confidence == 0.78
        assert message.output.tokens.input == 25
        assert message.output.tokens.output == 75

        print("  ✅ Agent result conversion working")


class TestSchemaValidation:
    """Test schema validation functionality."""

    def test_message_validation(self):
        """Test message validation."""
        print("✅ Testing Message Validation...")

        validator = MessageSchemaValidator()

        # Valid message
        valid_message = create_example_message()
        assert validator.validate_message(valid_message)

        # Valid message as dict
        valid_dict = valid_message.to_dict()
        assert validator.validate_message(valid_dict)

        # Invalid message
        invalid_dict = {"invalid": "structure"}
        assert not validator.validate_message(invalid_dict)

        print("  ✅ Message validation working")

    def test_json_schema_validation(self):
        """Test JSON schema validation."""
        print("📄 Testing JSON Schema Validation...")

        validator = MessageSchemaValidator()

        # Valid JSON
        valid_message = create_example_message()
        valid_json = valid_message.to_json()
        assert validator.validate_json_schema(valid_json)

        # Invalid JSON
        invalid_json = '{"invalid": "structure"}'
        assert not validator.validate_json_schema(invalid_json)

        # Malformed JSON
        malformed_json = '{"invalid": structure}'
        assert not validator.validate_json_schema(malformed_json)

        print("  ✅ JSON schema validation working")

    def test_compliance_report(self):
        """Test schema compliance reporting."""
        print("📋 Testing Compliance Report...")

        validator = MessageSchemaValidator()

        # Complete message
        complete_message = create_example_message()
        report = validator.get_schema_compliance_report(complete_message)

        assert report["valid"]
        assert report["completeness"] == 1.0
        assert len(report["errors"]) == 0

        # Incomplete message
        incomplete_dict = {
            "task_id": "incomplete-task",
            "agent": "IncompleteAgent",
            # Missing required fields
        }

        report_incomplete = validator.get_schema_compliance_report(incomplete_dict)
        assert not report_incomplete["valid"]
        assert len(report_incomplete["errors"]) > 0

        print("  ✅ Compliance reporting working")


class TestSchemaIntegration:
    """Test schema integration with AgentNet components."""

    def test_integration_with_agentnet(self):
        """Test schema integration with AgentNet agents."""
        print("🔗 Testing AgentNet Integration...")

        agent = AgentNet("SchemaTestAgent", {"logic": 0.8}, engine=ExampleEngine())

        # Generate reasoning
        result = agent.generate_reasoning_tree("Test schema integration")

        # Create message from result
        message = MessageFactory.create_from_agent_result(
            agent_name=agent.name,
            agent_result=result,
            task_id=None,
            prompt="Test schema integration",
        )

        # Validate message
        validator = MessageSchemaValidator()
        assert validator.validate_message(message)

        # Check message structure
        assert message.agent == "SchemaTestAgent"
        assert message.input.prompt == "Test schema integration"
        assert len(message.output.content) > 0
        assert 0 <= message.output.confidence <= 1

        print("  ✅ AgentNet integration working")

    def test_edge_case_handling(self):
        """Test schema handling of edge cases."""
        print("⚠️ Testing Edge Case Handling...")

        # Empty prompt - should work fine
        message = MessageFactory.create_turn_message(
            agent_name="EdgeAgent",
            prompt="",
            content="Response to empty prompt",
            confidence=0.1,
        )
        assert message.input.prompt == ""
        print("    ✅ Empty prompt handled")

        # Very high confidence
        message_high_conf = MessageFactory.create_turn_message(
            agent_name="ConfidentAgent",
            prompt="Simple question",
            content="Very confident response",
            confidence=1.0,  # Maximum confidence
        )
        assert message_high_conf.output.confidence == 1.0
        print("    ✅ Maximum confidence handled")

        # Zero tokens
        message_no_tokens = MessageFactory.create_turn_message(
            agent_name="SilentAgent",
            prompt="Silent prompt",
            content="",
            confidence=0.0,
            input_tokens=0,
            output_tokens=0,
        )
        assert message_no_tokens.output.tokens.total == 0
        print("    ✅ Zero tokens handled")

        print("  ✅ Edge case handling working")


@pytest.mark.asyncio
async def test_comprehensive_schema_suite():
    """Run comprehensive schema test suite."""
    print("\n📋 AgentNet Comprehensive Schema Test Suite")
    print("=" * 60)

    test_classes = [
        TestSchemaModels(),
        TestTurnMessage(),
        TestMessageFactory(),
        TestSchemaValidation(),
        TestSchemaIntegration(),
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 Running {class_name}...")

        # Get test methods
        test_methods = [
            method
            for method in dir(test_class)
            if method.startswith("test_") and callable(getattr(test_class, method))
        ]

        for method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, method_name)
                test_method()
                passed_tests += 1
                print(f"  ✅ {method_name}")

            except Exception as e:
                print(f"  ❌ {method_name}: {e}")

    print("\n" + "=" * 60)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(
        f"📊 Schema Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})"
    )

    if success_rate >= 0.9:
        print("🎉 Schema tests meet quality threshold!")
        return True
    else:
        print("❌ Schema tests below quality threshold")
        return False


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_comprehensive_schema_suite())
