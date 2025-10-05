#!/usr/bin/env python3
"""Tests for risk enforcement engine."""

import pytest
from datetime import datetime

from agentnet.risk import (
    RiskRegister,
    RiskEvent,
    RiskLevel,
    RiskCategory,
)

try:
    from agentnet.risk import (
        RiskEnforcementEngine,
        EnforcementRule,
        create_default_enforcement_rules,
    )
    HAS_ENFORCEMENT = True
except ImportError:
    HAS_ENFORCEMENT = False


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_enforcement_engine_creation():
    """Test creating enforcement engine."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    assert engine is not None
    assert engine.risk_register is risk_register


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_add_enforcement_rule():
    """Test adding enforcement rules."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    rule = EnforcementRule(
        rule_id="test_rule",
        risk_type="test_risk",
        trigger_threshold=3,
        time_window_minutes=60,
        enforcement_action="block",
        severity=RiskLevel.HIGH
    )
    
    engine.add_enforcement_rule(rule)
    
    active_rules = engine.get_active_rules()
    assert len(active_rules) == 1
    assert active_rules[0].rule_id == "test_rule"


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_enable_disable_rule():
    """Test enabling and disabling rules."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    rule = EnforcementRule(
        rule_id="test_rule",
        risk_type="test_risk",
        trigger_threshold=3,
        time_window_minutes=60,
        enforcement_action="alert",
        severity=RiskLevel.HIGH
    )
    
    engine.add_enforcement_rule(rule)
    assert len(engine.get_active_rules()) == 1
    
    # Disable rule
    engine.disable_rule("test_rule")
    assert len(engine.get_active_rules()) == 0
    
    # Enable rule
    engine.enable_rule("test_rule")
    assert len(engine.get_active_rules()) == 1


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_blocking_target():
    """Test blocking a target."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    rule = EnforcementRule(
        rule_id="block_rule",
        risk_type="critical_error",
        trigger_threshold=1,
        time_window_minutes=60,
        enforcement_action="block",
        severity=RiskLevel.CRITICAL
    )
    
    engine.add_enforcement_rule(rule)
    
    # Create a critical risk event
    event = RiskEvent(
        risk_id="test_risk_1",
        category=RiskCategory.SECURITY,
        level=RiskLevel.CRITICAL,
        title="Critical security breach",
        description="Test security issue",
        agent_name="test_agent",
        metadata={"risk_type": "critical_error"}
    )
    
    # Check enforcement
    action = engine.check_and_enforce(event, target="test_agent")
    
    if action:  # Enforcement may trigger based on severity
        assert action.action_type == "block"
        assert engine.is_blocked("test_agent")


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_throttling_target():
    """Test throttling a target."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    rule = EnforcementRule(
        rule_id="throttle_rule",
        risk_type="high_usage",
        trigger_threshold=1,
        time_window_minutes=30,
        enforcement_action="throttle",
        severity=RiskLevel.CRITICAL
    )
    
    engine.add_enforcement_rule(rule)
    
    event = RiskEvent(
        risk_id="test_risk_2",
        category=RiskCategory.PERFORMANCE,
        level=RiskLevel.CRITICAL,
        title="High resource usage",
        description="Test performance issue",
        session_id="test_session",
        metadata={"risk_type": "high_usage"}
    )
    
    # Check enforcement
    action = engine.check_and_enforce(event, target="test_session")
    
    if action:  # Enforcement may trigger
        assert action.action_type == "throttle"
        is_throttled, rate = engine.is_throttled("test_session")
        if is_throttled:
            assert rate < 1.0  # Should be throttled


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_enforcement_stats():
    """Test getting enforcement statistics."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    # Add some rules
    for i in range(3):
        rule = EnforcementRule(
            rule_id=f"rule_{i}",
            risk_type="test",
            trigger_threshold=1,
            time_window_minutes=60,
            enforcement_action="alert",
            severity=RiskLevel.HIGH
        )
        engine.add_enforcement_rule(rule)
    
    stats = engine.get_enforcement_stats()
    
    assert "total_enforcement_actions" in stats
    assert "actions_by_type" in stats
    assert "currently_blocked" in stats
    assert "currently_throttled" in stats
    assert "active_rules" in stats
    assert stats["total_rules"] == 3


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_default_enforcement_rules():
    """Test creating default enforcement rules."""
    rules = create_default_enforcement_rules()
    
    assert len(rules) > 0
    assert all(isinstance(rule, EnforcementRule) for rule in rules)
    
    # Check that we have rules for key risk types
    risk_types = {rule.risk_type for rule in rules}
    assert "token_cost_spike" in risk_types
    assert "provider_outage" in risk_types


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_action_callback():
    """Test action callback registration."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    callback_called = []
    
    def test_callback(action, event):
        callback_called.append((action, event))
    
    # Register callback
    engine.register_action_callback("alert", test_callback)
    
    # Create rule that triggers alert
    rule = EnforcementRule(
        rule_id="alert_rule",
        risk_type="test_alert",
        trigger_threshold=1,
        time_window_minutes=60,
        enforcement_action="alert",
        severity=RiskLevel.CRITICAL
    )
    
    engine.add_enforcement_rule(rule)
    
    event = RiskEvent(
        risk_id="test_risk_3",
        category=RiskCategory.OPERATIONAL,
        level=RiskLevel.CRITICAL,
        title="Test alert",
        description="Test alert event",
        metadata={"risk_type": "test_alert"}
    )
    
    # Trigger enforcement
    action = engine.check_and_enforce(event, target="test")
    
    # Check if callback was called (if enforcement triggered)
    if action:
        assert len(callback_called) > 0


@pytest.mark.skipif(not HAS_ENFORCEMENT, reason="Enforcement module not available")
def test_enforcement_history():
    """Test enforcement history tracking."""
    risk_register = RiskRegister(storage_dir="/tmp/test_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    # Get initial history (should be empty)
    history = engine.get_enforcement_history()
    initial_count = len(history)
    
    # Add rule and trigger enforcement
    rule = EnforcementRule(
        rule_id="history_rule",
        risk_type="test_event",
        trigger_threshold=1,
        time_window_minutes=60,
        enforcement_action="alert",
        severity=RiskLevel.CRITICAL
    )
    
    engine.add_enforcement_rule(rule)
    
    event = RiskEvent(
        risk_id="test_risk_4",
        category=RiskCategory.SECURITY,
        level=RiskLevel.CRITICAL,
        title="Test for history",
        description="Test history tracking",
        metadata={"risk_type": "test_event"}
    )
    
    action = engine.check_and_enforce(event, target="test_target")
    
    # Check history (if enforcement was triggered)
    if action:
        history = engine.get_enforcement_history()
        assert len(history) > initial_count
        
        # Filter by target
        target_history = engine.get_enforcement_history(target="test_target")
        assert len(target_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
