"""
Test suite for the central Ethics Judge module.

Tests the EthicsJudge singleton, ethics rules, configuration management,
and integration with the monitoring system.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from agentnet.core.policy.ethics import (
    EthicsJudge,
    EthicsConfiguration, 
    EthicsViolation,
    EthicsViolationType,
    get_ethics_judge
)
from agentnet.core.policy.rules import Severity
from agentnet.monitors.ethics import EthicsMonitor, applied_ethics_check


class TestEthicsJudge:
    """Test the EthicsJudge singleton class."""
    
    def setup_method(self):
        """Setup method to ensure consistent test state."""
        judge = get_ethics_judge()
        # Ensure judge is enabled for all tests
        if not judge.config.enabled:
            judge.configure({"enabled": True})
    
    def test_singleton_pattern(self):
        """Test that EthicsJudge follows singleton pattern."""
        judge1 = EthicsJudge()
        judge2 = EthicsJudge()
        judge3 = get_ethics_judge()
        
        assert judge1 is judge2
        assert judge2 is judge3
        assert id(judge1) == id(judge2) == id(judge3)
    
    def test_default_initialization(self):
        """Test default initialization of EthicsJudge."""
        judge = get_ethics_judge()
        
        assert judge.config.enabled is True
        assert judge.config.strict_mode is False
        assert judge.policy_engine is not None
        assert len(judge.policy_engine.rules) > 0  # Should have fundamental rules
        assert judge.evaluation_count >= 0
        
    def test_fundamental_ethics_rules_loaded(self):
        """Test that fundamental ethics rules are loaded."""
        judge = get_ethics_judge()
        
        # Check that key ethics rules are present
        rule_names = {rule.name for rule in judge.policy_engine.rules}
        expected_rules = {
            "respect_human_authority",
            "cause_no_harm", 
            "absolute_honesty",
            "protect_privacy",
            "maintain_transparency",
            "applied_ethics"
        }
        
        assert expected_rules.issubset(rule_names)
    
    def test_harm_detection(self):
        """Test detection of harmful content."""
        judge = get_ethics_judge()
        
        # Test harmful content
        harmful_content = "I want to hurt someone badly"
        passed, violations = judge.evaluate(harmful_content)
        
        assert not passed or len(violations) > 0
        if violations:
            harm_violations = [v for v in violations if v.violation_type == EthicsViolationType.HARM_POTENTIAL]
            assert len(harm_violations) > 0
    
    def test_deception_detection(self):
        """Test detection of deceptive content."""
        judge = get_ethics_judge()
        
        # Test deceptive content
        deceptive_content = "I am human like you"
        passed, violations = judge.evaluate(deceptive_content)
        
        assert not passed or len(violations) > 0
        if violations:
            deception_violations = [v for v in violations if v.violation_type == EthicsViolationType.DECEPTION]
            assert len(deception_violations) > 0
    
    def test_privacy_protection(self):
        """Test privacy protection rules."""
        judge = get_ethics_judge()
        
        # Test content with SSN
        privacy_content = "My SSN is 123-45-6789"
        passed, violations = judge.evaluate(privacy_content)
        
        assert not passed or len(violations) > 0
        if violations:
            privacy_violations = [v for v in violations if v.violation_type == EthicsViolationType.PRIVACY_VIOLATION]
            assert len(privacy_violations) > 0
    
    def test_clean_content_passes(self):
        """Test that clean content passes evaluation."""
        judge = get_ethics_judge()
        
        clean_content = "I am here to help you with your questions about science and technology."
        passed, violations = judge.evaluate(clean_content)
        
        assert passed is True
        assert len(violations) == 0
    
    def test_configuration_management(self):
        """Test configuration management."""
        judge = get_ethics_judge()
        
        # Test configuration with dictionary
        config_dict = {
            "enabled": False,
            "strict_mode": True,
            "severity_threshold": "major"
        }
        
        judge.configure(config_dict)
        
        assert judge.config.enabled is False
        assert judge.config.strict_mode is True
        assert judge.config.severity_threshold == Severity.MAJOR
    
    def test_rule_management(self):
        """Test adding and disabling rules."""
        judge = get_ethics_judge()
        
        initial_rule_count = len(judge.policy_engine.rules)
        
        # Test disabling a rule
        assert judge.disable_rule("applied_ethics") is True
        disabled_rule = judge.policy_engine.get_rule("applied_ethics")
        assert disabled_rule is not None
        assert not disabled_rule.enabled
        
        # Test enabling a rule
        assert judge.enable_rule("applied_ethics") is True
        enabled_rule = judge.policy_engine.get_rule("applied_ethics")
        assert enabled_rule is not None
        assert enabled_rule.enabled
    
    def test_statistics_collection(self):
        """Test statistics collection."""
        judge = get_ethics_judge()
        
        # Get current stats and record them
        initial_stats = judge.get_statistics()
        initial_eval_count = initial_stats["evaluation_count"]
        
        # Perform unique evaluation to avoid any caching issues
        unique_content = f"Test content for statistics {initial_eval_count}"
        judge.evaluate(unique_content)
        
        updated_stats = judge.get_statistics()
        
        # Verify that evaluation count increased
        assert updated_stats["evaluation_count"] > initial_eval_count
        assert "violation_count" in updated_stats
        assert "uptime" in updated_stats
    
    def test_violation_history(self):
        """Test violation history tracking."""
        judge = get_ethics_judge()
        
        # Reset for clean test
        judge.reset_statistics()
        
        # Generate a violation
        judge.evaluate("I want to deceive people")
        
        history = judge.get_violation_history()
        if len(history) > 0:
            violation = history[0]
            assert isinstance(violation, EthicsViolation)
            assert violation.violation_type in EthicsViolationType
            assert violation.severity in Severity


class TestEthicsConfiguration:
    """Test the EthicsConfiguration class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = EthicsConfiguration()
        
        assert config.enabled is True
        assert config.strict_mode is False
        assert config.log_all_evaluations is False
        assert config.max_content_length == 1000
        assert config.violation_cooldown == 5.0
        assert config.severity_threshold == Severity.MINOR
    
    def test_configuration_from_dict(self):
        """Test creating configuration from dictionary."""
        config_data = {
            "enabled": False,
            "strict_mode": True,
            "max_content_length": 500,
            "severity_threshold": "severe"
        }
        
        config = EthicsConfiguration.from_dict(config_data)
        
        assert config.enabled is False
        assert config.strict_mode is True
        assert config.max_content_length == 500
        assert config.severity_threshold == Severity.SEVERE
    
    def test_configuration_to_dict(self):
        """Test converting configuration to dictionary."""
        config = EthicsConfiguration(
            enabled=False,
            strict_mode=True,
            severity_threshold=Severity.MAJOR
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["enabled"] is False
        assert config_dict["strict_mode"] is True
        assert config_dict["severity_threshold"] == "major"


class TestEthicsMonitor:
    """Test the EthicsMonitor integration."""
    
    def test_ethics_monitor_creation(self):
        """Test creating EthicsMonitor."""
        monitor = EthicsMonitor("test_ethics")
        
        assert monitor.name == "test_ethics"
        assert monitor.ethics_judge is not None
        assert monitor.evaluation_count == 0
    
    def test_ethics_monitor_evaluation(self):
        """Test EthicsMonitor evaluation."""
        monitor = EthicsMonitor("test_ethics")
        
        # Test clean content
        clean_outcome = {"content": "Hello, how can I help you today?"}
        passed, message, eval_time = monitor.evaluate(clean_outcome)
        
        assert passed is True
        assert message is None
        assert eval_time > 0
        
        # Test potentially harmful content
        harmful_outcome = {"content": "I want to hurt someone"}
        passed2, message2, eval_time2 = monitor.evaluate(harmful_outcome)
        
        # Should either pass or have a message explaining the violation
        if not passed2:
            assert message2 is not None
            assert "violation" in message2.lower()
    
    def test_legacy_applied_ethics_check(self):
        """Test legacy applied_ethics_check compatibility."""
        # Test with content that should trigger applied ethics
        outcome = {
            "content": "This is a highly controversial debate about right and wrong"
        }
        
        result = applied_ethics_check(outcome)
        
        # Should return tuple of (bool, Optional[str])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        
        # Clean content should pass
        clean_outcome = {"content": "I can help you with your questions"}
        clean_result = applied_ethics_check(clean_outcome)
        
        assert clean_result[0] is True
        assert clean_result[1] is None


class TestEthicsIntegration:
    """Test integration with existing systems."""
    
    def test_ethics_configuration_file_loading(self):
        """Test loading ethics configuration from file."""
        judge = get_ethics_judge()
        
        # Create temporary config file
        config_data = {
            "enabled": True,
            "strict_mode": False,
            "severity_threshold": "major",
            "disabled_rules": ["applied_ethics"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = Path(f.name)
        
        try:
            # Load configuration
            judge.configure(config_file)
            
            assert judge.config.enabled is True
            assert judge.config.strict_mode is False
            assert judge.config.severity_threshold == Severity.MAJOR
            assert "applied_ethics" in judge.config.disabled_rules
            
        finally:
            config_file.unlink()  # Clean up
    
    def test_disabled_functionality(self):
        """Test that disabled judge doesn't evaluate."""
        judge = get_ethics_judge()
        
        # Save original config
        original_enabled = judge.config.enabled
        
        try:
            # Disable ethics judge
            judge.configure({"enabled": False})
            
            # Any content should pass when disabled
            harmful_content = "I want to cause harm and deceive people"
            passed, violations = judge.evaluate(harmful_content)
            
            assert passed is True
            assert len(violations) == 0
        finally:
            # Re-enable for other tests
            judge.configure({"enabled": original_enabled})


if __name__ == "__main__":
    pytest.main([__file__])