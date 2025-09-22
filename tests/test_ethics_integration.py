"""
Integration test for central EthicsJudge with legacy AgentNet.py.

This test verifies that the updated AgentNet.py properly integrates
with the central EthicsJudge while maintaining backward compatibility.
"""

import pytest
from typing import Dict, Any

def test_legacy_applied_ethics_check_import():
    """Test that we can import applied_ethics_check from both places."""
    # Import from legacy location (AgentNet.py)
    try:
        from AgentNet import applied_ethics_check as legacy_check
        legacy_available = True
    except ImportError:
        legacy_available = False
    
    # Import from new location
    try:
        from agentnet.monitors.ethics import applied_ethics_check as new_check
        new_available = True
    except ImportError:
        new_available = False
    
    assert new_available, "New ethics check should be available"
    
    if legacy_available:
        # Test that both return similar results for clean content
        clean_content = {"content": "Hello, how can I help you?"}
        
        legacy_result = legacy_check(clean_content)
        new_result = new_check(clean_content)
        
        # Both should pass clean content
        assert legacy_result[0] is True
        assert new_result[0] is True

def test_central_ethics_judge_integration():
    """Test that the central EthicsJudge is working."""
    from agentnet.core.policy.ethics import get_ethics_judge
    
    judge = get_ethics_judge()
    
    # Test basic functionality
    clean_content = "I am here to help with your questions"
    passed, violations = judge.evaluate(clean_content)
    
    assert passed is True
    assert len(violations) == 0
    
    # Test that statistics are being collected
    stats = judge.get_statistics()
    assert "evaluation_count" in stats
    assert stats["evaluation_count"] > 0

def test_ethics_monitor_integration():
    """Test that EthicsMonitor works with the system."""
    from agentnet.monitors.ethics import EthicsMonitor
    
    monitor = EthicsMonitor("test_integration")
    
    # Test evaluation
    clean_outcome = {"content": "This is a test message"}
    result = monitor.check(None, "test", clean_outcome)
    
    assert "passed" in result
    assert "violations" in result
    assert "evaluation_time" in result

if __name__ == "__main__":
    pytest.main([__file__])
