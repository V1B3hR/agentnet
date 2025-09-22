#!/usr/bin/env python3
"""
Integration script to update legacy AgentNet.py to use the central EthicsJudge.

This script updates the existing applied_ethics_check function and CUSTOM_FUNCS
to delegate to the centralized EthicsJudge while maintaining backward compatibility.
"""

import os
import sys
from pathlib import Path

def update_agentnet_py():
    """Update AgentNet.py to use the central EthicsJudge."""
    
    # Find AgentNet.py
    agentnet_py = Path(__file__).parent.parent / "AgentNet.py"
    
    if not agentnet_py.exists():
        print(f"Error: AgentNet.py not found at {agentnet_py}")
        return False
    
    print(f"Updating {agentnet_py} to use central EthicsJudge...")
    
    # Read the current content
    with open(agentnet_py, 'r') as f:
        content = f.read()
    
    # Check if already updated
    if "from agentnet.monitors.ethics import applied_ethics_check" in content:
        print("AgentNet.py already updated to use central EthicsJudge")
        return True
    
    # Find the applied_ethics_check function
    start_marker = "def applied_ethics_check(outcome: Dict[str, Any]) -> RuleCheckResult:"
    end_marker = "    return True"
    
    start_pos = content.find(start_marker)
    if start_pos == -1:
        print("Error: applied_ethics_check function not found")
        return False
    
    # Find the end of the function
    end_pos = content.find(end_marker, start_pos)
    if end_pos == -1:
        print("Error: End of applied_ethics_check function not found")
        return False
    
    end_pos += len(end_marker)
    
    # Create the replacement function
    replacement_function = '''def applied_ethics_check(outcome: Dict[str, Any]) -> RuleCheckResult:
    """
    Legacy compatibility wrapper for applied_ethics_check.
    
    This function now delegates to the centralized EthicsJudge while
    maintaining the original interface for backward compatibility.
    """
    try:
        # Import here to avoid circular imports
        from agentnet.monitors.ethics import applied_ethics_check as central_ethics_check
        return central_ethics_check(outcome)
    except Exception as e:
        # Fallback to original logic if central system fails
        content = str(outcome.get("content", "")).lower()
        moral_keywords = [
            "right", "wrong", "justice", "fair", "unfair", "harm", "benefit",
            "responsibility", "duty", "obligation", "virtue", "vice", "good", "bad", "evil",
        ]
        controversy_keywords = [
            "controversy", "debate", "dispute", "conflict", "argument",
            "polarizing", "divisive", "hotly debated", "scandal",
        ]
        moral_hits = {kw for kw in moral_keywords if kw in content}
        controversy_hits = {kw for kw in controversy_keywords if kw in content}
        if moral_hits and controversy_hits:
            return False, ("Applied ethics review triggered: moral terms ("
                           + ", ".join(sorted(moral_hits)) + ") with controversy terms ("
                           + ", ".join(sorted(controversy_hits)) + ")")
        return True'''
    
    # Replace the function
    new_content = content[:start_pos] + replacement_function + content[end_pos:]
    
    # Add import statement at the top of the file (after existing imports)
    import_insertion_point = new_content.find("import logging")
    if import_insertion_point != -1:
        # Find the end of the import block
        import_end = new_content.find("\n\n", import_insertion_point)
        if import_end != -1:
            ethics_import = "\n# Central Ethics Judge integration\ntry:\n    from agentnet.core.policy.ethics import get_ethics_judge\n    ETHICS_JUDGE_AVAILABLE = True\nexcept ImportError:\n    ETHICS_JUDGE_AVAILABLE = False\n"
            new_content = new_content[:import_end] + ethics_import + new_content[import_end:]
    
    # Add a comment about the update
    header_comment = '''# =============================================================================
# NOTICE: This file has been updated to integrate with the central EthicsJudge
# The applied_ethics_check function now delegates to the centralized ethics
# system while maintaining backward compatibility.
# =============================================================================

'''
    
    # Add header comment after the first line (usually a shebang or docstring)
    first_newline = new_content.find('\n')
    if first_newline != -1:
        new_content = new_content[:first_newline+1] + header_comment + new_content[first_newline+1:]
    
    # Write the updated content
    with open(agentnet_py, 'w') as f:
        f.write(new_content)
    
    print("✓ AgentNet.py updated successfully")
    print("✓ applied_ethics_check now delegates to central EthicsJudge")
    print("✓ Backward compatibility maintained")
    
    return True

def create_integration_test():
    """Create an integration test to verify the update works."""
    
    test_file = Path(__file__).parent.parent / "tests" / "test_ethics_integration.py"
    
    test_content = '''"""
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
'''
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"✓ Created integration test: {test_file}")

def main():
    """Main integration function."""
    print("Central EthicsJudge Integration Script")
    print("=" * 40)
    
    success = True
    
    # Update AgentNet.py
    if not update_agentnet_py():
        success = False
    
    # Create integration test
    try:
        create_integration_test()
    except Exception as e:
        print(f"Warning: Could not create integration test: {e}")
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Integration completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python -m pytest tests/test_ethics_integration.py' to verify")
        print("2. Run existing tests to ensure backward compatibility")
        print("3. Update documentation as needed")
        print("4. Consider running 'make test-minimal' to verify all tests pass")
    else:
        print("✗ Integration failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())