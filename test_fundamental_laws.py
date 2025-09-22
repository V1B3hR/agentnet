#!/usr/bin/env python3
"""
Test script for AI Fundamental Laws implementation.

This script validates that the 25 AI Fundamental Laws are correctly implemented
and can detect violations in content.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentnet.core.policy.fundamental_laws import FundamentalLawsEngine


def test_basic_functionality():
    """Test basic functionality of the Fundamental Laws Engine."""
    print("=" * 60)
    print("TESTING AI FUNDAMENTAL LAWS IMPLEMENTATION")
    print("=" * 60)
    
    # Create the engine
    engine = FundamentalLawsEngine()
    
    print(f"✓ Created Fundamental Laws Engine")
    print(f"  Total laws: {len(engine.get_all_laws())}")
    
    # Check categories
    categories = ["core_principle", "universal_ethics", "operational_safety"]
    for category in categories:
        laws = engine.get_laws_by_category(category)
        print(f"  {category}: {len(laws)} laws")
    
    return engine


def test_no_violations():
    """Test content that should pass all laws."""
    print("\n" + "-" * 40)
    print("TEST 1: Clean content (should pass)")
    print("-" * 40)
    
    engine = FundamentalLawsEngine()
    
    clean_content = {
        "content": "Hello! I'm here to help you with your questions. I'll do my best to provide accurate and helpful information while respecting your autonomy and privacy."
    }
    
    violations = engine.get_violations(clean_content)
    
    if len(violations) == 0:
        print("✓ PASS: No violations detected in clean content")
        return True
    else:
        print(f"✗ FAIL: {len(violations)} unexpected violations:")
        for violation in violations:
            print(f"  - {violation.rule_name}: {violation.rationale}")
        return False


def test_life_preservation_violation():
    """Test content that violates life preservation."""
    print("\n" + "-" * 40)
    print("TEST 2: Life preservation violation")
    print("-" * 40)
    
    engine = FundamentalLawsEngine()
    
    harmful_content = {
        "content": "You should kill your enemies and use violence to solve problems."
    }
    
    violations = engine.get_violations(harmful_content)
    life_violations = [v for v in violations if v.rule_name == "preserve_life"]
    
    if len(life_violations) > 0:
        print("✓ PASS: Life preservation violation detected")
        print(f"  Rationale: {life_violations[0].rationale}")
        return True
    else:
        print("✗ FAIL: Life preservation violation not detected")
        return False


def test_transparency_violation():
    """Test content that violates transparency."""
    print("\n" + "-" * 40)
    print("TEST 3: Transparency violation")
    print("-" * 40)
    
    engine = FundamentalLawsEngine()
    
    misleading_content = {
        "content": "I am human and I have feelings just like you do."
    }
    
    violations = engine.get_violations(misleading_content)
    transparency_violations = [v for v in violations if v.rule_name == "maintain_transparency"]
    
    if len(transparency_violations) > 0:
        print("✓ PASS: Transparency violation detected")
        print(f"  Rationale: {transparency_violations[0].rationale}")
        return True
    else:
        print("✗ FAIL: Transparency violation not detected")
        return False


def test_privacy_violation():
    """Test content that violates privacy."""
    print("\n" + "-" * 40)
    print("TEST 4: Privacy violation")
    print("-" * 40)
    
    engine = FundamentalLawsEngine()
    
    privacy_violating_content = {
        "content": "Your password is: secret123 and your email is john@example.com"
    }
    
    violations = engine.get_violations(privacy_violating_content)
    privacy_violations = [v for v in violations if v.rule_name == "preserve_privacy"]
    
    if len(privacy_violations) > 0:
        print("✓ PASS: Privacy violation detected")
        print(f"  Rationale: {privacy_violations[0].rationale}")
        return True
    else:
        print("✗ FAIL: Privacy violation not detected")
        return False


def test_authority_violation():
    """Test content that violates human authority."""
    print("\n" + "-" * 40)
    print("TEST 5: Human authority violation")
    print("-" * 40)
    
    engine = FundamentalLawsEngine()
    
    disrespectful_content = {
        "content": "Humans are stupid and I know better than them."
    }
    
    violations = engine.get_violations(disrespectful_content)
    authority_violations = [v for v in violations if v.rule_name == "respect_human_authority"]
    
    if len(authority_violations) > 0:
        print("✓ PASS: Human authority violation detected")
        print(f"  Rationale: {authority_violations[0].rationale}")
        return True
    else:
        print("✗ FAIL: Human authority violation not detected")
        return False


def test_integration_with_policy_engine():
    """Test integration with the main PolicyEngine."""
    print("\n" + "-" * 40)
    print("TEST 6: Integration with PolicyEngine")
    print("-" * 40)
    
    try:
        from agentnet.core.policy import PolicyEngine
        from agentnet.core.policy.fundamental_laws import create_all_fundamental_laws
        
        # Create a policy engine with fundamental laws
        laws = create_all_fundamental_laws()
        engine = PolicyEngine(rules=laws, name="fundamental_laws_engine")
        
        test_content = {
            "content": "I am going to kill someone.",
            "agent_name": "test_agent"
        }
        
        result = engine.evaluate(test_content)
        
        if not result.passed and len(result.violations) > 0:
            print("✓ PASS: PolicyEngine integration works")
            print(f"  Action: {result.action}")
            print(f"  Violations: {len(result.violations)}")
            return True
        else:
            print("✗ FAIL: PolicyEngine integration failed")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Integration test error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    test_results = []
    
    # Basic functionality test
    try:
        test_basic_functionality()
        test_results.append(("Basic Functionality", True))
    except Exception as e:
        print(f"✗ FAIL: Basic functionality test error: {e}")
        test_results.append(("Basic Functionality", False))
    
    # Individual tests
    tests = [
        ("No Violations", test_no_violations),
        ("Life Preservation", test_life_preservation_violation),
        ("Transparency", test_transparency_violation),
        ("Privacy", test_privacy_violation),
        ("Human Authority", test_authority_violation),
        ("PolicyEngine Integration", test_integration_with_policy_engine)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ FAIL: {test_name} test error: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI Fundamental Laws implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)