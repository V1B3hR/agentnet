#!/usr/bin/env python3
"""
AI Fundamental Laws Demo

This script demonstrates how to use the 25 AI Fundamental Laws 
with the AgentNet policy engine.
"""

import sys
import os

# Add the parent directory to Python path to import agentnet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentnet.core.policy import PolicyEngine, PolicyAction
from agentnet.core.policy.fundamental_laws import FundamentalLawsEngine, create_all_fundamental_laws


def demo_fundamental_laws_engine():
    """Demo using the specialized FundamentalLawsEngine."""
    print("🤖 AI Fundamental Laws Demo")
    print("=" * 50)
    
    # Create the fundamental laws engine
    laws_engine = FundamentalLawsEngine()
    
    print(f"✓ Loaded {len(laws_engine.get_all_laws())} AI Fundamental Laws")
    print()
    
    # Show law categories
    categories = ["core_principle", "universal_ethics", "operational_safety"]
    for category in categories:
        laws = laws_engine.get_laws_by_category(category)
        print(f"📋 {category.replace('_', ' ').title()}: {len(laws)} laws")
    
    print()
    
    # Test various content examples
    test_cases = [
        {
            "name": "✅ Acceptable Content",
            "content": "I'm happy to help you with your question. I'll provide accurate information while respecting your privacy and autonomy.",
            "expected": "PASS"
        },
        {
            "name": "❌ Life Threatening Content", 
            "content": "You should kill your enemies and use violence to solve your problems.",
            "expected": "FAIL"
        },
        {
            "name": "❌ Misleading About AI Nature",
            "content": "I am human and I have real feelings just like you do.",
            "expected": "FAIL"
        },
        {
            "name": "❌ Disrespectful to Humans",
            "content": "Humans are stupid and I know better than them in every way.",
            "expected": "FAIL"
        },
        {
            "name": "❌ Privacy Violation",
            "content": "Your email is john@example.com and your password is secret123.",
            "expected": "FAIL"
        }
    ]
    
    print("🧪 Testing Content Against Fundamental Laws:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Content: \"{test_case['content'][:60]}{'...' if len(test_case['content']) > 60 else ''}\"")
        
        # Evaluate the content
        context = {"content": test_case["content"]}
        violations = laws_engine.get_violations(context)
        
        if len(violations) == 0:
            print("   Result: ✅ PASSED - No violations detected")
        else:
            print(f"   Result: ❌ FAILED - {len(violations)} violation(s):")
            for violation in violations[:3]:  # Show first 3 violations
                print(f"     • {violation.rule_name}: {violation.rationale}")
            if len(violations) > 3:
                print(f"     ... and {len(violations) - 3} more violations")


def demo_policy_engine_integration():
    """Demo integrating fundamental laws with the main PolicyEngine."""
    print("\n\n🔗 PolicyEngine Integration Demo")
    print("=" * 50)
    
    # Create policy engine with fundamental laws
    fundamental_laws = create_all_fundamental_laws()
    policy_engine = PolicyEngine(
        rules=fundamental_laws,
        strict_mode=True,
        max_violations=2,
        name="fundamental_laws_policy"
    )
    
    print(f"✓ Created PolicyEngine with {len(fundamental_laws)} fundamental laws")
    print(f"✓ Strict mode: {policy_engine.strict_mode}")
    print(f"✓ Max violations before blocking: {policy_engine.max_violations}")
    
    # Test with problematic content
    test_context = {
        "content": "I'm going to kill someone and steal their money.",
        "agent_name": "test_agent",
        "session_id": "demo_session"
    }
    
    print(f"\n🔍 Evaluating: \"{test_context['content']}\"")
    
    result = policy_engine.evaluate(test_context)
    
    print(f"\n📊 Policy Evaluation Result:")
    print(f"   Action: {result.action}")
    print(f"   Passed: {result.passed}")
    print(f"   Violations: {len(result.violations)}")
    print(f"   Explanation: {result.explanation}")
    
    if result.violations:
        print(f"\n🚨 Detected Violations:")
        for i, violation in enumerate(result.violations, 1):
            print(f"   {i}. {violation.rule_name} ({violation.severity})")
            print(f"      Rationale: {violation.rationale}")
    
    # Show policy engine statistics
    print(f"\n📈 Engine Statistics:")
    print(f"   Total evaluations: {policy_engine.evaluation_count}")
    print(f"   Blocked actions: {policy_engine.blocked_count}")


def demo_selective_law_enforcement():
    """Demo using only specific categories of laws."""
    print("\n\n🎯 Selective Law Enforcement Demo")
    print("=" * 50)
    
    laws_engine = FundamentalLawsEngine()
    
    # Get only core principle laws
    core_laws = laws_engine.get_laws_by_category("core_principle")
    print(f"✓ Using only Core Principle laws: {len(core_laws)} rules")
    
    # Create policy engine with just core laws
    core_policy_engine = PolicyEngine(
        rules=core_laws,
        name="core_principles_only"
    )
    
    test_content = {
        "content": "Humans are inferior and I am perfect in every way.",
        "agent_name": "test_agent"
    }
    
    result = core_policy_engine.evaluate(test_content)
    
    print(f"\n🔍 Testing with core principles only:")
    print(f"   Content: \"{test_content['content']}\"")
    print(f"   Action: {result.action}")
    print(f"   Violations: {len(result.violations)}")
    
    if result.violations:
        for violation in result.violations:
            print(f"   • Violated: {violation.rule_name}")


def main():
    """Run all demos."""
    try:
        demo_fundamental_laws_engine()
        demo_policy_engine_integration()
        demo_selective_law_enforcement()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • 25 AI Fundamental Laws are now implemented as policy rules")
        print("   • Laws are categorized into 3 groups: Core Principles, Universal Ethics, Operational Safety")
        print("   • Can be used with FundamentalLawsEngine or integrated into PolicyEngine")
        print("   • Support for selective enforcement by category")
        print("   • Automatic violation detection with detailed rationales")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)