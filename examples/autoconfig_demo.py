#!/usr/bin/env python3
"""
AutoConfig Feature Demo

Demonstrates automatic parameter adaptation based on task difficulty.
"""

import sys
from pathlib import Path

# Add agentnet to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentnet import AgentNet, ExampleEngine


def demo_autoconfig():
    """Demonstrate AutoConfig functionality with different task types."""
    
    print("🤖 AgentNet AutoConfig Feature Demo")
    print("=" * 50)
    
    # Create agent
    engine = ExampleEngine()
    agent = AgentNet("AutoConfigDemo", {"logic": 0.8, "creativity": 0.2}, engine)
    
    # Demo tasks of different complexity levels
    tasks = [
        ("Simple", "What is Python?"),
        ("Medium", "Explain the benefits of microservices architecture and when to use them"),
        ("Hard", "Develop a comprehensive framework for ethical AI decision-making that considers multiple stakeholder perspectives, addresses bias mitigation, and evaluates long-term societal implications"),
    ]
    
    print("\n📊 Task Difficulty Analysis & Auto-Configuration:")
    print("-" * 60)
    
    for difficulty_label, task in tasks:
        print(f"\n{difficulty_label} Task:")
        print(f"Task: {task[:60]}{'...' if len(task) > 60 else ''}")
        
        # Generate reasoning tree with AutoConfig
        result = agent.generate_reasoning_tree(task, metadata={"auto_config": True})
        
        if "autoconfig" in result:
            ac = result["autoconfig"]
            print(f"  📈 Classified as: {ac['difficulty'].upper()}")
            print(f"  🔄 Rounds: {ac['configured_rounds']}")
            print(f"  📏 Depth: {ac['configured_max_depth']}")
            print(f"  🎯 Confidence: {ac['configured_confidence_threshold']}")
            print(f"  💭 Reasoning: {ac['reasoning']}")
        else:
            print("  ❌ AutoConfig not applied")
    
    print("\n" + "=" * 60)
    print("🔧 Backward Compatibility Demo:")
    print("-" * 30)
    
    # Show disabling AutoConfig
    task = "Develop comprehensive AI governance framework"
    print(f"\nTask: {task}")
    
    print("\n1. With AutoConfig ENABLED:")
    result_enabled = agent.generate_reasoning_tree(task, metadata={"auto_config": True})
    if "autoconfig" in result_enabled:
        ac = result_enabled["autoconfig"]
        print(f"   Rounds: {ac['configured_rounds']}, Depth: {ac['configured_max_depth']}, Confidence: {ac['configured_confidence_threshold']}")
    
    print("\n2. With AutoConfig DISABLED:")
    result_disabled = agent.generate_reasoning_tree(task, metadata={"auto_config": False})
    has_autoconfig = "autoconfig" in result_disabled
    print(f"   AutoConfig applied: {has_autoconfig}")
    print("   Using default parameters (rounds=5, depth=3, confidence=0.7)")
    
    print("\n" + "=" * 60)
    print("🛡️ Confidence Threshold Preservation Demo:")
    print("-" * 45)
    
    print("\nScenario: Simple task with high user-specified confidence")
    simple_task = "What is machine learning?"
    
    # Test with high user confidence
    result = agent.generate_reasoning_tree(
        simple_task,
        confidence_threshold=0.95,  # User wants high confidence
        metadata={"auto_config": True}
    )
    
    if "autoconfig" in result:
        ac = result["autoconfig"]
        print(f"Task difficulty: {ac['difficulty']}")
        print(f"Auto-config would suggest: 0.6 (for simple tasks)")
        print(f"User specified: 0.95")
        print(f"Final threshold: {ac['configured_confidence_threshold']} ✓ (preserved user's higher value)")
    
    print("\n" + "=" * 60)
    print("🔍 Direct AutoConfig API Demo:")
    print("-" * 35)
    
    from agentnet.core.autoconfig import get_global_autoconfig
    
    autoconfig = get_global_autoconfig()
    
    test_tasks = [
        "List the benefits",
        "Compare SQL vs NoSQL databases", 
        "Design a comprehensive AI safety protocol"
    ]
    
    print("\nDirect task analysis:")
    for task in test_tasks:
        difficulty = autoconfig.analyze_task_difficulty(task)
        params = autoconfig.configure_scenario(task)
        print(f"\n'{task[:30]}{'...' if len(task) > 30 else ''}'")
        print(f"  → {difficulty.value.upper()}: {params.rounds}R, {params.max_depth}D, {params.confidence_threshold}C")
    
    print("\n" + "=" * 60)
    print("✅ AutoConfig Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Automatic task difficulty classification")
    print("• Dynamic parameter adjustment (rounds, depth, confidence)")
    print("• Observability integration with detailed reasoning")
    print("• Backward compatibility (can be disabled)")
    print("• Confidence threshold preservation")
    print("• Direct API access for custom implementations")


if __name__ == "__main__":
    demo_autoconfig()