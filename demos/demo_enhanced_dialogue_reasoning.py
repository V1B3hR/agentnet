#!/usr/bin/env python3
"""
Demonstration of Enhanced Dialogue Modes and Advanced Reasoning Types.

This script showcases the new capabilities implemented based on research from
managebetter.com and PMC articles to enhance AgentNet's multi-agent reasoning platform.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add agentnet to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agentnet'))

from AgentNet import AgentNet
from reasoning.types import ReasoningEngine, ReasoningType
from reasoning.modulation import ReasoningStyleModulator, ReasoningAwareStyleInfluence
from dialogue.enhanced_modes import OuterDialogue, ModulatedConversation, InterpolationConversation, InnerDialogue, DialogueMapping
from dialogue.solution_focused import SolutionFocusedDialogue, ConversationPhase
from dialogue.mapping import DialogueMapper


def demo_reasoning_types():
    """Demonstrate the five core reasoning types."""
    print("🧠 Advanced Reasoning Types Demonstration")
    print("=" * 50)
    
    # Create agents with different cognitive profiles
    logical_agent = AgentNet("Logic", {"logic": 0.9, "creativity": 0.3, "analytical": 0.8})
    creative_agent = AgentNet("Creative", {"logic": 0.4, "creativity": 0.9, "analytical": 0.6})
    analytical_agent = AgentNet("Analytical", {"logic": 0.7, "creativity": 0.5, "analytical": 0.9})
    
    agents = [logical_agent, creative_agent, analytical_agent]
    
    print(f"\n📊 Agent Profiles:")
    for agent in agents:
        print(f"  {agent.name}: {agent.style}")
    
    # Test reasoning types with different tasks
    reasoning_tasks = [
        ("Deductive Task", "All microservices need monitoring. Our payment service is a microservice. Therefore, our payment service needs monitoring.", ReasoningType.DEDUCTIVE),
        ("Inductive Task", "We've observed that services with caching have 90% fewer timeouts. Our new service should implement caching.", ReasoningType.INDUCTIVE),
        ("Abductive Task", "The API response time suddenly increased by 300%. The most likely explanation is database connection pooling issues.", ReasoningType.ABDUCTIVE),
        ("Analogical Task", "Designing a distributed system is like planning a city - you need good infrastructure, traffic management, and emergency services.", ReasoningType.ANALOGICAL),
        ("Causal Task", "High memory usage caused garbage collection pressure, which led to increased latency, resulting in user complaints.", ReasoningType.CAUSAL)
    ]
    
    print(f"\n🎯 Reasoning Type Demonstrations:")
    
    for task_name, task_description, reasoning_type in reasoning_tasks:
        print(f"\n--- {task_name} ({reasoning_type.value.upper()}) ---")
        print(f"Task: {task_description}")
        
        for agent in agents:
            reasoning_engine = ReasoningEngine(agent.style)
            result = reasoning_engine.reason(task_description, reasoning_type)
            
            print(f"\n{agent.name}'s {reasoning_type.value} reasoning:")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Steps: {len(result.reasoning_steps)}")
            print(f"  Content: {result.content[:100]}...")
            
            if hasattr(result, 'metadata') and result.metadata:
                print(f"  Metadata: {result.metadata}")


def demo_enhanced_dialogue_modes():
    """Demonstrate the five enhanced dialogue modes."""
    print("\n\n💬 Enhanced Dialogue Modes Demonstration")
    print("=" * 50)
    
    # Create agents with complementary styles
    manager = AgentNet("Manager", {"logic": 0.7, "creativity": 0.6, "analytical": 0.8})
    architect = AgentNet("Architect", {"logic": 0.8, "creativity": 0.7, "analytical": 0.9})
    developer = AgentNet("Developer", {"logic": 0.9, "creativity": 0.5, "analytical": 0.8})
    
    agents = [manager, architect, developer]
    topic = "Designing a resilient microservices architecture for high-scale e-commerce"
    
    print(f"\nTopic: {topic}")
    print(f"Participants: {[agent.name for agent in agents]}")
    
    # 1. Outer Dialogue - Enhanced standard communication
    print(f"\n--- 1. OUTER DIALOGUE (Enhanced Standard Communication) ---")
    outer_dialogue = OuterDialogue()
    outer_result = outer_dialogue.conduct_dialogue(
        agents, topic, rounds=3, conversation_pattern="collaborative"
    )
    
    print(f"Conversation Pattern: {outer_result['conversation_pattern']}")
    print(f"Rounds Completed: {outer_result['rounds_completed']}")
    print(f"Analysis: {outer_result['analysis']['dialogue_quality']}")
    print(f"Reasoning Diversity: {outer_result['analysis']['reasoning_diversity']}")
    
    # 2. Modulated Conversation - Tension-building dialogue
    print(f"\n--- 2. MODULATED CONVERSATION (Tension-Building) ---")
    modulated_dialogue = ModulatedConversation()
    modulated_result = modulated_dialogue.conduct_dialogue(
        agents[:2], topic, rounds=4,
        initial_intensity=0.3, max_intensity=0.9, tension_strategy="gradual_build"
    )
    
    print(f"Tension Strategy: {modulated_result['tension_strategy']}")  
    print(f"Max Intensity Reached: {modulated_result['analysis']['max_intensity_reached']:.2f}")
    print(f"Tension Building Success: {modulated_result['analysis']['tension_building_success']}")
    print(f"Conversation Climax Round: {modulated_result['analysis']['conversation_climax_round']}")
    
    # 3. Interpolation Conversation - Context filling
    print(f"\n--- 3. INTERPOLATION CONVERSATION (Context Filling) ---")
    interpolation_dialogue = InterpolationConversation()
    context_gaps = [
        "Missing performance requirements specification",
        "Unclear disaster recovery procedures",
        "Undefined scalability targets"
    ]
    
    interpolation_result = interpolation_dialogue.conduct_dialogue(
        agents[:2], topic, rounds=3,
        context_gaps=context_gaps, interpolation_strategy="memory_guided"
    )
    
    print(f"Initial Gaps: {len(interpolation_result['initial_gaps'])}")
    print(f"Remaining Gaps: {len(interpolation_result['remaining_gaps'])}")
    print(f"Interpolations Made: {len(interpolation_result['interpolations_made'])}")
    print(f"Gap Filling Rate: {interpolation_result['analysis']['gap_filling_rate']:.2f}")
    
    # 4. Inner Dialogue - Self-reflection
    print(f"\n--- 4. INNER DIALOGUE (Self-Reflection) ---")
    inner_dialogue = InnerDialogue()
    inner_result = inner_dialogue.conduct_dialogue(
        [architect], topic, rounds=3,
        reflection_depth="deep", focus_areas=["reasoning_process", "assumptions", "biases"]
    )
    
    print(f"Reflection Depth: {inner_result['reflection_depth']}")
    print(f"Focus Areas: {inner_result['focus_areas']}")
    print(f"Reflection Quality: {inner_result['analysis']['reflection_quality']}")
    print(f"Metacognitive Depth: {inner_result['analysis']['metacognitive_depth']}")
    print(f"Self-Awareness Progression: {inner_result['analysis']['self_awareness_progression']}")
    
    # 5. Dialogue Mapping - Decision visualization
    print(f"\n--- 5. DIALOGUE MAPPING (Decision Visualization) ---")
    mapping_dialogue = DialogueMapping()
    mapping_result = mapping_dialogue.conduct_dialogue(
        agents, topic, rounds=3,
        mapping_focus="decision_making", track_arguments=True
    )
    
    print(f"Mapping Focus: {mapping_result['mapping_focus']}")
    print(f"Decision Points Identified: {len(mapping_result['decision_map']['decision_points'])}")
    print(f"Alternatives Proposed: {len(mapping_result['decision_map']['alternatives'])}")
    print(f"Arguments Tracked: {sum(len(args) for args in mapping_result['decision_map']['arguments'].values())}")
    print(f"Mapping Effectiveness: {mapping_result['mapping_analysis']['mapping_effectiveness']}")


def demo_solution_focused_dialogue():
    """Demonstrate solution-focused conversation patterns."""
    print("\n\n🎯 Solution-Focused Dialogue Demonstration")
    print("=" * 50)
    
    # Create manager-employee style agents
    manager = AgentNet("ProjectManager", {"logic": 0.7, "creativity": 0.6, "analytical": 0.8})
    senior_dev = AgentNet("SeniorDev", {"logic": 0.8, "creativity": 0.7, "analytical": 0.9})
    devops = AgentNet("DevOps", {"logic": 0.9, "creativity": 0.5, "analytical": 0.9})
    
    agents = [manager, senior_dev, devops]
    
    # Problem-focused scenario
    problem_scenario = "Our CI/CD pipeline is failing 40% of the time, causing deployment delays and developer frustration"
    
    print(f"Problem Scenario: {problem_scenario}")
    print(f"Team: {[agent.name for agent in agents]}")
    
    solution_dialogue = SolutionFocusedDialogue()
    result = solution_dialogue.conduct_solution_focused_dialogue(
        agents, problem_scenario, rounds=8,
        initial_phase=ConversationPhase.PROBLEM_IDENTIFICATION,
        transition_strategy="adaptive"
    )
    
    print(f"\n📈 Solution-Focused Analysis:")
    print(f"Phase Progression: {' → '.join(result['phase_progression'])}")
    print(f"Problems Identified: {result['analysis']['problems_identified']}")
    print(f"Solutions Proposed: {result['analysis']['solutions_proposed']}")
    print(f"Questions Asked: {result['analysis']['questions_asked']}")
    print(f"Solution Focus Ratio: {result['analysis']['solution_focus_ratio']:.2f}")
    print(f"Transition Effectiveness: {result['analysis']['transition_effectiveness']}")
    print(f"Overall Effectiveness: {result['analysis']['overall_effectiveness']}")
    
    print(f"\n🔍 Key Solutions Proposed:")
    for i, solution in enumerate(result['solutions_proposed'][:3], 1):
        print(f"  {i}. {solution['description'][:80]}...")
        print(f"     Proposed by: {solution['proposed_by']}, Confidence: {solution['confidence']:.2f}")


def demo_reasoning_style_modulation():
    """Demonstrate reasoning-aware style modulation."""
    print("\n\n🎛️ Reasoning-Style Modulation Demonstration")
    print("=" * 50)
    
    # Create agents with extreme style profiles
    profiles = [
        ("LogicMaster", {"logic": 0.95, "creativity": 0.2, "analytical": 0.8}),
        ("CreativeGenius", {"logic": 0.3, "creativity": 0.95, "analytical": 0.5}),
        ("Analyst", {"logic": 0.6, "creativity": 0.4, "analytical": 0.95})
    ]
    
    modulator = ReasoningStyleModulator()
    
    print("🧪 Style Modulation for Different Reasoning Types:")
    
    for agent_name, base_style in profiles:
        print(f"\n--- {agent_name}: {base_style} ---")
        
        # Show optimal reasoning types for this style
        suggested_types = modulator.suggest_reasoning_types(base_style)
        print(f"Naturally suited for: {[rt.value for rt in suggested_types[:3]]}")
        
        # Show modulation for each reasoning type
        for reasoning_type in [ReasoningType.DEDUCTIVE, ReasoningType.ABDUCTIVE, ReasoningType.ANALOGICAL]:
            modulated_style = modulator.modulate_style_for_reasoning(
                base_style, reasoning_type, modulation_strength=0.6
            )
            
            print(f"  {reasoning_type.value.title()}: {modulated_style}")
            
            # Calculate improvement
            original_alignment = sum(abs(base_style[dim] - modulator.reasoning_profiles[reasoning_type][dim]) 
                                   for dim in ["logic", "creativity", "analytical"])
            modulated_alignment = sum(abs(modulated_style[dim] - modulator.reasoning_profiles[reasoning_type][dim]) 
                                    for dim in ["logic", "creativity", "analytical"])
            
            improvement = (original_alignment - modulated_alignment) / original_alignment * 100
            print(f"    Alignment improvement: {improvement:.1f}%")


def demo_dialogue_mapping_visualization():
    """Demonstrate dialogue mapping and visualization capabilities."""
    print("\n\n🗺️ Dialogue Mapping & Visualization Demonstration")
    print("=" * 50)
    
    # Create a strategic decision scenario
    ceo = AgentNet("CEO", {"logic": 0.8, "creativity": 0.7, "analytical": 0.8})
    cto = AgentNet("CTO", {"logic": 0.9, "creativity": 0.6, "analytical": 0.9})
    cfo = AgentNet("CFO", {"logic": 0.8, "creativity": 0.4, "analytical": 0.9})
    
    agents = [ceo, cto, cfo]
    strategic_topic = "Should we migrate to cloud-native architecture or optimize current infrastructure?"
    
    print(f"Strategic Decision: {strategic_topic}")
    print(f"Stakeholders: {[agent.name for agent in agents]}")
    
    # Conduct mapping dialogue
    mapping_dialogue = DialogueMapping()
    mapping_result = mapping_dialogue.conduct_dialogue(
        agents, strategic_topic, rounds=4,
        mapping_focus="decision_making", track_arguments=True
    )
    
    print(f"\n📊 Decision Map Analysis:")
    print(f"Decision Points: {len(mapping_result['decision_map']['decision_points'])}")
    print(f"Alternatives: {len(mapping_result['decision_map']['alternatives'])}")
    print(f"Criteria: {len(mapping_result['decision_map']['criteria'])}")
    print(f"Constraints: {len(mapping_result['decision_map']['constraints'])}")
    
    # Show argument balance
    arguments = mapping_result['decision_map']['arguments']
    total_args = sum(len(args) for args in arguments.values())
    print(f"\nArgument Distribution (Total: {total_args}):")
    print(f"  Supporting: {len(arguments['pro'])}")
    print(f"  Opposing: {len(arguments['con'])}")
    print(f"  Neutral/Analysis: {len(arguments['neutral'])}")
    
    # Create visualization
    mapper = DialogueMapper()
    
    # Create mock dialogue state for demonstration
    from dialogue.enhanced_modes import DialogueState, DialogueTurn, DialogueMode
    import time
    
    dialogue_state = DialogueState(
        session_id="demo_mapping",
        mode=DialogueMode.MAPPING,
        participants=[agent.name for agent in agents],
        topic=strategic_topic,
        current_round=4,
        transcript=[]
    )
    
    # Create representative turns
    demo_turns = [
        DialogueTurn("CEO", "We need to decide on our infrastructure strategy for the next 3 years", 0.9, DialogueMode.MAPPING, 1, time.time()),
        DialogueTurn("CTO", "What are our main technical constraints and requirements?", 0.8, DialogueMode.MAPPING, 1, time.time()),
        DialogueTurn("CFO", "We need to consider the budget impact - cloud migration costs vs. optimization costs", 0.85, DialogueMode.MAPPING, 2, time.time()),
        DialogueTurn("CTO", "I propose we evaluate three alternatives: full cloud migration, hybrid approach, or infrastructure optimization", 0.9, DialogueMode.MAPPING, 2, time.time())
    ]
    dialogue_state.transcript = demo_turns
    
    # Generate dialogue map
    dialogue_map = mapper.create_dialogue_map(dialogue_state)
    
    print(f"\n🎨 Visualization Data Generated:")
    print(f"Map Nodes: {len(dialogue_map['nodes'])}")
    print(f"Map Edges: {len(dialogue_map['edges'])}")
    
    # Generate visualization data
    vis_data = mapper.generate_visualization_data(dialogue_map['id'])
    print(f"Visualization Nodes: {len(vis_data['nodes'])}")
    print(f"Visualization Edges: {len(vis_data['edges'])}")
    print(f"Layout Type: {vis_data['layout']['type']}")
    
    # Show node types distribution
    node_types = {}
    for node in vis_data['nodes']:
        node_type = node['type']
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\nNode Type Distribution:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")


def demo_multi_perspective_analysis():
    """Demonstrate multi-perspective reasoning analysis."""
    print("\n\n🔬 Multi-Perspective Reasoning Analysis")
    print("=" * 50)
    
    # Complex scenario requiring multiple reasoning approaches
    complex_scenario = """
    Our e-commerce platform experiences intermittent slowdowns during peak traffic.
    Symptoms include: increased response times, occasional timeouts, user complaints.
    Recent changes: new recommendation algorithm, database schema updates, increased traffic by 40%.
    Need to identify root cause and implement solution.
    """
    
    print(f"Complex Scenario Analysis:")
    print(complex_scenario.strip())
    
    # Create reasoning engine
    balanced_style = {"logic": 0.7, "creativity": 0.7, "analytical": 0.8}
    reasoning_engine = ReasoningEngine(balanced_style)
    
    # Apply all reasoning types
    print(f"\n🧠 Multi-Perspective Analysis:")
    
    results = reasoning_engine.multi_perspective_reasoning(complex_scenario)
    
    for result in results:
        print(f"\n--- {result.reasoning_type.value.upper()} PERSPECTIVE ---")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Analysis: {result.content[:150]}...")
        print(f"Key Steps: {', '.join(result.reasoning_steps[:2])}")
        
        if hasattr(result, 'metadata') and result.metadata:
            print(f"Metadata: {result.metadata}")
    
    # Show reasoning type alignment with scenario
    print(f"\n📈 Reasoning Type Effectiveness:")
    sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. {result.reasoning_type.value.title()}: {result.confidence:.2f}")
    
    print(f"\nRecommendation: Use {sorted_results[0].reasoning_type.value} reasoning as primary approach, "
          f"with {sorted_results[1].reasoning_type.value} as supporting perspective.")


async def demo_async_capabilities():
    """Demonstrate async dialogue capabilities."""
    print("\n\n⚡ Async Dialogue Capabilities Demonstration")
    print("=" * 50)
    
    # Create async-capable agents
    async_agents = [
        AgentNet("AsyncAgent1", {"logic": 0.8, "creativity": 0.6, "analytical": 0.7}),
        AgentNet("AsyncAgent2", {"logic": 0.6, "creativity": 0.8, "analytical": 0.8})
    ]
    
    topic = "Real-time system architecture design"
    
    print(f"Async Topic: {topic}")
    print(f"Async Agents: {[agent.name for agent in async_agents]}")
    
    # Test async outer dialogue
    print(f"\n🔄 Async Outer Dialogue:")
    outer_dialogue = OuterDialogue()
    async_result = await outer_dialogue.conduct_async_dialogue(
        async_agents, topic, rounds=3, conversation_pattern="exploratory"
    )
    
    print(f"Async rounds completed: {async_result['rounds_completed']}")
    print(f"Dialogue quality: {async_result['analysis']['dialogue_quality']}")
    
    # Test async modulated conversation
    print(f"\n🌊 Async Modulated Conversation:")
    modulated_dialogue = ModulatedConversation()
    async_modulated = await modulated_dialogue.conduct_async_dialogue(
        async_agents, topic, rounds=3,
        initial_intensity=0.4, max_intensity=0.8, tension_strategy="oscillate"
    )
    
    print(f"Max intensity reached: {async_modulated['analysis']['max_intensity_reached']:.2f}")
    print(f"Tension building success: {async_modulated['analysis']['tension_building_success']}")


def main():
    """Run all demonstrations."""
    print("🚀 Enhanced Dialogue Modes and Advanced Reasoning Types")
    print("🔬 Research-Based Multi-Agent Platform Demonstration")
    print("=" * 80)
    
    try:
        # Core reasoning demonstrations
        demo_reasoning_types()
        demo_reasoning_style_modulation()
        
        # Enhanced dialogue demonstrations
        demo_enhanced_dialogue_modes()
        demo_solution_focused_dialogue()
        demo_dialogue_mapping_visualization()
        
        # Advanced analysis demonstrations
        demo_multi_perspective_analysis()
        
        # Async capabilities
        print("\n⏳ Running async demonstrations...")
        asyncio.run(demo_async_capabilities())
        
        print("\n\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("✨ Successfully demonstrated:")
        print("  🧠 5 Advanced Reasoning Types (Deductive, Inductive, Abductive, Analogical, Causal)")
        print("  💬 5 Enhanced Dialogue Modes (Outer, Modulated, Interpolation, Inner, Mapping)")
        print("  🎯 Solution-Focused Conversation Patterns")
        print("  🗺️ Dialogue Mapping & Visualization")
        print("  🎛️ Reasoning-Aware Style Modulation")
        print("  ⚡ Async Dialogue Capabilities")
        print("  🔬 Multi-Perspective Analysis")
        print("\n🏆 AgentNet Enhanced: Ready for advanced multi-agent reasoning!")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()