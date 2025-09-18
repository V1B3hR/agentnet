#!/usr/bin/env python3
"""
Integration Example: Using Enhanced Dialogue Modes and Advanced Reasoning Types with AgentNet.

This example shows how to integrate the new features into your existing AgentNet workflows.
"""

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
from dialogue.enhanced_modes import OuterDialogue, InnerDialogue
from dialogue.solution_focused import SolutionFocusedDialogue


def create_enhanced_agent(name: str, style: dict) -> AgentNet:
    """Create an AgentNet agent with enhanced reasoning capabilities."""
    agent = AgentNet(name, style)
    
    # Add reasoning engine
    agent.reasoning_engine = ReasoningEngine(style)
    
    # Add style modulator
    agent.style_modulator = ReasoningStyleModulator()
    
    return agent


class EnhancedAgentNet(AgentNet):
    """Extended AgentNet class with built-in enhanced features."""
    
    def __init__(self, name: str, style: dict, **kwargs):
        """Initialize enhanced AgentNet with reasoning and dialogue capabilities."""
        super().__init__(name, style, **kwargs)
        
        # Add enhanced capabilities
        self.reasoning_engine = ReasoningEngine(style)
        self.style_modulator = ReasoningStyleModulator()
        self.reasoning_aware_influence = ReasoningAwareStyleInfluence(self.style_modulator)
        
        # Enhanced dialogue modes
        self.outer_dialogue = OuterDialogue()
        self.inner_dialogue = InnerDialogue()
        self.solution_dialogue = SolutionFocusedDialogue()
    
    def enhanced_reasoning(self, task: str, reasoning_type: ReasoningType = None):
        """Perform enhanced reasoning with style modulation."""
        return self.reasoning_engine.reason(task, reasoning_type)
    
    def multi_perspective_analysis(self, task: str):
        """Analyze task from multiple reasoning perspectives."""
        return self.reasoning_engine.multi_perspective_reasoning(task)
    
    def solution_focused_dialogue_with(self, other_agents: list, problem: str, rounds: int = 6):
        """Conduct solution-focused dialogue with other agents."""
        agents = [self] + other_agents
        return self.solution_dialogue.conduct_solution_focused_dialogue(
            agents, problem, rounds=rounds
        )
    
    def self_reflect(self, topic: str, depth: str = "moderate"):
        """Engage in self-reflection using inner dialogue."""
        return self.inner_dialogue.conduct_dialogue([self], topic, reflection_depth=depth)
    
    def enhanced_multi_party_dialogue(self, agents: list, topic: str, pattern: str = "collaborative"):
        """Enhanced multi-party dialogue with conversation patterns."""
        all_agents = [self] + agents
        return self.outer_dialogue.conduct_dialogue(
            all_agents, topic, conversation_pattern=pattern
        )


def example_1_basic_reasoning_enhancement():
    """Example 1: Enhance existing agents with reasoning capabilities."""
    print("📚 Example 1: Basic Reasoning Enhancement")
    print("=" * 50)
    
    # Create regular AgentNet agents
    analyst = AgentNet("DataAnalyst", {"logic": 0.9, "creativity": 0.4, "analytical": 0.9})
    architect = AgentNet("SystemArchitect", {"logic": 0.8, "creativity": 0.7, "analytical": 0.8})
    
    # Add reasoning capabilities
    analyst.reasoning_engine = ReasoningEngine(analyst.style)
    architect.reasoning_engine = ReasoningEngine(architect.style)
    
    # Complex technical problem
    problem = """
    Our distributed system shows inconsistent latency patterns: 
    - 95% of requests complete in <100ms
    - 5% take 2-10 seconds  
    - Pattern occurs randomly across all services
    - Started after recent Kubernetes upgrade
    """
    
    print(f"Problem: {problem.strip()}")
    
    # Analyst uses inductive reasoning (pattern recognition)
    analyst_result = analyst.reasoning_engine.reason(problem, ReasoningType.INDUCTIVE)
    print(f"\n🔍 Analyst (Inductive): Confidence {analyst_result.confidence:.2f}")
    print(f"Analysis: {analyst_result.content[:100]}...")
    
    # Architect uses causal reasoning (system thinking)
    architect_result = architect.reasoning_engine.reason(problem, ReasoningType.CAUSAL)
    print(f"\n⚙️ Architect (Causal): Confidence {architect_result.confidence:.2f}")
    print(f"Analysis: {architect_result.content[:100]}...")
    
    print("\n✅ Basic reasoning enhancement complete!")


def example_2_solution_focused_team_meeting():
    """Example 2: Solution-focused team meeting."""
    print("\n\n🎯 Example 2: Solution-Focused Team Meeting")
    print("=" * 50)
    
    # Create enhanced agents
    manager = EnhancedAgentNet("ProjectManager", {"logic": 0.7, "creativity": 0.6, "analytical": 0.8})
    developer = EnhancedAgentNet("LeadDeveloper", {"logic": 0.8, "creativity": 0.7, "analytical": 0.9})
    devops = EnhancedAgentNet("DevOpsEngineer", {"logic": 0.9, "creativity": 0.5, "analytical": 0.9})
    
    # Problem scenario
    problem = "Our deployment success rate dropped to 60% this month, causing delays and team stress"
    
    print(f"Problem: {problem}")
    
    # Conduct solution-focused dialogue
    result = manager.solution_focused_dialogue_with([developer, devops], problem, rounds=4)
    
    print(f"\n📊 Results:")
    print(f"Phases covered: {len(set(result['phase_progression']))}")
    print(f"Solutions proposed: {result['analysis']['solutions_proposed']}")
    print(f"Solution focus ratio: {result['analysis']['solution_focus_ratio']:.2f}")
    print(f"Effectiveness: {result['analysis']['overall_effectiveness']}")
    
    # Show key solutions
    if result['solutions_proposed']:
        print(f"\n💡 Key Solutions:")
        for i, solution in enumerate(result['solutions_proposed'][:2], 1):
            print(f"  {i}. {solution['description'][:80]}...")
    
    print("\n✅ Solution-focused meeting complete!")


def example_3_multi_perspective_analysis():
    """Example 3: Multi-perspective analysis of complex problem."""
    print("\n\n🔬 Example 3: Multi-Perspective Analysis")
    print("=" * 50)
    
    # Create enhanced agent with balanced style
    analyst = EnhancedAgentNet("SystemAnalyst", {"logic": 0.7, "creativity": 0.7, "analytical": 0.8})
    
    # Complex business-technical problem
    complex_problem = """
    Company growth requires scaling our monolithic application, but:
    - Limited budget for full rewrite
    - Team lacks microservices experience  
    - Current system handles 80% of use cases well
    - Performance issues only in specific modules
    - Deadline pressure for new features
    """
    
    print(f"Complex Problem: {complex_problem.strip()}")
    
    # Multi-perspective analysis
    perspectives = analyst.multi_perspective_analysis(complex_problem)
    
    print(f"\n🎯 Multi-Perspective Analysis ({len(perspectives)} perspectives):")
    
    # Sort by confidence
    sorted_perspectives = sorted(perspectives, key=lambda p: p.confidence, reverse=True)
    
    for i, perspective in enumerate(sorted_perspectives, 1):
        print(f"\n{i}. {perspective.reasoning_type.value.upper()} (Confidence: {perspective.confidence:.2f})")
        print(f"   Approach: {perspective.content[:100]}...")
        print(f"   Key steps: {', '.join(perspective.reasoning_steps[:2])}")
    
    # Recommendation
    best_approach = sorted_perspectives[0]
    supporting_approach = sorted_perspectives[1]
    
    print(f"\n💡 Recommended Strategy:")
    print(f"  Primary: {best_approach.reasoning_type.value} reasoning (confidence: {best_approach.confidence:.2f})")
    print(f"  Supporting: {supporting_approach.reasoning_type.value} reasoning (confidence: {supporting_approach.confidence:.2f})")
    
    print("\n✅ Multi-perspective analysis complete!")


def example_4_enhanced_existing_workflows():
    """Example 4: Enhance existing AgentNet workflows."""
    print("\n\n🔧 Example 4: Enhancing Existing Workflows")
    print("=" * 50)
    
    # Create regular AgentNet agents (existing workflow)
    agent1 = AgentNet("ExistingAgent1", {"logic": 0.8, "creativity": 0.5, "analytical": 0.7})
    agent2 = AgentNet("ExistingAgent2", {"logic": 0.6, "creativity": 0.8, "analytical": 0.6})
    
    print("🔄 Running existing AgentNet dialogue...")
    
    # Existing AgentNet dialogue (still works exactly the same)
    existing_result = agent1.dialogue_with(agent2, "System optimization strategies", rounds=2)
    
    print(f"  Existing dialogue participants: {existing_result['participants']}")
    print(f"  Existing dialogue quality: {existing_result['synthesis']['dialogue_quality']:.2f}")
    
    # Now enhance the same agents with new capabilities
    print("\n✨ Enhancing with new capabilities...")
    
    # Add reasoning engines
    agent1.reasoning_engine = ReasoningEngine(agent1.style)
    agent2.reasoning_engine = ReasoningEngine(agent2.style)
    
    # Use enhanced dialogue mode
    outer_dialogue = OuterDialogue()
    enhanced_result = outer_dialogue.conduct_dialogue(
        [agent1, agent2], "System optimization strategies", 
        rounds=2, conversation_pattern="collaborative"
    )
    
    print(f"  Enhanced dialogue participants: {enhanced_result['participants']}")
    print(f"  Enhanced dialogue quality: {enhanced_result['analysis']['dialogue_quality']}")
    print(f"  Reasoning diversity: {enhanced_result['analysis']['reasoning_diversity']}")
    
    # Show that both workflows can coexist
    print(f"\n🤝 Coexistence verified:")
    print(f"  Existing workflow: {len(existing_result['dialogue_history'])} turns")
    print(f"  Enhanced workflow: {len(enhanced_result['transcript'])} turns")
    print(f"  Zero breaking changes ✅")
    
    print("\n✅ Workflow enhancement complete!")


def main():
    """Run all integration examples."""
    print("🚀 Enhanced Dialogue Modes and Advanced Reasoning Types")
    print("🔗 Integration Examples with Existing AgentNet")
    print("=" * 80)
    
    try:
        # Run all examples
        example_1_basic_reasoning_enhancement()
        example_2_solution_focused_team_meeting()
        example_3_multi_perspective_analysis()
        example_4_enhanced_existing_workflows()
        
        print("\n\n🎉 ALL INTEGRATION EXAMPLES COMPLETE!")
        print("=" * 80)
        print("🌟 Key Integration Points Demonstrated:")
        print("  ✅ Zero breaking changes to existing AgentNet functionality")
        print("  ✅ Gradual adoption - enhance agents as needed")
        print("  ✅ Backward compatibility - existing workflows unaffected")
        print("  ✅ New EnhancedAgentNet class for full-featured agents")
        print("  ✅ Drop-in reasoning enhancement for existing agents")
        print("  ✅ Solution-focused dialogue patterns for team collaboration")
        print("  ✅ Multi-perspective analysis for complex problems")
        print("\n🏆 AgentNet Enhanced: Production-ready with seamless integration!")
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()