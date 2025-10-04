#!/usr/bin/env python3
"""
Phase 7 Advanced Intelligence & Reasoning Demo

Demonstrates the new Phase 7 capabilities:
- Advanced Reasoning Engine with chain-of-thought and multi-hop reasoning
- Enhanced Memory Systems with temporal reasoning and cross-modal linking
- AI-Powered Agent Evolution with self-improvement capabilities
"""

import json
import time
from datetime import datetime
from pathlib import Path

import agentnet
from agentnet import (
    AgentNet,
    AdvancedReasoningEngine,
    ChainOfThoughtReasoning,
    MultiHopReasoning,
    EnhancedEpisodicMemory,
    AgentEvolutionManager,
    TemporalReasoning,
)


def demo_advanced_reasoning():
    """Demonstrate advanced reasoning capabilities."""
    print("\n🧠 Phase 7: Advanced Reasoning Engine Demo")
    print("=" * 50)
    
    # Create style weights for reasoning
    style_weights = {
        "logic": 0.8,
        "creativity": 0.6, 
        "analytical": 0.9
    }
    
    # Initialize advanced reasoning engine
    reasoning_engine = AdvancedReasoningEngine(style_weights)
    
    # Test chain-of-thought reasoning
    print("\n1. Chain-of-Thought Reasoning:")
    cot_task = "How can we solve climate change through technological innovation?"
    cot_result = reasoning_engine.advanced_reason(cot_task, "chain_of_thought")
    
    print(f"Task: {cot_task}")
    print(f"Result: {cot_result.content}")
    print(f"Confidence: {cot_result.confidence:.2f}")
    print(f"Steps: {len(cot_result.reasoning_steps)}")
    for i, step in enumerate(cot_result.reasoning_steps[:3], 1):
        print(f"  {i}. {step}")
    
    # Test multi-hop reasoning
    print("\n2. Multi-Hop Reasoning:")
    multihop_task = "Connect renewable energy to economic growth through job creation"
    context = {
        "knowledge_graph": {
            "nodes": [
                {"id": "renewable_energy", "content": "Clean energy sources", "type": "technology"},
                {"id": "job_creation", "content": "Employment opportunities", "type": "economic"},
                {"id": "economic_growth", "content": "GDP and prosperity", "type": "outcome"}
            ],
            "edges": [
                {"source": "renewable_energy", "target": "job_creation", "relation": "enables"},
                {"source": "job_creation", "target": "economic_growth", "relation": "drives"}
            ]
        }
    }
    
    multihop_result = reasoning_engine.advanced_reason(multihop_task, "multi_hop", context)
    print(f"Task: {multihop_task}")
    print(f"Result: {multihop_result.content}")
    print(f"Confidence: {multihop_result.confidence:.2f}")
    
    # Test counterfactual reasoning
    print("\n3. Counterfactual Reasoning:")
    counterfactual_task = "If electric vehicles were never invented, how would urban transportation develop?"
    cf_result = reasoning_engine.advanced_reason(counterfactual_task, "counterfactual")
    
    print(f"Task: {counterfactual_task}")
    print(f"Result: {cf_result.content}")
    print(f"Confidence: {cf_result.confidence:.2f}")
    
    # Test hybrid reasoning
    print("\n4. Hybrid Reasoning (Multiple Modes):")
    hybrid_task = "Analyze the future of artificial intelligence"
    modes = ["chain_of_thought", "multi_hop", "counterfactual"]
    hybrid_results = reasoning_engine.hybrid_reasoning(hybrid_task, modes)
    
    print(f"Task: {hybrid_task}")
    print(f"Modes applied: {modes}")
    print(f"Results obtained: {len(hybrid_results)}")
    
    if hybrid_results:
        best_result = max(hybrid_results, key=lambda r: r.confidence)
        print(f"Best mode: {best_result.reasoning_type.value}")
        print(f"Best confidence: {best_result.confidence:.2f}")
        avg_confidence = sum(r.confidence for r in hybrid_results) / len(hybrid_results)
        print(f"Average confidence: {avg_confidence:.2f}")


def demo_enhanced_memory():
    """Demonstrate enhanced memory capabilities."""
    print("\n🧠 Phase 7: Enhanced Memory Systems Demo")
    print("=" * 50)
    
    # Initialize enhanced episodic memory
    memory_config = {
        "storage_path": "/tmp/demo_enhanced_memory.json",
        "enhanced_episodic": True,
        "hierarchy_depth": 3,
        "consolidation_interval": 10.0  # 10 seconds for demo
    }
    
    enhanced_memory = EnhancedEpisodicMemory(memory_config)
    
    # Store different types of memories
    memories = [
        {
            "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "metadata": {"importance_score": 0.8, "language": "python"},
            "tags": ["code", "algorithm", "recursion"],
            "agent_name": "CodeAgent"
        },
        {
            "content": "Climate change data shows rising temperatures over the last century",
            "metadata": {"importance_score": 0.9, "data_type": "environmental"},
            "tags": ["data", "climate", "trend"],
            "agent_name": "DataAnalyst"
        },
        {
            "content": "Meeting scheduled for tomorrow at 2 PM to discuss project milestones",
            "metadata": {"importance_score": 0.6, "event_type": "meeting"},
            "tags": ["temporal", "schedule", "project"],
            "agent_name": "ProjectManager"
        }
    ]
    
    # Store memories
    print("\n1. Storing Cross-Modal Memories:")
    for i, mem_data in enumerate(memories):
        memory_entry = agentnet.MemoryEntry(
            content=mem_data["content"],
            metadata=mem_data["metadata"],
            timestamp=time.time() + i,  # Slight time offset
            agent_name=mem_data["agent_name"],
            tags=mem_data["tags"]
        )
        enhanced_memory.store(memory_entry)
        print(f"  Stored: {mem_data['content'][:50]}...")
    
    # Test retrieval with temporal reasoning
    print("\n2. Enhanced Retrieval with Temporal Reasoning:")
    retrieval_result = enhanced_memory.retrieve(
        query="algorithm data meeting",
        limit=5,
        use_temporal_reasoning=True
    )
    
    print(f"Found {len(retrieval_result.entries)} memories:")
    for entry in retrieval_result.entries:
        print(f"  - {entry.content[:50]}... (by {entry.agent_name})")
    
    # Get memory hierarchy
    print("\n3. Hierarchical Knowledge Organization:")
    hierarchy = enhanced_memory.get_memory_hierarchy()
    print(f"Memory organized into {len(hierarchy)} levels:")
    for level, groups in hierarchy.items():
        print(f"  {level}: {len(groups)} concept groups")
    
    # Force consolidation to run
    print("\n4. Forcing Memory Consolidation:")
    consolidation_result = enhanced_memory.force_consolidate()
    print(f"  Status: {consolidation_result.get('status', 'completed')}")
    print(f"  Clusters created: {consolidation_result.get('clusters_created', 0)}")
    print(f"  Memories consolidated: {consolidation_result.get('memories_consolidated', 0)}")
    print(f"  Memories forgotten: {consolidation_result.get('memories_forgotten', 0)}")
    
    # Get consolidation status
    print("\n5. Memory Consolidation Status:")
    consolidation_status = enhanced_memory.get_consolidation_status()
    print(f"  Clusters: {consolidation_status['clusters_count']}")
    print(f"  Rules: {consolidation_status['consolidation_rules']}")


def demo_agent_evolution():
    """Demonstrate AI-powered agent evolution."""
    print("\n🧠 Phase 7: AI-Powered Agent Evolution Demo")
    print("=" * 50)
    
    # Initialize evolution manager
    evolution_config = {
        "learning_rate": 0.2,
        "min_pattern_frequency": 2,
        "rl_learning_rate": 0.15,
        "discount_factor": 0.9
    }
    
    evolution_manager = AgentEvolutionManager(evolution_config)
    
    # Simulate task results for learning
    task_results = [
        {
            "task_id": "analysis_1",
            "task_type": "data_analysis", 
            "content": "Analyze sales data trends",
            "success": True,
            "duration": 45.0,
            "difficulty": 0.7,
            "skills_used": ["data_analysis", "visualization"],
            "confidence": 0.85,
            "metadata": {"complexity": "medium"}
        },
        {
            "task_id": "analysis_2",
            "task_type": "data_analysis",
            "content": "Process customer feedback data", 
            "success": True,
            "duration": 38.0,
            "difficulty": 0.6,
            "skills_used": ["data_analysis", "nlp"],
            "confidence": 0.92,
            "metadata": {"complexity": "medium"}
        },
        {
            "task_id": "coding_1",
            "task_type": "code_generation",
            "content": "Write Python function for data processing",
            "success": False,
            "duration": 120.0,
            "difficulty": 0.8,
            "skills_used": ["programming", "python"],
            "confidence": 0.45,
            "metadata": {"complexity": "high"}
        },
        {
            "task_id": "analysis_3",
            "task_type": "data_analysis",
            "content": "Statistical analysis of survey results",
            "success": True,
            "duration": 52.0,
            "difficulty": 0.75,
            "skills_used": ["data_analysis", "statistics"],
            "confidence": 0.88,
            "metadata": {"complexity": "high"}
        }
    ]
    
    print("\n1. Evolving Agent Capabilities:")
    agent_id = "DataAgent"
    
    # Evolve agent based on task results
    evolution_report = evolution_manager.evolve_agent(agent_id, task_results)
    
    print(f"Evolution report for {agent_id}:")
    print(f"  New skills: {evolution_report.get('new_skills', [])}")
    print(f"  Improvements: {len(evolution_report.get('improvements', []))}")
    print(f"  Specialization changes: {len(evolution_report.get('specialization_changes', []))}")
    
    # Get current capabilities
    print("\n2. Current Agent Capabilities:")
    capabilities = evolution_manager.get_agent_capabilities(agent_id)
    
    print(f"  Total skills: {capabilities['skill_count']}")
    print(f"  Average proficiency: {capabilities['avg_proficiency']:.2f}")
    
    if capabilities['most_used_skills']:
        print("  Most used skills:")
        for skill_name, skill in capabilities['most_used_skills'][:3]:
            print(f"    - {skill_name}: proficiency {skill.proficiency:.2f}, used {skill.usage_count} times")
    
    # Get improvement recommendations
    print("\n3. Improvement Recommendations:")
    recommendations = evolution_manager.recommend_agent_improvements(agent_id)
    
    print(f"  Skill recommendations: {recommendations.get('skill_recommendations', [])}")
    
    bottlenecks = recommendations.get('performance_bottlenecks', [])
    if bottlenecks:
        print("  Performance bottlenecks:")
        for bottleneck in bottlenecks[:2]:
            print(f"    - {bottleneck['type']}: {bottleneck.get('recommendation', 'N/A')}")
    
    # Test skill transfer
    print("\n4. Skill Transfer Learning:")
    skill_engine = evolution_manager.skill_engine
    
    # Demonstrate skill transfer
    transfer_success = skill_engine.transfer_skill(
        source_skill="data_analysis",
        target_skill="business_analysis", 
        similarity=0.7
    )
    
    if transfer_success:
        print("  Successfully transferred 'data_analysis' skill to 'business_analysis'")
        
        if "business_analysis" in skill_engine.skill_library:
            new_skill = skill_engine.skill_library["business_analysis"]
            print(f"  New skill proficiency: {new_skill.proficiency:.2f}")


def demo_integrated_agent():
    """Demonstrate integrated AgentNet with Phase 7 capabilities."""
    print("\n🧠 Phase 7: Integrated AgentNet Demo")
    print("=" * 50)
    
    # Create agent with Phase 7 capabilities
    style_weights = {
        "logic": 0.8,
        "creativity": 0.7,
        "analytical": 0.85
    }
    
    memory_config = {
        "enhanced_episodic": True,
        "storage_path": "/tmp/demo_agent_memory.json",
        "evolution": {
            "learning_rate": 0.15,
            "min_pattern_frequency": 2
        }
    }
    
    agent = AgentNet(
        name="Phase7Agent",
        style=style_weights,
        memory_config=memory_config
    )
    
    print(f"\n1. Created agent: {agent.name}")
    print(f"   Phase 7 enabled: {agent.advanced_reasoning_engine is not None}")
    print(f"   Enhanced memory: {agent.enhanced_memory is not None}")
    print(f"   Evolution manager: {agent.evolution_manager is not None}")
    
    # Test advanced reasoning
    if agent.advanced_reasoning_engine:
        print("\n2. Advanced Reasoning Test:")
        task = "How can AI help solve global challenges while ensuring ethical use?"
        
        result = agent.advanced_reason(
            task=task,
            reasoning_mode="auto",
            use_temporal=True
        )
        
        print(f"   Task: {task}")
        print(f"   Mode used: {result['primary_reasoning']['mode']}")
        print(f"   Confidence: {result['primary_reasoning']['confidence']:.2f}")
        print(f"   Temporal reasoning: {'Yes' if result['temporal_reasoning'] else 'No'}")
        
        # Test hybrid reasoning
        print("\n3. Hybrid Reasoning Test:")
        hybrid_result = agent.hybrid_reasoning(
            task=task,
            modes=["chain_of_thought", "counterfactual"]
        )
        
        print(f"   Modes applied: {hybrid_result['modes_applied']}")
        print(f"   Results count: {len(hybrid_result['individual_results'])}")
        print(f"   Best result confidence: {hybrid_result['synthesis']['avg_confidence']:.2f}")
    
    # Test agent evolution
    if agent.evolution_manager:
        print("\n4. Agent Evolution Test:")
        
        # Simulate some task results
        mock_results = [
            {
                "task_type": "reasoning",
                "success": True,
                "confidence": 0.8,
                "skills_used": ["advanced_reasoning", "problem_solving"]
            }
        ]
        
        evolution_report = agent.evolve_capabilities(mock_results)
        if "error" not in evolution_report:
            print(f"   Evolution successful: {len(evolution_report.get('improvements', []))} improvements")
        
        # Get capabilities
        capabilities = agent.get_agent_capabilities()
        if "error" not in capabilities:
            basic_info = capabilities.get('basic_info', {})
            print(f"   Agent has memory: {basic_info.get('has_memory', False)}")
            print(f"   Phase 7 enabled: {capabilities.get('phase7_enabled', False)}")


def main():
    """Run all Phase 7 demos."""
    print("🚀 AgentNet Phase 7 – Advanced Intelligence & Reasoning Demo")
    print("=" * 70)
    print(f"AgentNet Version: {agentnet.__version__}")
    print(f"Phase Status: {agentnet.__phase_status__}")
    
    # Create demo output directory
    Path("/tmp").mkdir(exist_ok=True)
    
    try:
        # Run individual component demos
        demo_advanced_reasoning()
        demo_enhanced_memory()
        demo_agent_evolution()
        demo_integrated_agent()
        
        print("\n✅ Phase 7 Demo Completed Successfully!")
        print("\nPhase 7 Features Demonstrated:")
        print("  ✓ Chain-of-thought reasoning with step validation")
        print("  ✓ Multi-hop reasoning across knowledge graphs")
        print("  ✓ Counterfactual analysis")
        print("  ✓ Hybrid reasoning modes")
        print("  ✓ Enhanced episodic memory with temporal reasoning")
        print("  ✓ Hierarchical knowledge organization")
        print("  ✓ Cross-modal memory linking")
        print("  ✓ Memory consolidation mechanisms")
        print("  ✓ AI-powered agent evolution")
        print("  ✓ Dynamic skill acquisition and transfer")
        print("  ✓ Task pattern analysis")
        print("  ✓ Performance-based agent optimization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()