#!/usr/bin/env python3
"""
Demo script showcasing P1 multi-agent polish features.

This demonstrates:
1. Enhanced convergence detection with different strategies
2. Parallel vs sequential execution performance
3. Basic API usage
"""

import asyncio
import json
import time
from AgentNet import AgentNet, ExampleEngine
from api.server import AgentNetAPI


async def demo_convergence_strategies():
    """Demonstrate different convergence strategies."""
    print("🎯 Convergence Strategy Demonstration")
    print("=" * 50)
    
    engine = ExampleEngine()
    agents = [
        AgentNet('DataScientist', {'logic': 0.9, 'creativity': 0.5, 'analytical': 0.9}, engine=engine),
        AgentNet('ProductManager', {'logic': 0.7, 'creativity': 0.8, 'analytical': 0.6}, engine=engine),
        AgentNet('Engineer', {'logic': 0.8, 'creativity': 0.6, 'analytical': 0.8}, engine=engine)
    ]
    
    strategies = [
        ("lexical_only", "Traditional lexical overlap"),
        ("confidence_gated", "Quality-gated convergence"),
        ("lexical_and_semantic", "Combined approach")
    ]
    
    topic = "Machine learning pipeline optimization"
    
    for strategy, description in strategies:
        print(f"\n🧪 Testing: {description}")
        
        # Configure strategy
        for agent in agents:
            agent.dialogue_config.update({
                'convergence_strategy': strategy,
                'convergence_min_overlap': 0.3,
                'convergence_window': 3,
                'use_semantic_convergence': strategy != 'lexical_only',
                'convergence_min_confidence': 0.6
            })
        
        start_time = time.perf_counter()
        session = await agents[0].async_multi_party_dialogue(
            agents=agents,
            topic=f"{topic} - {strategy}",
            rounds=6,
            convergence=True,
            parallel_round=False
        )
        duration = time.perf_counter() - start_time
        
        print(f"  Strategy: {strategy}")
        print(f"  Rounds: {session['rounds_executed']}")
        print(f"  Converged: {session['converged']}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Participants: {len(session['participants'])}")


async def demo_parallel_performance():
    """Demonstrate parallel vs sequential performance."""
    print("\n⚡ Parallel Execution Performance Demo")
    print("=" * 50)
    
    engine = ExampleEngine()
    agents = [
        AgentNet('Architect', {'logic': 0.8, 'creativity': 0.7, 'analytical': 0.8}, engine=engine),
        AgentNet('Security', {'logic': 0.9, 'creativity': 0.4, 'analytical': 0.9}, engine=engine),
        AgentNet('DevOps', {'logic': 0.7, 'creativity': 0.6, 'analytical': 0.7}, engine=engine),
        AgentNet('UX', {'logic': 0.6, 'creativity': 0.9, 'analytical': 0.5}, engine=engine)
    ]
    
    # Configure for fair comparison
    for agent in agents:
        agent.dialogue_config.update({
            'convergence_min_overlap': 0.4,
            'convergence_window': 3,
            'parallel_timeout': 30.0
        })
    
    topic = "Microservice architecture best practices"
    
    # Test parallel execution
    print("\n🔄 Parallel Execution:")
    start_time = time.perf_counter()
    parallel_session = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic=topic,
        rounds=4,
        convergence=True,
        parallel_round=True
    )
    parallel_duration = time.perf_counter() - start_time
    
    print(f"  Duration: {parallel_duration:.2f}s")
    print(f"  Rounds: {parallel_session['rounds_executed']}")
    print(f"  Converged: {parallel_session['converged']}")
    
    # Test sequential execution
    print("\n🔄 Sequential Execution:")
    start_time = time.perf_counter()
    sequential_session = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic=f"{topic} (sequential)",
        rounds=4,
        convergence=True,
        parallel_round=False
    )
    sequential_duration = time.perf_counter() - start_time
    
    print(f"  Duration: {sequential_duration:.2f}s")
    print(f"  Rounds: {sequential_session['rounds_executed']}")
    print(f"  Converged: {sequential_session['converged']}")
    
    # Performance summary
    speedup = sequential_duration / parallel_duration if parallel_duration > 0 else 1
    print(f"\n📊 Performance Summary:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {sequential_duration - parallel_duration:.2f}s")


async def demo_api_usage():
    """Demonstrate basic API usage."""
    print("\n🌐 Basic API Demonstration")
    print("=" * 50)
    
    # Create API instance
    api = AgentNetAPI()
    
    # Create a session
    request_data = {
        "topic": "Sustainable software development",
        "agents": [
            {"name": "GreenTech", "style": {"logic": 0.8, "creativity": 0.7, "analytical": 0.8}},
            {"name": "Efficiency", "style": {"logic": 0.9, "creativity": 0.5, "analytical": 0.9}}
        ],
        "mode": "consensus",
        "max_rounds": 5,
        "convergence": True,
        "parallel_round": True,
        "convergence_config": {
            "convergence_min_overlap": 0.3,
            "convergence_window": 3,
            "convergence_strategy": "lexical_and_semantic"
        }
    }
    
    print("📝 Creating session...")
    session_response = api.create_session(request_data)
    session_id = session_response["session_id"]
    print(f"  Session ID: {session_id}")
    print(f"  Status: {session_response['status']}")
    print(f"  Participants: {session_response['participants']}")
    
    print("\n📊 Session status before execution:")
    status = api.get_session_status(session_id)
    print(f"  Status: {status['status']}")
    print(f"  Current round: {status['current_round']}")
    print(f"  Total rounds: {status['total_rounds']}")
    
    print("\n🚀 Running session...")
    
    result = await api.run_session(session_id)
    
    print(f"  Final status: {result['status']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Rounds executed: {result['rounds_executed']}")
    print(f"  Topic evolution: {result.get('topic_start', 'N/A')} → {result.get('topic_final', 'N/A')}")
    
    print("\n📋 Final session state:")
    final_session = api.get_session(session_id)
    print(f"  Status: {final_session['status']}")
    print(f"  Transcript entries: {len(final_session.get('transcript', []))}")


async def main():
    """Run the complete P1 feature demonstration."""
    print("🚀 AgentNet P1 Multi-Agent Polish Demo")
    print("=" * 60)
    print("Showcasing: Enhanced Convergence, Parallel Execution, Basic API")
    print("=" * 60)
    
    try:
        # Demo 1: Convergence strategies
        await demo_convergence_strategies()
        
        # Demo 2: Parallel performance
        await demo_parallel_performance()
        
        # Demo 3: API usage
        await demo_api_usage()
        
        print("\n" + "=" * 60)
        print("🎉 P1 Feature Demo Complete!")
        print("✅ Enhanced convergence detection")
        print("✅ Parallel execution with monitoring")
        print("✅ Basic API foundation")
        print("✅ Comprehensive testing framework")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())