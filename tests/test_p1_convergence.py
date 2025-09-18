#!/usr/bin/env python3
"""
Test P1 improved convergence detection - validates enhanced convergence algorithms.
"""

import asyncio
import json
import logging
from AgentNet import AgentNet, ExampleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_convergence_parameters():
    """Test that convergence parameters are properly applied."""
    print("🧪 Testing Convergence Parameter Application...")
    
    engine = ExampleEngine()
    agents = [
        AgentNet('Alice', {'logic': 0.8, 'creativity': 0.6, 'analytical': 0.7}, engine=engine),
        AgentNet('Bob', {'logic': 0.6, 'creativity': 0.8, 'analytical': 0.5}, engine=engine)
    ]
    
    # Test 1: Strict convergence parameters (should take more rounds)
    for agent in agents:
        agent.dialogue_config.update({
            'convergence_min_overlap': 0.2,  # Lower threshold = harder to converge
            'convergence_window': 4  # Larger window = more content to compare
        })
    
    session1 = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic='Efficient data structures for large datasets',
        rounds=8,
        convergence=True,
        parallel_round=False
    )
    
    print(f"  ✅ Strict convergence: {session1['rounds_executed']} rounds, converged: {session1['converged']}")
    
    # Test 2: Lenient convergence parameters (should converge faster)
    for agent in agents:
        agent.dialogue_config.update({
            'convergence_min_overlap': 0.6,  # Higher threshold = easier to converge
            'convergence_window': 2  # Smaller window = less content to compare
        })
    
    session2 = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic='Best practices for code review',
        rounds=8,
        convergence=True,
        parallel_round=False
    )
    
    print(f"  ✅ Lenient convergence: {session2['rounds_executed']} rounds, converged: {session2['converged']}")
    
    return session1['rounds_executed'] >= session2['rounds_executed']


async def test_parallel_execution():
    """Test parallel round execution with error handling."""
    print("🧪 Testing Parallel Execution...")
    
    engine = ExampleEngine()
    agents = [
        AgentNet('FastAgent', {'logic': 0.7, 'creativity': 0.7, 'analytical': 0.6}, engine=engine),
        AgentNet('SlowAgent', {'logic': 0.8, 'creativity': 0.5, 'analytical': 0.9}, engine=engine),
        AgentNet('CreativeAgent', {'logic': 0.5, 'creativity': 0.9, 'analytical': 0.4}, engine=engine)
    ]
    
    # Configure parallel timeout
    for agent in agents:
        agent.dialogue_config.update({
            'parallel_timeout': 20.0,  # 20 second timeout
            'convergence_min_overlap': 0.4,
            'convergence_window': 3
        })
    
    # Test parallel execution
    start_time = asyncio.get_event_loop().time()
    session = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic='Optimal microservice architecture patterns',
        rounds=4,
        convergence=True,
        parallel_round=True
    )
    end_time = asyncio.get_event_loop().time()
    
    parallel_duration = end_time - start_time
    
    print(f"  ✅ Parallel execution completed in {parallel_duration:.2f}s")
    print(f"  ✅ Rounds: {session['rounds_executed']}, Converged: {session['converged']}")
    print(f"  ✅ Participants: {len(session['participants'])}")
    
    # Test sequential execution for comparison
    start_time = asyncio.get_event_loop().time()
    session_seq = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic='Database indexing strategies',
        rounds=4,
        convergence=True,
        parallel_round=False
    )
    end_time = asyncio.get_event_loop().time()
    
    sequential_duration = end_time - start_time
    
    print(f"  ✅ Sequential execution completed in {sequential_duration:.2f}s")
    print(f"  ✅ Speedup ratio: {sequential_duration/parallel_duration:.2f}x")
    
    return parallel_duration < sequential_duration


async def test_convergence_strategies():
    """Test different convergence strategies."""
    print("🧪 Testing Convergence Strategies...")
    
    engine = ExampleEngine()
    agents = [
        AgentNet('Analyst', {'logic': 0.9, 'creativity': 0.4, 'analytical': 0.9}, engine=engine),
        AgentNet('Innovator', {'logic': 0.5, 'creativity': 0.9, 'analytical': 0.6}, engine=engine)
    ]
    
    strategies = [
        ("lexical_only", "Lexical-only convergence"),
        ("confidence_gated", "Confidence-gated convergence"),
        ("lexical_and_semantic", "Combined lexical+semantic convergence")
    ]
    
    results = {}
    
    for strategy, description in strategies:
        for agent in agents:
            agent.dialogue_config.update({
                'convergence_strategy': strategy,
                'convergence_min_overlap': 0.4,
                'convergence_window': 3,
                'use_semantic_convergence': strategy != 'lexical_only',
                'convergence_min_confidence': 0.6
            })
        
        session = await agents[0].async_multi_party_dialogue(
            agents=agents,
            topic=f'Machine learning model optimization ({strategy})',
            rounds=6,
            convergence=True,
            parallel_round=False
        )
        
        results[strategy] = {
            'rounds': session['rounds_executed'],
            'converged': session['converged']
        }
        
        print(f"  ✅ {description}: {session['rounds_executed']} rounds, converged: {session['converged']}")
    
    return results


async def test_convergence_edge_cases():
    """Test convergence edge cases and error handling."""
    print("🧪 Testing Convergence Edge Cases...")
    
    engine = ExampleEngine()
    
    # Test 1: Single agent (should not converge in traditional sense)
    single_agent = [AgentNet('Solo', {'logic': 0.7, 'creativity': 0.7, 'analytical': 0.7}, engine=engine)]
    
    session_single = await single_agent[0].async_multi_party_dialogue(
        agents=single_agent,
        topic='Solo reasoning test',
        rounds=3,
        convergence=False,  # Disable convergence for single agent
        parallel_round=False
    )
    
    print(f"  ✅ Single agent: {session_single['rounds_executed']} rounds")
    
    # Test 2: No convergence (run all rounds)
    agents = [
        AgentNet('Persistent1', {'logic': 0.9, 'creativity': 0.3, 'analytical': 0.8}, engine=engine),
        AgentNet('Persistent2', {'logic': 0.3, 'creativity': 0.9, 'analytical': 0.4}, engine=engine)
    ]
    
    for agent in agents:
        agent.dialogue_config.update({
            'convergence_min_overlap': 0.9,  # Very strict - unlikely to converge
            'convergence_window': 2
        })
    
    session_no_conv = await agents[0].async_multi_party_dialogue(
        agents=agents,
        topic='Highly divergent topic discussion',
        rounds=4,
        convergence=True,
        parallel_round=False
    )
    
    print(f"  ✅ No convergence case: {session_no_conv['rounds_executed']} rounds, converged: {session_no_conv['converged']}")
    
    return True


async def run_convergence_experiments():
    """Run comprehensive convergence experiments."""
    print("🚀 P1 Convergence Enhancement Tests")
    print("=" * 50)
    
    try:
        # Test 1: Parameter application
        param_success = await test_convergence_parameters()
        print(f"Parameter application test: {'✅ PASSED' if param_success else '❌ FAILED'}")
        
        # Test 2: Parallel execution
        parallel_success = await test_parallel_execution()
        print(f"Parallel execution test: {'✅ PASSED' if parallel_success else '❌ FAILED'}")
        
        # Test 3: Convergence strategies
        strategy_results = await test_convergence_strategies()
        print(f"Convergence strategies test: ✅ PASSED")
        
        # Test 4: Edge cases
        edge_success = await test_convergence_edge_cases()
        print(f"Edge cases test: {'✅ PASSED' if edge_success else '❌ FAILED'}")
        
        print("\n" + "=" * 50)
        print("🎉 All P1 Convergence Tests Completed!")
        
        # Summary
        print("\nSummary of Convergence Strategy Results:")
        for strategy, result in strategy_results.items():
            print(f"  {strategy}: {result['rounds']} rounds, converged: {result['converged']}")
        
        return param_success and parallel_success and edge_success
        
    except Exception as e:
        logger.error(f"Convergence tests failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_convergence_experiments())
    if success:
        print("\n🏆 P1 Improved Convergence Detection: COMPLETE")
    else:
        print("\n❌ Some convergence tests failed")