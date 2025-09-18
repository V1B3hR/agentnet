#!/usr/bin/env python3
"""Phase 0 Demo: Bootstrap & Skeleton functionality showcase.

This demo demonstrates the core Phase 0 features:
- Single agent reasoning with style modulation
- Basic monitoring and event emission
- Session persistence
- Structured logging
- Simple orchestrator loop (linear turns)
"""

import json
import logging
import time
from pathlib import Path

from agentnet import AgentNet, ExampleEngine, MonitorFactory, MonitorSpec, Severity


def setup_logging():
    """Setup structured logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path('demo_output') / 'phase0_demo.log')
        ]
    )
    return logging.getLogger("phase0_demo")


def create_demo_agent(logger):
    """Create a demonstration agent with monitors."""
    logger.info("🤖 Creating demo agent with monitors...")
    
    # Create engine
    engine = ExampleEngine()
    
    # Create monitors
    keyword_monitor = MonitorFactory.build(MonitorSpec(
        name="demo_content_filter",
        type="keyword", 
        params={"keywords": ["error", "fail", "bad"], "violation_name": "negative_content"},
        severity=Severity.MINOR,
        description="Filter negative content in demo"
    ))
    
    # Create agent with style and monitors
    agent = AgentNet(
        name="PhaseZeroBot",
        style={"logic": 0.8, "creativity": 0.6, "analytical": 0.9},
        engine=engine,
        monitors=[keyword_monitor]
    )
    
    logger.info(f"✅ Agent '{agent.name}' created with style: {agent.style}")
    return agent


def demonstrate_single_reasoning(agent, logger):
    """Demonstrate single agent reasoning."""
    logger.info("🧠 Demonstrating single agent reasoning...")
    
    tasks = [
        "Explain the benefits of modular software architecture",
        "Design a simple monitoring system for distributed applications",
        "What are the key principles of good API design?"
    ]
    
    results = []
    for i, task in enumerate(tasks, 1):
        logger.info(f"  Task {i}: {task}")
        
        start_time = time.time()
        result = agent.generate_reasoning_tree(
            task, 
            include_monitor_trace=True,
            max_depth=2
        )
        duration = time.time() - start_time
        
        results.append({
            "task": task,
            "result": result,
            "duration": duration
        })
        
        logger.info(f"  ✅ Completed in {duration:.2f}s")
        logger.info(f"  📝 Content preview: {result['result']['content'][:100]}...")
        
        if result.get('monitor_trace'):
            logger.info(f"  🔍 Monitor events: {len(result['monitor_trace'])}")
    
    return results


def demonstrate_linear_orchestration(agent, logger):
    """Demonstrate minimal orchestrator loop (linear turns)."""
    logger.info("🔄 Demonstrating linear orchestration...")
    
    # Simulate a conversation where each turn builds on the previous
    conversation_context = ""
    turns = [
        "What is a multi-agent system?",
        "How do agents communicate with each other?", 
        "What are the challenges in multi-agent coordination?",
        "How can we address these challenges?"
    ]
    
    conversation_history = []
    
    for turn_num, prompt in enumerate(turns, 1):
        logger.info(f"  Turn {turn_num}: {prompt}")
        
        # Build context-aware prompt
        if conversation_context:
            full_prompt = f"Previous context: {conversation_context}\n\nCurrent question: {prompt}"
        else:
            full_prompt = prompt
            
        # Generate response
        result = agent.generate_reasoning_tree(full_prompt)
        response = result['result']['content']
        
        # Update context for next turn
        conversation_context += f"\nQ{turn_num}: {prompt}\nA{turn_num}: {response}"
        
        conversation_history.append({
            "turn": turn_num,
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        })
        
        logger.info(f"  ✅ Turn {turn_num} completed")
        logger.info(f"  📝 Response preview: {response[:80]}...")
    
    return conversation_history


def demonstrate_persistence(agent, logger):
    """Demonstrate session persistence."""
    logger.info("💾 Demonstrating session persistence...")
    
    # Create session data
    session_data = {
        "session_id": f"phase0_demo_{int(time.time())}",
        "agent_name": agent.name,
        "agent_style": agent.style,
        "created_at": time.time(),
        "tasks_completed": 3,
        "demo_type": "Phase 0 Bootstrap",
        "metadata": {
            "python_version": "3.8+",
            "agentnet_version": "0.5.0",
            "features_demonstrated": [
                "single_agent_reasoning",
                "monitoring",
                "linear_orchestration", 
                "session_persistence"
            ]
        }
    }
    
    # Persist session
    session_file = agent.persist_session(session_data)
    logger.info(f"  ✅ Session persisted to: {session_file}")
    
    # Verify file exists
    from pathlib import Path
    if Path(session_file).exists():
        logger.info(f"  ✅ Session file verified: {Path(session_file).stat().st_size} bytes")
    else:
        logger.warning(f"  ⚠️ Session file not found: {session_file}")
    
    return session_file


def demonstrate_structured_events(logger):
    """Demonstrate structured event model."""
    logger.info("📊 Demonstrating structured event model...")
    
    # Example structured events
    events = [
        {
            "event_type": "agent_created",
            "timestamp": time.time(),
            "agent_name": "PhaseZeroBot",
            "agent_style": {"logic": 0.8, "creativity": 0.6},
            "monitors_attached": 1
        },
        {
            "event_type": "reasoning_started", 
            "timestamp": time.time(),
            "task": "Architecture explanation",
            "max_depth": 2
        },
        {
            "event_type": "monitor_triggered",
            "timestamp": time.time(),
            "monitor_name": "demo_content_filter",
            "severity": "minor",
            "violation": "negative_content"
        },
        {
            "event_type": "session_persisted",
            "timestamp": time.time(),
            "session_id": "demo_session",
            "file_path": "sessions/demo.json"
        }
    ]
    
    # Log structured events
    for event in events:
        logger.info(f"  📋 Event: {json.dumps(event, indent=2)}")
    
    # Save to file for analysis
    events_file = Path('demo_output') / 'structured_events.json'
    events_file.parent.mkdir(exist_ok=True)
    
    with open(events_file, 'w') as f:
        json.dump(events, f, indent=2)
        
    logger.info(f"  ✅ Structured events saved to: {events_file}")
    return events


def main():
    """Run the Phase 0 demonstration."""
    print("🚀 AgentNet Phase 0 Demo: Bootstrap & Skeleton")
    print("=" * 60)
    
    # Setup
    Path('demo_output').mkdir(exist_ok=True)
    logger = setup_logging()
    
    logger.info("Starting Phase 0 demonstration")
    
    try:
        # 1. Create agent
        agent = create_demo_agent(logger)
        
        # 2. Single agent reasoning
        reasoning_results = demonstrate_single_reasoning(agent, logger)
        
        # 3. Linear orchestration
        conversation = demonstrate_linear_orchestration(agent, logger)
        
        # 4. Session persistence
        session_file = demonstrate_persistence(agent, logger)
        
        # 5. Structured events
        events = demonstrate_structured_events(logger)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 Phase 0 Demo Complete!")
        print("=" * 60)
        print(f"✅ Agent created: {agent.name}")
        print(f"✅ Reasoning tasks completed: {len(reasoning_results)}")
        print(f"✅ Conversation turns: {len(conversation)}")
        print(f"✅ Session persisted: {session_file}")
        print(f"✅ Structured events: {len(events)}")
        print(f"\n📁 Output files in: demo_output/")
        print(f"📋 Log file: demo_output/phase0_demo.log")
        
        logger.info("Phase 0 demonstration completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())