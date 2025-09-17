"""Backward compatibility layer for original AgentNet.py functionality.

This module maintains compatibility with existing code that imports from AgentNet.py
while using the new modular structure internally.
"""

# Import everything from the new modular structure
from agentnet import *

# Import from original file for complex functionality not yet refactored
import sys
import os

# Add current directory to path to import from original file
original_file = os.path.join(os.path.dirname(__file__), 'AgentNet.py')

# For backward compatibility, we'll selectively import complex functions
# that haven't been refactored yet but keep the core classes from the new structure

# Re-export the main classes for compatibility
__all__ = [
    "AgentNet", "ExampleEngine", "Severity", "CognitiveFault", 
    "MonitorFactory", "MonitorManager", "SessionManager"
]

# Legacy compatibility function
def _demo():
    """Legacy demo function for backward compatibility."""
    from agentnet import AgentNet, ExampleEngine
    
    print("Running P0 refactored demo...")
    engine = ExampleEngine()
    
    # Create agents
    agent_a = AgentNet("LogicEngine", {"logic": 0.9, "creativity": 0.3}, engine=engine)
    agent_b = AgentNet("CreativeSpeak", {"logic": 0.4, "creativity": 0.85}, engine=engine)
    
    # Single agent reasoning
    print("\n=== Single Agent Reasoning ===")
    result = agent_a.generate_reasoning_tree("How to design secure distributed systems?")
    print(f"Agent A reasoning: {result['result']['content']}")
    
    result = agent_b.generate_reasoning_tree("Creative approaches to system resilience?")
    print(f"Agent B reasoning: {result['result']['content']}")
    
    # Session persistence
    print("\n=== Session Persistence ===")
    session_data = {
        "session_id": f"demo_{int(time.time())}",
        "participants": [agent_a.name, agent_b.name],
        "topic": "System design discussion",
        "rounds_executed": 2,
        "converged": False,
        "timestamp": time.time(),
        "transcript": [
            {"agent": agent_a.name, "content": result['result']['content'], "round": 1},
            {"agent": agent_b.name, "content": result['result']['content'], "round": 2}
        ]
    }
    
    filepath = agent_a.persist_session(session_data)
    print(f"Session persisted to: {filepath}")
    
    print("\nP0 refactoring demonstration complete!")

if __name__ == "__main__":
    import time
    _demo()