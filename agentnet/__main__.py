#!/usr/bin/env python3
"""
AgentNet CLI entry point.

This module provides command-line access to AgentNet functionality
via `python -m agentnet`. It delegates to the main CLI implementation
in the root AgentNet.py file.

Usage:
    python -m agentnet --help
    python -m agentnet --demo sync
    python -m agentnet --demo async --rounds 5
    python -m agentnet --demo experimental
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import AgentNet.py
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

try:
    # Import and execute the main CLI function from AgentNet.py
    from AgentNet import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing AgentNet CLI: {e}")
    print("Make sure AgentNet.py is available in the root directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error running AgentNet CLI: {e}")
    sys.exit(1)