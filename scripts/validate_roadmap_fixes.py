#!/usr/bin/env python3
"""
Validation script to verify roadmap issues have been resolved.

This script checks:
1. networkx dependency is available
2. DAG planner can import and use networkx
3. Docker deployment files exist
4. All tests pass

DEPRECATED: This script is deprecated and will be removed in a future release.
Please use the unified CLI instead:
    python -m cli.main validate-roadmap
    or: python cli/main.py validate-roadmap
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def main():
    """Run all validation checks."""
    print("⚠️  DEPRECATION WARNING: This script is deprecated.")
    print("    Please use: python -m cli.main validate-roadmap")
    print()

    # Import and run the validation
    from devtools.validation import validate_roadmap

    return validate_roadmap()


def check_networkx():
    """Verify networkx is available."""
    try:
        import networkx as nx

        print(f"✅ networkx is available (version {nx.__version__})")
        return True
    except ImportError as e:
        print(f"❌ networkx is NOT available: {e}")
        return False


def check_dag_planner():
    """Verify DAG planner can use networkx."""
    try:
        from agentnet.core.orchestration.dag_planner import DAGPlanner

        print("✅ DAGPlanner successfully imports and uses networkx")
        return True
    except ImportError as e:
        print(f"❌ DAGPlanner import failed: {e}")
        return False


def check_docker_files():
    """Verify Docker deployment files exist."""
    files = [
        ("Dockerfile", "Main Docker image definition"),
        ("docker-compose.yml", "Multi-service deployment configuration"),
        (".dockerignore", "Docker build exclusions"),
        ("DOCKER.md", "Docker deployment documentation"),
        ("configs/prometheus.yml", "Prometheus configuration"),
    ]

    all_exist = True
    for filename, description in files:
        if Path(filename).exists():
            print(f"✅ {filename} exists ({description})")
        else:
            print(f"❌ {filename} is MISSING ({description})")
            all_exist = False

    return all_exist


def check_requirements_txt():
    """Verify networkx is in requirements.txt."""
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            if "networkx" in content:
                print("✅ networkx is listed in requirements.txt")
                return True
            else:
                print("❌ networkx is NOT in requirements.txt")
                return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False


if __name__ == "__main__":
    sys.exit(main())
