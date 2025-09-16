"""
Unified Experiment Runner for AgentNet.
Registry dispatcher that runs experiments via `python -m experiments.scripts.runner --exp <name>`.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Callable, Any

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))


def run_smoke_tests() -> int:
    """Run smoke/sanity tests."""
    from experiments.scripts.run_smoke import main
    return main()


def run_style_exploration() -> int:
    """Run style influence exploration."""
    from experiments.scripts.run_style_grid import main
    return main()


def run_convergence_dynamics() -> int:
    """Run convergence dynamics experiments."""
    from experiments.scripts.run_convergence import main
    return main()


def run_monitor_stress() -> int:
    """Run monitor stress tests."""
    from experiments.scripts.run_monitor_stress import main
    return main()


def run_fault_injection() -> int:
    """Run fault injection & resilience tests."""
    print("Fault injection experiments not yet implemented")
    return 0


def run_async_benchmark() -> int:
    """Run async vs sync performance benchmarks."""
    print("Async benchmarking experiments not yet implemented")
    return 0


def run_analytics_generator() -> int:
    """Run analytics index generator."""
    print("Analytics generator not yet implemented")
    return 0


def run_extensibility_demo() -> int:
    """Run extensibility experiments with custom monitors."""
    print("Extensibility experiments not yet implemented")
    return 0


def run_policy_tiers_demo() -> int:
    """Run policy/safety modeling extensions demo."""
    print("Policy tiers experiments not yet implemented")
    return 0


def run_quality_gates() -> int:
    """Run validation/quality gates."""
    from experiments.scripts.run_quality_gates import main
    return main()


# Registry of available experiments
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "name": "Quick Smoke / Sanity Tests",
        "description": "Single-agent reasoning with/without monitors, forced rule failures, resource overruns",
        "function": run_smoke_tests,
        "implemented": True
    },
    "style": {
        "name": "Style Influence Exploration", 
        "description": "Grid exploration over style weight combinations with JSONL output",
        "function": run_style_exploration,
        "implemented": True
    },
    "convergence": {
        "name": "Convergence Dynamics (Multi-Agent)",
        "description": "Multi-agent convergence trials with various parameters",
        "function": run_convergence_dynamics,
        "implemented": True
    },
    "monitor_stress": {
        "name": "Monitoring Stress & Failure Mode Mapping",
        "description": "High-sensitivity monitor config with stacked violations",
        "function": run_monitor_stress,
        "implemented": True
    },
    "fault_injection": {
        "name": "Fault Injection & Resilience",
        "description": "Engine wrapper with probabilistic exceptions and delays",
        "function": run_fault_injection,
        "implemented": False
    },
    "async_benchmark": {
        "name": "Async vs Sync Performance Benchmark",
        "description": "Sequential vs parallel async dialogue performance comparison",
        "function": run_async_benchmark,
        "implemented": False
    },
    "analytics": {
        "name": "Analytics Index Generator",
        "description": "Scan persisted sessions and compute diversity/repetition metrics",
        "function": run_analytics_generator,
        "implemented": False
    },
    "extensibility": {
        "name": "Extensibility Experiments",
        "description": "Custom repetition/semantic similarity monitor demonstrations",
        "function": run_extensibility_demo,
        "implemented": False
    },
    "policy_tiers": {
        "name": "Policy/Safety Modeling Extensions",
        "description": "Multi-tier monitoring (pre_style + post_style) demonstration",
        "function": run_policy_tiers_demo,
        "implemented": False
    },
    "quality_gates": {
        "name": "Validation / Quality Gates",
        "description": "Load JSONL metrics and apply threshold gates for CI",
        "function": run_quality_gates,
        "implemented": True
    }
}


def list_experiments() -> None:
    """List all available experiments."""
    print("Available AgentNet Experiments:")
    print("=" * 50)
    
    for exp_id, exp_info in EXPERIMENTS.items():
        status = "✓" if exp_info["implemented"] else "○"
        print(f"{status} {exp_id:15} - {exp_info['name']}")
        print(f"  {'':17} {exp_info['description']}")
        print()
    
    print("Legend: ✓ = Implemented, ○ = Planned")
    print("\nUsage:")
    print("  python -m experiments.scripts.runner --exp <experiment_name>")
    print("  python -m experiments.scripts.runner --list")


def main() -> int:
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="AgentNet Experimental Harness Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.scripts.runner --exp smoke
  python -m experiments.scripts.runner --exp style
  python -m experiments.scripts.runner --exp convergence
  python -m experiments.scripts.runner --list
        """
    )
    
    parser.add_argument(
        "--exp", "--experiment",
        type=str,
        help="Experiment to run (use --list to see available experiments)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiments"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return 0
    
    if not args.exp:
        print("Error: No experiment specified. Use --exp <name> or --list to see available experiments.")
        parser.print_help()
        return 1
    
    experiment_id = args.exp.lower()
    
    if experiment_id not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment_id}'")
        print(f"Available experiments: {', '.join(EXPERIMENTS.keys())}")
        print("Use --list for detailed information.")
        return 1
    
    experiment = EXPERIMENTS[experiment_id]
    
    if not experiment["implemented"]:
        print(f"Error: Experiment '{experiment_id}' is planned but not yet implemented.")
        print(f"Description: {experiment['description']}")
        return 1
    
    print(f"Running experiment: {experiment['name']}")
    print(f"Description: {experiment['description']}")
    print("-" * 60)
    
    try:
        return experiment["function"]()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 130
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())