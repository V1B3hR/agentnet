"""
Convergence Dynamics (Multi-Agent) for AgentNet.
Convergence trial runner with parameters: overlap thresholds, window sizes, agent diversity profiles, trials count.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path to import AgentNet
sys.path.append(str(Path(__file__).parent.parent.parent))

from AgentNet import AgentNet, ExampleEngine
from experiments.utils.analytics import extract_session_metrics, write_metrics_jsonl, compute_diversity_index


def create_diverse_agents(
    agent_count: int, 
    diversity_profile: str,
    engine: Any
) -> List[AgentNet]:
    """Create agents with different diversity profiles."""
    agents = []
    
    if diversity_profile == "high_diversity":
        # Maximum style differences
        style_configs = [
            {"logic": 0.9, "creativity": 0.1, "analytical": 0.9},  # Logical analyst
            {"logic": 0.2, "creativity": 0.9, "analytical": 0.3},  # Creative thinker
            {"logic": 0.5, "creativity": 0.5, "analytical": 0.8},  # Balanced analyst
            {"logic": 0.8, "creativity": 0.6, "analytical": 0.4},  # Logical creative
            {"logic": 0.3, "creativity": 0.7, "analytical": 0.6},  # Creative analyst
        ]
    elif diversity_profile == "medium_diversity":
        # Moderate style differences
        style_configs = [
            {"logic": 0.8, "creativity": 0.4, "analytical": 0.7},
            {"logic": 0.6, "creativity": 0.6, "analytical": 0.5},
            {"logic": 0.5, "creativity": 0.7, "analytical": 0.6},
            {"logic": 0.7, "creativity": 0.5, "analytical": 0.8},
            {"logic": 0.4, "creativity": 0.8, "analytical": 0.4},
        ]
    elif diversity_profile == "low_diversity":
        # Similar styles
        base_style = {"logic": 0.7, "creativity": 0.5, "analytical": 0.6}
        style_configs = []
        for i in range(5):
            style = base_style.copy()
            # Small variations
            for key in style:
                style[key] += (i - 2) * 0.05  # -0.1 to +0.1 variation
                style[key] = max(0.1, min(1.0, style[key]))  # Clamp to valid range
            style_configs.append(style)
    else:
        raise ValueError(f"Unknown diversity profile: {diversity_profile}")
    
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    
    for i in range(min(agent_count, len(style_configs))):
        agent = AgentNet(
            name=names[i],
            style=style_configs[i],
            engine=engine,
            monitors=[]
        )
        agents.append(agent)
    
    return agents


async def run_convergence_trial(
    agents: List[AgentNet],
    topic: str,
    max_rounds: int = 10,
    overlap_threshold: float = 0.5,
    window_size: int = 3,
    trial_id: str = ""
) -> Dict[str, Any]:
    """Run a single convergence trial."""
    try:
        session = await agents[0].async_multi_party_dialogue(
            agents=agents,
            topic=topic,
            rounds=max_rounds,
            convergence=True,
            parallel_round=False
        )
        
        # Extract convergence metrics
        converged = session.get("converged", False)
        rounds_executed = session.get("rounds_executed", 0)
        transcript = session.get("transcript", [])
        
        # Compute additional metrics
        agent_contributions = {}
        total_tokens = 0
        confidences = []
        
        for turn in transcript:
            if isinstance(turn, dict):
                agent_name = turn.get("agent", "unknown")
                content = turn.get("content", "")
                
                if agent_name not in agent_contributions:
                    agent_contributions[agent_name] = {"turns": 0, "tokens": 0, "confidences": []}
                
                agent_contributions[agent_name]["turns"] += 1
                token_count = len(content.split())
                agent_contributions[agent_name]["tokens"] += token_count
                total_tokens += token_count
                
                if "confidence" in turn:
                    confidence = turn["confidence"]
                    agent_contributions[agent_name]["confidences"].append(confidence)
                    confidences.append(confidence)
        
        # Compute participation balance (how evenly agents participated)
        turn_counts = [contrib["turns"] for contrib in agent_contributions.values()]
        participation_balance = min(turn_counts) / max(turn_counts) if max(turn_counts) > 0 else 0
        
        # Compute confidence progression (are confidences increasing over time?)
        confidence_progression = 0
        if len(confidences) > 1:
            early_conf = sum(confidences[:len(confidences)//2]) / (len(confidences)//2)
            late_conf = sum(confidences[len(confidences)//2:]) / (len(confidences) - len(confidences)//2)
            confidence_progression = late_conf - early_conf
        
        # Compute diversity index for the session
        diversity_index = compute_diversity_index([session])
        
        return {
            "trial_id": trial_id,
            "success": True,
            "converged": converged,
            "rounds_executed": rounds_executed,
            "total_turns": len(transcript),
            "total_tokens": total_tokens,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_progression": confidence_progression,
            "participation_balance": participation_balance,
            "diversity_index": diversity_index,
            "agent_count": len(agents),
            "agent_contributions": agent_contributions,
            "session_data": session
        }
        
    except Exception as e:
        return {
            "trial_id": trial_id,
            "success": False,
            "error": str(e),
            "converged": False,
            "rounds_executed": 0
        }


async def run_convergence_experiments(
    output_dir: Path,
    topics: List[str] = None,
    diversity_profiles: List[str] = None,
    overlap_thresholds: List[float] = None,
    window_sizes: List[int] = None,
    agent_counts: List[int] = None,
    trials_per_config: int = 3,
    max_rounds: int = 10
) -> Dict[str, Any]:
    """Run comprehensive convergence experiments."""
    
    if topics is None:
        topics = [
            "Optimal distributed system architecture",
            "Ethical AI decision-making framework",
            "Climate change mitigation strategies",
            "Future of human-AI collaboration",
            "Space exploration priorities"
        ]
    
    if diversity_profiles is None:
        diversity_profiles = ["high_diversity", "medium_diversity", "low_diversity"]
    
    if overlap_thresholds is None:
        overlap_thresholds = [0.3, 0.5, 0.7]
    
    if window_sizes is None:
        window_sizes = [2, 3, 4]
    
    if agent_counts is None:
        agent_counts = [2, 3, 4]
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_type": "convergence_dynamics",
        "parameters": {
            "topics": topics,
            "diversity_profiles": diversity_profiles,
            "overlap_thresholds": overlap_thresholds,
            "window_sizes": window_sizes,
            "agent_counts": agent_counts,
            "trials_per_config": trials_per_config,
            "max_rounds": max_rounds
        },
        "total_configurations": 0,
        "total_trials": 0,
        "results": []
    }
    
    engine = ExampleEngine()
    config_count = 0
    
    for topic in topics:
        for diversity_profile in diversity_profiles:
            for agent_count in agent_counts:
                for overlap_threshold in overlap_thresholds:
                    for window_size in window_sizes:
                        config_count += 1
                        
                        print(f"Configuration {config_count}: {topic[:30]}..., {diversity_profile}, "
                              f"{agent_count} agents, overlap={overlap_threshold}, window={window_size}")
                        
                        # Create agents for this configuration
                        agents = create_diverse_agents(agent_count, diversity_profile, engine)
                        
                        # Update convergence parameters
                        for agent in agents:
                            agent.dialogue_config.update({
                                "convergence_min_overlap": overlap_threshold,
                                "convergence_window": window_size
                            })
                        
                        config_results = []
                        
                        # Run multiple trials for this configuration
                        for trial in range(trials_per_config):
                            trial_id = f"config_{config_count}_trial_{trial}"
                            
                            trial_result = await run_convergence_trial(
                                agents=agents,
                                topic=topic,
                                max_rounds=max_rounds,
                                overlap_threshold=overlap_threshold,
                                window_size=window_size,
                                trial_id=trial_id
                            )
                            
                            config_results.append(trial_result)
                            results["total_trials"] += 1
                            
                            # Write individual trial metrics
                            if trial_result["success"]:
                                session_data = trial_result["session_data"]
                                metrics = extract_session_metrics(session_data)
                                
                                metrics_entry = {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "experiment_type": "convergence_dynamics",
                                    "session_id": f"convergence_{trial_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                                    "agent_count": agent_count,
                                    "metrics": metrics,
                                    "parameters": {
                                        "topic": topic,
                                        "diversity_profile": diversity_profile,
                                        "overlap_threshold": overlap_threshold,
                                        "window_size": window_size,
                                        "max_rounds": max_rounds,
                                        "trial_id": trial_id
                                    },
                                    "outcomes": {
                                        "success": True,
                                        "converged": trial_result["converged"],
                                        "participation_balance": trial_result["participation_balance"],
                                        "diversity_index": trial_result["diversity_index"]
                                    }
                                }
                                
                                write_metrics_jsonl(metrics_entry, output_dir / "convergence_metrics.jsonl")
                        
                        # Aggregate results for this configuration
                        successful_trials = [r for r in config_results if r["success"]]
                        
                        if successful_trials:
                            convergence_rate = sum(1 for r in successful_trials if r["converged"]) / len(successful_trials)
                            avg_rounds = sum(r["rounds_executed"] for r in successful_trials) / len(successful_trials)
                            avg_confidence = sum(r["avg_confidence"] for r in successful_trials) / len(successful_trials)
                            avg_participation = sum(r["participation_balance"] for r in successful_trials) / len(successful_trials)
                            avg_diversity = sum(r["diversity_index"] for r in successful_trials) / len(successful_trials)
                            
                            config_summary = {
                                "configuration": {
                                    "topic": topic,
                                    "diversity_profile": diversity_profile,
                                    "agent_count": agent_count,
                                    "overlap_threshold": overlap_threshold,
                                    "window_size": window_size
                                },
                                "successful_trials": len(successful_trials),
                                "total_trials": len(config_results),
                                "success_rate": len(successful_trials) / len(config_results),
                                "convergence_rate": convergence_rate,
                                "avg_rounds_to_convergence": avg_rounds,
                                "avg_confidence": avg_confidence,
                                "avg_participation_balance": avg_participation,
                                "avg_diversity_index": avg_diversity,
                                "trials": config_results
                            }
                        else:
                            config_summary = {
                                "configuration": {
                                    "topic": topic,
                                    "diversity_profile": diversity_profile,
                                    "agent_count": agent_count,
                                    "overlap_threshold": overlap_threshold,
                                    "window_size": window_size
                                },
                                "successful_trials": 0,
                                "total_trials": len(config_results),
                                "success_rate": 0.0,
                                "convergence_rate": 0.0,
                                "trials": config_results
                            }
                        
                        results["results"].append(config_summary)
                        results["total_configurations"] += 1
    
    return results


def analyze_convergence_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze convergence results to identify patterns."""
    analysis = {
        "best_configurations": [],
        "parameter_effects": {},
        "insights": []
    }
    
    successful_configs = [r for r in results["results"] if r["success_rate"] > 0]
    if not successful_configs:
        return analysis
    
    # Find best configurations by different metrics
    best_convergence = max(successful_configs, key=lambda x: x.get("convergence_rate", 0))
    best_efficiency = min([r for r in successful_configs if r.get("convergence_rate", 0) > 0], 
                         key=lambda x: x.get("avg_rounds_to_convergence", float('inf')))
    best_participation = max(successful_configs, key=lambda x: x.get("avg_participation_balance", 0))
    
    analysis["best_configurations"] = [
        {"metric": "convergence_rate", "config": best_convergence["configuration"], "value": best_convergence.get("convergence_rate", 0)},
        {"metric": "efficiency", "config": best_efficiency["configuration"], "value": best_efficiency.get("avg_rounds_to_convergence", 0)},
        {"metric": "participation", "config": best_participation["configuration"], "value": best_participation.get("avg_participation_balance", 0)}
    ]
    
    # Analyze parameter effects
    parameters = ["diversity_profile", "agent_count", "overlap_threshold", "window_size"]
    for param in parameters:
        param_values = {}
        for config_result in successful_configs:
            param_val = config_result["configuration"][param]
            if param_val not in param_values:
                param_values[param_val] = {"convergence_rates": [], "avg_rounds": [], "participation": []}
            
            param_values[param_val]["convergence_rates"].append(config_result.get("convergence_rate", 0))
            param_values[param_val]["avg_rounds"].append(config_result.get("avg_rounds_to_convergence", 0))
            param_values[param_val]["participation"].append(config_result.get("avg_participation_balance", 0))
        
        # Compute averages for each parameter value
        param_analysis = {}
        for val, metrics in param_values.items():
            param_analysis[str(val)] = {
                "avg_convergence_rate": sum(metrics["convergence_rates"]) / len(metrics["convergence_rates"]),
                "avg_rounds": sum(metrics["avg_rounds"]) / len(metrics["avg_rounds"]),
                "avg_participation": sum(metrics["participation"]) / len(metrics["participation"]),
                "count": len(metrics["convergence_rates"])
            }
        
        analysis["parameter_effects"][param] = param_analysis
    
    return analysis


def main():
    """Main entry point for convergence dynamics experiments."""
    # Create output directory
    output_dir = Path("experiments/data/raw_sessions") / f"convergence_dynamics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running AgentNet Convergence Dynamics Experiments...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Run convergence experiments
    results = asyncio.run(run_convergence_experiments(
        output_dir,
        topics=["Network security analysis", "AI ethics framework"],  # Reduced for demo
        trials_per_config=2,
        max_rounds=8
    ))
    
    # Analyze results
    analysis = analyze_convergence_results(results)
    
    # Save detailed results
    results_file = output_dir / "convergence_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    analysis_file = output_dir / "convergence_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"\nConvergence Dynamics Results:")
    print(f"Total configurations tested: {results['total_configurations']}")
    print(f"Total trials: {results['total_trials']}")
    print(f"Successful configurations: {len([r for r in results['results'] if r['success_rate'] > 0])}")
    
    # Print best configurations
    if analysis["best_configurations"]:
        print("\nBest Configurations:")
        for best in analysis["best_configurations"]:
            print(f"  {best['metric']}: {best['config']} (value: {best['value']:.3f})")
    
    # Print parameter effects
    if analysis["parameter_effects"]:
        print("\nParameter Effects:")
        for param, effects in analysis["parameter_effects"].items():
            print(f"  {param}:")
            for val, metrics in effects.items():
                print(f"    {val}: conv_rate={metrics['avg_convergence_rate']:.3f}, rounds={metrics['avg_rounds']:.1f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Analysis saved to: {analysis_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())