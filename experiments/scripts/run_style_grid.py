"""
Style Influence Exploration for AgentNet.
Grid exploration over style weight combinations with JSONL output of results.
"""

import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path to import AgentNet
sys.path.append(str(Path(__file__).parent.parent.parent))

from AgentNet import AgentNet, ExampleEngine
from experiments.utils.analytics import extract_session_metrics, write_metrics_jsonl


def generate_style_grid(
    dimensions: List[str],
    min_val: float = 0.1,
    max_val: float = 1.0,
    steps: int = 3
) -> List[Dict[str, float]]:
    """Generate a grid of style combinations."""
    values = [min_val + i * (max_val - min_val) / (steps - 1) for i in range(steps)]
    
    style_combinations = []
    for combo in itertools.product(values, repeat=len(dimensions)):
        style = dict(zip(dimensions, combo))
        style_combinations.append(style)
    
    return style_combinations


def run_style_exploration(
    output_dir: Path,
    tasks: List[str] = None,
    style_dimensions: List[str] = None,
    grid_steps: int = 3,
    trials_per_combination: int = 2
) -> Dict[str, Any]:
    """Run comprehensive style exploration."""
    
    if tasks is None:
        tasks = [
            "Analyze network security vulnerabilities",
            "Design creative problem-solving approach", 
            "Perform logical reasoning chain",
            "Generate innovative solutions",
            "Conduct systematic analysis"
        ]
    
    if style_dimensions is None:
        style_dimensions = ["logic", "creativity", "analytical"]
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_type": "style_exploration",
        "parameters": {
            "style_dimensions": style_dimensions,
            "grid_steps": grid_steps,
            "trials_per_combination": trials_per_combination,
            "total_tasks": len(tasks)
        },
        "combinations_tested": 0,
        "total_trials": 0,
        "results": []
    }
    
    engine = ExampleEngine()
    style_combinations = generate_style_grid(style_dimensions, steps=grid_steps)
    
    for i, style_combo in enumerate(style_combinations):
        print(f"Testing style combination {i+1}/{len(style_combinations)}: {style_combo}")
        
        combination_results = []
        
        for trial in range(trials_per_combination):
            for task_idx, task in enumerate(tasks):
                agent_name = f"StyleAgent_{i}_{trial}_{task_idx}"
                agent = AgentNet(
                    name=agent_name,
                    style=style_combo.copy(),
                    engine=engine,
                    monitors=[]
                )
                
                try:
                    result = agent.generate_reasoning_tree(task)
                    
                    # Extract result data
                    result_data = result.get("result", {})
                    
                    # Compute metrics
                    session_data = {
                        "transcript": [result_data],
                        "runtime_seconds": result_data.get("runtime", 0)
                    }
                    metrics = extract_session_metrics(session_data)
                    
                    # Prepare trial data
                    trial_data = {
                        "style_combination": style_combo,
                        "task": task,
                        "task_index": task_idx,
                        "trial": trial,
                        "agent_name": agent_name,
                        "confidence": result_data.get("confidence", 0),
                        "content_length": len(result_data.get("content", "")),
                        "style_insights": result_data.get("style_insights", []),
                        "style_insights_count": len(result_data.get("style_insights", [])),
                        "original_confidence": result_data.get("original_confidence", result_data.get("confidence", 0)),
                        "confidence_boost": result_data.get("confidence", 0) - result_data.get("original_confidence", result_data.get("confidence", 0)),
                        "runtime": result_data.get("runtime", 0),
                        "success": True
                    }
                    
                    combination_results.append(trial_data)
                    
                    # Write individual metrics entry
                    metrics_entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "experiment_type": "style_exploration",
                        "session_id": f"style_{agent_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        "agent_count": 1,
                        "metrics": metrics,
                        "parameters": {
                            "style_combination": style_combo,
                            "task": task,
                            "trial": trial,
                            "combination_index": i
                        },
                        "outcomes": {
                            "success": True,
                            "confidence_boost": trial_data["confidence_boost"],
                            "style_insights_count": trial_data["style_insights_count"]
                        }
                    }
                    
                    write_metrics_jsonl(metrics_entry, output_dir / "style_exploration_metrics.jsonl")
                    results["total_trials"] += 1
                    
                except Exception as e:
                    error_data = {
                        "style_combination": style_combo,
                        "task": task,
                        "task_index": task_idx,
                        "trial": trial,
                        "agent_name": agent_name,
                        "success": False,
                        "error": str(e)
                    }
                    combination_results.append(error_data)
                    results["total_trials"] += 1
        
        # Aggregate results for this style combination
        successful_trials = [r for r in combination_results if r.get("success", False)]
        if successful_trials:
            avg_confidence = sum(r["confidence"] for r in successful_trials) / len(successful_trials)
            avg_boost = sum(r["confidence_boost"] for r in successful_trials) / len(successful_trials)
            avg_insights = sum(r["style_insights_count"] for r in successful_trials) / len(successful_trials)
            avg_runtime = sum(r["runtime"] for r in successful_trials) / len(successful_trials)
            
            combo_summary = {
                "style_combination": style_combo,
                "successful_trials": len(successful_trials),
                "total_trials": len(combination_results),
                "success_rate": len(successful_trials) / len(combination_results),
                "avg_confidence": avg_confidence,
                "avg_confidence_boost": avg_boost,
                "avg_style_insights": avg_insights,
                "avg_runtime": avg_runtime,
                "trials": combination_results
            }
        else:
            combo_summary = {
                "style_combination": style_combo,
                "successful_trials": 0,
                "total_trials": len(combination_results),
                "success_rate": 0.0,
                "trials": combination_results
            }
        
        results["results"].append(combo_summary)
        results["combinations_tested"] += 1
    
    return results


def analyze_style_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze style exploration results to find patterns."""
    analysis = {
        "best_combinations": [],
        "dimension_effects": {},
        "insights": []
    }
    
    successful_combos = [r for r in results["results"] if r["success_rate"] > 0]
    if not successful_combos:
        return analysis
    
    # Find best combinations by different metrics
    best_confidence = max(successful_combos, key=lambda x: x.get("avg_confidence", 0))
    best_boost = max(successful_combos, key=lambda x: x.get("avg_confidence_boost", 0))
    best_insights = max(successful_combos, key=lambda x: x.get("avg_style_insights", 0))
    
    analysis["best_combinations"] = [
        {"metric": "confidence", "combination": best_confidence["style_combination"], "value": best_confidence.get("avg_confidence", 0)},
        {"metric": "confidence_boost", "combination": best_boost["style_combination"], "value": best_boost.get("avg_confidence_boost", 0)},
        {"metric": "style_insights", "combination": best_insights["style_combination"], "value": best_insights.get("avg_style_insights", 0)}
    ]
    
    # Analyze dimension effects
    style_dimensions = results["parameters"]["style_dimensions"]
    for dim in style_dimensions:
        dim_values = {}
        for combo_result in successful_combos:
            dim_val = combo_result["style_combination"][dim]
            if dim_val not in dim_values:
                dim_values[dim_val] = {"confidences": [], "boosts": [], "insights": []}
            
            dim_values[dim_val]["confidences"].append(combo_result.get("avg_confidence", 0))
            dim_values[dim_val]["boosts"].append(combo_result.get("avg_confidence_boost", 0))
            dim_values[dim_val]["insights"].append(combo_result.get("avg_style_insights", 0))
        
        # Compute averages for each dimension value
        dim_analysis = {}
        for val, metrics in dim_values.items():
            dim_analysis[val] = {
                "avg_confidence": sum(metrics["confidences"]) / len(metrics["confidences"]),
                "avg_boost": sum(metrics["boosts"]) / len(metrics["boosts"]),
                "avg_insights": sum(metrics["insights"]) / len(metrics["insights"]),
                "count": len(metrics["confidences"])
            }
        
        analysis["dimension_effects"][dim] = dim_analysis
    
    return analysis


def main():
    """Main entry point for style exploration."""
    # Create output directory
    output_dir = Path("experiments/data/raw_sessions") / f"style_exploration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running AgentNet Style Exploration...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Run style exploration
    results = run_style_exploration(output_dir, grid_steps=3, trials_per_combination=2)
    
    # Analyze results
    analysis = analyze_style_results(results)
    
    # Save detailed results
    results_file = output_dir / "style_exploration_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    analysis_file = output_dir / "style_exploration_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"\nStyle Exploration Results:")
    print(f"Total combinations tested: {results['combinations_tested']}")
    print(f"Total trials: {results['total_trials']}")
    print(f"Successful combinations: {len([r for r in results['results'] if r['success_rate'] > 0])}")
    
    # Print best combinations
    if analysis["best_combinations"]:
        print("\nBest Combinations:")
        for best in analysis["best_combinations"]:
            print(f"  {best['metric']}: {best['combination']} (value: {best['value']:.3f})")
    
    # Print dimension effects
    if analysis["dimension_effects"]:
        print("\nDimension Effects:")
        for dim, effects in analysis["dimension_effects"].items():
            print(f"  {dim}:")
            for val, metrics in effects.items():
                print(f"    {val}: avg_conf={metrics['avg_confidence']:.3f}, boost={metrics['avg_boost']:.3f}, insights={metrics['avg_insights']:.1f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Analysis saved to: {analysis_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())