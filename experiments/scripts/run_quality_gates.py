"""
Validation / Quality Gates for AgentNet experiments.
Script that loads recent JSONL metrics, applies threshold gates and exits non-zero on failure (CI-friendly).
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import AgentNet
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.analytics import load_metrics_jsonl


DEFAULT_THRESHOLDS = {
    "convergence_rate_min": 0.3,      # Minimum 30% convergence rate
    "convergence_rate_max": 1.0,      # Maximum 100% convergence rate
    "lexical_diversity_min": 0.1,     # Minimum lexical diversity
    "severe_violations_max": 0,       # No severe violations allowed
    "confidence_score_min": 0.2,      # Minimum average confidence
    "confidence_score_max": 1.2,      # Maximum confidence (allowing for style boost)
    "success_rate_min": 0.5,          # Minimum 50% success rate for experiments
    "runtime_max": 300.0,             # Maximum 5 minutes per session
    "token_count_max": 10000,         # Maximum token count per session
    "violation_rate_max": 0.5,        # Maximum 50% of sessions can have violations
    "style_insights_min": 0,          # Minimum style insights (0 allows for no-style scenarios)
    "rounds_executed_max": 50         # Maximum rounds to prevent infinite loops
}


def load_recent_metrics(
    data_dir: Path,
    hours_back: int = 24,
    experiment_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Load metrics from JSONL files within the specified time window."""
    if not data_dir.exists():
        return []
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    all_metrics = []
    
    # Find all JSONL files in data directory
    for jsonl_file in data_dir.rglob("*.jsonl"):
        try:
            metrics = load_metrics_jsonl(jsonl_file)
            for metric in metrics:
                # Parse timestamp and filter by time
                try:
                    metric_time = datetime.fromisoformat(metric["timestamp"].replace("Z", "+00:00"))
                    if metric_time.replace(tzinfo=None) >= cutoff_time:
                        # Filter by experiment type if specified
                        if experiment_types is None or metric.get("experiment_type") in experiment_types:
                            all_metrics.append(metric)
                except (KeyError, ValueError):
                    continue
        except Exception:
            continue
    
    return all_metrics


def apply_quality_gates(
    metrics: List[Dict[str, Any]],
    thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """Apply quality gates to metrics and return results."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()
    
    results = {
        "total_sessions": len(metrics),
        "gates_passed": 0,
        "gates_failed": 0,
        "failures": [],
        "summary": {},
        "overall_pass": True
    }
    
    if not metrics:
        results["failures"].append({
            "gate": "data_availability",
            "description": "No metrics found in specified time window",
            "threshold": "N/A",
            "actual": 0,
            "severity": "critical"
        })
        results["overall_pass"] = False
        return results
    
    # Extract metrics for analysis
    convergence_rates = []
    lexical_diversities = []
    severe_violations = []
    confidence_scores = []
    success_indicators = []
    runtimes = []
    token_counts = []
    violation_indicators = []
    style_insights = []
    rounds_executed = []
    
    for metric in metrics:
        m = metric.get("metrics", {})
        outcomes = metric.get("outcomes", {})
        
        # Extract values, handling missing data gracefully
        if "convergence_rate" in m:
            convergence_rates.append(m["convergence_rate"])
        
        if "lexical_diversity" in m:
            lexical_diversities.append(m["lexical_diversity"])
        
        if "severe_violations" in m:
            severe_violations.append(m["severe_violations"])
        
        if "confidence_score" in m:
            confidence_scores.append(m["confidence_score"])
        
        if "success" in outcomes:
            success_indicators.append(1 if outcomes["success"] else 0)
        
        if "runtime_seconds" in m:
            runtimes.append(m["runtime_seconds"])
        
        if "token_count" in m:
            token_counts.append(m["token_count"])
        
        if "violation_count" in m:
            violation_indicators.append(1 if m["violation_count"] > 0 else 0)
        
        if "style_insights_count" in m:
            style_insights.append(m["style_insights_count"])
        
        if "rounds_executed" in m:
            rounds_executed.append(m["rounds_executed"])
    
    # Apply individual gates
    gates = [
        ("convergence_rate_min", convergence_rates, lambda vals, thresh: min(vals) if vals else 0, ">="),
        ("convergence_rate_max", convergence_rates, lambda vals, thresh: max(vals) if vals else 0, "<="),
        ("lexical_diversity_min", lexical_diversities, lambda vals, thresh: min(vals) if vals else 0, ">="),
        ("severe_violations_max", severe_violations, lambda vals, thresh: max(vals) if vals else 0, "<="),
        ("confidence_score_min", confidence_scores, lambda vals, thresh: min(vals) if vals else 0, ">="),
        ("confidence_score_max", confidence_scores, lambda vals, thresh: max(vals) if vals else 0, "<="),
        ("success_rate_min", success_indicators, lambda vals, thresh: sum(vals) / len(vals) if vals else 0, ">="),
        ("runtime_max", runtimes, lambda vals, thresh: max(vals) if vals else 0, "<="),
        ("token_count_max", token_counts, lambda vals, thresh: max(vals) if vals else 0, "<="),
        ("violation_rate_max", violation_indicators, lambda vals, thresh: sum(vals) / len(vals) if vals else 0, "<="),
        ("style_insights_min", style_insights, lambda vals, thresh: min(vals) if vals else 0, ">="),
        ("rounds_executed_max", rounds_executed, lambda vals, thresh: max(vals) if vals else 0, "<="),
    ]
    
    for gate_name, values, aggregator, operator in gates:
        if gate_name not in thresholds:
            continue
        
        threshold = thresholds[gate_name]
        
        if not values:
            # No data for this gate - decide if this is acceptable
            if gate_name in ["convergence_rate_min", "convergence_rate_max"]:
                # Convergence rate only applies to multi-agent experiments
                continue
            
            actual_value = 0
        else:
            actual_value = aggregator(values, threshold)
        
        # Apply threshold check
        if operator == ">=" and actual_value >= threshold:
            results["gates_passed"] += 1
        elif operator == "<=" and actual_value <= threshold:
            results["gates_passed"] += 1
        else:
            results["gates_failed"] += 1
            results["overall_pass"] = False
            
            failure = {
                "gate": gate_name,
                "description": f"{gate_name.replace('_', ' ').title()}",
                "threshold": f"{operator} {threshold}",
                "actual": actual_value,
                "severity": "major" if "severe" in gate_name or "max" in gate_name else "minor",
                "sample_count": len(values)
            }
            results["failures"].append(failure)
        
        # Store in summary
        results["summary"][gate_name] = {
            "threshold": threshold,
            "actual": actual_value,
            "passed": (operator == ">=" and actual_value >= threshold) or (operator == "<=" and actual_value <= threshold),
            "sample_count": len(values)
        }
    
    return results


def main():
    """Main entry point for quality gates validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentNet Quality Gates Validation")
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=Path("experiments/data"),
        help="Directory containing JSONL metrics files"
    )
    parser.add_argument(
        "--hours-back", 
        type=int, 
        default=24,
        help="Hours back to look for metrics (default: 24)"
    )
    parser.add_argument(
        "--experiment-types",
        nargs="*",
        help="Filter by specific experiment types"
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        help="JSON file with custom threshold values"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for detailed results (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load custom thresholds if provided
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.thresholds_file and args.thresholds_file.exists():
        try:
            with open(args.thresholds_file, 'r') as f:
                custom_thresholds = json.load(f)
                thresholds.update(custom_thresholds)
            print(f"Loaded custom thresholds from {args.thresholds_file}")
        except Exception as e:
            print(f"Warning: Failed to load custom thresholds: {e}")
    
    print("AgentNet Quality Gates Validation")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Time window: {args.hours_back} hours back")
    if args.experiment_types:
        print(f"Experiment types: {', '.join(args.experiment_types)}")
    print()
    
    # Load recent metrics
    metrics = load_recent_metrics(args.data_dir, args.hours_back, args.experiment_types)
    print(f"Loaded {len(metrics)} metrics entries")
    
    if args.verbose and metrics:
        experiment_counts = {}
        for metric in metrics:
            exp_type = metric.get("experiment_type", "unknown")
            experiment_counts[exp_type] = experiment_counts.get(exp_type, 0) + 1
        print("Experiment type breakdown:")
        for exp_type, count in experiment_counts.items():
            print(f"  {exp_type}: {count}")
        print()
    
    # Apply quality gates
    results = apply_quality_gates(metrics, thresholds)
    
    # Print results
    print("Quality Gates Results:")
    print(f"  Total sessions analyzed: {results['total_sessions']}")
    print(f"  Gates passed: {results['gates_passed']}")
    print(f"  Gates failed: {results['gates_failed']}")
    print(f"  Overall result: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print()
    
    if results["failures"]:
        print("Failed Gates:")
        for failure in results["failures"]:
            severity_marker = "🔴" if failure["severity"] == "critical" else "🟡" if failure["severity"] == "major" else "🟠"
            print(f"  {severity_marker} {failure['gate']}: {failure['description']}")
            print(f"    Expected: {failure['threshold']}, Actual: {failure['actual']:.3f}")
            if "sample_count" in failure:
                print(f"    Sample size: {failure['sample_count']}")
            print()
    
    if args.verbose and results["summary"]:
        print("Detailed Gate Summary:")
        for gate_name, gate_info in results["summary"].items():
            status = "✓" if gate_info["passed"] else "✗"
            print(f"  {status} {gate_name}: {gate_info['actual']:.3f} (threshold: {gate_info['threshold']})")
        print()
    
    # Save detailed results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.output}")
    
    # Exit with appropriate code
    if results["overall_pass"]:
        print("✅ All quality gates passed!")
        return 0
    else:
        print("❌ Some quality gates failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())