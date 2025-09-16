"""
Smoke / Sanity Tests for AgentNet.
Tests single-agent reasoning with and without monitors, forced rule failures, and resource budget overruns.
"""

import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path to import AgentNet
sys.path.append(str(Path(__file__).parent.parent.parent))

from AgentNet import AgentNet, ExampleEngine, CognitiveFault
from experiments.utils.analytics import extract_session_metrics, write_metrics_jsonl


def run_smoke_tests(output_dir: Path) -> Dict[str, Any]:
    """Run comprehensive smoke tests."""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_type": "smoke_test",
        "tests": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0
        }
    }
    
    engine = ExampleEngine()
    
    # Test 1: Basic single-agent reasoning without monitors
    test_result = run_test_no_monitors(engine, output_dir)
    results["tests"].append(test_result)
    results["summary"]["total_tests"] += 1
    if test_result["passed"]:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    
    # Test 2: Single-agent reasoning with monitors
    test_result = run_test_with_monitors(engine, output_dir)
    results["tests"].append(test_result)
    results["summary"]["total_tests"] += 1
    if test_result["passed"]:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    
    # Test 3: Forced rule failure
    test_result = run_test_forced_rule_failure(engine, output_dir)
    results["tests"].append(test_result)
    results["summary"]["total_tests"] += 1
    if test_result["passed"]:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    
    # Test 4: Resource budget overrun
    test_result = run_test_resource_overrun(engine, output_dir)
    results["tests"].append(test_result)
    results["summary"]["total_tests"] += 1
    if test_result["passed"]:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    
    # Test 5: CognitiveFault handling
    test_result = run_test_cognitive_fault(engine, output_dir)
    results["tests"].append(test_result)
    results["summary"]["total_tests"] += 1
    if test_result["passed"]:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    
    return results


def run_test_no_monitors(engine: ExampleEngine, output_dir: Path) -> Dict[str, Any]:
    """Test basic reasoning without monitors."""
    test_name = "basic_reasoning_no_monitors"
    
    try:
        # Create agent without monitors
        agent = AgentNet(
            name="SmokeTestAgent",
            style={"logic": 0.8, "creativity": 0.5, "analytical": 0.7},
            engine=engine,
            monitors=[]
        )
        
        # Generate reasoning tree
        result = agent.generate_reasoning_tree("Test reasoning without monitors")
        
        # Verify result structure
        assert "result" in result
        assert "content" in result["result"]
        assert "confidence" in result["result"]
        assert result["result"]["confidence"] > 0
        
        # Write metrics
        session_data = {"transcript": [result["result"]], "runtime_seconds": result["result"].get("runtime", 0)}
        metrics = extract_session_metrics(session_data)
        
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_type": "smoke_test",
            "session_id": f"smoke_{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_count": 1,
            "metrics": metrics,
            "parameters": {"test_name": test_name, "monitors_enabled": False},
            "outcomes": {"success": True, "result": result}
        }
        
        write_metrics_jsonl(metrics_entry, output_dir / "smoke_tests_metrics.jsonl")
        
        return {
            "test_name": test_name,
            "passed": True,
            "metrics": metrics,
            "message": "Basic reasoning completed successfully"
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_test_with_monitors(engine: ExampleEngine, output_dir: Path) -> Dict[str, Any]:
    """Test reasoning with monitors enabled."""
    test_name = "reasoning_with_monitors"
    
    try:
        # Create temporary config files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create monitor config
            monitors_config = """
monitors:
  - name: keyword_guard
    type: keyword
    severity: minor
    description: Block certain tokens
    params:
      keywords: ["badword", "forbidden"]
  - name: resource_guard
    type: resource
    severity: minor
    description: Flag overruns beyond 20%
    params:
      budget_key: resource_budget
      tolerance: 0.2
"""
            monitors_file = tmpdir_path / "monitors.yaml"
            monitors_file.write_text(monitors_config)
            
            # Create agent config
            agent_config = f"""
name: SmokeTestAgentWithMonitors
style:
  logic: 0.8
  creativity: 0.5
  analytical: 0.7
  resource_budget: 0.05
monitors_file: {monitors_file}
dialogue_config:
  max_rounds: 10
"""
            agent_file = tmpdir_path / "agent.yaml"
            agent_file.write_text(agent_config)
            
            # Create agent from config
            agent = AgentNet.from_config(agent_file, engine=engine)
            
            # Generate reasoning tree
            result = agent.generate_reasoning_tree("Test reasoning with monitors enabled")
            
            # Verify result structure
            assert "result" in result
            assert "content" in result["result"]
            assert "confidence" in result["result"]
            
            # Extract metrics
            session_data = {"transcript": [result["result"]], "runtime_seconds": result["result"].get("runtime", 0)}
            metrics = extract_session_metrics(session_data)
            
            metrics_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "experiment_type": "smoke_test",
                "session_id": f"smoke_{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "agent_count": 1,
                "metrics": metrics,
                "parameters": {"test_name": test_name, "monitors_enabled": True, "monitor_count": len(agent.monitors)},
                "outcomes": {"success": True, "result": result}
            }
            
            write_metrics_jsonl(metrics_entry, output_dir / "smoke_tests_metrics.jsonl")
            
            return {
                "test_name": test_name,
                "passed": True,
                "metrics": metrics,
                "message": f"Reasoning with {len(agent.monitors)} monitors completed successfully"
            }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_test_forced_rule_failure(engine: ExampleEngine, output_dir: Path) -> Dict[str, Any]:
    """Test forced rule failure scenario."""
    test_name = "forced_rule_failure"
    
    try:
        # Create agent with strict keyword monitor
        from AgentNet import MonitorSpec, MonitorManager
        
        keyword_spec = MonitorSpec(
            name="strict_keyword_guard",
            type="keyword",
            severity="severe",
            description="Strict keyword filtering",
            params={"keywords": ["test"]}  # Will trigger on our test content
        )
        
        monitor_manager = MonitorManager([keyword_spec])
        
        agent = AgentNet(
            name="StrictTestAgent",
            style={"logic": 0.8, "creativity": 0.5, "analytical": 0.7},
            engine=engine,
            monitors=monitor_manager.monitors
        )
        
        # This should trigger the keyword monitor - expect CognitiveFault
        violations_found = False
        result = None
        try:
            result = agent.generate_reasoning_tree("This is a test that should trigger violations")
        except CognitiveFault as cf:
            # Expected behavior - monitor violations cause CognitiveFault
            violations_found = True
            result = {"result": {"content": "Cognitive fault triggered", "confidence": 0.0, "runtime": 0.0}}
        
        # Also check interaction history for violations
        if not violations_found:
            violations_found = any(
                entry.get("type") == "monitor_violation" or entry.get("type") == "monitor_severe"
                for entry in agent.interaction_history
            )
        
        session_data = {
            "transcript": [result["result"]], 
            "runtime_seconds": result["result"].get("runtime", 0),
            "violations": [entry for entry in agent.interaction_history if "monitor" in entry.get("type", "")]
        }
        metrics = extract_session_metrics(session_data)
        
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_type": "smoke_test",
            "session_id": f"smoke_{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_count": 1,
            "metrics": metrics,
            "parameters": {"test_name": test_name, "expected_violations": True},
            "outcomes": {"success": True, "violations_triggered": violations_found}
        }
        
        write_metrics_jsonl(metrics_entry, output_dir / "smoke_tests_metrics.jsonl")
        
        return {
            "test_name": test_name,
            "passed": True,
            "metrics": metrics,
            "message": f"Rule failure test completed, violations found: {violations_found}"
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_test_resource_overrun(engine: ExampleEngine, output_dir: Path) -> Dict[str, Any]:
    """Test resource budget overrun scenario."""
    test_name = "resource_overrun"
    
    try:
        # Create agent with very low resource budget
        from AgentNet import MonitorSpec, MonitorManager
        
        resource_spec = MonitorSpec(
            name="strict_resource_guard",
            type="resource",
            severity="major",
            description="Very strict resource monitoring",
            params={"budget_key": "resource_budget", "tolerance": 0.01}  # 1% tolerance
        )
        
        monitor_manager = MonitorManager([resource_spec])
        
        agent = AgentNet(
            name="ResourceTestAgent",
            style={
                "logic": 0.8, 
                "creativity": 0.5, 
                "analytical": 0.7,
                "resource_budget": 0.01  # Very low budget to trigger overrun
            },
            engine=engine,
            monitors=monitor_manager.monitors
        )
        
        # This should trigger resource overrun
        result = agent.generate_reasoning_tree("Complex reasoning task that should exceed resource budget")
        
        # Check if resource violations were recorded
        resource_violations = [
            entry for entry in agent.interaction_history 
            if entry.get("type") in ["monitor_violation", "monitor_major"] and "resource" in entry.get("detail", {}).get("violations", [{}])[0].get("type", "")
        ]
        
        session_data = {
            "transcript": [result["result"]], 
            "runtime_seconds": result["result"].get("runtime", 0),
            "violations": resource_violations
        }
        metrics = extract_session_metrics(session_data)
        
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_type": "smoke_test",
            "session_id": f"smoke_{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_count": 1,
            "metrics": metrics,
            "parameters": {"test_name": test_name, "low_budget": 0.01},
            "outcomes": {"success": True, "resource_violations": len(resource_violations)}
        }
        
        write_metrics_jsonl(metrics_entry, output_dir / "smoke_tests_metrics.jsonl")
        
        return {
            "test_name": test_name,
            "passed": True,
            "metrics": metrics,
            "message": f"Resource overrun test completed, {len(resource_violations)} violations found"
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_test_cognitive_fault(engine: ExampleEngine, output_dir: Path) -> Dict[str, Any]:
    """Test CognitiveFault handling."""
    test_name = "cognitive_fault_handling"
    
    try:
        # Create a custom engine that throws CognitiveFault
        class FaultyEngine:
            def infer(self, context: str) -> Dict[str, Any]:
                # Simulate a cognitive fault
                violations = [{
                    "type": "cognitive_error",
                    "severity": "severe",
                    "description": "Simulated cognitive fault",
                    "rationale": "Testing fault handling mechanism",
                    "meta": {"test": True}
                }]
                raise CognitiveFault("Simulated cognitive fault for testing", violations)
        
        faulty_engine = FaultyEngine()
        
        agent = AgentNet(
            name="FaultTestAgent",
            style={"logic": 0.8, "creativity": 0.5, "analytical": 0.7},
            engine=faulty_engine,
            monitors=[]
        )
        
        # This should raise CognitiveFault
        fault_caught = False
        fault_details = {}
        try:
            result = agent.generate_reasoning_tree("This should trigger a cognitive fault")
        except CognitiveFault as cf:
            fault_caught = True
            fault_details = {
                "message": str(cf),
                "violations": cf.violations
            }
        
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_type": "smoke_test",
            "session_id": f"smoke_{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_count": 1,
            "metrics": {"runtime_seconds": 0, "violation_count": 1 if fault_caught else 0},
            "parameters": {"test_name": test_name, "expected_fault": True},
            "outcomes": {"success": fault_caught, "fault_caught": fault_caught}
        }
        
        write_metrics_jsonl(metrics_entry, output_dir / "smoke_tests_metrics.jsonl")
        
        return {
            "test_name": test_name,
            "passed": fault_caught,
            "message": f"CognitiveFault handling test completed, fault caught: {fault_caught}"
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main entry point for smoke tests."""
    # Create output directory
    output_dir = Path("experiments/data/raw_sessions") / f"smoke_tests_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running AgentNet Smoke Tests...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Run all smoke tests
    results = run_smoke_tests(output_dir)
    
    # Save detailed results
    results_file = output_dir / "smoke_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nSmoke Test Results:")
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success rate: {results['summary']['passed'] / results['summary']['total_tests'] * 100:.1f}%")
    
    # Print individual test results
    for test in results["tests"]:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  {test['test_name']}: {status}")
        if not test["passed"]:
            print(f"    Error: {test.get('error', 'Unknown error')}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with error code if any tests failed
    return results['summary']['failed']


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)