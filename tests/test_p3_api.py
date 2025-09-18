#!/usr/bin/env python3
"""
P3 API Tests

Tests for the new DAG and Evaluation API endpoints.
"""

import asyncio
import json
import requests
import threading
import time
from pathlib import Path

from api.server import AgentNetAPI


def start_api_server():
    """Start API server in background thread."""
    api = AgentNetAPI()
    
    def run_server():
        api.run(host="127.0.0.1", port=8081)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # Give server time to start
    return api


def test_dag_planning_endpoint():
    """Test /tasks/plan endpoint."""
    print("🧪 Testing /tasks/plan endpoint...")
    
    # Test task graph from roadmap
    task_graph = {
        "nodes": [
            {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
            {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]},
            {"id": "mitigations", "prompt": "Propose mitigations", "agent": "Apollo", "deps": ["analysis"]},
            {"id": "summary", "prompt": "Integrate plan & mitigations", "agent": "Synthesizer", "deps": ["mitigations"]}
        ]
    }
    
    response = requests.post("http://127.0.0.1:8081/tasks/plan", json=task_graph)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "graph_id" in data, "Response should include graph_id"
    assert data["valid"] == True, "Graph should be valid"
    assert "analysis" in data, "Response should include analysis"
    assert "execution_order" in data, "Response should include execution_order"
    
    expected_order = [["root"], ["analysis"], ["mitigations"], ["summary"]]
    assert data["execution_order"] == expected_order, f"Unexpected execution order: {data['execution_order']}"
    
    print(f"  ✅ Task graph planned: {data['analysis']['node_count']} nodes")
    print(f"  ✅ Execution order: {data['execution_order']}")
    return data["graph_id"]


def test_dag_execution_endpoint():
    """Test /tasks/execute endpoint."""
    print("🧪 Testing /tasks/execute endpoint...")
    
    # Simple task graph for execution
    task_graph = {
        "nodes": [
            {"id": "plan", "prompt": "Create a system architecture plan", "agent": "Planner", "deps": []},
            {"id": "review", "prompt": "Review the architecture plan", "agent": "Reviewer", "deps": ["plan"]}
        ]
    }
    
    request_data = {
        "task_graph": task_graph,
        "context": {"project": "test_execution"}
    }
    
    response = requests.post("http://127.0.0.1:8081/tasks/execute", json=request_data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "execution_id" in data, "Response should include execution_id"
    assert data["status"] == "completed", f"Expected completed status, got {data['status']}"
    assert len(data["completed_tasks"]) == 2, f"Expected 2 completed tasks, got {len(data['completed_tasks'])}"
    assert len(data["failed_tasks"]) == 0, f"Expected 0 failed tasks, got {len(data['failed_tasks'])}"
    
    print(f"  ✅ DAG executed: {len(data['completed_tasks'])} tasks completed in {data['total_time'] or 0:.2f}s")
    print(f"  ✅ Execution ID: {data['execution_id']}")
    return data


def test_evaluation_endpoint():
    """Test /eval/run endpoint."""
    print("🧪 Testing /eval/run endpoint...")
    
    # Test single scenario evaluation
    scenario = {
        "name": "api_test_scenario",
        "mode": "brainstorm",
        "agents": ["Analyst", "Designer"],
        "topic": "API testing strategy",
        "success_criteria": [
            {
                "type": "keyword_presence",
                "name": "api_keywords",
                "must_include": ["API", "testing", "strategy"],
                "weight": 2.0
            },
            {
                "type": "length_check",
                "name": "response_length",
                "min_length": 50,
                "max_length": 1000,
                "weight": 1.0
            }
        ],
        "max_rounds": 3,
        "timeout": 30
    }
    
    request_data = {
        "scenario": scenario,
        "context": {"test_mode": True}
    }
    
    response = requests.post("http://127.0.0.1:8081/eval/run", json=request_data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "scenario_name" in data, "Response should include scenario_name"
    assert "execution_id" in data, "Response should include execution_id"
    assert data["status"] == "completed", f"Expected completed status, got {data['status']}"
    assert "metrics" in data, "Response should include metrics"
    
    metrics = data["metrics"]
    assert "success_rate" in metrics, "Metrics should include success_rate"
    assert "total_score" in metrics, "Metrics should include total_score"
    
    print(f"  ✅ Scenario evaluated: {data['scenario_name']}")
    print(f"  ✅ Success rate: {metrics['success_rate']:.2f}")
    print(f"  ✅ Execution time: {data['execution_time'] or 0:.2f}s")
    return data


def test_evaluation_suite_endpoint():
    """Test /eval/run endpoint with suite."""
    print("🧪 Testing /eval/run endpoint with suite...")
    
    # Test suite with multiple scenarios
    suite = {
        "suite": "api_test_suite",
        "description": "Test suite for API validation",
        "scenarios": [
            {
                "name": "quick_brainstorm",
                "mode": "brainstorm",
                "agents": ["Thinker1", "Thinker2"],
                "topic": "Quick brainstorming session",
                "success_criteria": [
                    {
                        "type": "keyword_presence",
                        "name": "brainstorm_keywords",
                        "must_include": ["idea", "think"],
                        "weight": 1.0
                    }
                ],
                "max_rounds": 2,
                "timeout": 20
            },
            {
                "name": "simple_debate",
                "mode": "debate",
                "agents": ["Advocate", "Critic"],
                "topic": "Simple debate topic",
                "success_criteria": [
                    {
                        "type": "length_check",
                        "name": "substantive_response",
                        "min_length": 30,
                        "weight": 1.0
                    }
                ],
                "max_rounds": 2,
                "timeout": 20
            }
        ]
    }
    
    request_data = {
        "suite": suite,
        "context": {"batch_test": True},
        "parallel": False
    }
    
    response = requests.post("http://127.0.0.1:8081/eval/run", json=request_data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "suite_name" in data, "Response should include suite_name"
    assert "execution_id" in data, "Response should include execution_id"
    assert "summary" in data, "Response should include summary"
    assert data["scenario_count"] == 2, f"Expected 2 scenarios, got {data['scenario_count']}"
    
    summary = data["summary"]
    assert "total_scenarios" in summary, "Summary should include total_scenarios"
    assert "success_rate" in summary, "Summary should include success_rate"
    
    print(f"  ✅ Suite evaluated: {data['suite_name']}")
    print(f"  ✅ Scenarios: {data['scenario_count']}")
    print(f"  ✅ Success rate: {summary['success_rate']:.2f}")
    print(f"  ✅ Total time: {data['total_time'] or 0:.2f}s")
    return data


def test_invalid_requests():
    """Test error handling for invalid requests."""
    print("🧪 Testing error handling...")
    
    # Test invalid task graph
    invalid_graph = {
        "nodes": [
            {"id": "a", "prompt": "Task A", "agent": "Agent1", "deps": ["b"]},
            {"id": "b", "prompt": "Task B", "agent": "Agent2", "deps": ["a"]}  # Cycle!
        ]
    }
    
    response = requests.post("http://127.0.0.1:8081/tasks/plan", json=invalid_graph)
    assert response.status_code == 200, "Should return 200 but with error in body"
    
    data = response.json()
    assert "error" in data or not data.get("valid", True), "Should indicate invalid graph"
    print(f"  ✅ Invalid graph rejected: {data.get('validation_errors', ['cycle detected'])}")
    
    # Test missing evaluation data
    response = requests.post("http://127.0.0.1:8081/eval/run", json={})
    assert response.status_code == 200, "Should return 200 but with error in body"
    
    data = response.json()
    assert "error" in data, "Should return error for missing data"
    print(f"  ✅ Missing evaluation data rejected: {data['error']}")


def main():
    """Run all P3 API tests."""
    print("🚀 P3 API Implementation Tests")
    print("=" * 50)
    
    # Start API server
    print("🔧 Starting API server...")
    api = start_api_server()
    print("  ✅ API server started on http://127.0.0.1:8081")
    print()
    
    start_time = time.time()
    
    try:
        # Test health endpoint first
        print("🧪 Testing health endpoint...")
        response = requests.get("http://127.0.0.1:8081/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        health_data = response.json()
        assert health_data["status"] == "healthy", "API should be healthy"
        print("  ✅ Health endpoint working")
        print()
        
        # Test P3 endpoints
        test_dag_planning_endpoint()
        print()
        
        test_dag_execution_endpoint()
        print()
        
        test_evaluation_endpoint()
        print()
        
        test_evaluation_suite_endpoint()
        print()
        
        test_invalid_requests()
        print()
        
        total_time = time.time() - start_time
        
        print("=" * 50)
        print("🎉 All P3 API Tests Passed!")
        print(f"⏱️  Total test time: {total_time:.2f}s")
        print()
        print("P3 API Endpoints Successfully Implemented:")
        print("  ✅ POST /tasks/plan - DAG planning and validation")
        print("  ✅ POST /tasks/execute - DAG execution with AgentNet")
        print("  ✅ POST /eval/run - Evaluation scenario and suite runner")
        print("  ✅ Error handling for invalid requests")
        print("  ✅ Integration with existing AgentNet infrastructure")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()