#!/usr/bin/env python3
"""
P3 Features Demo

Demonstrates the complete P3 DAG & Eval implementation:
- Task Graph Planning and Execution
- Evaluation Harness with multiple criteria
- API endpoints for workflow automation
- Integration with existing AgentNet components
"""

import asyncio
import json
import time
from pathlib import Path

# Import P3 components
from agentnet import (
    # DAG components
    DAGPlanner, TaskNode, TaskGraph, TaskScheduler, ExecutionResult,
    # Evaluation components
    EvaluationRunner, EvaluationScenario, EvaluationSuite, 
    MetricsCalculator, EvaluationMetrics, SuccessCriteria,
    # Existing components
    AgentNet, ExampleEngine, SessionManager
)
from agentnet.core.eval.metrics import CriteriaType


def demo_header(title: str):
    """Print a formatted demo section header."""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)


async def demo_dag_planner():
    """Demonstrate DAG planning capabilities."""
    demo_header("DAG Planner & Task Graph Execution")
    
    print("📋 Creating Task Graph from Roadmap Example...")
    
    # Example from roadmap: high availability design workflow
    task_graph_json = """
    {
      "metadata": {"project": "high_availability_system", "priority": "critical"},
      "nodes": [
        {
          "id": "requirements", 
          "prompt": "Define high availability requirements and constraints", 
          "agent": "RequirementsAnalyst", 
          "deps": [],
          "metadata": {"phase": "planning", "estimated_time": "30min"}
        },
        {
          "id": "architecture", 
          "prompt": "Design system architecture for high availability", 
          "agent": "SystemArchitect", 
          "deps": ["requirements"],
          "metadata": {"phase": "design", "estimated_time": "45min"}
        },
        {
          "id": "failure_analysis", 
          "prompt": "Analyze potential failure modes and impact", 
          "agent": "ReliabilityEngineer", 
          "deps": ["architecture"],
          "metadata": {"phase": "analysis", "estimated_time": "60min"}
        },
        {
          "id": "mitigation_strategies", 
          "prompt": "Develop failure mitigation and recovery strategies", 
          "agent": "RecoverySpecialist", 
          "deps": ["failure_analysis"],
          "metadata": {"phase": "mitigation", "estimated_time": "45min"}
        },
        {
          "id": "implementation_plan", 
          "prompt": "Create detailed implementation roadmap", 
          "agent": "ProjectManager", 
          "deps": ["architecture", "mitigation_strategies"],
          "metadata": {"phase": "planning", "estimated_time": "30min"}
        },
        {
          "id": "final_review", 
          "prompt": "Review complete high availability design", 
          "agent": "TechnicalReviewer", 
          "deps": ["implementation_plan"],
          "metadata": {"phase": "review", "estimated_time": "20min"}
        }
      ]
    }
    """
    
    # Create and validate DAG
    planner = DAGPlanner()
    task_graph = planner.create_graph_from_json(task_graph_json)
    
    print(f"  ✅ Task Graph Created: {len(task_graph.nodes)} nodes")
    print(f"  ✅ Validation Status: {'Valid' if task_graph.is_valid else 'Invalid'}")
    
    if not task_graph.is_valid:
        print(f"  ❌ Validation Errors: {task_graph.validation_errors}")
        return
    
    # Analyze the graph
    analysis = planner.analyze_graph(task_graph)
    execution_order = planner.get_execution_order(task_graph)
    
    print(f"\n📊 Graph Analysis:")
    print(f"  • Total nodes: {analysis['node_count']}")
    print(f"  • Max depth: {analysis['max_depth']}")
    print(f"  • Root nodes: {analysis['root_nodes']}")
    print(f"  • Leaf nodes: {analysis['leaf_nodes']}")
    print(f"  • Agents involved: {', '.join(analysis['agents_involved'])}")
    print(f"  • Execution stages: {analysis['execution_stages']}")
    
    print(f"\n🔄 Execution Order:")
    for i, stage in enumerate(execution_order, 1):
        print(f"  Stage {i}: {stage}")
    
    # Create AgentNet-powered task executor
    engine = ExampleEngine()
    
    async def agentnet_task_executor(task_id: str, prompt: str, agent_name: str, context: dict) -> dict:
        """Task executor using AgentNet with enhanced context."""
        # Create specialized agent based on role
        agent_styles = {
            "RequirementsAnalyst": {"logic": 0.9, "creativity": 0.4, "analytical": 0.95},
            "SystemArchitect": {"logic": 0.85, "creativity": 0.7, "analytical": 0.9},
            "ReliabilityEngineer": {"logic": 0.95, "creativity": 0.3, "analytical": 0.98},
            "RecoverySpecialist": {"logic": 0.8, "creativity": 0.6, "analytical": 0.85},
            "ProjectManager": {"logic": 0.7, "creativity": 0.5, "analytical": 0.8},
            "TechnicalReviewer": {"logic": 0.9, "creativity": 0.4, "analytical": 0.95}
        }
        
        style = agent_styles.get(agent_name, {"logic": 0.8, "creativity": 0.6, "analytical": 0.8})
        
        agent = AgentNet(
            name=agent_name,
            style=style,
            engine=engine
        )
        
        # Build enhanced prompt with context
        enhanced_prompt = f"{prompt}\n\nProject Context: High Availability System Design"
        
        if context.get("dependency_results"):
            enhanced_prompt += "\n\nPrevious Work Context:"
            for dep_id, dep_result in context["dependency_results"].items():
                content = dep_result.get("content", "No content available")
                enhanced_prompt += f"\n- From {dep_id}: {content[:200]}..."
        
        if context.get("metadata"):
            task_meta = context["metadata"]
            if "phase" in task_meta:
                enhanced_prompt += f"\n\nPhase: {task_meta['phase']}"
            if "estimated_time" in task_meta:
                enhanced_prompt += f"\nEstimated Time: {task_meta['estimated_time']}"
        
        # Execute with AgentNet
        result = agent.generate_reasoning_tree(enhanced_prompt)
        
        return {
            "content": result["result"]["content"],
            "confidence": result["result"]["confidence"],
            "agent": agent_name,
            "task_id": task_id,
            "meta_insights": result["result"].get("meta_insights", []),
            "style_applied": style,
            "execution_context": context.get("metadata", {})
        }
    
    # Execute the task graph
    print(f"\n⚙️  Executing Task Graph...")
    
    scheduler = TaskScheduler(
        max_retries=2, 
        parallel_execution=True,
        execution_timeout=30.0
    )
    scheduler.set_task_executor(agentnet_task_executor)
    
    start_time = time.time()
    execution_result = await scheduler.execute_graph(task_graph, {"project_priority": "critical"})
    execution_time = time.time() - start_time
    
    print(f"  ✅ Execution Status: {execution_result.status}")
    print(f"  ✅ Completed Tasks: {len(execution_result.get_completed_tasks())}")
    print(f"  ✅ Failed Tasks: {len(execution_result.get_failed_tasks())}")
    print(f"  ✅ Total Execution Time: {execution_time:.2f}s")
    
    # Display task results
    print(f"\n📋 Task Results Summary:")
    for task_id, task_result in execution_result.task_results.items():
        status_icon = "✅" if task_result.status == "completed" else "❌"
        print(f"  {status_icon} {task_id}: {task_result.status}")
        if task_result.status == "completed" and task_result.result:
            confidence = task_result.result.get("confidence", 0)
            agent = task_result.result.get("agent", "Unknown")
            print(f"      Agent: {agent} | Confidence: {confidence:.2f}")
            print(f"      Result: {task_result.result.get('content', '')[:100]}...")
    
    return execution_result


async def demo_evaluation_harness():
    """Demonstrate evaluation harness capabilities."""
    demo_header("Evaluation Harness & Metrics")
    
    print("📊 Testing Multiple Evaluation Criteria...")
    
    # Create comprehensive evaluation scenario
    scenario = EvaluationScenario(
        name="comprehensive_design_evaluation",
        mode="consensus",
        agents=["SystemArchitect", "SecurityExpert", "PerformanceEngineer"],
        topic="Design a scalable microservices architecture",
        success_criteria=[
            # Keyword presence - architectural terms
            SuccessCriteria(
                type=CriteriaType.KEYWORD_PRESENCE,
                name="architecture_keywords",
                must_include=["microservices", "scalable", "architecture", "design"],
                weight=2.0
            ),
            # Keyword absence - avoid problematic terms
            SuccessCriteria(
                type=CriteriaType.KEYWORD_ABSENCE,
                name="avoid_antipatterns",
                must_exclude=["monolith", "single-point-of-failure", "hardcoded"],
                weight=1.5
            ),
            # Response quality - substantial content
            SuccessCriteria(
                type=CriteriaType.LENGTH_CHECK,
                name="substantive_response",
                min_length=300,
                max_length=2000,
                weight=1.0
            ),
            # Confidence threshold
            SuccessCriteria(
                type=CriteriaType.CONFIDENCE_THRESHOLD,
                name="high_confidence",
                min_confidence=0.7,
                weight=1.5
            ),
            # No rule violations
            SuccessCriteria(
                type=CriteriaType.RULE_VIOLATIONS_COUNT,
                name="clean_execution",
                max_violations=0,
                weight=2.0
            )
        ],
        max_rounds=4,
        timeout=60
    )
    
    # Create evaluation runner with AgentNet integration
    runner = EvaluationRunner(results_dir="demo_eval_results")
    
    # Set up dialogue executor
    engine = ExampleEngine()
    
    async def demo_dialogue_executor(agents: list, topic: str, config: dict) -> dict:
        """Demo dialogue executor with enhanced agents."""
        agent_configs = []
        for agent_name in agents:
            if agent_name == "SystemArchitect":
                style = {"logic": 0.85, "creativity": 0.7, "analytical": 0.9}
            elif agent_name == "SecurityExpert":
                style = {"logic": 0.9, "creativity": 0.4, "analytical": 0.95}
            elif agent_name == "PerformanceEngineer":
                style = {"logic": 0.9, "creativity": 0.5, "analytical": 0.9}
            else:
                style = {"logic": 0.8, "creativity": 0.6, "analytical": 0.8}
            
            agent_configs.append({
                "name": agent_name,
                "style": style
            })
        
        # Simulate a comprehensive session (would normally use SessionManager)
        session_result = {
            "transcript": [
                {
                    "agent": "SystemArchitect",
                    "content": f"For {topic}, I recommend a microservices architecture with API gateways, service discovery, and container orchestration. This design provides scalable, maintainable services that can be developed and deployed independently.",
                    "confidence": 0.85
                },
                {
                    "agent": "SecurityExpert", 
                    "content": f"The architecture should include OAuth2/JWT authentication, mTLS between services, API rate limiting, and network segmentation. Avoid hardcoded secrets and implement zero-trust principles.",
                    "confidence": 0.9
                },
                {
                    "agent": "PerformanceEngineer",
                    "content": f"For optimal performance, implement caching strategies, database connection pooling, asynchronous processing, and circuit breakers. Monitor service latency and implement auto-scaling based on demand.",
                    "confidence": 0.88
                }
            ],
            "converged": True,
            "rounds": 3,
            "violations": [],
            "final_synthesis": {
                "content": f"Consensus reached on {topic}: A secure, scalable microservices architecture with proper authentication, performance optimization, and monitoring. The design avoids monolithic patterns and single points of failure.",
                "confidence": 0.87
            }
        }
        
        return session_result
    
    runner.set_dialogue_executor(demo_dialogue_executor)
    
    # Run the evaluation
    print(f"  🎯 Running scenario: {scenario.name}")
    print(f"  👥 Agents: {', '.join(scenario.agents)}")
    print(f"  📝 Topic: {scenario.topic}")
    print(f"  ⚖️  Criteria: {len(scenario.success_criteria)} evaluation criteria")
    
    start_time = time.time()
    result = await runner.run_scenario(scenario)
    eval_time = time.time() - start_time
    
    print(f"\n📊 Evaluation Results:")
    print(f"  ✅ Status: {result.status}")
    print(f"  ⏱️  Execution Time: {eval_time:.2f}s")
    
    if result.metrics:
        metrics = result.metrics
        print(f"  🎯 Overall Success Rate: {metrics.success_rate:.2f}")
        print(f"  📈 Total Score: {metrics.total_score:.2f}/{metrics.max_possible_score:.2f}")
        print(f"  📊 Coverage Score: {metrics.coverage_score:.2f}")
        print(f"  🆕 Novelty Score: {metrics.novelty_score:.2f}")
        print(f"  🎭 Coherence Score: {metrics.coherence_score:.2f}")
        print(f"  ⚠️  Rule Violations: {metrics.rule_violations_count}")
        
        print(f"\n🔍 Detailed Criteria Results:")
        for criteria_result in metrics.criteria_results:
            status_icon = "✅" if criteria_result.passed else "❌"
            print(f"    {status_icon} {criteria_result.criteria_name}: {criteria_result.score:.2f} (weight: {criteria_result.weight})")
            if criteria_result.details:
                # Show relevant details
                if criteria_result.criteria_type == CriteriaType.KEYWORD_PRESENCE:
                    found = criteria_result.details.get("found_keywords", [])
                    missing = criteria_result.details.get("missing_keywords", [])
                    print(f"        Found: {found}")
                    if missing:
                        print(f"        Missing: {missing}")
                elif criteria_result.criteria_type == CriteriaType.LENGTH_CHECK:
                    length = criteria_result.details.get("content_length", 0)
                    print(f"        Content length: {length} characters")
    
    return result


async def demo_yaml_evaluation_suite():
    """Demonstrate YAML-based evaluation suite."""
    demo_header("YAML Evaluation Suite")
    
    print("📁 Loading Evaluation Suite from YAML...")
    
    # Load the baseline evaluation suite
    suite_path = Path("configs/eval_scenarios/baseline_design_eval.yaml")
    if not suite_path.exists():
        print(f"  ⚠️  YAML suite file not found: {suite_path}")
        return
    
    suite = EvaluationSuite.from_yaml_file(suite_path)
    print(f"  ✅ Loaded suite: {suite.name}")
    print(f"  📄 Description: {suite.description}")
    print(f"  🧪 Scenarios: {len(suite.scenarios)}")
    
    # Display scenario summary
    print(f"\n📋 Scenario Overview:")
    for scenario in suite.scenarios:
        print(f"    • {scenario.name} ({scenario.mode})")
        print(f"      Agents: {', '.join(scenario.agents)}")
        print(f"      Topic: {scenario.topic}")
        print(f"      Criteria: {len(scenario.success_criteria)} checks")
        if scenario.mode == "workflow" and scenario.task_graph:
            print(f"      Task Graph: {len(scenario.task_graph.get('nodes', []))} nodes")
    
    # Create runner with demo executors
    runner = EvaluationRunner(results_dir="demo_eval_results")
    
    # Set up executors (simplified for demo)
    async def demo_dialogue_executor(agents: list, topic: str, config: dict) -> dict:
        mode = config.get("mode", "general")
        
        if mode == "brainstorm":
            content = f"Brainstorming {topic}: We need redundancy, failover mechanisms, and recovery procedures for robust systems."
        elif mode == "debate":
            content = f"Debating {topic}: Ethics and responsibility are crucial in autonomous systems. We must ensure proper decision hierarchies."
        elif mode == "consensus":
            content = f"Building consensus on {topic}: Sustainable and renewable energy transition requires balanced approaches. We agree on the need for comprehensive planning."
        else:
            content = f"Discussing {topic}: Comprehensive analysis and strategic planning are essential."
        
        return {
            "transcript": [
                {"agent": agents[0], "content": content, "confidence": 0.85},
                {"agent": agents[1], "content": f"I agree. {content}", "confidence": 0.8}
            ],
            "converged": True,
            "rounds": 2,
            "violations": []
        }
    
    async def demo_workflow_executor(task_graph: dict, config: dict) -> dict:
        nodes = task_graph.get("nodes", [])
        task_results = {}
        
        for node in nodes:
            task_results[node["id"]] = {
                "status": "completed",
                "result": {
                    "content": f"Completed {node['prompt']} - architecture design includes key components and system integration points.",
                    "confidence": 0.8
                }
            }
        
        return {
            "status": "completed",
            "task_results": task_results,
            "execution_time": 0.5
        }
    
    runner.set_dialogue_executor(demo_dialogue_executor)
    runner.set_workflow_executor(demo_workflow_executor)
    
    # Run first few scenarios (to keep demo time reasonable)
    demo_scenarios = suite.scenarios[:2]  # Run first 2 scenarios
    
    print(f"\n🚀 Running {len(demo_scenarios)} Demo Scenarios...")
    
    for scenario in demo_scenarios:
        print(f"\n  🧪 Running: {scenario.name}")
        result = await runner.run_scenario(scenario)
        
        if result.status == "completed" and result.metrics:
            print(f"    ✅ Success Rate: {result.metrics.success_rate:.2f}")
            print(f"    📊 Score: {result.metrics.total_score:.2f}/{result.metrics.max_possible_score:.2f}")
        else:
            print(f"    ❌ Status: {result.status}")
            if result.error:
                print(f"    Error: {result.error}")


async def demo_api_integration():
    """Demonstrate P3 API integration (without actual server)."""
    demo_header("API Integration Showcase")
    
    print("🔌 P3 API Endpoints Available:")
    print("  📋 POST /tasks/plan - DAG planning and validation")
    print("  ⚙️  POST /tasks/execute - DAG execution with dependency resolution")
    print("  🧪 POST /eval/run - Evaluation scenario and suite runner")
    
    print(f"\n📊 Example API Usage:")
    
    # Example task planning request
    task_plan_request = {
        "nodes": [
            {"id": "design", "prompt": "Create system design", "agent": "Architect", "deps": []},
            {"id": "review", "prompt": "Review the design", "agent": "Reviewer", "deps": ["design"]}
        ]
    }
    
    print(f"  📋 Task Planning Request:")
    print(f"     POST /tasks/plan")
    print(f"     {json.dumps(task_plan_request, indent=6)}")
    
    # Example evaluation request
    eval_request = {
        "scenario": {
            "name": "api_design_review",
            "mode": "consensus",
            "agents": ["Designer", "Reviewer"],
            "topic": "API design best practices",
            "success_criteria": [
                {
                    "type": "keyword_presence",
                    "name": "api_keywords",
                    "must_include": ["REST", "API", "design"],
                    "weight": 1.0
                }
            ],
            "max_rounds": 3
        }
    }
    
    print(f"\n  🧪 Evaluation Request:")  
    print(f"     POST /eval/run")
    print(f"     {json.dumps(eval_request, indent=6)}")
    
    print(f"\n🏗️  API Integration Benefits:")
    print("  • Programmatic workflow automation")
    print("  • Batch evaluation processing")
    print("  • Integration with CI/CD pipelines")
    print("  • Quality gates and regression testing")
    print("  • Scalable multi-agent orchestration")


async def main():
    """Run the complete P3 features demo."""
    print("🎉 AgentNet P3 Features Demo")
    print("Task Graph Planner & Evaluation Harness")
    print("=" * 60)
    
    start_time = time.time()
    
    # Demo 1: DAG Planning and Execution
    dag_result = await demo_dag_planner()
    
    # Demo 2: Evaluation Harness
    eval_result = await demo_evaluation_harness()
    
    # Demo 3: YAML Evaluation Suite
    await demo_yaml_evaluation_suite()
    
    # Demo 4: API Integration
    await demo_api_integration()
    
    total_time = time.time() - start_time
    
    # Final Summary
    demo_header("P3 Features Summary")
    
    print("🎯 Implemented Capabilities:")
    print("  ✅ DAG Task Graph Planning with networkx")
    print("  ✅ Task scheduling with dependency resolution")
    print("  ✅ Retry logic and fallback agent support")
    print("  ✅ AgentNet integration for task execution")
    print("  ✅ Multi-criteria evaluation harness")
    print("  ✅ YAML-based scenario configuration")
    print("  ✅ Comprehensive metrics calculation")
    print("  ✅ API endpoints for workflow automation")
    print("  ✅ Integration with existing P0/P1/P2 features")
    
    print(f"\n📊 Demo Statistics:")
    if dag_result:
        print(f"  • DAG Tasks Executed: {len(dag_result.get_completed_tasks())}")
    if eval_result and eval_result.metrics:
        print(f"  • Evaluation Success Rate: {eval_result.metrics.success_rate:.2f}")
        print(f"  • Criteria Evaluated: {len(eval_result.metrics.criteria_results)}")
    print(f"  • Total Demo Time: {total_time:.2f}s")
    
    print(f"\n🚀 P3 Implementation Complete!")
    print("Ready for production deployment and integration with:")
    print("  - CI/CD pipelines for automated evaluation")
    print("  - Workflow orchestration systems")
    print("  - Quality assurance and regression testing")
    print("  - Multi-agent task automation")


if __name__ == "__main__":
    asyncio.run(main())