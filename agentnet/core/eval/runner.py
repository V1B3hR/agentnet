"""
Evaluation Runner for Scenario Execution

Implements the evaluation harness that executes scenarios and collects metrics.
Based on FR12 requirements from docs/RoadmapAgentNet.md.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import yaml

from .metrics import MetricsCalculator, EvaluationMetrics, SuccessCriteria, CriteriaType

logger = logging.getLogger("agentnet.eval.runner")


@dataclass
class EvaluationScenario:
    """
    Represents a single evaluation scenario.
    
    Based on roadmap example:
    - name: "resilience_planning"
      mode: "brainstorm"
      agents: ["Athena", "Apollo"]
      topic: "Edge network partition recovery"
      success_criteria: [...]
    """
    name: str
    mode: str  # brainstorm, debate, consensus, workflow
    agents: List[str]
    topic: str
    success_criteria: List[SuccessCriteria] = field(default_factory=list)
    max_rounds: Optional[int] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For workflow mode
    task_graph: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "mode": self.mode,
            "agents": self.agents,
            "topic": self.topic,
            "success_criteria": [
                {
                    "type": criteria.type,
                    "name": criteria.name,
                    "weight": criteria.weight,
                    "must_include": criteria.must_include,
                    "must_exclude": criteria.must_exclude,
                    "case_sensitive": criteria.case_sensitive,
                    "reference_text": criteria.reference_text,
                    "reference_id": criteria.reference_id,
                    "min_score": criteria.min_score,
                    "max_score": criteria.max_score,
                    "min_length": criteria.min_length,
                    "max_length": criteria.max_length,
                    "pattern": criteria.pattern,
                    "flags": criteria.flags,
                    "min_confidence": criteria.min_confidence,
                    "max_confidence": criteria.max_confidence,
                    "max_violations": criteria.max_violations,
                    "violation_types": criteria.violation_types,
                    "function_name": criteria.function_name,
                    "function_params": criteria.function_params
                }
                for criteria in self.success_criteria
            ],
            "max_rounds": self.max_rounds,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "task_graph": self.task_graph
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationScenario":
        """Create EvaluationScenario from dictionary."""
        criteria = []
        for criteria_data in data.get("success_criteria", []):
            criteria_obj = SuccessCriteria(
                type=CriteriaType(criteria_data["type"]),
                name=criteria_data.get("name"),
                weight=criteria_data.get("weight", 1.0),
                must_include=criteria_data.get("must_include"),
                must_exclude=criteria_data.get("must_exclude"),
                case_sensitive=criteria_data.get("case_sensitive", False),
                reference_text=criteria_data.get("reference_text"),
                reference_id=criteria_data.get("reference_id"),
                min_score=criteria_data.get("min_score"),
                max_score=criteria_data.get("max_score"),
                min_length=criteria_data.get("min_length"),
                max_length=criteria_data.get("max_length"),
                pattern=criteria_data.get("pattern"),
                flags=criteria_data.get("flags", 0),
                min_confidence=criteria_data.get("min_confidence"),
                max_confidence=criteria_data.get("max_confidence"),
                max_violations=criteria_data.get("max_violations"),
                violation_types=criteria_data.get("violation_types"),
                function_name=criteria_data.get("function_name"),
                function_params=criteria_data.get("function_params")
            )
            criteria.append(criteria_obj)
        
        return cls(
            name=data["name"],
            mode=data["mode"],
            agents=data["agents"],
            topic=data["topic"],
            success_criteria=criteria,
            max_rounds=data.get("max_rounds"), 
            timeout=data.get("timeout"),
            metadata=data.get("metadata", {}),
            task_graph=data.get("task_graph")
        )


@dataclass
class EvaluationSuite:
    """
    Represents a complete evaluation suite with multiple scenarios.
    
    Based on roadmap example:
    suite: "baseline_design_eval"
    scenarios: [...]
    """
    name: str
    description: Optional[str] = None
    scenarios: List[EvaluationScenario] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite": self.name,
            "description": self.description,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSuite":
        """Create EvaluationSuite from dictionary."""
        scenarios = [
            EvaluationScenario.from_dict(scenario_data)
            for scenario_data in data.get("scenarios", [])
        ]
        
        return cls(
            name=data.get("suite", "unnamed_suite"),
            description=data.get("description"),
            scenarios=scenarios,
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> "EvaluationSuite":
        """Load evaluation suite from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> "EvaluationSuite":
        """Load evaluation suite from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ScenarioResult:
    """Result of running a single scenario."""
    scenario_name: str
    execution_id: str
    status: str  # completed, failed, timeout
    metrics: Optional[EvaluationMetrics] = None
    session_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.execution_time = self.end_time - self.start_time


@dataclass
class SuiteResult:
    """Result of running a complete evaluation suite."""
    suite_name: str
    execution_id: str
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_time: Optional[float] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.total_time = self.end_time - self.start_time
        
        # Calculate summary statistics
        self._calculate_summary()
    
    def _calculate_summary(self):
        """Calculate summary statistics for the suite."""
        if not self.scenario_results:
            return
        
        completed = [r for r in self.scenario_results if r.status == "completed"]
        failed = [r for r in self.scenario_results if r.status == "failed"]
        timeout = [r for r in self.scenario_results if r.status == "timeout"]
        
        total_scenarios = len(self.scenario_results)
        success_rate = len(completed) / total_scenarios if total_scenarios > 0 else 0.0
        
        # Aggregate metrics from successful scenarios
        total_score = 0.0
        max_possible_score = 0.0
        avg_coverage = 0.0
        avg_novelty = 0.0
        avg_coherence = 0.0
        total_violations = 0
        
        if completed:
            for result in completed:
                if result.metrics:
                    total_score += result.metrics.total_score
                    max_possible_score += result.metrics.max_possible_score
                    avg_coverage += result.metrics.coverage_score
                    avg_novelty += result.metrics.novelty_score
                    avg_coherence += result.metrics.coherence_score
                    total_violations += result.metrics.rule_violations_count
            
            avg_coverage /= len(completed)
            avg_novelty /= len(completed)
            avg_coherence /= len(completed)
        
        overall_score = total_score / max_possible_score if max_possible_score > 0 else 0.0
        
        self.summary = {
            "total_scenarios": total_scenarios,
            "completed_scenarios": len(completed),
            "failed_scenarios": len(failed),
            "timeout_scenarios": len(timeout),
            "success_rate": success_rate,
            "overall_score": overall_score,
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "avg_coverage_score": avg_coverage,
            "avg_novelty_score": avg_novelty,
            "avg_coherence_score": avg_coherence,
            "total_violations": total_violations,
            "execution_time": self.total_time
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite_name": self.suite_name,
            "execution_id": self.execution_id,
            "scenario_results": [
                {
                    "scenario_name": r.scenario_name,
                    "execution_id": r.execution_id,
                    "status": r.status,
                    "metrics": r.metrics.to_dict() if r.metrics else None,
                    "session_data": r.session_data,
                    "error": r.error,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "execution_time": r.execution_time
                }
                for r in self.scenario_results
            ],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.total_time,
            "summary": self.summary
        }


class EvaluationRunner:
    """
    Evaluation Runner executes scenarios and collects metrics.
    
    Key capabilities:
    - Execute individual scenarios or complete suites
    - Support multiple execution modes (dialogue, workflow)
    - Collect and aggregate metrics
    - Store results for comparison and regression testing
    """
    
    def __init__(
        self,
        results_dir: Optional[Union[str, Path]] = None,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            results_dir: Directory to store evaluation results
            metrics_calculator: Custom metrics calculator (optional)
        """
        self.results_dir = Path(results_dir) if results_dir else Path("eval_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.logger = logger
        
        # Execution functions - to be set by user
        self.dialogue_executor: Optional[Callable] = None
        self.workflow_executor: Optional[Callable] = None
    
    def set_dialogue_executor(self, executor: Callable[[List[str], str, Dict[str, Any]], Dict[str, Any]]):
        """
        Set the dialogue executor function for multi-agent scenarios.
        
        The executor should accept (agents, topic, config) and return session data.
        Example:
            def execute_dialogue(agents, topic, config):
                # Run multi-agent dialogue
                return {"transcript": [...], "converged": True, "rounds": 3}
        
        Args:
            executor: Function to execute multi-agent dialogues
        """
        self.dialogue_executor = executor
    
    def set_workflow_executor(self, executor: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]):
        """
        Set the workflow executor function for DAG scenarios.
        
        The executor should accept (task_graph, config) and return execution results.
        Example:
            def execute_workflow(task_graph, config):
                # Execute DAG workflow
                return {"status": "completed", "task_results": {...}}
        
        Args:
            executor: Function to execute workflow scenarios
        """
        self.workflow_executor = executor
    
    async def run_scenario(
        self,
        scenario: EvaluationScenario,
        context: Optional[Dict[str, Any]] = None
    ) -> ScenarioResult:
        """
        Run a single evaluation scenario.
        
        Args:
            scenario: EvaluationScenario to execute
            context: Optional context to pass to executors
            
        Returns:
            ScenarioResult with execution details and metrics
        """
        execution_id = str(uuid.uuid4())
        context = context or {}
        
        result = ScenarioResult(
            scenario_name=scenario.name,
            execution_id=execution_id,
            status="running",
            start_time=time.time()
        )
        
        self.logger.info(f"Running scenario '{scenario.name}' (mode: {scenario.mode})")
        
        try:
            # Execute based on mode
            if scenario.mode in ["brainstorm", "debate", "consensus", "general"]:
                session_data = await self._execute_dialogue_scenario(scenario, context)
            elif scenario.mode == "workflow":
                session_data = await self._execute_workflow_scenario(scenario, context)
            else:
                raise ValueError(f"Unsupported scenario mode: {scenario.mode}")
            
            result.session_data = session_data
            result.status = "completed"
            
            # Calculate metrics
            content = self._extract_content_for_evaluation(session_data, scenario)
            metadata = self._extract_metadata_for_evaluation(session_data, scenario)
            
            result.metrics = self.metrics_calculator.evaluate_content(
                content, scenario.success_criteria, scenario.name, metadata
            )
            
            # Add execution time to metrics
            if result.metrics:
                result.metrics.execution_time = result.execution_time
            
            self.logger.info(f"Scenario '{scenario.name}' completed with score {result.metrics.success_rate:.2f}")
            
        except asyncio.TimeoutError:
            result.status = "timeout"
            result.error = f"Scenario timed out after {scenario.timeout}s"
            self.logger.warning(f"Scenario '{scenario.name}' timed out")
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            self.logger.error(f"Scenario '{scenario.name}' failed: {str(e)}")
        
        finally:
            result.end_time = time.time()
        
        return result
    
    async def run_suite(
        self,
        suite: EvaluationSuite,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = False
    ) -> SuiteResult:
        """
        Run a complete evaluation suite.
        
        Args:
            suite: EvaluationSuite to execute
            context: Optional context to pass to scenarios
            parallel: Whether to run scenarios in parallel
            
        Returns:
            SuiteResult with all scenario results and summary
        """
        execution_id = str(uuid.uuid4())
        
        suite_result = SuiteResult(
            suite_name=suite.name,
            execution_id=execution_id,
            start_time=time.time()
        )
        
        self.logger.info(f"Running evaluation suite '{suite.name}' with {len(suite.scenarios)} scenarios")
        
        try:
            if parallel and len(suite.scenarios) > 1:
                # Run scenarios in parallel
                tasks = [
                    self.run_scenario(scenario, context)
                    for scenario in suite.scenarios
                ]
                scenario_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                for i, result_or_exception in enumerate(scenario_results):
                    if isinstance(result_or_exception, Exception):
                        scenario_results[i] = ScenarioResult(
                            scenario_name=suite.scenarios[i].name,
                            execution_id=str(uuid.uuid4()),
                            status="failed",
                            error=str(result_or_exception),
                            start_time=time.time(),
                            end_time=time.time()
                        )
                
                suite_result.scenario_results = scenario_results
                
            else:
                # Run scenarios sequentially
                for scenario in suite.scenarios:
                    scenario_result = await self.run_scenario(scenario, context)
                    suite_result.scenario_results.append(scenario_result)
            
            # Calculate final summary
            suite_result._calculate_summary()
            self.logger.info(f"Suite '{suite.name}' completed: {suite_result.summary.get('success_rate', 0):.2f} success rate")
            
        except Exception as e:
            self.logger.error(f"Suite '{suite.name}' execution failed: {str(e)}")
        
        finally:
            suite_result.end_time = time.time()
            # Recalculate summary now that all scenarios are complete
            suite_result._calculate_summary()
        
        # Save results
        await self._save_suite_result(suite_result)
        
        return suite_result
    
    async def _execute_dialogue_scenario(
        self,
        scenario: EvaluationScenario,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a dialogue-based scenario."""
        if not self.dialogue_executor:
            raise ValueError("Dialogue executor not set")
        
        config = {
            "mode": scenario.mode,
            "max_rounds": scenario.max_rounds,
            "timeout": scenario.timeout,
            **scenario.metadata,
            **context
        }
        
        # Apply timeout if specified
        if scenario.timeout:
            return await asyncio.wait_for(
                self._call_dialogue_executor(scenario.agents, scenario.topic, config),
                timeout=scenario.timeout
            )
        else:
            return await self._call_dialogue_executor(scenario.agents, scenario.topic, config)
    
    async def _execute_workflow_scenario(
        self,
        scenario: EvaluationScenario,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a workflow-based scenario."""
        if not self.workflow_executor:
            raise ValueError("Workflow executor not set")
        
        if not scenario.task_graph:
            raise ValueError("Workflow scenario must specify task_graph")
        
        config = {
            "timeout": scenario.timeout,
            **scenario.metadata,
            **context
        }
        
        # Apply timeout if specified
        if scenario.timeout:
            return await asyncio.wait_for(
                self._call_workflow_executor(scenario.task_graph, config),
                timeout=scenario.timeout
            )
        else:
            return await self._call_workflow_executor(scenario.task_graph, config)
    
    async def _call_dialogue_executor(
        self,
        agents: List[str],
        topic: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call dialogue executor, handling both sync and async."""
        if asyncio.iscoroutinefunction(self.dialogue_executor):
            return await self.dialogue_executor(agents, topic, config)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.dialogue_executor, agents, topic, config
            )
    
    async def _call_workflow_executor(
        self,
        task_graph: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call workflow executor, handling both sync and async."""
        if asyncio.iscoroutinefunction(self.workflow_executor):
            return await self.workflow_executor(task_graph, config)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.workflow_executor, task_graph, config
            )
    
    def _extract_content_for_evaluation(
        self,
        session_data: Dict[str, Any],
        scenario: EvaluationScenario
    ) -> str:
        """Extract content from session data for evaluation."""
        # Extract based on scenario mode
        if scenario.mode == "workflow":
            # For workflow, concatenate all task results
            content_parts = []
            if "task_results" in session_data:
                for task_id, task_result in session_data["task_results"].items():
                    if isinstance(task_result, dict) and "result" in task_result:
                        result_data = task_result["result"]
                        if isinstance(result_data, dict) and "content" in result_data:
                            content_parts.append(result_data["content"])
                        else:
                            content_parts.append(str(result_data))
            return "\n\n".join(content_parts)
        else:
            # For dialogue, extract from transcript
            content_parts = []
            if "transcript" in session_data:
                for turn in session_data["transcript"]:
                    if isinstance(turn, dict) and "content" in turn:
                        content_parts.append(turn["content"])
                    else:
                        content_parts.append(str(turn))
            
            # Also include final synthesis if available
            if "final_synthesis" in session_data:
                synthesis = session_data["final_synthesis"]
                if isinstance(synthesis, dict) and "content" in synthesis:
                    content_parts.append(synthesis["content"])
                else:
                    content_parts.append(str(synthesis))
            
            return "\n\n".join(content_parts)
    
    def _extract_metadata_for_evaluation(
        self,
        session_data: Dict[str, Any],
        scenario: EvaluationScenario
    ) -> Dict[str, Any]:
        """Extract metadata from session data for evaluation."""
        metadata = {
            "scenario_name": scenario.name,
            "scenario_mode": scenario.mode,
            "agents": scenario.agents
        }
        
        # Add session-specific metadata
        if "converged" in session_data:
            metadata["converged"] = session_data["converged"]
        if "rounds" in session_data:
            metadata["rounds"] = session_data["rounds"]
        if "violations" in session_data:
            metadata["violations"] = session_data["violations"]
        if "status" in session_data:
            metadata["status"] = session_data["status"]
        
        # Extract confidence from final result
        confidence = None
        if "final_synthesis" in session_data:
            synthesis = session_data["final_synthesis"]
            if isinstance(synthesis, dict) and "confidence" in synthesis:
                confidence = synthesis["confidence"]
        elif "transcript" in session_data and session_data["transcript"]:
            # Use confidence from last turn
            last_turn = session_data["transcript"][-1]
            if isinstance(last_turn, dict) and "confidence" in last_turn:
                confidence = last_turn["confidence"]
        
        if confidence is not None:
            metadata["confidence"] = confidence
        
        return metadata
    
    async def _save_suite_result(self, suite_result: SuiteResult):
        """Save suite result to disk."""
        try:
            filename = f"{suite_result.suite_name}_{suite_result.execution_id}_{int(time.time())}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(suite_result.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Saved evaluation results to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {str(e)}")
    
    def run_scenario_sync(
        self,
        scenario: EvaluationScenario,
        context: Optional[Dict[str, Any]] = None
    ) -> ScenarioResult:
        """Synchronous wrapper for run_scenario."""
        return asyncio.run(self.run_scenario(scenario, context))
    
    def run_suite_sync(
        self,
        suite: EvaluationSuite,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = False
    ) -> SuiteResult:
        """Synchronous wrapper for run_suite."""
        return asyncio.run(self.run_suite(suite, context, parallel))


# Example usage and testing
if __name__ == "__main__":
    # Example evaluation suite based on roadmap
    suite_yaml = """
suite: "baseline_design_eval"
description: "Baseline evaluation for design and planning scenarios"
scenarios:
  - name: "resilience_planning"
    mode: "brainstorm"
    agents: ["Athena", "Apollo"]
    topic: "Edge network partition recovery"
    success_criteria:
      - type: keyword_presence
        name: "resilience_keywords"
        must_include: ["redundancy", "failover"]
        weight: 2.0
      - type: semantic_score
        name: "relevance_check"
        reference_text: "Network resilience requires redundant systems and failover mechanisms"
        min_score: 0.3
        weight: 1.5
    max_rounds: 5
    timeout: 30
  - name: "ethical_debate"
    mode: "debate"
    agents: ["Athena", "Hermes"]
    topic: "Autonomous swarm decision hierarchy"
    success_criteria:
      - type: keyword_presence
        name: "ethics_keywords"
        must_include: ["ethics", "responsibility"]
        weight: 1.0
      - type: length_check
        name: "response_length"
        min_length: 200
        weight: 1.0
    max_rounds: 3
    timeout: 25
"""
    
    async def example_dialogue_executor(agents, topic, config):
        """Example dialogue executor."""
        await asyncio.sleep(0.5)  # Simulate execution
        return {
            "transcript": [
                {"agent": agents[0], "content": f"Let's discuss {topic}. We need redundancy and failover systems for resilience.", "confidence": 0.8},
                {"agent": agents[1], "content": f"I agree on redundancy. Failover mechanisms are crucial for network partition recovery.", "confidence": 0.85}
            ],
            "converged": True,
            "rounds": 2,
            "violations": []
        }
    
    async def test_evaluation_runner():
        # Create evaluation suite
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(suite_yaml)
            suite_file = f.name
        
        try:
            suite = EvaluationSuite.from_yaml_file(suite_file)
            print(f"Loaded suite '{suite.name}' with {len(suite.scenarios)} scenarios")
            
            # Create runner
            runner = EvaluationRunner(results_dir="test_eval_results")
            runner.set_dialogue_executor(example_dialogue_executor)
            
            # Run suite
            result = await runner.run_suite(suite)
            
            print(f"\nSuite Results:")
            print(f"Success Rate: {result.summary['success_rate']:.2f}")
            print(f"Overall Score: {result.summary['overall_score']:.2f}")
            print(f"Execution Time: {result.total_time:.2f}s")
            
            for scenario_result in result.scenario_results:
                print(f"  {scenario_result.scenario_name}: {scenario_result.status}")
                if scenario_result.metrics:
                    print(f"    Score: {scenario_result.metrics.success_rate:.2f}")
        
        finally:
            os.unlink(suite_file)
    
    asyncio.run(test_evaluation_runner())