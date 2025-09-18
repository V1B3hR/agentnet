"""
Task Scheduler for DAG Execution

Implements the scheduler component that executes DAG tasks with dependency resolution,
retry logic, and fallback handling. Based on FR7 requirements from docs/RoadmapAgentNet.md.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from .dag_planner import TaskGraph, TaskNode

logger = logging.getLogger("agentnet.orchestration.scheduler")


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class ExecutionStatus(str, Enum):
    """Overall execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    attempt: int = 1
    agent_used: Optional[str] = None
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.execution_time = self.end_time - self.start_time


@dataclass 
class ExecutionResult:
    """Result of a complete DAG execution."""
    execution_id: str
    task_graph_id: str
    status: ExecutionStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.total_time = self.end_time - self.start_time
        else:
            self.total_time = None
    
    def get_completed_tasks(self) -> List[str]:
        """Get list of successfully completed task IDs."""
        return [
            task_id for task_id, result in self.task_results.items()
            if result.status == TaskStatus.COMPLETED
        ]
    
    def get_failed_tasks(self) -> List[str]:
        """Get list of failed task IDs."""
        return [
            task_id for task_id, result in self.task_results.items() 
            if result.status == TaskStatus.FAILED
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "execution_id": self.execution_id,
            "task_graph_id": self.task_graph_id,
            "status": self.status,
            "task_results": {
                task_id: {
                    "task_id": result.task_id,
                    "status": result.status,
                    "result": result.result,
                    "error": result.error,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "execution_time": result.execution_time,
                    "attempt": result.attempt,
                    "agent_used": result.agent_used
                }
                for task_id, result in self.task_results.items()
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.total_time,
            "error": self.error,
            "metadata": self.metadata
        }


class TaskScheduler:
    """
    Task Scheduler executes DAG tasks with dependency resolution.
    
    Key capabilities:
    - Execute ready nodes when dependencies complete
    - Retry failed tasks with configurable limits
    - Fallback agent support for persistent failures
    - Parallel execution of independent tasks
    - Aggregated final synthesis (optional)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_fallback: bool = True,
        fallback_agent: Optional[str] = None,
        execution_timeout: float = 300.0,  # 5 minutes default
        parallel_execution: bool = True
    ):
        """
        Initialize the task scheduler.
        
        Args:
            max_retries: Maximum retry attempts per task
            retry_delay: Delay between retry attempts (seconds)
            enable_fallback: Whether to use fallback agents on persistent failure
            fallback_agent: Default fallback agent name
            execution_timeout: Maximum time for single task execution
            parallel_execution: Whether to execute independent tasks in parallel
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_fallback = enable_fallback
        self.fallback_agent = fallback_agent
        self.execution_timeout = execution_timeout
        self.parallel_execution = parallel_execution
        self.logger = logger
        
        # Task executor function - will be set by user
        self.task_executor: Optional[Callable] = None
    
    def set_task_executor(self, executor: Callable[[str, str, str, Dict[str, Any]], Dict[str, Any]]):
        """
        Set the task executor function.
        
        The executor should accept (task_id, prompt, agent, context) and return a result dict.
        Example:
            def execute_task(task_id, prompt, agent, context):
                # Execute the task using the agent
                return {"content": "result", "confidence": 0.8}
        
        Args:
            executor: Function to execute individual tasks
        """
        self.task_executor = executor
    
    async def execute_graph(
        self,
        task_graph: TaskGraph,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a complete task graph.
        
        Args:
            task_graph: Valid TaskGraph to execute
            context: Optional context to pass to task executor
            
        Returns:
            ExecutionResult with complete execution details
            
        Raises:
            ValueError: If task graph is invalid or executor not set
        """
        if not task_graph.is_valid:
            raise ValueError(f"Cannot execute invalid task graph: {task_graph.validation_errors}")
        
        if not self.task_executor:
            raise ValueError("Task executor must be set before executing graph")
        
        execution_id = str(uuid.uuid4())
        context = context or {}
        
        # Initialize execution result
        execution = ExecutionResult(
            execution_id=execution_id,
            task_graph_id=task_graph.graph_id,
            status=ExecutionStatus.RUNNING,
            start_time=time.time(),
            metadata={"context": context}
        )
        
        self.logger.info(f"Starting DAG execution {execution_id} with {len(task_graph.nodes)} tasks")
        
        try:
            # Initialize task results
            for node in task_graph.nodes:
                execution.task_results[node.id] = TaskResult(
                    task_id=node.id,
                    status=TaskStatus.PENDING
                )
            
            # Execute tasks in dependency order
            completed_tasks: Set[str] = set()
            
            while len(completed_tasks) < len(task_graph.nodes):
                # Get ready tasks
                ready_tasks = self._get_ready_tasks(task_graph, completed_tasks, execution)
                
                if not ready_tasks:
                    # Check if we have any running tasks
                    running_tasks = [
                        task_id for task_id, result in execution.task_results.items()
                        if result.status in [TaskStatus.RUNNING, TaskStatus.RETRYING]
                    ]
                    
                    if not running_tasks:
                        # No ready tasks and no running tasks - might be stuck
                        failed_tasks = execution.get_failed_tasks()
                        if failed_tasks:
                            execution.status = ExecutionStatus.FAILED
                            execution.error = f"Execution blocked by failed tasks: {failed_tasks}"
                            break
                        else:
                            # This shouldn't happen with a valid DAG
                            execution.status = ExecutionStatus.FAILED
                            execution.error = "No ready tasks found but execution not complete"
                            break
                    
                    # Wait for running tasks
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute ready tasks
                if self.parallel_execution and len(ready_tasks) > 1:
                    await self._execute_tasks_parallel(ready_tasks, task_graph, execution, context)
                else:
                    for task_id in ready_tasks:
                        await self._execute_single_task(task_id, task_graph, execution, context)
                
                # Update completed tasks
                newly_completed = [
                    task_id for task_id, result in execution.task_results.items()
                    if result.status == TaskStatus.COMPLETED and task_id not in completed_tasks
                ]
                completed_tasks.update(newly_completed)
                
                # Check for failures that should stop execution
                failed_tasks = execution.get_failed_tasks()
                if failed_tasks:
                    # Check if these are critical failures (no retries left)
                    critical_failures = [
                        task_id for task_id in failed_tasks
                        if execution.task_results[task_id].attempt > self.max_retries
                    ]
                    if critical_failures:
                        execution.status = ExecutionStatus.FAILED
                        execution.error = f"Critical task failures: {critical_failures}"
                        break
            
            # Finalize execution
            if execution.status == ExecutionStatus.RUNNING:
                if len(completed_tasks) == len(task_graph.nodes):
                    execution.status = ExecutionStatus.COMPLETED
                    self.logger.info(f"DAG execution {execution_id} completed successfully")
                else:
                    execution.status = ExecutionStatus.FAILED
                    execution.error = "Not all tasks completed"
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = f"Execution error: {str(e)}"
            self.logger.error(f"DAG execution {execution_id} failed: {str(e)}")
        
        finally:
            execution.end_time = time.time()
        
        return execution
    
    def _get_ready_tasks(
        self,
        task_graph: TaskGraph,
        completed_tasks: Set[str],
        execution: ExecutionResult
    ) -> List[str]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        
        for node in task_graph.nodes:
            task_result = execution.task_results[node.id]
            
            # Skip if already completed, running, or permanently failed
            if task_result.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING]:
                continue
            
            # Skip if failed too many times
            if task_result.status == TaskStatus.FAILED and task_result.attempt > self.max_retries:
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_tasks for dep in node.deps):
                ready_tasks.append(node.id)
        
        return ready_tasks
    
    async def _execute_tasks_parallel(
        self,
        task_ids: List[str],
        task_graph: TaskGraph,
        execution: ExecutionResult,
        context: Dict[str, Any]
    ):
        """Execute multiple tasks in parallel."""
        tasks = []
        for task_id in task_ids:
            task = asyncio.create_task(
                self._execute_single_task(task_id, task_graph, execution, context)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_task(
        self,
        task_id: str,
        task_graph: TaskGraph,
        execution: ExecutionResult,
        context: Dict[str, Any]
    ):
        """Execute a single task with retry logic."""
        # Find the task node
        task_node = None
        for node in task_graph.nodes:
            if node.id == task_id:
                task_node = node
                break
        
        if not task_node:
            execution.task_results[task_id].status = TaskStatus.FAILED
            execution.task_results[task_id].error = f"Task node {task_id} not found"
            return
        
        task_result = execution.task_results[task_id]
        
        # Build context for this task
        task_context = context.copy()
        task_context["task_id"] = task_id
        task_context["dependencies"] = task_node.deps
        task_context["metadata"] = task_node.metadata
        
        # Add results from completed dependencies
        dependency_results = {}
        for dep_id in task_node.deps:
            if dep_id in execution.task_results:
                dep_result = execution.task_results[dep_id]
                if dep_result.status == TaskStatus.COMPLETED and dep_result.result:
                    dependency_results[dep_id] = dep_result.result
        task_context["dependency_results"] = dependency_results
        
        max_attempts = self.max_retries + 1
        agent_to_use = task_node.agent
        
        for attempt in range(1, max_attempts + 1):
            task_result.attempt = attempt
            task_result.status = TaskStatus.RUNNING if attempt == 1 else TaskStatus.RETRYING
            task_result.start_time = time.time()
            
            self.logger.info(f"Executing task {task_id} with agent {agent_to_use} (attempt {attempt})")
            
            try:
                # Execute the task
                result = await asyncio.wait_for(
                    asyncio.create_task(
                        self._call_task_executor(task_id, task_node.prompt, agent_to_use, task_context)
                    ),
                    timeout=self.execution_timeout
                )
                
                # Success
                task_result.status = TaskStatus.COMPLETED
                task_result.result = result
                task_result.agent_used = agent_to_use
                task_result.end_time = time.time()
                
                self.logger.info(f"Task {task_id} completed successfully")
                return
                
            except asyncio.TimeoutError:
                error_msg = f"Task {task_id} timed out after {self.execution_timeout}s"
                task_result.error = error_msg
                self.logger.warning(error_msg)
                
            except Exception as e:
                error_msg = f"Task {task_id} failed: {str(e)}"
                task_result.error = error_msg
                self.logger.warning(error_msg)
            
            task_result.end_time = time.time()
            
            # If not the last attempt, wait and potentially switch to fallback agent
            if attempt < max_attempts:
                if self.enable_fallback and self.fallback_agent and attempt > 1:
                    agent_to_use = self.fallback_agent
                    self.logger.info(f"Switching to fallback agent {self.fallback_agent} for task {task_id}")
                
                await asyncio.sleep(self.retry_delay)
        
        # All attempts failed
        task_result.status = TaskStatus.FAILED
        self.logger.error(f"Task {task_id} failed after {max_attempts} attempts")
    
    async def _call_task_executor(
        self,
        task_id: str,
        prompt: str,
        agent: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the task executor function, handling both sync and async executors."""
        if asyncio.iscoroutinefunction(self.task_executor):
            return await self.task_executor(task_id, prompt, agent, context)
        else:
            # Run sync executor in thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, self.task_executor, task_id, prompt, agent, context
            )
    
    def execute_graph_sync(
        self,
        task_graph: TaskGraph,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Synchronous wrapper for execute_graph.
        
        Args:
            task_graph: Valid TaskGraph to execute
            context: Optional context to pass to task executor
            
        Returns:
            ExecutionResult with complete execution details
        """
        return asyncio.run(self.execute_graph(task_graph, context))


# Example usage and testing
if __name__ == "__main__":
    import json
    from .dag_planner import DAGPlanner
    
    # Example task executor
    async def example_task_executor(task_id: str, prompt: str, agent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Example task executor that simulates agent execution."""
        await asyncio.sleep(0.1)  # Simulate work
        
        # Use dependency results if available
        dep_context = ""
        if context.get("dependency_results"):
            dep_context = f" (building on: {list(context['dependency_results'].keys())})"
        
        return {
            "content": f"[{agent}] Response to: {prompt}{dep_context}",
            "confidence": 0.85,
            "agent": agent,
            "task_id": task_id
        }
    
    # Example from roadmap
    example_json = """
    {
      "nodes": [
        {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
        {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]},
        {"id": "mitigations", "prompt": "Propose mitigations", "agent": "Apollo", "deps": ["analysis"]},
        {"id": "summary", "prompt": "Integrate plan & mitigations", "agent": "Synthesizer", "deps": ["mitigations"]}
      ]
    }
    """
    
    async def test_scheduler():
        # Create task graph
        planner = DAGPlanner()
        task_graph = planner.create_graph_from_json(example_json)
        
        print(f"Task graph valid: {task_graph.is_valid}")
        if not task_graph.is_valid:
            print(f"Errors: {task_graph.validation_errors}")
            return
        
        # Create scheduler
        scheduler = TaskScheduler(max_retries=2, parallel_execution=True)
        scheduler.set_task_executor(example_task_executor)
        
        # Execute graph
        result = await scheduler.execute_graph(task_graph)
        
        print(f"\nExecution Result:")
        print(f"Status: {result.status}")
        print(f"Total time: {result.total_time:.2f}s")
        print(f"Completed tasks: {len(result.get_completed_tasks())}")
        print(f"Failed tasks: {len(result.get_failed_tasks())}")
        
        for task_id, task_result in result.task_results.items():
            print(f"  {task_id}: {task_result.status} ({task_result.execution_time:.2f}s)")
            if task_result.result:
                print(f"    -> {task_result.result.get('content', '')}")
    
    asyncio.run(test_scheduler())