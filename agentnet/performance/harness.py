"""
Performance Harness for AgentNet

Provides configurable benchmarking framework for measuring turn latency,
token utilization, and multi-agent performance as specified in Phase 5.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import logging

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Types of benchmarks supported by the performance harness."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    MULTI_AGENT = "multi_agent"
    CONCURRENT_AGENTS = "concurrent_agents"
    TOOL_HEAVY = "tool_heavy"
    MEMORY_INTENSIVE = "memory_intensive"


@dataclass
class BenchmarkConfig:
    """Configuration for a performance benchmark."""
    
    name: str
    benchmark_type: BenchmarkType
    iterations: int = 10
    concurrency_level: int = 1
    warmup_iterations: int = 2
    timeout_seconds: float = 30.0
    
    # Agent configuration
    agent_count: int = 1
    turn_count: int = 3
    
    # Test parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Analyze the benefits of distributed systems",
        "Explain machine learning algorithms",
        "Describe cybersecurity best practices"
    ])
    
    # Performance thresholds
    max_turn_latency_ms: float = 5000.0
    max_tokens_per_turn: int = 1000
    min_success_rate: float = 0.95
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark run."""
    
    config: BenchmarkConfig
    start_time: float
    end_time: float
    total_duration: float
    
    # Performance metrics
    avg_turn_latency_ms: float
    min_turn_latency_ms: float
    max_turn_latency_ms: float
    p95_turn_latency_ms: float
    
    # Token metrics
    total_tokens_consumed: int
    avg_tokens_per_turn: float
    token_efficiency_score: float
    
    # Success metrics  
    total_operations: int
    successful_operations: int
    failed_operations: int
    success_rate: float
    
    # Throughput
    operations_per_second: float
    tokens_per_second: float
    
    # Detailed results
    individual_results: List[Dict[str, Any]] = field(default_factory=list)
    error_details: List[str] = field(default_factory=list)
    
    @property
    def passed_thresholds(self) -> bool:
        """Check if benchmark passed all configured thresholds."""
        return (
            self.avg_turn_latency_ms <= self.config.max_turn_latency_ms and
            self.avg_tokens_per_turn <= self.config.max_tokens_per_turn and
            self.success_rate >= self.config.min_success_rate
        )


class PerformanceHarness:
    """
    Configurable performance harness for AgentNet benchmarking.
    
    Supports various benchmark types including single/multi-turn operations,
    multi-agent scenarios, and concurrent execution patterns.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self._results_history: List[BenchmarkResult] = []
        
    async def run_benchmark(
        self, 
        config: BenchmarkConfig,
        agent_factory: Callable[[], Any],
        operation_func: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        Run a performance benchmark with the given configuration.
        
        Args:
            config: Benchmark configuration
            agent_factory: Function that creates agent instances
            operation_func: Optional custom operation function
            
        Returns:
            BenchmarkResult with detailed performance metrics
        """
        logger.info(f"Starting benchmark: {config.name}")
        start_time = time.time()
        
        # Warmup phase
        if config.warmup_iterations > 0:
            logger.info(f"Running {config.warmup_iterations} warmup iterations...")
            await self._run_warmup(config, agent_factory, operation_func)
        
        # Main benchmark phase
        individual_results = []
        errors = []
        
        if config.concurrency_level > 1:
            individual_results, errors = await self._run_concurrent_benchmark(
                config, agent_factory, operation_func
            )
        else:
            individual_results, errors = await self._run_sequential_benchmark(
                config, agent_factory, operation_func
            )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate metrics
        result = self._calculate_benchmark_metrics(
            config, start_time, end_time, total_duration,
            individual_results, errors
        )
        
        self._results_history.append(result)
        logger.info(f"Benchmark completed: {config.name} - Success rate: {result.success_rate:.2%}")
        
        return result
    
    async def _run_warmup(
        self,
        config: BenchmarkConfig,
        agent_factory: Callable,
        operation_func: Optional[Callable]
    ) -> None:
        """Run warmup iterations to stabilize performance measurements."""
        for i in range(config.warmup_iterations):
            try:
                agent = agent_factory()
                if operation_func:
                    await operation_func(agent, config.test_prompts[0])
                else:
                    await self._default_operation(agent, config.test_prompts[0], config)
            except Exception as e:
                logger.debug(f"Warmup iteration {i} failed: {e}")
    
    async def _run_sequential_benchmark(
        self,
        config: BenchmarkConfig, 
        agent_factory: Callable,
        operation_func: Optional[Callable]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Run benchmark iterations sequentially."""
        individual_results = []
        errors = []
        
        for i in range(config.iterations):
            try:
                prompt = config.test_prompts[i % len(config.test_prompts)]
                agent = agent_factory()
                
                iteration_start = time.time()
                
                if operation_func:
                    result = await operation_func(agent, prompt)
                else:
                    result = await self._default_operation(agent, prompt, config)
                
                iteration_duration = time.time() - iteration_start
                
                individual_results.append({
                    'iteration': i,
                    'duration_ms': iteration_duration * 1000,
                    'prompt': prompt,
                    'result': result,
                    'success': True
                })
                
            except Exception as e:
                error_msg = f"Iteration {i} failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                individual_results.append({
                    'iteration': i,
                    'duration_ms': 0.0,
                    'prompt': prompt,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        return individual_results, errors
    
    async def _run_concurrent_benchmark(
        self,
        config: BenchmarkConfig,
        agent_factory: Callable,
        operation_func: Optional[Callable]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Run benchmark iterations concurrently."""
        individual_results = []
        errors = []
        
        # Create tasks for concurrent execution
        tasks = []
        for i in range(config.iterations):
            prompt = config.test_prompts[i % len(config.test_prompts)]
            task = self._run_single_iteration(
                i, prompt, agent_factory, operation_func, config
            )
            tasks.append(task)
        
        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(config.concurrency_level)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[limited_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Iteration {i} failed: {str(result)}"
                errors.append(error_msg)
                individual_results.append({
                    'iteration': i,
                    'duration_ms': 0.0,
                    'success': False,
                    'error': str(result)
                })
            else:
                individual_results.append(result)
        
        return individual_results, errors
    
    async def _run_single_iteration(
        self,
        iteration: int,
        prompt: str,
        agent_factory: Callable,
        operation_func: Optional[Callable],
        config: BenchmarkConfig
    ) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        agent = agent_factory()
        iteration_start = time.time()
        
        if operation_func:
            result = await operation_func(agent, prompt)
        else:
            result = await self._default_operation(agent, prompt, config)
        
        iteration_duration = time.time() - iteration_start
        
        return {
            'iteration': iteration,
            'duration_ms': iteration_duration * 1000,
            'prompt': prompt,
            'result': result,
            'success': True
        }
    
    async def _default_operation(
        self, 
        agent: Any, 
        prompt: str, 
        config: BenchmarkConfig
    ) -> Dict[str, Any]:
        """Default operation for benchmarking - adapts to agent type."""
        
        # Try different agent operation methods
        if hasattr(agent, 'generate_reasoning_tree'):
            return agent.generate_reasoning_tree(prompt)
        elif hasattr(agent, 'process'):
            return await agent.process(prompt)
        elif hasattr(agent, 'infer'):
            return agent.infer(prompt)
        else:
            # Fallback - just return the agent response
            return {'content': f'Processed: {prompt}', 'confidence': 0.8}
    
    def _calculate_benchmark_metrics(
        self,
        config: BenchmarkConfig,
        start_time: float,
        end_time: float, 
        total_duration: float,
        individual_results: List[Dict[str, Any]],
        errors: List[str]
    ) -> BenchmarkResult:
        """Calculate comprehensive benchmark metrics from individual results."""
        
        successful_results = [r for r in individual_results if r.get('success', False)]
        
        # Latency metrics
        latencies = [r['duration_ms'] for r in successful_results]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            # Simple p95 calculation
            sorted_latencies = sorted(latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0.0
        else:
            avg_latency = min_latency = max_latency = p95_latency = 0.0
        
        # Token metrics (estimated - would need actual token counting)
        total_tokens = 0
        for result in successful_results:
            # Estimate tokens from result content if available
            result_data = result.get('result', {})
            if isinstance(result_data, dict):
                content = result_data.get('content', '')
                # Rough token estimation: ~4 chars per token
                estimated_tokens = len(str(content)) // 4
                total_tokens += estimated_tokens
        
        avg_tokens_per_turn = total_tokens / len(successful_results) if successful_results else 0.0
        
        # Success metrics
        total_ops = len(individual_results)
        successful_ops = len(successful_results)
        failed_ops = total_ops - successful_ops
        success_rate = successful_ops / total_ops if total_ops > 0 else 0.0
        
        # Throughput
        ops_per_second = total_ops / total_duration if total_duration > 0 else 0.0
        tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0.0
        
        # Token efficiency (higher is better - more content per token)
        token_efficiency = 1.0 / avg_tokens_per_turn if avg_tokens_per_turn > 0 else 0.0
        
        return BenchmarkResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            avg_turn_latency_ms=avg_latency,
            min_turn_latency_ms=min_latency,
            max_turn_latency_ms=max_latency,
            p95_turn_latency_ms=p95_latency,
            total_tokens_consumed=total_tokens,
            avg_tokens_per_turn=avg_tokens_per_turn,
            token_efficiency_score=token_efficiency,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            success_rate=success_rate,
            operations_per_second=ops_per_second,
            tokens_per_second=tokens_per_second,
            individual_results=individual_results,
            error_details=errors
        )
    
    def get_results_history(self) -> List[BenchmarkResult]:
        """Get all benchmark results from this session."""
        return self._results_history.copy()
    
    def get_latest_result(self) -> Optional[BenchmarkResult]:
        """Get the most recent benchmark result."""
        return self._results_history[-1] if self._results_history else None