"""
Production-Ready Performance Harness for AgentNet

Provides a statistically robust, async-native benchmarking framework that
integrates directly with the observability stack (latency and token trackers)
to produce accurate and actionable performance reports.
"""

import asyncio
import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, AsyncGenerator

from ..observability.latency import get_latency_tracker, LatencyComponent
from ..observability.tokens import get_token_tracker

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Types of benchmarks supported by the performance harness."""
    SINGLE_TURN = "single_turn"
    CONCURRENT_AGENTS = "concurrent_agents"


@dataclass
class BenchmarkConfig:
    """Configuration for a performance benchmark run."""
    name: str
    benchmark_type: BenchmarkType
    
    # Execution control
    min_rounds: int = 5
    min_time_per_round_s: float = 1.0
    warmup_rounds: int = 1
    concurrency_level: int = 1
    
    # Test data
    test_prompts: List[str] = field(default_factory=lambda: ["Analyze the benefits of distributed systems."])
    
    # Performance thresholds for pass/fail
    max_p95_latency_ms: float = 5000.0
    max_avg_cost_usd: float = 0.05
    min_success_rate: float = 0.98


@dataclass
class BenchmarkResult:
    """Detailed results from a performance benchmark run."""
    config: BenchmarkConfig
    total_duration_s: float
    
    # Success metrics
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    
    # Latency statistics (from LatencyTracker)
    latency_stats: Dict[str, float]
    
    # Cost & Token statistics (from TokenUtilizationTracker)
    token_stats: Dict[str, Any]
    
    # Detailed breakdowns
    component_latency_breakdown: Dict[str, Dict[str, float]]
    error_details: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful_iterations / self.total_iterations if self.total_iterations > 0 else 0.0

    @property
    def passed(self) -> bool:
        """Check if the benchmark passed all configured thresholds."""
        latency_ok = self.latency_stats.get("p95_ms", 0) <= self.config.max_p95_latency_ms
        cost_ok = self.token_stats.get("avg_cost_per_turn_usd", 0) <= self.config.max_avg_cost_usd
        success_ok = self.success_rate >= self.config.min_success_rate
        return latency_ok and cost_ok and success_ok


# Type alias for the agent setup/teardown fixture
AgentFixture = Callable[[], AsyncGenerator[Any, None]]

class PerformanceHarness:
    """
    A statistically robust performance harness for AgentNet that integrates
    with the observability stack for precise measurements.
    """
    def __init__(self):
        self._results_history: List[BenchmarkResult] = []
        self.latency_tracker = get_latency_tracker()
        self.token_tracker = get_token_tracker()

    async def run_benchmark(
        self,
        config: BenchmarkConfig,
        agent_fixture: AgentFixture,
        operation_func: Callable[[Any, str], Awaitable[Any]],
    ) -> BenchmarkResult:
        """
        Run a performance benchmark using the specified configuration.

        Args:
            config: The benchmark configuration.
            agent_fixture: An async generator function that yields an agent instance.
            operation_func: The async function to benchmark, which takes an agent and a prompt.

        Returns:
            A BenchmarkResult with detailed, accurate performance metrics.
        """
        logger.info(f"--- Starting Benchmark: {config.name} ---")
        
        # Clear trackers to isolate this benchmark run
        self.latency_tracker.clear_measurements()
        self.token_tracker.clear_metrics()

        start_time = time.monotonic()
        
        # --- Warmup Phase ---
        if config.warmup_rounds > 0:
            logger.info(f"Running {config.warmup_rounds} warmup round(s)...")
            await self._execute_round(config.warmup_rounds, config, agent_fixture, operation_func)
            # Clear trackers again after warmup
            self.latency_tracker.clear_measurements()
            self.token_tracker.clear_metrics()

        # --- Main Benchmark Phase ---
        logger.info(f"Running main benchmark for at least {config.min_rounds} rounds...")
        all_errors = await self._execute_round(config.min_rounds, config, agent_fixture, operation_func)

        total_duration_s = time.monotonic() - start_time
        
        # --- Analysis Phase ---
        logger.info("Benchmark execution complete. Analyzing results...")
        result = self._analyze_results(config, total_duration_s, all_errors)
        self._results_history.append(result)

        logger.info(f"--- Benchmark Complete: {config.name} ---")
        logger.info(f"  - Passed: {result.passed}")
        logger.info(f"  - Success Rate: {result.success_rate:.2%}")
        logger.info(f"  - P95 Latency: {result.latency_stats.get('p95_ms', 0):.2f} ms")
        logger.info(f"  - Avg Cost/Turn: ${result.token_stats.get('avg_cost_per_turn_usd', 0):.6f}")
        
        return result

    async def _execute_round(
        self, num_rounds: int, config: BenchmarkConfig, agent_fixture: AgentFixture, operation_func: Callable
    ) -> List[str]:
        """Execute a number of benchmark rounds."""
        all_errors = []
        for i in range(num_rounds):
            iterations_in_round = self._determine_iterations_for_round(i)
            logger.debug(f"Round {i+1}/{num_rounds}, running {iterations_in_round} iterations...")
            
            tasks = [
                self._run_single_iteration(
                    f"round{i}-iter{j}",
                    config.test_prompts[j % len(config.test_prompts)],
                    agent_fixture,
                    operation_func,
                )
                for j in range(iterations_in_round)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                if isinstance(res, Exception):
                    all_errors.append(str(res))
        return all_errors

    def _determine_iterations_for_round(self, round_index: int) -> int:
        """Simple logic to increase iterations in later rounds for stability."""
        # A more advanced version could adapt based on timing of previous rounds.
        return 1 * (2 ** min(round_index, 3))

    async def _run_single_iteration(
        self, turn_id: str, prompt: str, agent_fixture: AgentFixture, operation_func: Callable
    ):
        """Run and measure a single, isolated iteration."""
        async with self.latency_tracker.measure(turn_id, LatencyComponent.ORCHESTRATION):
            async for agent in agent_fixture():
                try:
                    self.latency_tracker.start_turn_measurement(turn_id, agent.name)
                    await operation_func(agent, prompt)
                finally:
                    self.latency_tracker.end_turn_measurement(turn_id)

    def _analyze_results(
        self, config: BenchmarkConfig, total_duration_s: float, errors: List[str]
    ) -> BenchmarkResult:
        """Analyze data from trackers to produce the final result."""
        measurements = self.latency_tracker.get_measurements()
        successful_iterations = len(measurements)
        failed_iterations = len(errors)
        total_iterations = successful_iterations + failed_iterations

        token_summary = self.token_tracker.get_system_overview()

        return BenchmarkResult(
            config=config,
            total_duration_s=total_duration_s,
            total_iterations=total_iterations,
            successful_iterations=successful_iterations,
            failed_iterations=failed_iterations,
            latency_stats=self.latency_tracker.get_latency_statistics(),
            token_stats=token_summary.get("overview", {}),
            component_latency_breakdown=self.latency_tracker.get_component_breakdown(),
            error_details=errors,
        )

    def get_latest_result(self) -> Optional[BenchmarkResult]:
        """Get the most recent benchmark result."""
        return self._results_history[-1] if self._results_history else None
