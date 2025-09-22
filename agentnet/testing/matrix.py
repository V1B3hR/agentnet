"""
Test Matrix Framework for AgentNet

Provides systematic testing across different agent configurations,
feature combinations, and scenarios for comprehensive validation.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import logging
import itertools

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of a test execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class FeatureSet(str, Enum):
    """AgentNet feature sets to test."""
    PHASE_0 = "phase_0"  # Core functionality
    PHASE_1 = "phase_1"  # Turn engine, policy, events
    PHASE_2 = "phase_2"  # Memory, tools, critique
    PHASE_3 = "phase_3"  # DAG, eval, persistence
    PHASE_4 = "phase_4"  # Governance, RBAC, cost
    PHASE_5 = "phase_5"  # Observability, performance
    PHASE_6 = "phase_6"  # Enterprise features


class AgentType(str, Enum):
    """Types of agents to test."""
    SINGLE = "single"
    DEBATE_PAIR = "debate_pair"
    MULTI_PARTY = "multi_party"
    HIERARCHICAL = "hierarchical"
    ASYNC = "async"


@dataclass
class TestConfiguration:
    """Configuration for a single test case."""
    
    name: str
    feature_set: FeatureSet
    agent_type: AgentType
    
    # Agent configuration
    agent_count: int = 1
    agent_styles: List[Dict[str, float]] = field(default_factory=lambda: [
        {"logic": 0.8, "creativity": 0.6, "analytical": 0.7}
    ])
    
    # Test parameters
    test_scenario: str = "basic_reasoning"
    test_prompts: List[str] = field(default_factory=lambda: [
        "Analyze the benefits of microservices architecture"
    ])
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Feature toggles
    enable_memory: bool = False
    enable_tools: bool = False
    enable_policies: bool = False
    enable_monitoring: bool = False
    
    # Performance expectations
    max_latency_ms: float = 5000.0
    min_success_rate: float = 0.95
    max_cost_usd: float = 0.10
    
    # Test metadata
    priority: str = "medium"  # low, medium, high, critical
    tags: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0


@dataclass
class TestResult:
    """Result of a test execution."""
    
    configuration: TestConfiguration
    status: TestStatus
    
    # Execution metrics
    start_time: float
    end_time: float
    duration_seconds: float
    
    # Test outcomes
    success: bool
    error_message: Optional[str] = None
    exception: Optional[Exception] = None
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Test-specific results
    agent_results: List[Dict[str, Any]] = field(default_factory=list)
    assertions_passed: int = 0
    assertions_failed: int = 0
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed_performance_requirements(self) -> bool:
        """Check if test passed performance requirements."""
        return (
            self.avg_latency_ms <= self.configuration.max_latency_ms and
            self.total_cost_usd <= self.configuration.max_cost_usd
        )


class TestMatrix:
    """
    Comprehensive test matrix for systematic AgentNet testing.
    
    Generates and executes test configurations across different
    combinations of features, agent types, and scenarios.
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        self._configurations: List[TestConfiguration] = []
        self._results: List[TestResult] = []
        self._test_functions: Dict[str, Callable] = {}
    
    def add_configuration(self, config: TestConfiguration) -> None:
        """Add a test configuration to the matrix."""
        self._configurations.append(config)
        logger.debug(f"Added test configuration: {config.name}")
    
    def add_configurations(self, configs: List[TestConfiguration]) -> None:
        """Add multiple test configurations."""
        for config in configs:
            self.add_configuration(config)
    
    def register_test_function(self, scenario: str, test_func: Callable) -> None:
        """Register a test function for a specific scenario."""
        self._test_functions[scenario] = test_func
        logger.debug(f"Registered test function for scenario: {scenario}")
    
    def generate_standard_matrix(self) -> List[TestConfiguration]:
        """Generate standard test matrix covering common scenarios."""
        
        configurations = []
        
        # Phase 0 - Core functionality tests
        for agent_type in [AgentType.SINGLE]:
            config = TestConfiguration(
                name=f"P0_Core_{agent_type.value}",
                feature_set=FeatureSet.PHASE_0,
                agent_type=agent_type,
                test_scenario="basic_reasoning",
                priority="critical",
                tags=["phase_0", "core", "critical"]
            )
            configurations.append(config)
        
        # Phase 1 - Turn engine and policy tests
        for agent_type in [AgentType.SINGLE, AgentType.DEBATE_PAIR, AgentType.MULTI_PARTY]:
            config = TestConfiguration(
                name=f"P1_TurnEngine_{agent_type.value}",
                feature_set=FeatureSet.PHASE_1,
                agent_type=agent_type,
                agent_count=1 if agent_type == AgentType.SINGLE else (2 if agent_type == AgentType.DEBATE_PAIR else 3),
                test_scenario="multi_agent_dialogue",
                enable_policies=True,
                priority="high",
                tags=["phase_1", "turn_engine", "policy"]
            )
            configurations.append(config)
        
        # Phase 2 - Memory and tools tests
        for memory_enabled, tools_enabled in [(True, False), (False, True), (True, True)]:
            config = TestConfiguration(
                name=f"P2_Memory{memory_enabled}_Tools{tools_enabled}",
                feature_set=FeatureSet.PHASE_2,
                agent_type=AgentType.SINGLE,
                test_scenario="memory_tool_integration",
                enable_memory=memory_enabled,
                enable_tools=tools_enabled,
                priority="high",
                tags=["phase_2", "memory", "tools"]
            )
            configurations.append(config)
        
        # Phase 3 - Advanced features
        config = TestConfiguration(
            name="P3_DAG_Evaluation",
            feature_set=FeatureSet.PHASE_3,
            agent_type=AgentType.SINGLE,
            test_scenario="dag_evaluation",
            enable_memory=True,
            enable_tools=True,
            priority="medium",
            tags=["phase_3", "dag", "evaluation"]
        )
        configurations.append(config)
        
        # Phase 4 - Governance tests
        config = TestConfiguration(
            name="P4_Governance_RBAC",
            feature_set=FeatureSet.PHASE_4,
            agent_type=AgentType.SINGLE,
            test_scenario="governance_rbac",
            enable_policies=True,
            enable_monitoring=True,
            priority="medium",
            tags=["phase_4", "governance", "rbac"]
        )
        configurations.append(config)
        
        # Phase 5 - Observability tests
        config = TestConfiguration(
            name="P5_Observability_Performance",
            feature_set=FeatureSet.PHASE_5,
            agent_type=AgentType.SINGLE,
            test_scenario="performance_monitoring",
            enable_monitoring=True,
            priority="high",
            tags=["phase_5", "observability", "performance"]
        )
        configurations.append(config)
        
        # Integration tests combining multiple phases
        config = TestConfiguration(
            name="Integration_Full_Stack",
            feature_set=FeatureSet.PHASE_5,  # Latest available
            agent_type=AgentType.MULTI_PARTY,
            agent_count=3,
            test_scenario="full_stack_integration",
            enable_memory=True,
            enable_tools=True,
            enable_policies=True,
            enable_monitoring=True,
            priority="critical",
            tags=["integration", "full_stack", "critical"],
            timeout_seconds=60.0
        )
        configurations.append(config)
        
        return configurations
    
    def generate_performance_matrix(self) -> List[TestConfiguration]:
        """Generate performance-focused test matrix."""
        
        configurations = []
        
        # Latency stress tests
        for agent_count in [1, 2, 5, 10]:
            config = TestConfiguration(
                name=f"Performance_Latency_{agent_count}_agents",
                feature_set=FeatureSet.PHASE_5,
                agent_type=AgentType.MULTI_PARTY if agent_count > 1 else AgentType.SINGLE,
                agent_count=agent_count,
                test_scenario="latency_stress_test",
                max_latency_ms=2000.0,  # Stricter requirement
                enable_monitoring=True,
                priority="high",
                tags=["performance", "latency", "stress"]
            )
            configurations.append(config)
        
        # Memory usage tests
        for memory_intensive in [True, False]:
            config = TestConfiguration(
                name=f"Performance_Memory_{'intensive' if memory_intensive else 'light'}",
                feature_set=FeatureSet.PHASE_2,
                agent_type=AgentType.SINGLE,
                test_scenario="memory_usage_test",
                enable_memory=memory_intensive,
                enable_tools=memory_intensive,
                priority="medium",
                tags=["performance", "memory"]
            )
            configurations.append(config)
        
        return configurations
    
    async def execute_matrix(
        self,
        configurations: Optional[List[TestConfiguration]] = None,
        parallel_execution: bool = False,
        max_concurrent: int = 4
    ) -> List[TestResult]:
        """Execute test matrix with given configurations."""
        
        if configurations is None:
            configurations = self._configurations
        
        if not configurations:
            logger.warning("No test configurations to execute")
            return []
        
        logger.info(f"Executing test matrix with {len(configurations)} configurations")
        
        if parallel_execution:
            results = await self._execute_parallel(configurations, max_concurrent)
        else:
            results = await self._execute_sequential(configurations)
        
        self._results.extend(results)
        return results
    
    async def _execute_sequential(
        self, 
        configurations: List[TestConfiguration]
    ) -> List[TestResult]:
        """Execute test configurations sequentially."""
        
        results = []
        for i, config in enumerate(configurations, 1):
            logger.info(f"Executing test {i}/{len(configurations)}: {config.name}")
            
            result = await self._execute_single_test(config)
            results.append(result)
            
            # Log immediate result
            status_emoji = "✅" if result.status == TestStatus.PASSED else "❌"
            logger.info(
                f"{status_emoji} {config.name}: {result.status.value} "
                f"({result.duration_seconds:.2f}s)"
            )
        
        return results
    
    async def _execute_parallel(
        self,
        configurations: List[TestConfiguration],
        max_concurrent: int
    ) -> List[TestResult]:
        """Execute test configurations in parallel with concurrency limit."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_execute(config):
            async with semaphore:
                return await self._execute_single_test(config)
        
        tasks = [limited_execute(config) for config in configurations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                config = configurations[i]
                error_result = TestResult(
                    configuration=config,
                    status=TestStatus.ERROR,
                    start_time=time.time(),
                    end_time=time.time(),
                    duration_seconds=0.0,
                    success=False,
                    error_message=str(result),
                    exception=result
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_single_test(self, config: TestConfiguration) -> TestResult:
        """Execute a single test configuration."""
        
        start_time = time.time()
        
        try:
            # Get test function for scenario
            test_func = self._test_functions.get(config.test_scenario)
            if not test_func:
                # Use default test function
                test_func = self._default_test_function
            
            # Execute test with timeout
            result_data = await asyncio.wait_for(
                test_func(config),
                timeout=config.timeout_seconds
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Create result
            result = TestResult(
                configuration=config,
                status=TestStatus.PASSED if result_data.get('success', False) else TestStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=result_data.get('success', False),
                error_message=result_data.get('error_message'),
                avg_latency_ms=result_data.get('avg_latency_ms', 0.0),
                total_cost_usd=result_data.get('total_cost_usd', 0.0),
                agent_results=result_data.get('agent_results', []),
                assertions_passed=result_data.get('assertions_passed', 0),
                assertions_failed=result_data.get('assertions_failed', 0),
                metadata=result_data.get('metadata', {})
            )
            
            return result
            
        except asyncio.TimeoutError:
            end_time = time.time()
            return TestResult(
                configuration=config,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=config.timeout_seconds,
                success=False,
                error_message=f"Test timed out after {config.timeout_seconds}s"
            )
            
        except Exception as e:
            end_time = time.time()
            return TestResult(
                configuration=config,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_time - start_time,
                success=False,
                error_message=str(e),
                exception=e
            )
    
    async def _default_test_function(self, config: TestConfiguration) -> Dict[str, Any]:
        """Default test function when no specific test is registered."""
        
        # Import here to avoid circular imports
        from ..core.agent import AgentNet
        from ..providers.example import ExampleEngine
        
        # Create agents based on configuration
        agents = []
        engine = ExampleEngine()
        
        for i in range(config.agent_count):
            style = config.agent_styles[i % len(config.agent_styles)]
            agent = AgentNet(
                name=f"TestAgent_{i}",
                style=style,
                engine=engine
            )
            agents.append(agent)
        
        # Execute basic test scenario
        results = []
        total_latency = 0.0
        
        for prompt in config.test_prompts:
            if config.agent_type == AgentType.SINGLE:
                # Single agent test
                start = time.time()
                result = agents[0].generate_reasoning_tree(prompt)
                latency = (time.time() - start) * 1000
                
                results.append({
                    'agent_id': agents[0].name, 
                    'result': result,
                    'latency_ms': latency
                })
                total_latency += latency
                
            elif config.agent_type in [AgentType.DEBATE_PAIR, AgentType.MULTI_PARTY]:
                # Multi-agent test
                if len(agents) >= 2:
                    start = time.time()
                    dialogue_result = agents[0].dialogue_with(
                        agents[1], 
                        prompt, 
                        rounds=2
                    )
                    latency = (time.time() - start) * 1000
                    
                    results.append({
                        'type': 'dialogue',
                        'result': dialogue_result,
                        'latency_ms': latency
                    })
                    total_latency += latency
        
        # Calculate success based on basic criteria
        success = (
            len(results) > 0 and
            all(r.get('result') is not None for r in results) and
            (total_latency / len(results) if results else 0) <= config.max_latency_ms
        )
        
        return {
            'success': success,
            'agent_results': results,
            'avg_latency_ms': total_latency / len(results) if results else 0.0,
            'total_cost_usd': 0.001 * len(results),  # Estimated
            'assertions_passed': len(results) if success else 0,
            'assertions_failed': 0 if success else len(results)
        }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        
        if not self._results:
            return {}
        
        total_tests = len(self._results)
        passed = len([r for r in self._results if r.status == TestStatus.PASSED])
        failed = len([r for r in self._results if r.status == TestStatus.FAILED])
        errors = len([r for r in self._results if r.status == TestStatus.ERROR])
        
        # Performance metrics
        avg_duration = sum(r.duration_seconds for r in self._results) / total_tests
        avg_latency = sum(r.avg_latency_ms for r in self._results) / total_tests
        
        # By feature set
        by_feature = {}
        for result in self._results:
            feature = result.configuration.feature_set.value
            if feature not in by_feature:
                by_feature[feature] = {'passed': 0, 'failed': 0, 'error': 0}
            
            if result.status == TestStatus.PASSED:
                by_feature[feature]['passed'] += 1
            elif result.status == TestStatus.FAILED:
                by_feature[feature]['failed'] += 1
            else:
                by_feature[feature]['error'] += 1
        
        return {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'success_rate': passed / total_tests if total_tests > 0 else 0.0,
            'avg_duration_seconds': avg_duration,
            'avg_latency_ms': avg_latency,
            'by_feature_set': by_feature,
            'failed_tests': [
                {'name': r.configuration.name, 'error': r.error_message}
                for r in self._results
                if r.status in [TestStatus.FAILED, TestStatus.ERROR]
            ]
        }
    
    def get_all_results(self) -> List[TestResult]:
        """Get all test results."""
        return self._results.copy()
    
    def clear_results(self) -> None:
        """Clear all test results."""
        self._results.clear()
        logger.info("Cleared all test results")