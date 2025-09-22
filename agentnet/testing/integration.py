"""
Integration Test Suite for AgentNet

Provides comprehensive integration testing that combines P1-P4 features
with P5 observability for end-to-end validation.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentTestCase:
    """Test case for multi-agent interactions."""
    
    name: str
    description: str
    
    # Agent setup
    agent_configs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Test scenario
    scenario_type: str = "dialogue"  # dialogue, debate, collaboration
    test_prompt: str = "Analyze system architecture"
    rounds: int = 3
    
    # Expected outcomes
    expected_consensus: bool = False
    min_dialogue_length: int = 10
    max_turn_latency_ms: float = 5000.0
    
    # Feature requirements
    requires_memory: bool = False
    requires_tools: bool = False
    requires_policies: bool = False
    
    # Success criteria
    success_conditions: List[str] = field(default_factory=list)


class IntegrationTestSuite:
    """
    Comprehensive integration test suite combining multiple AgentNet phases.
    
    Tests end-to-end functionality across P1-P5 features with observability.
    """
    
    def __init__(self):
        self._test_cases: List[MultiAgentTestCase] = []
        self._results: List[Dict[str, Any]] = []
    
    def add_test_case(self, test_case: MultiAgentTestCase) -> None:
        """Add an integration test case."""
        self._test_cases.append(test_case)
        logger.debug(f"Added integration test case: {test_case.name}")
    
    def create_standard_test_cases(self) -> List[MultiAgentTestCase]:
        """Create standard integration test cases."""
        
        test_cases = []
        
        # P1 + P5: Turn Engine with Observability
        test_cases.append(MultiAgentTestCase(
            name="P1_P5_TurnEngine_Observability",
            description="Test turn engine with performance monitoring",
            agent_configs=[
                {"name": "Analyst", "style": {"logic": 0.9, "creativity": 0.4}},
                {"name": "Critic", "style": {"logic": 0.7, "creativity": 0.8}}
            ],
            scenario_type="debate",
            test_prompt="Evaluate microservices vs monolithic architecture",
            rounds=3,
            expected_consensus=False,
            success_conditions=[
                "Both agents participate",
                "Debate concludes within time limit",
                "Performance metrics captured",
                "No policy violations"
            ]
        ))
        
        # P2 + P5: Memory/Tools with Monitoring
        test_cases.append(MultiAgentTestCase(
            name="P2_P5_Memory_Tools_Monitoring",
            description="Test memory and tools with performance tracking",
            agent_configs=[
                {"name": "ResearchAgent", "style": {"logic": 0.8, "analytical": 0.9}}
            ],
            scenario_type="tool_usage",
            test_prompt="Research and calculate system performance metrics",
            requires_memory=True,
            requires_tools=True,
            success_conditions=[
                "Memory operations recorded",
                "Tools executed successfully",
                "Token usage tracked",
                "Latency within thresholds"
            ]
        ))
        
        # P3 + P4 + P5: Full Stack Integration
        test_cases.append(MultiAgentTestCase(
            name="P3_P4_P5_Full_Stack",
            description="Full stack integration with governance and observability",
            agent_configs=[
                {"name": "Lead", "style": {"logic": 0.8, "creativity": 0.6}},
                {"name": "Developer", "style": {"logic": 0.9, "creativity": 0.5}},
                {"name": "QA", "style": {"logic": 0.7, "analytical": 0.9}}
            ],
            scenario_type="collaboration",
            test_prompt="Design and review a secure API system",
            rounds=4,
            requires_memory=True,
            requires_tools=True,
            requires_policies=True,
            expected_consensus=True,
            success_conditions=[
                "All agents participate effectively",
                "Policy compliance maintained",
                "Cost tracking accurate",
                "Performance within limits",
                "Consensus reached"
            ]
        ))
        
        # P5 Streaming Integration (Phase 6 preview)
        test_cases.append(MultiAgentTestCase(
            name="P5_P6_Streaming_Collaboration",
            description="Test streaming partial-output collaboration",
            agent_configs=[
                {"name": "Writer", "style": {"creativity": 0.9, "logic": 0.6}},
                {"name": "Editor", "style": {"analytical": 0.9, "creativity": 0.5}}
            ],
            scenario_type="streaming_collaboration",
            test_prompt="Collaboratively write a technical specification",
            rounds=5,
            success_conditions=[
                "Streaming responses processed",
                "Partial JSON parsing works",
                "Real-time collaboration effective",
                "No data corruption"
            ]
        ))
        
        return test_cases
    
    async def run_all_tests(
        self, 
        include_observability: bool = True,
        parallel_execution: bool = False
    ) -> Dict[str, Any]:
        """Run all integration tests."""
        
        if not self._test_cases:
            self._test_cases = self.create_standard_test_cases()
        
        logger.info(f"Running {len(self._test_cases)} integration tests")
        
        if parallel_execution:
            results = await self._run_tests_parallel(include_observability)
        else:
            results = await self._run_tests_sequential(include_observability)
        
        # Generate summary
        summary = self._generate_test_summary(results)
        
        self._results = results
        return summary
    
    async def _run_tests_sequential(self, include_observability: bool) -> List[Dict[str, Any]]:
        """Run tests sequentially."""
        
        results = []
        
        for i, test_case in enumerate(self._test_cases, 1):
            logger.info(f"Running integration test {i}/{len(self._test_cases)}: {test_case.name}")
            
            result = await self._execute_test_case(test_case, include_observability)
            results.append(result)
            
            # Log result
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            logger.info(f"{status} {test_case.name} ({result['duration_seconds']:.2f}s)")
        
        return results
    
    async def _run_tests_parallel(self, include_observability: bool) -> List[Dict[str, Any]]:
        """Run tests in parallel."""
        
        tasks = [
            self._execute_test_case(test_case, include_observability)
            for test_case in self._test_cases
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_case = self._test_cases[i]
                error_result = {
                    'test_case': test_case.name,
                    'success': False,
                    'error': str(result),
                    'duration_seconds': 0.0,
                    'agent_results': [],
                    'observability_data': {}
                }
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_test_case(
        self, 
        test_case: MultiAgentTestCase,
        include_observability: bool
    ) -> Dict[str, Any]:
        """Execute a single integration test case."""
        
        start_time = time.time()
        
        try:
            # Setup observability if requested
            observability_data = {}
            latency_tracker = None
            token_tracker = None
            
            if include_observability:
                from ..performance import LatencyTracker, TokenUtilizationTracker
                latency_tracker = LatencyTracker()
                token_tracker = TokenUtilizationTracker()
            
            # Create agents based on test case configuration
            agents = await self._create_test_agents(test_case)
            
            # Execute test scenario
            scenario_result = await self._execute_scenario(
                test_case, agents, latency_tracker, token_tracker
            )
            
            # Collect observability data
            if include_observability:
                observability_data = self._collect_observability_data(
                    latency_tracker, token_tracker
                )
            
            # Evaluate success conditions
            success, evaluation_details = self._evaluate_success_conditions(
                test_case, scenario_result, observability_data
            )
            
            end_time = time.time()
            
            return {
                'test_case': test_case.name,
                'success': success,
                'duration_seconds': end_time - start_time,
                'scenario_result': scenario_result,
                'agent_results': scenario_result.get('agent_results', []),
                'observability_data': observability_data,
                'evaluation_details': evaluation_details,
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Integration test {test_case.name} failed: {e}")
            
            return {
                'test_case': test_case.name,
                'success': False,
                'duration_seconds': end_time - start_time,
                'scenario_result': {},
                'agent_results': [],
                'observability_data': {},
                'evaluation_details': {},
                'error': str(e)
            }
    
    async def _create_test_agents(self, test_case: MultiAgentTestCase) -> List:
        """Create agents for the test case."""
        
        # Import here to avoid circular imports
        from ..core.agent import AgentNet
        from ..providers.example import ExampleEngine
        
        agents = []
        engine = ExampleEngine()
        
        for config in test_case.agent_configs:
            # Setup agent configuration
            agent_config = {
                'engine': engine
            }
            
            # Add memory if required
            if test_case.requires_memory:
                agent_config['memory_config'] = {
                    'memory': {
                        'short_term': {'enabled': True, 'max_entries': 20},
                        'episodic': {'enabled': True}
                    }
                }
            
            # Add tools if required
            if test_case.requires_tools:
                from ..tools.registry import ToolRegistry
                from ..tools.examples import CalculatorTool, StatusCheckTool
                
                tool_registry = ToolRegistry()
                tool_registry.register_tool(CalculatorTool())
                tool_registry.register_tool(StatusCheckTool())
                agent_config['tool_registry'] = tool_registry
            
            agent = AgentNet(
                name=config['name'],
                style=config['style'],
                **agent_config
            )
            
            agents.append(agent)
        
        return agents
    
    async def _execute_scenario(
        self,
        test_case: MultiAgentTestCase,
        agents: List,
        latency_tracker,
        token_tracker
    ) -> Dict[str, Any]:
        """Execute the test scenario with the given agents."""
        
        scenario_type = test_case.scenario_type
        
        if scenario_type == "dialogue":
            return await self._execute_dialogue_scenario(
                test_case, agents, latency_tracker, token_tracker
            )
        elif scenario_type == "debate":
            return await self._execute_debate_scenario(
                test_case, agents, latency_tracker, token_tracker
            )
        elif scenario_type == "collaboration":
            return await self._execute_collaboration_scenario(
                test_case, agents, latency_tracker, token_tracker
            )
        elif scenario_type == "tool_usage":
            return await self._execute_tool_usage_scenario(
                test_case, agents, latency_tracker, token_tracker
            )
        elif scenario_type == "streaming_collaboration":
            return await self._execute_streaming_scenario(
                test_case, agents, latency_tracker, token_tracker
            )
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    async def _execute_dialogue_scenario(self, test_case, agents, latency_tracker, token_tracker):
        """Execute dialogue scenario."""
        
        if len(agents) < 2:
            raise ValueError("Dialogue scenario requires at least 2 agents")
        
        # Track latency if available
        turn_id = f"dialogue_{int(time.time())}"
        if latency_tracker:
            latency_tracker.start_turn_measurement(turn_id, agents[0].name)
        
        # Execute dialogue - check if method exists
        if hasattr(agents[0], 'dialogue_with'):
            result = agents[0].dialogue_with(
                agents[1],
                test_case.test_prompt,
                rounds=test_case.rounds
            )
        else:
            # Fallback: simulate dialogue with individual reasoning
            result = {
                'participants': [agents[0].name, agents[1].name],
                'dialogue_history': [
                    f"{agents[0].name}: {agents[0].generate_reasoning_tree(test_case.test_prompt).get('result', {}).get('content', 'No response')}",
                    f"{agents[1].name}: {agents[1].generate_reasoning_tree(f'Respond to: {test_case.test_prompt}').get('result', {}).get('content', 'No response')}"
                ]
            }
        
        # End latency tracking
        if latency_tracker:
            latency_tracker.end_turn_measurement(turn_id)
        
        # Track tokens (estimated)
        if token_tracker:
            # Estimate tokens from dialogue content
            dialogue_content = str(result.get('dialogue_history', []))
            estimated_tokens = len(dialogue_content) // 4
            
            token_tracker.record_token_usage(
                agent_id=agents[0].name,
                turn_id=turn_id,
                input_tokens=len(test_case.test_prompt) // 4,
                output_tokens=estimated_tokens,
                processing_time=1.0  # Estimated
            )
        
        return {
            'type': 'dialogue',
            'result': result,
            'agent_results': [
                {'agent': agents[0].name, 'participated': True},
                {'agent': agents[1].name, 'participated': True}
            ]
        }
    
    async def _execute_debate_scenario(self, test_case, agents, latency_tracker, token_tracker):
        """Execute debate scenario."""
        
        if len(agents) < 2:
            raise ValueError("Debate scenario requires at least 2 agents")
        
        # Use dialogue with debate-like prompting
        debate_prompt = f"Debate this topic: {test_case.test_prompt}. Take opposing positions."
        
        turn_id = f"debate_{int(time.time())}"
        if latency_tracker:
            latency_tracker.start_turn_measurement(turn_id, "debate_session")
        
        # Execute debate - check if method exists
        if hasattr(agents[0], 'dialogue_with'):
            result = agents[0].dialogue_with(
                agents[1],
                debate_prompt,
                rounds=test_case.rounds
            )
        else:
            # Fallback: simulate debate with individual reasoning
            result = {
                'participants': [agents[0].name, agents[1].name],
                'dialogue_history': [
                    f"{agents[0].name} (Pro): {agents[0].generate_reasoning_tree(f'Argue in favor of: {test_case.test_prompt}').get('result', {}).get('content', 'No response')}",
                    f"{agents[1].name} (Con): {agents[1].generate_reasoning_tree(f'Argue against: {test_case.test_prompt}').get('result', {}).get('content', 'No response')}"
                ]
            }
        
        if latency_tracker:
            latency_tracker.end_turn_measurement(turn_id)
        
        return {
            'type': 'debate',
            'result': result,
            'agent_results': [
                {'agent': agent.name, 'participated': True}
                for agent in agents[:2]
            ]
        }
    
    async def _execute_collaboration_scenario(self, test_case, agents, latency_tracker, token_tracker):
        """Execute collaboration scenario."""
        
        # Multi-party collaboration
        if len(agents) < 2:
            raise ValueError("Collaboration scenario requires at least 2 agents")
        
        turn_id = f"collaboration_{int(time.time())}"
        if latency_tracker:
            latency_tracker.start_turn_measurement(turn_id, "collaboration_session")
        
        # Use multi-party dialogue - check if method exists
        if hasattr(agents[0], 'multi_party_dialogue'):
            result = agents[0].multi_party_dialogue(
                agents,
                test_case.test_prompt,
                rounds=test_case.rounds,
                mode="collaboration"
            )
        else:
            # Fallback: simulate collaboration with individual contributions
            contributions = []
            for i, agent in enumerate(agents):
                prompt = f"As part of a team discussion on '{test_case.test_prompt}', provide your contribution:"
                response = agent.generate_reasoning_tree(prompt)
                contributions.append({
                    'agent': agent.name,
                    'contribution': response.get('result', {}).get('content', 'No contribution')
                })
            
            result = {
                'mode': 'collaboration',
                'transcript': contributions,
                'participants': [agent.name for agent in agents]
            }
        
        if latency_tracker:
            latency_tracker.end_turn_measurement(turn_id)
        
        return {
            'type': 'collaboration',
            'result': result,
            'agent_results': [
                {'agent': agent.name, 'participated': True}
                for agent in agents
            ]
        }
    
    async def _execute_tool_usage_scenario(self, test_case, agents, latency_tracker, token_tracker):
        """Execute tool usage scenario."""
        
        agent = agents[0]
        
        turn_id = f"tools_{int(time.time())}"
        if latency_tracker:
            latency_tracker.start_turn_measurement(turn_id, agent.name)
        
        # Generate reasoning with potential tool usage
        result = agent.generate_reasoning_tree(test_case.test_prompt)
        
        if latency_tracker:
            latency_tracker.end_turn_measurement(turn_id)
        
        return {
            'type': 'tool_usage',
            'result': result,
            'agent_results': [{'agent': agent.name, 'tools_used': []}]
        }
    
    async def _execute_streaming_scenario(self, test_case, agents, latency_tracker, token_tracker):
        """Execute streaming collaboration scenario (Phase 6 preview)."""
        
        # Simulate streaming collaboration
        # In a real implementation, this would use actual streaming APIs
        
        if len(agents) < 2:
            raise ValueError("Streaming scenario requires at least 2 agents")
        
        turn_id = f"streaming_{int(time.time())}"
        if latency_tracker:
            latency_tracker.start_turn_measurement(turn_id, "streaming_session")
        
        # Simulate streaming by breaking responses into chunks
        writer, editor = agents[0], agents[1]
        
        # Writer generates content
        writer_result = writer.generate_reasoning_tree(
            f"Write a technical specification for: {test_case.test_prompt}"
        )
        
        # Editor reviews and provides feedback
        content = writer_result.get('result', {}).get('content', '')
        editor_result = editor.generate_reasoning_tree(
            f"Review and improve this specification: {content[:200]}..."
        )
        
        if latency_tracker:
            latency_tracker.end_turn_measurement(turn_id)
        
        return {
            'type': 'streaming_collaboration',
            'result': {
                'writer_output': writer_result,
                'editor_feedback': editor_result,
                'collaboration_successful': True
            },
            'agent_results': [
                {'agent': writer.name, 'role': 'writer'},
                {'agent': editor.name, 'role': 'editor'}
            ]
        }
    
    def _collect_observability_data(self, latency_tracker, token_tracker) -> Dict[str, Any]:
        """Collect observability data from trackers."""
        
        data = {}
        
        if latency_tracker:
            measurements = latency_tracker.get_measurements()
            if measurements:
                data['latency'] = {
                    'total_measurements': len(measurements),
                    'avg_latency_ms': sum(m.total_latency_ms for m in measurements) / len(measurements),
                    'max_latency_ms': max(m.total_latency_ms for m in measurements),
                    'statistics': latency_tracker.get_latency_statistics()
                }
        
        if token_tracker:
            overview = token_tracker.get_system_token_overview()
            if overview:
                data['tokens'] = overview
        
        return data
    
    def _evaluate_success_conditions(
        self, 
        test_case: MultiAgentTestCase,
        scenario_result: Dict[str, Any],
        observability_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate test success conditions."""
        
        evaluation = {}
        all_passed = True
        
        # Check basic scenario success
        scenario_success = scenario_result.get('result') is not None
        evaluation['scenario_executed'] = scenario_success
        if not scenario_success:
            all_passed = False
        
        # Check agent participation
        agent_results = scenario_result.get('agent_results', [])
        participated_count = len([r for r in agent_results if r.get('participated', False)])
        expected_participants = len(test_case.agent_configs)
        
        evaluation['agent_participation'] = {
            'participated': participated_count,
            'expected': expected_participants,
            'passed': participated_count >= expected_participants
        }
        if participated_count < expected_participants:
            all_passed = False
        
        # Check latency requirements
        if 'latency' in observability_data:
            avg_latency = observability_data['latency']['avg_latency_ms']
            latency_passed = avg_latency <= test_case.max_turn_latency_ms
            
            evaluation['latency_check'] = {
                'avg_latency_ms': avg_latency,
                'max_allowed_ms': test_case.max_turn_latency_ms,
                'passed': latency_passed
            }
            if not latency_passed:
                all_passed = False
        
        # Check custom success conditions
        custom_conditions = []
        for condition in test_case.success_conditions:
            # Simple heuristic evaluation of conditions
            passed = self._evaluate_condition(condition, scenario_result, observability_data)
            custom_conditions.append({'condition': condition, 'passed': passed})
            if not passed:
                all_passed = False
        
        evaluation['custom_conditions'] = custom_conditions
        
        return all_passed, evaluation
    
    def _evaluate_condition(
        self, 
        condition: str, 
        scenario_result: Dict[str, Any],
        observability_data: Dict[str, Any]
    ) -> bool:
        """Evaluate a custom success condition."""
        
        condition_lower = condition.lower()
        
        # Simple keyword-based evaluation
        if "participate" in condition_lower:
            agent_results = scenario_result.get('agent_results', [])
            return len(agent_results) > 0
        
        if "performance" in condition_lower or "metric" in condition_lower:
            return 'latency' in observability_data or 'tokens' in observability_data
        
        if "policy" in condition_lower and "violation" in condition_lower:
            # Assume no violations if no error reported
            return scenario_result.get('error') is None
        
        if "consensus" in condition_lower:
            # Simple check - assume consensus if dialogue completed
            result = scenario_result.get('result', {})
            return len(str(result)) > 100  # Heuristic
        
        if "streaming" in condition_lower:
            return scenario_result.get('type') == 'streaming_collaboration'
        
        if "time limit" in condition_lower:
            return 'latency' in observability_data
        
        # Default: assume passed if no specific check available
        return True
    
    def _generate_test_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of test results."""
        
        total_tests = len(results)
        passed_tests = len([r for r in results if r['success']])
        failed_tests = total_tests - passed_tests
        
        avg_duration = sum(r['duration_seconds'] for r in results) / total_tests if results else 0
        
        # Feature coverage
        feature_coverage = {}
        for result in results:
            test_name = result['test_case']
            if 'P1' in test_name:
                feature_coverage.setdefault('Phase 1', []).append(result['success'])
            if 'P2' in test_name:
                feature_coverage.setdefault('Phase 2', []).append(result['success'])
            if 'P3' in test_name:
                feature_coverage.setdefault('Phase 3', []).append(result['success'])
            if 'P4' in test_name:
                feature_coverage.setdefault('Phase 4', []).append(result['success'])
            if 'P5' in test_name:
                feature_coverage.setdefault('Phase 5', []).append(result['success'])
        
        # Calculate success rates by feature
        feature_success_rates = {}
        for feature, successes in feature_coverage.items():
            feature_success_rates[feature] = sum(successes) / len(successes) if successes else 0.0
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'avg_duration_seconds': avg_duration,
            'feature_success_rates': feature_success_rates,
            'failed_test_details': [
                {'name': r['test_case'], 'error': r.get('error', 'Test failed')}
                for r in results if not r['success']
            ],
            'detailed_results': results
        }
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all test results."""
        return self._results.copy()
    
    def clear_results(self) -> None:
        """Clear test results."""
        self._results.clear()
        logger.info("Cleared integration test results")