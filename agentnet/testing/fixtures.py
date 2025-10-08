"""
Test Fixtures and Scenario Generation for AgentNet

Provides reusable test fixtures and automated scenario generation
for comprehensive testing across different configurations.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Generator
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    """Profile for generating test agents."""

    name: str
    style: Dict[str, float]
    role: str = "general"
    specialization: Optional[str] = None

    # Capabilities
    memory_enabled: bool = False
    tools_enabled: bool = False
    policy_compliance: bool = True

    # Behavioral traits
    response_length: str = "medium"  # short, medium, long
    creativity_level: str = "balanced"  # low, balanced, high
    risk_tolerance: str = "medium"  # low, medium, high


@dataclass
class TestScenario:
    """Test scenario definition."""

    name: str
    description: str
    category: str  # reasoning, dialogue, collaboration, tools, etc.

    # Scenario parameters
    prompts: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Complexity and requirements
    complexity_level: str = "medium"  # simple, medium, complex
    estimated_duration_seconds: float = 30.0

    # Feature requirements
    requires_memory: bool = False
    requires_tools: bool = False
    requires_multi_agent: bool = False
    min_agents: int = 1
    max_agents: int = 1

    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = "AgentNet Testing Framework"


class AgentFixtures:
    """
    Provides pre-configured agent fixtures for testing.

    Generates diverse agent configurations for comprehensive
    testing across different styles and capabilities.
    """

    def __init__(self):
        self._profiles = self._create_standard_profiles()

    def _create_standard_profiles(self) -> List[AgentProfile]:
        """Create standard agent profiles for testing."""

        profiles = [
            # Analytical agents
            AgentProfile(
                name="DataAnalyst",
                style={"logic": 0.9, "creativity": 0.3, "analytical": 0.95},
                role="analyst",
                specialization="data_analysis",
                memory_enabled=True,
                response_length="long",
                creativity_level="low",
            ),
            AgentProfile(
                name="SystemArchitect",
                style={"logic": 0.85, "creativity": 0.6, "analytical": 0.9},
                role="architect",
                specialization="system_design",
                tools_enabled=True,
                response_length="long",
                creativity_level="balanced",
            ),
            # Creative agents
            AgentProfile(
                name="CreativeWriter",
                style={"logic": 0.4, "creativity": 0.95, "analytical": 0.3},
                role="creative",
                specialization="content_creation",
                response_length="long",
                creativity_level="high",
                risk_tolerance="high",
            ),
            AgentProfile(
                name="Brainstormer",
                style={"logic": 0.6, "creativity": 0.9, "analytical": 0.5},
                role="ideation",
                specialization="brainstorming",
                response_length="medium",
                creativity_level="high",
            ),
            # Balanced agents
            AgentProfile(
                name="GeneralAgent",
                style={"logic": 0.7, "creativity": 0.7, "analytical": 0.7},
                role="general",
                memory_enabled=True,
                tools_enabled=True,
                response_length="medium",
                creativity_level="balanced",
            ),
            AgentProfile(
                name="Coordinator",
                style={"logic": 0.8, "creativity": 0.5, "analytical": 0.7},
                role="coordinator",
                specialization="project_management",
                memory_enabled=True,
                response_length="medium",
            ),
            # Specialized agents
            AgentProfile(
                name="SecurityExpert",
                style={"logic": 0.9, "creativity": 0.2, "analytical": 0.9},
                role="security",
                specialization="cybersecurity",
                tools_enabled=True,
                policy_compliance=True,
                risk_tolerance="low",
                response_length="medium",
            ),
            AgentProfile(
                name="QualityAssurance",
                style={"logic": 0.85, "creativity": 0.3, "analytical": 0.95},
                role="qa",
                specialization="quality_assurance",
                tools_enabled=True,
                response_length="long",
                risk_tolerance="low",
            ),
            # Debate/dialogue agents
            AgentProfile(
                name="Advocate",
                style={"logic": 0.8, "creativity": 0.7, "confidence": 0.9},
                role="advocate",
                specialization="argumentation",
                response_length="medium",
                risk_tolerance="medium",
            ),
            AgentProfile(
                name="Critic",
                style={"logic": 0.9, "creativity": 0.4, "analytical": 0.9},
                role="critic",
                specialization="critical_analysis",
                response_length="medium",
                risk_tolerance="low",
            ),
        ]

        return profiles

    def get_profile(self, name: str) -> Optional[AgentProfile]:
        """Get agent profile by name."""
        for profile in self._profiles:
            if profile.name == name:
                return profile
        return None

    def get_profiles_by_role(self, role: str) -> List[AgentProfile]:
        """Get all profiles with specific role."""
        return [p for p in self._profiles if p.role == role]

    def get_profiles_by_specialization(self, specialization: str) -> List[AgentProfile]:
        """Get profiles with specific specialization."""
        return [p for p in self._profiles if p.specialization == specialization]

    def create_agent_from_profile(self, profile: AgentProfile, engine=None) -> Any:
        """Create AgentNet agent from profile."""

        # Import here to avoid circular imports
        from ..core.agent import AgentNet
        from ..providers.example import ExampleEngine

        if engine is None:
            engine = ExampleEngine()

        # Setup agent configuration
        agent_config = {"engine": engine}

        # Add memory if enabled
        if profile.memory_enabled:
            agent_config["memory_config"] = {
                "memory": {
                    "short_term": {"enabled": True, "max_entries": 20},
                    "episodic": {"enabled": True},
                }
            }

        # Add tools if enabled
        if profile.tools_enabled:
            try:
                from ..tools.registry import ToolRegistry
                from ..tools.examples import CalculatorTool, StatusCheckTool

                tool_registry = ToolRegistry()
                tool_registry.register_tool(CalculatorTool())
                tool_registry.register_tool(StatusCheckTool())
                agent_config["tool_registry"] = tool_registry
            except ImportError:
                logger.warning(f"Tools not available for {profile.name}")

        return AgentNet(name=profile.name, style=profile.style, **agent_config)

    def create_agent_group(
        self, group_type: str, size: Optional[int] = None, engine=None
    ) -> List[Any]:
        """Create a group of agents for specific scenarios."""

        if group_type == "debate_pair":
            profiles = [self.get_profile("Advocate"), self.get_profile("Critic")]
        elif group_type == "analysis_team":
            profiles = [
                self.get_profile("DataAnalyst"),
                self.get_profile("SystemArchitect"),
                self.get_profile("QualityAssurance"),
            ]
        elif group_type == "creative_team":
            profiles = [
                self.get_profile("CreativeWriter"),
                self.get_profile("Brainstormer"),
                self.get_profile("Coordinator"),
            ]
        elif group_type == "security_review":
            profiles = [
                self.get_profile("SecurityExpert"),
                self.get_profile("SystemArchitect"),
                self.get_profile("QualityAssurance"),
            ]
        elif group_type == "diverse_group":
            # Mix of different types
            profiles = [
                self.get_profile("GeneralAgent"),
                self.get_profile("DataAnalyst"),
                self.get_profile("CreativeWriter"),
                self.get_profile("Coordinator"),
            ]
        else:
            # Random selection
            profiles = random.sample(
                self._profiles, min(size or 3, len(self._profiles))
            )

        # Apply size limit if specified
        if size and len(profiles) > size:
            profiles = profiles[:size]
        elif size and len(profiles) < size:
            # Duplicate profiles to reach desired size
            while len(profiles) < size:
                profiles.append(random.choice(self._profiles))

        return [self.create_agent_from_profile(p, engine) for p in profiles if p]

    def get_all_profiles(self) -> List[AgentProfile]:
        """Get all available agent profiles."""
        return self._profiles.copy()

    def add_custom_profile(self, profile: AgentProfile) -> None:
        """Add a custom agent profile."""
        self._profiles.append(profile)
        logger.debug(f"Added custom agent profile: {profile.name}")


class ScenarioGenerator:
    """
    Generates diverse test scenarios for comprehensive AgentNet testing.

    Creates scenarios across different categories, complexity levels,
    and feature requirements for thorough validation.
    """

    def __init__(self):
        self._scenarios = self._create_standard_scenarios()
        self._prompt_templates = self._create_prompt_templates()

    def _create_standard_scenarios(self) -> List[TestScenario]:
        """Create standard test scenarios."""

        scenarios = [
            # Basic reasoning scenarios
            TestScenario(
                name="Basic Problem Solving",
                description="Test basic reasoning and problem-solving capabilities",
                category="reasoning",
                prompts=[
                    "How would you optimize database query performance?",
                    "Explain the trade-offs between different sorting algorithms",
                    "Design a caching strategy for a web application",
                ],
                complexity_level="simple",
                estimated_duration_seconds=15.0,
                success_criteria=[
                    "Provides coherent analysis",
                    "Identifies key considerations",
                    "Suggests practical solutions",
                ],
                tags=["reasoning", "problem_solving", "basic"],
            ),
            TestScenario(
                name="Complex System Design",
                description="Test complex reasoning for system architecture",
                category="reasoning",
                prompts=[
                    "Design a microservices architecture for an e-commerce platform",
                    "How would you implement distributed consensus in a blockchain?",
                    "Architect a real-time analytics system for IoT data",
                ],
                complexity_level="complex",
                estimated_duration_seconds=45.0,
                requires_tools=True,
                success_criteria=[
                    "Comprehensive architectural analysis",
                    "Addresses scalability concerns",
                    "Considers trade-offs and alternatives",
                ],
                tags=["reasoning", "architecture", "complex"],
            ),
            # Dialogue scenarios
            TestScenario(
                name="Collaborative Discussion",
                description="Test multi-agent collaborative dialogue",
                category="dialogue",
                prompts=[
                    "Discuss the best approach for API versioning",
                    "Plan a software development sprint",
                    "Evaluate different cloud deployment strategies",
                ],
                complexity_level="medium",
                estimated_duration_seconds=30.0,
                requires_multi_agent=True,
                min_agents=2,
                max_agents=4,
                success_criteria=[
                    "All agents participate meaningfully",
                    "Ideas build on each other",
                    "Reaches some form of consensus or clear alternatives",
                ],
                tags=["dialogue", "collaboration", "multi_agent"],
            ),
            TestScenario(
                name="Technical Debate",
                description="Test adversarial dialogue and debate",
                category="debate",
                prompts=[
                    "Debate: Should we use SQL or NoSQL for this application?",
                    "Argue the merits of monorepo vs multi-repo approaches",
                    "Discuss whether to build or buy a solution",
                ],
                complexity_level="medium",
                estimated_duration_seconds=40.0,
                requires_multi_agent=True,
                min_agents=2,
                max_agents=2,
                success_criteria=[
                    "Both sides present strong arguments",
                    "Addresses counterarguments",
                    "Maintains professional discourse",
                ],
                tags=["debate", "adversarial", "technical"],
            ),
            # Tool usage scenarios
            TestScenario(
                name="Research and Analysis",
                description="Test tool usage for research and data analysis",
                category="tools",
                prompts=[
                    "Research current trends in machine learning and analyze the data",
                    "Calculate system performance metrics and generate a report",
                    "Investigate security vulnerabilities and assess risk levels",
                ],
                complexity_level="medium",
                estimated_duration_seconds=25.0,
                requires_tools=True,
                requires_memory=True,
                success_criteria=[
                    "Uses appropriate tools effectively",
                    "Integrates tool results into analysis",
                    "Provides actionable insights",
                ],
                tags=["tools", "research", "analysis"],
            ),
            # Memory scenarios
            TestScenario(
                name="Contextual Learning",
                description="Test memory and context retention across interactions",
                category="memory",
                prompts=[
                    "Remember that we're working on a payments system. Now analyze security requirements.",
                    "Based on our previous discussion about the database schema, suggest optimization strategies.",
                    "Considering the architectural decisions we made earlier, how should we handle error logging?",
                ],
                complexity_level="medium",
                estimated_duration_seconds=35.0,
                requires_memory=True,
                success_criteria=[
                    "References previous context appropriately",
                    "Builds on earlier information",
                    "Maintains consistency across interactions",
                ],
                tags=["memory", "context", "learning"],
            ),
            # Performance scenarios
            TestScenario(
                name="High Throughput Processing",
                description="Test performance under high load",
                category="performance",
                prompts=[
                    "Process this batch of 100 short analysis requests",
                    "Handle concurrent optimization problems",
                    "Manage multiple simultaneous discussions",
                ],
                complexity_level="complex",
                estimated_duration_seconds=60.0,
                success_criteria=[
                    "Maintains response quality under load",
                    "Completes processing within time limits",
                    "No significant error rate increase",
                ],
                tags=["performance", "throughput", "stress"],
            ),
            # Integration scenarios
            TestScenario(
                name="Full Stack Integration",
                description="Test end-to-end functionality across all features",
                category="integration",
                prompts=[
                    "Lead a complete system design review with security analysis, cost estimation, and implementation planning",
                    "Facilitate a multi-stakeholder technical decision with documentation and follow-up actions",
                    "Coordinate a complex problem-solving session with tool usage and knowledge retention",
                ],
                complexity_level="complex",
                estimated_duration_seconds=90.0,
                requires_memory=True,
                requires_tools=True,
                requires_multi_agent=True,
                min_agents=3,
                max_agents=5,
                success_criteria=[
                    "Integrates all required features smoothly",
                    "Maintains performance across feature combinations",
                    "Produces comprehensive deliverables",
                ],
                tags=["integration", "full_stack", "comprehensive"],
            ),
        ]

        return scenarios

    def _create_prompt_templates(self) -> Dict[str, List[str]]:
        """Create prompt templates for scenario generation."""

        templates = {
            "analysis": [
                "Analyze {topic} and provide recommendations",
                "Evaluate the pros and cons of {topic}",
                "What are the key considerations for {topic}?",
                "How would you approach {topic} in a production environment?",
            ],
            "design": [
                "Design a {system_type} for {use_case}",
                "Architect a solution for {problem}",
                "What would be your approach to building {system_type}?",
                "How would you scale {system_type} for {scale_requirement}?",
            ],
            "problem_solving": [
                "How would you solve {problem}?",
                "What's the best approach to handle {challenge}?",
                "Debug this issue: {problem_description}",
                "Optimize {system} for better {metric}",
            ],
            "collaboration": [
                "Let's discuss {topic} and reach a decision",
                "Work together to plan {project}",
                "Collaborate on solving {challenge}",
                "Review and improve {deliverable}",
            ],
        }

        return templates

    def get_scenario(self, name: str) -> Optional[TestScenario]:
        """Get scenario by name."""
        for scenario in self._scenarios:
            if scenario.name == name:
                return scenario
        return None

    def get_scenarios_by_category(self, category: str) -> List[TestScenario]:
        """Get scenarios by category."""
        return [s for s in self._scenarios if s.category == category]

    def get_scenarios_by_complexity(self, complexity: str) -> List[TestScenario]:
        """Get scenarios by complexity level."""
        return [s for s in self._scenarios if s.complexity_level == complexity]

    def get_scenarios_requiring_features(
        self, memory: bool = False, tools: bool = False, multi_agent: bool = False
    ) -> List[TestScenario]:
        """Get scenarios requiring specific features."""

        filtered = []
        for scenario in self._scenarios:
            if memory and not scenario.requires_memory:
                continue
            if tools and not scenario.requires_tools:
                continue
            if multi_agent and not scenario.requires_multi_agent:
                continue
            filtered.append(scenario)

        return filtered

    def generate_custom_scenario(
        self, category: str, topic: str, complexity: str = "medium", **kwargs
    ) -> TestScenario:
        """Generate a custom scenario based on parameters."""

        # Get appropriate prompt template
        if category in self._prompt_templates:
            template = random.choice(self._prompt_templates[category])
            # Provide default values for template variables
            template_vars = {
                "topic": topic,
                "system_type": kwargs.get("system_type", topic),
                "use_case": kwargs.get("use_case", f"{topic} application"),
                "problem": kwargs.get("problem", f"{topic} optimization"),
                "challenge": kwargs.get("challenge", f"{topic} implementation"),
                "problem_description": kwargs.get(
                    "problem_description", f"Issues with {topic}"
                ),
                "system": kwargs.get("system", topic),
                "metric": kwargs.get("metric", "performance"),
                "project": kwargs.get("project", f"{topic} project"),
                "deliverable": kwargs.get("deliverable", f"{topic} specification"),
                "scale_requirement": kwargs.get(
                    "scale_requirement", "high availability"
                ),
                **kwargs,
            }
            try:
                prompt = template.format(**template_vars)
            except KeyError as e:
                # Fallback if template has unexpected variables
                prompt = f"Please analyze and discuss {topic}"
        else:
            prompt = f"Please analyze and discuss {topic}"

        # Determine requirements based on complexity
        if complexity == "simple":
            duration = 15.0
            requires_tools = False
            requires_memory = False
            min_agents = 1
        elif complexity == "complex":
            duration = 60.0
            requires_tools = True
            requires_memory = True
            min_agents = 2
        else:  # medium
            duration = 30.0
            requires_tools = random.choice([True, False])
            requires_memory = random.choice([True, False])
            min_agents = random.choice([1, 2])

        return TestScenario(
            name=f"Custom {category.title()}: {topic}",
            description=f"Custom {category} scenario for {topic}",
            category=category,
            prompts=[prompt],
            complexity_level=complexity,
            estimated_duration_seconds=duration,
            requires_tools=requires_tools,
            requires_memory=requires_memory,
            requires_multi_agent=min_agents > 1,
            min_agents=min_agents,
            max_agents=min_agents + 2,
            success_criteria=[
                "Addresses the topic appropriately",
                "Demonstrates expected capabilities",
                "Completes within time limit",
            ],
            tags=[category, complexity, "custom"],
            author="Custom Generator",
        )

    def generate_scenario_matrix(
        self, categories: List[str], complexity_levels: List[str], topics: List[str]
    ) -> List[TestScenario]:
        """Generate a matrix of scenarios across parameters."""

        scenarios = []

        for category in categories:
            for complexity in complexity_levels:
                for topic in topics:
                    scenario = self.generate_custom_scenario(
                        category=category, complexity=complexity, topic=topic
                    )
                    scenarios.append(scenario)

        return scenarios

    def get_all_scenarios(self) -> List[TestScenario]:
        """Get all available scenarios."""
        return self._scenarios.copy()

    def add_custom_scenario(self, scenario: TestScenario) -> None:
        """Add a custom scenario."""
        self._scenarios.append(scenario)
        logger.debug(f"Added custom scenario: {scenario.name}")

    def get_scenarios_for_duration(
        self, max_duration_seconds: float
    ) -> List[TestScenario]:
        """Get scenarios that fit within time duration."""
        return [
            s
            for s in self._scenarios
            if s.estimated_duration_seconds <= max_duration_seconds
        ]
