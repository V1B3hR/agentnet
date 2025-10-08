"""
AgentNet Testing Framework

Provides comprehensive test matrix and integration testing capabilities
for systematic validation across different agent configurations, features,
and scenarios as specified in Phase 5 requirements.
"""

from .matrix import TestMatrix, TestConfiguration, TestResult, FeatureSet, AgentType
from .integration import IntegrationTestSuite, MultiAgentTestCase
from .regression import RegressionTestSuite, PerformanceRegression, PerformanceBaseline
from .fixtures import AgentFixtures, ScenarioGenerator, AgentProfile, TestScenario

__all__ = [
    "TestMatrix",
    "TestConfiguration",
    "TestResult",
    "FeatureSet",
    "AgentType",
    "IntegrationTestSuite",
    "MultiAgentTestCase",
    "RegressionTestSuite",
    "PerformanceRegression",
    "PerformanceBaseline",
    "AgentFixtures",
    "ScenarioGenerator",
    "AgentProfile",
    "TestScenario",
]
