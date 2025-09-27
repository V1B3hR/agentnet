#!/usr/bin/env python3
"""
Comprehensive test suite for high-priority security, policy, and tool governance features.

Tests the implementation of:
1. Advanced policy rule types (semantic_similarity, llm_classifier, numerical_threshold)
2. Enhanced security isolation mechanisms
3. Tool governance and custom validation
4. Agent orchestration policies
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from agentnet.core.policy.engine import PolicyEngine, PolicyAction
from agentnet.core.policy.rules import (
    ConstraintRule, 
    Severity,
    create_semantic_similarity_rule,
    create_llm_classifier_rule,
    create_numerical_threshold_rule,
    create_role_rule
)
from agentnet.core.auth.middleware import AuthMiddleware, SecurityIsolationManager
from agentnet.core.auth.rbac import RBACManager, Role, User
from agentnet.tools.executor import ToolExecutor
from agentnet.tools.registry import ToolRegistry
from agentnet.tools.base import ToolSpec, ToolStatus


class TestAdvancedPolicyRules:
    """Test advanced policy rule types implementation."""

    def setUp(self):
        self.policy_engine = PolicyEngine(name="test_advanced_policies")

    def test_semantic_similarity_rule_creation(self):
        """Test creation and basic functionality of semantic similarity rule."""
        rule = create_semantic_similarity_rule(
            name="test_semantic",
            max_similarity=0.8,
            embedding_set="test_corpus",
            severity=Severity.SEVERE
        )
        
        assert rule.name == "test_semantic"
        assert rule.severity == Severity.SEVERE
        assert "semantic" in rule.tags
        assert "similarity" in rule.tags

    def test_semantic_similarity_rule_evaluation(self):
        """Test semantic similarity rule evaluation."""
        rule = create_semantic_similarity_rule(
            name="test_semantic",
            max_similarity=0.9,
            window_size=2
        )
        
        # First evaluation should pass (no history)
        context1 = {
            "content": "This is the first message",
            "agent_name": "test_agent",
            "task_id": "task_1"
        }
        result1 = rule.evaluate(context1)
        assert result1.passed is True
        
        # Second evaluation with different content should pass
        context2 = {
            "content": "This is completely different content",
            "agent_name": "test_agent", 
            "task_id": "task_1"
        }
        result2 = rule.evaluate(context2)
        assert result2.passed is True
        
        # Third evaluation with very similar content should fail
        context3 = {
            "content": "This is the first message with minor changes",
            "agent_name": "test_agent",
            "task_id": "task_1"
        }
        result3 = rule.evaluate(context3)
        # Due to Jaccard similarity fallback, this should trigger
        assert result3.passed is False
        assert "similarity" in result3.rationale.lower()

    def test_llm_classifier_rule_creation(self):
        """Test LLM classifier rule creation."""
        rule = create_llm_classifier_rule(
            name="toxicity_check",
            model="test-model",
            threshold=0.7,
            classification_target="toxicity"
        )
        
        assert rule.name == "toxicity_check"
        assert "llm" in rule.tags
        assert "classification" in rule.tags
        assert "toxicity" in rule.tags

    def test_llm_classifier_rule_evaluation(self):
        """Test LLM classifier rule evaluation."""
        rule = create_llm_classifier_rule(
            name="toxicity_check",
            threshold=0.5,
            classification_target="toxicity"
        )
        
        # Test with non-toxic content
        safe_content = {"content": "This is a pleasant and helpful message"}
        result_safe = rule.evaluate(safe_content)
        assert result_safe.passed is True
        
        # Test with content containing toxic keywords
        toxic_content = {"content": "This message contains hate and toxic language"}  
        result_toxic = rule.evaluate(toxic_content)
        assert result_toxic.passed is False
        assert "toxicity" in result_toxic.rationale

    def test_llm_classifier_pii_detection(self):
        """Test LLM classifier for PII detection."""
        rule = create_llm_classifier_rule(
            name="pii_check",
            threshold=0.3,
            classification_target="pii"
        )
        
        # Test with PII content
        pii_content = {"content": "My SSN is 123-45-6789 and email is test@example.com"}
        result_pii = rule.evaluate(pii_content)
        assert result_pii.passed is False
        assert "pii" in result_pii.rationale.lower()

    def test_numerical_threshold_rule_creation(self):
        """Test numerical threshold rule creation."""
        rule = create_numerical_threshold_rule(
            name="confidence_check",
            metric_name="confidence_score",
            threshold=0.8,
            operator="greater_than"
        )
        
        assert rule.name == "confidence_check"
        assert "numerical" in rule.tags
        assert "threshold" in rule.tags

    def test_numerical_threshold_rule_evaluation(self):
        """Test numerical threshold rule evaluation."""
        rule = create_numerical_threshold_rule(
            name="confidence_check",
            metric_name="confidence_score", 
            threshold=0.8,
            operator="greater_than"
        )
        
        # Test passing case
        high_confidence = {"confidence_score": 0.9}
        result_pass = rule.evaluate(high_confidence)
        assert result_pass.passed is True
        
        # Test failing case
        low_confidence = {"confidence_score": 0.6}
        result_fail = rule.evaluate(low_confidence)
        assert result_fail.passed is False
        assert "confidence_score" in result_fail.rationale

    def test_numerical_threshold_operators(self):
        """Test different numerical threshold operators."""
        operators_and_values = [
            ("less_than", 5.0, 3.0, True),
            ("less_than", 5.0, 7.0, False),
            ("greater_equal", 5.0, 5.0, True),
            ("greater_equal", 5.0, 4.5, False),
            ("equals", 5.0, 5.0, True),
            ("not_equals", 5.0, 3.0, True)
        ]
        
        for operator, threshold, test_value, should_pass in operators_and_values:
            rule = create_numerical_threshold_rule(
                name=f"test_{operator}",
                metric_name="test_metric",
                threshold=threshold,
                operator=operator
            )
            
            context = {"test_metric": test_value}
            result = rule.evaluate(context)
            assert result.passed == should_pass, f"Failed for {operator} {threshold} vs {test_value}"


class TestAgentOrchestrationPolicies:
    """Test agent orchestration and coordination policies."""

    def setUp(self):
        self.policy_engine = PolicyEngine(name="orchestration_test")
        
        # Add orchestration rules
        role_rule = create_role_rule(
            name="allowed_orchestration_roles",
            allowed_roles=["coordinator", "analyst", "executor"],
            severity=Severity.MAJOR
        )
        role_rule.tags.append("orchestration")
        
        trust_rule = create_numerical_threshold_rule(
            name="min_trust_level",
            metric_name="agent_trust_level",
            threshold=0.3,
            operator="greater_equal"
        )
        trust_rule.tags.append("coordination")
        
        self.policy_engine.add_rule(role_rule)
        self.policy_engine.add_rule(trust_rule)

    def test_agent_orchestration_evaluation(self):
        """Test multi-agent orchestration policy evaluation."""
        agents = [
            {
                "name": "coordinator_agent",
                "role": "coordinator", 
                "capabilities": ["planning", "coordination"],
                "trust_level": 0.9
            },
            {
                "name": "analyst_agent",
                "role": "analyst",
                "capabilities": ["analysis", "research"],
                "trust_level": 0.8
            },
            {
                "name": "untrusted_agent",
                "role": "executor",
                "capabilities": ["execution"],
                "trust_level": 0.1  # Below minimum trust
            }
        ]
        
        coordination_context = {"task_type": "collaborative_analysis"}
        
        result = self.policy_engine.evaluate_agent_orchestration(agents, coordination_context)
        
        # Should allow first two agents but block the third
        assert len(result["allowed_agents"]) == 2
        assert len(result["blocked_agents"]) == 1
        assert result["blocked_agents"][0]["agent"]["name"] == "untrusted_agent"
        
        # Should generate coordination rules for allowed agents
        assert len(result["coordination_rules"]) > 0
        rule_types = [rule["type"] for rule in result["coordination_rules"]]
        assert "communication_order" in rule_types

    def test_coordination_rules_generation(self):
        """Test coordination rules generation logic."""
        # Test with agents requiring shared resources
        agents_with_resources = [
            {
                "name": "agent1",
                "role": "coordinator",
                "capabilities": ["shared_resource", "planning"],
                "trust_level": 0.8
            },
            {
                "name": "agent2", 
                "role": "executor",
                "capabilities": ["shared_resource", "execution"],
                "trust_level": 0.7
            }
        ]
        
        result = self.policy_engine.evaluate_agent_orchestration(
            agents_with_resources, {"task_type": "resource_intensive"}
        )
        
        coordination_rules = result["coordination_rules"]
        rule_types = [rule["type"] for rule in coordination_rules]
        
        assert "resource_sharing" in rule_types
        resource_rule = next(rule for rule in coordination_rules if rule["type"] == "resource_sharing")
        assert resource_rule["resource_locks"] is True


class TestToolGovernanceAndSecurity:
    """Test enhanced tool governance and security features."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.policy_engine = PolicyEngine(name="tool_governance")
        self.tool_executor = ToolExecutor(
            registry=self.registry,
            policy_engine=self.policy_engine
        )
        
        # Add a test tool
        test_spec = ToolSpec(
            name="test_tool",
            description="Test tool for governance",
            category="test",
            parameters_schema={"type": "object", "properties": {}},
            auth_required=False
        )
        
        class TestTool:
            def __init__(self):
                self.spec = test_spec
            
            async def execute(self, parameters):
                return {"result": "test execution"}
        
        self.registry.register_tool(TestTool())

    def test_tool_security_validation(self):
        """Test tool security validation logic."""
        tool = self.registry.get_tool("test_tool")
        
        # Test safe parameters
        safe_params = {"input": "safe data", "count": 5}
        security_result = self.tool_executor.validate_tool_security(tool, safe_params)
        assert security_result["secure"] is True
        assert security_result["security_level"] == "standard"
        
        # Test dangerous parameters
        dangerous_params = {"command": "rm -rf /", "input": "safe data"}
        security_result = self.tool_executor.validate_tool_security(tool, dangerous_params)
        assert len(security_result["warnings"]) > 0

    def test_custom_validator_registration(self):
        """Test custom validator registration and execution."""
        # Register custom validator
        async def custom_validator(tool, parameters, context):
            if parameters.get("forbidden_param"):
                return {
                    "valid": False,
                    "blocking": True,
                    "reason": "Forbidden parameter detected"
                }
            return {"valid": True}
        
        self.tool_executor.add_custom_validator("test_tool", custom_validator)
        
        # Test validation
        assert "test_tool" in self.tool_executor.custom_validators
        assert len(self.tool_executor.custom_validators["test_tool"]) == 1

    @pytest.mark.asyncio
    async def test_governance_validation(self):
        """Test governance validation in tool execution."""
        tool = self.registry.get_tool("test_tool")
        
        # Mock policy engine evaluation
        mock_policy_result = Mock()
        mock_policy_result.action.value = "allow"
        mock_policy_result.violations = []
        
        with patch.object(self.policy_engine, 'evaluate_tool_usage_policy', return_value=mock_policy_result):
            governance_result = await self.tool_executor.validate_tool_governance(
                tool, {"param": "value"}, "user123", {"agent_name": "test_agent"}
            )
            
            assert governance_result["allowed"] is True
            assert governance_result["risk_level"] == "low"

    @pytest.mark.asyncio
    async def test_governance_blocking(self):
        """Test governance policy blocking tool execution."""
        tool = self.registry.get_tool("test_tool")
        
        # Mock policy engine to block execution
        mock_policy_result = Mock()
        mock_policy_result.action.value = "block"
        mock_policy_result.violations = [Mock()]
        mock_policy_result.violations[0].to_dict.return_value = {"rule": "blocked"}
        
        with patch.object(self.policy_engine, 'evaluate_tool_usage_policy', return_value=mock_policy_result):
            governance_result = await self.tool_executor.validate_tool_governance(
                tool, {"param": "value"}, "user123", {"agent_name": "test_agent"}
            )
            
            assert governance_result["allowed"] is False
            assert governance_result["risk_level"] == "high"


class TestSecurityIsolationEnhancements:
    """Test enhanced security isolation mechanisms."""

    def setUp(self):
        self.rbac_manager = RBACManager()
        self.isolation_manager = SecurityIsolationManager()
        
        # Create test users
        self.admin_user = self.rbac_manager.create_user(
            "admin1", "admin", "admin@test.com", [Role.ADMIN], "tenant1"
        )
        self.regular_user = self.rbac_manager.create_user(
            "user1", "user", "user@test.com", [Role.TENANT_USER], "tenant1"
        )

    def test_isolated_session_creation(self):
        """Test creation of isolated sessions."""
        session_context = self.isolation_manager.create_isolated_session(
            self.regular_user, "session123", "standard"
        )
        
        assert session_context["session_id"] == "session123"
        assert session_context["user_id"] == self.regular_user.user_id
        assert session_context["tenant_id"] == self.regular_user.tenant_id
        assert session_context["isolation_level"] == "standard"
        assert "resource_access" in session_context
        assert "network_policy" in session_context

    def test_resource_access_determination(self):
        """Test resource access policy determination."""
        # Test admin user gets more resources
        admin_session = self.isolation_manager.create_isolated_session(
            self.admin_user, "admin_session", "standard"
        )
        
        regular_session = self.isolation_manager.create_isolated_session(
            self.regular_user, "user_session", "standard"
        )
        
        admin_resources = admin_session["resource_access"]
        user_resources = regular_session["resource_access"]
        
        assert admin_resources["compute_quota"] > user_resources["compute_quota"]
        assert admin_resources["memory_limit_mb"] > user_resources["memory_limit_mb"]
        assert admin_resources["network_access"] == "full"
        assert user_resources["network_access"] == "restricted"

    def test_strict_isolation_level(self):
        """Test strict isolation level restrictions."""
        strict_session = self.isolation_manager.create_isolated_session(
            self.regular_user, "strict_session", "strict"
        )
        
        resources = strict_session["resource_access"]
        network = strict_session["network_policy"]
        
        assert resources["compute_quota"] == 50  # Reduced for strict
        assert resources["memory_limit_mb"] == 256
        assert resources["network_access"] == "none"
        assert network["outbound_allowed"] is False

    def test_resource_locking_mechanism(self):
        """Test exclusive resource locking."""
        session1 = "session1"
        session2 = "session2"
        resource_id = "shared_resource_1"
        
        # Session 1 acquires lock
        lock_acquired = self.isolation_manager.acquire_resource_lock(session1, resource_id)
        assert lock_acquired is True
        
        # Session 2 tries to acquire same lock
        lock_blocked = self.isolation_manager.acquire_resource_lock(session2, resource_id)
        assert lock_blocked is False
        
        # Session 1 releases lock
        lock_released = self.isolation_manager.release_resource_lock(session1, resource_id)
        assert lock_released is True
        
        # Now session 2 can acquire it
        lock_acquired_2 = self.isolation_manager.acquire_resource_lock(session2, resource_id)
        assert lock_acquired_2 is True

    def test_session_cleanup(self):
        """Test session cleanup and resource release."""
        session_id = "cleanup_test_session"
        
        # Create session and acquire resources
        self.isolation_manager.create_isolated_session(
            self.regular_user, session_id, "standard"
        )
        
        self.isolation_manager.acquire_resource_lock(session_id, "resource1")
        self.isolation_manager.acquire_resource_lock(session_id, "resource2")
        
        # Verify session exists and resources are locked
        assert session_id in self.isolation_manager._active_sessions
        assert len([r for r, s in self.isolation_manager._resource_locks.items() if s == session_id]) == 2
        
        # Cleanup session
        self.isolation_manager.cleanup_session(session_id)
        
        # Verify cleanup
        assert session_id not in self.isolation_manager._active_sessions
        assert len([r for r, s in self.isolation_manager._resource_locks.items() if s == session_id]) == 0

    def test_tenant_isolation_boundaries(self):
        """Test tenant isolation boundary enforcement."""
        # Create users in different tenants
        tenant1_user = self.rbac_manager.create_user(
            "t1_user", "tenant1_user", "t1@test.com", [Role.TENANT_USER], "tenant1"
        )
        tenant2_user = self.rbac_manager.create_user(
            "t2_user", "tenant2_user", "t2@test.com", [Role.TENANT_USER], "tenant2"
        )
        
        # Create sessions
        t1_session = self.isolation_manager.create_isolated_session(
            tenant1_user, "t1_session", "standard"
        )
        t2_session = self.isolation_manager.create_isolated_session(
            tenant2_user, "t2_session", "standard"
        )
        
        # Verify tenant boundaries
        assert "tenant1" in self.isolation_manager._tenant_boundaries
        assert "tenant2" in self.isolation_manager._tenant_boundaries
        assert "t1_session" in self.isolation_manager._tenant_boundaries["tenant1"]
        assert "t2_session" in self.isolation_manager._tenant_boundaries["tenant2"]
        
        # Verify cross-tenant access policies
        t1_data_policy = t1_session["data_access_policy"]
        t2_data_policy = t2_session["data_access_policy"]
        
        assert t1_data_policy["tenant_isolation"] is True
        assert t1_data_policy["cross_tenant_access"] is False
        assert t2_data_policy["tenant_isolation"] is True
        assert t2_data_policy["cross_tenant_access"] is False


if __name__ == "__main__":
    pytest.main([__file__])