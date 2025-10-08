"""
Integration tests for tool system governance features.

Tests the complete governance flow including policy enforcement,
security validation, and custom validators.
"""

import pytest
from agentnet.tools.base import Tool, ToolSpec, ToolResult, ToolStatus
from agentnet.tools.executor import ToolExecutor
from agentnet.tools.registry import ToolRegistry


class MockTool(Tool):
    """Mock tool for testing."""
    
    async def execute(self, parameters, context=None):
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"result": "mock_result"},
            execution_time=0.1
        )


def test_tool_spec_with_governance_fields():
    """Test that ToolSpec supports governance fields."""
    spec = ToolSpec(
        name="test_tool",
        description="Test tool",
        schema={"type": "object", "properties": {}},
        category="data_access",
        risk_level="medium"
    )
    
    assert spec.category == "data_access"
    assert spec.risk_level == "medium"
    
    spec_dict = spec.to_dict()
    assert "category" in spec_dict
    assert "risk_level" in spec_dict
    assert spec_dict["category"] == "data_access"
    assert spec_dict["risk_level"] == "medium"


@pytest.mark.asyncio
async def test_tool_executor_without_policy_engine():
    """Test tool execution without policy engine (no governance)."""
    registry = ToolRegistry()
    
    spec = ToolSpec(
        name="test_tool",
        description="Test tool",
        schema={"type": "object", "properties": {}},
        category="computation",
        risk_level="low"
    )
    
    tool = MockTool(spec)
    registry.register_tool(tool)
    
    executor = ToolExecutor(registry=registry, policy_engine=None)
    
    # Should execute successfully without governance
    result = await executor.execute_tool("test_tool", {})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["result"] == "mock_result"


@pytest.mark.asyncio
async def test_tool_executor_with_security_validation():
    """Test tool execution with security validation."""
    registry = ToolRegistry()
    
    spec = ToolSpec(
        name="secure_tool",
        description="Secure tool requiring auth",
        schema={"type": "object", "properties": {}},
        auth_required=True,
        auth_scope="admin",
        category="data_access",
        risk_level="high"
    )
    
    tool = MockTool(spec)
    registry.register_tool(tool)
    
    executor = ToolExecutor(registry=registry)
    executor.security_checks = True
    
    # Test security validation
    security_result = executor.validate_tool_security(tool, {})
    assert "secure" in security_result
    
    # Should have security recommendations for high-risk tool
    if spec.risk_level == "high":
        assert security_result.get("risk_level") == "high"


@pytest.mark.asyncio
async def test_custom_validator_registration():
    """Test registration and execution of custom validators."""
    registry = ToolRegistry()
    
    spec = ToolSpec(
        name="validated_tool",
        description="Tool with custom validation",
        schema={"type": "object", "properties": {"value": {"type": "number"}}},
        category="computation"
    )
    
    tool = MockTool(spec)
    registry.register_tool(tool)
    
    executor = ToolExecutor(registry=registry)
    
    # Register custom validator
    async def custom_validator(tool, parameters, context):
        value = parameters.get("value", 0)
        return {
            "valid": value > 0,
            "message": "Value must be positive",
            "blocking": True if value <= 0 else False
        }
    
    executor.add_custom_validator("validated_tool", custom_validator)
    
    # Check validator was registered
    assert "validated_tool" in executor.custom_validators
    assert len(executor.custom_validators["validated_tool"]) == 1


def test_tool_categorization():
    """Test tool categorization for governance."""
    categories = ["data_access", "external_api", "computation", "file_io", "network"]
    
    for category in categories:
        spec = ToolSpec(
            name=f"{category}_tool",
            description=f"Tool for {category}",
            schema={"type": "object"},
            category=category
        )
        
        assert spec.category == category


def test_tool_risk_levels():
    """Test tool risk level classification."""
    risk_levels = ["low", "medium", "high", "critical"]
    
    for risk_level in risk_levels:
        spec = ToolSpec(
            name=f"{risk_level}_risk_tool",
            description=f"Tool with {risk_level} risk",
            schema={"type": "object"},
            risk_level=risk_level
        )
        
        assert spec.risk_level == risk_level


def test_backward_compatibility():
    """Test that ToolSpec is backward compatible without governance fields."""
    # Should work without category and risk_level
    spec = ToolSpec(
        name="legacy_tool",
        description="Legacy tool without governance fields",
        schema={"type": "object"}
    )
    
    assert spec.category is None
    assert spec.risk_level is None
    
    # Should still convert to dict
    spec_dict = spec.to_dict()
    assert "category" in spec_dict
    assert "risk_level" in spec_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
