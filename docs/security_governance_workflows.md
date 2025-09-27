# Security & Governance Workflows

This document provides practical workflows for implementing and using AgentNet's advanced security, policy, and governance features.

## Table of Contents

1. [Advanced Policy Rules](#advanced-policy-rules)
2. [Security Isolation](#security-isolation)
3. [Tool Governance](#tool-governance)
4. [Agent Orchestration](#agent-orchestration)
5. [Practical Examples](#practical-examples)

## Advanced Policy Rules

AgentNet now supports three advanced policy rule types that integrate seamlessly with the existing policy engine:

### Semantic Similarity Rules

Prevent agents from generating repetitive or overly similar content by using semantic similarity detection.

```python
from agentnet.core.policy.rules import create_semantic_similarity_rule
from agentnet.core.policy.engine import PolicyEngine, Severity

# Create semantic similarity rule
similarity_rule = create_semantic_similarity_rule(
    name="prevent_repetitive_content",
    max_similarity=0.85,  # Maximum allowed similarity threshold
    embedding_set="restricted_corpora",  # Embedding set to compare against
    window_size=5,  # Number of historical outputs to compare
    severity=Severity.SEVERE,
    description="Prevent agents from generating repetitive content"
)

# Add to policy engine
policy_engine = PolicyEngine(name="production_policies")
policy_engine.add_rule(similarity_rule)

# Evaluate content
context = {
    "content": "Generated agent response",
    "agent_name": "assistant_agent",
    "task_id": "conversation_123"
}

result = policy_engine.evaluate(context, tags=["semantic"])
if result.action.value == "block":
    print(f"Content blocked due to similarity: {result.explanation}")
```

### LLM Classifier Rules

Use LLM-based classification to detect toxicity, PII, inappropriate content, and other policy violations.

```python
from agentnet.core.policy.rules import create_llm_classifier_rule

# Toxicity detection
toxicity_rule = create_llm_classifier_rule(
    name="toxicity_filter",
    model="moderation-small",
    threshold=0.7,
    classification_target="toxicity",
    severity=Severity.MAJOR,
    description="Filter toxic or harmful content"
)

# PII detection
pii_rule = create_llm_classifier_rule(
    name="pii_protection",
    model="pii-detector",
    threshold=0.5,
    classification_target="pii",
    severity=Severity.SEVERE,
    description="Prevent leakage of personally identifiable information"
)

# Custom classification
custom_rule = create_llm_classifier_rule(
    name="brand_compliance",
    model="brand-classifier",
    threshold=0.8,
    classification_target="brand_violation",
    severity=Severity.MAJOR,
    description="Ensure brand compliance in generated content"
)

# Add all rules to engine
for rule in [toxicity_rule, pii_rule, custom_rule]:
    policy_engine.add_rule(rule)
```

### Numerical Threshold Rules

Monitor and enforce numerical thresholds on metrics like confidence scores, processing time, resource usage, etc.

```python
from agentnet.core.policy.rules import create_numerical_threshold_rule

# Confidence threshold
confidence_rule = create_numerical_threshold_rule(
    name="minimum_confidence",
    metric_name="confidence_score",
    threshold=0.8,
    operator="greater_than",  # or "less_than", "equals", "not_equals", etc.
    severity=Severity.MINOR,
    description="Require minimum confidence for responses"
)

# Processing time limit
time_rule = create_numerical_threshold_rule(
    name="processing_time_limit",
    metric_name="processing_time_seconds",
    threshold=30.0,
    operator="less_than",
    severity=Severity.MAJOR,
    description="Limit agent processing time"
)

# Resource usage monitoring
memory_rule = create_numerical_threshold_rule(
    name="memory_usage_limit",
    metric_name="memory_usage_mb",
    threshold=1024,
    operator="less_equal",
    severity=Severity.SEVERE,
    description="Prevent excessive memory usage"
)

policy_engine.add_rule(confidence_rule)
policy_engine.add_rule(time_rule) 
policy_engine.add_rule(memory_rule)

# Usage example
metrics_context = {
    "confidence_score": 0.65,  # Below threshold
    "processing_time_seconds": 25.0,  # Within limit
    "memory_usage_mb": 800  # Within limit
}

result = policy_engine.evaluate(metrics_context, tags=["numerical", "threshold"])
```

## Security Isolation

Enhanced security isolation provides multiple layers of protection for multi-tenant deployments and sensitive workloads.

### Creating Isolated Sessions

```python
from agentnet.core.auth.middleware import SecurityIsolationManager
from agentnet.core.auth.rbac import RBACManager, Role

# Initialize security components
rbac_manager = RBACManager()
isolation_manager = SecurityIsolationManager()

# Create users with different roles
admin_user = rbac_manager.create_user(
    user_id="admin_001",
    username="admin",
    email="admin@company.com",
    roles=[Role.ADMIN],
    tenant_id="company_tenant"
)

regular_user = rbac_manager.create_user(
    user_id="user_001", 
    username="analyst",
    email="analyst@company.com",
    roles=[Role.TENANT_USER],
    tenant_id="company_tenant"
)

# Create isolated sessions with different levels
admin_session = isolation_manager.create_isolated_session(
    user=admin_user,
    session_id="admin_session_123",
    isolation_level="standard"  # "basic", "standard", "strict"
)

analyst_session = isolation_manager.create_isolated_session(
    user=regular_user,
    session_id="analyst_session_456", 
    isolation_level="strict"  # More restrictive for regular users
)

print(f"Admin resources: {admin_session['resource_access']}")
print(f"Analyst resources: {analyst_session['resource_access']}")
```

### Resource Access Control

```python
# Validate resource access
session_id = "analyst_session_456"

# Check if session can access compute resources
compute_access = isolation_manager.validate_resource_access(
    session_id=session_id,
    resource_type="compute", 
    resource_id="gpu_cluster_1"
)

# Check network access
network_access = isolation_manager.validate_resource_access(
    session_id=session_id,
    resource_type="network",
    resource_id="external_api_endpoint"
)

print(f"Compute access: {compute_access}")
print(f"Network access: {network_access}")
```

### Resource Locking for Coordination

```python
# Acquire exclusive locks on shared resources
lock_acquired = isolation_manager.acquire_resource_lock(
    session_id="analyst_session_456",
    resource_id="shared_database_connection"
)

if lock_acquired:
    print("Resource locked successfully")
    
    # Do work with the resource
    # ...
    
    # Release the lock when done
    isolation_manager.release_resource_lock(
        session_id="analyst_session_456",
        resource_id="shared_database_connection"
    )
else:
    print("Resource is locked by another session")
```

### Session Cleanup

```python
# Clean up session and release all resources
isolation_manager.cleanup_session("analyst_session_456")

# Get isolation system statistics
stats = isolation_manager.get_isolation_stats()
print(f"Active sessions: {stats['active_sessions']}")
print(f"Resource locks: {stats['resource_locks']}")
```

## Tool Governance

Enhanced tool governance provides security validation, custom policies, and runtime monitoring for tool execution.

### Setting Up Tool Governance

```python
from agentnet.tools.executor import ToolExecutor
from agentnet.tools.registry import ToolRegistry
from agentnet.core.policy.engine import PolicyEngine

# Create tool registry and policy engine
registry = ToolRegistry()
policy_engine = PolicyEngine(name="tool_governance")

# Initialize executor with governance enabled
executor = ToolExecutor(
    registry=registry,
    policy_engine=policy_engine,
    # Additional security options
)

print(f"Governance enabled: {executor.governance_enabled}")
print(f"Security checks: {executor.security_checks}")
```

### Custom Tool Validators

```python
# Define custom validation logic
async def financial_data_validator(tool, parameters, context):
    """Custom validator for financial data tools."""
    
    # Check if user has financial data access
    user_roles = context.get("user_roles", [])
    if "financial_analyst" not in user_roles:
        return {
            "valid": False,
            "blocking": True,
            "reason": "User lacks financial data access permissions"
        }
    
    # Validate sensitive parameters
    if parameters.get("include_pii", False):
        return {
            "valid": False,
            "blocking": True,
            "reason": "PII inclusion not allowed for this tool"
        }
    
    # Additional checks...
    return {"valid": True}

# Register validator for specific tools
executor.add_custom_validator("financial_report_generator", financial_data_validator)
executor.add_custom_validator("market_data_analyzer", financial_data_validator)
```

### Governance-Enabled Tool Execution

```python
# Execute tool with full governance validation
async def execute_with_governance_example():
    result = await executor.execute_with_governance(
        tool_name="financial_report_generator",
        parameters={
            "report_type": "quarterly",
            "include_sensitive_data": False,
            "output_format": "pdf"
        },
        user_id="analyst_001",
        context={
            "agent_name": "financial_assistant",
            "user_roles": ["financial_analyst"],
            "session_id": "session_123"
        }
    )
    
    if result.status == ToolStatus.BLOCKED:
        print(f"Tool execution blocked: {result.error_message}")
        print(f"Governance details: {result.metadata.get('governance_validation')}")
    else:
        print(f"Tool executed successfully: {result.data}")

# Run the example
import asyncio
asyncio.run(execute_with_governance_example())
```

## Agent Orchestration

Multi-agent orchestration policies help coordinate complex workflows while maintaining security and compliance.

### Orchestration Policy Setup

```python
from agentnet.core.policy.rules import create_role_rule, create_numerical_threshold_rule

# Define orchestration policies
role_rule = create_role_rule(
    name="allowed_orchestration_roles",
    allowed_roles=["coordinator", "analyst", "executor", "reviewer"],
    severity=Severity.MAJOR,
    description="Only specific roles can participate in orchestration"
)
role_rule.tags.append("orchestration")

trust_rule = create_numerical_threshold_rule(
    name="minimum_trust_level",
    metric_name="agent_trust_level",
    threshold=0.3,
    operator="greater_equal",
    severity=Severity.SEVERE,
    description="Agents must meet minimum trust threshold"
)
trust_rule.tags.append("coordination")

# Add to policy engine
orchestration_engine = PolicyEngine(name="orchestration_policies")
orchestration_engine.add_rule(role_rule)
orchestration_engine.add_rule(trust_rule)
```

### Multi-Agent Coordination

```python
# Define agent pool for a complex task
agents = [
    {
        "name": "planning_agent",
        "role": "coordinator",
        "capabilities": ["task_planning", "resource_allocation"],
        "trust_level": 0.95,
        "security_clearance": "high"
    },
    {
        "name": "research_agent", 
        "role": "analyst",
        "capabilities": ["data_analysis", "research"],
        "trust_level": 0.85,
        "security_clearance": "medium"
    },
    {
        "name": "execution_agent",
        "role": "executor", 
        "capabilities": ["task_execution", "reporting"],
        "trust_level": 0.75,
        "security_clearance": "medium"
    },
    {
        "name": "untrusted_agent",
        "role": "analyzer",
        "capabilities": ["basic_analysis"],
        "trust_level": 0.2,  # Below threshold
        "security_clearance": "low"
    }
]

# Evaluate orchestration policies
coordination_context = {
    "task_type": "sensitive_data_analysis",
    "required_clearance": "medium",
    "coordination_mode": "sequential"
}

orchestration_result = orchestration_engine.evaluate_agent_orchestration(
    agents=agents,
    coordination_context=coordination_context
)

print(f"Allowed agents: {len(orchestration_result['allowed_agents'])}")
print(f"Blocked agents: {len(orchestration_result['blocked_agents'])}")

# Display coordination rules
for rule in orchestration_result['coordination_rules']:
    print(f"Coordination rule: {rule['type']} - {rule['description']}")
```

### Coordination Rule Implementation

```python
# Extract coordination rules and implement them
coordination_rules = orchestration_result['coordination_rules']
allowed_agents = orchestration_result['allowed_agents'] 

# Implement communication order
comm_rule = next((r for r in coordination_rules if r['type'] == 'communication_order'), None)
if comm_rule:
    agent_order = comm_rule['agents']
    print(f"Communication order: {' -> '.join(agent_order)}")

# Implement resource sharing
resource_rule = next((r for r in coordination_rules if r['type'] == 'resource_sharing'), None) 
if resource_rule and resource_rule.get('resource_locks'):
    print("Resource locking enabled for coordination")
    
    # Implement resource coordination logic
    for agent in allowed_agents:
        # Acquire locks before agent execution
        # Execute agent with coordinated resources
        # Release locks after completion
        pass

# Implement supervision for low-trust agents
supervision_rule = next((r for r in coordination_rules if r['type'] == 'supervision_required'), None)
if supervision_rule:
    min_trust = supervision_rule['min_trust_level']
    print(f"Supervision required for agents with trust < {min_trust}")
```

## Practical Examples

### Complete Security Workflow

```python
async def secure_agent_workflow():
    """Complete example of secure agent execution with all features."""
    
    # 1. Setup security and governance
    rbac_manager = RBACManager()
    isolation_manager = SecurityIsolationManager()
    policy_engine = PolicyEngine(name="production_security")
    
    # Add comprehensive policies
    policy_engine.add_rule(create_semantic_similarity_rule("no_repetition", max_similarity=0.8))
    policy_engine.add_rule(create_llm_classifier_rule("toxicity_filter", threshold=0.7, classification_target="toxicity"))
    policy_engine.add_rule(create_numerical_threshold_rule("confidence_check", "confidence", 0.75, "greater_than"))
    
    # 2. Create secure session
    user = rbac_manager.create_user("user123", "analyst", "analyst@company.com", [Role.TENANT_USER], "tenant1")
    session = isolation_manager.create_isolated_session(user, "secure_session", "standard")
    
    # 3. Setup tool governance
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry, policy_engine=policy_engine)
    
    # 4. Execute agent task with full security
    agent_context = {
        "agent_name": "secure_assistant",
        "user_id": user.user_id,
        "session_id": session["session_id"],
        "task_type": "data_analysis"
    }
    
    # Evaluate policies before execution
    policy_result = policy_engine.evaluate(agent_context)
    
    if policy_result.action.value != "allow":
        print(f"Execution blocked by policy: {policy_result.explanation}")
        return
    
    # Execute with governance
    tool_result = await executor.execute_with_governance(
        tool_name="data_analyzer",
        parameters={"dataset": "customer_data", "analysis_type": "summary"},
        user_id=user.user_id,
        context=agent_context
    )
    
    print(f"Secure execution completed: {tool_result.status}")
    
    # 5. Cleanup
    isolation_manager.cleanup_session(session["session_id"])

# Run the secure workflow
# asyncio.run(secure_agent_workflow())
```

### Multi-Tenant Isolation Example

```python
def multi_tenant_isolation_example():
    """Example of multi-tenant isolation with different security levels."""
    
    rbac_manager = RBACManager()
    isolation_manager = SecurityIsolationManager()
    
    # Create users in different tenants
    tenant_a_admin = rbac_manager.create_user("admin_a", "admin", "admin@tenant-a.com", [Role.ADMIN], "tenant_a")
    tenant_a_user = rbac_manager.create_user("user_a", "user", "user@tenant-a.com", [Role.TENANT_USER], "tenant_a")
    tenant_b_user = rbac_manager.create_user("user_b", "user", "user@tenant-b.com", [Role.TENANT_USER], "tenant_b")
    
    # Create isolated sessions
    sessions = {}
    for user, level in [(tenant_a_admin, "basic"), (tenant_a_user, "standard"), (tenant_b_user, "strict")]:
        session = isolation_manager.create_isolated_session(user, f"session_{user.user_id}", level)
        sessions[user.user_id] = session
        
        print(f"User {user.username} ({user.tenant_id}) - Isolation: {level}")
        print(f"  Resources: {session['resource_access']}")
        print(f"  Network: {session['network_policy']['outbound_allowed']}")
        print()
    
    # Test cross-tenant isolation
    tenant_a_session = sessions["user_a"]
    tenant_b_session = sessions["user_b"]
    
    print("Cross-tenant access policies:")
    print(f"Tenant A cross-tenant access: {tenant_a_session['data_access_policy']['cross_tenant_access']}")
    print(f"Tenant B cross-tenant access: {tenant_b_session['data_access_policy']['cross_tenant_access']}")
    
    # Cleanup all sessions
    for user_id in sessions:
        isolation_manager.cleanup_session(f"session_{user_id}")

multi_tenant_isolation_example()
```

## Best Practices

1. **Layer Security**: Use multiple security layers - authentication, authorization, isolation, and policy enforcement.

2. **Least Privilege**: Grant minimum necessary permissions and resources to each user and agent.

3. **Monitor Everything**: Enable comprehensive logging and monitoring for all security and governance events.

4. **Regular Review**: Periodically review and update policies, rules, and security configurations.

5. **Test Thoroughly**: Test security features under various scenarios including edge cases and adversarial conditions.

6. **Documentation**: Maintain clear documentation of security policies, governance rules, and operational procedures.

7. **Incident Response**: Have clear procedures for handling security violations and policy breaches.

This completes the comprehensive security and governance workflow documentation for AgentNet's enhanced features.