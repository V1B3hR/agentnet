#!/usr/bin/env python3
"""
Comprehensive demo of roadmap implementation features.

This demo showcases:
1. Provider ecosystem (OpenAI, Anthropic, Azure adapters)
2. Tool governance and lifecycle management
3. Risk register runtime enforcement integration
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_provider_adapters():
    """Demonstrate provider adapter functionality."""
    print("=" * 70)
    print("1. PROVIDER ADAPTER DEMONSTRATION")
    print("=" * 70)
    print()
    
    from agentnet.providers import ProviderAdapter
    from agentnet.providers.openai import OpenAIAdapter
    from agentnet.providers.anthropic import AnthropicAdapter
    from agentnet.providers.azure import AzureOpenAIAdapter
    
    print("✓ Available Provider Adapters:")
    print("  - OpenAI (GPT-4, GPT-3.5)")
    print("  - Anthropic (Claude 3 - Opus, Sonnet, Haiku)")
    print("  - Azure OpenAI (Regional deployments)")
    print()
    
    # Demo OpenAI adapter
    print("📋 OpenAI Adapter Configuration:")
    openai_adapter = OpenAIAdapter(config={
        "api_key": "test-key-not-real",
        "model": "gpt-4"
    })
    info = openai_adapter.get_provider_info()
    print(f"  Name: {info['name']}")
    print(f"  Streaming: {info['supports_streaming']}")
    print(f"  Models: {', '.join(info['models'])}")
    print(f"  Configured: {info['configured']}")
    print()
    
    # Demo cost calculation
    print("💰 Cost Calculation Example:")
    mock_result = {
        "model": "gpt-4",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
    cost_info = openai_adapter.get_cost_info(mock_result)
    print(f"  Tokens: {cost_info['tokens']}")
    print(f"  Estimated Cost: ${cost_info['cost']:.4f}")
    print(f"  Prompt Tokens: {cost_info['breakdown']['prompt_tokens']}")
    print(f"  Completion Tokens: {cost_info['breakdown']['completion_tokens']}")
    print()
    
    # Demo Anthropic adapter
    print("📋 Anthropic Adapter Configuration:")
    anthropic_adapter = AnthropicAdapter(config={
        "api_key": "test-key-not-real",
        "model": "claude-3-opus-20240229"
    })
    info = anthropic_adapter.get_provider_info()
    print(f"  Name: {info['name']}")
    print(f"  Streaming: {info['supports_streaming']}")
    print(f"  Default Model: {info['default_model']}")
    print()
    
    # Demo Azure adapter
    print("📋 Azure OpenAI Adapter Configuration:")
    azure_adapter = AzureOpenAIAdapter(config={
        "api_key": "test-key-not-real",
        "endpoint": "https://example.openai.azure.com/",
        "deployment": "gpt-4"
    })
    info = azure_adapter.get_provider_info()
    print(f"  Name: {info['name']}")
    print(f"  Endpoint: {info['endpoint']}")
    print(f"  Default Deployment: {info['default_deployment']}")
    print()


def demo_tool_governance():
    """Demonstrate tool governance and lifecycle management."""
    print("=" * 70)
    print("2. TOOL GOVERNANCE & LIFECYCLE DEMONSTRATION")
    print("=" * 70)
    print()
    
    from agentnet.tools.governance import (
        ToolGovernanceManager,
        ToolMetadata,
        ToolStatus,
        GovernanceLevel,
    )
    
    manager = ToolGovernanceManager()
    
    # Register a tool in draft status
    print("🔧 Creating new tool in DRAFT status:")
    metadata = ToolMetadata(
        status=ToolStatus.DRAFT,
        governance_level=GovernanceLevel.INTERNAL,
        owner="data-team",
        team="analytics",
        version="1.0.0"
    )
    manager.register_tool_metadata("analytics_query_tool", metadata)
    print("  ✓ Tool 'analytics_query_tool' registered")
    print(f"  Status: {metadata.status.value}")
    print(f"  Governance Level: {metadata.governance_level.value}")
    print(f"  Owner: {metadata.owner}")
    print()
    
    # Progress through lifecycle
    print("📈 Tool Lifecycle Progression:")
    
    # Move to testing
    manager.update_tool_status("analytics_query_tool", ToolStatus.TESTING, updated_by="qa-team")
    print("  ✓ Status: DRAFT → TESTING")
    
    # Approve the tool
    manager.approve_tool("analytics_query_tool", approver="platform-admin", make_active=True)
    print("  ✓ Tool approved and activated")
    
    # Check usage permissions
    can_use, reason = manager.can_use_tool("analytics_query_tool")
    print(f"  ✓ Can use tool: {can_use}")
    print()
    
    # Demo usage tracking
    print("📊 Usage Tracking:")
    for i in range(5):
        manager.track_usage("analytics_query_tool")
    stats = manager.get_usage_stats("analytics_query_tool")
    today = datetime.now().date().isoformat()
    print(f"  Usage today: {stats.get(today, 0)} times")
    print()
    
    # Demo quota enforcement
    print("🚫 Quota Enforcement:")
    restricted_tool = ToolMetadata(
        status=ToolStatus.ACTIVE,
        usage_quota=10,
        daily_quota=5
    )
    manager.register_tool_metadata("restricted_tool", restricted_tool)
    print("  ✓ Created tool with quotas:")
    print(f"    Total quota: {restricted_tool.usage_quota}")
    print(f"    Daily quota: {restricted_tool.daily_quota}")
    
    # Use up some quota
    for i in range(3):
        manager.track_usage("restricted_tool")
    
    can_use, reason = manager.can_use_tool("restricted_tool")
    print(f"  ✓ Can still use tool: {can_use}")
    print()
    
    # Demo deprecation
    print("⚠️  Tool Deprecation:")
    manager.deprecate_tool(
        "analytics_query_tool",
        reason="Replaced by v2 with better performance"
    )
    tool_meta = manager.get_tool_metadata("analytics_query_tool")
    print(f"  Status: {tool_meta.status.value}")
    print(f"  Deprecation date: {tool_meta.deprecation_date}")
    print()
    
    # Get audit log
    print("📝 Audit Log (last 5 events):")
    log = manager.get_audit_log(limit=5)
    for entry in log[-5:]:
        print(f"  - {entry['event_type']}: {entry.get('tool_name', 'N/A')}")
    print()


def demo_risk_enforcement():
    """Demonstrate risk register runtime enforcement."""
    print("=" * 70)
    print("3. RISK REGISTER RUNTIME ENFORCEMENT DEMONSTRATION")
    print("=" * 70)
    print()
    
    from agentnet.risk import (
        RiskRegister,
        RiskEvent,
        RiskLevel,
        RiskCategory,
    )
    
    try:
        from agentnet.risk import (
            RiskEnforcementEngine,
            EnforcementRule,
            create_default_enforcement_rules,
        )
    except ImportError:
        print("⚠️  Risk enforcement module not available")
        return
    
    # Initialize risk register and enforcement engine
    risk_register = RiskRegister(storage_dir="/tmp/demo_risk_enforcement")
    engine = RiskEnforcementEngine(risk_register)
    
    print("🛡️  Risk Enforcement Engine initialized")
    print()
    
    # Load default rules
    print("📋 Loading default enforcement rules:")
    default_rules = create_default_enforcement_rules()
    for rule in default_rules:
        engine.add_enforcement_rule(rule)
        print(f"  ✓ {rule.rule_id}: {rule.enforcement_action} on {rule.risk_type}")
    print()
    
    # Show active rules
    active_rules = engine.get_active_rules()
    print(f"🎯 Active Rules: {len(active_rules)}")
    print()
    
    # Simulate a cost spike risk event
    print("💸 Simulating token cost spike event:")
    cost_event = RiskEvent(
        risk_id="cost_spike_001",
        category=RiskCategory.FINANCIAL,
        level=RiskLevel.CRITICAL,
        title="Token cost spike detected",
        description="Usage exceeded budget threshold",
        agent_name="production_agent",
        tenant_id="tenant_123",
        metadata={
            "risk_type": "token_cost_spike",
            "current_cost": 50.0,
            "threshold": 20.0
        }
    )
    print(f"  Risk: {cost_event.title}")
    print(f"  Level: {cost_event.level.value}")
    print(f"  Agent: {cost_event.agent_name}")
    print()
    
    # Check enforcement
    print("⚡ Checking enforcement rules:")
    action = engine.check_and_enforce(cost_event, target="production_agent")
    if action:
        print(f"  ✓ Enforcement triggered!")
        print(f"    Action: {action.action_type}")
        print(f"    Target: {action.target}")
        print(f"    Executed at: {action.executed_at}")
    else:
        print("  ℹ️  No enforcement action triggered (threshold not met)")
    print()
    
    # Show enforcement stats
    print("📊 Enforcement Statistics:")
    stats = engine.get_enforcement_stats()
    print(f"  Total enforcement actions: {stats['total_enforcement_actions']}")
    print(f"  Active rules: {stats['active_rules']}")
    print(f"  Currently blocked targets: {stats['currently_blocked']}")
    print(f"  Currently throttled targets: {stats['currently_throttled']}")
    print()
    
    # Demo callback system
    print("🔔 Callback System:")
    
    callback_triggered = []
    
    def alert_callback(action, event):
        callback_triggered.append(f"Alert for {event.title}")
    
    engine.register_action_callback("alert", alert_callback)
    print("  ✓ Registered custom alert callback")
    print()


def demo_integration():
    """Demonstrate integration of all features."""
    print("=" * 70)
    print("4. INTEGRATED WORKFLOW DEMONSTRATION")
    print("=" * 70)
    print()
    
    print("🔄 Typical Production Workflow:")
    print()
    print("1. Configure provider adapter (OpenAI/Anthropic/Azure)")
    print("   → Select model based on requirements and cost")
    print()
    print("2. Register tools with governance")
    print("   → Set appropriate governance level and quotas")
    print("   → Progress through approval workflow")
    print()
    print("3. Enable risk enforcement")
    print("   → Configure rules for cost, performance, security")
    print("   → Set up callbacks for alerts and actions")
    print()
    print("4. Monitor runtime execution")
    print("   → Track provider costs automatically")
    print("   → Enforce tool usage quotas")
    print("   → Trigger enforcement on risk events")
    print()
    print("5. Audit and compliance")
    print("   → Review tool governance audit log")
    print("   → Check enforcement history")
    print("   → Analyze cost trends")
    print()


def main():
    """Run all demonstrations."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "AGENTNET ROADMAP IMPLEMENTATION DEMO" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        demo_provider_adapters()
        demo_tool_governance()
        demo_risk_enforcement()
        demo_integration()
        
        print("=" * 70)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Features Demonstrated:")
        print("  ✓ Provider adapters (OpenAI, Anthropic, Azure)")
        print("  ✓ Tool governance and lifecycle management")
        print("  ✓ Risk register runtime enforcement")
        print("  ✓ Integrated workflow")
        print()
        print("For more information, see:")
        print("  - docs/RoadmapAgentNet.md (Section 31: Recent Implementation Updates)")
        print("  - tests/test_provider_adapters.py")
        print("  - tests/test_tool_governance.py")
        print("  - tests/test_risk_enforcement.py")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
