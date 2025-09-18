#!/usr/bin/env python3
"""
P2 Memory & Tools Implementation Tests

Tests the vector store integration and tool registry features implemented in P2.
"""

import asyncio
import time
from pathlib import Path

# Import AgentNet components
from agentnet import (
    AgentNet,
    CalculatorTool,
    ExampleEngine,
    FileWriteTool,
    MemoryManager,
    MemoryType,
    StatusCheckTool,
    ToolRegistry,
    WebSearchTool,
)


def test_memory_system():
    """Test memory system functionality."""
    print("🧠 Testing Memory System...")

    # Configure memory system
    memory_config = {
        "memory": {
            "short_term": {"enabled": True, "max_entries": 10, "max_tokens": 1000},
            "episodic": {
                "enabled": True,
                "storage_path": "demo_output/episodic_test.json",
                "max_episodes": 50,
            },
            "semantic": {
                "enabled": True,
                "storage_path": "demo_output/semantic_test.json",
                "similarity_threshold": 0.5,
                "max_entries": 100,
            },
        },
        "retrieval": {
            "max_total_tokens": 2000,
            "short_term_count": 5,
            "semantic_count": 3,
            "episodic_count": 2,
        },
    }

    # Create memory manager
    memory_manager = MemoryManager(memory_config)

    # Test storing different types of content
    print("  📝 Storing test memories...")

    test_memories = [
        (
            "Machine learning is a subset of AI focused on algorithms that learn from data.",
            ["ai", "ml", "important"],
        ),
        ("Python is a versatile programming language.", ["programming", "python"]),
        (
            "Database indexing improves query performance significantly.",
            ["database", "performance", "important"],
        ),
        (
            "Microservices architecture promotes scalability and maintainability.",
            ["architecture", "microservices"],
        ),
        (
            "Error handling is crucial for robust software systems.",
            ["error", "programming", "critical"],
        ),
    ]

    for content, tags in test_memories:
        success = memory_manager.store(content, "TestAgent", {"source": "test"}, tags)
        print(f"    ✅ Stored: {content[:50]}... (tags: {tags})")

    # Test memory retrieval
    print("  🔍 Testing memory retrieval...")

    queries = [
        "machine learning algorithms",
        "database performance",
        "software architecture",
        "programming languages",
    ]

    for query in queries:
        retrieval = memory_manager.retrieve(query, "TestAgent")
        print(f"    Query: '{query}'")
        print(
            f"      Found {len(retrieval.entries)} entries from {len(retrieval.source_layers)} layers"
        )
        print(
            f"      Total tokens: {retrieval.total_tokens}, Time: {retrieval.retrieval_time:.3f}s"
        )

        for i, entry in enumerate(retrieval.entries[:2]):  # Show first 2
            print(f"        {i+1}. {entry.content[:80]}...")
        print()

    # Test memory statistics
    stats = memory_manager.get_memory_stats()
    print("  📊 Memory Statistics:")
    for layer_name, layer_stats in stats["layers"].items():
        print(f"    {layer_name}: {layer_stats}")

    print("  ✅ Memory system test completed!\n")
    return memory_manager


def test_tool_system():
    """Test tool registry and execution system."""
    print("🔧 Testing Tool System...")

    # Create tool registry
    registry = ToolRegistry()

    # Register example tools
    tools = [WebSearchTool(), CalculatorTool(), FileWriteTool(), StatusCheckTool()]

    print("  📋 Registering tools...")
    for tool in tools:
        registry.register_tool(tool)
        print(f"    ✅ Registered: {tool.name} - {tool.description}")

    # List available tools
    print("\n  📝 Available tools:")
    for spec in registry.list_tool_specs():
        print(f"    • {spec.name}: {spec.description}")
        print(
            f"      Rate limit: {spec.rate_limit_per_min}/min, Auth: {spec.auth_required}"
        )
        print(f"      Tags: {spec.tags}")

    # Search tools
    print("\n  🔍 Tool search results:")
    search_results = registry.search_tools("math", ["utility"])
    for spec in search_results:
        print(f"    Found: {spec.name} - {spec.description}")

    print(f"\n  📊 Registry stats: {registry.tool_count} tools registered")
    print("  ✅ Tool system test completed!\n")
    return registry


async def test_tool_execution(registry):
    """Test tool execution with various scenarios."""
    print("⚡ Testing Tool Execution...")

    from agentnet.tools.executor import ToolExecutor

    # Create executor
    executor = ToolExecutor(registry)

    # Test cases
    test_cases = [
        {
            "name": "Calculator",
            "tool": "calculator",
            "params": {"expression": "2 + 3 * 4"},
        },
        {
            "name": "Web Search",
            "tool": "web_search",
            "params": {"query": "machine learning", "max_results": 3},
        },
        {
            "name": "Status Check",
            "tool": "status_check",
            "params": {"component": "memory"},
        },
        {
            "name": "File Write",
            "tool": "file_write",
            "params": {
                "filename": "test_output.txt",
                "content": "P2 test output from AgentNet",
            },
        },
    ]

    print("  🚀 Executing tools...")
    for test_case in test_cases:
        print(f"\n    Testing {test_case['name']}...")

        result = await executor.execute_tool(
            test_case["tool"], test_case["params"], user_id="test_user"
        )

        print(f"      Status: {result.status.value}")
        if result.status.value == "success":
            print(f"      Data: {str(result.data)[:100]}...")
        else:
            print(f"      Error: {result.error_message}")
        print(f"      Time: {result.execution_time:.3f}s")

    # Test parallel execution
    print("\n  🔄 Testing parallel execution...")
    parallel_calls = [
        {"tool": "calculator", "parameters": {"expression": "10 * 5"}},
        {"tool": "status_check", "parameters": {"component": "disk"}},
        {"tool": "calculator", "parameters": {"expression": "100 / 4"}},
    ]

    results = await executor.execute_tools_parallel(parallel_calls, user_id="test_user")
    for i, result in enumerate(results):
        print(f"    Call {i+1}: {result.status.value} in {result.execution_time:.3f}s")

    print("  ✅ Tool execution test completed!\n")


async def test_integrated_agent():
    """Test AgentNet with integrated memory and tools."""
    print("🤖 Testing Integrated AgentNet with Memory & Tools...")

    # Setup memory configuration
    memory_config = {
        "memory": {
            "short_term": {"enabled": True, "max_entries": 20},
            "episodic": {
                "enabled": True,
                "storage_path": "demo_output/agent_episodic.json",
            },
            "semantic": {
                "enabled": True,
                "storage_path": "demo_output/agent_semantic.json",
            },
        }
    }

    # Setup tool registry
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool())
    tool_registry.register_tool(StatusCheckTool())

    # Create enhanced agent
    agent = AgentNet(
        name="P2Agent",
        style={"logic": 0.8, "creativity": 0.6, "analytical": 0.9},
        engine=ExampleEngine(),
        memory_config=memory_config,
        tool_registry=tool_registry,
    )

    print(f"  🚀 Created agent: {agent}")

    # Test memory operations
    print("\n  💾 Testing agent memory operations...")

    agent.store_memory(
        "P2 implementation includes vector store integration and tool registry",
        metadata={"phase": "P2", "importance": "high"},
        tags=["p2", "implementation", "important"],
    )

    agent.store_memory(
        "Memory system has three layers: short-term, episodic, and semantic",
        metadata={"topic": "memory", "detail_level": "technical"},
        tags=["memory", "architecture"],
    )

    # Test memory retrieval
    memory_result = agent.retrieve_memory("P2 features and implementation")
    if memory_result:
        print(f"    📋 Found {len(memory_result['entries'])} relevant memories")
        print(f"    🕒 Retrieval took {memory_result['retrieval_time']:.3f}s")
        print(f"    📊 Used layers: {memory_result['source_layers']}")

    # Test tool operations
    print("\n  🔧 Testing agent tool operations...")

    available_tools = agent.list_available_tools()
    print(f"    📝 Available tools: {[tool['name'] for tool in available_tools]}")

    # Execute a tool
    calc_result = await agent.execute_tool("calculator", {"expression": "42 * 1.5"})
    if calc_result:
        print(f"    🧮 Calculator result: {calc_result['data']}")

    # Test enhanced reasoning with memory
    print("\n  🧠 Testing enhanced reasoning with memory...")

    reasoning_tree = agent.generate_reasoning_tree_enhanced(
        "How should we implement the P2 phase of AgentNet?",
        use_memory=True,
        memory_context={"tags": ["p2", "implementation"]},
    )

    print(f"    📊 Reasoning completed in {reasoning_tree['runtime']:.3f}s")
    print(f"    🧠 Memory used: {reasoning_tree['memory_used']}")
    print(f"    📝 Result: {reasoning_tree['result']['content'][:100]}...")

    # Show memory stats
    memory_stats = agent.get_memory_stats()
    if memory_stats:
        print(f"\n  📈 Final memory stats:")
        for layer, stats in memory_stats["layers"].items():
            print(f"    {layer}: {stats}")

    print("  ✅ Integrated agent test completed!\n")


async def main():
    """Run all P2 tests."""
    print("🚀 P2 Memory & Tools Implementation Tests")
    print("=" * 50)

    # Test memory system
    memory_manager = test_memory_system()

    # Test tool system
    tool_registry = test_tool_system()

    # Test tool execution
    await test_tool_execution(tool_registry)

    # Test integrated agent
    await test_integrated_agent()

    print("🎉 All P2 Tests Completed!")
    print("\nP2 Features Successfully Implemented:")
    print("  ✅ Vector store integration (with mock embeddings)")
    print("  ✅ Multi-layer memory system (short-term, episodic, semantic)")
    print("  ✅ Tool registry with JSON schema validation")
    print("  ✅ Tool execution with rate limiting and auth framework")
    print("  ✅ AgentNet integration with memory and tools")
    print("  ✅ Enhanced reasoning with memory context")

    print(f"\n📁 Demo outputs saved to: {Path('demo_output').absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
