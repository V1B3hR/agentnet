#!/usr/bin/env python3
"""
P2 Features Demo: Memory & Tools Integration

Demonstrates the vector store integration and tool registry capabilities
added in P2, showing how agents can leverage memory and tools for enhanced reasoning.
"""

import asyncio
import json
from pathlib import Path

from agentnet import (
    AgentNet, ExampleEngine, MemoryManager, ToolRegistry,
    WebSearchTool, CalculatorTool, FileWriteTool, StatusCheckTool
)


async def demo_memory_enhanced_reasoning():
    """Demonstrate memory-enhanced reasoning capabilities."""
    print("🧠 Memory-Enhanced Reasoning Demo")
    print("=" * 50)
    
    # Setup memory-enabled agent
    memory_config = {
        "memory": {
            "short_term": {"enabled": True, "max_entries": 20},
            "episodic": {"enabled": True, "storage_path": "demo_output/demo_episodic.json"},
            "semantic": {"enabled": True, "storage_path": "demo_output/demo_semantic.json"}
        }
    }
    
    agent = AgentNet(
        name="MemoryAgent",
        style={"logic": 0.8, "creativity": 0.7, "analytical": 0.9},
        engine=ExampleEngine(),
        memory_config=memory_config
    )
    
    print(f"Created memory-enabled agent: {agent.name}")
    
    # Seed the agent's memory with some knowledge
    knowledge_base = [
        ("AgentNet is a multi-agent reasoning platform with policy governance", ["platform", "architecture"]),
        ("P2 phase introduces vector store integration and tool registry", ["p2", "features", "implementation"]),
        ("Memory system has three layers: short-term, episodic, and semantic", ["memory", "architecture"]),
        ("Tools are executed with rate limiting and authentication", ["tools", "security"]),
        ("Semantic memory uses vector similarity for content retrieval", ["semantic", "vectors", "search"])
    ]
    
    print("\n📚 Seeding agent memory...")
    for content, tags in knowledge_base:
        agent.store_memory(content, {"source": "knowledge_base"}, tags)
        print(f"  Stored: {content[:60]}...")
    
    # Ask questions that should leverage memory
    questions = [
        "What are the key features of P2?",
        "How does the memory system work in AgentNet?",
        "What security measures are implemented for tools?",
        "Explain vector similarity in semantic memory"
    ]
    
    print("\n🤔 Testing memory-enhanced reasoning...")
    for question in questions:
        print(f"\nQ: {question}")
        
        # Get reasoning with memory context
        result = agent.generate_reasoning_tree_enhanced(
            question, 
            use_memory=True,
            memory_context={"tags": ["p2", "memory", "tools"]}
        )
        
        print(f"A: {result['result']['content']}")
        print(f"   Memory used: {result['memory_used']}")
        print(f"   Confidence: {result['result'].get('confidence', 0):.2f}")
        print(f"   Time: {result['runtime']:.3f}s")
    
    # Show memory stats
    stats = agent.get_memory_stats()
    print(f"\n📊 Final memory statistics:")
    for layer, data in stats['layers'].items():
        print(f"  {layer}: {data}")


async def demo_tool_augmented_agents():
    """Demonstrate tool-augmented agent capabilities."""
    print("\n\n🔧 Tool-Augmented Agent Demo")
    print("=" * 50)
    
    # Setup tool registry
    tool_registry = ToolRegistry()
    
    # Register tools
    tools = [
        WebSearchTool(),
        CalculatorTool(),
        FileWriteTool(),
        StatusCheckTool()
    ]
    
    for tool in tools:
        tool_registry.register_tool(tool)
    
    # Create tool-enabled agent
    agent = AgentNet(
        name="ToolAgent",
        style={"logic": 0.9, "creativity": 0.5, "analytical": 0.8},
        engine=ExampleEngine(),
        tool_registry=tool_registry
    )
    
    print(f"Created tool-enabled agent: {agent.name}")
    
    # Show available tools
    tools_list = agent.list_available_tools()
    print(f"\n🛠️  Available tools ({len(tools_list)}):")
    for tool in tools_list:
        print(f"  • {tool['name']}: {tool['description']}")
        print(f"    Rate limit: {tool['rate_limit_per_min']}/min")
    
    # Demonstrate tool usage
    print("\n⚡ Demonstrating tool execution...")
    
    # Calculator tool
    print("\n1. Mathematical calculation:")
    calc_result = await agent.execute_tool("calculator", {"expression": "15 * 8 + 42"})
    if calc_result and calc_result['status'] == 'success':
        print(f"   Expression: {calc_result['data']['expression']}")
        print(f"   Result: {calc_result['data']['result']}")
    
    # Status check tool
    print("\n2. System status check:")
    status_result = await agent.execute_tool("status_check", {"component": "overall"})
    if status_result and status_result['status'] == 'success':
        data = status_result['data']
        print(f"   Component: {data['component']}")
        print(f"   Status: {data['status']}")
        print(f"   Details: {data['details']}")
    
    # File write tool
    print("\n3. File operation:")
    file_content = f"""# P2 Demo Output

Agent: {agent.name}
Timestamp: {agent.generate_reasoning_tree('current time')['timestamp']}

This file was created by the P2 tool system demonstration.

Available tools:
{chr(10).join([f"- {tool['name']}" for tool in tools_list])}
"""
    
    file_result = await agent.execute_tool("file_write", {
        "filename": "p2_demo_output.md",
        "content": file_content
    })
    
    if file_result and file_result['status'] == 'success':
        print(f"   File: {file_result['data']['filename']}")
        print(f"   Path: {file_result['data']['filepath']}")
        print(f"   Size: {file_result['data']['bytes_written']} bytes")
    
    # Tool search
    print("\n🔍 Tool search capabilities:")
    search_results = agent.search_tools("math")
    print(f"   Search 'math': {[t['name'] for t in search_results]}")
    
    search_results = agent.search_tools("system")
    print(f"   Search 'system': {[t['name'] for t in search_results]}")


async def demo_integrated_workflow():
    """Demonstrate integrated memory + tools workflow."""
    print("\n\n🚀 Integrated Memory + Tools Workflow Demo")
    print("=" * 50)
    
    # Setup fully-featured agent
    memory_config = {
        "memory": {
            "short_term": {"enabled": True, "max_entries": 15},
            "episodic": {"enabled": True, "storage_path": "demo_output/workflow_episodic.json"},
            "semantic": {"enabled": True, "storage_path": "demo_output/workflow_semantic.json"}
        }
    }
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool())
    tool_registry.register_tool(FileWriteTool())
    tool_registry.register_tool(StatusCheckTool())
    
    agent = AgentNet(
        name="IntegratedAgent",
        style={"logic": 0.8, "creativity": 0.6, "analytical": 0.9},
        engine=ExampleEngine(),
        memory_config=memory_config,
        tool_registry=tool_registry
    )
    
    print(f"Created integrated agent: {agent.name}")
    print(f"Features: Memory ({agent.memory_manager is not None}), Tools ({agent.tool_executor is not None})")
    
    # Workflow: Analyze system performance
    print("\n📈 Workflow: System Performance Analysis")
    
    # Step 1: Check system status
    print("\n1. Checking system status...")
    status = await agent.execute_tool("status_check", {"component": "overall"})
    if status['status'] == 'success':
        uptime = status['data']['details']['uptime_hours']
        load = status['data']['details']['load_average']
        
        # Store in memory for later reference
        memory_content = f"System status: {uptime:.1f}h uptime, {load} load average"
        agent.store_memory(memory_content, {"type": "system_status"}, ["system", "performance"])
        print(f"   Status: {memory_content}")
    
    # Step 2: Calculate performance metrics
    print("\n2. Calculating performance metrics...")
    if status['status'] == 'success':
        uptime_hours = status['data']['details']['uptime_hours']
        
        # Calculate uptime in days
        calc_result = await agent.execute_tool("calculator", {"expression": f"{uptime_hours} / 24"})
        if calc_result['status'] == 'success':
            uptime_days = calc_result['data']['result']
            
            # Store calculation in memory
            memory_content = f"System uptime: {uptime_days:.1f} days ({uptime_hours:.1f} hours)"
            agent.store_memory(memory_content, {"type": "calculation"}, ["uptime", "metrics"])
            print(f"   Uptime: {uptime_days:.1f} days")
    
    # Step 3: Generate memory-enhanced analysis
    print("\n3. Generating analysis with memory context...")
    analysis = agent.generate_reasoning_tree_enhanced(
        "Provide a system performance analysis based on recent status checks",
        use_memory=True,
        memory_context={"tags": ["system", "performance", "metrics"]}
    )
    
    print(f"   Analysis: {analysis['result']['content']}")
    print(f"   Memory entries used: {len(analysis.get('memory_retrieval', {}).get('entries', []))}")
    
    # Step 4: Generate report file
    print("\n4. Generating performance report...")
    
    # Retrieve all performance-related memories
    perf_memories = agent.retrieve_memory("system performance metrics")
    
    report_content = f"""# System Performance Report
Generated by: {agent.name}
Timestamp: {analysis['timestamp']}

## Analysis
{analysis['result']['content']}

## Memory Context
Retrieved {len(perf_memories['entries']) if perf_memories else 0} relevant memories:
"""
    
    if perf_memories:
        for i, entry in enumerate(perf_memories['entries'], 1):
            report_content += f"\n{i}. {entry['content']}"
    
    report_content += f"""

## Statistics
- Analysis time: {analysis['runtime']:.3f}s
- Memory retrieval time: {perf_memories['retrieval_time'] if perf_memories else 0:.3f}s
- Confidence: {analysis['result'].get('confidence', 0):.2f}
"""
    
    file_result = await agent.execute_tool("file_write", {
        "filename": "performance_report.md",
        "content": report_content
    })
    
    if file_result['status'] == 'success':
        print(f"   Report saved: {file_result['data']['filepath']}")
    
    # Show final statistics
    memory_stats = agent.get_memory_stats()
    print(f"\n📊 Final statistics:")
    print(f"   Memory layers: {list(memory_stats['layers'].keys())}")
    for layer, stats in memory_stats['layers'].items():
        if 'entry_count' in stats:
            print(f"   {layer}: {stats['entry_count']} entries")


async def main():
    """Run all P2 feature demonstrations."""
    print("🎯 AgentNet P2 Features Demonstration")
    print("Memory & Tools Integration")
    print("=" * 60)
    
    # Ensure demo output directory exists
    Path("demo_output").mkdir(exist_ok=True)
    
    # Run demonstrations
    await demo_memory_enhanced_reasoning()
    await demo_tool_augmented_agents()
    await demo_integrated_workflow()
    
    print("\n" + "=" * 60)
    print("🎉 P2 Features Demonstration Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("  ✅ Multi-layer memory system (short-term, episodic, semantic)")
    print("  ✅ Vector-based semantic memory retrieval")
    print("  ✅ Tool registry with JSON schema validation")
    print("  ✅ Rate-limited and authenticated tool execution")
    print("  ✅ Memory-enhanced reasoning")
    print("  ✅ Integrated memory + tools workflows")
    
    print(f"\n📁 Demo files created in: {Path('demo_output').absolute()}")
    
    # List created files
    demo_files = list(Path("demo_output").glob("*"))
    if demo_files:
        print("   Created files:")
        for file in sorted(demo_files):
            if file.is_file():
                print(f"   • {file.name}")


if __name__ == "__main__":
    asyncio.run(main())