"""
Demonstration of Phase 1 + Phase 2 Integration

Shows the complete flow:
1. Turn-based orchestration with policy enforcement
2. Event system capturing all interactions  
3. Critique and revision loops
4. Structured analyst vs critic debates
5. Multi-strategy arbitration

This demonstrates the core value proposition of AgentNet's governed multi-agent reasoning.
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import AgentNet components
from agentnet import ExampleEngine
from agentnet.core.agent import AgentNet
from agentnet.core.orchestration.turn_engine import TurnEngine, TurnMode
from agentnet.core.policy.engine import PolicyEngine, PolicyAction
from agentnet.core.policy.rules import create_keyword_rule, create_confidence_rule, Severity
from agentnet.events.bus import EventBus, EventType
from agentnet.events.sinks import ConsoleSink, FileSink
from agentnet.critique.evaluator import CritiqueEvaluator, CritiqueType
from agentnet.critique.debate import DebateManager, DebateRole
from agentnet.critique.arbitrator import Arbitrator, ArbitrationStrategy


class DemoAgent:
    """Enhanced demo agent with realistic responses."""
    
    def __init__(self, name: str, role: str, knowledge_domain: str):
        self.name = name
        self.role = role
        self.knowledge_domain = knowledge_domain
        self.engine = ExampleEngine()
        self.response_history = []
    
    async def async_generate_reasoning_tree(self, prompt: str):
        """Generate role-appropriate responses."""
        # Store prompt for context
        self.response_history.append(prompt[:100])
        
        # Generate response based on role
        if self.role == "analyst":
            content = self._generate_analyst_response(prompt)
        elif self.role == "critic":
            content = self._generate_critic_response(prompt)
        elif self.role == "expert":
            content = self._generate_expert_response(prompt)
        else:
            content = f"General response from {self.name}: {prompt[:100]}..."
        
        # Vary confidence based on content length and role
        confidence = 0.9 if len(content) > 200 else 0.7
        if self.role == "critic":
            confidence -= 0.1  # Critics are more cautious
        
        return {
            "result": {
                "content": content,
                "confidence": confidence
            },
            "metadata": {
                "agent_role": self.role,
                "knowledge_domain": self.knowledge_domain,
                "response_count": len(self.response_history)
            }
        }
    
    def _generate_analyst_response(self, prompt: str) -> str:
        """Generate analytical response."""
        if "artificial intelligence" in prompt.lower() or "ai" in prompt.lower():
            return (
                f"From an analytical perspective on AI in {self.knowledge_domain}: "
                f"The current state shows significant progress in machine learning capabilities, "
                f"particularly in natural language processing and computer vision. "
                f"Key benefits include automation of complex tasks, enhanced decision-making support, "
                f"and improved efficiency in data processing. "
                f"Evidence from recent studies indicates 40% improvement in diagnostic accuracy "
                f"when AI assists medical professionals. "
                f"The technology appears ready for broader adoption with proper safeguards."
            )
        
        return (
            f"Analytical assessment by {self.name}: "
            f"Based on systematic evaluation of the available data and evidence, "
            f"I observe several key patterns and implications that warrant consideration. "
            f"The analysis suggests multiple factors at play, including technological, "
            f"economic, and social dimensions that interconnect in complex ways."
        )
    
    def _generate_critic_response(self, prompt: str) -> str:
        """Generate critical response."""
        if "artificial intelligence" in prompt.lower() or "ai" in prompt.lower():
            return (
                f"Critical evaluation of AI in {self.knowledge_domain}: "
                f"While the potential benefits are significant, we must carefully consider "
                f"the substantial risks and limitations. Privacy concerns are paramount, "
                f"especially with sensitive medical data. Bias in AI systems could "
                f"perpetuate or amplify existing healthcare disparities. "
                f"The 'black box' nature of many AI systems raises questions about "
                f"accountability and transparency in critical decisions. "
                f"We should proceed cautiously with robust oversight mechanisms."
            )
        
        return (
            f"Critical perspective from {self.name}: "
            f"I must raise several concerns about the proposed approach. "
            f"The assumptions underlying this analysis may not hold under scrutiny. "
            f"Potential risks include unintended consequences, implementation challenges, "
            f"and the possibility of oversimplifying complex systems. "
            f"Alternative viewpoints deserve consideration before moving forward."
        )
    
    def _generate_expert_response(self, prompt: str) -> str:
        """Generate expert judgment response."""
        if "winner" in prompt.lower() or "judgment" in prompt.lower():
            return (
                f"Expert judgment: After careful evaluation of both positions, "
                f"the analyst presents a more comprehensive and evidence-based argument. "
                f"While the critic raises valid concerns, the analytical approach "
                f"provides a more balanced assessment of benefits and risks. "
                f"Winner: Analyst. Confidence: 0.8. "
                f"The analytical perspective better addresses the complexity "
                f"of the issue while acknowledging limitations."
            )
        
        return (
            f"Expert opinion from {self.name} ({self.knowledge_domain}): "
            f"Based on extensive experience in this domain, I observe that "
            f"both perspectives have merit. The key is finding the optimal balance "
            f"between innovation and caution, ensuring that progress is sustainable "
            f"and beneficial to all stakeholders involved."
        )


async def demonstrate_phase1_orchestration():
    """Demonstrate Phase 1: Turn engine with policy enforcement and events."""
    print("\n" + "="*80)
    print("PHASE 1 DEMONSTRATION: Turn Engine + Policy + Events")
    print("="*80)
    
    # Create event bus with console output
    event_bus = EventBus(name="demo_bus", async_processing=False)
    console_sink = ConsoleSink(
        format_template="[{datetime}] {event_type} | {source} | {data}",
        log_level="INFO"
    )
    event_bus.add_sink(console_sink)
    
    # Create policy engine with rules
    policy = PolicyEngine(name="demo_policy", strict_mode=False)
    
    # Add content quality rules
    policy.add_rule(create_confidence_rule(
        name="min_confidence",
        min_confidence=0.6,
        severity=Severity.MINOR,
        description="Require minimum confidence for responses"
    ))
    
    policy.add_rule(create_keyword_rule(
        name="no_negative_words",
        keywords=["terrible", "awful", "disaster", "catastrophic"],
        severity=Severity.MAJOR,
        description="Avoid overly negative language"
    ))
    
    # Create turn engine with policy and events
    turn_engine = TurnEngine(
        max_turns=6,
        max_rounds=2,
        policy_engine=policy,
        event_callbacks={
            "on_turn_start": lambda session_id, agent_name, turn_num, round_num: 
                event_bus.emit_turn_start(session_id, agent_name, turn_num, round_num),
            "on_turn_end": lambda session_id, agent_name, turn_result:
                event_bus.emit_turn_end(session_id, agent_name, turn_result)
        }
    )
    
    # Create demo agents
    agents = [
        DemoAgent("Dr. Sarah Chen", "analyst", "healthcare"),
        DemoAgent("Prof. Michael Torres", "analyst", "technology"),
        DemoAgent("Dr. Aisha Patel", "critic", "ethics")
    ]
    
    print("\n🚀 Starting multi-agent session with policy enforcement...")
    
    # Execute multi-agent session
    session_result = await turn_engine.execute_multi_agent_session(
        agents=agents,
        topic="The role of artificial intelligence in modern healthcare",
        mode=TurnMode.ROUND_ROBIN,
        context={"demo": True, "domain": "healthcare_ai"}
    )
    
    print(f"\n📊 Session Results:")
    print(f"   Status: {session_result.status.value}")
    print(f"   Duration: {session_result.duration:.2f} seconds")
    print(f"   Total turns: {session_result.total_turns}")
    print(f"   Participants: {', '.join(session_result.agents_involved)}")
    
    # Show policy violations if any
    violations = [turn for turn in session_result.turns if turn.policy_violations]
    if violations:
        print(f"\n⚠️  Policy violations detected in {len(violations)} turns")
        for turn in violations[:2]:  # Show first 2
            print(f"   Agent {turn.agent_id}: {len(turn.policy_violations)} violations")
    else:
        print("\n✅ No policy violations detected")
    
    # Show recent events
    recent_events = event_bus.get_events(limit=5)
    print(f"\n📡 Recent events captured: {len(recent_events)}")
    
    return session_result, event_bus, policy


async def demonstrate_phase2_critique_system(session_result):
    """Demonstrate Phase 2: Critique system with revision loops."""
    print("\n" + "="*80)
    print("PHASE 2 DEMONSTRATION: Critique and Revision System")
    print("="*80)
    
    # Create critique evaluator
    critique_evaluator = CritiqueEvaluator(
        name="demo_critic",
        quality_threshold=0.7,
        truthiness_threshold=0.6,
        enable_automated_scoring=True
    )
    
    print("\n🔍 Analyzing agent responses for quality...")
    
    # Critique each turn from the session
    critiques = []
    for turn in session_result.turns[:3]:  # Analyze first 3 turns
        critique = await critique_evaluator.evaluate_content(
            content=turn.content,
            context={
                "confidence": turn.confidence,
                "agent_name": turn.agent_id,
                "agent_role": turn.metadata.get("agent_role", "unknown")
            },
            critique_type=CritiqueType.AUTOMATED_CRITIQUE,
            critiqued_by="demo_system"
        )
        
        critiques.append((turn.agent_id, critique))
        
        print(f"\n📝 Critique for {turn.agent_id}:")
        print(f"   Overall Score: {critique.overall_score:.2f}")
        print(f"   Quality: {critique.quality_score:.2f} | Truthiness: {critique.truthiness_score:.2f}")
        print(f"   Coherence: {critique.coherence_score:.2f} | Completeness: {critique.completeness_score:.2f}")
        print(f"   Needs Revision: {'Yes' if critique.needs_revision else 'No'}")
        
        if critique.needs_revision:
            print(f"   Triggers: {[t.value for t in critique.revision_triggers]}")
    
    # Find the best and worst responses
    best_critique = max(critiques, key=lambda x: x[1].overall_score)
    worst_critique = min(critiques, key=lambda x: x[1].overall_score)
    
    print(f"\n🏆 Best Response: {best_critique[0]} (Score: {best_critique[1].overall_score:.2f})")
    print(f"🔧 Needs Most Improvement: {worst_critique[0]} (Score: {worst_critique[1].overall_score:.2f})")
    
    return critiques


async def demonstrate_phase2_debate_system():
    """Demonstrate Phase 2: Structured debate with arbitration."""
    print("\n" + "="*80)
    print("PHASE 2 DEMONSTRATION: Analyst vs Critic Debate + Arbitration")
    print("="*80)
    
    # Create debate manager
    debate_manager = DebateManager(
        name="demo_debate",
        max_exchanges_per_phase=3,
        enable_position_evolution=True
    )
    
    # Create specialized debate agents
    analyst_agent = DemoAgent("Dr. Innovation", "analyst", "AI_technology")
    critic_agent = DemoAgent("Dr. Caution", "critic", "AI_ethics")
    expert_agent = DemoAgent("Dr. Judge", "expert", "healthcare_policy")
    
    print("\n🎭 Starting structured analyst vs critic debate...")
    
    # Conduct debate
    debate_result = await debate_manager.conduct_analyst_critic_debate(
        topic="Should AI systems be given autonomous decision-making authority in emergency medical situations?",
        analyst_agent=analyst_agent,
        critic_agent=critic_agent,
        context={"scenario": "emergency_medicine", "stakes": "high"},
        rounds=2
    )
    
    print(f"\n📋 Debate Results:")
    print(f"   Topic: {debate_result.topic}")
    print(f"   Duration: {debate_result.duration:.2f} seconds")
    print(f"   Total exchanges: {len(debate_result.exchanges)}")
    print(f"   Phases completed: {len(debate_result.completed_phases)}")
    print(f"   Positions evolved: {sum(1 for p in debate_result.positions.values() if p.revised_times > 0)}")
    
    # Show key exchanges
    print(f"\n💬 Key Exchanges:")
    for i, exchange in enumerate(debate_result.exchanges[:4]):  # Show first 4
        speaker_role = debate_result.participants[exchange.speaker_id].value
        print(f"   {i+1}. [{speaker_role.title()}] {exchange.speaker_id}: {exchange.content[:100]}...")
    
    # Create arbitrator and resolve debate
    print(f"\n⚖️  Arbitrating debate outcome...")
    
    arbitrator = Arbitrator(
        name="demo_arbitrator",
        default_strategy=ArbitrationStrategy.HYBRID,
        expert_agent=expert_agent
    )
    
    arbitration_result = await arbitrator.arbitrate_debate(
        debate_result,
        context={"agent_names": [analyst_agent.name, critic_agent.name]}
    )
    
    print(f"\n🏛️  Arbitration Results:")
    print(f"   Strategy: {arbitration_result.strategy.value}")
    print(f"   Winner: {arbitration_result.winning_agent}")
    print(f"   Confidence: {arbitration_result.confidence:.2f}")
    print(f"   Consensus: {'Yes' if arbitration_result.consensus_achieved else 'No'}")
    print(f"   Reasoning: {arbitration_result.arbitration_reasoning[:200]}...")
    
    return debate_result, arbitration_result


async def demonstrate_integrated_flow():
    """Demonstrate the complete integrated flow."""
    print("\n" + "="*80)
    print("INTEGRATED DEMONSTRATION: Complete AgentNet Flow")
    print("="*80)
    
    print("\n🌟 This demonstration shows AgentNet's core value proposition:")
    print("   1. Governed multi-agent interactions with policy enforcement")
    print("   2. Comprehensive observability through event capture")  
    print("   3. Intelligent critique and quality assessment")
    print("   4. Structured debates for complex reasoning")
    print("   5. Smart arbitration for conflict resolution")
    
    # Run all demonstrations
    session_result, event_bus, policy = await demonstrate_phase1_orchestration()
    critiques = await demonstrate_phase2_critique_system(session_result)
    debate_result, arbitration_result = await demonstrate_phase2_debate_system()
    
    # Show final summary
    print(f"\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"✅ Phase 1 MVP Components:")
    print(f"   • Turn Engine: {session_result.total_turns} turns across {session_result.total_rounds} rounds")
    print(f"   • Policy Engine: {policy.evaluation_count} evaluations, {policy.violation_count} violations")
    print(f"   • Event System: {event_bus.total_events} events captured")
    
    print(f"\n✅ Phase 2 Critique & Debate Components:")
    print(f"   • Critique System: {len(critiques)} responses analyzed")
    print(f"   • Debate System: {len(debate_result.exchanges)} exchanges across {len(debate_result.completed_phases)} phases")
    print(f"   • Arbitration: {arbitration_result.strategy.value} strategy, winner: {arbitration_result.winning_agent}")
    
    avg_quality = sum(c[1].overall_score for c in critiques) / len(critiques)
    print(f"\n📊 Quality Metrics:")
    print(f"   • Average Response Quality: {avg_quality:.2f}")
    print(f"   • Debate Resolution Confidence: {arbitration_result.confidence:.2f}")
    print(f"   • System Observability: {event_bus.total_events} events tracked")
    
    print(f"\n🎯 AgentNet successfully demonstrates:")
    print(f"   ✓ Scalable multi-agent orchestration") 
    print(f"   ✓ Real-time policy enforcement")
    print(f"   ✓ Comprehensive system observability")
    print(f"   ✓ Intelligent quality assessment")
    print(f"   ✓ Structured reasoning through debate")
    print(f"   ✓ Automated conflict resolution")
    
    return {
        "session_result": session_result,
        "critiques": critiques,
        "debate_result": debate_result,
        "arbitration_result": arbitration_result,
        "event_bus_stats": event_bus.get_stats(),
        "policy_stats": policy.get_stats()
    }


async def main():
    """Main demonstration function."""
    print("🚀 AgentNet Phase 1 + Phase 2 Integration Demo")
    print("Demonstrating governed multi-agent reasoning with critique and debate")
    print("="*80)
    
    try:
        results = await demonstrate_integrated_flow()
        
        # Optionally save results to file
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        summary = {
            "demo_type": "phase1_phase2_integration",
            "timestamp": "2024-09-21",
            "components_tested": [
                "TurnEngine", "PolicyEngine", "EventBus", 
                "CritiqueEvaluator", "DebateManager", "Arbitrator"
            ],
            "metrics": {
                "total_turns": results["session_result"].total_turns,
                "total_events": results["event_bus_stats"]["total_events"],
                "debate_exchanges": len(results["debate_result"].exchanges),
                "average_quality": sum(c[1].overall_score for c in results["critiques"]) / len(results["critiques"]),
                "arbitration_confidence": results["arbitration_result"].confidence
            }
        }
        
        with open(output_dir / "integration_demo_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Demo summary saved to: {output_dir / 'integration_demo_summary.json'}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"AgentNet Phase 1 + Phase 2 integration is fully functional!")


if __name__ == "__main__":
    asyncio.run(main())