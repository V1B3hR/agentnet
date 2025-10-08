"""
Enhanced, LLM-driven Phase 7 Advanced Reasoning Engine.

This engine leverages the agent's core inference provider and memory to
execute powerful, structured reasoning techniques, including a hybrid mode
with a final synthesis step.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..types import BaseReasoning, ReasoningResult, ReasoningType
from ...providers.base import ProviderAdapter
from ...memory.manager import MemoryManager

logger = logging.getLogger("agentnet.reasoning.advanced")


class StepStatus(str, Enum):
    """The validation status of a reasoning step."""
    VALID = "valid"
    NEEDS_REFINEMENT = "needs_refinement"
    INVALID = "invalid"


@dataclass
class ReasoningStep:
    """A single, validated step in a reasoning chain."""
    step_id: int
    description: str
    justification: str
    status: StepStatus = StepStatus.VALID
    confidence: float = 0.8
    dependencies: List[int] = field(default_factory=list)


class AdvancedReasoningModule(BaseReasoning):
    """Base class for advanced reasoning modules that require an LLM provider."""
    def __init__(self, provider: ProviderAdapter, style_weights: Dict[str, float]):
        super().__init__(style_weights)
        self.provider = provider

    @abstractmethod
    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        pass

    async def _prompt_llm(self, prompt: str, agent_name: str) -> Dict[str, Any]:
        """Helper to call the LLM and parse JSON, with error handling."""
        try:
            response = await self.provider.async_infer(prompt, agent_name=agent_name)
            # Basic cleaning of common LLM artifacts before JSON parsing
            clean_content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_content)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}\nResponse: {response.content}")
            raise ValueError("LLM did not return valid JSON for reasoning step.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}")
            raise


class ChainOfThoughtReasoning(AdvancedReasoningModule):
    """Generates and validates a step-by-step chain of thought using an LLM."""

    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        prompt = (
            f"Generate a step-by-step chain of thought to solve the following task. "
            f"For each step, provide a description and a justification. "
            f"Finally, provide a consolidated final answer.\n\n"
            f"Task: {task}\n\n"
            f"Respond with a single JSON object with two keys: 'steps' (a list of objects, each with 'step_id', 'description', 'justification', and 'confidence') and 'final_answer'."
        )
        
        llm_result = await self._prompt_llm(prompt, "CoT-Reasoner")
        
        steps = [ReasoningStep(**step) for step in llm_result.get("steps", [])]
        
        # Optional: Add a validation/refinement pass here if needed
        
        final_confidence = self._calculate_confidence(
            statistics.mean(s.confidence for s in steps) if steps else 0.5
        )

        return ReasoningResult(
            reasoning_type=ReasoningType.DEDUCTIVE,
            content=llm_result.get("final_answer", "No conclusion reached."),
            confidence=final_confidence,
            reasoning_steps=[s.description for s in steps],
            metadata={"chain_of_thought_steps": [s.__dict__ for s in steps]},
        )


class MultiHopReasoning(AdvancedReasoningModule):
    """Performs multi-hop reasoning by building and traversing a dynamic knowledge graph from agent memory."""

    def __init__(self, provider: ProviderAdapter, style_weights: Dict[str, float], memory_manager: MemoryManager):
        super().__init__(provider, style_weights)
        self.memory = memory_manager

    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        # 1. Extract key entities from the task using the LLM
        entity_prompt = (
            f"Extract the key entities or concepts (max 5) from the following task. "
            f"Respond with a single JSON object with a key 'entities' containing a list of strings.\n\nTask: {task}"
        )
        entities_result = await self._prompt_llm(entity_prompt, "MH-EntityExtractor")
        entities = entities_result.get("entities", [])
        if not entities:
            return ReasoningResult(ReasoningType.ANALOGICAL, "Could not identify key entities for multi-hop reasoning.", 0.3)

        # 2. Retrieve relevant memories for each entity to build a knowledge graph
        knowledge_graph = "--- Knowledge Graph ---\n"
        for entity in entities:
            retrieved = self.memory.retrieve(query=entity, agent_name=context.get("agent_name", "system"))
            if retrieved and retrieved.entries:
                knowledge_graph += f"Entity '{entity}' is connected to memories:\n"
                for entry in retrieved.entries[:3]: # Limit for context size
                    knowledge_graph += f"- {entry.content[:150]}...\n"
        
        # 3. Prompt the LLM to reason over the graph
        reasoning_prompt = (
            f"Based on the following knowledge graph constructed from memory, find the most logical path or "
            f"set of connections to address the task. Explain the path and provide a final answer.\n\n"
            f"Task: {task}\n\n{knowledge_graph}\n\n"
            f"Respond with a single JSON object with keys: 'reasoning_path' (a list of strings explaining the hops) and 'final_answer'."
        )
        
        llm_result = await self._prompt_llm(reasoning_prompt, "MH-Pathfinder")
        
        return ReasoningResult(
            reasoning_type=ReasoningType.ANALOGICAL,
            content=llm_result.get("final_answer", "No conclusion reached."),
            confidence=self._calculate_confidence(0.75 if llm_result.get("reasoning_path") else 0.4),
            reasoning_steps=llm_result.get("reasoning_path", []),
            metadata={"entities": entities, "knowledge_graph_summary": knowledge_graph[:500]},
        )


class CounterfactualReasoning(AdvancedReasoningModule):
    """Analyzes a task by generating and evaluating counterfactual 'what if' scenarios."""

    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        prompt = (
            f"Analyze the following statement or task by generating three distinct counterfactual scenarios ('what if' questions). "
            f"For each scenario, briefly evaluate its likely outcome. "
            f"Finally, synthesize these evaluations to provide a final answer to the original task.\n\n"
            f"Task: {task}\n\n"
            f"Respond with a single JSON object with two keys: 'scenarios' (a list of objects, each with 'counterfactual' and 'evaluation') and 'final_answer'."
        )
        
        llm_result = await self._prompt_llm(prompt, "CF-Reasoner")
        
        scenarios = llm_result.get("scenarios", [])
        reasoning_steps = [f"What if: {s['counterfactual']} -> Evaluation: {s['evaluation']}" for s in scenarios]

        return ReasoningResult(
            reasoning_type=ReasoningType.CAUSAL,
            content=llm_result.get("final_answer", "No conclusion reached."),
            confidence=self._calculate_confidence(0.8 if scenarios else 0.4),
            reasoning_steps=reasoning_steps,
            metadata={"counterfactual_scenarios": scenarios},
        )


class AdvancedReasoningEngine:
    """Central engine coordinating LLM-driven advanced reasoning capabilities."""

    def __init__(self, provider: ProviderAdapter, style_weights: Dict[str, float], memory_manager: MemoryManager):
        self.provider = provider
        self.style_weights = style_weights
        self.memory_manager = memory_manager
        
        self.advanced_reasoners = {
            "chain_of_thought": ChainOfThoughtReasoning(provider, style_weights),
            "multi_hop": MultiHopReasoning(provider, style_weights, memory_manager),
            "counterfactual": CounterfactualReasoning(provider, style_weights),
        }

    async def advanced_reason(self, task: str, reasoning_mode: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        if reasoning_mode not in self.advanced_reasoners:
            raise ValueError(f"Unknown reasoning mode: {reasoning_mode}")
        
        reasoner = self.advanced_reasoners[reasoning_mode]
        return await reasoner.reason(task, context)

    async def hybrid_reasoning(self, task: str, modes: List[str], context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """Applies multiple reasoning modes and synthesizes the results for a more robust answer."""
        
        reasoning_tasks = [
            self.advanced_reason(task, mode, context) for mode in modes if mode in self.advanced_reasoners
        ]
        
        results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)
        
        # --- Synthesis Step ---
        synthesis_context = "--- Results from Different Reasoning Modes ---\n"
        valid_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                synthesis_context += f"Mode '{modes[i]}' failed: {res}\n"
            else:
                synthesis_context += f"Mode '{modes[i]}' Conclusion: {res.content} (Confidence: {res.confidence:.2f})\n"
                valid_results.append(res)

        if not valid_results:
            raise RuntimeError("All reasoning modes failed in hybrid execution.")

        synthesis_prompt = (
            f"You have received conclusions from multiple expert reasoning systems. Your task is to synthesize them into a single, comprehensive, and final answer. "
            f"Identify the consensus view, note any valuable unique insights, and resolve contradictions if possible.\n\n"
            f"Original Task: {task}\n\n{synthesis_context}\n\n"
            f"Respond with a single JSON object with one key: 'final_synthesized_answer'."
        )
        
        synthesis_llm_result = await self._prompt_llm(synthesis_prompt, "Hybrid-Synthesizer")
        
        final_confidence = statistics.mean(res.confidence for res in valid_results) * 1.1 # Boost confidence from synthesis
        
        return ReasoningResult(
            reasoning_type=ReasoningType.HYBRID,
            content=synthesis_llm_result.get("final_synthesized_answer", "Synthesis failed."),
            confidence=min(final_confidence, 0.98), # Cap confidence
            reasoning_steps=[f"Synthesized from modes: {', '.join(modes)}"],
            metadata={"individual_results": [res.__dict__ for res in valid_results]},
        )

    async def _prompt_llm(self, prompt: str, agent_name: str) -> Dict[str, Any]:
        """Wrapper for LLM calls, primarily for the synthesis step."""
        try:
            response = await self.provider.async_infer(prompt, agent_name=agent_name)
            clean_content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"Failed during synthesis LLM call: {e}")
            return {"final_synthesized_answer": "Could not synthesize the final answer due to an error."}

    async def auto_select_advanced_mode(self, task: str) -> str:
        """Auto-selects the most appropriate reasoning mode using an LLM."""
        prompt = (
            f"Analyze the following user task and determine the best advanced reasoning mode to solve it. "
            f"Choose from: 'chain_of_thought', 'multi_hop', 'counterfactual'.\n\n"
            f"- Use 'chain_of_thought' for step-by-step processes, explanations, or planning.\n"
            f"- Use 'multi_hop' for tasks involving finding relationships, connecting concepts, or using deep knowledge.\n"
            f"- Use 'counterfactual' for tasks involving causality, 'what if' scenarios, or exploring alternatives.\n\n"
            f"Task: \"{task}\"\n\n"
            f"Respond with a single JSON object with one key: 'best_mode'."
        )
        
        try:
            llm_result = await self._prompt_llm(prompt, "Mode-Selector")
            selected_mode = llm_result.get("best_mode")
            if selected_mode in self.advanced_reasoners:
                return selected_mode
        except Exception:
            pass # Fallback on failure
            
        # Fallback to heuristic if LLM fails
        task_lower = task.lower()
        if any(word in task_lower for word in ["what if", "would have"]): return "counterfactual"
        if any(word in task_lower for word in ["connect", "relationship"]): return "multi_hop"
        return "chain_of_thought"
