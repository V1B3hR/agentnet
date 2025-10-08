"""
Enhanced, LLM-driven Phase 7 Advanced Reasoning Engine.

This engine leverages the agent's core inference provider and memory to
execute structured reasoning techniques (chain-of-thought, multi-hop, counterfactual)
and offers hybrid synthesis and adaptive mode selection.

Advanced Improvements Implemented:
- Robust JSON parsing & schema validation
- Retry/backoff on provider failures
- Unified LLM call helper
- Confidence modeling refinements
- Rich metadata / traceability
- Extensible hooks for customization
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Sequence, Tuple, Union

from ..types import BaseReasoning, ReasoningResult, ReasoningType
from ...providers.base import ProviderAdapter
from ...memory.manager import MemoryManager

logger = logging.getLogger("agentnet.reasoning.advanced")

# ---------------------------------------------------------------------------
# Exceptions & Utilities
# ---------------------------------------------------------------------------

class ReasoningJSONError(RuntimeError):
    """Raised when the LLM response cannot be parsed into expected JSON form."""
    def __init__(self, message: str, raw: Optional[str] = None):
        super().__init__(message)
        self.raw = raw or ""


def _safe_json_extract(raw: str) -> Dict[str, Any]:
    """
    Attempt to extract JSON from a raw LLM response.
    Handles:
      - Markdown fences (```json ... ```)
      - Leading/trailing commentary
      - Partial extraneous text before/after JSON braces
    """
    if not raw:
        raise ReasoningJSONError("Empty response when JSON expected.", raw)

    cleaned = raw.strip()

    # Remove common fenced code block markers
    if cleaned.startswith("```"):
        # Strip initial fence line
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    # Heuristic: find first '{' and last '}' to isolate JSON
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ReasoningJSONError("Could not locate JSON object braces.", raw)

    candidate = cleaned[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ReasoningJSONError(f"JSON decoding failed: {e}", raw) from e


def exponential_backoff_delays(
    attempts: int,
    base: float = 0.4,
    factor: float = 2.0,
    jitter: Tuple[float, float] = (0.2, 0.6),
) -> List[float]:
    """
    Generate a list of sleep delays for retry attempts.
    The first attempt has no delay; subsequent ones scale exponentially.
    """
    delays = []
    for i in range(attempts):
        if i == 0:
            delays.append(0.0)
        else:
            exp = base * (factor ** (i - 1))
            j = random.uniform(*jitter)
            delays.append(exp + j)
    return delays


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class AdvancedReasoningModule(BaseReasoning, ABC):
    """
    Base class for advanced reasoning modules that require an LLM provider.
    Provides:
      - Robust LLM JSON prompt method with retries
      - Confidence utility
      - Extension hooks
    """

    def __init__(
        self,
        provider: ProviderAdapter,
        style_weights: Dict[str, float],
        *,
        max_retries: int = 2,
        timeout: Optional[float] = None,
        validate_schema: bool = True,
    ):
        super().__init__(style_weights)
        self.provider = provider
        self.max_retries = max_retries
        self.timeout = timeout
        self.validate_schema = validate_schema

    @abstractmethod
    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        ...

    # ------------------------- Hooks -------------------------

    def postprocess_steps(self, steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """
        Optional refinement hook for chain-of-thought steps.
        Override to implement pruning, merging, or revalidation.
        """
        return steps

    # --------------------- LLM Interaction -------------------

    async def _prompt_llm_json(
        self,
        prompt: str,
        agent_name: str,
        expected_keys: Optional[Sequence[str]] = None,
        *,
        mode: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call provider and parse JSON with robust retry & validation.
        Returns (json_payload, diagnostics).
        diagnostics includes: attempts, parsing_warnings, elapsed
        """
        delays = exponential_backoff_delays(self.max_retries + 1)
        parsing_warnings: List[str] = []
        last_error: Optional[Exception] = None
        start_time = time.time()

        for attempt, delay in enumerate(delays, start=1):
            if delay:
                await asyncio.sleep(delay)

            try:
                coro = self.provider.async_infer(prompt, agent_name=agent_name)
                if self.timeout:
                    response = await asyncio.wait_for(coro, timeout=self.timeout)
                else:
                    response = await coro

                raw_content = getattr(response, "content", "")
                data = _safe_json_extract(raw_content)

                # Simple key presence validation
                if self.validate_schema and expected_keys:
                    missing = [k for k in expected_keys if k not in data]
                    if missing:
                        parsing_warnings.append(
                            f"Missing expected keys {missing} in attempt {attempt}."
                        )

                diagnostics = {
                    "attempts": attempt,
                    "parsing_warnings": parsing_warnings,
                    "elapsed_seconds": round(time.time() - start_time, 3),
                    "mode": mode,
                }
                return data, diagnostics

            except asyncio.TimeoutError as e:
                last_error = e
                parsing_warnings.append(f"Timeout on attempt {attempt}.")
            except ReasoningJSONError as e:
                last_error = e
                parsing_warnings.append(f"JSON parse failure attempt {attempt}: {e}")
            except Exception as e:
                last_error = e
                parsing_warnings.append(f"Unexpected error attempt {attempt}: {e}")

        # All attempts failed
        logger.error(f"LLM JSON prompt failed after {self.max_retries+1} attempts: {last_error}")
        raise ReasoningJSONError(
            f"LLM prompt failed after retries. Last error: {last_error}", getattr(last_error, "raw", None)
        )

    # --------------------- Confidence Support ----------------

    def _blend_confidence(
        self,
        base: float,
        adjustments: Optional[List[float]] = None,
        style_weight_key: Optional[str] = None,
        cap: float = 0.98,
    ) -> float:
        weights_factor = self.style_weights.get(style_weight_key, 1.0) if style_weight_key else 1.0
        raw = base * weights_factor
        if adjustments:
            for adj in adjustments:
                raw *= adj
        return clamp(raw, 0.0, cap)


# ---------------------------------------------------------------------------
# Reasoner Implementations
# ---------------------------------------------------------------------------

class ChainOfThoughtReasoning(AdvancedReasoningModule):
    """Generates and validates a step-by-step chain of thought using an LLM."""

    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        prompt = (
            "Generate a step-by-step reasoning chain for the task below. "
            "Each step should include: step_id (int, sequential), description, justification, and confidence (0-1). "
            "Conclude with a final answer.\n\n"
            f"Task: {task}\n\n"
            "Respond as JSON: {\"steps\": [...], \"final_answer\": \"...\"}"
        )

        json_payload, diagnostics = self._safe_invoke(prompt, agent_name="CoT-Reasoner", expected_keys=["steps", "final_answer"], mode="chain_of_thought")

        raw_steps = json_payload.get("steps", [])
        steps: List[ReasoningStep] = []
        for item in raw_steps:
            try:
                step = ReasoningStep(
                    step_id=int(item.get("step_id", len(steps) + 1)),
                    description=str(item.get("description", "")).strip(),
                    justification=str(item.get("justification", "")).strip(),
                    confidence=clamp(float(item.get("confidence", 0.7))),
                )
            except Exception as e:
                diagnostics["parsing_warnings"].append(f"Failed to parse step: {e}")
                continue
            steps.append(step)

        steps = self.postprocess_steps(steps)

        avg_conf = statistics.mean([s.confidence for s in steps]) if steps else 0.5
        final_conf = self._blend_confidence(avg_conf, style_weight_key="chain_of_thought")

        return ReasoningResult(
            reasoning_type=ReasoningType.DEDUCTIVE,
            content=json_payload.get("final_answer", "No conclusion reached."),
            confidence=final_conf,
            reasoning_steps=[s.description for s in steps],
            metadata={
                "chain_of_thought_steps": [s.to_dict() for s in steps],
                "raw_llm_json": json_payload,
                "diagnostics": diagnostics,
            },
        )

    def _safe_invoke(self, prompt: str, agent_name: str, expected_keys: List[str], mode: str):
        return asyncio.ensure_future(self._prompt_llm_json(prompt, agent_name, expected_keys, mode=mode))


class MultiHopReasoning(AdvancedReasoningModule):
    """Performs multi-hop reasoning by building and traversing a dynamic knowledge graph from agent memory."""

    def __init__(
        self,
        provider: ProviderAdapter,
        style_weights: Dict[str, float],
        memory_manager: MemoryManager,
        **kwargs,
    ):
        super().__init__(provider, style_weights, **kwargs)
        self.memory = memory_manager

    def select_entities(self, entities: List[str], max_count: int = 5) -> List[str]:
        # Override for ranking or dedup logic; simple trimming for now
        seen = set()
        ordered: List[str] = []
        for e in entities:
            e_norm = e.strip()
            if e_norm and e_norm not in seen:
                ordered.append(e_norm)
                seen.add(e_norm)
            if len(ordered) >= max_count:
                break
        return ordered

    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        prompt_entities = (
            "Extract up to 7 key entities or concepts crucial for multi-hop reasoning.\n\n"
            f"Task: {task}\n\n"
            "Respond as JSON: {\"entities\": [\"...\"]}"
        )
        context = context or {}
        json_entities, diag_entities = await self._prompt_llm_json(
            prompt_entities, "MH-EntityExtractor", expected_keys=["entities"], mode="multi_hop"
        )
        entities_raw = json_entities.get("entities", [])
        entities = self.select_entities(entities_raw, max_count=5)

        if not entities:
            return ReasoningResult(
                reasoning_type=ReasoningType.ANALOGICAL,
                content="Could not identify key entities for multi-hop reasoning.",
                confidence=0.3,
                reasoning_steps=[],
                metadata={"raw_llm_json": json_entities, "diagnostics": diag_entities},
            )

        agent_name = context.get("agent_name", "system")

        knowledge_graph_lines: List[str] = ["--- Knowledge Graph ---"]
        aggregated_snippets: List[str] = []
        for entity in entities:
            try:
                retrieved = self.memory.retrieve(query=entity, agent_name=agent_name)
            except Exception as e:
                logger.warning(f"Memory retrieval failed for entity '{entity}': {e}")
                continue
            if retrieved and getattr(retrieved, "entries", None):
                knowledge_graph_lines.append(f"Entity '{entity}' linked to:")
                for entry in retrieved.entries[:3]:
                    snippet = (entry.content or "")[:160].replace("\n", " ")
                    aggregated_snippets.append(snippet)
                    knowledge_graph_lines.append(f"- {snippet}...")
        knowledge_graph = "\n".join(knowledge_graph_lines)

        reasoning_prompt = (
            "Using the knowledge graph and entities, perform multi-hop reasoning to connect relevant facts and derive an answer.\n"
            f"Task: {task}\n\n{knowledge_graph}\n\n"
            "Respond as JSON: {\"reasoning_path\": [\"hop1\", \"hop2\", ...], \"final_answer\": \"...\"}"
        )

        json_reason, diag_reason = await self._prompt_llm_json(
            reasoning_prompt, "MH-Pathfinder", expected_keys=["reasoning_path", "final_answer"], mode="multi_hop"
        )

        path = json_reason.get("reasoning_path", [])
        base_conf = 0.75 if path else 0.4
        confidence = self._blend_confidence(base_conf, style_weight_key="multi_hop")

        return ReasoningResult(
            reasoning_type=ReasoningType.ANALOGICAL,
            content=json_reason.get("final_answer", "No conclusion reached."),
            confidence=confidence,
            reasoning_steps=path,
            metadata={
                "entities": entities,
                "knowledge_graph_excerpt": knowledge_graph[:1200],
                "raw_llm_json": {"entities_extract": json_entities, "reasoning": json_reason},
                "diagnostics": {"entities": diag_entities, "reasoning": diag_reason},
            },
        )


class CounterfactualReasoning(AdvancedReasoningModule):
    """Analyzes a task by generating and evaluating counterfactual 'what if' scenarios."""

    async def reason(self, task: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        prompt = (
            "Generate three distinct counterfactual 'what if' scenarios for the statement/task below. "
            "For each scenario provide: counterfactual (string), evaluation (string), optional plausibility (0-1). "
            "Then synthesize a final answer.\n\n"
            f"Task: {task}\n\n"
            "Respond as JSON: {\"scenarios\": [{\"counterfactual\": \"...\", \"evaluation\": \"...\", \"plausibility\": 0.0}], \"final_answer\": \"...\"}"
        )
        json_payload, diagnostics = await self._prompt_llm_json(
            prompt, "CF-Reasoner", expected_keys=["scenarios", "final_answer"], mode="counterfactual"
        )

        scenarios = json_payload.get("scenarios", []) or []
        # Normalize to exactly 3 scenarios
        if len(scenarios) > 3:
            scenarios = scenarios[:3]
        elif len(scenarios) < 3:
            # Pad (rare case) with placeholders
            for i in range(len(scenarios), 3):
                scenarios.append(
                    {"counterfactual": f"Placeholder scenario {i+1}", "evaluation": "Insufficient data.", "plausibility": 0.3}
                )

        reasoning_steps = []
        plaus_scores = []
        for s in scenarios:
            cf = str(s.get("counterfactual", "")).strip()
            ev = str(s.get("evaluation", "")).strip()
            plaus = s.get("plausibility")
            if isinstance(plaus, (int, float)):
                plaus_scores.append(clamp(float(plaus)))
            reasoning_steps.append(f"What if: {cf} -> {ev}")

        base_conf = 0.8 if scenarios else 0.4
        # If plausibility scores available, blend them
        if plaus_scores:
            avg_plaus = statistics.mean(plaus_scores)
            base_conf = (base_conf + avg_plaus) / 2.0

        confidence = self._blend_confidence(base_conf, style_weight_key="counterfactual")

        return ReasoningResult(
            reasoning_type=ReasoningType.CAUSAL,
            content=json_payload.get("final_answer", "No conclusion reached."),
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            metadata={
                "counterfactual_scenarios": scenarios,
                "raw_llm_json": json_payload,
                "diagnostics": diagnostics,
            },
        )


# ---------------------------------------------------------------------------
# Hybrid & Mode Selection Engine
# ---------------------------------------------------------------------------

class AdvancedReasoningEngine:
    """
    Central engine coordinating LLM-driven advanced reasoning capabilities and hybrid synthesis.
    """

    def __init__(
        self,
        provider: ProviderAdapter,
        style_weights: Dict[str, float],
        memory_manager: MemoryManager,
        *,
        module_kwargs: Optional[Dict[str, Any]] = None,
    ):
        module_kwargs = module_kwargs or {}
        self.provider = provider
        self.style_weights = style_weights
        self.memory_manager = memory_manager

        self.advanced_reasoners: Dict[str, AdvancedReasoningModule] = {
            "chain_of_thought": ChainOfThoughtReasoning(provider, style_weights, **module_kwargs),
            "multi_hop": MultiHopReasoning(provider, style_weights, memory_manager, **module_kwargs),
            "counterfactual": CounterfactualReasoning(provider, style_weights, **module_kwargs),
        }

    async def advanced_reason(
        self,
        task: str,
        reasoning_mode: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        if reasoning_mode not in self.advanced_reasoners:
            raise ValueError(f"Unknown reasoning mode: {reasoning_mode}")
        return await self.advanced_reasoners[reasoning_mode].reason(task, context)

    async def hybrid_reasoning(
        self,
        task: str,
        modes: List[str],
        context: Optional[Dict[str, Any]] = None,
        *,
        synthesis_strategy: Optional[Callable[[List[ReasoningResult]], str]] = None,
    ) -> ReasoningResult:
        """
        Applies multiple reasoning modes and synthesizes results.
        Provides robust handling of partial failures.

        synthesis_strategy: optional custom function producing final string answer from results
        """

        if not modes:
            raise ValueError("At least one mode must be specified for hybrid reasoning.")

        # Deduplicate preserving order
        seen = set()
        unique_modes = []
        for m in modes:
            if m in self.advanced_reasoners and m not in seen:
                unique_modes.append(m)
                seen.add(m)

        if not unique_modes:
            raise ValueError("No valid reasoning modes supplied.")

        results: List[ReasoningResult] = []
        mode_errors: Dict[str, str] = {}

        async def run_mode(mode: str):
            try:
                res = await self.advanced_reason(task, mode, context)
                results.append(res)
            except Exception as e:
                logger.error(f"Mode '{mode}' failed during hybrid reasoning: {e}")
                mode_errors[mode] = str(e)

        # Python 3.11+ TaskGroup usage if available
        if hasattr(asyncio, "TaskGroup"):
            try:
                async with asyncio.TaskGroup() as tg:
                    for m in unique_modes:
                        tg.create_task(run_mode(m))
            except* Exception:
                # Exceptions already captured per-mode
                pass
        else:
            await asyncio.gather(*(run_mode(m) for m in unique_modes))

        if not results:
            raise RuntimeError(f"All reasoning modes failed. Errors: {mode_errors}")

        # Build synthesis context
        synthesis_context = ["--- Results from Different Reasoning Modes ---"]
        for m in unique_modes:
            match = next((r for r in results if r.reasoning_type.name.lower().startswith(m.split('_')[0])), None)
            # Fallback: search by presence in metadata
            mode_res = None
            for r in results:
                # Attach mode inference by metadata or reasoning_type guess
                if m in r.metadata.get("diagnostics", {}).get("mode", ""):
                    mode_res = r
                    break
            mode_res = mode_res or next((r for r in results if m in str(r.metadata)), None)

        # Just systematically list all results
        for res in results:
            synthesis_context.append(f"Mode-Inferred {res.reasoning_type.name}: {res.content} (Confidence: {res.confidence:.2f})")
        for m, err in mode_errors.items():
            synthesis_context.append(f"Mode '{m}' failed: {err}")
        synthesis_text = "\n".join(synthesis_context)

        if synthesis_strategy:
            final_text = synthesis_strategy(results)
        else:
            # Default LLM-based synthesis
            prompt = (
                "You are a synthesis engine combining conclusions from multiple reasoning subsystems. "
                "Unify their insights, reconcile contradictions, and produce a single authoritative answer.\n\n"
                f"Original Task: {task}\n\n"
                f"{synthesis_text}\n\n"
                "Respond as JSON: {\"final_synthesized_answer\": \"...\"}"
            )
            final_text = await self._synthesize(prompt)

        avg_conf = statistics.mean([r.confidence for r in results])
        # Slight boost due to multi-perspective synthesis
        boosted = clamp(avg_conf * 1.1, 0.0, 0.98)

        return ReasoningResult(
            reasoning_type=ReasoningType.HYBRID,
            content=final_text,
            confidence=boosted,
            reasoning_steps=[f"Synthesized from modes: {', '.join(unique_modes)}"],
            metadata={
                "individual_results": [r.__dict__ for r in results],
                "failed_modes": mode_errors,
                "synthesis_context_excerpt": synthesis_text[:1500],
            },
        )

    async def _synthesize(self, prompt: str) -> str:
        try:
            response = await self.provider.async_infer(prompt, agent_name="Hybrid-Synthesizer")
            data = _safe_json_extract(response.content)
            return data.get("final_synthesized_answer", "Synthesis failed.")
        except Exception as e:
            logger.error(f"Synthesis LLM call failed: {e}")
            return "Could not synthesize final answer due to an error."

    async def auto_select_advanced_mode(self, task: str) -> str:
        """
        Auto-selects the most appropriate reasoning mode using an LLM with heuristics fallback.
        Returns a mode string.
        """
        prompt = (
            "Select the best reasoning mode for the task from: chain_of_thought, multi_hop, counterfactual.\n"
            "Guidelines:\n"
            "- chain_of_thought: step-by-step explanation, planning, structured derivation\n"
            "- multi_hop: connecting multiple concepts, relational inference, bridging facts\n"
            "- counterfactual: causality, alternative outcomes, 'what if'\n\n"
            f"Task: \"{task}\"\n\n"
            "Respond as JSON: {\"best_mode\": \"...\", \"confidence\": 0.0}"
        )
        try:
            response = await self.provider.async_infer(prompt, agent_name="Mode-Selector")
            data = _safe_json_extract(response.content)
            candidate = data.get("best_mode")
            if candidate in self.advanced_reasoners:
                return candidate
        except Exception:
            pass

        # Heuristic fallback
        task_lower = task.lower()
        if any(k in task_lower for k in ["what if", "would have", "alternate", "alternative", "counterfactual"]):
            return "counterfactual"
        if any(k in task_lower for k in ["connect", "relationship", "relation", "link", "bridge", "chain"]):
            return "multi_hop"
        if any(k in task_lower for k in ["cause", "causal", "impact", "effect"]):
            return "counterfactual"
        return "chain_of_thought"
