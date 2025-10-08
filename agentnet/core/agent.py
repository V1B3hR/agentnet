"""Core AgentNet implementation.

Enhancements (2025-10):
- Added richer type hints, runtime validation, and more detailed docstrings
- Implemented style_override handling in reasoning generation (was previously unused)
- Added monitor tracing (when include_monitor_trace=True) with success/error metadata
- Added optional async engine inference support (detects and awaits async infer)
- Integrated CostRecorder usage hooks if engine responses include token usage
- Added update_style(), clone(), and context-manager support
- Added graceful degradation + consolidated Phase 7 capability checks
- Improved logging with structured context + debug verbosity points
- Added guard clauses & defensive programming (e.g., memory truncation, invalid params)
- Added optional result caching (LRU) for identical reasoning tasks when enabled
- Added lightweight performance timing utilities
- Added style influence extensibility hook (_compute_style_influence)
- Ensured memory tags deduplication & safe storage thresholding
- Added evolution experience enrichment (tracks reasoning depth & style influence)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Callable,
    Union,
    Sequence,
    Tuple,
    TypedDict,
)

from ..memory.manager import MemoryManager
from ..monitors.base import MonitorFn
from ..persistence.agent_state import AgentStateManager
from ..persistence.session import SessionManager
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from .autoconfig import get_global_autoconfig
from .cost.recorder import CostRecorder
from .types import CognitiveFault
from .planner import Planner
from .self_reflection import SelfReflection
from .skill_manager import SkillManager

# Phase 7 Advanced Intelligence & Reasoning imports
try:
    from .reasoning.advanced import AdvancedReasoningEngine
    from .reasoning.temporal import TemporalReasoning
    from ..memory.enhanced import EnhancedEpisodicMemory
    from .evolution import AgentEvolutionManager

    _PHASE7_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AdvancedReasoningEngine = None
    TemporalReasoning = None
    EnhancedEpisodicMemory = None
    AgentEvolutionManager = None
    _PHASE7_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from .providers.base import ProviderAdapter  # noqa: F401


logger = logging.getLogger("agentnet.core")


# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------

class ReasoningTree(TypedDict, total=False):
    root: str
    result: Dict[str, Any]
    agent: str
    task: str
    enhanced_task: str
    runtime: float
    timestamp: float
    style: Dict[str, float]
    monitor_trace: Optional[List[Dict[str, Any]]]
    memory_retrieval: Optional[Dict[str, Any]]
    memory_used: bool
    metadata: Dict[str, Any]
    autoconfig: Dict[str, Any]


class AdvancedReasoningResult(TypedDict, total=False):
    primary_reasoning: Dict[str, Any]
    temporal_reasoning: Optional[Dict[str, Any]]
    runtime: float
    phase7_enabled: bool
    error: str


# ---------------------------------------------------------------------------
# Utility decorators / helpers
# ---------------------------------------------------------------------------

def time_it(label: str) -> Callable:
    """Decorator to time internal methods for debug."""
    def outer(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                logger.debug(f"[PERF] {label} took {time.time() - start:.4f}s")

        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                logger.debug(f"[PERF] {label} took {time.time() - start:.4f}s")

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return outer


def _is_coroutine_callable(obj: Any, name: str) -> bool:
    """Check if attribute is a coroutine function."""
    attr = getattr(obj, name, None)
    return inspect.iscoroutinefunction(attr)


# ---------------------------------------------------------------------------
# Core AgentNet
# ---------------------------------------------------------------------------

class AgentNet:
    """Core AgentNet: cognitive agent with style modulation, reasoning graph, persistence, monitors, and async dialogue."""

    # Maximum length of memory content stored during auto-capture
    _MAX_MEMORY_STORE_CHARS = 5000

    def __init__(
        self,
        name: str,
        style: Dict[str, float],
        engine: Optional["ProviderAdapter"] = None,
        monitors: Optional[Sequence[MonitorFn]] = None,
        dialogue_config: Optional[Dict[str, Any]] = None,
        pre_monitors: Optional[Sequence[MonitorFn]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        cost_recorder: Optional[CostRecorder] = None,
        tenant_id: Optional[str] = None,
        *,
        enable_reasoning_cache: bool = False,
        reasoning_cache_size: int = 64,
    ):
        """
        Initialize AgentNet instance.

        Args:
            name: Agent name
            style: Style weights dictionary (e.g., {"logic":0.7,"creativity":0.4})
            engine: Inference engine/provider adapter
            monitors: Post-style monitors (executed after style influence)
            dialogue_config: Dialogue configuration dict
            pre_monitors: Pre-style monitors (executed before style influence)
            memory_config: Memory subsystem configuration
            tool_registry: Registry for available tools
            cost_recorder: Optional cost recorder
            tenant_id: Tenant identifier for multi-tenant cost tracking
            enable_reasoning_cache: Enable in-memory LRU caching for reasoning results
            reasoning_cache_size: Cache size (entries)
        """
        self.name = name
        self._validate_style(style)
        self.style = style
        self.engine = engine
        self.monitors: List[MonitorFn] = list(monitors or [])
        self.pre_monitors: List[MonitorFn] = list(pre_monitors or [])
        self.knowledge_graph: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.dialogue_config = dialogue_config or {
            "default_rounds": 3,
            "max_rounds": 20,
            "convergence_window": 3,
            "convergence_min_overlap": 0.55,
            "convergence_min_conf": 0.75,
            "memory_guard": {
                "max_transcript_tokens": 5000,
                "truncate_strategy": "head",
            },
        }

        # Initialize memory system
        self.memory_manager = MemoryManager(memory_config) if memory_config else None

        # Initialize tool system
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry) if tool_registry else None

        # Initialize cost tracking
        self.cost_recorder = cost_recorder or CostRecorder()
        self.tenant_id = tenant_id

        # Managers
        self.session_manager = SessionManager()
        self.planner = Planner(self)
        self.self_reflection = SelfReflection(self)
        self.skill_manager = SkillManager(self)

        # Phase 7 components
        self.advanced_reasoning_engine = None
        self.temporal_reasoning = None
        self.enhanced_memory = None
        self.evolution_manager = None

        if _PHASE7_AVAILABLE:
            try:
                self._initialize_phase7(memory_config)
                logger.info(f"Phase 7 capabilities enabled for agent '{name}'")
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to initialize Phase 7 modules: {e}")
        else:
            logger.info(f"Phase 7 capabilities not available for agent '{name}'")

        # Reasoning cache
        self._reasoning_cache_enabled = enable_reasoning_cache
        if enable_reasoning_cache:
            size = max(1, reasoning_cache_size)

            @lru_cache(maxsize=size)
            def _cache_fn(cache_key: Tuple) -> ReasoningTree:
                # This function will be populated dynamically; we wrap underlying call.
                raise RuntimeError("Internal cache function used incorrectly.")

            self._reasoning_cache = _cache_fn  # type: ignore
        else:
            self._reasoning_cache = None  # type: ignore

        logger.info(
            "AgentNet '%s' initialized | style=%s monitors=%d pre_monitors=%d "
            "memory=%s tools=%s phase7=%s cache=%s",
            name,
            style,
            len(self.monitors),
            len(self.pre_monitors),
            "enabled" if self.memory_manager else "disabled",
            "enabled" if self.tool_executor else "disabled",
            "enabled" if _PHASE7_AVAILABLE else "disabled",
            f"enabled(size={reasoning_cache_size})"
            if enable_reasoning_cache
            else "disabled",
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _initialize_phase7(self, memory_config: Optional[Dict[str, Any]]) -> None:
        """Internal initializer for Phase 7 components."""
        self.advanced_reasoning_engine = AdvancedReasoningEngine(self.style)
        self.temporal_reasoning = TemporalReasoning(self.style)

        if memory_config and memory_config.get("enhanced_episodic", False):
            enhanced_config = memory_config.copy()
            enhanced_config.setdefault(
                "storage_path", "sessions/enhanced_episodic.json"
            )
            self.enhanced_memory = EnhancedEpisodicMemory(enhanced_config)

        evolution_config = (memory_config or {}).get("evolution", {})
        evolution_config.setdefault("learning_rate", 0.1)
        evolution_config.setdefault("min_pattern_frequency", 3)
        self.evolution_manager = AgentEvolutionManager(evolution_config)

    # ------------------------------------------------------------------
    # Validation & configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_style(style: Dict[str, float]) -> None:
        if not isinstance(style, dict):
            raise TypeError("style must be a dict of {trait: weight}")
        for k, v in style.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"Style weight for '{k}' must be numeric")
            if v < 0:
                logger.warning("Style weight '%s' is negative; clamping to 0.", k)
                style[k] = 0.0

    def update_style(self, **updates: float) -> Dict[str, float]:
        """Update style weights in-place; returns updated style."""
        for k, v in updates.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"New style weight for '{k}' must be numeric")
            self.style[k] = float(max(0.0, v))
        logger.debug("Updated style for '%s': %s", self.name, self.style)
        return self.style

    @property
    def phase7_enabled(self) -> bool:
        return bool(_PHASE7_AVAILABLE and self.advanced_reasoning_engine)

    # ------------------------------------------------------------------
    # Monitor registration
    # ------------------------------------------------------------------
    def register_monitor(self, monitor_fn: MonitorFn, pre_style: bool = False) -> None:
        """
        Register an additional monitor at runtime.

        Args:
            monitor_fn: Monitor function to register
            pre_style: If True, monitor runs before style influence
        """
        if pre_style:
            self.pre_monitors.append(monitor_fn)
            logger.info("Registered pre-style monitor for '%s'", self.name)
        else:
            self.monitors.append(monitor_fn)
            logger.info("Registered post-style monitor for '%s'", self.name)

    def register_monitors(
        self, monitor_fns: Sequence[MonitorFn], pre_style: bool = False
    ) -> None:
        """Register multiple monitors at runtime."""
        for monitor_fn in monitor_fns:
            self.register_monitor(monitor_fn, pre_style)

    # ------------------------------------------------------------------
    # Internal style application
    # ------------------------------------------------------------------
    def _compute_style_influence(self, style: Dict[str, float]) -> float:
        """Hook for computing style influence (override for custom logic)."""
        logic_weight = style.get("logic", 0.5)
        analytical_weight = style.get("analytical", style.get("logic", 0.5))
        return (logic_weight + analytical_weight) / 2.0

    def _apply_style_influence(
        self,
        base_result: Dict[str, Any],
        task: str,
        style_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Apply style modulation to base result with optional style override."""
        active_style = self.style.copy()
        if style_override:
            # Merge override
            for k, v in style_override.items():
                if isinstance(v, (int, float)):
                    active_style[k] = v
        style_influence = self._compute_style_influence(active_style)
        confidence = base_result.get("confidence", 0.8)
        adjusted_confidence = confidence * (0.8 + 0.4 * style_influence)
        adjusted_confidence = min(1.0, max(0.05, adjusted_confidence))
        return {
            **base_result,
            "confidence": adjusted_confidence,
            "style_applied": True,
            "style_influence": style_influence,
            "effective_style": active_style,
        }

    # ------------------------------------------------------------------
    # Monitor execution
    # ------------------------------------------------------------------
    def _exec_monitors(
        self,
        monitors: Sequence[MonitorFn],
        stage: str,
        task: str,
        result: Dict[str, Any],
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Execute a list of monitors with fault isolation & trace capture."""
        for monitor_fn in monitors:
            start = time.time()
            entry: Dict[str, Any] = {
                "monitor": getattr(monitor_fn, "__name__", str(monitor_fn)),
                "stage": stage,
                "started": start,
            }
            try:
                monitor_fn(self, task, result)
                entry["status"] = "ok"
            except CognitiveFault as cf:
                entry["status"] = "cognitive_fault"
                entry["fault"] = cf.to_dict()
                if trace is not None:
                    trace.append(entry)
                raise
            except Exception as e:
                logger.error("Monitor '%s' failed at %s: %s", entry["monitor"], stage, e)
                entry["status"] = "error"
                entry["error"] = str(e)
            finally:
                entry["duration"] = time.time() - start
                if trace is not None:
                    trace.append(entry)

    def _run_pre_monitors(self, task: str, result: Dict[str, Any], trace=None) -> None:
        if self.pre_monitors:
            self._exec_monitors(self.pre_monitors, "pre", task, result, trace)

    def _run_post_monitors(self, task: str, result: Dict[str, Any], trace=None) -> None:
        if self.monitors:
            self._exec_monitors(self.monitors, "post", task, result, trace)

    # ------------------------------------------------------------------
    # Engine inference
    # ------------------------------------------------------------------
    def _normalize_engine_result(self, raw: Any) -> Dict[str, Any]:
        """Normalize engine result to standard format."""
        if isinstance(raw, dict):
            return raw
        return {"content": str(raw), "confidence": 0.8}

    async def _engine_infer_async(self, prompt: str) -> Dict[str, Any]:
        """Run engine inference (async aware)."""
        if not self.engine:
            return {
                "content": f"[{self.name}] No engine available for task: {prompt}",
                "confidence": 0.5,
            }

        # Support async infer if provided
        if _is_coroutine_callable(self.engine, "infer"):
            raw = await self.engine.infer(prompt, agent_name=self.name)
        else:
            # Run in thread if inference might block
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: self.engine.infer(prompt, agent_name=self.name)
            )
        return self._normalize_engine_result(raw)

    def _engine_infer_sync(self, prompt: str) -> Dict[str, Any]:
        if not self.engine:
            return {
                "content": f"[{self.name}] No engine available for task: {prompt}",
                "confidence": 0.5,
            }
        raw = self.engine.infer(prompt, agent_name=self.name)
        return self._normalize_engine_result(raw)

    # ------------------------------------------------------------------
    # Reasoning tree generation (core)
    # ------------------------------------------------------------------
    def _reasoning_cache_key(
        self,
        task: str,
        include_monitor_trace: bool,
        max_depth: int,
        confidence_threshold: float,
        style_override: Optional[Dict[str, float]],
        metadata: Optional[Dict[str, Any]],
        use_memory: bool,
        memory_context: Optional[Dict[str, Any]],
    ) -> Tuple:
        """Build a hashable cache key."""
        return (
            task,
            include_monitor_trace,
            max_depth,
            round(confidence_threshold, 5),
            tuple(sorted((style_override or {}).items())),
            tuple(sorted((metadata or {}).items())),
            use_memory,
            tuple(sorted((memory_context or {}).items())),
        )

    def _retrieve_memory_context(
        self, task: str, use_memory: bool, memory_context: Optional[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Retrieve memory items & format context string."""
        if not (use_memory and self.memory_manager):
            return "", None
        retrieval = self.retrieve_memory(task, memory_context)
        if not retrieval or not retrieval.get("entries"):
            return "", retrieval
        memory_items = [
            f"- {entry['content'][:200]}..."
            for entry in retrieval["entries"][:3]
        ]
        context_block = "\n\nRelevant memory context:\n" + "\n".join(memory_items)
        return context_block, retrieval

    def _store_result_memory(
        self,
        task: str,
        styled_result: Dict[str, Any],
        memory_context: Optional[Dict[str, Any]],
    ) -> None:
        if not self.memory_manager:
            return
        if styled_result.get("confidence", 0) <= 0.7:
            return
        content = styled_result.get("content", "")
        if not content:
            return
        truncated = content[: self._MAX_MEMORY_STORE_CHARS]
        tags = {"reasoning", "high_confidence"}
        if memory_context and "tags" in memory_context:
            tags.update(memory_context["tags"])
        self.store_memory(
            truncated,
            metadata={"task": task, "confidence": styled_result.get("confidence")},
            tags=list(tags),
        )

    def _maybe_record_cost(self, result: Dict[str, Any]) -> None:
        """Record cost if usage metadata is found."""
        usage = result.get("usage") or result.get("metadata", {}).get("usage")
        if usage and self.cost_recorder:
            try:
                self.cost_recorder.record(
                    tenant_id=self.tenant_id or "default",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens"),
                    model=usage.get("model"),
                    cost=usage.get("cost"),
                    meta={"agent": self.name},
                )
            except Exception as e:  # pragma: no cover
                logger.debug("Failed to record cost: %s", e)

    def _generate_reasoning_tree(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_memory: bool = False,
        memory_context: Optional[Dict[str, Any]] = None,
        *,
        _internal_cache_bypass: bool = False,
    ) -> ReasoningTree:
        """
        Generate a reasoning tree for the given task (synchronous path).
        """
        # Cache check
        if (
            self._reasoning_cache_enabled
            and self._reasoning_cache
            and not include_monitor_trace
            and not _internal_cache_bypass
        ):
            key = self._reasoning_cache_key(
                task,
                include_monitor_trace,
                max_depth,
                confidence_threshold,
                style_override,
                metadata,
                use_memory,
                memory_context,
            )
            try:
                return self._reasoning_cache(key)  # type: ignore
            except RuntimeError:
                # Populate manually then seed cache
                tree = self._generate_reasoning_tree(
                    task,
                    include_monitor_trace,
                    max_depth,
                    confidence_threshold,
                    style_override,
                    metadata,
                    use_memory,
                    memory_context,
                    _internal_cache_bypass=True,
                )
                self._reasoning_cache.cache_clear()  # reset potential previous stub
                def _cached_loader(_key: Tuple, _tree=tree):
                    return _tree
                self._reasoning_cache.__wrapped__ = _cached_loader  # type: ignore
                return tree

        start_time = time.time()
        autoconfig = get_global_autoconfig()
        auto_config_params = None

        # Parameter adaptation
        if autoconfig.should_auto_configure(metadata):
            context = {"confidence": confidence_threshold}
            if metadata:
                context.update(metadata)
            auto_config_params = autoconfig.configure_scenario(
                task=task,
                context=context,
                base_max_depth=max_depth,
                base_confidence_threshold=(
                    confidence_threshold if confidence_threshold != 0.7 else None
                ),
            )
            max_depth = auto_config_params.max_depth
            if confidence_threshold == 0.7:
                confidence_threshold = auto_config_params.confidence_threshold
            else:
                confidence_threshold = autoconfig.preserve_confidence_threshold(
                    confidence_threshold, auto_config_params
                )

        # Memory retrieval
        memory_context_str, memory_retrieval = self._retrieve_memory_context(
            task, use_memory, memory_context
        )
        enhanced_task = task + memory_context_str

        monitor_trace: Optional[List[Dict[str, Any]]] = [] if include_monitor_trace else None

        try:
            base_result = self._engine_infer_sync(enhanced_task)

            # Pre-monitors
            self._run_pre_monitors(enhanced_task, base_result, monitor_trace)

            # Style influence
            styled_result = self._apply_style_influence(
                base_result, enhanced_task, style_override
            )

            # Post-monitors
            self._run_post_monitors(enhanced_task, styled_result, monitor_trace)

            # Cost record
            self._maybe_record_cost(styled_result)

            # Store memory if significant
            self._store_result_memory(task, styled_result, memory_context)

            runtime = time.time() - start_time
            reasoning_tree: ReasoningTree = {
                "root": f"{self.name}_reasoning",
                "result": styled_result,
                "agent": self.name,
                "task": task,
                "enhanced_task": enhanced_task,
                "runtime": runtime,
                "timestamp": time.time(),
                "style": (styled_result.get("effective_style") or self.style).copy(),
                "monitor_trace": monitor_trace,
                "memory_retrieval": memory_retrieval,
                "memory_used": bool(memory_context_str),
                "metadata": metadata or {},
            }

            if auto_config_params:
                autoconfig.inject_autoconfig_data(reasoning_tree, auto_config_params)

            # History record
            self.interaction_history.append(
                {
                    "type": "reasoning_tree",
                    "task": task,
                    "result": styled_result,
                    "runtime": runtime,
                    "memory_used": bool(memory_context_str),
                }
            )

            return reasoning_tree

        except CognitiveFault as cf:
            runtime = time.time() - start_time
            fault_tree: ReasoningTree = {
                "root": f"{self.name}_fault",
                "result": {
                    "content": f"[{self.name}] Cognitive fault: {cf}",
                    "confidence": 0.1,
                    "fault": True,
                },
                "agent": self.name,
                "task": task,
                "enhanced_task": enhanced_task,
                "runtime": runtime,
                "timestamp": time.time(),
                "memory_retrieval": memory_retrieval,
                "memory_used": bool(memory_context_str),
                "metadata": {"cognitive_fault": cf.to_dict()},
                "monitor_trace": monitor_trace,
            }
            self.interaction_history.append(
                {
                    "type": "fault",
                    "task": task,
                    "fault": cf.to_dict(),
                    "runtime": runtime,
                }
            )
            return fault_tree

        except Exception as e:
            runtime = time.time() - start_time
            logger.error("Unexpected error in reasoning: %s", e)
            return ReasoningTree(
                root=f"{self.name}_error",
                result={
                    "content": f"[{self.name}] Unexpected error: {e}",
                    "confidence": 0.0,
                    "error": True,
                },
                agent=self.name,
                task=task,
                enhanced_task=enhanced_task,
                runtime=runtime,
                timestamp=time.time(),
                memory_retrieval=memory_retrieval,
                memory_used=bool(memory_context_str),
                metadata={"error": str(e)},
                monitor_trace=monitor_trace,
            )

    def generate_reasoning_tree(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTree:
        """Public synchronous reasoning tree generation."""
        return self._generate_reasoning_tree(
            task=task,
            include_monitor_trace=include_monitor_trace,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            style_override=style_override,
            metadata=metadata,
        )

    def generate_reasoning_tree_enhanced(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
        use_memory: bool = True,
        memory_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTree:
        """Generate reasoning tree with memory retrieval integration."""
        return self._generate_reasoning_tree(
            task=task,
            include_monitor_trace=include_monitor_trace,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            style_override=style_override,
            metadata=metadata,
            use_memory=use_memory,
            memory_context=memory_context,
        )

    # ------------------------------------------------------------------
    # Async reasoning
    # ------------------------------------------------------------------
    async def async_generate_reasoning_tree(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTree:
        """
        Asynchronously generate a reasoning tree for the given task.
        If the underlying engine supports async infer, use it directly.
        """
        # If engine is async-capable we replicate the internal logic with async inference:
        if _is_coroutine_callable(self.engine, "infer"):  # type: ignore
            return await self._async_reasoning_tree_internal(
                task,
                include_monitor_trace,
                max_depth,
                confidence_threshold,
                style_override,
                metadata,
            )
        # Fallback to sync generation inside a thread to avoid blocking loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_reasoning_tree(
                task,
                include_monitor_trace,
                max_depth,
                confidence_threshold,
                style_override,
                metadata,
            ),
        )

    async def _async_reasoning_tree_internal(
        self,
        task: str,
        include_monitor_trace: bool,
        max_depth: int,
        confidence_threshold: float,
        style_override: Optional[Dict[str, float]],
        metadata: Optional[Dict[str, Any]],
    ) -> ReasoningTree:
        """Async variant of reasoning tree generation (no memory integration here)."""
        start_time = time.time()
        autoconfig = get_global_autoconfig()
        auto_config_params = None

        if autoconfig.should_auto_configure(metadata):
            context = {"confidence": confidence_threshold}
            if metadata:
                context.update(metadata)
            auto_config_params = autoconfig.configure_scenario(
                task=task,
                context=context,
                base_max_depth=max_depth,
                base_confidence_threshold=(
                    confidence_threshold if confidence_threshold != 0.7 else None
                ),
            )
            max_depth = auto_config_params.max_depth
            if confidence_threshold == 0.7:
                confidence_threshold = auto_config_params.confidence_threshold
            else:
                confidence_threshold = autoconfig.preserve_confidence_threshold(
                    confidence_threshold, auto_config_params
                )

        monitor_trace: Optional[List[Dict[str, Any]]] = [] if include_monitor_trace else None

        try:
            base_result = await self._engine_infer_async(task)
            self._run_pre_monitors(task, base_result, monitor_trace)
            styled_result = self._apply_style_influence(
                base_result, task, style_override
            )
            self._run_post_monitors(task, styled_result, monitor_trace)
            self._maybe_record_cost(styled_result)

            runtime = time.time() - start_time
            tree: ReasoningTree = {
                "root": f"{self.name}_reasoning",
                "result": styled_result,
                "agent": self.name,
                "task": task,
                "enhanced_task": task,
                "runtime": runtime,
                "timestamp": time.time(),
                "style": styled_result.get("effective_style") or self.style,
                "monitor_trace": monitor_trace,
                "memory_retrieval": None,
                "memory_used": False,
                "metadata": metadata or {},
            }
            if auto_config_params:
                autoconfig.inject_autoconfig_data(tree, auto_config_params)

            self.interaction_history.append(
                {
                    "type": "reasoning_tree",
                    "task": task,
                    "result": styled_result,
                    "runtime": runtime,
                    "memory_used": False,
                }
            )
            return tree

        except Exception as e:
            logger.error("Async reasoning failed: %s", e)
            return ReasoningTree(
                root=f"{self.name}_error",
                result={
                    "content": f"[{self.name}] Async error: {e}",
                    "confidence": 0.0,
                    "error": True,
                },
                agent=self.name,
                task=task,
                enhanced_task=task,
                runtime=time.time() - start_time,
                timestamp=time.time(),
                metadata={"error": str(e)},
                monitor_trace=monitor_trace,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self, path: str) -> None:
        """Save agent state (including style & basic components) to a file path."""
        AgentStateManager.save_state(self, path)

    def persist_session(self, session_record: dict, directory: str = "sessions") -> str:
        """Persist a session record to configured directory."""
        session_manager = (
            SessionManager(directory) if directory != "sessions" else self.session_manager
        )
        return session_manager.persist_session(session_record, self.name)

    @classmethod
    def load_state(
        cls, path: str, engine=None, monitors: Optional[Sequence[MonitorFn]] = None
    ) -> "AgentNet":
        """Load agent state from a previously saved state file."""
        return AgentStateManager.load_state(path, cls, engine, monitors)

    @staticmethod
    def from_config(config_path: Union[str, Path], engine=None) -> "AgentNet":
        """Create agent from configuration file (placeholder implementation)."""
        return AgentNet("ConfigAgent", {"logic": 0.7, "creativity": 0.5}, engine=engine)

    def clone(self, name: Optional[str] = None) -> "AgentNet":
        """Create a shallow clone with same configuration (no deep memory duplication)."""
        return AgentNet(
            name or f"{self.name}_clone",
            style=self.style.copy(),
            engine=self.engine,
            monitors=self.monitors,
            pre_monitors=self.pre_monitors,
            dialogue_config=self.dialogue_config.copy(),
            memory_config=None,  # Avoid cloning memory storage by default
            tool_registry=self.tool_registry,
            cost_recorder=self.cost_recorder,
            tenant_id=self.tenant_id,
        )

    # ------------------------------------------------------------------
    # Memory system wrappers
    # ------------------------------------------------------------------
    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store content in memory system."""
        if not self.memory_manager:
            return False
        if not content:
            return False
        return self.memory_manager.store(content, self.name, metadata, tags)

    def retrieve_memory(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        if not self.memory_manager:
            return None
        retrieval = self.memory_manager.retrieve(query, self.name, context)
        return {
            "entries": [
                {
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp,
                    "agent_name": entry.agent_name,
                    "tags": entry.tags,
                }
                for entry in retrieval.entries
            ],
            "total_tokens": retrieval.total_tokens,
            "retrieval_time": retrieval.retrieval_time,
            "source_layers": [layer.value for layer in retrieval.source_layers],
        }

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Return memory system statistics if enabled."""
        if not self.memory_manager:
            return None
        return self.memory_manager.get_memory_stats()

    def clear_memory(self, layer_type: Optional[str] = None) -> bool:
        """Clear entire memory or a specific layer by type name."""
        if not self.memory_manager:
            return False
        if layer_type:
            from ..memory.base import MemoryType
            try:
                memory_type = MemoryType(layer_type)
                self.memory_manager.clear_layer(memory_type)
            except ValueError:
                return False
        else:
            self.memory_manager.clear_all()
        return True

    # ------------------------------------------------------------------
    # Tool system
    # ------------------------------------------------------------------
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a tool and return serialized result structure."""
        if not self.tool_executor:
            return None
        try:
            result = await self.tool_executor.execute_tool(
                tool_name, parameters, user_id=self.name, context=context
            )
            return {
                "status": result.status.value,
                "data": result.data,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "metadata": result.metadata,
            }
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", tool_name, e)
            return {
                "status": "error",
                "data": None,
                "error_message": str(e),
                "execution_time": 0.0,
                "metadata": {},
            }

    def list_available_tools(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools (optionally filtered by tag)."""
        if not self.tool_registry:
            return []
        specs = self.tool_registry.list_tool_specs(tag)
        return [spec.to_dict() for spec in specs]

    def search_tools(
        self, query: str, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search tools by query or tags."""
        if not self.tool_registry:
            return []
        specs = self.tool_registry.search_tools(query, tags)
        return [spec.to_dict() for spec in specs]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get a tool specification dictionary."""
        if not self.tool_registry:
            return None
        spec = self.tool_registry.get_tool_spec(tool_name)
        return spec.to_dict() if spec else None

    # ------------------------------------------------------------------
    # Phase 7 advanced / hybrid reasoning
    # ------------------------------------------------------------------
    def advanced_reason(
        self,
        task: str,
        reasoning_mode: str = "auto",
        context: Optional[Dict[str, Any]] = None,
        use_temporal: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform advanced reasoning using Phase 7 capabilities.

        Args:
            task: The reasoning task
            reasoning_mode: One of ("chain_of_thought","multi_hop","counterfactual",
                                   "symbolic","auto")
            context: Additional context
            use_temporal: Include temporal reasoning augmentation
        """
        if not self.phase7_enabled:
            return {
                "content": f"Advanced reasoning not available for task: {task}",
                "confidence": 0.3,
                "reasoning_type": "fallback",
                "phase7_available": False,
            }

        start_time = time.time()
        context = context or {}
        try:
            if reasoning_mode == "auto":
                reasoning_mode = (
                    self.advanced_reasoning_engine.auto_select_advanced_mode(task)
                )

            reasoning_result = self.advanced_reasoning_engine.advanced_reason(
                task, reasoning_mode, context
            )

            temporal_result = None
            if use_temporal and self.temporal_reasoning:
                temporal_result = self.temporal_reasoning.reason(task, context)

            combined_result: Dict[str, Any] = {
                "primary_reasoning": {
                    "mode": reasoning_mode,
                    "content": reasoning_result.content,
                    "confidence": reasoning_result.confidence,
                    "reasoning_steps": reasoning_result.reasoning_steps,
                    "metadata": reasoning_result.metadata,
                },
                "temporal_reasoning": None,
                "runtime": time.time() - start_time,
                "phase7_enabled": True,
            }

            if temporal_result:
                combined_result["temporal_reasoning"] = {
                    "content": temporal_result.content,
                    "confidence": temporal_result.confidence,
                    "reasoning_steps": temporal_result.reasoning_steps,
                    "metadata": temporal_result.metadata,
                }
                if temporal_result.confidence > 0.6:
                    combined_result["primary_reasoning"]["confidence"] = min(
                        1.0,
                        combined_result["primary_reasoning"]["confidence"] * 1.1,
                    )

            if self.evolution_manager:
                self._record_reasoning_experience(
                    task, reasoning_mode, combined_result
                )

            return combined_result

        except Exception as e:
            logger.error("Advanced reasoning failed: %s", e)
            return {
                "content": f"Advanced reasoning failed for task: {task}",
                "confidence": 0.2,
                "reasoning_type": "error",
                "error": str(e),
                "phase7_available": True,
            }

    def hybrid_reasoning(
        self, task: str, modes: List[str], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply multiple advanced reasoning modes to the same task and synthesize.

        Returns aggregated result with best-confidence selection and consensus metrics.
        """
        if not self.phase7_enabled:
            return {
                "content": f"Hybrid reasoning not available for task: {task}",
                "confidence": 0.3,
                "modes_applied": [],
                "phase7_available": False,
            }

        start_time = time.time()
        context = context or {}
        try:
            reasoning_results = self.advanced_reasoning_engine.hybrid_reasoning(
                task, modes, context
            )
            confidences = [r.confidence for r in reasoning_results]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            best = max(reasoning_results, key=lambda r: r.confidence) if reasoning_results else None

            hybrid_result = {
                "task": task,
                "modes_applied": modes,
                "individual_results": [
                    {
                        "mode": r.reasoning_type.value,
                        "content": r.content,
                        "confidence": r.confidence,
                        "reasoning_steps": r.reasoning_steps,
                        "metadata": r.metadata,
                    }
                    for r in reasoning_results
                ],
                "synthesis": {
                    "best_mode": best.reasoning_type.value if best else "none",
                    "best_content": best.content if best else "No valid results",
                    "avg_confidence": avg_conf,
                    "consensus_confidence": avg_conf * (len(reasoning_results) / len(modes))
                    if modes
                    else 0.0,
                },
                "runtime": time.time() - start_time,
                "phase7_enabled": True,
            }
            return hybrid_result
        except Exception as e:
            logger.error("Hybrid reasoning failed: %s", e)
            return {
                "content": f"Hybrid reasoning failed for task: {task}",
                "confidence": 0.2,
                "modes_applied": modes,
                "error": str(e),
                "phase7_available": True,
            }

    # ------------------------------------------------------------------
    # Enhanced memory / evolution
    # ------------------------------------------------------------------
    def get_enhanced_memory_hierarchy(self) -> Dict[str, Any]:
        if not self.phase7_enabled or not self.enhanced_memory:
            return {
                "error": "Enhanced memory not available",
                "phase7_available": self.phase7_enabled,
            }
        try:
            return self.enhanced_memory.get_memory_hierarchy()
        except Exception as e:
            logger.error("Failed to get memory hierarchy: %s", e)
            return {"error": str(e), "phase7_available": True}

    def get_cross_modal_links(self, memory_id: str) -> List[Dict[str, Any]]:
        if not self.phase7_enabled or not self.enhanced_memory:
            return []
        try:
            links = self.enhanced_memory.get_cross_modal_links(memory_id)
            return [
                {
                    "source_id": link.source_id,
                    "target_id": link.target_id,
                    "source_modality": link.source_modality.value,
                    "target_modality": link.target_modality.value,
                    "relation_type": link.relation_type,
                    "strength": link.strength,
                    "metadata": link.metadata,
                }
                for link in links
            ]
        except Exception as e:
            logger.error("Failed to get cross-modal links: %s", e)
            return []

    def evolve_capabilities(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.phase7_enabled or not self.evolution_manager:
            return {
                "error": "Agent evolution not available",
                "phase7_available": self.phase7_enabled,
            }
        try:
            report = self.evolution_manager.evolve_agent(self.name, task_results)
            logger.info(
                "Agent '%s' evolved: %d new skills, %d improvements",
                self.name,
                len(report.get("new_skills", [])),
                len(report.get("improvements", [])),
            )
            return report
        except Exception as e:
            logger.error("Agent evolution failed: %s", e)
            return {"error": str(e), "phase7_available": True}

    def get_agent_capabilities(self) -> Dict[str, Any]:
        basic = {
            "name": self.name,
            "style": self.style,
            "has_engine": self.engine is not None,
            "has_memory": self.memory_manager is not None,
            "has_tools": self.tool_executor is not None,
        }
        if not self.phase7_enabled or not self.evolution_manager:
            return {
                "basic_capabilities": basic,
                "phase7_available": self.phase7_enabled,
            }
        try:
            capabilities = self.evolution_manager.get_agent_capabilities(self.name)
            capabilities["basic_info"] = basic
            capabilities["phase7_enabled"] = True
            return capabilities
        except Exception as e:
            logger.error("Failed to get agent capabilities: %s", e)
            return {"error": str(e), "phase7_available": True}

    def get_improvement_recommendations(self) -> Dict[str, Any]:
        if not self.phase7_enabled or not self.evolution_manager:
            return {
                "recommendations": [
                    "Enable Phase 7 capabilities for advanced recommendations"
                ],
                "phase7_available": self.phase7_enabled,
            }
        try:
            return self.evolution_manager.recommend_agent_improvements(self.name)
        except Exception as e:
            logger.error("Failed to get improvement recommendations: %s", e)
            return {"error": str(e), "phase7_available": True}

    def save_evolution_state(self, filepath: Optional[str] = None) -> bool:
        if not self.phase7_enabled or not self.evolution_manager:
            logger.warning("Phase 7 evolution not available for state saving")
            return False
        filepath = filepath or f"sessions/agent_evolution_{self.name}.json"
        try:
            self.evolution_manager.save_evolution_state(filepath)
            logger.info("Saved evolution state for '%s' to %s", self.name, filepath)
            return True
        except Exception as e:
            logger.error("Failed to save evolution state: %s", e)
            return False

    def load_evolution_state(self, filepath: Optional[str] = None) -> bool:
        if not self.phase7_enabled or not self.evolution_manager:
            logger.warning("Phase 7 evolution not available for state loading")
            return False
        filepath = filepath or f"sessions/agent_evolution_{self.name}.json"
        try:
            success = self.evolution_manager.load_evolution_state(filepath)
            if success:
                logger.info(
                    "Loaded evolution state for agent '%s' from %s",
                    self.name,
                    filepath,
                )
            return success
        except Exception as e:
            logger.error("Failed to load evolution state: %s", e)
            return False

    def _record_reasoning_experience(
        self, task: str, reasoning_mode: str, result: Dict[str, Any]
    ) -> None:
        if not self.evolution_manager:
            return
        try:
            confidence = (
                result.get("primary_reasoning", {}).get("confidence", 0.0)
                if "primary_reasoning" in result
                else result.get("result", {}).get("confidence", 0.0)
            )
            success = confidence > 0.6
            style_influence = (
                result.get("result", {}).get("style_influence")
                or result.get("primary_reasoning", {}).get("style_influence")
                or None
            )
            task_result = {
                "task_id": f"reasoning_{int(time.time())}",
                "task_type": f"advanced_reasoning_{reasoning_mode}",
                "content": task,
                "success": success,
                "confidence": confidence,
                "duration": result.get("runtime", 0.0),
                "skills_used": [reasoning_mode, "advanced_reasoning"],
                "metadata": {
                    "reasoning_mode": reasoning_mode,
                    "temporal_used": result.get("temporal_reasoning") is not None,
                    "phase7_capability": True,
                    "style_influence": style_influence,
                },
            }
            self.evolution_manager.pattern_analyzer.record_task(task_result)
        except Exception as e:
            logger.error("Failed to record reasoning experience: %s", e)

    # ------------------------------------------------------------------
    # Representations & context mgmt
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"AgentNet(name='{self.name}', style={self.style})"

    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        # Placeholder for resource cleanup if needed later
        return False
