"""Core AgentNet implementation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..memory.manager import MemoryManager
from ..monitors.base import MonitorFn
from ..persistence.agent_state import AgentStateManager
from ..persistence.session import SessionManager
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from .cost.recorder import CostRecorder
from .types import CognitiveFault

if TYPE_CHECKING:
    from ..providers.base import ProviderAdapter

logger = logging.getLogger("agentnet.core")


class AgentNet:
    """Core AgentNet: cognitive agent with style modulation, reasoning graph, persistence, monitors, and async dialogue."""

    def __init__(
        self,
        name: str,
        style: Dict[str, float],
        engine: Optional["ProviderAdapter"] = None,
        monitors: Optional[List[MonitorFn]] = None,
        dialogue_config: Optional[Dict[str, Any]] = None,
        pre_monitors: Optional[List[MonitorFn]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        cost_recorder: Optional[CostRecorder] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize AgentNet instance.

        Args:
            name: Agent name
            style: Style weights dictionary
            engine: Inference engine/provider
            monitors: Post-style monitors (executed after style influence)
            dialogue_config: Dialogue configuration
            pre_monitors: Pre-style monitors (executed before style influence)
            memory_config: Memory system configuration
            tool_registry: Tool registry for available tools
            cost_recorder: Cost recorder for tracking inference costs
            tenant_id: Tenant ID for multi-tenant cost tracking
        """
        self.name = name
        self.style = style
        self.engine = engine
        self.monitors = monitors or []
        self.pre_monitors = pre_monitors or []
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
        self.memory_manager = None
        if memory_config:
            self.memory_manager = MemoryManager(memory_config)

        # Initialize tool system
        self.tool_registry = tool_registry
        self.tool_executor = None
        if tool_registry:
            self.tool_executor = ToolExecutor(tool_registry)

        # Initialize cost tracking
        self.cost_recorder = cost_recorder or CostRecorder()
        self.tenant_id = tenant_id

        # Initialize managers
        self.session_manager = SessionManager()

        logger.info(
            f"AgentNet instance '{name}' initialized with style {style}, "
            f"{len(self.monitors)} monitors, {len(self.pre_monitors)} pre-monitors, "
            f"memory={'enabled' if self.memory_manager else 'disabled'}, "
            f"tools={'enabled' if self.tool_executor else 'disabled'}"
        )

    def register_monitor(self, monitor_fn: MonitorFn, pre_style: bool = False) -> None:
        """
        Register an additional monitor at runtime.

        Args:
            monitor_fn: Monitor function to register
            pre_style: If True, add to pre_monitors (run before style influence).
        """
        if pre_style:
            self.pre_monitors.append(monitor_fn)
            logger.info(f"Registered pre-style monitor for agent '{self.name}'")
        else:
            self.monitors.append(monitor_fn)
            logger.info(f"Registered post-style monitor for agent '{self.name}'")

    def register_monitors(
        self, monitor_fns: List[MonitorFn], pre_style: bool = False
    ) -> None:
        """
        Register multiple monitors at runtime.

        Args:
            monitor_fns: List of monitor functions
            pre_style: If True, add to pre_monitors
        """
        for monitor_fn in monitor_fns:
            self.register_monitor(monitor_fn, pre_style)

    @staticmethod
    def _normalize_engine_result(raw: Any) -> Dict[str, Any]:
        """Normalize engine result to standard format."""
        if isinstance(raw, dict):
            return raw
        return {"content": str(raw), "confidence": 0.8}

    def _run_pre_monitors(self, task: str, result: Dict[str, Any]) -> None:
        """Run pre-style monitors."""
        for monitor_fn in self.pre_monitors:
            try:
                monitor_fn(self, task, result)
            except CognitiveFault:
                raise
            except Exception as e:
                logger.error(f"Pre-monitor failed: {e}")

    def _run_post_monitors(self, task: str, result: Dict[str, Any]) -> None:
        """Run post-style monitors."""
        for monitor_fn in self.monitors:
            try:
                monitor_fn(self, task, result)
            except CognitiveFault:
                raise
            except Exception as e:
                logger.error(f"Post-monitor failed: {e}")

    def _apply_style_influence(
        self, base_result: Dict[str, Any], task: str
    ) -> Dict[str, Any]:
        """Apply style modulation to base result."""
        # Simple style influence - can be enhanced later
        content = base_result.get("content", "")
        confidence = base_result.get("confidence", 0.8)

        # Adjust confidence based on style
        logic_weight = self.style.get("logic", 0.5)
        creativity_weight = self.style.get("creativity", 0.5)
        analytical_weight = self.style.get("analytical", 0.5)

        # Simple style influence calculation
        style_influence = (logic_weight + analytical_weight) / 2.0
        adjusted_confidence = confidence * (0.8 + 0.4 * style_influence)
        adjusted_confidence = min(1.0, max(0.1, adjusted_confidence))

        return {
            **base_result,
            "confidence": adjusted_confidence,
            "style_applied": True,
            "style_influence": style_influence,
        }

    def generate_reasoning_tree(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a reasoning tree for the given task.

        Args:
            task: The task/prompt to reason about
            include_monitor_trace: Whether to include monitor execution traces
            max_depth: Maximum reasoning depth
            confidence_threshold: Minimum confidence threshold
            style_override: Temporary style override

        Returns:
            Reasoning tree dictionary
        """
        start_time = time.time()

        try:
            # Use engine if available, otherwise create simple response
            if self.engine:
                raw_result = self.engine.infer(task, agent_name=self.name)
            else:
                raw_result = {
                    "content": f"[{self.name}] No engine available for task: {task}",
                    "confidence": 0.5,
                }

            base_result = self._normalize_engine_result(raw_result)

            # Run pre-style monitors
            self._run_pre_monitors(task, base_result)

            # Apply style influence
            styled_result = self._apply_style_influence(base_result, task)

            # Run post-style monitors
            self._run_post_monitors(task, styled_result)

            # Calculate runtime first
            runtime = time.time() - start_time

            # Record cost if engine was used
            cost_record = None
            if self.engine:
                try:
                    # Get cost information from engine
                    cost_info = self.engine.get_cost_info(raw_result)

                    # Record cost with cost recorder
                    cost_record = self.cost_recorder.record_inference_cost(
                        provider=cost_info.get("provider", "unknown"),
                        model=cost_info.get("model", "unknown"),
                        result=raw_result,
                        agent_name=self.name,
                        task_id=task[:50],  # Truncate task for ID
                        tenant_id=self.tenant_id,
                        metadata={
                            "runtime": runtime,
                            "confidence": styled_result.get("confidence", 0.0),
                            "style": self.style.copy(),
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to record cost: {e}")

            # Build reasoning tree
            reasoning_tree = {
                "root": f"{self.name}_reasoning",
                "result": styled_result,
                "agent": self.name,
                "task": task,
                "runtime": runtime,
                "timestamp": time.time(),
                "style": self.style.copy(),
                "monitor_trace": [] if include_monitor_trace else None,
                "cost_record": (
                    {
                        "total_cost": cost_record.total_cost if cost_record else 0.0,
                        "provider": cost_record.provider if cost_record else None,
                        "tokens_input": cost_record.tokens_input if cost_record else 0,
                        "tokens_output": (
                            cost_record.tokens_output if cost_record else 0
                        ),
                    }
                    if cost_record
                    else None
                ),
            }

            # Record in interaction history
            self.interaction_history.append(
                {
                    "type": "reasoning_tree",
                    "task": task,
                    "result": styled_result,
                    "runtime": runtime,
                }
            )

            return reasoning_tree

        except CognitiveFault as cf:
            # Handle cognitive faults
            runtime = time.time() - start_time
            fault_tree = {
                "root": f"{self.name}_fault",
                "result": {
                    "content": f"[{self.name}] Cognitive fault: {cf}",
                    "confidence": 0.1,
                    "fault": True,
                },
                "agent": self.name,
                "task": task,
                "runtime": runtime,
                "timestamp": time.time(),
                "cognitive_fault": cf.to_dict(),
            }

            self.interaction_history.append(
                {
                    "type": "cognitive_fault",
                    "task": task,
                    "fault": cf.to_dict(),
                    "runtime": runtime,
                }
            )

            return fault_tree

    # Persistence methods
    def save_state(self, path: str) -> None:
        """Save agent state to file."""
        AgentStateManager.save_state(self, path)

    def persist_session(self, session_record: dict, directory: str = "sessions") -> str:
        """Persist session record."""
        # Use the session manager if directory differs from default
        if directory != "sessions":
            session_manager = SessionManager(directory)
        else:
            session_manager = self.session_manager

        return session_manager.persist_session(session_record, self.name)

    @classmethod
    def load_state(
        cls, path: str, engine=None, monitors: Optional[List[MonitorFn]] = None
    ) -> "AgentNet":
        """Load agent state from file."""
        return AgentStateManager.load_state(path, cls, engine, monitors)

    @staticmethod
    def from_config(config_path: str | Path, engine=None) -> "AgentNet":
        """Create agent from configuration file (placeholder)."""
        # This would load from a config file - for now return a default agent
        return AgentNet("ConfigAgent", {"logic": 0.7, "creativity": 0.5}, engine=engine)

    def __repr__(self) -> str:
        return f"AgentNet(name='{self.name}', style={self.style})"

    # Memory system methods
    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store content in memory system."""
        if not self.memory_manager:
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
        """Get memory system statistics."""
        if not self.memory_manager:
            return None

        return self.memory_manager.get_memory_stats()

    def clear_memory(self, layer_type: Optional[str] = None) -> bool:
        """Clear memory (all layers or specific layer)."""
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

    # Tool system methods
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a tool and return the result."""
        if not self.tool_executor:
            return None

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

    def list_available_tools(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.tool_registry:
            return []

        specs = self.tool_registry.list_tool_specs(tag)
        return [spec.to_dict() for spec in specs]

    def search_tools(
        self, query: str, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for tools by name or description."""
        if not self.tool_registry:
            return []

        specs = self.tool_registry.search_tools(query, tags)
        return [spec.to_dict() for spec in specs]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        if not self.tool_registry:
            return None

        spec = self.tool_registry.get_tool_spec(tool_name)
        if not spec:
            return None

        return spec.to_dict()

    # Enhanced reasoning with memory and tools
    def generate_reasoning_tree_enhanced(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
        use_memory: bool = True,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate reasoning tree with memory retrieval integration.

        Args:
            task: The task/prompt to reason about
            include_monitor_trace: Whether to include monitor execution traces
            max_depth: Maximum reasoning depth
            confidence_threshold: Minimum confidence threshold
            style_override: Temporary style override
            use_memory: Whether to use memory retrieval
            memory_context: Additional context for memory retrieval

        Returns:
            Enhanced reasoning tree with memory context
        """
        start_time = time.time()

        # Retrieve relevant memories if enabled
        memory_context_str = ""
        memory_retrieval = None

        if use_memory and self.memory_manager:
            memory_retrieval = self.retrieve_memory(task, memory_context)
            if memory_retrieval and memory_retrieval["entries"]:
                memory_items = []
                for entry in memory_retrieval["entries"]:
                    memory_items.append(f"- {entry['content'][:200]}...")

                memory_context_str = f"\n\nRelevant memory context:\n" + "\n".join(
                    memory_items[:3]
                )

        # Enhanced task with memory context
        enhanced_task = task + memory_context_str

        try:
            # Use engine if available, otherwise create simple response
            if self.engine:
                raw_result = self.engine.infer(enhanced_task, agent_name=self.name)
            else:
                raw_result = {
                    "content": f"[{self.name}] No engine available for task: {enhanced_task}",
                    "confidence": 0.5,
                }

            base_result = self._normalize_engine_result(raw_result)

            # Run pre-style monitors
            self._run_pre_monitors(enhanced_task, base_result)

            # Apply style influence
            styled_result = self._apply_style_influence(base_result, enhanced_task)

            # Run post-style monitors
            self._run_post_monitors(enhanced_task, styled_result)

            # Store in memory if significant result
            if self.memory_manager and styled_result.get("confidence", 0) > 0.7:
                tags = ["reasoning", "high_confidence"]
                if memory_context and "tags" in memory_context:
                    tags.extend(memory_context["tags"])

                self.store_memory(
                    styled_result["content"],
                    metadata={
                        "task": task,
                        "confidence": styled_result.get("confidence"),
                    },
                    tags=tags,
                )

            # Build enhanced reasoning tree
            runtime = time.time() - start_time
            reasoning_tree = {
                "root": f"{self.name}_reasoning_enhanced",
                "result": styled_result,
                "agent": self.name,
                "task": task,
                "enhanced_task": enhanced_task,
                "runtime": runtime,
                "timestamp": time.time(),
                "style": self.style.copy(),
                "monitor_trace": [] if include_monitor_trace else None,
                "memory_retrieval": memory_retrieval,
                "memory_used": bool(memory_context_str),
            }

            # Record in interaction history
            self.interaction_history.append(
                {
                    "type": "reasoning_tree_enhanced",
                    "task": task,
                    "result": styled_result,
                    "runtime": runtime,
                    "memory_used": bool(memory_context_str),
                }
            )

            return reasoning_tree

        except CognitiveFault as cf:
            # Handle cognitive faults
            runtime = time.time() - start_time
            fault_tree = {
                "root": f"{self.name}_fault_enhanced",
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
                "cognitive_fault": cf.to_dict(),
                "memory_retrieval": memory_retrieval,
            }

            self.interaction_history.append(
                {
                    "type": "fault_enhanced",
                    "task": task,
                    "fault": cf.to_dict(),
                    "runtime": runtime,
                }
            )

            return fault_tree

        except Exception as e:
            # Handle unexpected errors
            runtime = time.time() - start_time
            error_tree = {
                "root": f"{self.name}_error_enhanced",
                "result": {
                    "content": f"[{self.name}] Unexpected error: {e}",
                    "confidence": 0.0,
                    "error": True,
                },
                "agent": self.name,
                "task": task,
                "enhanced_task": enhanced_task,
                "runtime": runtime,
                "timestamp": time.time(),
                "error": str(e),
                "memory_retrieval": memory_retrieval,
            }

            return error_tree
