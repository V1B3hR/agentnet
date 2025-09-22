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
from .autoconfig import get_global_autoconfig
from .cost.recorder import CostRecorder
from .types import CognitiveFault

# Phase 7 Advanced Intelligence & Reasoning imports
try:
    from ..reasoning.advanced import AdvancedReasoningEngine
    from ..reasoning.temporal import TemporalReasoning
    from ..memory.enhanced import EnhancedEpisodicMemory
    from .evolution import AgentEvolutionManager
    _PHASE7_AVAILABLE = True
except ImportError:
    AdvancedReasoningEngine = None
    TemporalReasoning = None
    EnhancedEpisodicMemory = None
    AgentEvolutionManager = None
    _PHASE7_AVAILABLE = False

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

        # Phase 7: Advanced Intelligence & Reasoning initialization
        self.advanced_reasoning_engine = None
        self.temporal_reasoning = None
        self.enhanced_memory = None
        self.evolution_manager = None
        
        if _PHASE7_AVAILABLE:
            # Initialize advanced reasoning engine
            self.advanced_reasoning_engine = AdvancedReasoningEngine(style)
            
            # Initialize temporal reasoning
            self.temporal_reasoning = TemporalReasoning(style)
            
            # Initialize enhanced episodic memory if memory config supports it
            if memory_config and memory_config.get("enhanced_episodic", False):
                enhanced_config = memory_config.copy()
                enhanced_config["storage_path"] = enhanced_config.get("storage_path", "sessions/enhanced_episodic.json")
                self.enhanced_memory = EnhancedEpisodicMemory(enhanced_config)
            
            # Initialize evolution manager
            evolution_config = memory_config.get("evolution", {}) if memory_config else {}
            evolution_config.setdefault("learning_rate", 0.1)
            evolution_config.setdefault("min_pattern_frequency", 3)
            self.evolution_manager = AgentEvolutionManager(evolution_config)
            
            logger.info(f"Phase 7 capabilities enabled for agent '{name}'")
        else:
            logger.info(f"Phase 7 capabilities not available for agent '{name}'")

        logger.info(
            f"AgentNet instance '{name}' initialized with style {style}, "
            f"{len(self.monitors)} monitors, {len(self.pre_monitors)} pre-monitors, "
            f"memory={'enabled' if self.memory_manager else 'disabled'}, "
            f"tools={'enabled' if self.tool_executor else 'disabled'}, "
            f"phase7={'enabled' if _PHASE7_AVAILABLE else 'disabled'}"
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a reasoning tree for the given task.

        Args:
            task: The task/prompt to reason about
            include_monitor_trace: Whether to include monitor execution traces
            max_depth: Maximum reasoning depth
            confidence_threshold: Minimum confidence threshold
            style_override: Temporary style override
            metadata: Optional metadata that may include auto_config setting

        Returns:
            Reasoning tree dictionary
        """
        start_time = time.time()
        
        # Auto-configure parameters based on task difficulty if enabled
        autoconfig = get_global_autoconfig()
        auto_config_params = None
        
        if autoconfig.should_auto_configure(metadata):
            # Create context from metadata and confidence
            context = {"confidence": confidence_threshold}
            if metadata:
                context.update(metadata)
            
            auto_config_params = autoconfig.configure_scenario(
                task=task,
                context=context,
                base_max_depth=max_depth,
                base_confidence_threshold=confidence_threshold if confidence_threshold != 0.7 else None
            )
            
            # Apply auto-configured parameters
            max_depth = auto_config_params.max_depth
            # For confidence threshold, use auto-config value if it's for default threshold
            if confidence_threshold == 0.7:  # Default threshold
                confidence_threshold = auto_config_params.confidence_threshold
            else:  # User-specified threshold, preserve or raise
                confidence_threshold = autoconfig.preserve_confidence_threshold(
                    confidence_threshold, auto_config_params
                )

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
                "metadata": metadata or {},
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
            
            # Inject auto-configuration data for observability
            if auto_config_params:
                autoconfig.inject_autoconfig_data(reasoning_tree, auto_config_params)

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

    async def async_generate_reasoning_tree(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously generate a reasoning tree for the given task.
        
        This is an async wrapper around generate_reasoning_tree for compatibility
        with async dialogue systems.

        Args:
            task: The task/prompt to reason about
            include_monitor_trace: Whether to include monitor execution traces
            max_depth: Maximum reasoning depth
            confidence_threshold: Minimum confidence threshold
            style_override: Temporary style override
            metadata: Optional metadata that may include auto_config setting

        Returns:
            Reasoning tree dictionary
        """
        # For now, this is a simple async wrapper around the sync method
        # In the future, this could be extended to support truly async inference
        return self.generate_reasoning_tree(
            task=task,
            include_monitor_trace=include_monitor_trace,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            style_override=style_override,
            metadata=metadata,
        )

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
        metadata: Optional[Dict[str, Any]] = None,
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
            metadata: Optional metadata that may include auto_config setting

        Returns:
            Enhanced reasoning tree with memory context
        """
        start_time = time.time()
        
        # Auto-configure parameters based on task difficulty if enabled
        autoconfig = get_global_autoconfig()
        auto_config_params = None
        
        if autoconfig.should_auto_configure(metadata):
            # Create context from metadata and confidence
            context = {"confidence": confidence_threshold}
            if metadata:
                context.update(metadata)
            
            auto_config_params = autoconfig.configure_scenario(
                task=task,
                context=context,
                base_max_depth=max_depth,
                base_confidence_threshold=confidence_threshold if confidence_threshold != 0.7 else None
            )
            
            # Apply auto-configured parameters
            max_depth = auto_config_params.max_depth
            # For confidence threshold, use auto-config value if it's for default threshold
            if confidence_threshold == 0.7:  # Default threshold
                confidence_threshold = auto_config_params.confidence_threshold
            else:  # User-specified threshold, preserve or raise
                confidence_threshold = autoconfig.preserve_confidence_threshold(
                    confidence_threshold, auto_config_params
                )

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
                "metadata": metadata or {},
            }
            
            # Inject auto-configuration data for observability
            if auto_config_params:
                autoconfig.inject_autoconfig_data(reasoning_tree, auto_config_params)

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
    
    # Phase 7: Advanced Intelligence & Reasoning Methods
    
    def advanced_reason(
        self,
        task: str,
        reasoning_mode: str = "auto",
        context: Optional[Dict[str, Any]] = None,
        use_temporal: bool = False
    ) -> Dict[str, Any]:
        """
        Perform advanced reasoning using Phase 7 capabilities.
        
        Args:
            task: The reasoning task
            reasoning_mode: Reasoning mode ("chain_of_thought", "multi_hop", "counterfactual", "symbolic", "auto")
            context: Additional context for reasoning
            use_temporal: Whether to include temporal reasoning
            
        Returns:
            Advanced reasoning result
        """
        if not _PHASE7_AVAILABLE or not self.advanced_reasoning_engine:
            return {
                "content": f"Advanced reasoning not available for task: {task}",
                "confidence": 0.3,
                "reasoning_type": "fallback",
                "phase7_available": False
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            # Auto-select reasoning mode if needed
            if reasoning_mode == "auto":
                reasoning_mode = self.advanced_reasoning_engine.auto_select_advanced_mode(task)
            
            # Perform advanced reasoning
            reasoning_result = self.advanced_reasoning_engine.advanced_reason(task, reasoning_mode, context)
            
            # Add temporal reasoning if requested
            temporal_result = None
            if use_temporal and self.temporal_reasoning:
                temporal_result = self.temporal_reasoning.reason(task, context)
            
            # Combine results
            combined_result = {
                "primary_reasoning": {
                    "mode": reasoning_mode,
                    "content": reasoning_result.content,
                    "confidence": reasoning_result.confidence,
                    "reasoning_steps": reasoning_result.reasoning_steps,
                    "metadata": reasoning_result.metadata
                },
                "temporal_reasoning": None,
                "runtime": time.time() - start_time,
                "phase7_enabled": True
            }
            
            if temporal_result:
                combined_result["temporal_reasoning"] = {
                    "content": temporal_result.content,
                    "confidence": temporal_result.confidence,
                    "reasoning_steps": temporal_result.reasoning_steps,
                    "metadata": temporal_result.metadata
                }
                # Boost overall confidence if temporal reasoning supports primary reasoning
                if temporal_result.confidence > 0.6:
                    combined_result["primary_reasoning"]["confidence"] = min(1.0, 
                        combined_result["primary_reasoning"]["confidence"] * 1.1)
            
            # Store advanced reasoning experience for evolution
            if self.evolution_manager:
                self._record_reasoning_experience(task, reasoning_mode, combined_result)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Advanced reasoning failed: {e}")
            return {
                "content": f"Advanced reasoning failed for task: {task}",
                "confidence": 0.2,
                "reasoning_type": "error",
                "error": str(e),
                "phase7_available": True
            }
    
    def hybrid_reasoning(
        self,
        task: str,
        modes: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply multiple advanced reasoning modes to the same task.
        
        Args:
            task: The reasoning task
            modes: List of reasoning modes to apply
            context: Additional context for reasoning
            
        Returns:
            Hybrid reasoning results from multiple modes
        """
        if not _PHASE7_AVAILABLE or not self.advanced_reasoning_engine:
            return {
                "content": f"Hybrid reasoning not available for task: {task}",
                "confidence": 0.3,
                "modes_applied": [],
                "phase7_available": False
            }
        
        start_time = time.time()
        context = context or {}
        
        try:
            # Apply multiple reasoning modes
            reasoning_results = self.advanced_reasoning_engine.hybrid_reasoning(task, modes, context)
            
            # Synthesize results
            all_confidences = [r.confidence for r in reasoning_results]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            # Find consensus or best result
            best_result = max(reasoning_results, key=lambda r: r.confidence) if reasoning_results else None
            
            hybrid_result = {
                "task": task,
                "modes_applied": modes,
                "individual_results": [
                    {
                        "mode": r.reasoning_type.value,
                        "content": r.content,
                        "confidence": r.confidence,
                        "reasoning_steps": r.reasoning_steps,
                        "metadata": r.metadata
                    }
                    for r in reasoning_results
                ],
                "synthesis": {
                    "best_mode": best_result.reasoning_type.value if best_result else "none",
                    "best_content": best_result.content if best_result else "No valid results",
                    "avg_confidence": avg_confidence,
                    "consensus_confidence": avg_confidence * (len(reasoning_results) / len(modes))
                },
                "runtime": time.time() - start_time,
                "phase7_enabled": True
            }
            
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Hybrid reasoning failed: {e}")
            return {
                "content": f"Hybrid reasoning failed for task: {task}",
                "confidence": 0.2,
                "modes_applied": modes,
                "error": str(e),
                "phase7_available": True
            }
    
    def get_enhanced_memory_hierarchy(self) -> Dict[str, Any]:
        """Get hierarchical organization of enhanced memories."""
        if not _PHASE7_AVAILABLE or not self.enhanced_memory:
            return {"error": "Enhanced memory not available", "phase7_available": _PHASE7_AVAILABLE}
        
        try:
            return self.enhanced_memory.get_memory_hierarchy()
        except Exception as e:
            logger.error(f"Failed to get memory hierarchy: {e}")
            return {"error": str(e), "phase7_available": True}
    
    def get_cross_modal_links(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get cross-modal links for a specific memory."""
        if not _PHASE7_AVAILABLE or not self.enhanced_memory:
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
                    "metadata": link.metadata
                }
                for link in links
            ]
        except Exception as e:
            logger.error(f"Failed to get cross-modal links: {e}")
            return []
    
    def evolve_capabilities(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evolve agent capabilities based on task performance.
        
        Args:
            task_results: List of task results for learning
            
        Returns:
            Evolution report with improvements and recommendations
        """
        if not _PHASE7_AVAILABLE or not self.evolution_manager:
            return {
                "error": "Agent evolution not available",
                "phase7_available": _PHASE7_AVAILABLE
            }
        
        try:
            evolution_report = self.evolution_manager.evolve_agent(self.name, task_results)
            
            # Log evolution progress
            logger.info(f"Agent '{self.name}' evolved: {len(evolution_report.get('new_skills', []))} new skills, "
                       f"{len(evolution_report.get('improvements', []))} improvements")
            
            return evolution_report
            
        except Exception as e:
            logger.error(f"Agent evolution failed: {e}")
            return {"error": str(e), "phase7_available": True}
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get current agent capabilities and evolution status."""
        if not _PHASE7_AVAILABLE or not self.evolution_manager:
            return {
                "basic_capabilities": {
                    "name": self.name,
                    "style": self.style,
                    "has_engine": self.engine is not None,
                    "has_memory": self.memory_manager is not None,
                    "has_tools": self.tool_executor is not None
                },
                "phase7_available": _PHASE7_AVAILABLE
            }
        
        try:
            capabilities = self.evolution_manager.get_agent_capabilities(self.name)
            capabilities["basic_info"] = {
                "name": self.name,
                "style": self.style,
                "has_engine": self.engine is not None,
                "has_memory": self.memory_manager is not None,
                "has_tools": self.tool_executor is not None
            }
            capabilities["phase7_enabled"] = True
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to get agent capabilities: {e}")
            return {"error": str(e), "phase7_available": True}
    
    def get_improvement_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for agent improvements."""
        if not _PHASE7_AVAILABLE or not self.evolution_manager:
            return {
                "recommendations": ["Enable Phase 7 capabilities for advanced recommendations"],
                "phase7_available": _PHASE7_AVAILABLE
            }
        
        try:
            return self.evolution_manager.recommend_agent_improvements(self.name)
        except Exception as e:
            logger.error(f"Failed to get improvement recommendations: {e}")
            return {"error": str(e), "phase7_available": True}
    
    def save_evolution_state(self, filepath: Optional[str] = None) -> bool:
        """Save agent evolution state to file."""
        if not _PHASE7_AVAILABLE or not self.evolution_manager:
            logger.warning("Phase 7 evolution not available for state saving")
            return False
        
        if filepath is None:
            filepath = f"sessions/agent_evolution_{self.name}.json"
        
        try:
            self.evolution_manager.save_evolution_state(filepath)
            logger.info(f"Saved evolution state for agent '{self.name}' to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
            return False
    
    def load_evolution_state(self, filepath: Optional[str] = None) -> bool:
        """Load agent evolution state from file."""
        if not _PHASE7_AVAILABLE or not self.evolution_manager:
            logger.warning("Phase 7 evolution not available for state loading")
            return False
        
        if filepath is None:
            filepath = f"sessions/agent_evolution_{self.name}.json"
        
        try:
            success = self.evolution_manager.load_evolution_state(filepath)
            if success:
                logger.info(f"Loaded evolution state for agent '{self.name}' from {filepath}")
            return success
        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")
            return False
    
    def _record_reasoning_experience(
        self,
        task: str,
        reasoning_mode: str,
        result: Dict[str, Any]
    ) -> None:
        """Record reasoning experience for evolution learning."""
        if not self.evolution_manager:
            return
        
        try:
            # Create learning experience from reasoning result
            confidence = result.get("primary_reasoning", {}).get("confidence", 0.0)
            success = confidence > 0.6  # Consider high confidence as success
            
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
                    "phase7_capability": True
                }
            }
            
            # Record the task for pattern analysis
            self.evolution_manager.pattern_analyzer.record_task(task_result)
            
        except Exception as e:
            logger.error(f"Failed to record reasoning experience: {e}")
