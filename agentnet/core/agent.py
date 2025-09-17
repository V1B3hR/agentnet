"""Core AgentNet implementation."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .types import CognitiveFault
from ..monitors.base import MonitorFn
from ..persistence.session import SessionManager
from ..persistence.agent_state import AgentStateManager

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
                "truncate_strategy": "head"
            }
        }
        
        # Initialize managers
        self.session_manager = SessionManager()
        
        logger.info(
            f"AgentNet instance '{name}' initialized with style {style}, "
            f"{len(self.monitors)} monitors, {len(self.pre_monitors)} pre-monitors"
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

    def register_monitors(self, monitor_fns: List[MonitorFn], pre_style: bool = False) -> None:
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

    def _apply_style_influence(self, base_result: Dict[str, Any], task: str) -> Dict[str, Any]:
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
            "style_influence": style_influence
        }

    def generate_reasoning_tree(
        self,
        task: str,
        include_monitor_trace: bool = False,
        max_depth: int = 3,
        confidence_threshold: float = 0.7,
        style_override: Optional[Dict[str, float]] = None
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
                    "confidence": 0.5
                }
            
            base_result = self._normalize_engine_result(raw_result)
            
            # Run pre-style monitors
            self._run_pre_monitors(task, base_result)
            
            # Apply style influence
            styled_result = self._apply_style_influence(base_result, task)
            
            # Run post-style monitors  
            self._run_post_monitors(task, styled_result)
            
            # Build reasoning tree
            runtime = time.time() - start_time
            reasoning_tree = {
                "root": f"{self.name}_reasoning",
                "result": styled_result,
                "agent": self.name,
                "task": task,
                "runtime": runtime,
                "timestamp": time.time(),
                "style": self.style.copy(),
                "monitor_trace": [] if include_monitor_trace else None
            }
            
            # Record in interaction history
            self.interaction_history.append({
                "type": "reasoning_tree",
                "task": task,
                "result": styled_result,
                "runtime": runtime
            })
            
            return reasoning_tree
            
        except CognitiveFault as cf:
            # Handle cognitive faults
            runtime = time.time() - start_time
            fault_tree = {
                "root": f"{self.name}_fault",
                "result": {
                    "content": f"[{self.name}] Cognitive fault: {cf}",
                    "confidence": 0.1,
                    "fault": True
                },
                "agent": self.name,
                "task": task,
                "runtime": runtime,
                "timestamp": time.time(),
                "cognitive_fault": cf.to_dict()
            }
            
            self.interaction_history.append({
                "type": "cognitive_fault",
                "task": task,
                "fault": cf.to_dict(),
                "runtime": runtime
            })
            
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
        cls,
        path: str,
        engine=None,
        monitors: Optional[List[MonitorFn]] = None
    ) -> 'AgentNet':
        """Load agent state from file."""
        return AgentStateManager.load_state(path, cls, engine, monitors)

    @staticmethod
    def from_config(config_path: str | Path, engine=None) -> 'AgentNet':
        """Create agent from configuration file (placeholder)."""
        # This would load from a config file - for now return a default agent
        return AgentNet("ConfigAgent", {"logic": 0.7, "creativity": 0.5}, engine=engine)

    def __repr__(self) -> str:
        return f"AgentNet(name='{self.name}', style={self.style})"