"""
Robust, dynamic monitor manager for organizing, controlling, and executing monitors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .base import MonitorFn, MonitorSpec
from .factory import MonitorFactory

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors")


def _load_data_file(path: Path) -> Dict[str, Any]:
    """Load data file (JSON or YAML), handling potential errors."""
    import json
    content = path.read_text()

    try:
        if path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                logger.warning("PyYAML is not installed; .yaml/.yml config will be treated as JSON.")
        return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to load or parse monitor configuration file at '{path}': {e}")
        return {}


class MonitorManager:
    """
    Manages the lifecycle, configuration, and execution of monitors.

    This class provides a robust, dynamic way to handle monitoring. It supports
    loading configurations from files, enabling/disabling monitors at runtime,
    grouping monitors by tags, and resiliently handling invalid specs.
    """

    def __init__(self, specs: Optional[List[Union[MonitorSpec, Dict[str, Any]]]] = None):
        """
        Initialize the manager with a list of monitor specifications.

        Args:
            specs: A list of MonitorSpec objects or dictionaries to build them from.
        """
        self.specs_by_name: Dict[str, MonitorSpec] = {}
        self.monitors_by_name: Dict[str, MonitorFn] = {}

        if specs:
            self._build_monitors_from_specs(specs)

    def _build_monitors_from_specs(self, specs: List[Union[MonitorSpec, Dict[str, Any]]]):
        """Builds monitor functions from specifications, handling errors gracefully."""
        seen_names = set()
        for i, spec_data in enumerate(specs):
            try:
                spec = MonitorSpec(**spec_data) if isinstance(spec_data, dict) else spec_data

                if spec.name in seen_names:
                    logger.warning(f"Duplicate monitor name '{spec.name}' found. Overwriting previous definition.")
                
                self.specs_by_name[spec.name] = spec
                self.monitors_by_name[spec.name] = MonitorFactory.build(spec)
                seen_names.add(spec.name)

            except Exception as e:
                name = spec_data.get("name", f"unnamed_spec_{i}") if isinstance(spec_data, dict) else f"spec_{i}"
                logger.error(f"Failed to build monitor '{name}': {e}", exc_info=True)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "MonitorManager":
        """
        Load monitor specifications from a JSON or YAML file.

        Args:
            path: Path to the monitor configuration file.

        Returns:
            A new MonitorManager instance.
        """
        config_path = Path(path)
        if not config_path.exists():
            logger.error(f"Monitor configuration file not found at: {config_path}")
            return cls()

        data = _load_data_file(config_path)
        specs_data = data.get("monitors", [])
        if not isinstance(specs_data, list):
            logger.error(f"Expected 'monitors' to be a list in '{config_path}', but found {type(specs_data)}.")
            return cls()
            
        return cls(specs=specs_data)

    def run_monitors(self, agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        """
        Execute all enabled monitors against the agent's result.

        This is the primary method for applying monitoring checks.

        Args:
            agent: The AgentNet instance that produced the result.
            task: The task/prompt given to the agent.
            result: The result dictionary from the agent's reasoning process.
        """
        for monitor_name, monitor_fn in self.monitors_by_name.items():
            spec = self.specs_by_name.get(monitor_name)
            if spec and spec.enabled:
                try:
                    monitor_fn(agent, task, result)
                except Exception as e:
                    logger.error(f"Monitor '{monitor_name}' raised an unhandled exception: {e}", exc_info=True)

    def get_active_callables(self, tags: Optional[List[str]] = None) -> List[MonitorFn]:
        """
        Get a list of all enabled monitor functions, optionally filtered by tags.

        Args:
            tags: If provided, only return monitors that have at least one of
                  the specified tags.

        Returns:
            A list of executable monitor functions.
        """
        callables = []
        for name, spec in self.specs_by_name.items():
            if spec.enabled:
                if tags and not set(spec.tags or []).intersection(tags):
                    continue
                callables.append(self.monitors_by_name[name])
        return callables

    def add_spec(self, spec: MonitorSpec, overwrite: bool = False) -> None:
        """
        Add a new monitor specification at runtime.

        Args:
            spec: The MonitorSpec to add.
            overwrite: If True, replace an existing monitor with the same name.
        """
        if spec.name in self.specs_by_name and not overwrite:
            raise ValueError(f"Monitor with name '{spec.name}' already exists. Use overwrite=True to replace it.")
        
        try:
            self.specs_by_name[spec.name] = spec
            self.monitors_by_name[spec.name] = MonitorFactory.build(spec)
            logger.info(f"Successfully added/updated monitor spec: '{spec.name}'")
        except Exception as e:
            logger.error(f"Failed to build and add monitor '{spec.name}': {e}")
            self.specs_by_name.pop(spec.name, None)
            self.monitors_by_name.pop(spec.name, None)
            raise

    def get_spec(self, name: str) -> Optional[MonitorSpec]:
        """Get a monitor specification by its unique name."""
        return self.specs_by_name.get(name)

    def _toggle_status_by_query(self, query: Union[str, List[str]], enabled: bool):
        """Generic helper to enable/disable monitors by name or tag."""
        queries = {query} if isinstance(query, str) else set(query)
        found = False
        for spec in self.specs_by_name.values():
            if spec.name in queries or set(spec.tags or []).intersection(queries):
                spec.enabled = enabled
                found = True
        if not found:
            logger.warning(f"No monitors found matching query: {query}")

    def enable(self, name_or_tag: Union[str, List[str]]):
        """Enable one or more monitors by name or tag."""
        self._toggle_status_by_query(name_or_tag, enabled=True)
        logger.info(f"Enabled monitors matching: {name_or_tag}")

    def disable(self, name_or_tag: Union[str, List[str]]):
        """Disable one or more monitors by name or tag."""
        self._toggle_status_by_query(name_or_tag, enabled=False)
        logger.info(f"Disabled monitors matching: {name_or_tag}")

    def get_status(self) -> List[Dict[str, Any]]:
        """
        Get the status of all managed monitors.

        Returns:
            A list of dictionaries, each detailing a monitor's status.
        """
        return [
            {
                "name": spec.name,
                "type": spec.type,
                "enabled": spec.enabled,
                "tags": spec.tags or [],
                "severity": spec.severity,
            }
            for spec in self.specs_by_name.values()
        ]
