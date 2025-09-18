"""Monitor manager for organizing and executing monitors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import MonitorFn, MonitorSpec
from .factory import MonitorFactory

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors")


def _load_data_file(path: Path) -> Dict[str, Any]:
    """Load data file (JSON or YAML)."""
    import json

    try:
        import yaml
    except ImportError:
        yaml = None

    if path.suffix.lower() in [".yaml", ".yml"] and yaml:
        return yaml.safe_load(path.read_text())
    else:
        return json.loads(path.read_text())


class MonitorManager:
    """Manages monitor specifications and creates monitor functions."""

    def __init__(self, specs: Optional[List[MonitorSpec]] = None):
        """Initialize with monitor specifications.

        Args:
            specs: List of monitor specifications
        """
        self.specs = specs or []
        self.monitors: List[MonitorFn] = [MonitorFactory.build(s) for s in self.specs]

    @staticmethod
    def load_from_file(path: str | Path) -> "MonitorManager":
        """Load monitor specifications from file.

        Args:
            path: Path to monitor configuration file

        Returns:
            MonitorManager instance
        """
        data = _load_data_file(Path(path))
        specs = [MonitorSpec(**m) for m in data.get("monitors", [])]
        return MonitorManager(specs)

    def get_callables(self) -> List[MonitorFn]:
        """Get list of monitor functions.

        Returns:
            List of monitor functions
        """
        return self.monitors

    def add_spec(self, spec: MonitorSpec) -> None:
        """Add a monitor specification.

        Args:
            spec: Monitor specification to add
        """
        self.specs.append(spec)
        self.monitors.append(MonitorFactory.build(spec))
        logger.info(f"Added monitor spec: {spec.name}")

    def get_spec_names(self) -> List[str]:
        """Get names of all monitor specifications."""
        return [spec.name for spec in self.specs]
