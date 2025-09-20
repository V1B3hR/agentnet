"""
Custom monitor implementations for AgentNet experiments.
Uses the refactored monitor system with shared implementations.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

# Add parent directory to path to import AgentNet
sys.path.append(str(Path(__file__).parent.parent.parent))

from agentnet.monitors.base import MonitorSpec, MonitorFn


def create_repetition_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a repetition detection monitor using semantic similarity.
    
    This is now just a wrapper around the semantic similarity monitor
    with appropriate default parameters for repetition detection.
    """
    # Import from the refactored monitor system
    from agentnet.monitors.semantic import create_semantic_similarity_monitor
    
    # Override parameters for repetition detection defaults
    repetition_spec = MonitorSpec(
        name=spec.name,
        type="semantic_similarity",
        params={
            "max_similarity": spec.params.get("max_similarity", 0.8),  # Lower threshold for repetition
            "window_size": spec.params.get("window_size", 3),          # Smaller window for repetition
            "violation_name": spec.params.get("violation_name", f"{spec.name}_repetition"),
            **spec.params  # Allow override of any parameter
        },
        severity=spec.severity,
        description=spec.description or "Repetition detection using semantic similarity",
        cooldown_seconds=spec.cooldown_seconds
    )
    
    return create_semantic_similarity_monitor(repetition_spec)


def create_semantic_similarity_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a semantic similarity monitor.
    
    This is now just a wrapper around the refactored semantic monitor.
    """
    # Import from the refactored monitor system
    from agentnet.monitors.semantic import create_semantic_similarity_monitor as create_semantic
    
    return create_semantic(spec)


def create_custom_monitor(spec: MonitorSpec) -> MonitorFn:
    """Factory function to create custom monitors based on spec type."""
    if spec.type == "repetition":
        return create_repetition_monitor(spec)
    elif spec.type == "semantic_similarity":
        return create_semantic_similarity_monitor(spec)
    else:
        # Delegate to the main monitor factory for other types
        from agentnet.monitors import MonitorFactory
        return MonitorFactory.build(spec)


def load_monitor_config(config_path: Path) -> List[MonitorSpec]:
    """Load monitor configuration from YAML file."""
    import yaml
    
    if not config_path.exists():
        raise FileNotFoundError(f"Monitor config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    monitors = []
    for monitor_data in config.get('monitors', []):
        spec = MonitorSpec(
            name=monitor_data['name'],
            type=monitor_data['type'],
            severity=monitor_data.get('severity', 'minor'),
            description=monitor_data.get('description', ''),
            params=monitor_data.get('params', {})
        )
        monitors.append(spec)
    
    return monitors