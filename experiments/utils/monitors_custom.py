"""
Custom monitor implementations for AgentNet experiments.
Includes repetition detection and semantic similarity monitors.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

# Add parent directory to path to import AgentNet
sys.path.append(str(Path(__file__).parent.parent.parent))

from AgentNet import MonitorSpec, MonitorFn
from experiments.utils.analytics import jaccard_similarity


# Global storage for repetition monitor state
_repetition_history: Dict[str, List[str]] = {}


def create_repetition_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a repetition detection monitor using Jaccard similarity."""
    max_similarity = spec.params.get("max_similarity", 0.8)
    window_size = spec.params.get("window_size", 3)
    
    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        content = result.get("content", "")
        if not content:
            return
        
        agent_key = f"{agent.name}_{task}"
        
        # Initialize history for this agent-task combination
        if agent_key not in _repetition_history:
            _repetition_history[agent_key] = []
        
        history = _repetition_history[agent_key]
        
        # Check similarity against recent history
        violations = []
        for i, historical_content in enumerate(history[-window_size:]):
            similarity = jaccard_similarity(content, historical_content)
            if similarity > max_similarity:
                violations.append({
                    "type": "repetition",
                    "severity": spec.severity,
                    "description": f"Content similarity {similarity:.2f} exceeds threshold {max_similarity}",
                    "rationale": f"Current content too similar to content from {len(history)-i} turns ago",
                    "meta": {
                        "similarity_score": similarity,
                        "threshold": max_similarity,
                        "historical_index": len(history) - i
                    }
                })
        
        # Add current content to history
        history.append(content)
        
        # Limit history size to prevent unbounded growth
        if len(history) > window_size * 2:
            history.pop(0)
        
        # Handle violations if any
        if violations:
            from AgentNet import MonitorFactory
            detail = {
                "outcome": {"content": content},
                "violations": violations
            }
            MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
    
    return monitor


def create_semantic_similarity_monitor(spec: MonitorSpec) -> MonitorFn:
    """
    Create a semantic similarity monitor.
    Falls back to Jaccard similarity if sentence-transformers is not available.
    """
    max_similarity = spec.params.get("max_similarity", 0.9)
    window_size = spec.params.get("window_size", 5)
    
    # Try to import sentence-transformers for semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Use a lightweight model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        use_semantic = True
    except ImportError:
        model = None
        use_semantic = False
    
    def semantic_similarity(text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence transformers."""
        if not use_semantic:
            return jaccard_similarity(text1, text2)
        
        embeddings = model.encode([text1, text2])
        # Compute cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        content = result.get("content", "")
        if not content:
            return
        
        agent_key = f"{agent.name}_{task}_semantic"
        
        # Initialize history for this agent-task combination
        if agent_key not in _repetition_history:
            _repetition_history[agent_key] = []
        
        history = _repetition_history[agent_key]
        
        # Check semantic similarity against recent history
        violations = []
        for i, historical_content in enumerate(history[-window_size:]):
            similarity = semantic_similarity(content, historical_content)
            if similarity > max_similarity:
                violations.append({
                    "type": "semantic_repetition",
                    "severity": spec.severity,
                    "description": f"Semantic similarity {similarity:.2f} exceeds threshold {max_similarity}",
                    "rationale": f"Current content semantically too similar to content from {len(history)-i} turns ago",
                    "meta": {
                        "similarity_score": similarity,
                        "threshold": max_similarity,
                        "historical_index": len(history) - i,
                        "similarity_type": "semantic" if use_semantic else "jaccard"
                    }
                })
        
        # Add current content to history
        history.append(content)
        
        # Limit history size
        if len(history) > window_size * 2:
            history.pop(0)
        
        # Handle violations if any
        if violations:
            from AgentNet import MonitorFactory
            detail = {
                "outcome": {"content": content},
                "violations": violations
            }
            MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
    
    return monitor


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


def create_custom_monitor(spec: MonitorSpec) -> MonitorFn:
    """Factory function to create custom monitors based on spec type."""
    if spec.type == "repetition":
        return create_repetition_monitor(spec)
    elif spec.type == "semantic_similarity":
        return create_semantic_similarity_monitor(spec)
    else:
        raise ValueError(f"Unknown custom monitor type: {spec.type}")