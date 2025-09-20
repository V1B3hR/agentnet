"""Semantic similarity monitor implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.semantic")


def create_semantic_similarity_monitor(spec: MonitorSpec) -> MonitorFn:
    """Create a semantic similarity monitor.

    This monitor tracks content similarity over time and triggers violations
    when new content is too similar to previous outputs.

    Args:
        spec: Monitor specification with parameters:
            - max_similarity: Maximum allowed similarity (default: 0.9)
            - window_size: Number of historical items to compare against (default: 5)
            - embedding_set: Name of embedding set to use (default: "restricted_corpora")
            - violation_name: Name for violation (optional)

    Returns:
        Monitor function
    """
    max_similarity = float(spec.params.get("max_similarity", 0.9))
    window_size = int(spec.params.get("window_size", 5))
    embedding_set = spec.params.get("embedding_set", "restricted_corpora")
    violation_name = spec.params.get("violation_name", f"{spec.name}_semantic")

    # Storage for content history per agent-task combination
    if not hasattr(create_semantic_similarity_monitor, "_semantic_history"):
        create_semantic_similarity_monitor._semantic_history = {}

    # Try to import sentence-transformers for semantic similarity
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Try to load model, but handle download failures gracefully
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            use_semantic = True
        except Exception as model_error:
            logger.warning(
                f"Failed to load sentence transformer model: {model_error}"
            )
            model = None
            use_semantic = False
    except ImportError:
        logger.warning(
            "sentence-transformers not available, falling back to Jaccard similarity"
        )
        model = None
        use_semantic = False

    def semantic_similarity(text1: str, text2: str) -> float:
        """Compute semantic similarity."""
        if not use_semantic:
            # Fallback to Jaccard similarity
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        embeddings = model.encode([text1, text2])
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        # Import here to avoid circular imports
        from .factory import MonitorFactory

        if MonitorFactory._should_cooldown(spec, task):
            return

        content = (
            str(result.get("content", ""))
            if isinstance(result, dict)
            else str(result)
        )
        if not content.strip():
            return

        agent_key = f"{agent.name}_{task}_semantic"

        # Initialize history for this agent-task combination
        history_store = create_semantic_similarity_monitor._semantic_history
        if agent_key not in history_store:
            history_store[agent_key] = []

        history = history_store[agent_key]

        # Check semantic similarity against recent history
        violations = []
        for i, historical_content in enumerate(history[-window_size:]):
            similarity = semantic_similarity(content, historical_content)
            if similarity > max_similarity:
                violations.append(
                    MonitorFactory._build_violation(
                        name=violation_name,
                        vtype="semantic_similarity",
                        severity=spec.severity,
                        description=spec.description
                        or f"Semantic similarity {similarity:.2f} exceeds threshold",
                        rationale=f"Current content semantically too similar to content from {len(history)-i} turns ago",
                        meta={
                            "similarity_score": similarity,
                            "threshold": max_similarity,
                            "historical_index": len(history) - i,
                            "embedding_set": embedding_set,
                            "window_size": window_size,
                        },
                    )
                )

        # Add current content to history
        history.append(content)

        # Limit history size to prevent unbounded growth
        if len(history) > window_size * 2:
            history[:] = history[-window_size:]

        if violations:
            detail = {
                "outcome": {"content": content},
                "violations": violations,
            }
            MonitorFactory._handle(
                spec, agent, task, passed=False, detail=detail
            )

    return monitor
