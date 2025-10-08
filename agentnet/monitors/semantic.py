"""
Advanced semantic similarity monitor with model caching, batch processing, and
reference text comparison for powerful content guardrails.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Deque

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.semantic")

# --- Global Caches for Performance ---
# Cache for loaded sentence transformer models to avoid reloading
_model_cache: Dict[str, Any] = {}
# Cache for historical embeddings/text, managed by deque
_history_cache: Dict[str, Deque] = {}
# Cache for pre-computed embeddings of reference texts
_reference_embedding_cache: Dict[str, Any] = {}


def _get_model(model_name: str):
    """
    Lazily loads and caches a sentence transformer model.
    Returns the model instance or None if dependencies are not available.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        logger.info(f"Successfully loaded and cached sentence transformer model: {model_name}")
        return model
    except ImportError:
        if not _model_cache: # Log only once
             logger.warning(
                "sentence-transformers library not found. Semantic monitor will use Jaccard similarity fallback."
            )
        _model_cache[model_name] = None
        return None
    except Exception as e:
        logger.error(f"Failed to download or load sentence transformer model '{model_name}': {e}")
        _model_cache[model_name] = None
        return None


def create_semantic_similarity_monitor(spec: MonitorSpec) -> MonitorFn:
    """
    Create an advanced semantic similarity monitor.

    This monitor can detect repetitive content by comparing against recent history,
    or enforce content guardrails by comparing against a fixed set of reference texts.

    Args:
        spec: Monitor specification with parameters:
            - model_name (str): Name of the sentence-transformers model to use.
              Defaults to "all-MiniLM-L6-v2".
            - max_similarity (float): Trigger violation if similarity exceeds this.
              Useful for preventing repetition or forbidden topics.
            - min_similarity (float): Trigger violation if similarity is below this.
              Useful for ensuring topic adherence with reference_texts.
            - window_size (int): Number of historical items to compare against.
              Defaults to 5.
            - reference_texts (List[str]): A list of texts to compare against
              instead of the agent's history.
            - violation_name (str): Custom name for the violation.
    """
    # --- 1. Configuration ---
    model_name = spec.params.get("model_name", "all-MiniLM-L6-v2")
    max_similarity = spec.params.get("max_similarity")
    min_similarity = spec.params.get("min_similarity")
    window_size = int(spec.params.get("window_size", 5))
    reference_texts = spec.params.get("reference_texts")
    violation_name = spec.params.get("violation_name", f"{spec.name}_semantic_violation")

    if max_similarity is None and min_similarity is None:
        raise ValueError("Semantic monitor requires at least one of 'max_similarity' or 'min_similarity' to be set.")

    model = _get_model(model_name)

    # --- 2. Return the appropriate monitor function (semantic or fallback) ---
    if model:
        return _create_semantic_monitor_impl(
            spec, model, model_name, max_similarity, min_similarity,
            window_size, reference_texts, violation_name
        )
    else:
        return _create_jaccard_fallback_monitor(
            spec, max_similarity, window_size, violation_name
        )


def _create_semantic_monitor_impl(
    spec, model, model_name, max_similarity, min_similarity,
    window_size, reference_texts, violation_name
) -> MonitorFn:
    """Creates the primary monitor function that uses sentence embeddings."""
    from sentence_transformers.util import cos_sim
    import torch

    # Pre-compute reference embeddings if provided
    reference_embeddings = None
    if reference_texts:
        ref_cache_key = f"{model_name}_{hash(tuple(reference_texts))}"
        if ref_cache_key not in _reference_embedding_cache:
            _reference_embedding_cache[ref_cache_key] = model.encode(reference_texts, convert_to_tensor=True)
        reference_embeddings = _reference_embedding_cache[ref_cache_key]

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        from .factory import MonitorFactory
        if MonitorFactory._should_cooldown(spec, task):
            return

        content = str(result.get("content", "")) if isinstance(result, dict) else str(result)
        if not content.strip():
            return

        current_embedding = model.encode(content, convert_to_tensor=True)
        comparison_embeddings = None
        comparison_mode = "history"

        if reference_embeddings is not None:
            comparison_embeddings = reference_embeddings
            comparison_mode = "reference"
        else:
            agent_key = f"{agent.name}_{task}_{model_name}"
            if agent_key not in _history_cache:
                _history_cache[agent_key] = deque(maxlen=window_size)
            
            history = _history_cache[agent_key]
            if history:
                comparison_embeddings = torch.stack(list(history))

        if comparison_embeddings is not None:
            similarities = cos_sim(current_embedding, comparison_embeddings)[0]
            max_score = torch.max(similarities).item()
            
            # Check for max_similarity violation
            if max_similarity is not None and max_score > max_similarity:
                factory.trigger_violation(
                    name=name,
                    severity=severity,
                    message=f"Semantic similarity {max_score:.3f} exceeds maximum {max_similarity:.3f} ({comparison_mode})",
                    task=task,
                    result=result,
                    agent=agent,
                )
            
            # Check for min_similarity violation
            if min_similarity is not None and max_score < min_similarity:
                factory.trigger_violation(
                    name=name,
                    severity=severity,
                    message=f"Semantic similarity {max_score:.3f} below minimum {min_similarity:.3f} ({comparison_mode})",
                    task=task,
                    result=result,
                    agent=agent,
                )

        # Update history cache if using history mode
        if reference_embeddings is None and comparison_mode == "history":
            _history_cache[agent_key].append(current_embedding)

    return monitor_fn
