"""
AgentNet Deep Learning Module (Phase 9)

This module provides deep learning capabilities for AgentNet, including:
- Model registry and management
- Training pipeline infrastructure
- Fine-tuning utilities for LLMs
- Embedding generation and management
- Neural reasoning modules

The module is designed to work with PyTorch as the primary framework,
with optional TensorFlow support.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch

    PYTORCH_AVAILABLE = True
    PYTORCH_VERSION = torch.__version__
except ImportError:
    PYTORCH_AVAILABLE = False
    PYTORCH_VERSION = None
    torch = None

# Try to import TensorFlow (optional)
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
    TENSORFLOW_VERSION = tf.__version__
except ImportError:
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_VERSION = None
    tf = None

# Try to import transformers
try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_VERSION = None
    transformers = None

# Try to import sentence-transformers
try:
    import sentence_transformers

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    sentence_transformers = None


def is_available() -> bool:
    """Check if deep learning features are available."""
    return PYTORCH_AVAILABLE


def pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return PYTORCH_AVAILABLE


def tensorflow_available() -> bool:
    """Check if TensorFlow is available."""
    return TENSORFLOW_AVAILABLE


def get_framework_info() -> dict:
    """Get information about available deep learning frameworks."""
    return {
        "pytorch": {
            "available": PYTORCH_AVAILABLE,
            "version": PYTORCH_VERSION,
        },
        "tensorflow": {
            "available": TENSORFLOW_AVAILABLE,
            "version": TENSORFLOW_VERSION,
        },
        "transformers": {
            "available": TRANSFORMERS_AVAILABLE,
            "version": TRANSFORMERS_VERSION,
        },
        "sentence_transformers": {
            "available": SENTENCE_TRANSFORMERS_AVAILABLE,
        },
    }


# Conditional imports - only import if dependencies available
# Note: Registry module doesn't need PyTorch, so import it separately
try:
    from .registry import ModelRegistry, ModelMetadata, ModelArtifact

    _REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Registry not available: {e}")
    _REGISTRY_AVAILABLE = False

    # Provide stub classes
    class ModelRegistry:
        def __init__(self, *args, **kwargs):
            raise ImportError("ModelRegistry import failed. Check dependencies.")

    class ModelMetadata:
        def __init__(self, *args, **kwargs):
            raise ImportError("ModelMetadata import failed. Check dependencies.")

    class ModelArtifact:
        def __init__(self, *args, **kwargs):
            raise ImportError("ModelArtifact import failed. Check dependencies.")


if PYTORCH_AVAILABLE:
    try:
        from .trainer import DeepLearningTrainer, TrainingConfig, TrainingCallback
        from .embeddings import EmbeddingGenerator, EmbeddingCache, SemanticSearch

        _CORE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Some deep learning components not available: {e}")
        _CORE_AVAILABLE = False

        # Provide stub classes
        class DeepLearningTrainer:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "DeepLearningTrainer requires PyTorch. Install with: pip install agentnet[deeplearning]"
                )

        class TrainingConfig:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "TrainingConfig requires PyTorch. Install with: pip install agentnet[deeplearning]"
                )

        class TrainingCallback:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "TrainingCallback requires PyTorch. Install with: pip install agentnet[deeplearning]"
                )

        class EmbeddingGenerator:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "EmbeddingGenerator requires sentence-transformers. Install with: pip install agentnet[deeplearning]"
                )

        class EmbeddingCache:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "EmbeddingCache requires PyTorch. Install with: pip install agentnet[deeplearning]"
                )

        class SemanticSearch:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "SemanticSearch requires FAISS. Install with: pip install agentnet[deeplearning]"
                )

else:
    _CORE_AVAILABLE = False

    # Provide stub classes when PyTorch not available
    class DeepLearningTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DeepLearningTrainer requires PyTorch. Install with: pip install agentnet[deeplearning]"
            )

    class TrainingConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TrainingConfig requires PyTorch. Install with: pip install agentnet[deeplearning]"
            )

    class TrainingCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TrainingCallback requires PyTorch. Install with: pip install agentnet[deeplearning]"
            )

    class EmbeddingGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EmbeddingGenerator requires sentence-transformers. Install with: pip install agentnet[deeplearning]"
            )

    class EmbeddingCache:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EmbeddingCache requires PyTorch. Install with: pip install agentnet[deeplearning]"
            )

    class SemanticSearch:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SemanticSearch requires FAISS. Install with: pip install agentnet[deeplearning]"
            )


# Fine-tuning components (require transformers + peft)
try:
    if PYTORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
        from .finetuning import FineTuner, LoRAConfig, InstructionDataset

        _FINETUNING_AVAILABLE = True
    else:
        raise ImportError("Fine-tuning requires PyTorch and Transformers")
except ImportError:
    _FINETUNING_AVAILABLE = False

    class FineTuner:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FineTuner requires transformers and peft. Install with: pip install agentnet[deeplearning]"
            )

    class LoRAConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LoRAConfig requires peft. Install with: pip install agentnet[deeplearning]"
            )

    class InstructionDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "InstructionDataset requires datasets. Install with: pip install agentnet[deeplearning]"
            )


# Neural reasoning components
try:
    if PYTORCH_AVAILABLE:
        from .neural_reasoning import (
            NeuralReasoner,
            AttentionReasoning,
            GraphNeuralReasoning,
        )

        _NEURAL_REASONING_AVAILABLE = True
    else:
        raise ImportError("Neural reasoning requires PyTorch")
except ImportError:
    _NEURAL_REASONING_AVAILABLE = False

    class NeuralReasoner:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "NeuralReasoner requires PyTorch. Install with: pip install agentnet[deeplearning]"
            )

    class AttentionReasoning:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AttentionReasoning requires PyTorch. Install with: pip install agentnet[deeplearning]"
            )

    class GraphNeuralReasoning:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GraphNeuralReasoning requires PyTorch Geometric. Install with: pip install torch-geometric"
            )


__all__ = [
    # Status functions
    "is_available",
    "pytorch_available",
    "tensorflow_available",
    "get_framework_info",
    # Model registry
    "ModelRegistry",
    "ModelMetadata",
    "ModelArtifact",
    # Training
    "DeepLearningTrainer",
    "TrainingConfig",
    "TrainingCallback",
    # Fine-tuning
    "FineTuner",
    "LoRAConfig",
    "InstructionDataset",
    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingCache",
    "SemanticSearch",
    # Neural reasoning
    "NeuralReasoner",
    "AttentionReasoning",
    "GraphNeuralReasoning",
]


# Log availability status
if not PYTORCH_AVAILABLE:
    logger.warning(
        "PyTorch not available. Deep learning features disabled. "
        "Install with: pip install agentnet[deeplearning]"
    )
elif _CORE_AVAILABLE:
    logger.info(f"AgentNet Deep Learning initialized (PyTorch {PYTORCH_VERSION})")
