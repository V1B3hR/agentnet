"""
Tests for Phase 9 Deep Learning Integration

Tests the deep learning module structure, model registry,
training configuration, and integration points.
"""

import pytest
import tempfile
from pathlib import Path
import json

# Try to import deep learning components
try:
    from agentnet.deeplearning import (
        is_available,
        pytorch_available,
        get_framework_info,
        ModelRegistry,
        ModelMetadata,
        TrainingConfig,
        LoRAConfig,
        EmbeddingGenerator,
        EmbeddingCache,
    )
    DEEPLEARNING_IMPORTED = True
except ImportError:
    DEEPLEARNING_IMPORTED = False

# Try to import from submodules for registry testing
try:
    from agentnet.deeplearning.registry import ModelType, ModelStatus
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class TestDeepLearningAvailability:
    """Test deep learning module availability and status functions."""
    
    def test_module_imports(self):
        """Test that deep learning module can be imported."""
        assert DEEPLEARNING_IMPORTED, "Deep learning module should be importable"
    
    def test_is_available_function(self):
        """Test is_available status function."""
        if DEEPLEARNING_IMPORTED:
            result = is_available()
            assert isinstance(result, bool)
    
    def test_pytorch_available_function(self):
        """Test pytorch_available status function."""
        if DEEPLEARNING_IMPORTED:
            result = pytorch_available()
            assert isinstance(result, bool)
    
    def test_get_framework_info(self):
        """Test get_framework_info returns valid structure."""
        if DEEPLEARNING_IMPORTED:
            info = get_framework_info()
            assert isinstance(info, dict)
            assert "pytorch" in info
            assert "tensorflow" in info
            assert "transformers" in info
            assert "available" in info["pytorch"]


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry module not available")
class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_registry_creation(self):
        """Test creating a model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            assert registry.registry_dir == Path(tmpdir)
            assert registry.metadata_file.exists()
    
    def test_model_metadata_creation(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            model_id="test-model-001",
            name="Test Model",
            version="1.0.0",
            model_type=ModelType.LANGUAGE_MODEL,
            status=ModelStatus.READY,
            base_model="test-base",
            description="Test model",
            tags=["test"],
            metrics={"accuracy": 0.9}
        )
        
        assert metadata.model_id == "test-model-001"
        assert metadata.name == "Test Model"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == ModelType.LANGUAGE_MODEL
        assert metadata.status == ModelStatus.READY
        assert metadata.metrics["accuracy"] == 0.9
    
    def test_model_registration(self):
        """Test registering a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            
            metadata = ModelMetadata(
                model_id="test-model-002",
                name="Test Model 2",
                version="1.0.0",
                model_type=ModelType.LANGUAGE_MODEL,
                status=ModelStatus.READY,
            )
            
            registry.register(metadata)
            
            # Verify registration
            retrieved = registry.get("test-model-002")
            assert retrieved is not None
            assert retrieved.model_id == "test-model-002"
            assert retrieved.name == "Test Model 2"
    
    def test_list_models(self):
        """Test listing models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            
            # Register multiple models
            for i in range(3):
                metadata = ModelMetadata(
                    model_id=f"test-model-{i}",
                    name=f"Test Model {i}",
                    version="1.0.0",
                    model_type=ModelType.LANGUAGE_MODEL,
                    status=ModelStatus.READY,
                )
                registry.register(metadata)
            
            # List all models
            models = registry.list_models()
            assert len(models) == 3
    
    def test_list_models_with_filter(self):
        """Test listing models with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            
            # Register models with different statuses
            metadata1 = ModelMetadata(
                model_id="test-model-ready",
                name="Ready Model",
                version="1.0.0",
                model_type=ModelType.LANGUAGE_MODEL,
                status=ModelStatus.READY,
            )
            metadata2 = ModelMetadata(
                model_id="test-model-training",
                name="Training Model",
                version="1.0.0",
                model_type=ModelType.LANGUAGE_MODEL,
                status=ModelStatus.TRAINING,
            )
            
            registry.register(metadata1)
            registry.register(metadata2)
            
            # Filter by status
            ready_models = registry.list_models(status=ModelStatus.READY)
            assert len(ready_models) == 1
            assert ready_models[0].status == ModelStatus.READY
    
    def test_update_status(self):
        """Test updating model status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            
            metadata = ModelMetadata(
                model_id="test-model-status",
                name="Status Test Model",
                version="1.0.0",
                model_type=ModelType.LANGUAGE_MODEL,
                status=ModelStatus.TRAINING,
            )
            
            registry.register(metadata)
            
            # Update status
            registry.update_status("test-model-status", ModelStatus.READY)
            
            # Verify update
            updated = registry.get("test-model-status")
            assert updated.status == ModelStatus.READY
    
    def test_update_metrics(self):
        """Test updating model metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            
            metadata = ModelMetadata(
                model_id="test-model-metrics",
                name="Metrics Test Model",
                version="1.0.0",
                model_type=ModelType.LANGUAGE_MODEL,
                status=ModelStatus.READY,
                metrics={"accuracy": 0.8}
            )
            
            registry.register(metadata)
            
            # Update metrics
            registry.update_metrics("test-model-metrics", {"loss": 0.5})
            
            # Verify update
            updated = registry.get("test-model-metrics")
            assert updated.metrics["accuracy"] == 0.8
            assert updated.metrics["loss"] == 0.5
    
    def test_metadata_serialization(self):
        """Test metadata to_dict and from_dict."""
        metadata = ModelMetadata(
            model_id="test-model-serial",
            name="Serial Test Model",
            version="1.0.0",
            model_type=ModelType.LANGUAGE_MODEL,
            status=ModelStatus.READY,
        )
        
        # Serialize to dict
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["model_id"] == "test-model-serial"
        
        # Deserialize from dict
        restored = ModelMetadata.from_dict(metadata_dict)
        assert restored.model_id == metadata.model_id
        assert restored.name == metadata.name
        assert restored.model_type == metadata.model_type


@pytest.mark.skipif(not DEEPLEARNING_IMPORTED, reason="Deep learning module not imported")
class TestTrainingConfig:
    """Test training configuration."""
    
    def test_training_config_creation(self):
        """Test creating training configuration."""
        try:
            from agentnet.deeplearning.trainer import TrainingConfig
            
            config = TrainingConfig(
                model_name="test-model",
                output_dir="./output",
                learning_rate=1e-5,
                batch_size=16,
                num_epochs=5
            )
            
            assert config.model_name == "test-model"
            assert config.learning_rate == 1e-5
            assert config.batch_size == 16
            assert config.num_epochs == 5
        except ImportError:
            pytest.skip("TrainingConfig not available")
    
    def test_training_config_to_dict(self):
        """Test training config serialization."""
        try:
            from agentnet.deeplearning.trainer import TrainingConfig
            
            config = TrainingConfig(
                model_name="test-model",
                output_dir="./output"
            )
            
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert "model_name" in config_dict
            assert "learning_rate" in config_dict
        except ImportError:
            pytest.skip("TrainingConfig not available")


@pytest.mark.skipif(not DEEPLEARNING_IMPORTED, reason="Deep learning module not imported")
class TestLoRAConfig:
    """Test LoRA configuration."""
    
    def test_lora_config_creation(self):
        """Test creating LoRA configuration."""
        try:
            from agentnet.deeplearning.finetuning import LoRAConfig
            
            config = LoRAConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1
            )
            
            assert config.r == 8
            assert config.lora_alpha == 32
            assert len(config.target_modules) == 2
            assert config.lora_dropout == 0.1
        except ImportError:
            pytest.skip("LoRAConfig not available")


@pytest.mark.skipif(not DEEPLEARNING_IMPORTED, reason="Deep learning module not imported")
class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_embedding_cache_creation(self):
        """Test creating embedding cache."""
        try:
            from agentnet.deeplearning.embeddings import EmbeddingCache
            
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = EmbeddingCache(cache_dir=tmpdir, max_size=100)
                assert cache.cache_dir == Path(tmpdir)
                assert cache.max_size == 100
        except ImportError:
            pytest.skip("EmbeddingCache not available")


class TestPhase9Integration:
    """Test Phase 9 integration with main AgentNet module."""
    
    def test_phase9_in_phase_status(self):
        """Test that Phase 9 is in phase status."""
        import agentnet
        assert "P9" in agentnet.__phase_status__
    
    def test_phase9_exports(self):
        """Test that Phase 9 exports are available."""
        import agentnet
        
        # These should be available even if stubbed
        assert hasattr(agentnet, 'ModelRegistry')
        assert hasattr(agentnet, 'DeepLearningTrainer')
        assert hasattr(agentnet, 'FineTuner')
        assert hasattr(agentnet, 'EmbeddingGenerator')
        assert hasattr(agentnet, 'NeuralReasoner')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
