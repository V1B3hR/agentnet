"""
Model Registry - Central repository for managing trained models

This module provides model versioning, metadata tracking, and storage
management for deep learning models in AgentNet.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models that can be registered."""
    LANGUAGE_MODEL = "language_model"
    EMBEDDING_MODEL = "embedding_model"
    REASONING_MODEL = "reasoning_model"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """Status of a model in the registry."""
    TRAINING = "training"
    READY = "ready"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    
    # Identifiers
    model_id: str
    name: str
    version: str
    
    # Model information
    model_type: ModelType
    status: ModelStatus = ModelStatus.TRAINING
    
    # Training information
    base_model: Optional[str] = None
    training_date: datetime = field(default_factory=datetime.now)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Storage
    local_path: Optional[Path] = None
    remote_url: Optional[str] = None
    
    # Lineage
    parent_model_id: Optional[str] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    
    # File information
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "status": self.status.value,
            "base_model": self.base_model,
            "training_date": self.training_date.isoformat(),
            "training_config": self.training_config,
            "metrics": self.metrics,
            "local_path": str(self.local_path) if self.local_path else None,
            "remote_url": self.remote_url,
            "parent_model_id": self.parent_model_id,
            "tags": self.tags,
            "description": self.description,
            "author": self.author,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        data = data.copy()
        data["model_type"] = ModelType(data["model_type"])
        data["status"] = ModelStatus(data["status"])
        data["training_date"] = datetime.fromisoformat(data["training_date"])
        if data.get("local_path"):
            data["local_path"] = Path(data["local_path"])
        return cls(**data)


@dataclass
class ModelArtifact:
    """Represents a serialized model artifact."""
    
    metadata: ModelMetadata
    model_path: Path
    config_path: Optional[Path] = None
    tokenizer_path: Optional[Path] = None
    
    def validate(self) -> bool:
        """Validate that all required files exist."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        if self.config_path and not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            return False
        
        if self.tokenizer_path and not self.tokenizer_path.exists():
            logger.error(f"Tokenizer file not found: {self.tokenizer_path}")
            return False
        
        return True
    
    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of model file."""
        sha256 = hashlib.sha256()
        with open(self.model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


class ModelRegistry:
    """
    Central repository for managing trained models.
    
    Features:
    - Model versioning and metadata tracking
    - Local and remote model storage
    - Model lineage tracking
    - Integration with Hugging Face Hub
    """
    
    def __init__(self, registry_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store registry metadata and models.
                         Defaults to ~/.agentnet/models
        """
        if registry_dir is None:
            registry_dir = Path.home() / ".agentnet" / "models"
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_dir / "registry.json"
        self._models: Dict[str, ModelMetadata] = {}
        self._load_registry()
        
        logger.info(f"Model registry initialized at {self.registry_dir}")
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, metadata_dict in data.items():
                    self._models[model_id] = ModelMetadata.from_dict(metadata_dict)
                
                logger.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._models = {}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {
                model_id: metadata.to_dict()
                for model_id, metadata in self._models.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Registry saved to disk")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register(self, metadata: ModelMetadata) -> None:
        """
        Register a new model.
        
        Args:
            metadata: Model metadata to register
        """
        if metadata.model_id in self._models:
            logger.warning(f"Model {metadata.model_id} already registered, updating")
        
        self._models[metadata.model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {metadata.name} v{metadata.version}")
    
    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Model metadata or None if not found
        """
        return self._models.get(model_id)
    
    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            tags: Filter by tags (any match)
            
        Returns:
            List of matching model metadata
        """
        models = list(self._models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        if tags:
            models = [
                m for m in models
                if any(tag in m.tags for tag in tags)
            ]
        
        return sorted(models, key=lambda m: m.training_date, reverse=True)
    
    def update_status(self, model_id: str, status: ModelStatus) -> None:
        """
        Update model status.
        
        Args:
            model_id: Model identifier
            status: New status
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self._models[model_id].status = status
        self._save_registry()
        
        logger.info(f"Updated {model_id} status to {status.value}")
    
    def update_metrics(self, model_id: str, metrics: Dict[str, float]) -> None:
        """
        Update model performance metrics.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics to update
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self._models[model_id].metrics.update(metrics)
        self._save_registry()
        
        logger.info(f"Updated metrics for {model_id}")
    
    def delete(self, model_id: str, delete_files: bool = False) -> None:
        """
        Delete model from registry.
        
        Args:
            model_id: Model identifier
            delete_files: Whether to delete model files from disk
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self._models[model_id]
        
        if delete_files and metadata.local_path:
            try:
                if metadata.local_path.is_file():
                    metadata.local_path.unlink()
                elif metadata.local_path.is_dir():
                    import shutil
                    shutil.rmtree(metadata.local_path)
                logger.info(f"Deleted model files at {metadata.local_path}")
            except Exception as e:
                logger.error(f"Failed to delete model files: {e}")
        
        del self._models[model_id]
        self._save_registry()
        
        logger.info(f"Deleted model {model_id} from registry")
    
    def get_lineage(self, model_id: str) -> List[ModelMetadata]:
        """
        Get the lineage (ancestry) of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of ancestor models, from oldest to newest
        """
        lineage = []
        current_id = model_id
        
        while current_id:
            metadata = self.get(current_id)
            if not metadata:
                break
            
            lineage.insert(0, metadata)
            current_id = metadata.parent_model_id
        
        return lineage
