"""MLops workflow management for AgentNet."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("agentnet.mlops")


class ModelStage(Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStatus(Enum):
    """Deployment status states."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """Model version metadata."""

    model_id: str
    version: str
    stage: ModelStage
    provider: str
    model_name: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentRecord:
    """Deployment tracking record."""

    deployment_id: str
    model_version: ModelVersion
    status: DeploymentStatus
    deployed_at: datetime = field(default_factory=datetime.now)
    environment: str = "production"
    health_check_url: Optional[str] = None
    rollback_version: Optional[str] = None
    deployment_logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLopsWorkflow:
    """MLops workflow management system."""

    def __init__(self, storage_dir: str = "mlops_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.storage_dir / "models").mkdir(exist_ok=True)
        (self.storage_dir / "deployments").mkdir(exist_ok=True)
        (self.storage_dir / "validation").mkdir(exist_ok=True)

        logger.info(f"MLopsWorkflow initialized with storage_dir: {storage_dir}")

        # In-memory caches for frequently accessed data
        self._model_cache: Dict[str, ModelVersion] = {}
        self._deployment_cache: Dict[str, DeploymentRecord] = {}

    def register_model(
        self,
        model_id: str,
        version: str,
        provider: str,
        model_name: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        performance_metrics: Optional[Dict[str, float]] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Register a new model version."""
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            stage=stage,
            provider=provider,
            model_name=model_name,
            performance_metrics=performance_metrics or {},
            deployment_config=deployment_config or {},
            metadata=metadata or {},
        )

        # Save to storage
        self._save_model_version(model_version)

        # Update cache
        cache_key = f"{model_id}:{version}"
        self._model_cache[cache_key] = model_version

        logger.info(f"Registered model {model_id}:{version} in stage {stage.value}")
        return model_version

    def validate_model(
        self,
        model_id: str,
        version: str,
        validation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run model validation pipeline."""
        try:
            model = self.get_model_version(model_id, version)
            if not model:
                return {
                    "status": "error",
                    "message": f"Model {model_id}:{version} not found",
                }

            validation_config = validation_config or {}

            # Basic validation pipeline
            results = {
                "model_id": model_id,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "validations": {},
            }

            # Schema validation
            schema_valid = self._validate_model_schema(model)
            results["validations"]["schema"] = {
                "passed": schema_valid,
                "message": "Schema validation "
                + ("passed" if schema_valid else "failed"),
            }

            # Performance validation
            perf_valid = self._validate_model_performance(model, validation_config)
            results["validations"]["performance"] = perf_valid

            # Security validation
            security_valid = self._validate_model_security(model)
            results["validations"]["security"] = security_valid

            # Overall status
            all_passed = all(
                v.get("passed", False) for v in results["validations"].values()
            )
            results["status"] = "passed" if all_passed else "failed"

            # Save validation results
            self._save_validation_results(results)

            logger.info(
                f"Validation for {model_id}:{version} - Status: {results['status']}"
            )
            return results

        except Exception as e:
            logger.error(f"Error validating {model_id}:{version}: {e}")
            return {"status": "error", "message": str(e)}

    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        cache_key = f"{model_id}:{version}"

        # Check cache first
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Load from storage
        model_file = self.storage_dir / "models" / f"{model_id}_{version}.json"
        if model_file.exists():
            try:
                with open(model_file) as f:
                    data = json.load(f)
                    model = ModelVersion(
                        model_id=data["model_id"],
                        version=data["version"],
                        stage=ModelStage(data["stage"]),
                        provider=data["provider"],
                        model_name=data["model_name"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        metadata=data.get("metadata", {}),
                        performance_metrics=data.get("performance_metrics", {}),
                        deployment_config=data.get("deployment_config", {}),
                    )
                    self._model_cache[cache_key] = model
                    return model
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to load model {cache_key}: {e}")

        return None

    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List all model versions, optionally filtered by stage."""
        models = []
        models_dir = self.storage_dir / "models"

        for model_file in models_dir.glob("*.json"):
            try:
                with open(model_file) as f:
                    data = json.load(f)
                    model_stage = ModelStage(data["stage"])

                    if stage is None or model_stage == stage:
                        model = ModelVersion(
                            model_id=data["model_id"],
                            version=data["version"],
                            stage=model_stage,
                            provider=data["provider"],
                            model_name=data["model_name"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            metadata=data.get("metadata", {}),
                            performance_metrics=data.get("performance_metrics", {}),
                            deployment_config=data.get("deployment_config", {}),
                        )
                        models.append(model)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load model from {model_file}: {e}")
                continue

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    # Private helper methods

    def _save_model_version(self, model: ModelVersion):
        """Save model version to storage."""
        model_file = (
            self.storage_dir / "models" / f"{model.model_id}_{model.version}.json"
        )
        data = {
            "model_id": model.model_id,
            "version": model.version,
            "stage": model.stage.value,
            "provider": model.provider,
            "model_name": model.model_name,
            "created_at": model.created_at.isoformat(),
            "metadata": model.metadata,
            "performance_metrics": model.performance_metrics,
            "deployment_config": model.deployment_config,
        }

        with open(model_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to storage."""
        filename = f"validation_{results['model_id']}_{results['version']}_{int(datetime.now().timestamp())}.json"
        validation_file = self.storage_dir / "validation" / filename

        with open(validation_file, "w") as f:
            json.dump(results, f, indent=2)

    def _validate_model_schema(self, model: ModelVersion) -> bool:
        """Validate model schema."""
        # Basic schema validation - ensure required fields are present
        required_fields = ["model_id", "version", "provider", "model_name"]
        for field in required_fields:
            if not getattr(model, field, None):
                return False
        return True

    def _validate_model_performance(
        self, model: ModelVersion, validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model performance metrics."""
        thresholds = validation_config.get("performance_thresholds", {})
        metrics = model.performance_metrics

        results = {"passed": True, "details": {}}

        # Define which metrics are "higher is better" vs "lower is better"
        lower_is_better_metrics = {"latency_ms", "response_time", "error_rate"}

        for metric, threshold in thresholds.items():
            if metric in metrics:
                if metric in lower_is_better_metrics:
                    # For latency, error rate, etc. - lower values are better
                    passed = metrics[metric] <= threshold
                else:
                    # For accuracy, f1_score, etc. - higher values are better
                    passed = metrics[metric] >= threshold

                results["details"][metric] = {
                    "value": metrics[metric],
                    "threshold": threshold,
                    "passed": passed,
                    "comparison": "≤" if metric in lower_is_better_metrics else "≥",
                }
                if not passed:
                    results["passed"] = False
            else:
                results["details"][metric] = {
                    "value": None,
                    "threshold": threshold,
                    "passed": False,
                    "message": f"Metric {metric} not found",
                }
                results["passed"] = False

        if not thresholds:
            # No thresholds specified, just check if metrics exist
            results["message"] = "No performance thresholds specified"

        return results

    def _validate_model_security(self, model: ModelVersion) -> Dict[str, Any]:
        """Validate model security requirements."""
        # Basic security validation
        security_checks = {
            "has_metadata": bool(model.metadata),
            "provider_trusted": model.provider
            in ["openai", "anthropic", "example", "local"],
            "deployment_config_present": bool(model.deployment_config),
        }

        all_passed = all(security_checks.values())

        return {
            "passed": all_passed,
            "checks": security_checks,
            "message": "Security validation " + ("passed" if all_passed else "failed"),
        }


__all__ = [
    "MLopsWorkflow",
    "ModelVersion",
    "DeploymentRecord",
    "ModelStage",
    "DeploymentStatus",
]
