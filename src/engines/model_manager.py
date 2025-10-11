"""
ML Model Manager.

This module provides model persistence, loading, and lifecycle management
for trained machine learning models.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import joblib

from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import (
    ModelNotFoundError,
    ModelSerializationError,
    ValidationError
)


class ModelManager:
    """
    Manages ML model persistence, loading, and lifecycle.

    Handles:
    - Model saving and loading
    - Metadata management
    - User isolation
    - Model discovery and listing
    - Model deletion and cleanup
    """

    def __init__(self, config: MLEngineConfig):
        """
        Initialize model manager.

        Args:
            config: ML Engine configuration
        """
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_user_models_dir(self, user_id: int) -> Path:
        """
        Get directory for user's models.

        Args:
            user_id: User identifier

        Returns:
            Path to user's models directory
        """
        user_dir = self.models_dir / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_model_dir(self, user_id: int, model_id: str) -> Path:
        """
        Get directory for specific model.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Returns:
            Path to model directory

        Raises:
            ModelNotFoundError: If model directory doesn't exist
        """
        model_dir = self.get_user_models_dir(user_id) / model_id
        if not model_dir.exists():
            raise ModelNotFoundError(f"Model '{model_id}' not found for user {user_id}", model_id=model_id, user_id=user_id)
        return model_dir

    def _save_auxiliary_files(
        self,
        model_dir: Path,
        scaler: Optional[Any],
        feature_info: Optional[Dict[str, Any]],
        encoders: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save scaler, feature_info, and encoders if provided."""
        if scaler is not None:
            joblib.dump(scaler, model_dir / "scaler.pkl")
        if feature_info is not None:
            with open(model_dir / "feature_names.json", 'w') as f:
                json.dump(feature_info, f, indent=2)
        if encoders is not None and len(encoders) > 0:
            joblib.dump(encoders, model_dir / "encoders.pkl")

    def _load_auxiliary_files(self, model_dir: Path) -> tuple:
        """Load scaler, feature_info, and encoders if they exist."""
        scaler = None
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        feature_info = {}
        feature_info_path = model_dir / "feature_names.json"
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)

        encoders = {}
        encoders_path = model_dir / "encoders.pkl"
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)

        return scaler, feature_info, encoders

    def save_model(
        self,
        user_id: int,
        model_id: str,
        model: Any,
        metadata: Dict[str, Any],
        scaler: Optional[Any] = None,
        feature_info: Optional[Dict[str, Any]] = None,
        encoders: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a trained model with all artifacts.

        Args:
            user_id: User identifier
            model_id: Model identifier
            model: Trained model object to save
            metadata: Model metadata (metrics, config, etc.)
            scaler: Preprocessing scaler (optional)
            feature_info: Feature names and configuration (optional)
            encoders: Categorical feature encoders (optional)

        Raises:
            ModelSerializationError: If saving fails
        """
        try:
            # Create model directory
            model_dir = self.get_user_models_dir(user_id) / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Detect if Keras model
            is_keras = hasattr(model, 'to_json') and hasattr(model, 'save_weights')

            if is_keras:
                # Save Keras model
                # 1. Save architecture as JSON
                model_json = model.to_json()
                with open(model_dir / "model.json", "w") as json_file:
                    json_file.write(model_json)

                # 2. Save weights as H5 (Keras 3.x requires .weights.h5 extension)
                model.save_weights(str(model_dir / "model.weights.h5"))

                # 3. Mark as Keras in metadata
                metadata["model_format"] = "keras"
            else:
                # Save sklearn model
                model_path = model_dir / "model.pkl"
                joblib.dump(model, model_path)
                metadata["model_format"] = "sklearn"

            # Add timestamp to metadata
            metadata["model_id"] = model_id
            metadata["user_id"] = user_id
            metadata["created_at"] = datetime.utcnow().isoformat() + "Z"

            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save auxiliary files
            self._save_auxiliary_files(model_dir, scaler, feature_info, encoders)

        except Exception as e:
            # Clean up on failure
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise ModelSerializationError(
                f"Failed to save model '{model_id}': {e}",
                model_id=model_id,
                operation="save"
            )

    def load_model(self, user_id: int, model_id: str) -> Dict[str, Any]:
        """
        Load a trained model with all artifacts.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Returns:
            Dictionary containing:
                - model: Trained model object
                - metadata: Model metadata
                - scaler: Preprocessing scaler (if used)
                - feature_info: Feature names and configuration

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelSerializationError: If loading fails
        """
        try:
            model_dir = self.get_model_dir(user_id, model_id)

            # Load metadata first to check format
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                raise ModelNotFoundError(f"Metadata not found for model '{model_id}'", model_id=model_id, user_id=user_id)

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check model format
            model_format = metadata.get("model_format", "sklearn")

            if model_format == "keras":
                # Load Keras model
                model_json_path = model_dir / "model.json"
                model_weights_path = model_dir / "model.weights.h5"

                if not model_json_path.exists() or not model_weights_path.exists():
                    raise ModelNotFoundError(f"Keras model files not found for model '{model_id}'", model_id=model_id, user_id=user_id)

                # Import Keras
                from tensorflow.keras.models import model_from_json

                # Load architecture
                with open(model_json_path, 'r') as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json)

                # Load weights
                model.load_weights(str(model_weights_path))

            else:
                # Load sklearn model
                model_path = model_dir / "model.pkl"
                if not model_path.exists():
                    raise ModelNotFoundError(f"Model file not found for model '{model_id}'", model_id=model_id, user_id=user_id)

                model = joblib.load(model_path)

            # Load auxiliary files
            scaler, feature_info, encoders = self._load_auxiliary_files(model_dir)

            return {
                "model": model,
                "metadata": metadata,
                "scaler": scaler,
                "feature_info": feature_info,
                "encoders": encoders
            }

        except (ModelNotFoundError, ValidationError):
            raise
        except Exception as e:
            raise ModelSerializationError(
                f"Failed to load model '{model_id}': {e}",
                model_id=model_id,
                error_details=str(e)
            )

    def get_model_metadata(self, user_id: int, model_id: str) -> Dict[str, Any]:
        """
        Get model metadata without loading the model.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Returns:
            Model metadata dictionary

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model_dir = self.get_model_dir(user_id, model_id)
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            raise ModelNotFoundError(f"Metadata not found for model '{model_id}'", model_id=model_id, user_id=user_id)

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_user_models(
        self,
        user_id: int,
        task_type: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all models for a user with optional filtering.

        Args:
            user_id: User identifier
            task_type: Filter by task type (e.g., "regression", "classification")
            model_type: Filter by model type (e.g., "linear", "random_forest")

        Returns:
            List of model metadata dictionaries
        """
        user_dir = self.get_user_models_dir(user_id)
        models = []

        # Iterate through model directories
        for model_dir in user_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Apply filters
                if task_type and metadata.get("task_type") != task_type:
                    continue

                if model_type and metadata.get("model_type") != model_type:
                    continue

                models.append(metadata)

            except Exception:
                # Skip models with corrupted metadata
                continue

        # Sort by creation date (newest first)
        models.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )

        return models

    def delete_model(self, user_id: int, model_id: str) -> None:
        """
        Delete a model and all its artifacts.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model_dir = self.get_model_dir(user_id, model_id)

        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            raise ModelSerializationError(
                f"Failed to delete model '{model_id}': {e}",
                model_id=model_id,
                error_details=str(e)
            )

    def get_model_size(self, user_id: int, model_id: str) -> float:
        """
        Get total size of model in MB.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Returns:
            Model size in MB

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model_dir = self.get_model_dir(user_id, model_id)

        total_size = 0
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)  # Convert to MB

    def count_user_models(self, user_id: int) -> int:
        """
        Count total models for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of models
        """
        return len(self.list_user_models(user_id))

    def enforce_model_limits(self, user_id: int) -> None:
        """
        Enforce per-user model limits by deleting oldest models.

        Args:
            user_id: User identifier
        """
        models = self.list_user_models(user_id)

        # Check if we exceed the limit
        if len(models) <= self.config.max_models_per_user:
            return

        # Delete oldest models
        models_to_delete = len(models) - self.config.max_models_per_user

        for i in range(models_to_delete):
            model_id = models[-(i + 1)]["model_id"]
            try:
                self.delete_model(user_id, model_id)
            except Exception:
                # Continue even if deletion fails
                pass

    def validate_model_size(self, user_id: int, model_id: str) -> None:
        """
        Validate that model size is within limits.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Raises:
            ValidationError: If model exceeds size limit
        """
        size_mb = self.get_model_size(user_id, model_id)

        if size_mb > self.config.max_model_size_mb:
            raise ValidationError(
                f"Model size ({size_mb:.2f}MB) exceeds limit "
                f"({self.config.max_model_size_mb}MB)",
                field="model_size",
                value=size_mb
            )

    def get_model_summary(self, user_id: int, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive model summary.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Returns:
            Dictionary with model summary information
        """
        metadata = self.get_model_metadata(user_id, model_id)
        size_mb = self.get_model_size(user_id, model_id)

        return {
            "model_id": model_id,
            "user_id": user_id,
            "model_type": metadata.get("model_type"),
            "task_type": metadata.get("task_type"),
            "created_at": metadata.get("created_at"),
            "size_mb": round(size_mb, 2),
            "metrics": metadata.get("metrics", {}),
            "n_features": len(metadata.get("feature_columns", [])),
            "features": metadata.get("feature_columns", []),
            "target": metadata.get("target_column")
        }

    def cleanup_old_models(self, user_id: int, days: int = 30) -> int:
        """
        Delete models older than specified days.

        Args:
            user_id: User identifier
            days: Age threshold in days

        Returns:
            Number of models deleted
        """
        models = self.list_user_models(user_id)
        deleted_count = 0

        cutoff_date = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)

        for metadata in models:
            created_at_str = metadata.get("created_at", "")
            try:
                # Parse ISO format timestamp
                created_at = datetime.fromisoformat(
                    created_at_str.replace("Z", "+00:00")
                ).timestamp()

                if created_at < cutoff_date:
                    model_id = metadata["model_id"]
                    self.delete_model(user_id, model_id)
                    deleted_count += 1

            except Exception:
                # Skip models with invalid timestamps
                continue

        return deleted_count

    def __repr__(self) -> str:
        """String representation."""
        return f"ModelManager(models_dir='{self.models_dir}')"
