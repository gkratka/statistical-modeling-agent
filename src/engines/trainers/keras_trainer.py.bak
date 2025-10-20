"""
Keras Neural Network Model Trainer.

This module provides the trainer for Keras Sequential models for classification
and regression tasks, matching the user's workflow for custom neural network architectures.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError, ValidationError


class KerasNeuralNetworkTrainer(ModelTrainer):
    """
    Trainer for Keras Sequential neural network models.

    Supports:
    - keras_binary_classification
    - keras_multiclass_classification
    - keras_regression

    Uses TensorFlow/Keras for custom architecture specification and training.
    """

    # Supported model types
    SUPPORTED_MODELS = [
        "keras_binary_classification",
        "keras_multiclass_classification",
        "keras_regression"
    ]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize Keras neural network trainer.

        Args:
            config: ML Engine configuration
        """
        super().__init__(config)

        # Lazy import to avoid dependency issues
        self._keras_imported = False
        self._keras = None
        self._Sequential = None
        self._Dense = None
        self._Dropout = None

    def _import_keras(self):
        """Lazy import of Keras modules."""
        if not self._keras_imported:
            try:
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout

                self._keras = keras
                self._Sequential = Sequential
                self._Dense = Dense
                self._Dropout = Dropout
                self._keras_imported = True
            except ImportError as e:
                import sys
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

                raise TrainingError(
                    "TensorFlow/Keras import failed. This may indicate a version conflict or missing dependencies.",
                    error_details=(
                        f"Import error: {str(e)}\n"
                        f"Python version: {python_version}\n"
                        f"Required: tensorflow>=2.12.0,<2.16.0\n"
                        f"Troubleshooting: Try 'pip install --upgrade tensorflow' or check if TensorFlow is compatible with your Python version"
                    )
                )

    def _add_dense_layer(self, model: Any, layer_spec: Dict[str, Any], input_dim: int = None):
        """Add Dense layer to model with optional input_dim for first layer."""
        kwargs = {
            "units": layer_spec.get("units", 64),
            "activation": layer_spec.get("activation", "relu"),
            "kernel_initializer": layer_spec.get("kernel_initializer", "glorot_uniform")
        }
        if input_dim is not None:
            kwargs["input_dim"] = input_dim
        model.add(self._Dense(**kwargs))

    def build_model_from_architecture(
        self,
        architecture: Dict[str, Any],
        n_features: int
    ) -> Any:
        """
        Build Keras Sequential model from architecture specification.

        Args:
            architecture: Dict with "layers" and "compile" keys
            n_features: Number of input features

        Returns:
            Compiled Keras Sequential model

        Raises:
            ValidationError: If architecture spec is invalid
            TrainingError: If model build fails
        """
        self._import_keras()

        try:
            model = self._Sequential()

            layers = architecture.get("layers", [])
            if not layers:
                raise ValidationError(
                    "Architecture must contain at least one layer",
                    field="architecture.layers",
                    value=layers
                )

            # Build layers
            for i, layer_spec in enumerate(layers):
                layer_type = layer_spec.get("type", "Dense")

                if layer_type == "Dense":
                    # First layer needs input_dim
                    self._add_dense_layer(model, layer_spec, n_features if i == 0 else None)

                elif layer_type == "Dropout":
                    model.add(self._Dropout(rate=layer_spec.get("rate", 0.5)))

                else:
                    raise ValidationError(
                        f"Unsupported layer type: '{layer_type}'",
                        field="architecture.layers.type",
                        value=layer_type
                    )

            # Compile model
            compile_config = architecture.get("compile", {})
            model.compile(
                loss=compile_config.get("loss", "binary_crossentropy"),
                optimizer=compile_config.get("optimizer", "adam"),
                metrics=compile_config.get("metrics", ["accuracy"])
            )

            return model

        except (ValidationError, TrainingError):
            raise
        except Exception as e:
            raise TrainingError(
                f"Failed to build Keras model from architecture: {e}",
                error_details=str(e)
            )

    def get_model_instance(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """
        Create Keras model instance from architecture specification.

        Args:
            model_type: Type of Keras model
            hyperparameters: Must contain "architecture" key with model spec

        Returns:
            Compiled Keras model

        Raises:
            TrainingError: If model_type is not supported
            ValidationError: If architecture is missing or invalid
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown Keras model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        architecture = hyperparameters.get("architecture")
        if not architecture:
            raise ValidationError(
                "Keras models require 'architecture' in hyperparameters",
                field="hyperparameters.architecture",
                value=None
            )

        n_features = hyperparameters.get("n_features", 1)

        return self.build_model_from_architecture(architecture, n_features)

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Any:
        """
        Train Keras model.

        Args:
            model: Keras Sequential model
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters (epochs, batch_size, verbose, etc.)

        Returns:
            Trained model

        Raises:
            TrainingError: If training fails
        """
        self._import_keras()

        try:
            # Extract training parameters
            epochs = kwargs.get("epochs", 100)
            batch_size = kwargs.get("batch_size", 32)
            verbose = kwargs.get("verbose", 1)
            validation_split = kwargs.get("validation_split", 0.0)

            # Train model
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                validation_split=validation_split
            )

            return model

        except Exception as e:
            raise TrainingError(
                f"Keras model training failed: {e}",
                model_type=str(type(model).__name__),
                error_details=str(e)
            )

    def validate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Validate trained Keras model on test set.

        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of validation metrics
        """
        return self.calculate_metrics(y_test, model.predict(X_test), model, X_test, y_test)

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: Any,
        model: Any = None,
        X: pd.DataFrame = None,
        y: pd.Series = None
    ) -> Dict[str, float]:
        """
        Calculate metrics for Keras model.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model: Keras model (optional, for evaluate())
            X: Features for evaluation (optional)
            y: Targets for evaluation (optional)

        Returns:
            Dictionary of metrics
        """
        self._import_keras()

        try:
            metrics = {}

            # Use Keras evaluate if model and data provided
            if model is not None and X is not None and y is not None:
                eval_result = model.evaluate(X, y, verbose=0)

                # eval_result is [loss, metric1, metric2, ...]
                if isinstance(eval_result, list):
                    metrics["loss"] = abs(float(eval_result[0]))  # Ensure positive loss
                    if len(eval_result) > 1:
                        metrics["accuracy"] = float(eval_result[1])
                else:
                    metrics["loss"] = abs(float(eval_result))  # Ensure positive loss

            # Calculate additional metrics using sklearn
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix
            )

            # Determine if classification or regression
            unique_vals = len(np.unique(y_true))
            is_classification = unique_vals < 20

            if is_classification:
                # Convert predictions to class labels
                if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                    y_pred_class = (y_pred > 0.5).astype(int).flatten()
                else:
                    y_pred_class = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred

                # Classification metrics
                if "accuracy" not in metrics:
                    metrics["accuracy"] = float(accuracy_score(y_true, y_pred_class))

                n_classes = unique_vals
                average_method = 'binary' if n_classes == 2 else 'weighted'

                metrics["precision"] = float(precision_score(
                    y_true, y_pred_class, average=average_method, zero_division=0
                ))
                metrics["recall"] = float(recall_score(
                    y_true, y_pred_class, average=average_method, zero_division=0
                ))
                metrics["f1"] = float(f1_score(
                    y_true, y_pred_class, average=average_method, zero_division=0
                ))

                # Confusion matrix
                conf_matrix = confusion_matrix(y_true, y_pred_class)
                metrics["confusion_matrix"] = conf_matrix.tolist()

            else:
                # Regression metrics
                y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

                mse = mean_squared_error(y_true, y_pred_flat)
                metrics["mse"] = float(mse)
                metrics["rmse"] = float(np.sqrt(mse))
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred_flat))
                metrics["r2"] = float(r2_score(y_true, y_pred_flat))

            return metrics

        except Exception as e:
            # Return basic metrics on error
            return {
                "error": str(e),
                "loss": float('nan')
            }

    def get_model_summary(
        self,
        model: Any,
        model_type: str,
        feature_names: list
    ) -> Dict[str, Any]:
        """
        Get summary information about trained Keras model.

        Args:
            model: Trained Keras model
            model_type: Type of model
            feature_names: List of feature names

        Returns:
            Dictionary with model summary information
        """
        summary = {
            "model_type": model_type,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "framework": "keras"
        }

        # Add model architecture info
        if hasattr(model, 'layers'):
            summary["n_layers"] = len(model.layers)

        # Add total parameters
        if hasattr(model, 'count_params'):
            summary["n_parameters"] = int(model.count_params())

        # Add optimizer info
        if hasattr(model, 'optimizer'):
            summary["optimizer"] = str(model.optimizer.__class__.__name__)

        return summary

    @classmethod
    def get_supported_models(cls) -> list:
        """
        Get list of supported model types.

        Returns:
            List of supported model type names
        """
        return cls.SUPPORTED_MODELS.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"KerasNeuralNetworkTrainer(models={self.SUPPORTED_MODELS})"
