"""
Neural Network Model Trainer.

This module provides the trainer for neural network models using scikit-learn's
MLPClassifier and MLPRegressor for both classification and regression tasks.
"""

from typing import Any, Dict, Literal
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError


class NeuralNetworkTrainer(ModelTrainer):
    """
    Trainer for neural network models.

    Supports: mlp_regression, mlp_classification
    Uses scikit-learn's Multi-layer Perceptron (MLP) for both tasks.
    """

    # Supported model types
    SUPPORTED_MODELS = ["mlp_regression", "mlp_classification"]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize neural network trainer.

        Args:
            config: ML Engine configuration
        """
        super().__init__(config)

    def get_model_instance(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """
        Create neural network model instance.

        Args:
            model_type: Type of neural network model
            hyperparameters: Model hyperparameters

        Returns:
            Instantiated model object

        Raises:
            TrainingError: If model_type is not supported
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown neural network model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        # Merge with defaults
        params = self.merge_hyperparameters(model_type, hyperparameters)

        # Get hidden layer sizes
        hidden_layers = params.get('hidden_layers', [100])
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        hidden_layer_sizes = tuple(hidden_layers)

        try:
            if model_type == "mlp_regression":
                return MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=params.get('activation', 'relu'),
                    solver=params.get('solver', 'adam'),
                    alpha=params.get('alpha', 0.0001),
                    learning_rate=params.get('learning_rate', 'constant'),
                    learning_rate_init=params.get('learning_rate_init', 0.001),
                    max_iter=params.get('max_iter', 200),
                    early_stopping=params.get('early_stopping', False),
                    validation_fraction=params.get('validation_fraction', 0.1),
                    random_state=42
                )

            elif model_type == "mlp_classification":
                return MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=params.get('activation', 'relu'),
                    solver=params.get('solver', 'adam'),
                    alpha=params.get('alpha', 0.0001),
                    learning_rate=params.get('learning_rate', 'constant'),
                    learning_rate_init=params.get('learning_rate_init', 0.001),
                    max_iter=params.get('max_iter', 200),
                    early_stopping=params.get('early_stopping', False),
                    validation_fraction=params.get('validation_fraction', 0.1),
                    random_state=42
                )

        except Exception as e:
            raise TrainingError(
                f"Failed to create {model_type} model: {e}",
                model_type=model_type,
                error_details=str(e)
            )

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: Any = None
    ) -> Dict[str, float]:
        """
        Calculate metrics (regression or classification based on data).

        Args:
            y_true: True target values
            y_pred: Predicted values or class labels
            y_proba: Predicted probabilities (for classification, optional)

        Returns:
            Dictionary of metrics
        """
        try:
            # Detect if this is regression or classification
            # Check if predictions are continuous or discrete
            unique_true = len(np.unique(y_true))
            unique_pred = len(np.unique(y_pred))

            # If few unique values and predictions match true labels closely,
            # treat as classification
            is_classification = (unique_true < 20 and unique_pred < 20)

            if is_classification:
                return self._calculate_classification_metrics(y_true, y_pred, y_proba)
            else:
                return self._calculate_regression_metrics(y_true, y_pred)

        except Exception as e:
            # Return default metrics on error
            return {
                "error": str(e),
                "accuracy": float('nan') if is_classification else None,
                "mse": float('nan') if not is_classification else None
            }

    def _calculate_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mae),
            "r2": float(r2),
            "explained_variance": float(explained_var)
        }

    def _calculate_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: Any = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        n_classes = len(np.unique(y_true))
        average_method = 'binary' if n_classes == 2 else 'weighted'

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

        # Add ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class='ovr',
                        average='weighted'
                    )
                metrics["roc_auc"] = float(roc_auc)
            except Exception:
                pass

        # Add confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = conf_matrix.tolist()

        return metrics

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train neural network model.

        Args:
            model: Neural network model instance
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model

        Raises:
            TrainingError: If training fails
        """
        try:
            model.fit(X_train, y_train)
            return model

        except Exception as e:
            raise TrainingError(
                f"Neural network training failed: {e}",
                model_type=str(type(model).__name__),
                error_details=str(e)
            )

    def get_model_summary(
        self,
        model: Any,
        model_type: str,
        feature_names: list[str]
    ) -> Dict[str, Any]:
        """
        Get summary information about trained neural network.

        Args:
            model: Trained model instance
            model_type: Type of model
            feature_names: List of feature names

        Returns:
            Dictionary with model summary information
        """
        summary = {
            "model_type": model_type,
            "n_features": len(feature_names),
            "feature_names": feature_names
        }

        # Add neural network architecture info
        if hasattr(model, 'hidden_layer_sizes'):
            summary["hidden_layer_sizes"] = list(model.hidden_layer_sizes)
            summary["n_layers"] = len(model.hidden_layer_sizes) + 1  # +1 for output layer

        if hasattr(model, 'n_iter_'):
            summary["n_iterations"] = int(model.n_iter_)

        if hasattr(model, 'activation'):
            summary["activation"] = str(model.activation)

        if hasattr(model, 'solver'):
            summary["solver"] = str(model.solver)

        if hasattr(model, 'alpha'):
            summary["alpha"] = float(model.alpha)

        if hasattr(model, 'learning_rate'):
            summary["learning_rate"] = str(model.learning_rate)

        if hasattr(model, 'learning_rate_init'):
            summary["learning_rate_init"] = float(model.learning_rate_init)

        # Add number of parameters
        if hasattr(model, 'coefs_'):
            n_params = sum(coef.size for coef in model.coefs_)
            n_params += sum(intercept.size for intercept in model.intercepts_)
            summary["n_parameters"] = int(n_params)

        # Add convergence info
        if hasattr(model, 'loss_'):
            summary["final_loss"] = float(model.loss_)

        # Add number of classes for classification
        if hasattr(model, 'classes_'):
            summary["n_classes"] = len(model.classes_)
            summary["classes"] = model.classes_.tolist()

        # Note: MLPs don't have traditional feature importance
        # We could compute permutation importance but that's expensive
        summary["feature_importance"] = None

        return summary

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """
        Get list of supported model types.

        Returns:
            List of supported model type names
        """
        return cls.SUPPORTED_MODELS.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"NeuralNetworkTrainer(models={self.SUPPORTED_MODELS})"
