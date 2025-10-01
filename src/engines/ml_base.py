"""
ML Base Classes and Abstractions.

This module provides the abstract base class for all ML model trainers,
defining the interface for training, validation, and model persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.engines.ml_config import MLEngineConfig


class ModelTrainer(ABC):
    """
    Abstract base class for ML model trainers.

    This class defines the interface that all specialized trainers
    (Regression, Classification, NeuralNetwork) must implement.

    Responsibilities:
    - Data preparation (split, validation)
    - Preprocessing coordination
    - Training orchestration
    - Metrics calculation
    - Model persistence coordination

    Attributes:
        config: MLEngineConfig instance with training parameters
    """

    def __init__(self, config: MLEngineConfig):
        """
        Initialize model trainer.

        Args:
            config: ML Engine configuration
        """
        self.config = config

    @abstractmethod
    def get_model_instance(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """
        Create model instance based on type and hyperparameters.

        Args:
            model_type: Type of model to create (e.g., "linear", "ridge")
            hyperparameters: Hyperparameter dictionary

        Returns:
            Instantiated model object

        Raises:
            ValueError: If model_type is not supported
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: Any = None
    ) -> Dict[str, float]:
        """
        Calculate appropriate metrics for model type.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_proba: Predicted probabilities (for classification)

        Returns:
            Dictionary of metric names to values
        """
        pass

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        test_size: float = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into train/test sets.

        Args:
            data: Full dataset
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Proportion of data for testing (default from config)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Raises:
            ValueError: If columns don't exist in data
        """
        if test_size is None:
            test_size = self.config.default_test_size

        # Extract features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def validate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Validate model on test set and calculate metrics.

        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of validation metrics
        """
        # Generate predictions
        y_pred = model.predict(X_test)

        # Get probabilities for classification if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass  # Some models may not support predict_proba

        # Calculate and return metrics
        return self.calculate_metrics(y_test, y_pred, y_proba)

    def get_feature_importance(
        self,
        model: Any,
        feature_names: list[str]
    ) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model if available.

        Args:
            model: Trained model instance
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance
        """
        # Check for feature_importances_ (tree-based models)
        if hasattr(model, "feature_importances_"):
            return {
                name: float(importance)
                for name, importance in zip(feature_names, model.feature_importances_)
            }

        # Check for coef_ (linear models)
        if hasattr(model, "coef_"):
            # Handle both 1D and 2D coefficient arrays
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multi-class: use mean absolute coefficient across classes
                coef = abs(coef).mean(axis=0)
            return {
                name: float(importance)
                for name, importance in zip(feature_names, coef)
            }

        return None

    def merge_hyperparameters(
        self,
        model_type: str,
        user_hyperparameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge user-provided hyperparameters with defaults.

        Args:
            model_type: Type of model
            user_hyperparameters: User-provided hyperparameters (may be None)

        Returns:
            Merged hyperparameter dictionary
        """
        # Start with defaults from config
        defaults = self.config.get_default_hyperparameters(model_type)

        # Merge with user-provided values
        if user_hyperparameters:
            defaults.update(user_hyperparameters)

        return defaults

    def __repr__(self) -> str:
        """String representation of trainer."""
        return f"{self.__class__.__name__}(config={self.config.models_dir})"
