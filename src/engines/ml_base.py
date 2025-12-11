"""
ML Base Classes and Abstractions.

This module provides the abstract base class for all ML model trainers,
defining the interface for training, validation, and model persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
        feature_columns: List[str],
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

        # Normalize binary classification targets to [0,1] range
        # Check for common binary patterns: {1,2} or {'1','2'}
        unique_vals = set(y.unique())
        if unique_vals == {1, 2}:
            # Map {1,2} to {0,1} for binary classification
            y = y.map({1: 0, 2: 1})
        elif unique_vals == {'1', '2'}:
            # Map {'1','2'} to {0,1} for binary classification
            y = y.map({'1': 0, '2': 1}).astype(int)
        # NOTE: Categorical encoding moved to AFTER train/test split (see below)

        # Handle test_size=0 (no train/test split)
        if test_size == 0.0 or test_size is None or test_size < 0.01:
            # No split - use all data for training
            X_train = X.copy()
            y_train = y.copy()
            # Return empty test sets with same structure
            X_test = X.iloc[:0].copy()  # Empty with same columns
            y_test = y.iloc[:0].copy()  # Empty with same structure
        else:
            # Split data with stratification for classification targets
            # Stratification ensures all classes appear in both train and test sets
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y  # Maintain class proportions
                )
            except ValueError:
                # Fallback to random split if stratification fails
                # (e.g., rare classes with <2 samples)
                import logging
                logging.warning(
                    "Stratified split failed (rare classes?). Using random split."
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
                )

        # Encode categorical target AFTER split (fit on train only)
        if y_train.dtype == 'object' or pd.api.types.is_categorical_dtype(y_train):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train), index=y_train.index, name=y_train.name)
            # Transform test set - handle unseen labels by mapping to most frequent
            if len(y_test) > 0:
                # Map test labels to known classes, unknown to -1 temporarily
                y_test_transformed = []
                for val in y_test:
                    if val in le.classes_:
                        y_test_transformed.append(le.transform([val])[0])
                    else:
                        # Map unseen label to most frequent training class
                        y_test_transformed.append(0)
                y_test = pd.Series(y_test_transformed, index=y_test.index, name=y_test.name)

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
        feature_names: List[str]
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
