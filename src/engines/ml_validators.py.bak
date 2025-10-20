"""
ML Input Validation Functions.

This module provides validation functions for ML operations including
data validation, hyperparameter validation, and model access validation.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.utils.exceptions import (
    DataValidationError,
    HyperparameterError,
    ModelNotFoundError,
    ValidationError
)


class MLValidators:
    """Input validation for ML operations."""

    @staticmethod
    def validate_training_data(
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        min_samples: int = 10
    ) -> None:
        """
        Validate training data for ML operations.

        Args:
            data: Training dataframe
            target_column: Name of target column
            feature_columns: List of feature column names
            min_samples: Minimum required samples

        Raises:
            DataValidationError: If data is invalid for training
        """
        # Check dataframe not empty
        if data.empty:
            raise DataValidationError(
                "Training data is empty",
                data_shape=(0, 0)
            )

        # Check sufficient samples
        if len(data) < min_samples:
            raise DataValidationError(
                f"Insufficient training data: need at least {min_samples} samples, "
                f"but only {len(data)} provided",
                data_shape=data.shape
            )

        # Check target column exists
        if target_column not in data.columns:
            raise DataValidationError(
                f"Target column '{target_column}' not found in data",
                data_shape=data.shape,
                missing_columns=[target_column]
            )

        # Check feature columns exist
        missing_features = set(feature_columns) - set(data.columns)
        if missing_features:
            raise DataValidationError(
                f"Feature columns not found in data: {missing_features}",
                data_shape=data.shape,
                missing_columns=list(missing_features)
            )

        # Check for target variability
        if data[target_column].nunique() == 1:
            raise DataValidationError(
                "Target column has no variance (all values are the same). "
                "Cannot train a model with constant target.",
                data_shape=data.shape
            )

        # Check for sufficient non-null values in target
        null_count = data[target_column].isnull().sum()
        if null_count == len(data):
            raise DataValidationError(
                "Target column contains only null values",
                data_shape=data.shape
            )

        # Warn if too many nulls in target (>50%)
        if null_count > len(data) * 0.5:
            raise DataValidationError(
                f"Target column has {null_count} null values "
                f"({null_count/len(data)*100:.1f}% of data). "
                "Consider handling missing values before training.",
                data_shape=data.shape
            )

    @staticmethod
    def validate_hyperparameters(
        model_type: str,
        hyperparameters: Dict[str, Any],
        allowed_ranges: Optional[Dict[str, list]] = None
    ) -> None:
        """
        Validate hyperparameter values against allowed ranges.

        Args:
            model_type: Type of model being configured
            hyperparameters: Hyperparameter dictionary to validate
            allowed_ranges: Dictionary of parameter_name -> [min, max]

        Raises:
            HyperparameterError: If any hyperparameter is invalid
        """
        if not hyperparameters:
            return  # Nothing to validate

        if not allowed_ranges:
            return  # No validation rules defined

        for param_name, param_value in hyperparameters.items():
            if param_name in allowed_ranges:
                range_vals = allowed_ranges[param_name]
                if len(range_vals) == 2:
                    min_val, max_val = range_vals

                    # Validate numeric parameters
                    if isinstance(param_value, (int, float)):
                        if not (min_val <= param_value <= max_val):
                            raise HyperparameterError(
                                f"Hyperparameter '{param_name}' value {param_value} "
                                f"outside allowed range [{min_val}, {max_val}]",
                                parameter_name=param_name,
                                parameter_value=param_value,
                                allowed_range=(min_val, max_val)
                            )

    @staticmethod
    def validate_model_id(model_id: str) -> None:
        """
        Validate model ID format.

        Args:
            model_id: Model identifier to validate

        Raises:
            ValidationError: If model ID format is invalid
        """
        # Model IDs should match pattern: model_[8 hex chars]
        if not re.match(r'^model_[a-f0-9]{8}$', model_id):
            raise ValidationError(
                f"Invalid model ID format: '{model_id}'. "
                "Expected format: model_xxxxxxxx (8 hex characters)",
                field="model_id",
                value=model_id
            )

    @staticmethod
    def validate_model_exists(
        model_id: str,
        user_id: int,
        models_dir: Path
    ) -> None:
        """
        Validate that model exists and user owns it.

        Args:
            model_id: Model identifier
            user_id: User identifier (for ownership check)
            models_dir: Base directory for model storage

        Raises:
            ModelNotFoundError: If model doesn't exist or user doesn't own it
        """
        # First validate format
        MLValidators.validate_model_id(model_id)

        # Check path exists within user's directory
        model_path = models_dir / f"user_{user_id}" / model_id

        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model '{model_id}' not found. "
                "Use /list_models to see your available models.",
                model_id=model_id,
                user_id=user_id
            )

        # Check that it's a directory
        if not model_path.is_dir():
            raise ModelNotFoundError(
                f"Model path '{model_id}' is not a valid model directory",
                model_id=model_id,
                user_id=user_id
            )

        # Check for required files
        required_files = ["model.pkl", "metadata.json", "feature_names.json"]
        missing_files = [
            f for f in required_files
            if not (model_path / f).exists()
        ]

        if missing_files:
            raise ModelNotFoundError(
                f"Model '{model_id}' is incomplete. Missing files: {missing_files}",
                model_id=model_id,
                user_id=user_id
            )

    @staticmethod
    def validate_prediction_data(
        data: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Validate prediction data matches model schema.

        Args:
            data: Input data for prediction
            metadata: Model metadata with feature information

        Raises:
            DataValidationError: If data doesn't match model schema
        """
        required_features = metadata.get('feature_columns', [])

        # Check required features present
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise DataValidationError(
                f"Prediction data missing required features: {missing_features}",
                data_shape=data.shape,
                missing_columns=list(missing_features)
            )

        # Check data not empty
        if data.empty:
            raise DataValidationError(
                "Prediction data is empty",
                data_shape=(0, 0)
            )

        # Check for excessive null values in features
        for feature in required_features:
            null_count = data[feature].isnull().sum()
            if null_count == len(data):
                raise DataValidationError(
                    f"Feature '{feature}' contains only null values in prediction data",
                    data_shape=data.shape,
                    invalid_columns=[feature]
                )

    @staticmethod
    def sanitize_column_name(name: str) -> str:
        """
        Sanitize column names for safe script generation.

        Args:
            name: Original column name

        Returns:
            Sanitized column name (alphanumeric + underscore only)
        """
        # Replace non-alphanumeric characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # Ensure it starts with letter or underscore (not digit)
        if sanitized and sanitized[0].isdigit():
            sanitized = f'_{sanitized}'

        # Handle empty case
        if not sanitized:
            sanitized = 'column'

        return sanitized

    @staticmethod
    def validate_test_size(test_size: float) -> None:
        """
        Validate train/test split ratio.

        Args:
            test_size: Proportion of data for testing

        Raises:
            ValidationError: If test_size is invalid
        """
        if not (0.0 <= test_size < 1.0):
            raise ValidationError(
                f"test_size must be between 0 and 1, got {test_size}",
                field="test_size",
                value=str(test_size)
            )

    @staticmethod
    def validate_cv_folds(cv_folds: int, data_size: int) -> None:
        """
        Validate cross-validation fold count.

        Args:
            cv_folds: Number of CV folds
            data_size: Size of training data

        Raises:
            ValidationError: If cv_folds is invalid
        """
        if cv_folds < 2:
            raise ValidationError(
                f"cv_folds must be at least 2, got {cv_folds}",
                field="cv_folds",
                value=str(cv_folds)
            )

        if cv_folds > data_size:
            raise ValidationError(
                f"cv_folds ({cv_folds}) cannot exceed data size ({data_size})",
                field="cv_folds",
                value=str(cv_folds)
            )
