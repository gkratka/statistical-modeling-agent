"""
ML Engine - Main Orchestrator.

This module provides the main interface for ML operations including
training, prediction, and model management.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

from src.engines.ml_config import MLEngineConfig
from src.engines.model_manager import ModelManager
from src.engines.ml_validators import MLValidators
from src.engines.ml_preprocessors import MLPreprocessors
from src.engines.trainers.regression_trainer import RegressionTrainer
from src.engines.trainers.classification_trainer import ClassificationTrainer
from src.engines.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.utils.exceptions import (
    TrainingError,
    PredictionError,
    DataValidationError,
    ValidationError
)


class MLEngine:
    """
    Main ML Engine orchestrator.

    Coordinates training, prediction, and model management operations
    across all trainer types.
    """

    def __init__(self, config: MLEngineConfig):
        """
        Initialize ML Engine.

        Args:
            config: ML Engine configuration
        """
        self.config = config
        self.model_manager = ModelManager(config)

        # Initialize trainers
        self.trainers = {
            "regression": RegressionTrainer(config),
            "classification": ClassificationTrainer(config),
            "neural_network": NeuralNetworkTrainer(config)
        }

    def get_trainer(self, task_type: str) -> Any:
        """
        Get appropriate trainer for task type.

        Args:
            task_type: Task type (regression, classification, neural_network)

        Returns:
            Trainer instance

        Raises:
            ValidationError: If task_type is unknown
        """
        if task_type not in self.trainers:
            raise ValidationError(
                f"Unknown task type: '{task_type}'. "
                f"Supported: {list(self.trainers.keys())}",
                field="task_type",
                value=task_type
            )
        return self.trainers[task_type]

    def train_model(
        self,
        data: pd.DataFrame,
        task_type: str,
        model_type: str,
        target_column: str,
        feature_columns: List[str],
        user_id: int,
        hyperparameters: Optional[Dict[str, Any]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        test_size: Optional[float] = None,
        validation_type: str = "hold_out",
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.

        Args:
            data: Training data
            task_type: Task type (regression, classification, neural_network)
            model_type: Model type (e.g., linear, random_forest, mlp_regression)
            target_column: Name of target column
            feature_columns: List of feature column names
            user_id: User identifier
            hyperparameters: Model hyperparameters (optional)
            preprocessing_config: Preprocessing configuration (optional)
            test_size: Proportion for test set (optional, uses default)
            validation_type: Validation strategy (hold_out or cross_validation)
            cv_folds: Number of CV folds (optional, uses default)

        Returns:
            Dictionary containing:
                - model_id: Generated model identifier
                - metrics: Training and test metrics
                - training_time: Time taken to train
                - model_info: Model configuration information

        Raises:
            DataValidationError: If data validation fails
            TrainingError: If training fails
            ValidationError: If parameters are invalid
        """
        # Use defaults if not provided
        if test_size is None:
            test_size = self.config.default_test_size
        if cv_folds is None:
            cv_folds = self.config.default_cv_folds
        if hyperparameters is None:
            hyperparameters = {}
        if preprocessing_config is None:
            preprocessing_config = {
                "missing_strategy": self.config.default_missing_strategy,
                "scaling": self.config.default_scaling
            }

        # Validate data
        MLValidators.validate_training_data(
            data,
            target_column=target_column,
            feature_columns=feature_columns,
            min_samples=self.config.min_training_samples
        )

        # Validate test size
        MLValidators.validate_test_size(test_size)

        # Get appropriate trainer
        trainer = self.get_trainer(task_type)

        # Verify model type is supported
        if model_type not in trainer.SUPPORTED_MODELS:
            raise ValidationError(
                f"Model type '{model_type}' not supported for task '{task_type}'. "
                f"Supported: {trainer.SUPPORTED_MODELS}",
                field="model_type",
                value=model_type
            )

        # Validate hyperparameters if ranges defined
        if self.config.hyperparameter_ranges:
            try:
                MLValidators.validate_hyperparameters(
                    model_type,
                    hyperparameters,
                    self.config.hyperparameter_ranges
                )
            except Exception:
                # Hyperparameter validation is advisory, continue
                pass

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            data,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size
        )

        # Handle missing values
        missing_strategy = preprocessing_config.get(
            "missing_strategy",
            self.config.default_missing_strategy
        )
        X_train = MLPreprocessors.handle_missing_values(
            X_train,
            strategy=missing_strategy
        )
        X_test = MLPreprocessors.handle_missing_values(
            X_test,
            strategy=missing_strategy
        )

        # Scale features
        scaling_method = preprocessing_config.get(
            "scaling",
            self.config.default_scaling
        )
        X_train_scaled, X_test_scaled, scaler = MLPreprocessors.scale_features(
            X_train,
            X_test,
            method=scaling_method
        )

        # Create model
        model = trainer.get_model_instance(model_type, hyperparameters)

        # Train model
        trained_model = trainer.train(model, X_train_scaled, y_train)

        # Validate model
        validation_results = trainer.validate_model(
            trained_model,
            X_test_scaled,
            y_test
        )

        # This would normally generate a script and execute it,
        # but for now we return the results directly
        return {
            "model_id": f"model_{user_id}_{model_type}",
            "metrics": validation_results,
            "training_time": 0.0,  # Would be calculated by executor
            "model_info": {
                "model_type": model_type,
                "task_type": task_type,
                "features": feature_columns,
                "target": target_column
            }
        }

    def predict(
        self,
        user_id: int,
        model_id: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Make predictions with a trained model.

        Args:
            user_id: User identifier
            model_id: Model identifier
            data: Input data for prediction

        Returns:
            Dictionary containing:
                - predictions: Array of predictions
                - probabilities: Class probabilities (classification only)
                - model_id: Model identifier
                - n_predictions: Number of predictions made

        Raises:
            ModelNotFoundError: If model doesn't exist
            PredictionError: If prediction fails
            DataValidationError: If input data is invalid
        """
        try:
            # Load model
            model_artifacts = self.model_manager.load_model(user_id, model_id)
            model = model_artifacts["model"]
            metadata = model_artifacts["metadata"]
            scaler = model_artifacts["scaler"]
            feature_info = model_artifacts["feature_info"]

            # Get expected features
            expected_features = metadata.get("feature_columns", [])

            # Validate prediction data
            MLValidators.validate_prediction_data(
                data,
                expected_features
            )

            # Extract features
            X = data[expected_features].copy()

            # Handle missing values (same strategy as training)
            missing_strategy = metadata.get("preprocessing", {}).get(
                "missing_value_strategy",
                "mean"
            )
            X = MLPreprocessors.handle_missing_values(X, strategy=missing_strategy)

            # Scale if scaler was used
            if scaler is not None:
                X_scaled = pd.DataFrame(
                    scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_scaled = X

            # Make predictions
            predictions = model.predict(X_scaled)

            result = {
                "predictions": predictions.tolist(),
                "model_id": model_id,
                "n_predictions": len(predictions)
            }

            # Get probabilities for classification
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                result["probabilities"] = probabilities.tolist()

                # Add class labels
                if hasattr(model, 'classes_'):
                    result["classes"] = model.classes_.tolist()

            return result

        except (ValidationError, DataValidationError):
            raise
        except Exception as e:
            raise PredictionError(
                f"Prediction failed for model '{model_id}': {e}",
                model_id=model_id,
                error_details=str(e)
            )

    def list_models(
        self,
        user_id: int,
        task_type: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List user's models with optional filtering.

        Args:
            user_id: User identifier
            task_type: Filter by task type (optional)
            model_type: Filter by model type (optional)

        Returns:
            List of model metadata dictionaries
        """
        return self.model_manager.list_user_models(
            user_id,
            task_type=task_type,
            model_type=model_type
        )

    def get_model_info(self, user_id: int, model_id: str) -> Dict[str, Any]:
        """
        Get detailed model information.

        Args:
            user_id: User identifier
            model_id: Model identifier

        Returns:
            Model summary dictionary
        """
        return self.model_manager.get_model_summary(user_id, model_id)

    def delete_model(self, user_id: int, model_id: str) -> None:
        """
        Delete a model.

        Args:
            user_id: User identifier
            model_id: Model identifier
        """
        self.model_manager.delete_model(user_id, model_id)

    def get_supported_models(self, task_type: str) -> List[str]:
        """
        Get list of supported models for a task type.

        Args:
            task_type: Task type

        Returns:
            List of supported model names
        """
        trainer = self.get_trainer(task_type)
        return trainer.get_supported_models()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MLEngine(trainers={list(self.trainers.keys())}, "
            f"models_dir='{self.config.models_dir}')"
        )
