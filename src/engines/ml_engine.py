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

    def get_trainer(self, task_type: str, model_type: Optional[str] = None) -> Any:
        """
        Get appropriate trainer for task type and model type.

        Args:
            task_type: Task type (regression, classification, neural_network)
            model_type: Model type (optional, used for Keras routing)

        Returns:
            Trainer instance

        Raises:
            ValidationError: If task_type is unknown
        """
        # Check if Keras model (prefix-based detection)
        if model_type and model_type.startswith("keras_"):
            from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
            return KerasNeuralNetworkTrainer(self.config)

        # Otherwise use existing trainers
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
        data: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
        task_type: str = None,
        model_type: str = None,
        target_column: str = None,
        feature_columns: List[str] = None,
        user_id: int = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        test_size: Optional[float] = None,
        validation_type: str = "hold_out",
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train a machine learning model with support for lazy loading.

        Args:
            data: Training data (optional if file_path provided)
            file_path: Path to data file for lazy loading (optional if data provided)
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
        # Lazy loading support: load data from file_path if data not provided
        if data is None and file_path is not None:
            try:
                # Import data loader and handle async loading
                import pandas as pd

                # Simple synchronous loading for ML engine
                # (DataLoader.load_from_local_path is async, so we use pandas directly)
                file_ext = Path(file_path).suffix.lower()

                if file_ext == '.csv':
                    data = pd.read_csv(file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    data = pd.read_excel(file_path)
                elif file_ext == '.parquet':
                    data = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")

            except Exception as e:
                raise DataValidationError(
                    f"Failed to load data from {file_path}: {str(e)}"
                )

        # Validate that we have data one way or another
        if data is None:
            raise ValidationError("Either 'data' or 'file_path' must be provided")
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

        # Get appropriate trainer (pass model_type for Keras routing)
        trainer = self.get_trainer(task_type, model_type)

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

        # Handle missing values BEFORE train/test split
        # This is critical for 'drop' strategy to maintain consistent indices
        missing_strategy = preprocessing_config.get(
            "missing_strategy",
            self.config.default_missing_strategy
        )

        if missing_strategy == "drop":
            # Drop rows with missing values in features or target
            data_clean = data.dropna(subset=feature_columns + [target_column])
        else:
            # For other strategies, we'll handle after split
            data_clean = data.copy()

        # Prepare data (train/test split)
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            data_clean,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size
        )

        # Handle missing values for imputation strategies (mean, median, etc.)
        if missing_strategy != "drop":
            X_train = MLPreprocessors.handle_missing_values(
                X_train,
                strategy=missing_strategy
            )
            X_test = MLPreprocessors.handle_missing_values(
                X_test,
                strategy=missing_strategy
            )

        # Encode categorical variables (must be done before scaling)
        X_train, X_test, encoders = MLPreprocessors.encode_categorical(
            X_train,
            X_test
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

        # Check if Keras model
        is_keras = model_type.startswith("keras_")

        if is_keras:
            # Keras-specific training path
            # Add n_features to hyperparameters for architecture building
            hyperparameters["n_features"] = len(feature_columns)

            # Create model
            model = trainer.get_model_instance(model_type, hyperparameters)

            # Extract Keras training parameters
            epochs = hyperparameters.get("epochs", 100)
            batch_size = hyperparameters.get("batch_size", 32)
            verbose = hyperparameters.get("verbose", 1)
            validation_split = hyperparameters.get("validation_split", 0.0)

            # Train model with Keras parameters
            trained_model = trainer.train(
                model,
                X_train_scaled,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                validation_split=validation_split
            )

            # Validate model (use test set if available)
            if len(X_test_scaled) > 0:
                validation_results = trainer.validate_model(
                    trained_model,
                    X_test_scaled,
                    y_test
                )
            else:
                # No test set, evaluate on training data
                validation_results = trainer.calculate_metrics(
                    y_train,
                    trained_model.predict(X_train_scaled),
                    trained_model,
                    X_train_scaled,
                    y_train
                )
        else:
            # sklearn training path
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

        # Get model summary (includes coefficients, intercept, etc.)
        model_summary = trainer.get_model_summary(
            trained_model,
            model_type,
            feature_columns
        )

        # Generate unique model ID
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{user_id}_{model_type}_{timestamp}"

        # Prepare metadata for saving
        metadata = {
            "model_type": model_type,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": validation_results,
            "preprocessing": {
                "missing_value_strategy": preprocessing_config.get("missing_strategy"),
                "scaling_method": preprocessing_config.get("scaling")
            },
            "hyperparameters": hyperparameters,
            "test_size": test_size,
            **model_summary
        }

        # Prepare feature info
        feature_info = {
            "feature_names": feature_columns,
            "n_features": len(feature_columns)
        }

        # Save model with all artifacts
        self.model_manager.save_model(
            user_id=user_id,
            model_id=model_id,
            model=trained_model,
            metadata=metadata,
            scaler=scaler,
            feature_info=feature_info,
            encoders=encoders
        )

        # Return training results
        return {
            "success": True,
            "model_id": model_id,
            "metrics": validation_results,
            "training_time": 0.0,  # Would be calculated by executor
            "model_info": {
                "model_type": model_type,
                "task_type": task_type,
                "features": feature_columns,
                "target": target_column,
                **model_summary  # Include coefficients, intercept, etc.
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
            encoders = model_artifacts.get("encoders", {})

            # Validate prediction data (validator extracts features from metadata)
            MLValidators.validate_prediction_data(data, metadata)

            # Get expected features and extract them
            expected_features = metadata.get("feature_columns", [])
            X = data[expected_features].copy()

            # Handle missing values (same strategy as training)
            missing_strategy = metadata.get("preprocessing", {}).get(
                "missing_value_strategy",
                "mean"
            )
            X = MLPreprocessors.handle_missing_values(X, strategy=missing_strategy)

            # Apply categorical encoding if encoders exist (must be done before scaling)
            if encoders and len(encoders) > 0:
                for col, encoder in encoders.items():
                    if col in X.columns:
                        try:
                            # Transform categorical column using fitted encoder
                            X[col] = encoder.transform(X[col].astype(str))
                        except ValueError:
                            # Handle unseen categories: map to most frequent class
                            # This preserves model compatibility while handling edge cases
                            most_frequent = encoder.classes_[0]
                            X[col] = encoder.transform([most_frequent] * len(X))

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
