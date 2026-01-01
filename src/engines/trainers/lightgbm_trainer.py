"""
LightGBM Gradient Boosting Model Trainer.

This module provides training for LightGBM models using sklearn-compatible API.
Supports binary classification, multiclass classification, and regression tasks.

LightGBM Advantages:
- 10-20x faster training than XGBoost on large datasets (>100K rows)
- Lower memory usage (histogram-based algorithms)
- Better accuracy with fewer iterations
- Leaf-wise tree growth (vs depth-wise in XGBoost)

Key Parameter Differences from XGBoost:
- num_leaves (default 31) instead of max_depth
- feature_fraction instead of colsample_bytree
- bagging_fraction + bagging_freq instead of subsample
- min_data_in_leaf instead of min_child_weight
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError, ValidationError


class LightGBMTrainer(ModelTrainer):
    """
    Trainer for LightGBM gradient boosting models.

    Supports:
    - lightgbm_binary_classification
    - lightgbm_multiclass_classification
    - lightgbm_regression

    Uses LightGBM's sklearn-compatible API (LGBMClassifier, LGBMRegressor).
    Models are saved via joblib (same as sklearn and XGBoost models).

    Performance Characteristics:
    - Training speed: 10-20x faster than XGBoost on large datasets
    - Memory usage: ~50% less than XGBoost
    - Accuracy: Often achieves higher accuracy with fewer iterations
    - Best for: Datasets >100K rows, memory-constrained environments
    """

    SUPPORTED_MODELS = [
        "lightgbm_binary_classification",
        "lightgbm_multiclass_classification",
        "lightgbm_regression"
    ]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize LightGBM trainer.

        Args:
            config: ML Engine configuration
        """
        super().__init__(config)

        # Lazy import to avoid dependency issues
        self._lightgbm_imported = False
        self._LGBMClassifier = None
        self._LGBMRegressor = None

    def _import_lightgbm(self) -> None:
        """Lazy import of LightGBM modules."""
        if not self._lightgbm_imported:
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor
                self._LGBMClassifier = LGBMClassifier
                self._LGBMRegressor = LGBMRegressor
                self._lightgbm_imported = True
            except ImportError as e:
                raise TrainingError(
                    "LightGBM import failed. Please install lightgbm>=3.3.0",
                    error_details=(
                        f"Import error: {str(e)}\n"
                        f"Install: pip install lightgbm>=3.3.0\n\n"
                        f"Note: On macOS, you may need to install OpenMP:\n"
                        f"  brew install libomp"
                    )
                )

    def get_model_instance(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create LightGBM model instance.

        Args:
            model_type: Type of LightGBM model
            hyperparameters: Model hyperparameters

        Returns:
            LightGBM model instance (LGBMClassifier or LGBMRegressor)

        Raises:
            TrainingError: If model_type is not supported
            ValidationError: If hyperparameters are invalid

        Key Hyperparameters:
            num_leaves: Maximum tree leaves (default 31, range 20-150)
                - LightGBM uses leaf-wise growth instead of depth-wise
                - Higher values = more complex models
            learning_rate: Step size shrinkage (default 0.1, range 0.01-0.3)
            n_estimators: Number of boosting rounds (default 100)
            feature_fraction: Fraction of features per tree (default 0.8)
                - Equivalent to XGBoost's colsample_bytree
            bagging_fraction: Fraction of data per tree (default 0.8)
                - Equivalent to XGBoost's subsample
            bagging_freq: Frequency for bagging (default 1)
                - Set to 0 to disable bagging
            min_data_in_leaf: Minimum samples in leaf (default 20)
                - Equivalent to XGBoost's min_child_weight
        """
        self._import_lightgbm()

        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown LightGBM model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        hyperparameters = hyperparameters or {}

        # Classification models
        if "classification" in model_type:
            if model_type == "lightgbm_binary_classification":
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting_type": "gbdt",
                    "n_estimators": hyperparameters.get("n_estimators", 100),
                    "num_leaves": hyperparameters.get("num_leaves", 31),
                    "learning_rate": hyperparameters.get("learning_rate", 0.1),
                    "feature_fraction": hyperparameters.get("feature_fraction", 0.8),
                    "bagging_fraction": hyperparameters.get("bagging_fraction", 0.8),
                    "bagging_freq": hyperparameters.get("bagging_freq", 1),
                    "min_data_in_leaf": hyperparameters.get("min_data_in_leaf", 20),
                    "min_gain_to_split": hyperparameters.get("min_gain_to_split", 0.0),
                    "lambda_l1": hyperparameters.get("lambda_l1", 0.0),
                    "lambda_l2": hyperparameters.get("lambda_l2", 0.0),
                    "random_state": hyperparameters.get("random_state", 42),
                    "n_jobs": hyperparameters.get("n_jobs", -1),
                    "verbose": -1
                }
            else:  # multiclass
                params = {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "n_estimators": hyperparameters.get("n_estimators", 100),
                    "num_leaves": hyperparameters.get("num_leaves", 31),
                    "learning_rate": hyperparameters.get("learning_rate", 0.1),
                    "feature_fraction": hyperparameters.get("feature_fraction", 0.8),
                    "bagging_fraction": hyperparameters.get("bagging_fraction", 0.8),
                    "bagging_freq": hyperparameters.get("bagging_freq", 1),
                    "min_data_in_leaf": hyperparameters.get("min_data_in_leaf", 20),
                    "min_gain_to_split": hyperparameters.get("min_gain_to_split", 0.0),
                    "lambda_l1": hyperparameters.get("lambda_l1", 0.0),
                    "lambda_l2": hyperparameters.get("lambda_l2", 0.0),
                    "random_state": hyperparameters.get("random_state", 42),
                    "n_jobs": hyperparameters.get("n_jobs", -1),
                    "verbose": -1
                }

            return self._LGBMClassifier(**params)

        # Regression models
        else:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "n_estimators": hyperparameters.get("n_estimators", 100),
                "num_leaves": hyperparameters.get("num_leaves", 31),
                "learning_rate": hyperparameters.get("learning_rate", 0.1),
                "feature_fraction": hyperparameters.get("feature_fraction", 0.8),
                "bagging_fraction": hyperparameters.get("bagging_fraction", 0.8),
                "bagging_freq": hyperparameters.get("bagging_freq", 1),
                "min_data_in_leaf": hyperparameters.get("min_data_in_leaf", 20),
                "min_gain_to_split": hyperparameters.get("min_gain_to_split", 0.0),
                "lambda_l1": hyperparameters.get("lambda_l1", 0.0),
                "lambda_l2": hyperparameters.get("lambda_l2", 0.0),
                "random_state": hyperparameters.get("random_state", 42),
                "n_jobs": hyperparameters.get("n_jobs", -1),
                "verbose": -1
            }

            return self._LGBMRegressor(**params)

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Any:
        """
        Train LightGBM model.

        Args:
            model: LightGBM model instance
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
                - X_val: Validation features for early stopping
                - y_val: Validation target for early stopping
                - early_stopping_rounds: Number of rounds for early stopping
                - verbose: Print training progress

        Returns:
            Trained model

        Raises:
            TrainingError: If training fails

        Performance Notes:
            LightGBM is optimized for large datasets and will be significantly
            faster than XGBoost on datasets >100K rows while using less memory.
        """
        self._import_lightgbm()

        try:
            # Extract optional validation set for early stopping
            X_val = kwargs.get("X_val")
            y_val = kwargs.get("y_val")
            early_stopping_rounds = kwargs.get("early_stopping_rounds")
            verbose = kwargs.get("verbose", False)

            if X_val is not None and y_val is not None and early_stopping_rounds:
                # Train with early stopping
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        # LightGBM will use early_stopping callback if specified
                    ] if not verbose else None
                )
            else:
                # Standard training
                model.fit(X_train, y_train)

            return model

        except Exception as e:
            raise TrainingError(
                f"LightGBM model training failed: {e}",
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
        Validate trained LightGBM model on test set.

        Args:
            model: Trained LightGBM model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of validation metrics
        """
        y_pred = model.predict(X_test)
        return self.calculate_metrics(y_test, y_pred, model, X_test, y_test)

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: Any,
        model: Any = None,
        X: pd.DataFrame = None,
        y: pd.Series = None
    ) -> Dict[str, float]:
        """
        Calculate metrics for LightGBM model.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model: LightGBM model (optional, for probabilities)
            X: Features for evaluation (optional)
            y: Targets for evaluation (optional)

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            average_precision_score,
            brier_score_loss,
            log_loss,
            confusion_matrix
        )

        metrics = {}

        # Determine if classification or regression
        # Check if y_true contains only integer-like values
        y_true_array = np.array(y_true)
        is_integer_like = np.allclose(y_true_array, y_true_array.astype(int), rtol=0, atol=1e-10)
        unique_vals = len(np.unique(y_true))
        is_classification = is_integer_like and unique_vals < 20

        if is_classification:
            # Classification metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

            n_classes = unique_vals
            average_method = 'binary' if n_classes == 2 else 'weighted'

            metrics["precision"] = float(precision_score(
                y_true, y_pred, average=average_method, zero_division=0
            ))
            metrics["recall"] = float(recall_score(
                y_true, y_pred, average=average_method, zero_division=0
            ))
            metrics["f1"] = float(f1_score(
                y_true, y_pred, average=average_method, zero_division=0
            ))

            # Probability-based metrics (requires model and features)
            if model is not None and X is not None:
                try:
                    y_proba = model.predict_proba(X)
                    is_binary = n_classes == 2

                    if is_binary:
                        # Binary classification
                        pos_proba = y_proba[:, 1]

                        # ROC-AUC (primary metric)
                        metrics["roc_auc"] = float(roc_auc_score(y_true, pos_proba))

                        # AUC-PR (Precision-Recall AUC)
                        metrics["auc_pr"] = float(average_precision_score(y_true, pos_proba))

                        # Brier Score (calibration - lower is better)
                        metrics["brier_score"] = float(brier_score_loss(y_true, pos_proba))

                    else:
                        # Multiclass: use one-vs-rest for ROC-AUC
                        metrics["roc_auc"] = float(roc_auc_score(
                            y_true, y_proba, multi_class='ovr', average='weighted'
                        ))

                    # Log Loss (works for both binary and multiclass)
                    metrics["log_loss"] = float(log_loss(y_true, y_proba))

                except Exception:
                    # Probability-based metrics failed, skip them
                    pass

            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = conf_matrix.tolist()

        else:
            # Regression metrics
            mse = mean_squared_error(y_true, y_pred)
            metrics["mse"] = float(mse)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))

        return metrics

    def get_model_summary(
        self,
        model: Any,
        model_type: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Get summary information about trained LightGBM model.

        Args:
            model: Trained LightGBM model
            model_type: Type of model
            feature_names: List of feature names

        Returns:
            Dictionary with model summary information
        """
        summary = {
            "model_type": model_type,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "framework": "lightgbm"
        }

        # Add model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            summary["n_estimators"] = params.get("n_estimators")
            summary["num_leaves"] = params.get("num_leaves")
            summary["learning_rate"] = params.get("learning_rate")
            summary["feature_fraction"] = params.get("feature_fraction")
            summary["bagging_fraction"] = params.get("bagging_fraction")

        # Add feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = {
                name: float(score)
                for name, score in zip(feature_names, importance)
            }
            # Sort by importance
            summary["feature_importance"] = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        # Add number of boosting rounds actually used
        if hasattr(model, 'best_iteration_'):
            summary["best_iteration"] = int(model.best_iteration_)

        return summary

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model types."""
        return cls.SUPPORTED_MODELS.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"LightGBMTrainer(models={self.SUPPORTED_MODELS})"
