"""
XGBoost Gradient Boosting Model Trainer.

This module provides training for XGBoost models using sklearn-compatible API.
Supports binary classification, multiclass classification, and regression tasks.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError, ValidationError


class XGBoostTrainer(ModelTrainer):
    """
    Trainer for XGBoost gradient boosting models.

    Supports:
    - xgboost_binary_classification
    - xgboost_multiclass_classification
    - xgboost_regression

    Uses XGBoost's sklearn-compatible API (XGBClassifier, XGBRegressor).
    Models are saved via joblib (same as sklearn models).
    """

    SUPPORTED_MODELS = [
        "xgboost_binary_classification",
        "xgboost_multiclass_classification",
        "xgboost_regression"
    ]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize XGBoost trainer.

        Args:
            config: ML Engine configuration
        """
        super().__init__(config)

        # Lazy import to avoid dependency issues
        self._xgboost_imported = False
        self._XGBClassifier = None
        self._XGBRegressor = None

    def _import_xgboost(self) -> None:
        """Lazy import of XGBoost modules."""
        if not self._xgboost_imported:
            try:
                from xgboost import XGBClassifier, XGBRegressor
                self._XGBClassifier = XGBClassifier
                self._XGBRegressor = XGBRegressor
                self._xgboost_imported = True
            except Exception as e:
                error_str = str(e)

                # Check for OpenMP/libomp missing error
                if 'libomp' in error_str or 'OpenMP' in error_str.lower():
                    raise TrainingError(
                        "XGBoost requires OpenMP runtime (libomp)",
                        error_details=(
                            "macOS Setup Required:\n\n"
                            "1. Install Homebrew:\n"
                            "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n\n"
                            "2. Install OpenMP:\n"
                            "   brew install libomp\n\n"
                            "3. Restart bot:\n"
                            "   pkill -9 -f telegram_bot && ./scripts/dev_start.sh\n\n"
                            "Alternative: Use 'Gradient Boosting (sklearn)' instead (no setup needed)\n\n"
                            f"Full guide: XGBOOST_SETUP.md\n\n"
                            f"Technical error: {error_str[:200]}"
                        )
                    )

                # Check for standard import error
                elif isinstance(e, ImportError):
                    raise TrainingError(
                        "XGBoost import failed. Please install xgboost>=1.7.0",
                        error_details=(
                            f"Import error: {error_str}\n"
                            f"Install: pip install \"xgboost>=1.7.0\""
                        )
                    )

                # Other errors
                else:
                    raise TrainingError(
                        f"XGBoost initialization failed: {error_str}",
                        error_details=error_str
                    )

    def get_model_instance(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create XGBoost model instance.

        Args:
            model_type: Type of XGBoost model
            hyperparameters: Model hyperparameters

        Returns:
            XGBoost model instance (XGBClassifier or XGBRegressor)

        Raises:
            TrainingError: If model_type is not supported
            ValidationError: If hyperparameters are invalid
        """
        self._import_xgboost()

        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown XGBoost model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        hyperparameters = hyperparameters or {}

        # Classification models
        if "classification" in model_type:
            if model_type == "xgboost_binary_classification":
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "n_estimators": hyperparameters.get("n_estimators", 100),
                    "max_depth": hyperparameters.get("max_depth", 6),
                    "learning_rate": hyperparameters.get("learning_rate", 0.1),
                    "subsample": hyperparameters.get("subsample", 0.8),
                    "colsample_bytree": hyperparameters.get("colsample_bytree", 0.8),
                    "min_child_weight": hyperparameters.get("min_child_weight", 1),
                    "gamma": hyperparameters.get("gamma", 0),
                    "reg_alpha": hyperparameters.get("reg_alpha", 0),
                    "reg_lambda": hyperparameters.get("reg_lambda", 1),
                    "random_state": hyperparameters.get("random_state", 42),
                    "n_jobs": hyperparameters.get("n_jobs", -1)
                }
            else:  # multiclass
                params = {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "n_estimators": hyperparameters.get("n_estimators", 100),
                    "max_depth": hyperparameters.get("max_depth", 6),
                    "learning_rate": hyperparameters.get("learning_rate", 0.1),
                    "subsample": hyperparameters.get("subsample", 0.8),
                    "colsample_bytree": hyperparameters.get("colsample_bytree", 0.8),
                    "min_child_weight": hyperparameters.get("min_child_weight", 1),
                    "gamma": hyperparameters.get("gamma", 0),
                    "reg_alpha": hyperparameters.get("reg_alpha", 0),
                    "reg_lambda": hyperparameters.get("reg_lambda", 1),
                    "random_state": hyperparameters.get("random_state", 42),
                    "n_jobs": hyperparameters.get("n_jobs", -1)
                }

            return self._XGBClassifier(**params)

        # Regression models
        else:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "n_estimators": hyperparameters.get("n_estimators", 100),
                "max_depth": hyperparameters.get("max_depth", 6),
                "learning_rate": hyperparameters.get("learning_rate", 0.1),
                "subsample": hyperparameters.get("subsample", 0.8),
                "colsample_bytree": hyperparameters.get("colsample_bytree", 0.8),
                "min_child_weight": hyperparameters.get("min_child_weight", 1),
                "gamma": hyperparameters.get("gamma", 0),
                "reg_alpha": hyperparameters.get("reg_alpha", 0),
                "reg_lambda": hyperparameters.get("reg_lambda", 1),
                "random_state": hyperparameters.get("random_state", 42),
                "n_jobs": hyperparameters.get("n_jobs", -1)
            }

            return self._XGBRegressor(**params)

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Any:
        """
        Train XGBoost model.

        Args:
            model: XGBoost model instance
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters

        Returns:
            Trained model

        Raises:
            TrainingError: If training fails
        """
        self._import_xgboost()

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
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose
                )
            else:
                # Standard training
                model.fit(X_train, y_train, verbose=verbose)

            return model

        except Exception as e:
            raise TrainingError(
                f"XGBoost model training failed: {e}",
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
        Validate trained XGBoost model on test set.

        Args:
            model: Trained XGBoost model
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
        Calculate metrics for XGBoost model.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model: XGBoost model (optional, for probabilities)
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
        Get summary information about trained XGBoost model.

        Args:
            model: Trained XGBoost model
            model_type: Type of model
            feature_names: List of feature names

        Returns:
            Dictionary with model summary information
        """
        summary = {
            "model_type": model_type,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "framework": "xgboost"
        }

        # Add model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            summary["n_estimators"] = params.get("n_estimators")
            summary["max_depth"] = params.get("max_depth")
            summary["learning_rate"] = params.get("learning_rate")

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
        if hasattr(model, 'best_iteration'):
            summary["best_iteration"] = int(model.best_iteration)

        return summary

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model types."""
        return cls.SUPPORTED_MODELS.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"XGBoostTrainer(models={self.SUPPORTED_MODELS})"
