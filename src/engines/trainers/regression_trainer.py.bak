"""
Regression Model Trainer.

This module provides the trainer for regression models including
linear regression, ridge, lasso, elastic net, and polynomial regression.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError


class RegressionTrainer(ModelTrainer):
    """
    Trainer for regression models.

    Supports: linear, ridge, lasso, elasticnet, polynomial
    """

    # Supported model types
    SUPPORTED_MODELS = ["linear", "ridge", "lasso", "elasticnet", "polynomial"]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize regression trainer.

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
        Create regression model instance.

        Args:
            model_type: Type of regression model
            hyperparameters: Model hyperparameters

        Returns:
            Instantiated model object

        Raises:
            TrainingError: If model_type is not supported
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown regression model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        # Merge with defaults
        params = self.merge_hyperparameters(model_type, hyperparameters)

        try:
            if model_type == "linear":
                return LinearRegression()

            elif model_type == "ridge":
                return Ridge(
                    alpha=params.get('alpha', 1.0),
                    max_iter=params.get('max_iter', 1000),
                    random_state=42
                )

            elif model_type == "lasso":
                return Lasso(
                    alpha=params.get('alpha', 1.0),
                    max_iter=params.get('max_iter', 1000),
                    random_state=42
                )

            elif model_type == "elasticnet":
                return ElasticNet(
                    alpha=params.get('alpha', 1.0),
                    l1_ratio=params.get('l1_ratio', 0.5),
                    max_iter=params.get('max_iter', 1000),
                    random_state=42
                )

            elif model_type == "polynomial":
                degree = params.get('degree', 2)
                return Pipeline([
                    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                    ('linear', LinearRegression())
                ])

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
        Calculate regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_proba: Not used for regression (kept for interface compatibility)

        Returns:
            Dictionary of regression metrics
        """
        try:
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

        except Exception as e:
            # Return default metrics on error
            return {
                "mse": float('nan'),
                "rmse": float('nan'),
                "mae": float('nan'),
                "r2": float('nan'),
                "explained_variance": float('nan'),
                "error": str(e)
            }

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train regression model.

        Args:
            model: Regression model instance
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
                f"Model training failed: {e}",
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
        Get summary information about trained model.

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

        # Add model-specific information
        if model_type == "polynomial":
            # For pipeline, extract the polynomial degree
            if hasattr(model, 'named_steps'):
                poly_features = model.named_steps.get('poly')
                if poly_features:
                    summary["polynomial_degree"] = poly_features.degree

        elif hasattr(model, 'alpha'):
            # Ridge, Lasso, ElasticNet
            summary["alpha"] = float(model.alpha)

        if hasattr(model, 'l1_ratio'):
            # ElasticNet
            summary["l1_ratio"] = float(model.l1_ratio)

        # Add coefficients and intercept for linear models
        if hasattr(model, 'coef_'):
            # Extract coefficients (slope values)
            coef = model.coef_
            if hasattr(coef, 'tolist'):
                summary["coefficients"] = {
                    feature_names[i]: float(coef[i])
                    for i in range(len(feature_names))
                }
            else:
                summary["coefficients"] = {feature_names[0]: float(coef)}

        if hasattr(model, 'intercept_'):
            summary["intercept"] = float(model.intercept_)

        # Add feature importance if available
        feature_importance = self.get_feature_importance(model, feature_names)
        if feature_importance:
            summary["feature_importance"] = feature_importance

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
        return f"RegressionTrainer(models={self.SUPPORTED_MODELS})"
