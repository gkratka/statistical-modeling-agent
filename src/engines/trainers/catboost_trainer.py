"""
CatBoost Gradient Boosting Model Trainer.

This module provides training for CatBoost models using sklearn-compatible API.
Supports binary classification, multiclass classification, and regression tasks.

CatBoost Advantages:
- Best accuracy on tabular data (often outperforms XGBoost/LightGBM)
- Native categorical feature support (no encoding needed)
- GPU acceleration with automatic CPU fallback
- Robust to overfitting with ordered boosting
- Fast inference speed
- Excellent handling of missing values

Key Parameter Differences from XGBoost/LightGBM:
- iterations (not n_estimators)
- depth (not max_depth)
- l2_leaf_reg (not reg_lambda)
- bootstrap_type (MVS, Bayesian, Bernoulli)
- rsm for feature sampling (not colsample_bytree/feature_fraction)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError, ValidationError


class CatBoostTrainer(ModelTrainer):
    """
    Trainer for CatBoost gradient boosting models.

    Supports:
    - catboost_binary_classification
    - catboost_multiclass_classification
    - catboost_regression

    Uses CatBoost's sklearn-compatible API (CatBoostClassifier, CatBoostRegressor).
    Models are saved via joblib (same as sklearn, XGBoost, and LightGBM models).

    Features:
    - Automatic GPU detection with CPU fallback
    - Automatic categorical feature detection
    - Native categorical feature handling (no encoding needed)
    - Ordered boosting for better generalization
    """

    SUPPORTED_MODELS = [
        "catboost_binary_classification",
        "catboost_multiclass_classification",
        "catboost_regression"
    ]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize CatBoost trainer.

        Args:
            config: ML Engine configuration
        """
        super().__init__(config)

        # Lazy import to avoid dependency issues
        self._catboost_imported = False
        self._CatBoostClassifier = None
        self._CatBoostRegressor = None
        self._get_gpu_device_count = None
        self._gpu_available = False

    def _import_catboost(self) -> None:
        """Lazy import of CatBoost modules with GPU detection."""
        if not self._catboost_imported:
            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
                self._CatBoostClassifier = CatBoostClassifier
                self._CatBoostRegressor = CatBoostRegressor

                # Try GPU detection
                try:
                    from catboost import get_gpu_device_count
                    self._get_gpu_device_count = get_gpu_device_count
                    self._gpu_available = get_gpu_device_count() > 0
                except Exception:
                    # GPU detection failed, use CPU
                    self._gpu_available = False

                self._catboost_imported = True

            except ImportError as e:
                raise TrainingError(
                    "CatBoost import failed. Please install catboost>=1.2.0",
                    error_details=(
                        f"Import error: {str(e)}\n"
                        f"Install: pip install catboost>=1.2.0\n\n"
                        f"Note: CatBoost supports GPU acceleration on CUDA-enabled systems"
                    )
                )

    def _detect_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        Detect categorical features in dataframe.

        CatBoost can handle categorical features natively without encoding.
        This method identifies columns with object or category dtypes.

        Args:
            X: Input features dataframe

        Returns:
            List of categorical feature column names

        Examples:
            >>> df = pd.DataFrame({
            ...     "cat1": ["A", "B", "C"],
            ...     "num1": [1, 2, 3],
            ...     "cat2": pd.Categorical(["X", "Y", "Z"])
            ... })
            >>> trainer._detect_categorical_features(df)
            ['cat1', 'cat2']
        """
        categorical_features = []

        for col in X.columns:
            # Check for object dtype (strings)
            if X[col].dtype == 'object':
                categorical_features.append(col)
            # Check for categorical dtype
            elif X[col].dtype.name == 'category':
                categorical_features.append(col)

        return categorical_features

    def get_model_instance(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create CatBoost model instance.

        Args:
            model_type: Type of CatBoost model
            hyperparameters: Model hyperparameters

        Returns:
            CatBoost model instance (CatBoostClassifier or CatBoostRegressor)

        Raises:
            TrainingError: If model_type is not supported
            ValidationError: If hyperparameters are invalid

        Key Hyperparameters:
            iterations: Number of boosting rounds (default 1000)
                - CatBoost default is higher than XGBoost/LightGBM
                - Uses early stopping by default
            depth: Maximum tree depth (default 6)
                - CatBoost is less prone to overfitting
            learning_rate: Learning rate (default 0.03)
                - Lower than XGBoost/LightGBM default
            l2_leaf_reg: L2 regularization (default 3)
                - Equivalent to XGBoost's reg_lambda
            bootstrap_type: Sampling method
                - MVS (Minimum Variance Sampling) - default
                - Bayesian - Bayesian bootstrap
                - Bernoulli - subsample without replacement
            border_count: Number of splits for numerical features (default 254)
                - Higher = more accurate but slower
        """
        self._import_catboost()

        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown CatBoost model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        hyperparameters = hyperparameters or {}

        # Determine task type (GPU or CPU)
        task_type = "GPU" if self._gpu_available else "CPU"

        # Classification models
        if "classification" in model_type:
            if model_type == "catboost_binary_classification":
                params = {
                    "iterations": hyperparameters.get("iterations", 1000),
                    "depth": hyperparameters.get("depth", 6),
                    "learning_rate": hyperparameters.get("learning_rate", 0.03),
                    "l2_leaf_reg": hyperparameters.get("l2_leaf_reg", 3),
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "bootstrap_type": hyperparameters.get("bootstrap_type", "MVS"),
                    "subsample": hyperparameters.get("subsample", 0.8),
                    "rsm": hyperparameters.get("rsm", 0.8),
                    "border_count": hyperparameters.get("border_count", 254),
                    "task_type": task_type,
                    "random_seed": hyperparameters.get("random_seed", 42),
                    "verbose": hyperparameters.get("verbose", False)
                }
            else:  # multiclass
                params = {
                    "iterations": hyperparameters.get("iterations", 1000),
                    "depth": hyperparameters.get("depth", 6),
                    "learning_rate": hyperparameters.get("learning_rate", 0.03),
                    "l2_leaf_reg": hyperparameters.get("l2_leaf_reg", 3),
                    "loss_function": "MultiClass",
                    "eval_metric": "MultiClass",
                    "bootstrap_type": hyperparameters.get("bootstrap_type", "MVS"),
                    "subsample": hyperparameters.get("subsample", 0.8),
                    "rsm": hyperparameters.get("rsm", 0.8),
                    "border_count": hyperparameters.get("border_count", 254),
                    "task_type": task_type,
                    "random_seed": hyperparameters.get("random_seed", 42),
                    "verbose": hyperparameters.get("verbose", False)
                }

            return self._CatBoostClassifier(**params)

        # Regression models
        else:
            params = {
                "iterations": hyperparameters.get("iterations", 1000),
                "depth": hyperparameters.get("depth", 6),
                "learning_rate": hyperparameters.get("learning_rate", 0.03),
                "l2_leaf_reg": hyperparameters.get("l2_leaf_reg", 3),
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "bootstrap_type": hyperparameters.get("bootstrap_type", "MVS"),
                "subsample": hyperparameters.get("subsample", 0.8),
                "rsm": hyperparameters.get("rsm", 0.8),
                "border_count": hyperparameters.get("border_count", 254),
                "task_type": task_type,
                "random_seed": hyperparameters.get("random_seed", 42),
                "verbose": hyperparameters.get("verbose", False)
            }

            return self._CatBoostRegressor(**params)

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Any:
        """
        Train CatBoost model.

        CatBoost automatically handles categorical features without encoding.
        Detects categorical columns and passes them to the model.

        Args:
            model: CatBoost model instance
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
                - X_val: Validation features for early stopping
                - y_val: Validation target for early stopping
                - early_stopping_rounds: Number of rounds for early stopping
                - verbose: Print training progress
                - cat_features: List of categorical feature names (auto-detected if not provided)

        Returns:
            Trained model

        Raises:
            TrainingError: If training fails

        Performance Notes:
            CatBoost is optimized for categorical features and often achieves
            best accuracy on tabular data. GPU acceleration is automatically
            enabled when available.
        """
        self._import_catboost()

        try:
            # Extract optional validation set for early stopping
            X_val = kwargs.get("X_val")
            y_val = kwargs.get("y_val")
            early_stopping_rounds = kwargs.get("early_stopping_rounds")
            verbose = kwargs.get("verbose", False)

            # Auto-detect categorical features if not provided
            cat_features = kwargs.get("cat_features")
            if cat_features is None:
                cat_features = self._detect_categorical_features(X_train)

            if X_val is not None and y_val is not None and early_stopping_rounds:
                # Train with early stopping and use_best_model
                model.set_params(use_best_model=True)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    cat_features=cat_features,
                    verbose=verbose
                )
            else:
                # Standard training (explicitly disable use_best_model since no eval_set)
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features,
                    use_best_model=False,
                    verbose=verbose
                )

            return model

        except Exception as e:
            raise TrainingError(
                f"CatBoost model training failed: {e}",
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
        Validate trained CatBoost model on test set.

        Args:
            model: Trained CatBoost model
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
        Calculate metrics for CatBoost model.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model: CatBoost model (optional, for probabilities)
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

            # AUC-ROC for binary classification
            if n_classes == 2 and model is not None and X is not None:
                try:
                    y_proba = model.predict_proba(X)[:, 1]
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
                except Exception:
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
        Get summary information about trained CatBoost model.

        Args:
            model: Trained CatBoost model
            model_type: Type of model
            feature_names: List of feature names

        Returns:
            Dictionary with model summary information
        """
        summary = {
            "model_type": model_type,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "framework": "catboost"
        }

        # Add model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            summary["iterations"] = params.get("iterations")
            summary["depth"] = params.get("depth")
            summary["learning_rate"] = params.get("learning_rate")
            summary["l2_leaf_reg"] = params.get("l2_leaf_reg")
            summary["task_type"] = params.get("task_type")  # GPU or CPU

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

        # Add number of trees actually used (if early stopping was applied)
        if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
            summary["best_iteration"] = int(model.best_iteration_)

        return summary

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model types."""
        return cls.SUPPORTED_MODELS.copy()

    def __repr__(self) -> str:
        """String representation."""
        gpu_status = "GPU" if self._gpu_available else "CPU"
        return f"CatBoostTrainer(models={self.SUPPORTED_MODELS}, task_type={gpu_status})"
