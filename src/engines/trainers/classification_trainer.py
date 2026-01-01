"""
Classification Model Trainer.

This module provides the trainer for classification models including
logistic regression, decision trees, random forests, gradient boosting, and SVM.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report
)

from src.engines.ml_base import ModelTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError


class ClassificationTrainer(ModelTrainer):
    """
    Trainer for classification models.

    Supports: logistic, decision_tree, random_forest, gradient_boosting, svm, naive_bayes
    """

    # Supported model types
    SUPPORTED_MODELS = [
        "logistic",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svm",
        "naive_bayes"
    ]

    def __init__(self, config: MLEngineConfig):
        """
        Initialize classification trainer.

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
        Create classification model instance.

        Args:
            model_type: Type of classification model
            hyperparameters: Model hyperparameters

        Returns:
            Instantiated model object

        Raises:
            TrainingError: If model_type is not supported
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise TrainingError(
                f"Unknown classification model type: '{model_type}'. "
                f"Supported types: {self.SUPPORTED_MODELS}",
                model_type=model_type
            )

        # Merge with defaults
        params = self.merge_hyperparameters(model_type, hyperparameters)

        try:
            if model_type == "logistic":
                return LogisticRegression(
                    C=params.get('C', 1.0),
                    max_iter=params.get('max_iter', 1000),
                    random_state=42,
                    solver=params.get('solver', 'lbfgs')
                )

            elif model_type == "decision_tree":
                return DecisionTreeClassifier(
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42
                )

            elif model_type == "random_forest":
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42
                )

            elif model_type == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    max_depth=params.get('max_depth', 3),
                    random_state=42
                )

            elif model_type == "svm":
                return SVC(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    gamma=params.get('gamma', 'scale'),
                    probability=True,  # Enable probability estimates
                    random_state=42
                )

            elif model_type == "naive_bayes":
                return GaussianNB(
                    var_smoothing=params.get('var_smoothing', 1e-9)
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
        Calculate classification metrics.

        Args:
            y_true: True target values
            y_pred: Predicted class labels
            y_proba: Predicted probabilities (optional, for probability-based metrics)

        Returns:
            Dictionary of classification metrics including:
            - accuracy, precision, recall, f1 (basic metrics)
            - roc_auc: Area Under ROC Curve (primary classification metric)
            - auc_pr: Area Under Precision-Recall Curve
            - brier_score: Calibration metric (binary only, lower is better)
            - log_loss: Probabilistic accuracy (lower is better)
            - confusion_matrix: Raw confusion matrix
        """
        try:
            # Determine if binary or multiclass
            n_classes = len(np.unique(y_true))
            is_binary = n_classes == 2
            average_method = 'binary' if is_binary else 'weighted'

            # Calculate basic metrics
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

            # Add probability-based metrics if probabilities available
            if y_proba is not None:
                try:
                    if is_binary:
                        # Binary classification: use positive class probability
                        pos_proba = y_proba[:, 1]

                        # ROC-AUC (primary metric)
                        roc_auc = roc_auc_score(y_true, pos_proba)
                        metrics["roc_auc"] = float(roc_auc)

                        # AUC-PR (Precision-Recall AUC)
                        auc_pr = average_precision_score(y_true, pos_proba)
                        metrics["auc_pr"] = float(auc_pr)

                        # Brier Score (calibration - lower is better)
                        brier = brier_score_loss(y_true, pos_proba)
                        metrics["brier_score"] = float(brier)

                    else:
                        # Multiclass: use one-vs-rest for ROC-AUC
                        roc_auc = roc_auc_score(
                            y_true,
                            y_proba,
                            multi_class='ovr',
                            average='weighted'
                        )
                        metrics["roc_auc"] = float(roc_auc)

                    # Log Loss (works for both binary and multiclass)
                    ll = log_loss(y_true, y_proba)
                    metrics["log_loss"] = float(ll)

                except Exception:
                    # Probability-based metrics failed, skip them
                    pass

            # Add confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = conf_matrix.tolist()

            return metrics

        except Exception as e:
            # Return default metrics on error
            return {
                "accuracy": float('nan'),
                "precision": float('nan'),
                "recall": float('nan'),
                "f1": float('nan'),
                "error": str(e)
            }

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train classification model.

        Args:
            model: Classification model instance
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
        feature_names: List[str]
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
        if hasattr(model, 'n_estimators'):
            # Tree ensemble models
            summary["n_estimators"] = int(model.n_estimators)

        if hasattr(model, 'max_depth') and model.max_depth is not None:
            # Tree-based models
            summary["max_depth"] = int(model.max_depth)

        if hasattr(model, 'C'):
            # Logistic regression, SVM
            summary["C"] = float(model.C)

        if hasattr(model, 'kernel'):
            # SVM
            summary["kernel"] = str(model.kernel)

        if hasattr(model, 'learning_rate'):
            # Gradient boosting
            summary["learning_rate"] = float(model.learning_rate)

        # Add number of classes
        if hasattr(model, 'classes_'):
            summary["n_classes"] = len(model.classes_)
            summary["classes"] = model.classes_.tolist()

        # Add feature importance if available
        feature_importance = self.get_feature_importance(model, feature_names)
        if feature_importance:
            summary["feature_importance"] = feature_importance

        return summary

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        Get list of supported model types.

        Returns:
            List of supported model type names
        """
        return cls.SUPPORTED_MODELS.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"ClassificationTrainer(models={self.SUPPORTED_MODELS})"
