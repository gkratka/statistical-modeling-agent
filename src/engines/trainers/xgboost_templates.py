"""
XGBoost Hyperparameter Templates.

Provides pre-configured hyperparameter sets for common XGBoost tasks.
"""

from typing import Dict, Any, Literal


def get_binary_classification_template(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template for binary classification.

    Default hyperparameters optimized for binary classification tasks.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)

    Returns:
        Hyperparameter dict ready for XGBoostTrainer
    """
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "random_state": 42,
        "n_jobs": -1
    }


def get_multiclass_classification_template(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template for multiclass classification.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)

    Returns:
        Hyperparameter dict ready for XGBoostTrainer
    """
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "random_state": 42,
        "n_jobs": -1
    }


def get_regression_template(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template for regression.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)

    Returns:
        Hyperparameter dict ready for XGBoostTrainer
    """
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "random_state": 42,
        "n_jobs": -1
    }


def get_template(
    model_type: Literal[
        "xgboost_binary_classification",
        "xgboost_multiclass_classification",
        "xgboost_regression"
    ],
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get hyperparameter template based on model type.

    Args:
        model_type: Type of XGBoost model
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)

    Returns:
        Hyperparameter dict

    Raises:
        ValueError: If model_type is invalid
    """
    if model_type == "xgboost_binary_classification":
        return get_binary_classification_template(n_estimators, max_depth, learning_rate)
    elif model_type == "xgboost_multiclass_classification":
        return get_multiclass_classification_template(n_estimators, max_depth, learning_rate)
    elif model_type == "xgboost_regression":
        return get_regression_template(n_estimators, max_depth, learning_rate)
    else:
        raise ValueError(f"Unknown XGBoost model type: {model_type}")
