"""
CatBoost Hyperparameter Templates.

Provides pre-configured hyperparameter sets for common tasks using CatBoost.

CatBoost Advantages:
- Best accuracy on tabular data (often outperforms XGBoost/LightGBM)
- Native categorical feature support (no encoding needed)
- GPU acceleration with automatic detection
- Robust to overfitting (ordered boosting)
- Fast inference speed

CatBoost Key Parameters:
- iterations (not n_estimators) - number of boosting rounds
- depth (not max_depth) - tree depth
- learning_rate - step size
- l2_leaf_reg - L2 regularization
- border_count - number of splits for numerical features
- bootstrap_type - MVS, Bayesian, Bernoulli
"""

from typing import Dict, Any, Literal


def get_binary_classification_template(
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03
) -> Dict[str, Any]:
    """
    Get template for binary classification.

    Default hyperparameters optimized for binary classification tasks.

    Args:
        iterations: Number of boosting rounds (default 1000)
            - More rounds = better fit but longer training
            - CatBoost uses early stopping by default
            - Typical range: 100-5000
        depth: Maximum tree depth (default 6)
            - Higher = more complex model
            - Typical range: 4-10
            - CatBoost is less prone to overfitting than XGBoost
        learning_rate: Learning rate / step size (default 0.03)
            - Lower = more conservative, better generalization
            - Higher = faster convergence, risk of overfitting
            - Typical range: 0.01-0.1

    Returns:
        Hyperparameter dict ready for CatBoostTrainer

    Note:
        CatBoost uses different parameter names than XGBoost/LightGBM:
        - iterations (not n_estimators)
        - depth (not max_depth)
        - l2_leaf_reg (not reg_lambda)
        - bootstrap_type (not boosting_type)
    """
    return {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": 3,              # L2 regularization
        "bootstrap_type": "MVS",       # Minimum Variance Sampling
        "subsample": 0.8,              # Data sampling rate
        "rsm": 0.8,                    # Random Subspace Method (feature sampling)
        "border_count": 254,           # Number of splits for numerical features
        "random_seed": 42,
        "verbose": False
    }


def get_multiclass_classification_template(
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03
) -> Dict[str, Any]:
    """
    Get template for multiclass classification.

    Args:
        iterations: Number of boosting rounds (default 1000)
        depth: Maximum tree depth (default 6)
        learning_rate: Learning rate (default 0.03)

    Returns:
        Hyperparameter dict ready for CatBoostTrainer
    """
    return {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": 3,
        "bootstrap_type": "MVS",
        "subsample": 0.8,
        "rsm": 0.8,
        "border_count": 254,
        "use_best_model": True,
        "random_seed": 42,
        "verbose": False
    }


def get_regression_template(
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03
) -> Dict[str, Any]:
    """
    Get template for regression.

    Args:
        iterations: Number of boosting rounds (default 1000)
        depth: Maximum tree depth (default 6)
        learning_rate: Learning rate (default 0.03)

    Returns:
        Hyperparameter dict ready for CatBoostTrainer
    """
    return {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": 3,
        "bootstrap_type": "MVS",
        "subsample": 0.8,
        "rsm": 0.8,
        "border_count": 254,
        "use_best_model": True,
        "random_seed": 42,
        "verbose": False
    }


def get_template(
    model_type: Literal[
        "catboost_binary_classification",
        "catboost_multiclass_classification",
        "catboost_regression"
    ],
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03
) -> Dict[str, Any]:
    """
    Get hyperparameter template based on model type.

    Args:
        model_type: Type of CatBoost model
        iterations: Number of boosting rounds (default 1000)
        depth: Maximum tree depth (default 6)
        learning_rate: Learning rate (default 0.03)

    Returns:
        Hyperparameter dict

    Raises:
        ValueError: If model_type is invalid

    Examples:
        >>> params = get_template("catboost_binary_classification")
        >>> params['iterations']
        1000

        >>> params = get_template(
        ...     "catboost_regression",
        ...     iterations=2000,
        ...     depth=8,
        ...     learning_rate=0.01
        ... )
        >>> params['iterations']
        2000
    """
    if model_type == "catboost_binary_classification":
        return get_binary_classification_template(iterations, depth, learning_rate)
    elif model_type == "catboost_multiclass_classification":
        return get_multiclass_classification_template(iterations, depth, learning_rate)
    elif model_type == "catboost_regression":
        return get_regression_template(iterations, depth, learning_rate)
    else:
        raise ValueError(f"Unknown CatBoost model type: {model_type}")
