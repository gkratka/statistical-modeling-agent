"""
LightGBM Hyperparameter Templates.

Provides pre-configured hyperparameter sets for common tasks using LightGBM.

LightGBM uses different parameter naming than XGBoost:
- num_leaves (not max_depth) - leaf-wise vs depth-wise growth
- feature_fraction (not colsample_bytree)
- bagging_fraction + bagging_freq (not subsample)
- min_data_in_leaf (not min_child_weight)

LightGBM Advantages:
- 10-20x faster training on large datasets (>100K rows)
- Lower memory usage (~50% less than XGBoost)
- Better accuracy with fewer iterations
- Native categorical feature support
"""

from typing import Dict, Any, Literal


def get_binary_classification_template(
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template for binary classification.

    Default hyperparameters optimized for binary classification tasks.

    Args:
        n_estimators: Number of boosting rounds (default 100)
            - More rounds = better fit but longer training
            - Typical range: 50-500
        num_leaves: Maximum tree leaves (default 31)
            - LightGBM uses leaf-wise growth (not depth-wise)
            - Higher = more complex model
            - Typical range: 20-150
            - Relationship: num_leaves < 2^max_depth
        learning_rate: Learning rate / step size (default 0.1)
            - Lower = more conservative, better generalization
            - Higher = faster convergence, risk of overfitting
            - Typical range: 0.01-0.3

    Returns:
        Hyperparameter dict ready for LightGBMTrainer

    Note:
        LightGBM parameters differ from XGBoost:
        - num_leaves (not max_depth) - controls tree complexity
        - feature_fraction (not colsample_bytree) - feature sampling
        - bagging_fraction (not subsample) - data sampling
        - bagging_freq - when to perform bagging
    """
    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,  # Use 80% of features per tree
        "bagging_fraction": 0.8,  # Use 80% of data per tree
        "bagging_freq": 1,         # Perform bagging every iteration
        "min_data_in_leaf": 20,    # Minimum samples required in leaf
        "min_gain_to_split": 0.0,  # Minimum gain required for split
        "lambda_l1": 0.0,          # L1 regularization
        "lambda_l2": 0.0,          # L2 regularization
        "random_state": 42,
        "n_jobs": -1               # Use all CPU cores
    }


def get_multiclass_classification_template(
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template for multiclass classification.

    Args:
        n_estimators: Number of boosting rounds (default 100)
        num_leaves: Maximum tree leaves (default 31)
        learning_rate: Learning rate (default 0.1)

    Returns:
        Hyperparameter dict ready for LightGBMTrainer
    """
    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "min_gain_to_split": 0.0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "random_state": 42,
        "n_jobs": -1
    }


def get_regression_template(
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template for regression.

    Args:
        n_estimators: Number of boosting rounds (default 100)
        num_leaves: Maximum tree leaves (default 31)
        learning_rate: Learning rate (default 0.1)

    Returns:
        Hyperparameter dict ready for LightGBMTrainer
    """
    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "min_gain_to_split": 0.0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "random_state": 42,
        "n_jobs": -1
    }


def get_large_dataset_template(
    n_estimators: int = 100,
    num_leaves: int = 63,
    learning_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Get template optimized for large datasets (>100K rows).

    Uses more conservative parameters to prevent overfitting
    and take advantage of LightGBM's speed on large data.

    Args:
        n_estimators: Number of boosting rounds (default 100)
        num_leaves: Maximum tree leaves (default 63, higher for large data)
            - Larger datasets can support more complex trees
        learning_rate: Learning rate (default 0.05, lower for stability)
            - Lower learning rate with more rounds = better generalization

    Returns:
        Hyperparameter dict optimized for large datasets

    Use Cases:
        - Datasets >100K rows
        - Memory-constrained environments
        - Need fastest possible training
        - High-dimensional data

    Performance Expectations:
        - Training speed: 10-20x faster than XGBoost
        - Memory usage: ~50% less than XGBoost
        - Accuracy: Often better with fewer iterations
    """
    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,        # Higher for large datasets
        "learning_rate": learning_rate,  # Lower for stability
        "feature_fraction": 0.7,         # More aggressive feature sampling
        "bagging_fraction": 0.7,         # More aggressive data sampling
        "bagging_freq": 5,               # More frequent bagging
        "min_data_in_leaf": 100,         # Higher minimum for large datasets
        "min_gain_to_split": 0.01,       # Require meaningful splits
        "lambda_l1": 0.1,                # L1 regularization to prevent overfitting
        "lambda_l2": 0.1,                # L2 regularization to prevent overfitting
        "random_state": 42,
        "n_jobs": -1
    }


def get_fast_template(
    n_estimators: int = 50,
    num_leaves: int = 15,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Get template optimized for speed over accuracy.

    Use when:
    - Need quick initial results
    - Prototyping / experimentation
    - Real-time or near real-time training required

    Trade-offs:
    - Faster training (2-3x faster than default)
    - Lower accuracy (~2-5% drop)
    - Less overfitting risk

    Args:
        n_estimators: Number of boosting rounds (default 50, half of standard)
        num_leaves: Maximum tree leaves (default 15, half of standard)
        learning_rate: Learning rate (default 0.1, same as standard)

    Returns:
        Hyperparameter dict optimized for speed
    """
    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.7,   # Less features = faster
        "bagging_fraction": 0.7,   # Less data = faster
        "bagging_freq": 2,         # Less frequent bagging = faster
        "min_data_in_leaf": 50,    # Higher minimum = faster
        "min_gain_to_split": 0.05, # Higher threshold = faster
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "random_state": 42,
        "n_jobs": -1
    }


def get_high_accuracy_template(
    n_estimators: int = 300,
    num_leaves: int = 50,
    learning_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Get template optimized for maximum accuracy.

    Use when:
    - Accuracy is critical
    - Training time is not a constraint
    - Have enough data to support complex models

    Trade-offs:
    - Higher accuracy (~2-5% improvement)
    - Longer training time (3-5x slower than default)
    - Higher overfitting risk (needs more data)

    Args:
        n_estimators: Number of boosting rounds (default 300, 3x standard)
        num_leaves: Maximum tree leaves (default 50, ~1.5x standard)
        learning_rate: Learning rate (default 0.05, half of standard)
            - Lower learning rate compensates for more rounds

    Returns:
        Hyperparameter dict optimized for accuracy
    """
    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.9,   # Use more features
        "bagging_fraction": 0.9,   # Use more data
        "bagging_freq": 1,         # Bag every iteration
        "min_data_in_leaf": 10,    # Lower minimum = more granular splits
        "min_gain_to_split": 0.0,  # Allow all meaningful splits
        "lambda_l1": 0.01,         # Light regularization
        "lambda_l2": 0.01,         # Light regularization
        "random_state": 42,
        "n_jobs": -1
    }


def get_template(
    model_type: Literal["lightgbm_binary_classification",
                       "lightgbm_multiclass_classification",
                       "lightgbm_regression"],
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    dataset_size: Literal["small", "medium", "large"] = "medium",
    optimization: Literal["balanced", "speed", "accuracy"] = "balanced"
) -> Dict[str, Any]:
    """
    Get hyperparameter template based on model type, dataset size, and optimization goal.

    Args:
        model_type: Type of LightGBM model
        n_estimators: Number of boosting rounds (default 100)
        num_leaves: Maximum tree leaves (default 31)
        learning_rate: Learning rate (default 0.1)
        dataset_size: Dataset size category
            - "small": <10K rows (conservative parameters)
            - "medium": 10K-100K rows (default parameters)
            - "large": >100K rows (optimized large dataset parameters)
        optimization: Optimization goal
            - "balanced": Balance between speed and accuracy (default)
            - "speed": Optimize for faster training
            - "accuracy": Optimize for maximum accuracy

    Returns:
        Hyperparameter dict

    Raises:
        ValueError: If model_type is invalid

    Examples:
        >>> # Small dataset, need accuracy
        >>> params = get_template(
        ...     "lightgbm_binary_classification",
        ...     dataset_size="small",
        ...     optimization="accuracy"
        ... )

        >>> # Large dataset, need speed
        >>> params = get_template(
        ...     "lightgbm_regression",
        ...     dataset_size="large",
        ...     optimization="speed"
        ... )

        >>> # Medium dataset, balanced approach
        >>> params = get_template(
        ...     "lightgbm_multiclass_classification"
        ... )
    """
    # Prioritize optimization goal over dataset size
    if optimization == "speed":
        return get_fast_template()  # Use fast template's own defaults
    elif optimization == "accuracy":
        return get_high_accuracy_template()  # Use accuracy template's own defaults

    # For balanced optimization, use dataset size
    if dataset_size == "large":
        return get_large_dataset_template()  # Use large dataset template's own defaults

    # Otherwise use task-specific templates
    if model_type == "lightgbm_binary_classification":
        return get_binary_classification_template(n_estimators, num_leaves, learning_rate)
    elif model_type == "lightgbm_multiclass_classification":
        return get_multiclass_classification_template(n_estimators, num_leaves, learning_rate)
    elif model_type == "lightgbm_regression":
        return get_regression_template(n_estimators, num_leaves, learning_rate)
    else:
        raise ValueError(f"Unknown LightGBM model type: {model_type}")


# Comparison guide for choosing between templates
TEMPLATE_COMPARISON = {
    "default": {
        "speed": "1x (baseline)",
        "accuracy": "baseline",
        "use_case": "General purpose, balanced performance"
    },
    "large_dataset": {
        "speed": "0.8x (20% faster on large data)",
        "accuracy": "baseline or better",
        "use_case": "Datasets >100K rows, memory constraints"
    },
    "fast": {
        "speed": "3x (3x faster)",
        "accuracy": "baseline -2-5%",
        "use_case": "Prototyping, real-time training, quick results"
    },
    "high_accuracy": {
        "speed": "0.3x (3x slower)",
        "accuracy": "baseline +2-5%",
        "use_case": "Critical applications, competitions, final models"
    }
}
