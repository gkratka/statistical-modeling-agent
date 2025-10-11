"""
Task Type Detection Utility.

Automatically detects whether a target variable is suitable for regression
or classification based on its characteristics.
"""

from typing import Literal, Tuple
import pandas as pd
import numpy as np


def detect_task_type(
    target_series: pd.Series,
    unique_threshold: int = 20
) -> Literal["regression", "classification"]:
    """
    Detect whether target variable is for regression or classification.

    Logic:
    - Classification indicators:
      * Categorical dtypes (object, category, bool)
      * Few unique values (< unique_threshold)
      * String values
    - Regression indicators:
      * Numeric dtypes (int, float)
      * Many unique values (>= unique_threshold)

    Args:
        target_series: Target column from dataset
        unique_threshold: Maximum unique values for classification (default: 20)

    Returns:
        "regression" or "classification"

    Examples:
        >>> prices = pd.Series([100000, 150000, 200000, ...])  # Many unique values
        >>> detect_task_type(prices)
        'regression'

        >>> labels = pd.Series(['spam', 'ham', 'spam', ...])  # String categories
        >>> detect_task_type(labels)
        'classification'

        >>> binary = pd.Series([0, 1, 0, 1, 1, 0])  # Few unique numeric values
        >>> detect_task_type(binary)
        'classification'
    """
    # Get basic statistics
    dtype = target_series.dtype
    n_unique = target_series.nunique()

    # Strong classification indicators
    if dtype in ['object', 'category', 'bool']:
        return 'classification'

    # String values = classification
    if dtype == 'object':
        return 'classification'

    # Boolean = classification
    if dtype == 'bool':
        return 'classification'

    # Few unique numeric values = likely classification
    # Examples: 0/1 for binary, 0/1/2 for multiclass, ratings 1-5
    if n_unique < unique_threshold:
        return 'classification'

    # Many unique numeric values = regression
    # Examples: prices, temperatures, measurements
    return 'regression'


def validate_model_compatibility(
    model_type: str,
    detected_task: Literal["regression", "classification"]
) -> Tuple[bool, str]:
    """
    Validate if model type matches detected task type.

    Args:
        model_type: Selected model type (e.g., "keras_binary_classification")
        detected_task: Detected task type ("regression" or "classification")

    Returns:
        Tuple of (is_compatible, warning_message)
        - is_compatible: True if model matches task type
        - warning_message: Empty if compatible, warning text if not

    Examples:
        >>> validate_model_compatibility("linear", "regression")
        (True, "")

        >>> validate_model_compatibility("keras_binary_classification", "regression")
        (False, "Binary classification model selected for regression task...")
    """
    # Define model type mappings
    regression_models = {
        'linear', 'ridge', 'lasso', 'elasticnet', 'polynomial',
        'mlp_regression', 'keras_regression'
    }

    classification_models = {
        'logistic', 'decision_tree', 'random_forest', 'gradient_boosting',
        'svm', 'naive_bayes', 'mlp_classification',
        'keras_binary_classification', 'keras_multiclass_classification'
    }

    # Check compatibility
    if detected_task == 'regression':
        if model_type in regression_models:
            return (True, "")
        else:
            return (
                False,
                f"⚠️ **Mismatch Detected**\n\n"
                f"Your target appears to be **continuous** (many unique values), "
                f"which indicates a **regression** task (predicting numerical values).\n\n"
                f"However, you selected a **classification** model that predicts categories.\n\n"
                f"**Recommendation**: Choose a regression model instead:\n"
                f"• Linear Regression\n"
                f"• MLP Regression\n"
                f"• Keras Regression"
            )
    else:  # classification
        if model_type in classification_models:
            return (True, "")
        else:
            return (
                False,
                f"⚠️ **Mismatch Detected**\n\n"
                f"Your target appears to be **categorical** (few unique values), "
                f"which indicates a **classification** task (predicting categories).\n\n"
                f"However, you selected a **regression** model that predicts continuous values.\n\n"
                f"**Recommendation**: Choose a classification model instead:\n"
                f"• Logistic Regression\n"
                f"• Decision Tree\n"
                f"• Keras Binary/Multiclass Classification"
            )


def get_recommended_models(
    detected_task: Literal["regression", "classification"],
    category: Literal["regression", "classification", "neural"]
) -> list[Tuple[str, str]]:
    """
    Get recommended model options based on detected task type and category.

    Filters the model options to show only compatible models for the detected task.

    Args:
        detected_task: Detected task type from target analysis
        category: User-selected model category

    Returns:
        List of (display_name, model_type) tuples filtered for compatibility

    Examples:
        >>> get_recommended_models("regression", "neural")
        [("MLP Regression", "mlp_regression"), ("Keras Regression", "keras_regression")]

        >>> get_recommended_models("classification", "neural")
        [("MLP Classification", "mlp_classification"), ...]
    """
    # All available models by category
    all_models = {
        "regression": [
            ("Linear Regression", "linear"),
            ("Ridge Regression (L2)", "ridge"),
            ("Lasso Regression (L1)", "lasso"),
            ("ElasticNet (L1+L2)", "elasticnet"),
            ("Polynomial Regression", "polynomial")
        ],
        "classification": [
            ("Logistic Regression", "logistic"),
            ("Decision Tree", "decision_tree"),
            ("Random Forest", "random_forest"),
            ("Gradient Boosting", "gradient_boosting"),
            ("Support Vector Machine", "svm"),
            ("Naive Bayes", "naive_bayes")
        ],
        "neural": [
            ("MLP Regression", "mlp_regression"),
            ("MLP Classification", "mlp_classification"),
            ("Keras Binary Classification", "keras_binary_classification"),
            ("Keras Multiclass Classification", "keras_multiclass_classification"),
            ("Keras Regression", "keras_regression")
        ]
    }

    models = all_models.get(category, [])

    # Filter based on detected task type
    if category == "neural":
        if detected_task == "regression":
            # Only show regression models
            return [m for m in models if "regression" in m[1].lower()]
        else:  # classification
            # Only show classification models
            return [m for m in models if "classification" in m[1].lower()]

    # For non-neural categories, return all (already filtered by category)
    return models
