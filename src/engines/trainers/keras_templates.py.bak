"""
Keras Architecture Templates.

Provides pre-configured neural network architectures for common tasks.
"""

from typing import Dict, Any, Literal


def get_binary_classification_template(
    n_features: int,
    kernel_initializer: str = "random_normal"
) -> Dict[str, Any]:
    """
    Get template for binary classification.

    Architecture:
    - Input: n_features
    - Hidden: Dense(n_features, relu)
    - Output: Dense(1, sigmoid)
    - Loss: binary_crossentropy

    Args:
        n_features: Number of input features
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict ready for KerasNeuralNetworkTrainer
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "units": n_features,
                "activation": "relu",
                "kernel_initializer": kernel_initializer
            },
            {
                "type": "Dense",
                "units": 1,
                "activation": "sigmoid",
                "kernel_initializer": kernel_initializer
            }
        ],
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"]
        }
    }


def get_multiclass_classification_template(
    n_features: int,
    n_classes: int,
    kernel_initializer: str = "glorot_uniform"
) -> Dict[str, Any]:
    """
    Get template for multiclass classification.

    Architecture:
    - Input: n_features
    - Hidden: Dense(n_features, relu)
    - Output: Dense(n_classes, softmax)
    - Loss: categorical_crossentropy

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict ready for KerasNeuralNetworkTrainer
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "units": n_features,
                "activation": "relu",
                "kernel_initializer": kernel_initializer
            },
            {
                "type": "Dense",
                "units": n_classes,
                "activation": "softmax",
                "kernel_initializer": kernel_initializer
            }
        ],
        "compile": {
            "loss": "categorical_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"]
        }
    }


def get_regression_template(
    n_features: int,
    kernel_initializer: str = "glorot_uniform"
) -> Dict[str, Any]:
    """
    Get template for regression.

    Architecture:
    - Input: n_features
    - Hidden: Dense(n_features, relu)
    - Output: Dense(1, linear)
    - Loss: mse

    Args:
        n_features: Number of input features
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict ready for KerasNeuralNetworkTrainer
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "units": n_features,
                "activation": "relu",
                "kernel_initializer": kernel_initializer
            },
            {
                "type": "Dense",
                "units": 1,
                "activation": "linear",
                "kernel_initializer": kernel_initializer
            }
        ],
        "compile": {
            "loss": "mse",
            "optimizer": "adam",
            "metrics": ["mae"]
        }
    }


def get_template(
    model_type: Literal["keras_binary_classification", "keras_multiclass_classification", "keras_regression"],
    n_features: int,
    n_classes: int = 2,
    kernel_initializer: str = "random_normal"
) -> Dict[str, Any]:
    """
    Get architecture template based on model type.

    Args:
        model_type: Type of Keras model
        n_features: Number of input features
        n_classes: Number of classes (for classification)
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict

    Raises:
        ValueError: If model_type is invalid
    """
    if model_type == "keras_binary_classification":
        return get_binary_classification_template(n_features, kernel_initializer)
    elif model_type == "keras_multiclass_classification":
        return get_multiclass_classification_template(n_features, n_classes, kernel_initializer)
    elif model_type == "keras_regression":
        return get_regression_template(n_features, kernel_initializer)
    else:
        raise ValueError(f"Unknown Keras model type: {model_type}")
