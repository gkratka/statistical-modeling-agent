"""
ML Neural Network Training Script Template.

Uses base template with neural network-specific model creation supporting both regression and classification.
"""

from typing import List, Dict
from src.generators.templates.base_ml_template import create_ml_script

# Model-specific imports
NN_IMPORTS = """from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)"""

# Neural network metrics functions (both regression and classification)
NN_METRICS = """def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {{
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred))
    }}

def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    n_classes = len(np.unique(y_true))
    average_method = 'binary' if n_classes == 2 else 'weighted'

    metrics = {{
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average_method, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average_method, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average_method, zero_division=0))
    }}

    if y_proba is not None:
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            metrics["roc_auc"] = float(roc_auc)
        except Exception:
            pass

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics"""

# Neural network model creation
NN_MODEL_CREATION = """hidden_layers = hyperparameters.get('hidden_layers', [100])
    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]
    hidden_layer_sizes = tuple(hidden_layers)

    if model_type == 'mlp_regression':
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=hyperparameters.get('activation', 'relu'),
            solver=hyperparameters.get('solver', 'adam'),
            alpha=hyperparameters.get('alpha', 0.0001),
            learning_rate=hyperparameters.get('learning_rate', 'constant'),
            learning_rate_init=hyperparameters.get('learning_rate_init', 0.001),
            max_iter=hyperparameters.get('max_iter', 200),
            early_stopping=hyperparameters.get('early_stopping', False),
            validation_fraction=hyperparameters.get('validation_fraction', 0.1),
            random_state=42
        )
    elif model_type == 'mlp_classification':
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=hyperparameters.get('activation', 'relu'),
            solver=hyperparameters.get('solver', 'adam'),
            alpha=hyperparameters.get('alpha', 0.0001),
            learning_rate=hyperparameters.get('learning_rate', 'constant'),
            learning_rate_init=hyperparameters.get('learning_rate_init', 0.001),
            max_iter=hyperparameters.get('max_iter', 200),
            early_stopping=hyperparameters.get('early_stopping', False),
            validation_fraction=hyperparameters.get('validation_fraction', 0.1),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {{model_type}}")"""

# Neural network feature info (includes architecture)
NN_FEATURE_INFO = """architecture_info = {{
        "features": feature_columns,
        "target": target_column,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "n_layers": len(hidden_layer_sizes) + 1,
        "n_iterations": int(model.n_iter_),
        "activation": hyperparameters.get('activation', 'relu'),
        "solver": hyperparameters.get('solver', 'adam')
    }}
    if task_type == 'classification' and hasattr(model, 'classes_'):
        architecture_info["classes"] = model.classes_.tolist()

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(architecture_info, f)"""

# Neural network feature importance (no feature importance for MLPs)
NN_FEATURE_IMPORTANCE = """feature_importance = None"""


def generate_neural_network_training_script(
    model_type: str,
    task_type: str,
    target_column: str,
    feature_columns: List[str],
    test_size: float = 0.2,
    hyperparameters: Dict = None,
    preprocessing_config: Dict = None,
    validation_type: str = "hold_out",
    cv_folds: int = 5,
    user_id: int = 0
) -> str:
    """Generate neural network training script using base template."""
    if hyperparameters is None:
        hyperparameters = {
            "hidden_layers": [100],
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 200
        }

    if preprocessing_config is None:
        preprocessing_config = {"scaling": "standard", "missing_strategy": "mean"}

    # Task-specific configurations
    is_classification = task_type == 'classification'

    train_test_split_code = (
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)"
        if is_classification else
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)"
    )

    get_proba_code = """y_train_proba = None
    y_test_proba = None
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)""" if is_classification else ""

    calc_train_metrics = (
        "calculate_classification_metrics(y_train, y_train_pred, y_train_proba)" if is_classification else
        "calculate_regression_metrics(y_train, y_train_pred)"
    )

    calc_test_metrics = (
        "calculate_classification_metrics(y_test, y_test_pred, y_test_proba)" if is_classification else
        "calculate_regression_metrics(y_test, y_test_pred)"
    )

    cv_scoring_val = "'accuracy'" if is_classification else "'r2'"
    metric_name = "accuracy" if is_classification else "r2"

    # Calculate number of parameters
    additional_metadata_str = f""""architecture": {{{{
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "n_layers": len(hidden_layer_sizes) + 1,
            "n_parameters": int(sum(coef.size for coef in model.coefs_) + sum(intercept.size for intercept in model.intercepts_)),
            "activation": hyperparameters.get('activation', 'relu'),
            "solver": hyperparameters.get('solver', 'adam')
        }}}},
        "training_info": {{{{
            "data_shape": list(X.shape),
            "training_time_seconds": training_time,
            "n_iterations": int(model.n_iter_),
            "final_loss": float(model.loss_) if hasattr(model, 'loss_') else None
        }}}},
        "n_classes": len(model.classes_) if hasattr(model, 'classes_') else None,"""

    additional_results_str = """,
            "architecture": {{
                "hidden_layer_sizes": list(hidden_layer_sizes),
                "n_parameters": int(sum(coef.size for coef in model.coefs_) + sum(intercept.size for intercept in model.intercepts_)),
                "n_iterations": int(model.n_iter_)
            }},
            "n_classes": len(model.classes_) if hasattr(model, 'classes_') else None,
            "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None"""

    return create_ml_script(
        model_imports=NN_IMPORTS,
        metric_functions=NN_METRICS,
        model_creation=NN_MODEL_CREATION,
        task_type_param="task_type = {task_type!r}\n    ",
        task_type_value="task_type",
        train_test_split=train_test_split_code,
        get_probabilities=get_proba_code,
        calculate_train_metrics=calc_train_metrics,
        calculate_test_metrics=calc_test_metrics,
        cv_scoring=cv_scoring_val,
        cv_metric_name=f'"mean_{metric_name}"',
        cv_std_name=f'"std_{metric_name}"',
        save_feature_info=NN_FEATURE_INFO,
        feature_importance_logic=NN_FEATURE_IMPORTANCE,
        additional_metadata=additional_metadata_str,
        additional_results=additional_results_str,
        model_type=model_type,
        task_type=task_type,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=test_size,
        hyperparameters=hyperparameters,
        preprocessing_config=preprocessing_config,
        validation_type=validation_type,
        cv_folds=cv_folds,
        user_id=user_id
    )
