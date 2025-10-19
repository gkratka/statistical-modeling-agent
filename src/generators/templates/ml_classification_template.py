"""
ML Classification Training Script Template.

Uses base template with classification-specific model creation and metrics.
"""

from typing import List, Dict
from src.generators.templates.base_ml_template import create_ml_script

# Model-specific imports
CLASSIFICATION_IMPORTS = """from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix"""

# Classification metrics function
CLASSIFICATION_METRICS = """def calculate_metrics(y_true, y_pred, y_proba=None):
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

# Classification model creation
CLASSIFICATION_MODEL_CREATION = """if model_type == 'logistic':
        model = LogisticRegression(C=hyperparameters.get('C', 1.0), max_iter=hyperparameters.get('max_iter', 1000), solver=hyperparameters.get('solver', 'lbfgs'), random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=hyperparameters.get('max_depth', None), min_samples_split=hyperparameters.get('min_samples_split', 2), min_samples_leaf=hyperparameters.get('min_samples_leaf', 1), random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=hyperparameters.get('n_estimators', 100), max_depth=hyperparameters.get('max_depth', None), min_samples_split=hyperparameters.get('min_samples_split', 2), min_samples_leaf=hyperparameters.get('min_samples_leaf', 1), random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=hyperparameters.get('n_estimators', 100), learning_rate=hyperparameters.get('learning_rate', 0.1), max_depth=hyperparameters.get('max_depth', 3), random_state=42)
    elif model_type == 'svm':
        model = SVC(C=hyperparameters.get('C', 1.0), kernel=hyperparameters.get('kernel', 'rbf'), gamma=hyperparameters.get('gamma', 'scale'), probability=True, random_state=42)
    elif model_type == 'naive_bayes':
        model = GaussianNB(var_smoothing=hyperparameters.get('var_smoothing', 1e-9))
    else:
        raise ValueError(f"Unknown model type: {{model_type}}")"""

# Classification-specific feature info (includes classes)
CLASSIFICATION_FEATURE_INFO = """with open(model_dir / "feature_names.json", "w") as f:
        json.dump({{"features": feature_columns, "target": target_column, "classes": model.classes_.tolist() if hasattr(model, 'classes_') else []}}, f)"""

# Classification feature importance (handles both tree-based and linear)
CLASSIFICATION_FEATURE_IMPORTANCE = """feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = {{feat: float(imp) for feat, imp in zip(feature_columns, model.feature_importances_)}}
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = np.abs(coef).mean(axis=0)
        feature_importance = {{feat: float(imp) for feat, imp in zip(feature_columns, coef)}}"""


def generate_classification_training_script(
    model_type: str,
    target_column: str,
    feature_columns: List[str],
    test_size: float = 0.2,
    hyperparameters: Dict = None,
    preprocessing_config: Dict = None,
    validation_type: str = "hold_out",
    cv_folds: int = 5,
    user_id: int = 0
) -> str:
    """Generate classification training script using base template."""
    if hyperparameters is None:
        hyperparameters = {}

    if preprocessing_config is None:
        preprocessing_config = {"scaling": "standard", "missing_strategy": "mean"}

    return create_ml_script(
        model_imports=CLASSIFICATION_IMPORTS,
        metric_functions=CLASSIFICATION_METRICS,
        model_creation=CLASSIFICATION_MODEL_CREATION,
        task_type_value="'classification'",
        train_test_split="X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)",
        get_probabilities="""y_train_proba = None
    y_test_proba = None
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)""",
        calculate_train_metrics="calculate_metrics(y_train, y_train_pred, y_train_proba)",
        calculate_test_metrics="calculate_metrics(y_test, y_test_pred, y_test_proba)",
        cv_scoring="'accuracy'",
        cv_metric_name='"mean_accuracy"',
        cv_std_name='"std_accuracy"',
        save_feature_info=CLASSIFICATION_FEATURE_INFO,
        feature_importance_logic=CLASSIFICATION_FEATURE_IMPORTANCE,
        additional_metadata='"n_classes": len(model.classes_) if hasattr(model, \'classes_\') else None,',
        additional_results=',\n            "n_classes": len(model.classes_) if hasattr(model, \'classes_\') else None,\n            "classes": model.classes_.tolist() if hasattr(model, \'classes_\') else []',
        model_type=model_type,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=test_size,
        hyperparameters=hyperparameters,
        preprocessing_config=preprocessing_config,
        validation_type=validation_type,
        cv_folds=cv_folds,
        user_id=user_id
    )
