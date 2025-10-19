"""
ML Regression Training Script Template.

Uses base template with regression-specific model creation and metrics.
"""

from typing import List, Dict
from src.generators.templates.base_ml_template import create_ml_script

# Model-specific imports
REGRESSION_IMPORTS = """from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score"""

# Regression metrics function
REGRESSION_METRICS = """def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {{
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred))
    }}"""

# Regression model creation
REGRESSION_MODEL_CREATION = """if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=hyperparameters.get('alpha', 1.0), max_iter=hyperparameters.get('max_iter', 1000), random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=hyperparameters.get('alpha', 1.0), max_iter=hyperparameters.get('max_iter', 1000), random_state=42)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=hyperparameters.get('alpha', 1.0), l1_ratio=hyperparameters.get('l1_ratio', 0.5), max_iter=hyperparameters.get('max_iter', 1000), random_state=42)
    elif model_type == 'polynomial':
        degree = hyperparameters.get('degree', 2)
        model = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)), ('linear', LinearRegression())])
    else:
        raise ValueError(f"Unknown model type: {{model_type}}")"""

# Regression feature importance (handles polynomial specially)
REGRESSION_FEATURE_IMPORTANCE = """feature_importance = None
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = abs(coef).mean(axis=0)
        feature_importance = {{feat: float(imp) for feat, imp in zip(feature_columns, coef)}}
    elif model_type == 'polynomial' and hasattr(model.named_steps.get('linear'), 'coef_'):
        pass  # Polynomial features are expanded, skip for now"""


def generate_regression_training_script(
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
    """Generate regression training script using base template."""
    if hyperparameters is None:
        hyperparameters = {}

    if preprocessing_config is None:
        preprocessing_config = {"scaling": "standard", "missing_strategy": "mean"}

    return create_ml_script(
        model_imports=REGRESSION_IMPORTS,
        metric_functions=REGRESSION_METRICS,
        model_creation=REGRESSION_MODEL_CREATION,
        task_type_value="'regression'",
        feature_importance_logic=REGRESSION_FEATURE_IMPORTANCE,
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
