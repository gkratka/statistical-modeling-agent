"""
Base ML Training Script Template.

This module provides the common template structure shared across all ML model types.
Model-specific sections are injected via parameters.
"""

BASE_ML_TRAINING_TEMPLATE = """
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
{model_imports}
import time
import uuid
import warnings
warnings.filterwarnings('ignore')

{metric_functions}

try:
    # Read configuration
    config = json.loads(sys.stdin.read())
    df = pd.DataFrame(config['dataframe'])

    # Extract parameters
    model_type = {{model_type!r}}
    {task_type_param}target_column = {{target_column!r}}
    feature_columns = {{feature_columns!r}}
    test_size = {{test_size}}
    hyperparameters = {{hyperparameters!r}}
    preprocessing_config = {{preprocessing_config!r}}
    validation_type = {{validation_type!r}}
    cv_folds = {{cv_folds}}
    user_id = {{user_id}}

    # Prepare data
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values
    missing_strategy = preprocessing_config.get('missing_strategy', 'mean')
    if missing_strategy == 'mean':
        X = X.fillna(X.mean())
    elif missing_strategy == 'median':
        X = X.fillna(X.median())
    elif missing_strategy == 'drop':
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    elif missing_strategy == 'zero':
        X = X.fillna(0)

    # Split data
    {train_test_split}

    # Preprocessing - scaling
    scaling_type = preprocessing_config.get('scaling', 'standard')
    scaler = None
    if scaling_type == 'standard':
        scaler = StandardScaler()
    elif scaling_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_type == 'robust':
        scaler = RobustScaler()

    if scaler:
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Create model
    {model_creation}

    # Train model
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Predictions and metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    {get_probabilities}
    train_metrics = {calculate_train_metrics}
    test_metrics = {calculate_test_metrics}

    # Cross-validation
    cv_metrics = None
    if validation_type == 'cross_validation':
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=cv_folds,
            scoring={cv_scoring}
        )
        cv_metrics = {{{{
            {cv_metric_name}: float(cv_scores.mean()),
            {cv_std_name}: float(cv_scores.std()),
            "scores": [float(s) for s in cv_scores]
        }}}}

    # Generate model ID
    model_id = f"model_{{{{uuid.uuid4().hex[:8]}}}}"

    # Create model directory
    model_dir = Path(f"models/user_{{{{user_id}}}}/{{{{model_id}}}}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model and scaler
    joblib.dump(model, model_dir / "model.pkl")
    if scaler:
        joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save feature names
    {save_feature_info}

    # Get feature importance
    {feature_importance_logic}

    # Save metadata
    metadata = {{{{
        "model_id": model_id,
        "user_id": user_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": model_type,
        "task_type": {task_type_value},
        "target_column": target_column,
        "feature_columns": feature_columns,
        "preprocessing": {{{{
            "scaled": scaler is not None,
            "scaler_type": scaling_type if scaler else None,
            "missing_value_strategy": missing_strategy
        }}}},
        "hyperparameters": hyperparameters,
        "metrics": {{{{
            "train": train_metrics,
            "test": test_metrics,
            "cross_validation": cv_metrics
        }}}},
        {additional_metadata}
        "training_data_shape": list(X.shape),
        "training_time_seconds": training_time,
        "feature_importance": feature_importance
    }}}}

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Output results
    results = {{{{
        "success": True,
        "model_id": model_id,
        "metrics": {{{{
            "train": train_metrics,
            "test": test_metrics,
            "cross_validation": cv_metrics
        }}}},
        "training_time": training_time,
        "model_info": {{{{
            "model_type": model_type,
            "features": feature_columns,
            "target": target_column,
            "feature_importance": feature_importance{additional_results}
        }}}}
    }}}}

    print(json.dumps(results, indent=2))

except Exception as e:
    import traceback
    error_results = {{{{
        "success": False,
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    }}}}
    print(json.dumps(error_results, indent=2))
    sys.exit(1)
"""


def create_ml_script(
    model_imports: str,
    metric_functions: str,
    model_creation: str,
    task_type_param: str = "",
    task_type_value: str = "'regression'",
    train_test_split: str = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)",
    get_probabilities: str = "",
    calculate_train_metrics: str = "calculate_metrics(y_train, y_train_pred)",
    calculate_test_metrics: str = "calculate_metrics(y_test, y_test_pred)",
    cv_scoring: str = "'r2'",
    cv_metric_name: str = '"mean_r2"',
    cv_std_name: str = '"std_r2"',
    save_feature_info: str = None,
    feature_importance_logic: str = None,
    additional_metadata: str = "",
    additional_results: str = "",
    **format_params
) -> str:
    """
    Create ML training script from base template with model-specific sections.

    Args:
        model_imports: Model-specific import statements
        metric_functions: Metric calculation function definitions
        model_creation: Model instantiation logic
        task_type_param: Task type parameter extraction (for neural networks)
        task_type_value: Task type value for metadata
        train_test_split: Train-test split logic (stratified for classification)
        get_probabilities: Code to get probability predictions
        calculate_train_metrics: Training metrics calculation call
        calculate_test_metrics: Test metrics calculation call
        cv_scoring: Cross-validation scoring metric
        cv_metric_name: CV metric name in results
        cv_std_name: CV std metric name in results
        save_feature_info: Feature info save logic
        feature_importance_logic: Feature importance extraction logic
        additional_metadata: Additional metadata fields
        additional_results: Additional result fields
        **format_params: Template format parameters

    Returns:
        Complete Python script as string
    """
    if save_feature_info is None:
        save_feature_info = '''with open(model_dir / "feature_names.json", "w") as f:
        json.dump({{"features": feature_columns, "target": target_column}}, f)'''

    if feature_importance_logic is None:
        feature_importance_logic = '''feature_importance = None
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = abs(coef).mean(axis=0)
        feature_importance = {feat: float(imp) for feat, imp in zip(feature_columns, coef)}'''

    script = BASE_ML_TRAINING_TEMPLATE.format(
        model_imports=model_imports,
        metric_functions=metric_functions,
        model_creation=model_creation,
        task_type_param=task_type_param,
        task_type_value=task_type_value,
        train_test_split=train_test_split,
        get_probabilities=get_probabilities,
        calculate_train_metrics=calculate_train_metrics,
        calculate_test_metrics=calculate_test_metrics,
        cv_scoring=cv_scoring,
        cv_metric_name=cv_metric_name,
        cv_std_name=cv_std_name,
        save_feature_info=save_feature_info,
        feature_importance_logic=feature_importance_logic,
        additional_metadata=additional_metadata,
        additional_results=additional_results
    )

    return script.format(**format_params)
