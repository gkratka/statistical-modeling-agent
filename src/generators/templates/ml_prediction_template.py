"""
ML Prediction Script Template.

This template generates standalone Python scripts for making predictions
with trained machine learning models.
"""

from typing import List

# This is the template string that will be populated with parameters
PREDICTION_TEMPLATE = """
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    # Read configuration from stdin
    config = json.loads(sys.stdin.read())
    df = pd.DataFrame(config['dataframe'])

    # Extract parameters
    user_id = {user_id}
    model_id = {model_id!r}
    feature_columns = {feature_columns!r}

    # Model directory
    model_dir = Path(f"models/user_{{user_id}}/{{model_id}}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {{model_dir}}")

    # Load model
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {{model_path}}")

    model = joblib.load(model_path)

    # Load metadata
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load scaler if exists
    scaler = None
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    # Extract features from input data
    if not all(col in df.columns for col in feature_columns):
        missing = [col for col in feature_columns if col not in df.columns]
        raise ValueError(f"Missing required feature columns: {{missing}}")

    X = df[feature_columns].copy()

    # Handle missing values (same strategy as training)
    missing_strategy = metadata.get('preprocessing', {{}}).get('missing_value_strategy', 'mean')
    if missing_strategy == 'mean':
        X = X.fillna(X.mean())
    elif missing_strategy == 'median':
        X = X.fillna(X.median())
    elif missing_strategy == 'drop':
        # For prediction, we can't drop rows, so use mean instead
        X = X.fillna(X.mean())
    elif missing_strategy == 'zero':
        X = X.fillna(0)

    # Scale features if scaler was used
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = X

    # Make predictions
    predictions = model.predict(X_scaled)

    # Prepare results
    results = {{
        "success": True,
        "model_id": model_id,
        "predictions": predictions.tolist(),
        "n_predictions": len(predictions)
    }}

    # Get probabilities for classification models
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        results["probabilities"] = probabilities.tolist()

        # Add class labels if available
        if hasattr(model, 'classes_'):
            results["classes"] = model.classes_.tolist()

    # Add prediction statistics
    results["prediction_stats"] = {{
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
        "mean": float(np.mean(predictions)),
        "median": float(np.median(predictions))
    }}

    # Add model info
    results["model_info"] = {{
        "model_type": metadata.get("model_type"),
        "task_type": metadata.get("task_type"),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns)
    }}

    print(json.dumps(results, indent=2))

except Exception as e:
    # Output error information
    import traceback
    error_results = {{
        "success": False,
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(error_results, indent=2))
    sys.exit(1)
"""


def generate_prediction_script(
    user_id: int,
    model_id: str,
    feature_columns: List[str]
) -> str:
    """
    Generate a prediction script from the template.

    Args:
        user_id: User identifier
        model_id: Model identifier
        feature_columns: List of feature column names

    Returns:
        Complete Python script as string
    """
    # Format the template with parameters
    script = PREDICTION_TEMPLATE.format(
        user_id=user_id,
        model_id=model_id,
        feature_columns=feature_columns
    )

    return script
