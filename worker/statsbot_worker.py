#!/usr/bin/env python3
"""StatsBot Local Worker - Hybrid ML Execution Client.

This script connects to the StatsBot server and executes ML jobs locally.
It is designed to be self-contained and run via: curl URL | python3 - --token=TOKEN

Usage:
    python3 statsbot_worker.py --token=YOUR_TOKEN
    python3 statsbot_worker.py --token=YOUR_TOKEN --autostart

Requirements:
    - Python 3.8+
    - Optional: pandas, scikit-learn, xgboost, lightgbm (for ML operations)
"""

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Version
__version__ = "1.0.0"

# Default configuration
DEFAULT_WS_URL = os.getenv("STATSBOT_WS_URL", "ws://localhost:8765/ws")
DEFAULT_RECONNECT_DELAY = 5  # seconds
DEFAULT_MAX_RETRIES = 10


# ============================================================================
# Machine Identification
# ============================================================================


def get_machine_identifier() -> str:
    """
    Get unique machine identifier.

    Returns:
        Machine hostname as identifier
    """
    return socket.gethostname()


# ============================================================================
# Path Validation (Security)
# ============================================================================


def detect_path_traversal(path: str) -> bool:
    """
    Detect path traversal attempts in file paths.

    Args:
        path: File path to check

    Returns:
        True if traversal pattern detected, False otherwise
    """
    dangerous_patterns = [
        "../",
        "..\\",
        "%2e%2e",
        "..%2f",
        "..%5c",
        "%2e%2e%2f",
        "%2e%2e%5c",
    ]
    path_lower = path.lower()
    return any(pattern in path_lower for pattern in dangerous_patterns)


def validate_file_path(
    path: str, max_size_mb: int = 1000, allowed_extensions: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Validate local file path with security checks.

    Args:
        path: File path to validate
        max_size_mb: Maximum file size in MB (default: 1000)
        allowed_extensions: List of allowed extensions (default: ['.csv', '.xlsx', '.parquet'])

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    if allowed_extensions is None:
        allowed_extensions = [".csv", ".xlsx", ".xls", ".parquet"]

    try:
        # Check for path traversal
        if detect_path_traversal(path):
            return False, "Path traversal detected", None

        # Resolve path
        try:
            resolved_path = Path(path).resolve()
        except (ValueError, OSError) as e:
            return False, f"Invalid path format: {str(e)}", None

        # Check existence
        if not resolved_path.exists():
            return False, f"File not found: {resolved_path}", None

        # Check if it's a file (not directory)
        if not resolved_path.is_file():
            return False, f"Not a regular file: {resolved_path}", None

        # Check extension
        if resolved_path.suffix.lower() not in [
            ext.lower() for ext in allowed_extensions
        ]:
            return False, f"Invalid extension: {resolved_path.suffix}", None

        # Check readability
        if not os.access(resolved_path, os.R_OK):
            return False, f"File not readable: {resolved_path}", None

        # Check file size
        size_mb = resolved_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)", None

        # Check not empty
        if size_mb == 0:
            return False, "File is empty", None

        return True, None, resolved_path

    except Exception as e:
        return False, f"Validation error: {str(e)}", None


# ============================================================================
# Model Storage
# ============================================================================


def get_models_dir() -> Path:
    """
    Get or create models directory.

    Returns:
        Path to ~/.statsbot/models/
    """
    home = Path.home()
    models_dir = home / ".statsbot" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_config_dir() -> Path:
    """
    Get or create config directory.

    Returns:
        Path to ~/.statsbot/
    """
    home = Path.home()
    config_dir = home / ".statsbot"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Also ensure models subdirectory exists
    models_dir = config_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return config_dir


def save_worker_config(config: Dict[str, Any]) -> None:
    """
    Save worker configuration to file.

    Args:
        config: Configuration dictionary
    """
    config_dir = get_config_dir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=2))


def load_worker_config() -> Optional[Dict[str, Any]]:
    """
    Load worker configuration from file.

    Returns:
        Configuration dictionary or None if not found
    """
    config_dir = get_config_dir()
    config_file = config_dir / "config.json"

    if not config_file.exists():
        return None

    try:
        return json.loads(config_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ============================================================================
# ML Package Detection
# ============================================================================


def check_ml_packages() -> Dict[str, bool]:
    """
    Check availability of ML packages.

    Returns:
        Dictionary mapping package names to availability
    """
    packages = {}

    try:
        import pandas

        packages["pandas"] = True
    except ImportError:
        packages["pandas"] = False

    try:
        import sklearn

        packages["sklearn"] = True
    except ImportError:
        packages["sklearn"] = False

    try:
        import xgboost

        packages["xgboost"] = True
    except ImportError:
        packages["xgboost"] = False

    try:
        import lightgbm

        packages["lightgbm"] = True
    except ImportError:
        packages["lightgbm"] = False

    return packages


def get_available_packages() -> List[str]:
    """
    Get list of available ML packages.

    Returns:
        List of package names
    """
    packages = check_ml_packages()
    return [name for name, available in packages.items() if available]


# ============================================================================
# Message Handling
# ============================================================================


def create_auth_message(token: str, machine_id: str) -> str:
    """
    Create authentication message.

    Args:
        token: Authentication token
        machine_id: Machine identifier

    Returns:
        JSON-encoded auth message
    """
    msg = {"type": "auth", "token": token, "machine_id": machine_id}
    return json.dumps(msg)


def create_progress_message(
    job_id: str, status: str, progress: int, message: str
) -> str:
    """
    Create progress update message.

    Args:
        job_id: Job identifier
        status: Status string
        progress: Progress percentage (0-100)
        message: Status message

    Returns:
        JSON-encoded progress message
    """
    msg = {
        "type": "progress",
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
    }
    return json.dumps(msg)


def create_result_message(
    job_id: str, success: bool, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None
) -> str:
    """
    Create job result message.

    Args:
        job_id: Job identifier
        success: Whether job succeeded
        data: Result data (if success)
        error: Error message (if failure)

    Returns:
        JSON-encoded result message
    """
    msg = {"type": "result", "job_id": job_id, "success": success}

    if success and data:
        msg["data"] = data
    elif not success and error:
        msg["error"] = error

    return json.dumps(msg, default=str)  # default=str for numpy types


# ============================================================================
# Job Execution
# ============================================================================


def execute_file_info_job(job_id: str, params: Dict[str, Any]) -> str:
    """
    Get file metadata from local filesystem.

    Args:
        job_id: Job identifier
        params: Parameters with 'file_path' key

    Returns:
        JSON-encoded result message
    """
    try:
        file_path = params.get('file_path')
        if not file_path:
            return create_result_message(job_id, False, error="Missing file_path parameter")

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return create_result_message(job_id, False, error=f"File not found: {file_path}")
        if not path.is_file():
            return create_result_message(job_id, False, error=f"Not a file: {file_path}")

        stat = path.stat()
        return create_result_message(job_id, True, data={
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'file_path': str(path)
        })
    except Exception as e:
        return create_result_message(job_id, False, error=str(e))


def execute_load_data_job(job_id: str, params: Dict[str, Any]) -> str:
    """
    Load data from local filesystem and return as dataframe.

    Args:
        job_id: Job identifier
        params: Parameters with 'file_path' key

    Returns:
        JSON-encoded result message with dataframe
    """
    try:
        file_path = params.get('file_path')
        if not file_path:
            return create_result_message(job_id, False, error="Missing file_path parameter")

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return create_result_message(job_id, False, error=f"File not found: {file_path}")
        if not path.is_file():
            return create_result_message(job_id, False, error=f"Not a file: {file_path}")

        # Load data based on file extension
        suffix = path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(path)
        elif suffix in ('.xlsx', '.xls'):
            df = pd.read_excel(path)
        elif suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            return create_result_message(job_id, False, error=f"Unsupported file format: {suffix}")

        return create_result_message(job_id, True, data={
            'dataframe': df.to_dict('records'),
            'columns': list(df.columns),
            'rows': len(df)
        })
    except Exception as e:
        return create_result_message(job_id, False, error=str(e))


def execute_list_models_job(job_id: str) -> str:
    """
    Execute list_models job.

    Args:
        job_id: Job identifier

    Returns:
        JSON-encoded result message
    """
    try:
        models_dir = get_models_dir()
        model_dirs = [d.name for d in models_dir.iterdir() if d.is_dir()]

        # Get metadata for each model
        models = []
        for model_name in model_dirs:
            model_path = models_dir / model_name
            metadata_file = model_path / "metadata.json"

            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    models.append(
                        {
                            "model_id": model_name,
                            "model_type": metadata.get("model_type", "unknown"),
                            "task_type": metadata.get("task_type", "unknown"),
                            "created_at": metadata.get("created_at", "unknown"),
                            "feature_columns": metadata.get("feature_columns", []),
                            "target_column": metadata.get("target_column", "unknown"),
                            "custom_name": metadata.get("custom_name"),
                            "display_name": metadata.get("display_name"),
                        }
                    )
                except (json.JSONDecodeError, OSError):
                    # Skip models with invalid metadata
                    continue

        return create_result_message(job_id, True, data={"models": models})

    except Exception as e:
        return create_result_message(job_id, False, error=f"Failed to list models: {str(e)}")


def execute_delete_model_job(job_id: str, params: Dict[str, Any]) -> str:
    """
    Execute delete_model job.

    Args:
        job_id: Job identifier
        params: Job parameters containing model_id

    Returns:
        JSON-encoded result message
    """
    model_id = params.get("model_id")
    if not model_id:
        return create_result_message(job_id, False, error="Missing model_id parameter")

    try:
        models_dir = get_models_dir()
        model_path = models_dir / model_id

        if not model_path.exists():
            return create_result_message(job_id, False, error=f"Model not found: {model_id}")

        # Delete model directory and all contents
        import shutil
        shutil.rmtree(model_path)

        print(f"🗑️ Deleted model: {model_id}")
        return create_result_message(job_id, True, data={"deleted": model_id})

    except Exception as e:
        return create_result_message(job_id, False, error=f"Failed to delete model: {str(e)}")


def execute_set_model_name_job(job_id: str, params: Dict[str, Any]) -> str:
    """
    Execute set_model_name job - updates model metadata with custom name.

    Args:
        job_id: Job identifier
        params: Job parameters containing model_id and custom_name

    Returns:
        JSON-encoded result message
    """
    model_id = params.get("model_id")
    custom_name = params.get("custom_name")

    if not model_id:
        return create_result_message(job_id, False, error="Missing model_id parameter")
    if not custom_name:
        return create_result_message(job_id, False, error="Missing custom_name parameter")

    try:
        models_dir = get_models_dir()
        model_path = models_dir / model_id

        if not model_path.exists():
            return create_result_message(job_id, False, error=f"Model not found: {model_id}")

        # Update metadata.json with custom_name and display_name
        metadata_file = model_path / "metadata.json"
        if not metadata_file.exists():
            return create_result_message(job_id, False, error="Model metadata not found")

        metadata = json.loads(metadata_file.read_text())
        metadata["custom_name"] = custom_name
        metadata["display_name"] = custom_name
        metadata_file.write_text(json.dumps(metadata, indent=2))

        print(f"📝 Set model name: {model_id} → {custom_name}")
        return create_result_message(job_id, True, data={"model_id": model_id, "custom_name": custom_name})

    except Exception as e:
        return create_result_message(job_id, False, error=f"Failed to set model name: {str(e)}")


def execute_train_job(job_id: str, params: Dict[str, Any], ws_send_callback) -> str:
    """
    Execute training job.

    Args:
        job_id: Job identifier
        params: Training parameters
        ws_send_callback: Async callback to send progress updates

    Returns:
        JSON-encoded result message
    """
    # Check ML packages
    packages = check_ml_packages()
    if not packages.get("pandas") or not packages.get("sklearn"):
        return create_result_message(
            job_id,
            False,
            error="Required packages not available. Please install: pip install pandas scikit-learn",
        )

    try:
        # Import ML engine components
        # Note: This imports from the bot codebase - worker must have access
        # For self-contained worker, we'll implement minimal training logic here
        import pandas as pd

        # Validate file path
        file_path = params.get("file_path")
        if not file_path:
            return create_result_message(job_id, False, error="Missing file_path parameter")

        is_valid, error, resolved_path = validate_file_path(file_path)
        if not is_valid:
            return create_result_message(job_id, False, error=error)

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "loading", 10, "Loading data...")
            )
        )

        # Load data
        if resolved_path.suffix == ".csv":
            df = pd.read_csv(resolved_path)
        elif resolved_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(resolved_path)
        elif resolved_path.suffix == ".parquet":
            df = pd.read_parquet(resolved_path)
        else:
            return create_result_message(job_id, False, error=f"Unsupported format: {resolved_path.suffix}")

        # Extract parameters
        target_column = params.get("target_column")
        feature_columns = params.get("feature_columns")
        model_type = params.get("model_type")
        task_type = params.get("task_type", "regression")
        hyperparameters = params.get("hyperparameters", {})
        custom_name = params.get("custom_name")  # User-provided custom model name

        if not target_column or not feature_columns:
            return create_result_message(
                job_id, False, error="Missing target_column or feature_columns"
            )

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "training", 30, "Preparing data...")
            )
        )

        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Compute dataset stats for display
        from collections import Counter
        dataset_stats = {
            "n_rows": len(y_train) + len(y_test),
            "n_train": len(y_train),
            "n_test": len(y_test)
        }

        if task_type == "classification":
            # Class distribution (using full dataset before split for accurate counts)
            counts = Counter(y)
            total = len(y)
            dataset_stats["class_distribution"] = {
                str(k): {"count": int(v), "pct": round(100*v/total, 1)}
                for k, v in sorted(counts.items())
            }
        else:
            # Regression quartiles
            dataset_stats["quartiles"] = {
                "q1": float(np.percentile(y, 25)),
                "median": float(np.percentile(y, 50)),
                "q3": float(np.percentile(y, 75))
            }

        # Encode categorical columns for all ML models
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                encoders[col] = le

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "training", 50, "Training model...")
            )
        )

        # Train model - comprehensive support for all model types
        import pickle
        hp = hyperparameters or {}
        model = None

        # LightGBM models
        if model_type.startswith("lightgbm"):
            from lightgbm import LGBMClassifier, LGBMRegressor
            if "regression" in model_type:
                model = LGBMRegressor(**hp)
            else:
                model = LGBMClassifier(**hp)

        # XGBoost models
        elif model_type.startswith("xgboost"):
            from xgboost import XGBClassifier, XGBRegressor
            if "regression" in model_type:
                model = XGBRegressor(**hp)
            else:
                model = XGBClassifier(**hp)

        # CatBoost models
        elif model_type.startswith("catboost"):
            from catboost import CatBoostClassifier, CatBoostRegressor
            if "regression" in model_type:
                model = CatBoostRegressor(**hp, verbose=0)
            else:
                model = CatBoostClassifier(**hp, verbose=0)

        # Keras models
        elif model_type.startswith("keras"):
            from tensorflow import keras
            architecture = hp.get("architecture", [64, 32])
            epochs = hp.get("epochs", 50)
            batch_size = hp.get("batch_size", 32)

            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(X_train.shape[1],)))
            for units in architecture:
                model.add(keras.layers.Dense(units, activation="relu"))

            if "regression" in model_type:
                model.add(keras.layers.Dense(1))
                model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            else:
                n_classes = len(y_train.unique())
                if n_classes == 2:
                    model.add(keras.layers.Dense(1, activation="sigmoid"))
                    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                else:
                    model.add(keras.layers.Dense(n_classes, activation="softmax"))
                    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
            # Skip the fit below for Keras
            model._already_fitted = True

        # sklearn regression models
        elif model_type in ["linear", "ridge", "lasso", "elasticnet", "polynomial"]:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline

            if model_type == "linear":
                model = LinearRegression()
            elif model_type == "ridge":
                model = Ridge(**hp)
            elif model_type == "lasso":
                model = Lasso(**hp)
            elif model_type == "elasticnet":
                model = ElasticNet(**hp)
            elif model_type == "polynomial":
                degree = hp.get("degree", 2)
                model = Pipeline([
                    ("poly", PolynomialFeatures(degree=degree)),
                    ("linear", LinearRegression())
                ])

        # sklearn classification models
        elif model_type in ["logistic", "decision_tree", "random_forest", "gradient_boosting", "svm", "naive_bayes"]:
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB

            if model_type == "logistic":
                model = LogisticRegression(max_iter=1000, **hp)
            elif model_type == "decision_tree":
                model = DecisionTreeClassifier(**hp)
            elif model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, **hp)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(**hp)
            elif model_type == "svm":
                model = SVC(**hp)
            elif model_type == "naive_bayes":
                model = GaussianNB()

        # MLP neural networks
        elif model_type in ["mlp_regression", "mlp_classification"]:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            hidden_layers = hp.get("hidden_layer_sizes", (100, 50))
            if model_type == "mlp_regression":
                model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=500, **{k:v for k,v in hp.items() if k != "hidden_layer_sizes"})
            else:
                model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, **{k:v for k,v in hp.items() if k != "hidden_layer_sizes"})

        else:
            return create_result_message(job_id, False, error=f"Unknown model type: {model_type}")

        # Fit model (skip if Keras already fitted)
        if not getattr(model, "_already_fitted", False):
            model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import (
            r2_score, mean_squared_error, accuracy_score,
            precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, brier_score_loss, log_loss
        )
        import numpy as np

        if task_type == "regression":
            y_pred = model.predict(X_test)
            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
            }
        else:  # classification
            y_pred = model.predict(X_test)
            n_classes = len(np.unique(y_test))
            is_binary = n_classes == 2
            average_method = 'binary' if is_binary else 'weighted'

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average=average_method, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average=average_method, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average=average_method, zero_division=0)),
            }

            # Probability-based metrics (if model supports predict_proba)
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)

                    if is_binary:
                        pos_proba = y_proba[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_test, pos_proba))
                        metrics["auc_pr"] = float(average_precision_score(y_test, pos_proba))
                        metrics["brier_score"] = float(brier_score_loss(y_test, pos_proba))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(
                            y_test, y_proba, multi_class='ovr', average='weighted'
                        ))

                    metrics["log_loss"] = float(log_loss(y_test, y_proba))
            except Exception:
                pass  # Skip probability-based metrics if they fail

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "saving", 80, "Saving model...")
            )
        )

        # Save model
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{model_type}_{task_type}_{timestamp}"

        models_dir = get_models_dir()
        model_dir = models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model file
        model_file = model_dir / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Save encoders if any categorical columns were encoded
        if encoders:
            encoders_file = model_dir / "encoders.pkl"
            with open(encoders_file, "wb") as f:
                pickle.dump(encoders, f)

        # Save metadata
        # Generate display_name from custom_name or model_type
        display_name = custom_name if custom_name else f"{model_type.replace('_', ' ').title()} ({len(feature_columns)} features)"

        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "created_at": datetime.datetime.now().isoformat(),
            "custom_name": custom_name,
            "display_name": display_name,
        }
        metadata_file = model_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        # Send completion
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "complete", 100, "Training complete!")
            )
        )

        return create_result_message(
            job_id, True, data={
                "model_id": model_id,
                "metrics": metrics,
                "dataset_stats": dataset_stats,
                "model_info": {
                    "model_type": model_type,
                    "task_type": task_type,
                    "created_at": metadata["created_at"],
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                }
            }
        )

    except Exception as e:
        import traceback

        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        return create_result_message(job_id, False, error=error_msg)


def execute_predict_job(job_id: str, params: Dict[str, Any], ws_send_callback) -> str:
    """
    Execute prediction job.

    Args:
        job_id: Job identifier
        params: Prediction parameters
        ws_send_callback: Async callback to send progress updates

    Returns:
        JSON-encoded result message
    """
    # Check ML packages
    packages = check_ml_packages()
    if not packages.get("pandas") or not packages.get("sklearn"):
        return create_result_message(
            job_id,
            False,
            error="Required packages not available. Please install: pip install pandas scikit-learn",
        )

    try:
        import numpy as np
        import pandas as pd
        import pickle

        # Extract parameters
        model_id = params.get("model_id")
        file_path = params.get("file_path")

        if not model_id:
            return create_result_message(job_id, False, error="Missing model_id parameter")

        # Load model
        models_dir = get_models_dir()
        model_dir = models_dir / model_id

        if not model_dir.exists():
            return create_result_message(job_id, False, error=f"Model not found: {model_id}")

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "loading", 10, "Loading model...")
            )
        )

        # Load metadata
        metadata_file = model_dir / "metadata.json"
        if not metadata_file.exists():
            return create_result_message(job_id, False, error="Model metadata not found")

        metadata = json.loads(metadata_file.read_text())

        # Load model
        model_file = model_dir / "model.pkl"
        if not model_file.exists():
            return create_result_message(job_id, False, error="Model file not found")

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Load encoders if available (for categorical columns)
        encoders = {}
        encoders_file = model_dir / "encoders.pkl"
        if encoders_file.exists():
            with open(encoders_file, "rb") as f:
                encoders = pickle.load(f)

        # Validate file path
        is_valid, error, resolved_path = validate_file_path(file_path)
        if not is_valid:
            return create_result_message(job_id, False, error=error)

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "loading", 30, "Loading data...")
            )
        )

        # Load data
        if resolved_path.suffix == ".csv":
            df = pd.read_csv(resolved_path)
        elif resolved_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(resolved_path)
        elif resolved_path.suffix == ".parquet":
            df = pd.read_parquet(resolved_path)
        else:
            return create_result_message(job_id, False, error=f"Unsupported format: {resolved_path.suffix}")

        # Extract features
        feature_columns = metadata["feature_columns"]
        X = df[feature_columns].copy()

        # Apply encoders to categorical columns if available
        for col, encoder in encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))

        # Send progress update
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "predicting", 60, "Making predictions...")
            )
        )

        # Make predictions
        # For binary classification, output probability scores (0.0-1.0) with 8 decimals
        # For regression/multiclass, use standard predict()
        task_type = metadata.get("task_type", "")

        if task_type == "classification":
            # For binary classification, get probability of positive class
            if hasattr(model, 'predict_proba'):
                # sklearn, XGBoost, LightGBM, CatBoost classifiers
                predictions = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                # Keras already outputs probabilities via sigmoid activation
                raw_predictions = model.predict(X)
                predictions = raw_predictions.flatten()
            else:
                predictions = model.predict(X)

            # Round to 8 decimal places for consistency
            predictions = np.round(predictions, 8)
        else:
            # Regression and multiclass use standard predict
            predictions = model.predict(X)

        # Save full results to CSV file (prevents OOM on Railway bot)
        output_dir = resolved_path.parent
        output_filename = f"predictions_{job_id}.csv"
        output_path = output_dir / output_filename

        df_with_predictions = df.copy()
        df_with_predictions['prediction'] = predictions
        df_with_predictions.to_csv(output_path, index=False)

        # Send completion
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "complete", 100, "Prediction complete!")
            )
        )

        # Return truncated sample only (max 10 rows) to prevent bot OOM
        SAMPLE_SIZE = 10
        sample_size = min(SAMPLE_SIZE, len(df))

        return create_result_message(
            job_id,
            True,
            data={
                "predictions_sample": predictions[:sample_size].tolist(),
                "predictions_count": len(predictions),
                "dataframe_sample": df.head(sample_size).to_dict('records'),
                "dataframe_rows": len(df),
                "dataframe_columns": list(df.columns),
                "output_file": str(output_path),
            },
        )

    except Exception as e:
        import traceback

        error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        return create_result_message(job_id, False, error=error_msg)


def execute_join_job(job_id: str, params: Dict[str, Any], ws_send_callback) -> str:
    """
    Execute dataframe join/union/concat/merge operation.

    Args:
        job_id: Job identifier
        params: Join parameters
            - operation: str - "left_join", "right_join", "inner_join", "outer_join",
                              "cross_join", "union", "concat", "merge"
            - file_paths: List[str] - Paths to input dataframes
            - key_columns: Optional[List[str]] - Join key(s) for join operations
            - output_path: str - Where to save result
        ws_send_callback: Async callback to send progress updates

    Returns:
        JSON-encoded result message
    """
    # Check pandas is available
    packages = check_ml_packages()
    if not packages.get("pandas"):
        return create_result_message(
            job_id,
            False,
            error="Required package not available. Please install: pip install pandas",
        )

    try:
        import pandas as pd
        from pathlib import Path

        # Extract parameters
        operation = params.get("operation")
        file_paths = params.get("file_paths", [])
        key_columns = params.get("key_columns", [])
        output_path = params.get("output_path")
        filters = params.get("filters", [])  # NEW: Optional filters

        if not operation:
            return create_result_message(job_id, False, error="Missing operation parameter")
        if not file_paths or len(file_paths) < 2:
            return create_result_message(job_id, False, error="At least 2 file paths required")
        if not output_path:
            return create_result_message(job_id, False, error="Missing output_path parameter")

        # Send progress: loading dataframes
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "loading", 10, "Loading dataframes...")
            )
        )

        # Load all dataframes
        dfs = []
        for i, path in enumerate(file_paths):
            # Validate path
            is_valid, error, resolved_path = validate_file_path(path)
            if not is_valid:
                return create_result_message(job_id, False, error=f"File {i+1}: {error}")

            # Send progress
            progress_pct = 10 + (i * 20)
            asyncio.create_task(
                ws_send_callback(
                    create_progress_message(
                        job_id, "loading", progress_pct, f"Loading dataframe {i+1} of {len(file_paths)}..."
                    )
                )
            )

            # Load dataframe
            if resolved_path.suffix == ".csv":
                df = pd.read_csv(resolved_path)
            elif resolved_path.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(resolved_path)
            elif resolved_path.suffix == ".parquet":
                df = pd.read_parquet(resolved_path)
            else:
                return create_result_message(
                    job_id, False, error=f"Unsupported format: {resolved_path.suffix}"
                )

            dfs.append(df)

        # Apply filters if provided (NEW)
        if filters:
            asyncio.create_task(
                ws_send_callback(
                    create_progress_message(job_id, "filtering", 60, f"Applying {len(filters)} filter(s)...")
                )
            )

            # Apply each filter to all dataframes
            filtered_dfs = []
            for df in dfs:
                filtered_df = df
                for filter_expr in filters:
                    try:
                        # Convert simple equality to pandas-compatible format
                        # e.g., "month = 1" -> "month == 1" for query()
                        # Handle single = as ==
                        query_expr = filter_expr
                        if " = " in query_expr and " == " not in query_expr and " != " not in query_expr:
                            query_expr = query_expr.replace(" = ", " == ")
                        filtered_df = filtered_df.query(query_expr)
                    except Exception as filter_error:
                        # Skip filters for columns that don't exist in this dataframe
                        print(f"Filter '{filter_expr}' skipped for dataframe: {filter_error}", flush=True)
                        pass
                filtered_dfs.append(filtered_df)
            dfs = filtered_dfs

        # Send progress: executing operation
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "processing", 70, f"Executing {operation}...")
            )
        )

        # Execute operation
        if operation == "left_join":
            result = dfs[0]
            for df in dfs[1:]:
                result = result.merge(df, on=key_columns, how="left")

        elif operation == "right_join":
            result = dfs[0]
            for df in dfs[1:]:
                result = result.merge(df, on=key_columns, how="right")

        elif operation == "inner_join":
            result = dfs[0]
            for df in dfs[1:]:
                result = result.merge(df, on=key_columns, how="inner")

        elif operation == "outer_join":
            result = dfs[0]
            for df in dfs[1:]:
                result = result.merge(df, on=key_columns, how="outer")

        elif operation == "cross_join":
            result = dfs[0]
            for df in dfs[1:]:
                result = result.merge(df, how="cross")

        elif operation == "union":
            # Requires same columns
            result = pd.concat(dfs, ignore_index=True)

        elif operation == "concat":
            # Handles different columns (outer join)
            result = pd.concat(dfs, ignore_index=True, join="outer")

        elif operation == "merge":
            # Standard pandas merge (same as inner join by default)
            result = dfs[0]
            for df in dfs[1:]:
                if key_columns:
                    result = result.merge(df, on=key_columns)
                else:
                    result = result.merge(df)

        else:
            return create_result_message(job_id, False, error=f"Unknown operation: {operation}")

        # Send progress: saving result
        asyncio.create_task(
            ws_send_callback(
                create_progress_message(job_id, "saving", 90, "Saving result...")
            )
        )

        # Save result
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if output_path_obj.suffix == ".parquet":
            result.to_parquet(output_path_obj, index=False)
        elif output_path_obj.suffix in [".xlsx", ".xls"]:
            result.to_excel(output_path_obj, index=False)
        else:
            # Default to CSV
            result.to_csv(output_path_obj, index=False)

        # Return result
        return create_result_message(
            job_id,
            True,
            data={
                "output_file": str(output_path_obj),
                "row_count": len(result),
                "column_count": len(result.columns),
                "columns": list(result.columns)[:20],  # Sample first 20 columns
                "operation": operation,
            },
        )

    except Exception as e:
        import traceback

        error_msg = f"Join operation failed: {str(e)}\n{traceback.format_exc()}"
        return create_result_message(job_id, False, error=error_msg)


def execute_save_file_job(job_id: str, params: Dict[str, Any]) -> str:
    """
    Execute save file job - save dataframe to local file.

    Args:
        job_id: Job identifier
        params: Save parameters (file_path, dataframe)

    Returns:
        JSON-encoded result message
    """
    try:
        import pandas as pd
        from pathlib import Path

        file_path = params.get("file_path")
        dataframe = params.get("dataframe")

        if not file_path:
            return create_result_message(job_id, False, error="Missing file_path parameter")

        if not dataframe:
            return create_result_message(job_id, False, error="Missing dataframe parameter")

        # Convert dict to DataFrame
        df = pd.DataFrame(dataframe)

        # Validate directory exists
        path = Path(file_path)
        if not path.parent.exists():
            return create_result_message(
                job_id, False,
                error=f"Directory not found: {path.parent}"
            )

        # Save file based on extension
        if path.suffix.lower() == '.csv':
            df.to_csv(path, index=False)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df.to_excel(path, index=False)
        else:
            # Default to CSV
            df.to_csv(path, index=False)

        return create_result_message(
            job_id, True,
            data={"file_path": str(path), "rows": len(df)}
        )

    except Exception as e:
        import traceback
        error_msg = f"Save failed: {str(e)}\n{traceback.format_exc()}"
        return create_result_message(job_id, False, error=error_msg)


# ============================================================================
# WebSocket Client
# ============================================================================


class WorkerClient:
    """WebSocket client for worker communication."""

    def __init__(self, ws_url: str, token: str):
        """
        Initialize worker client.

        Args:
            ws_url: WebSocket URL
            token: Authentication token
        """
        self.ws_url = ws_url
        self.token = token
        self.machine_id = get_machine_identifier()
        self.ws = None
        self.authenticated = False
        self.running = False

    async def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns:
            True if connected successfully
        """
        try:
            # Import websockets (may not be available)
            try:
                import websockets
            except ImportError:
                print("ERROR: websockets package not available")
                print("Install with: pip install websockets")
                return False

            print(f"Connecting to {self.ws_url}...")
            self.ws = await websockets.connect(
                self.ws_url,
                max_size=10 * 1024 * 1024 * 1024,  # 10GB for large model results
            )
            print("Connected!")
            return True

        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    async def authenticate(self) -> bool:
        """
        Authenticate with server.

        Returns:
            True if authenticated successfully
        """
        try:
            # Send auth message
            auth_msg = create_auth_message(self.token, self.machine_id)
            await self.ws.send(auth_msg)

            # Wait for response
            response_str = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            response = json.loads(response_str)

            if response.get("type") == "auth_response" and response.get("success"):
                self.authenticated = True
                print(f"✅ Authenticated! User ID: {response.get('user_id')}")
                print(f"Machine: {self.machine_id}")
                print("Ready to receive jobs...")
                return True
            else:
                error = response.get("error", "Unknown error")
                print(f"❌ Authentication failed: {error}")
                return False

        except asyncio.TimeoutError:
            print("❌ Authentication timeout")
            return False
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False

    async def send_message(self, message: str) -> None:
        """
        Send message to server.

        Args:
            message: JSON-encoded message
        """
        if self.ws:
            await self.ws.send(message)

    async def send_message_with_retry(self, message: str, max_retries: int = 3) -> bool:
        """
        Send message with retry logic to handle connection drops.

        Uses EAFP pattern (try/except) since websockets 15.x doesn't have
        .open or .closed attributes on ClientConnection.

        Args:
            message: JSON-encoded message to send
            max_retries: Maximum number of retry attempts

        Returns:
            True if message was sent successfully, False otherwise
        """
        import websockets

        for attempt in range(max_retries):
            try:
                if not self.ws:
                    print(f"⚠️ No WebSocket, reconnecting... (attempt {attempt + 1})")
                    if not await self.connect() or not await self.authenticate():
                        raise ConnectionError("Reconnection failed")

                await self.ws.send(message)
                print(f"📤 Message sent successfully (attempt {attempt + 1})")
                return True

            except websockets.ConnectionClosed as e:
                print(f"⚠️ Connection closed: {e}, reconnecting... (attempt {attempt + 1})")
                self.ws = None
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Send failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    self.ws = None

        print(f"❌ All {max_retries} send attempts failed")
        return False

    async def _keep_alive_ping(self, job_id: str) -> None:
        """
        Send periodic pings to keep WebSocket connection alive during long jobs.

        Uses EAFP pattern (try/except) since websockets 15.x doesn't have
        .open or .closed attributes on ClientConnection.

        Args:
            job_id: Job ID for logging
        """
        import websockets

        try:
            while True:
                await asyncio.sleep(20)  # Ping every 20 seconds
                if self.ws:
                    try:
                        await self.ws.ping()
                        print(f"📡 Ping sent (job: {job_id})")
                    except websockets.ConnectionClosed:
                        print(f"⚠️ WebSocket closed during job {job_id}")
                        break
                else:
                    print(f"⚠️ No WebSocket during job {job_id}")
                    break
        except asyncio.CancelledError:
            pass  # Normal cancellation when job completes
        except Exception as e:
            print(f"⚠️ Ping error: {e}")

    async def listen_for_jobs(self) -> None:
        """Listen for incoming jobs and execute them."""
        self.running = True

        try:
            while self.running:
                try:
                    # Receive message
                    message_str = await self.ws.recv()
                    message = json.loads(message_str)

                    # Handle job
                    if message.get("type") == "job":
                        await self.handle_job(message)

                except json.JSONDecodeError as e:
                    print(f"Invalid JSON received: {str(e)}")
                    continue

        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.running = False

    async def handle_job(self, job: Dict[str, Any]) -> None:
        """
        Handle incoming job.

        Uses heartbeat pings to keep connection alive during long-running jobs
        and retry logic for sending results.

        Args:
            job: Job message
        """
        job_id = job.get("job_id")
        action = job.get("action")
        params = job.get("params", {})

        print(f"\n📋 Received job: {job_id} (action: {action})")

        # Start heartbeat ping task for long-running jobs
        ping_task = asyncio.create_task(self._keep_alive_ping(job_id))

        try:
            # Execute based on action
            if action == "list_models":
                result = execute_list_models_job(job_id)
            elif action == "train":
                result = execute_train_job(job_id, params, self.send_message)
            elif action == "predict":
                result = execute_predict_job(job_id, params, self.send_message)
            elif action == "file_info":
                result = execute_file_info_job(job_id, params)
            elif action == "save_file":
                result = execute_save_file_job(job_id, params)
            elif action == "delete_model":
                result = execute_delete_model_job(job_id, params)
            elif action == "set_model_name":
                result = execute_set_model_name_job(job_id, params)
            elif action == "join":
                result = execute_join_job(job_id, params, self.send_message)
            elif action == "load_data":
                result = execute_load_data_job(job_id, params)
            else:
                result = create_result_message(job_id, False, error=f"Unknown action: {action}")
        finally:
            # Cancel heartbeat task when job completes
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

        # Send result with retry logic for reliability
        print(f"📤 Sending result for job {job_id}...")
        success = await self.send_message_with_retry(result, max_retries=3)
        if success:
            print(f"✅ Job {job_id} completed and result sent")
        else:
            print(f"❌ Job {job_id} completed but failed to send result")

    async def run(self) -> None:
        """Run worker client with reconnection."""
        retry_count = 0
        max_retries = DEFAULT_MAX_RETRIES

        while retry_count < max_retries:
            # Connect
            if not await self.connect():
                retry_count += 1
                delay = DEFAULT_RECONNECT_DELAY * (2**retry_count)  # Exponential backoff
                print(f"Retrying in {delay} seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
                continue

            # Authenticate
            if not await self.authenticate():
                retry_count += 1
                delay = DEFAULT_RECONNECT_DELAY
                print(f"Retrying in {delay} seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
                continue

            # Listen for jobs
            try:
                await self.listen_for_jobs()
            except Exception as e:
                print(f"Error in job listener: {str(e)}")

            # Connection lost - retry
            print("\n⚠️  Connection lost. Reconnecting...")
            retry_count += 1
            await asyncio.sleep(DEFAULT_RECONNECT_DELAY)

        print(f"\n❌ Max retries ({max_retries}) reached. Exiting.")

    async def close(self) -> None:
        """Close WebSocket connection."""
        self.running = False
        if self.ws:
            await self.ws.close()


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="StatsBot Local Worker - Execute ML jobs locally"
    )
    parser.add_argument("--token", required=False, help="Authentication token from /connect")
    parser.add_argument("--autostart", nargs="?", const="on", choices=["on", "off"],
                        help="Install (on) or remove (off) auto-start service")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="WebSocket URL (default: from STATSBOT_WS_URL env)")

    args = parser.parse_args()

    print(f"StatsBot Worker v{__version__}")
    print(f"Machine: {get_machine_identifier()}")
    print()

    # Handle autostart
    if args.autostart:
        try:
            # Import autostart module
            sys.path.insert(0, str(Path(__file__).parent))
            from autostart import install_autostart, remove_autostart

            if args.autostart == "on":
                # Install auto-start
                print("Installing auto-start configuration...\n")

                # Get script path (this script)
                script_path = str(Path(__file__).resolve())

                # Load or create config
                config = load_worker_config()
                if not config:
                    # Need token and ws_url
                    if not args.token:
                        print("❌ Error: --token is required for first-time auto-start setup")
                        print("\nUsage:")
                        print(f"  python3 {Path(__file__).name} --token=YOUR_TOKEN --autostart")
                        sys.exit(1)

                    config = {
                        "ws_url": args.ws_url,
                        "token": args.token,
                        "machine_id": get_machine_identifier()
                    }
                    # Save for future use
                    save_worker_config(config)

                # Install auto-start
                success, message = install_autostart(script_path, config)

                if success:
                    print(f"✅ {message}")
                    print("\n📝 Note: Auto-start configuration has been created.")
                    print("   The worker will start automatically when you log in.")
                    sys.exit(0)
                else:
                    print(f"❌ {message}")
                    sys.exit(1)

            else:  # args.autostart == "off"
                # Remove auto-start
                print("Removing auto-start configuration...\n")

                success, message = remove_autostart()

                if success:
                    print(f"✅ {message}")
                    sys.exit(0)
                else:
                    print(f"❌ {message}")
                    sys.exit(1)

        except ImportError as e:
            print(f"❌ Error: Could not load auto-start module: {str(e)}")
            print("   Make sure autostart.py is in the same directory as the worker script.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during auto-start configuration: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Normal operation - require token
    if not args.token:
        print("❌ Error: --token is required")
        print("\nUsage:")
        print(f"  python3 {Path(__file__).name} --token=YOUR_TOKEN")
        print("\nTo set up auto-start:")
        print(f"  python3 {Path(__file__).name} --token=YOUR_TOKEN --autostart")
        sys.exit(1)

    # Check ML packages
    packages = check_ml_packages()
    available = get_available_packages()

    if available:
        print(f"✅ ML packages available: {', '.join(available)}")
    else:
        print("⚠️  No ML packages found. Install with:")
        print("   pip install pandas scikit-learn xgboost lightgbm")

    print()

    # Save config
    config = {"ws_url": args.ws_url, "token": args.token, "machine_id": get_machine_identifier()}
    save_worker_config(config)

    # Run worker
    try:
        client = WorkerClient(args.ws_url, args.token)
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n\n👋 Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
