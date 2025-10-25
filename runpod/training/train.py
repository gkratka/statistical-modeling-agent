"""
RunPod Training Script for ML Models.

This script runs on RunPod GPU pods to train ML models.
It downloads datasets from network volume, trains models,
and uploads results back to storage.

Environment Variables Required:
    STORAGE_ENDPOINT: RunPod storage endpoint URL
    STORAGE_ACCESS_KEY: Storage access key
    STORAGE_SECRET_KEY: Storage secret key
    VOLUME_ID: RunPod network volume ID
    DATASET_KEY: Storage key for dataset
    MODEL_ID: Unique model identifier
    MODEL_TYPE: Type of model to train
    TARGET_COLUMN: Target variable name
    FEATURE_COLUMNS: Comma-separated feature names
    HYPERPARAMETERS: JSON-encoded hyperparameters

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 6.3: RunPod Training Script)
"""

import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, Optional

import boto3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def download_from_storage(s3_client, volume_id: str, key: str) -> bytes:
    """Download file from RunPod network volume."""
    try:
        obj = s3_client.get_object(Bucket=volume_id, Key=key)
        return obj['Body'].read()
    except Exception as e:
        print(f"❌ Error downloading {key}: {e}")
        raise


def upload_to_storage(s3_client, volume_id: str, key: str, data: bytes) -> None:
    """Upload file to RunPod network volume."""
    try:
        s3_client.put_object(Bucket=volume_id, Key=key, Body=data)
    except Exception as e:
        print(f"❌ Error uploading {key}: {e}")
        raise


def create_model(model_type: str, hyperparameters: Dict[str, Any]):
    """
    Create ML model instance based on type.

    Args:
        model_type: Type of model (e.g., 'linear', 'random_forest')
        hyperparameters: Model hyperparameters

    Returns:
        Scikit-learn model instance
    """
    # Regression models
    if model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**hyperparameters)

    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(**hyperparameters)

    elif model_type == 'lasso':
        from sklearn.linear_model import Lasso
        return Lasso(**hyperparameters)

    elif model_type == 'elasticnet':
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**hyperparameters)

    # Classification models
    elif model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**hyperparameters)

    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**hyperparameters)

    elif model_type == 'random_forest':
        task_type = hyperparameters.pop('task_type', 'regression')
        if task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**hyperparameters)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**hyperparameters)

    elif model_type == 'gradient_boosting':
        task_type = hyperparameters.pop('task_type', 'regression')
        if task_type == 'classification':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**hyperparameters)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**hyperparameters)

    elif model_type == 'xgboost':
        import xgboost as xgb
        task_type = hyperparameters.pop('task_type', 'regression')
        if task_type == 'classification':
            return xgb.XGBClassifier(**hyperparameters)
        else:
            return xgb.XGBRegressor(**hyperparameters)

    elif model_type == 'lightgbm':
        import lightgbm as lgb
        task_type = hyperparameters.pop('task_type', 'regression')
        if task_type == 'classification':
            return lgb.LGBMClassifier(**hyperparameters)
        else:
            return lgb.LGBMRegressor(**hyperparameters)

    elif model_type == 'catboost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        task_type = hyperparameters.pop('task_type', 'regression')
        if task_type == 'classification':
            return CatBoostClassifier(**hyperparameters, verbose=False)
        else:
            return CatBoostRegressor(**hyperparameters, verbose=False)

    elif model_type == 'svm':
        from sklearn.svm import SVC
        return SVC(**hyperparameters)

    elif model_type == 'naive_bayes':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**hyperparameters)

    # Neural networks
    elif model_type in ['mlp_regression', 'mlp_classification']:
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        if model_type == 'mlp_classification':
            return MLPClassifier(**hyperparameters)
        else:
            return MLPRegressor(**hyperparameters)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test, y_test, model_type: str) -> Dict[str, float]:
    """
    Evaluate trained model.

    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test target
        model_type: Type of model

    Returns:
        Metrics dictionary
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score
    )

    # Determine if classification or regression
    is_classification = model_type in [
        'logistic', 'decision_tree', 'svm', 'naive_bayes',
        'mlp_classification'
    ] or (hasattr(model, 'predict_proba') and
          model_type in ['random_forest', 'gradient_boosting', 'xgboost',
                        'lightgbm', 'catboost'])

    if is_classification:
        y_pred = model.predict(X_test)

        # Multiclass or binary
        average = 'binary' if len(set(y_test)) == 2 else 'weighted'

        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average=average, zero_division=0))
        }
    else:
        # Regression
        y_pred = model.predict(X_test)
        return {
            'r2_score': float(r2_score(y_test, y_pred)),
            'mse': float(mean_squared_error(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(mean_squared_error(y_test, y_pred, squared=False))
        }


def main():
    """Main training script execution."""
    # Parse environment variables
    storage_endpoint = os.getenv('STORAGE_ENDPOINT')
    storage_access_key = os.getenv('STORAGE_ACCESS_KEY')
    storage_secret_key = os.getenv('STORAGE_SECRET_KEY')
    volume_id = os.getenv('VOLUME_ID')
    dataset_key = os.getenv('DATASET_KEY')
    model_id = os.getenv('MODEL_ID')
    model_type = os.getenv('MODEL_TYPE')
    target_column = os.getenv('TARGET_COLUMN')
    feature_columns_str = os.getenv('FEATURE_COLUMNS', '')
    hyperparameters_str = os.getenv('HYPERPARAMETERS', '{}')

    # Validate required variables
    required_vars = {
        'STORAGE_ENDPOINT': storage_endpoint,
        'STORAGE_ACCESS_KEY': storage_access_key,
        'STORAGE_SECRET_KEY': storage_secret_key,
        'VOLUME_ID': volume_id,
        'DATASET_KEY': dataset_key,
        'MODEL_ID': model_id,
        'MODEL_TYPE': model_type,
        'TARGET_COLUMN': target_column,
        'FEATURE_COLUMNS': feature_columns_str
    }

    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Parse parameters
    feature_columns = feature_columns_str.split(',')
    try:
        hyperparameters = json.loads(hyperparameters_str)
    except json.JSONDecodeError:
        print(f"❌ Invalid HYPERPARAMETERS JSON: {hyperparameters_str}")
        sys.exit(1)

    print("🚀 Starting ML training on RunPod GPU pod...")
    print(f"Model type: {model_type}")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_key}")
    print(f"Target: {target_column}")
    print(f"Features: {', '.join(feature_columns)}")
    print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    print()

    # Initialize S3 client
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=storage_endpoint,
            aws_access_key_id=storage_access_key,
            aws_secret_access_key=storage_secret_key
        )
        print("✅ Connected to RunPod storage")
    except Exception as e:
        print(f"❌ Failed to connect to storage: {e}")
        sys.exit(1)

    # Download dataset
    print(f"📥 Downloading dataset from {dataset_key}...")
    try:
        data_bytes = download_from_storage(s3_client, volume_id, dataset_key)

        # Detect file format and load
        if dataset_key.endswith('.csv'):
            df = pd.read_csv(BytesIO(data_bytes))
        elif dataset_key.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(data_bytes))
        elif dataset_key.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(BytesIO(data_bytes))
        else:
            # Default to CSV
            df = pd.read_csv(BytesIO(data_bytes))

        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    # Prepare data
    print("🔧 Preparing training data...")
    try:
        # Validate columns exist
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            print(f"❌ Missing feature columns: {', '.join(missing_cols)}")
            sys.exit(1)

        if target_column not in df.columns:
            print(f"❌ Missing target column: {target_column}")
            sys.exit(1)

        X = df[feature_columns]
        y = df[target_column]

        # Handle missing values
        if X.isnull().any().any():
            print("⚠️  Handling missing values in features...")
            X = X.fillna(X.mean())

        if y.isnull().any():
            print("⚠️  Handling missing values in target...")
            y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else y.mode()[0])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        sys.exit(1)

    # Preprocessing
    print("🔧 Preprocessing data...")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✅ Data scaled successfully")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        sys.exit(1)

    # Train model
    print(f"🎯 Training {model_type} model...")
    try:
        model = create_model(model_type, hyperparameters.copy())
        model.fit(X_train_scaled, y_train)
        print("✅ Model training complete!")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        sys.exit(1)

    # Evaluate
    print("📊 Evaluating model...")
    try:
        metrics = evaluate_model(model, X_test_scaled, y_test, model_type)
        print("✅ Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        metrics = {}

    # Save model
    print("💾 Saving model to storage...")
    try:
        model_bytes = BytesIO()
        joblib.dump(model, model_bytes)
        upload_to_storage(
            s3_client,
            volume_id,
            f"models/{model_id}/model.pkl",
            model_bytes.getvalue()
        )
        print(f"✅ Model saved to models/{model_id}/model.pkl")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        sys.exit(1)

    # Save preprocessor
    print("💾 Saving preprocessor...")
    try:
        prep_bytes = BytesIO()
        joblib.dump(scaler, prep_bytes)
        upload_to_storage(
            s3_client,
            volume_id,
            f"models/{model_id}/preprocessor.pkl",
            prep_bytes.getvalue()
        )
        print(f"✅ Preprocessor saved to models/{model_id}/preprocessor.pkl")
    except Exception as e:
        print(f"❌ Failed to save preprocessor: {e}")
        sys.exit(1)

    # Save metadata
    print("💾 Saving metadata...")
    try:
        metadata = {
            'model_type': model_type,
            'model_id': model_id,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'hyperparameters': hyperparameters,
            'metrics': metrics,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'dataset_key': dataset_key
        }
        upload_to_storage(
            s3_client,
            volume_id,
            f"models/{model_id}/metadata.json",
            json.dumps(metadata, indent=2).encode()
        )
        print(f"✅ Metadata saved to models/{model_id}/metadata.json")
    except Exception as e:
        print(f"❌ Failed to save metadata: {e}")
        sys.exit(1)

    print()
    print("="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Model ID: {model_id}")
    print(f"Model type: {model_type}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print("="*60)

    # Training complete - pod will auto-terminate
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
