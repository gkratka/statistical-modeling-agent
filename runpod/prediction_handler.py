"""
RunPod Serverless Handler for ML Predictions.

This handler runs on RunPod serverless endpoints to perform batch predictions
using trained models stored in RunPod network volumes.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.1: RunPod Serverless Handler)
"""

import os
import json
import sys
from io import BytesIO
from typing import Dict, Any, List

try:
    import runpod
    import boto3
    import pandas as pd
    import joblib
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    DEPENDENCIES_AVAILABLE = False


def download_from_storage(s3_client: Any, volume_id: str, key: str) -> bytes:
    """
    Download file from RunPod network volume.

    Args:
        s3_client: boto3 S3 client configured for RunPod storage
        volume_id: Network volume ID
        key: Object key in volume

    Returns:
        File contents as bytes
    """
    obj = s3_client.get_object(Bucket=volume_id, Key=key)
    return obj['Body'].read()


def upload_to_storage(s3_client: Any, volume_id: str, key: str, data: bytes) -> None:
    """
    Upload file to RunPod network volume.

    Args:
        s3_client: boto3 S3 client configured for RunPod storage
        volume_id: Network volume ID
        key: Object key in volume
        data: File contents as bytes
    """
    s3_client.put_object(Bucket=volume_id, Key=key, Body=data)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for ML predictions.

    Input event structure:
    {
        "input": {
            "model_key": "models/user_12345/model_abc",
            "data_key": "datasets/user_12345/test_data.csv",
            "output_key": "predictions/user_12345/predictions.csv",
            "volume_id": "vol-abc123",
            "prediction_column_name": "prediction",  # Optional
            "feature_columns": ["col1", "col2"]  # Optional
        }
    }

    Returns:
    {
        "output": {
            "success": True,
            "output_key": "predictions/user_12345/predictions.csv",
            "num_predictions": 1000,
            "volume_id": "vol-abc123"
        }
    }
    """
    try:
        # Validate dependencies
        if not DEPENDENCIES_AVAILABLE:
            return {
                "output": {
                    "success": False,
                    "error": "Missing required dependencies"
                }
            }

        # Parse input
        input_data = event.get('input', {})
        model_key = input_data.get('model_key')
        data_key = input_data.get('data_key')
        output_key = input_data.get('output_key')
        volume_id = input_data.get('volume_id')
        prediction_column_name = input_data.get('prediction_column_name', 'prediction')
        feature_columns = input_data.get('feature_columns')

        # Validate required parameters
        if not all([model_key, data_key, output_key, volume_id]):
            return {
                "output": {
                    "success": False,
                    "error": "Missing required parameters: model_key, data_key, output_key, volume_id"
                }
            }

        print(f"Starting prediction on RunPod serverless...")
        print(f"Model: {model_key}")
        print(f"Data: {data_key}")
        print(f"Output: {output_key}")

        # Initialize S3 client for RunPod storage
        storage_endpoint = os.getenv('STORAGE_ENDPOINT', 'https://storage.runpod.io')
        storage_access_key = os.getenv('STORAGE_ACCESS_KEY')
        storage_secret_key = os.getenv('STORAGE_SECRET_KEY')

        if not storage_access_key or not storage_secret_key:
            return {
                "output": {
                    "success": False,
                    "error": "Storage credentials not configured (STORAGE_ACCESS_KEY, STORAGE_SECRET_KEY)"
                }
            }

        s3_client = boto3.client(
            's3',
            endpoint_url=storage_endpoint,
            aws_access_key_id=storage_access_key,
            aws_secret_access_key=storage_secret_key
        )

        # Download model
        print("📥 Downloading model...")
        model_bytes = download_from_storage(s3_client, volume_id, f"{model_key}/model.pkl")
        model = joblib.load(BytesIO(model_bytes))
        print(f"✅ Model loaded: {type(model).__name__}")

        # Download preprocessor
        print("📥 Downloading preprocessor...")
        try:
            prep_bytes = download_from_storage(s3_client, volume_id, f"{model_key}/preprocessor.pkl")
            preprocessor = joblib.load(BytesIO(prep_bytes))
            print(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
        except Exception as e:
            print(f"⚠️  No preprocessor found: {e}")
            preprocessor = None

        # Download data
        print("📥 Downloading data...")
        data_bytes = download_from_storage(s3_client, volume_id, data_key)

        # Detect file format and load
        if data_key.endswith('.csv'):
            df = pd.read_csv(BytesIO(data_bytes))
        elif data_key.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(data_bytes))
        elif data_key.endswith('.xlsx') or data_key.endswith('.xls'):
            df = pd.read_excel(BytesIO(data_bytes))
        else:
            return {
                "output": {
                    "success": False,
                    "error": f"Unsupported file format: {data_key}"
                }
            }

        print(f"✅ Loaded {len(df)} rows")

        # Select features if specified
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df

        # Preprocess if preprocessor available
        if preprocessor:
            print("🔧 Preprocessing data...")
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X

        # Make predictions
        print("🎯 Making predictions...")
        predictions = model.predict(X_processed)
        print(f"✅ Generated {len(predictions)} predictions")

        # Add predictions to dataframe
        result_df = df.copy()
        result_df[prediction_column_name] = predictions

        # Also add prediction probabilities if available (for classification)
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_processed)
                # For binary classification
                if probabilities.shape[1] == 2:
                    result_df[f'{prediction_column_name}_probability'] = probabilities[:, 1]
                # For multiclass
                else:
                    for i in range(probabilities.shape[1]):
                        result_df[f'{prediction_column_name}_class_{i}_probability'] = probabilities[:, i]
            except Exception as e:
                print(f"⚠️  Could not add probabilities: {e}")

        # Upload results
        print("💾 Uploading results...")
        result_csv = result_df.to_csv(index=False)
        upload_to_storage(s3_client, volume_id, output_key, result_csv.encode())
        print(f"✅ Results saved to {output_key}")

        return {
            "output": {
                "success": True,
                "output_key": output_key,
                "num_predictions": len(predictions),
                "volume_id": volume_id,
                "input_rows": len(df),
                "output_columns": list(result_df.columns)
            }
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error: {e}")
        print(error_trace)

        return {
            "output": {
                "success": False,
                "error": str(e),
                "traceback": error_trace
            }
        }


# Start RunPod serverless worker
if __name__ == '__main__':
    if not DEPENDENCIES_AVAILABLE:
        print("ERROR: Required dependencies not available. Install requirements.txt")
        sys.exit(1)

    print("🚀 Starting RunPod serverless worker for ML predictions...")
    runpod.serverless.start({'handler': handler})
