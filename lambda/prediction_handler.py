"""
AWS Lambda handler for ML model predictions.

This Lambda function:
1. Downloads trained model from S3
2. Downloads input data from S3
3. Makes predictions using the model
4. Uploads results back to S3

Event payload structure:
{
    "model_s3_uri": "s3://bucket/path/to/model.pkl",
    "data_s3_uri": "s3://bucket/path/to/data.csv",
    "output_s3_uri": "s3://bucket/path/to/output.csv",
    "prediction_column_name": "predicted_value",
    "feature_columns": ["feature1", "feature2", ...]
}

Response structure:
{
    "statusCode": 200,
    "body": {
        "success": true,
        "predictions_generated": 100,
        "output_s3_uri": "s3://bucket/path/to/output.csv"
    }
}
"""

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse S3 URI into bucket and key components.

    Args:
        s3_uri: S3 URI in format s3://bucket/key/path

    Returns:
        Tuple of (bucket_name, key_path)

    Raises:
        ValueError: If URI format is invalid
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}. Must start with 's3://'")

    # Remove s3:// prefix
    path = s3_uri[5:]

    # Split into bucket and key
    parts = path.split("/", 1)

    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid S3 URI format: {s3_uri}. "
            "Expected format: s3://bucket-name/path/to/file"
        )

    bucket = parts[0]
    key = parts[1]

    return bucket, key


def validate_event(event: Dict[str, Any]) -> None:
    """
    Validate required fields in Lambda event.

    Args:
        event: Lambda event payload

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        "model_s3_uri",
        "data_s3_uri",
        "output_s3_uri",
        "prediction_column_name",
        "feature_columns"
    ]

    missing_fields = [field for field in required_fields if field not in event]

    if missing_fields:
        raise ValueError(
            f"Missing required fields in event: {', '.join(missing_fields)}"
        )

    # Validate feature_columns is a list
    if not isinstance(event["feature_columns"], list):
        raise ValueError("feature_columns must be a list")

    if len(event["feature_columns"]) == 0:
        raise ValueError("feature_columns must contain at least one column")


def download_from_s3(
    s3_client: Any,
    s3_uri: str,
    local_path: str
) -> None:
    """
    Download file from S3 to local filesystem.

    Args:
        s3_client: Boto3 S3 client
        s3_uri: S3 URI of source file
        local_path: Local filesystem path for downloaded file

    Raises:
        ClientError: If S3 download fails
    """
    bucket, key = parse_s3_uri(s3_uri)

    try:
        s3_client.download_file(bucket, key, local_path)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            raise ValueError(f"File not found in S3: {s3_uri}") from e
        elif error_code == "NoSuchBucket":
            raise ValueError(f"Bucket not found: {bucket}") from e
        else:
            raise ValueError(f"Failed to download from S3: {str(e)}") from e


def upload_to_s3(
    s3_client: Any,
    local_path: str,
    s3_uri: str
) -> None:
    """
    Upload file from local filesystem to S3.

    Args:
        s3_client: Boto3 S3 client
        local_path: Local filesystem path of file to upload
        s3_uri: S3 URI destination

    Raises:
        ClientError: If S3 upload fails
    """
    bucket, key = parse_s3_uri(s3_uri)

    try:
        s3_client.upload_file(local_path, bucket, key)
    except ClientError as e:
        raise ValueError(f"Failed to upload to S3: {str(e)}") from e


def load_model(model_path: str) -> Any:
    """
    Load trained model from disk.

    Args:
        model_path: Path to pickled model file

    Returns:
        Loaded model object

    Raises:
        Exception: If model loading fails
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}") from e


def load_data(data_path: str, feature_columns: list[str]) -> pd.DataFrame:
    """
    Load and validate input data.

    Args:
        data_path: Path to CSV data file
        feature_columns: Required feature columns

    Returns:
        DataFrame with validated features

    Raises:
        ValueError: If data loading or validation fails
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}") from e

    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("Input data is empty")

    # Validate feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns: {', '.join(missing_cols)}"
        )

    return df


def make_predictions(
    model: Any,
    data: pd.DataFrame,
    feature_columns: list[str]
) -> pd.Series:
    """
    Generate predictions using trained model.

    Args:
        model: Trained model object
        data: Input dataframe
        feature_columns: Features to use for prediction

    Returns:
        Series of predictions

    Raises:
        Exception: If prediction fails
    """
    try:
        # Extract features in correct order
        X = data[feature_columns]

        # Generate predictions
        predictions = model.predict(X)

        return pd.Series(predictions, index=data.index)
    except Exception as e:
        raise ValueError(f"Failed to generate predictions: {str(e)}") from e


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for ML predictions.

    Args:
        event: Lambda event payload with S3 URIs and configuration
        context: Lambda context object

    Returns:
        Lambda response with statusCode and body
    """
    try:
        # Validate event payload
        validate_event(event)

        # Extract event parameters
        model_s3_uri = event["model_s3_uri"]
        data_s3_uri = event["data_s3_uri"]
        output_s3_uri = event["output_s3_uri"]
        prediction_column_name = event["prediction_column_name"]
        feature_columns = event["feature_columns"]

        # Initialize S3 client
        s3_client = boto3.client("s3")

        # Define temporary file paths (Lambda has /tmp available)
        tmp_dir = Path("/tmp")
        model_path = tmp_dir / "model.pkl"
        data_path = tmp_dir / "data.csv"
        output_path = tmp_dir / "predictions.csv"

        # Step 1: Download model from S3
        print(f"Downloading model from {model_s3_uri}")
        download_from_s3(s3_client, model_s3_uri, str(model_path))

        # Step 2: Download input data from S3
        print(f"Downloading data from {data_s3_uri}")
        download_from_s3(s3_client, data_s3_uri, str(data_path))

        # Step 3: Load model
        print("Loading model")
        model = load_model(str(model_path))

        # Step 4: Load and validate data
        print("Loading and validating data")
        data = load_data(str(data_path), feature_columns)

        # Step 5: Generate predictions
        print(f"Generating predictions for {len(data)} rows")
        predictions = make_predictions(model, data, feature_columns)

        # Step 6: Add predictions to data
        data[prediction_column_name] = predictions

        # Step 7: Save results to disk
        print(f"Saving results to {output_path}")
        data.to_csv(output_path, index=False)

        # Step 8: Upload results to S3
        print(f"Uploading results to {output_s3_uri}")
        upload_to_s3(s3_client, str(output_path), output_s3_uri)

        # Success response
        response_body = {
            "success": True,
            "predictions_generated": len(predictions),
            "output_s3_uri": output_s3_uri,
            "prediction_column_name": prediction_column_name
        }

        return {
            "statusCode": 200,
            "body": json.dumps(response_body)
        }

    except ValueError as e:
        # User error - bad input
        print(f"Validation error: {str(e)}")
        return {
            "statusCode": 400,
            "body": json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "ValidationError"
            })
        }

    except Exception as e:
        # System error
        print(f"System error: {str(e)}")
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            })
        }
