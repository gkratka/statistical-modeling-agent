"""
Unit tests for AWS Lambda prediction handler.

Test Coverage:
- Lambda handler entry point
- S3 URI parsing
- Model download and loading
- Data download and processing
- Prediction generation
- Result upload to S3
- Error handling (missing model, invalid data, S3 errors)
"""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch, mock_open

import joblib
import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError


# Mock the lambda module before importing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lambda"))


class TestS3URIParsing:
    """Test S3 URI parsing functionality."""

    def test_parse_valid_s3_uri(self):
        """Test parsing valid S3 URI."""
        from prediction_handler import parse_s3_uri

        bucket, key = parse_s3_uri("s3://my-bucket/path/to/file.pkl")
        assert bucket == "my-bucket"
        assert key == "path/to/file.pkl"

    def test_parse_s3_uri_with_nested_path(self):
        """Test parsing S3 URI with nested directory structure."""
        from prediction_handler import parse_s3_uri

        bucket, key = parse_s3_uri("s3://bucket-name/models/user_123/model.pkl")
        assert bucket == "bucket-name"
        assert key == "models/user_123/model.pkl"

    def test_parse_s3_uri_invalid_format(self):
        """Test parsing invalid S3 URI raises ValueError."""
        from prediction_handler import parse_s3_uri

        with pytest.raises(ValueError, match="Invalid S3 URI format"):
            parse_s3_uri("https://bucket/key")

    def test_parse_s3_uri_missing_bucket(self):
        """Test parsing S3 URI without bucket raises ValueError."""
        from prediction_handler import parse_s3_uri

        with pytest.raises(ValueError, match="Invalid S3 URI format"):
            parse_s3_uri("s3:///path/to/file")

    def test_parse_s3_uri_missing_key(self):
        """Test parsing S3 URI without key raises ValueError."""
        from prediction_handler import parse_s3_uri

        with pytest.raises(ValueError, match="Invalid S3 URI format"):
            parse_s3_uri("s3://bucket/")


class TestLambdaHandler:
    """Test AWS Lambda handler functionality."""

    @pytest.fixture
    def sample_event(self) -> Dict[str, Any]:
        """Provide sample Lambda event payload."""
        return {
            "model_s3_uri": "s3://ml-models/user_123/model.pkl",
            "data_s3_uri": "s3://ml-data/user_123/input.csv",
            "output_s3_uri": "s3://ml-results/user_123/predictions.csv",
            "prediction_column_name": "predicted_price",
            "feature_columns": ["sqft", "bedrooms", "bathrooms"]
        }

    @pytest.fixture
    def sample_model(self):
        """Provide sample trained model."""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        # Train with dummy data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([100, 200, 300])
        model.fit(X, y)
        return model

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Provide sample input data."""
        return pd.DataFrame({
            "sqft": [1500, 2000, 2500],
            "bedrooms": [3, 4, 5],
            "bathrooms": [2, 2, 3],
            "extra_col": ["a", "b", "c"]  # Should be ignored
        })

    @pytest.fixture
    def mock_s3_client(self, sample_model, sample_data):
        """Provide mock S3 client."""
        mock_client = MagicMock()

        # Mock model download
        model_buffer = BytesIO()
        joblib.dump(sample_model, model_buffer)
        model_buffer.seek(0)

        # Mock data download
        data_buffer = BytesIO()
        sample_data.to_csv(data_buffer, index=False)
        data_buffer.seek(0)

        def mock_download(bucket, key, filename):
            if "model.pkl" in filename:
                with open(filename, "wb") as f:
                    f.write(model_buffer.read())
            elif "data.csv" in filename:
                with open(filename, "wb") as f:
                    f.write(data_buffer.read())

        mock_client.download_file.side_effect = mock_download
        return mock_client

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    def test_successful_prediction_workflow(
        self,
        mock_joblib_load,
        mock_boto3_client,
        sample_event,
        sample_model,
        sample_data,
        mock_s3_client,
        tmp_path
    ):
        """Test complete successful prediction workflow."""
        from prediction_handler import lambda_handler

        # Setup mocks
        mock_boto3_client.return_value = mock_s3_client
        mock_joblib_load.return_value = sample_model

        # Mock temp directory
        with patch("prediction_handler.Path") as mock_path:
            mock_path.return_value = tmp_path

            # Execute handler
            response = lambda_handler(sample_event, {})

            # Verify response structure
            assert response["statusCode"] == 200
            body = json.loads(response["body"])
            assert body["success"] is True
            assert "predictions_generated" in body
            assert body["predictions_generated"] == 3
            assert "output_s3_uri" in body

            # Verify S3 operations called
            assert mock_s3_client.download_file.call_count >= 2
            assert mock_s3_client.upload_file.call_count == 1

    @patch("prediction_handler.boto3.client")
    def test_model_download_from_s3(
        self,
        mock_boto3_client,
        sample_event,
        mock_s3_client
    ):
        """Test model is correctly downloaded from S3."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = mock_s3_client

        try:
            lambda_handler(sample_event, {})
        except Exception:
            pass  # We're only testing download call

        # Verify model download called with correct parameters
        download_calls = [
            call for call in mock_s3_client.download_file.call_args_list
            if "model.pkl" in str(call)
        ]
        assert len(download_calls) > 0

    @patch("prediction_handler.boto3.client")
    def test_data_download_from_s3(
        self,
        mock_boto3_client,
        sample_event,
        mock_s3_client
    ):
        """Test data is correctly downloaded from S3."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = mock_s3_client

        try:
            lambda_handler(sample_event, {})
        except Exception:
            pass

        # Verify data download called
        download_calls = [
            call for call in mock_s3_client.download_file.call_args_list
            if "data.csv" in str(call)
        ]
        assert len(download_calls) > 0

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    def test_prediction_generation(
        self,
        mock_joblib_load,
        mock_boto3_client,
        sample_event,
        sample_model,
        mock_s3_client
    ):
        """Test predictions are correctly generated."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = mock_s3_client
        mock_joblib_load.return_value = sample_model

        # Mock predict to return specific values
        expected_predictions = np.array([150.0, 250.0, 350.0])
        sample_model.predict = Mock(return_value=expected_predictions)

        with patch("prediction_handler.pd.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                "sqft": [1500, 2000, 2500],
                "bedrooms": [3, 4, 5],
                "bathrooms": [2, 2, 3]
            })

            response = lambda_handler(sample_event, {})

            # Verify predictions were generated
            assert response["statusCode"] == 200
            assert sample_model.predict.called

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    @patch("prediction_handler.pd.read_csv")
    def test_output_upload_to_s3(
        self,
        mock_read_csv,
        mock_joblib_load,
        mock_boto3_client,
        sample_event,
        sample_model,
        sample_data,
        mock_s3_client
    ):
        """Test prediction results are uploaded to S3."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = mock_s3_client
        mock_joblib_load.return_value = sample_model
        mock_read_csv.return_value = sample_data

        response = lambda_handler(sample_event, {})

        # Verify upload was called
        assert mock_s3_client.upload_file.called
        upload_call = mock_s3_client.upload_file.call_args
        assert upload_call is not None

        # Verify uploaded to correct S3 location
        assert "ml-results" in str(upload_call)

    @patch("prediction_handler.boto3.client")
    def test_error_handling_missing_model(
        self,
        mock_boto3_client,
        sample_event
    ):
        """Test error handling when model download fails."""
        from prediction_handler import lambda_handler

        mock_client = MagicMock()
        mock_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Model not found"}},
            "download_file"
        )
        mock_boto3_client.return_value = mock_client

        response = lambda_handler(sample_event, {})

        assert response["statusCode"] == 400  # Validation error
        body = json.loads(response["body"])
        assert body["success"] is False
        assert "error" in body

    @patch("prediction_handler.boto3.client")
    def test_error_handling_invalid_data(
        self,
        mock_boto3_client,
        sample_event,
        mock_s3_client
    ):
        """Test error handling when data file is invalid."""
        from prediction_handler import lambda_handler

        # Make data download fail
        def mock_download_fail(bucket, key, filename):
            if "data.csv" in filename:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "Data not found"}},
                    "download_file"
                )

        mock_s3_client.download_file.side_effect = mock_download_fail
        mock_boto3_client.return_value = mock_s3_client

        response = lambda_handler(sample_event, {})

        assert response["statusCode"] == 400  # Validation error
        body = json.loads(response["body"])
        assert body["success"] is False

    @patch("prediction_handler.boto3.client")
    def test_error_handling_s3_upload_failure(
        self,
        mock_boto3_client,
        sample_event,
        mock_s3_client
    ):
        """Test error handling when S3 upload fails."""
        from prediction_handler import lambda_handler

        # Make upload fail
        mock_s3_client.upload_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Upload denied"}},
            "upload_file"
        )
        mock_boto3_client.return_value = mock_s3_client

        response = lambda_handler(sample_event, {})

        assert response["statusCode"] == 400  # Validation error (access denied)
        body = json.loads(response["body"])
        assert body["success"] is False

    def test_error_handling_missing_required_fields(self):
        """Test error handling when required event fields are missing."""
        from prediction_handler import lambda_handler

        invalid_event = {
            "model_s3_uri": "s3://bucket/model.pkl"
            # Missing other required fields
        }

        response = lambda_handler(invalid_event, {})

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["success"] is False
        assert "error" in body

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    @patch("prediction_handler.pd.read_csv")
    def test_feature_column_selection(
        self,
        mock_read_csv,
        mock_joblib_load,
        mock_boto3_client,
        sample_event,
        sample_model,
        mock_s3_client
    ):
        """Test that only specified feature columns are used for prediction."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = mock_s3_client
        mock_joblib_load.return_value = sample_model

        # Data with extra columns
        data_with_extra = pd.DataFrame({
            "sqft": [1500, 2000],
            "bedrooms": [3, 4],
            "bathrooms": [2, 2],
            "extra_col1": ["a", "b"],
            "extra_col2": [10, 20]
        })
        mock_read_csv.return_value = data_with_extra

        # Mock predict to capture input
        captured_features = None

        def capture_predict(X):
            nonlocal captured_features
            captured_features = X
            return np.array([150.0, 250.0])

        sample_model.predict = capture_predict

        response = lambda_handler(sample_event, {})

        # Verify only specified features were used
        assert captured_features is not None
        assert captured_features.shape[1] == 3  # Only 3 feature columns
        assert response["statusCode"] == 200

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    @patch("prediction_handler.pd.read_csv")
    def test_prediction_column_added_to_output(
        self,
        mock_read_csv,
        mock_joblib_load,
        mock_boto3_client,
        sample_event,
        sample_model,
        sample_data,
        mock_s3_client,
        tmp_path
    ):
        """Test that prediction column is correctly added to output."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = mock_s3_client
        mock_joblib_load.return_value = sample_model
        mock_read_csv.return_value = sample_data

        # Capture saved dataframe
        saved_df = None

        def capture_upload(local_path, bucket, key):
            nonlocal saved_df
            saved_df = pd.read_csv(local_path)

        mock_s3_client.upload_file.side_effect = capture_upload

        with patch("prediction_handler.Path") as mock_path:
            mock_path.return_value = tmp_path

            response = lambda_handler(sample_event, {})

            # Verify prediction column exists in output
            assert saved_df is not None
            assert sample_event["prediction_column_name"] in saved_df.columns

    def test_lambda_response_format(self):
        """Test that Lambda response follows AWS Lambda format."""
        from prediction_handler import lambda_handler

        event = {"invalid": "event"}
        response = lambda_handler(event, {})

        # Verify response structure
        assert "statusCode" in response
        assert "body" in response
        assert isinstance(response["statusCode"], int)
        assert isinstance(response["body"], str)

        # Verify body is valid JSON
        body = json.loads(response["body"])
        assert "success" in body


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    @patch("prediction_handler.pd.read_csv")
    def test_empty_dataframe(
        self,
        mock_read_csv,
        mock_joblib_load,
        mock_boto3_client,
        tmp_path
    ):
        """Test handling of empty input dataframe."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = MagicMock()
        mock_joblib_load.return_value = MagicMock()
        mock_read_csv.return_value = pd.DataFrame()

        event = {
            "model_s3_uri": "s3://bucket/model.pkl",
            "data_s3_uri": "s3://bucket/data.csv",
            "output_s3_uri": "s3://bucket/output.csv",
            "prediction_column_name": "pred",
            "feature_columns": ["col1"]
        }

        response = lambda_handler(event, {})

        assert response["statusCode"] == 400  # Validation error
        body = json.loads(response["body"])
        assert body["success"] is False

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    @patch("prediction_handler.pd.read_csv")
    def test_missing_feature_columns(
        self,
        mock_read_csv,
        mock_joblib_load,
        mock_boto3_client
    ):
        """Test error when specified feature columns don't exist in data."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = MagicMock()
        mock_joblib_load.return_value = MagicMock()
        mock_read_csv.return_value = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })

        event = {
            "model_s3_uri": "s3://bucket/model.pkl",
            "data_s3_uri": "s3://bucket/data.csv",
            "output_s3_uri": "s3://bucket/output.csv",
            "prediction_column_name": "pred",
            "feature_columns": ["nonexistent_col"]
        }

        response = lambda_handler(event, {})

        assert response["statusCode"] == 400  # Validation error
        body = json.loads(response["body"])
        assert body["success"] is False
        assert "error" in body

    @patch("prediction_handler.boto3.client")
    @patch("prediction_handler.joblib.load")
    def test_corrupted_model_file(
        self,
        mock_joblib_load,
        mock_boto3_client
    ):
        """Test handling of corrupted model file."""
        from prediction_handler import lambda_handler

        mock_boto3_client.return_value = MagicMock()
        mock_joblib_load.side_effect = Exception("Corrupted model file")

        event = {
            "model_s3_uri": "s3://bucket/model.pkl",
            "data_s3_uri": "s3://bucket/data.csv",
            "output_s3_uri": "s3://bucket/output.csv",
            "prediction_column_name": "pred",
            "feature_columns": ["col1"]
        }

        response = lambda_handler(event, {})

        assert response["statusCode"] == 400  # Validation error (invalid model)
        body = json.loads(response["body"])
        assert body["success"] is False
