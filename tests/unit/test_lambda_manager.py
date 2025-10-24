"""
Unit tests for LambdaManager class.

Tests AWS Lambda orchestration capabilities including function invocation,
deployment, and layer management for serverless ML predictions.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.4-4.5: Lambda Manager TDD)
"""

import json
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import LambdaError
from src.cloud.lambda_manager import LambdaManager


class TestLambdaManagerInitialization:
    """Test LambdaManager initialization."""

    def test_init_success(self) -> None:
        """Test successful LambdaManager initialization."""
        # Arrange
        mock_aws_client = Mock(spec=AWSClient)
        mock_config = Mock(spec=CloudConfig)
        mock_config.lambda_function_name = "ml-prediction-function"
        mock_config.lambda_memory_mb = 2048
        mock_config.lambda_timeout_seconds = 300
        mock_config.iam_role_arn = "arn:aws:iam::123456789012:role/lambda-execution"

        # Act
        manager = LambdaManager(aws_client=mock_aws_client, config=mock_config)

        # Assert
        assert manager is not None
        assert manager._aws_client == mock_aws_client
        assert manager._config == mock_config
        assert manager._lambda_client is not None


class TestInvokePrediction:
    """Test synchronous Lambda invocation for predictions."""

    @pytest.fixture
    def mock_lambda_client(self) -> Mock:
        """Create mock Lambda client."""
        client = Mock()
        return client

    @pytest.fixture
    def lambda_manager(self, mock_lambda_client: Mock) -> LambdaManager:
        """Create LambdaManager with mocked dependencies."""
        mock_aws_client = Mock(spec=AWSClient)
        mock_aws_client.get_lambda_client.return_value = mock_lambda_client

        mock_config = Mock(spec=CloudConfig)
        mock_config.lambda_function_name = "ml-prediction-function"
        mock_config.lambda_memory_mb = 2048
        mock_config.lambda_timeout_seconds = 300
        mock_config.iam_role_arn = "arn:aws:iam::123456789012:role/lambda-execution"

        return LambdaManager(aws_client=mock_aws_client, config=mock_config)

    def test_invoke_prediction_success(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test successful synchronous prediction invocation."""
        # Arrange
        model_s3_uri = "s3://ml-models/user_123/model.pkl"
        data_s3_uri = "s3://ml-data/user_123/input.csv"
        output_s3_uri = "s3://ml-results/user_123/predictions.csv"
        prediction_column = "predicted_price"
        feature_columns = ["sqft", "bedrooms", "bathrooms"]

        mock_response = {
            "StatusCode": 200,
            "Payload": Mock(read=lambda: json.dumps({
                "statusCode": 200,
                "body": {
                    "output_s3_uri": output_s3_uri,
                    "num_predictions": 100,
                    "execution_time_seconds": 2.5
                }
            }).encode())
        }
        mock_lambda_client.invoke.return_value = mock_response

        # Act
        result = lambda_manager.invoke_prediction(
            model_s3_uri=model_s3_uri,
            data_s3_uri=data_s3_uri,
            output_s3_uri=output_s3_uri,
            prediction_column_name=prediction_column,
            feature_columns=feature_columns
        )

        # Assert
        assert result["statusCode"] == 200
        assert result["body"]["output_s3_uri"] == output_s3_uri
        assert result["body"]["num_predictions"] == 100
        assert result["body"]["execution_time_seconds"] == 2.5

        # Verify Lambda invoke call
        mock_lambda_client.invoke.assert_called_once()
        call_args = mock_lambda_client.invoke.call_args
        assert call_args[1]["FunctionName"] == "ml-prediction-function"
        assert call_args[1]["InvocationType"] == "RequestResponse"

        # Verify payload structure
        payload = json.loads(call_args[1]["Payload"])
        assert payload["model_s3_uri"] == model_s3_uri
        assert payload["data_s3_uri"] == data_s3_uri
        assert payload["output_s3_uri"] == output_s3_uri
        assert payload["prediction_column_name"] == prediction_column
        assert payload["feature_columns"] == feature_columns

    def test_invoke_prediction_without_feature_columns(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test prediction invocation without specifying feature columns."""
        # Arrange
        model_s3_uri = "s3://ml-models/user_123/model.pkl"
        data_s3_uri = "s3://ml-data/user_123/input.csv"
        output_s3_uri = "s3://ml-results/user_123/predictions.csv"
        prediction_column = "predicted_price"

        mock_response = {
            "StatusCode": 200,
            "Payload": Mock(read=lambda: json.dumps({
                "statusCode": 200,
                "body": {"output_s3_uri": output_s3_uri}
            }).encode())
        }
        mock_lambda_client.invoke.return_value = mock_response

        # Act
        result = lambda_manager.invoke_prediction(
            model_s3_uri=model_s3_uri,
            data_s3_uri=data_s3_uri,
            output_s3_uri=output_s3_uri,
            prediction_column_name=prediction_column
        )

        # Assert
        assert result["statusCode"] == 200

        # Verify payload doesn't include feature_columns
        call_args = mock_lambda_client.invoke.call_args
        payload = json.loads(call_args[1]["Payload"])
        assert "feature_columns" not in payload or payload["feature_columns"] is None

    def test_invoke_prediction_lambda_error_response(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test handling of Lambda function error response (statusCode != 200)."""
        # Arrange
        model_s3_uri = "s3://ml-models/user_123/model.pkl"
        data_s3_uri = "s3://ml-data/user_123/input.csv"
        output_s3_uri = "s3://ml-results/user_123/predictions.csv"
        prediction_column = "predicted_price"

        mock_response = {
            "StatusCode": 200,
            "Payload": Mock(read=lambda: json.dumps({
                "statusCode": 500,
                "body": {
                    "error": "Model file not found in S3",
                    "error_type": "ModelNotFoundError"
                }
            }).encode())
        }
        mock_lambda_client.invoke.return_value = mock_response

        # Act & Assert
        with pytest.raises(LambdaError) as exc_info:
            lambda_manager.invoke_prediction(
                model_s3_uri=model_s3_uri,
                data_s3_uri=data_s3_uri,
                output_s3_uri=output_s3_uri,
                prediction_column_name=prediction_column
            )

        assert "Lambda function returned error" in str(exc_info.value)
        assert "500" in str(exc_info.value)
        assert exc_info.value.function_name == "ml-prediction-function"

    def test_invoke_prediction_boto3_client_error(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test handling of boto3 ClientError during invocation."""
        # Arrange
        model_s3_uri = "s3://ml-models/user_123/model.pkl"
        data_s3_uri = "s3://ml-data/user_123/input.csv"
        output_s3_uri = "s3://ml-results/user_123/predictions.csv"
        prediction_column = "predicted_price"

        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Function not found: ml-prediction-function"
            },
            "ResponseMetadata": {"RequestId": "test-request-123"}
        }
        mock_lambda_client.invoke.side_effect = ClientError(
            error_response,
            "Invoke"
        )

        # Act & Assert
        with pytest.raises(LambdaError) as exc_info:
            lambda_manager.invoke_prediction(
                model_s3_uri=model_s3_uri,
                data_s3_uri=data_s3_uri,
                output_s3_uri=output_s3_uri,
                prediction_column_name=prediction_column
            )

        assert "Failed to invoke Lambda function" in str(exc_info.value)
        assert exc_info.value.error_code == "ResourceNotFoundException"
        assert exc_info.value.function_name == "ml-prediction-function"

    def test_invoke_prediction_boto3_generic_error(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test handling of generic boto3 exception during invocation."""
        # Arrange
        model_s3_uri = "s3://ml-models/user_123/model.pkl"
        data_s3_uri = "s3://ml-data/user_123/input.csv"
        output_s3_uri = "s3://ml-results/user_123/predictions.csv"
        prediction_column = "predicted_price"

        mock_lambda_client.invoke.side_effect = Exception("Network timeout")

        # Act & Assert
        with pytest.raises(LambdaError) as exc_info:
            lambda_manager.invoke_prediction(
                model_s3_uri=model_s3_uri,
                data_s3_uri=data_s3_uri,
                output_s3_uri=output_s3_uri,
                prediction_column_name=prediction_column
            )

        assert "Unexpected error invoking Lambda" in str(exc_info.value)
        assert "Network timeout" in str(exc_info.value)


class TestInvokeAsync:
    """Test asynchronous Lambda invocation for predictions."""

    @pytest.fixture
    def mock_lambda_client(self) -> Mock:
        """Create mock Lambda client."""
        return Mock()

    @pytest.fixture
    def lambda_manager(self, mock_lambda_client: Mock) -> LambdaManager:
        """Create LambdaManager with mocked dependencies."""
        mock_aws_client = Mock(spec=AWSClient)
        mock_aws_client.get_lambda_client.return_value = mock_lambda_client

        mock_config = Mock(spec=CloudConfig)
        mock_config.lambda_function_name = "ml-prediction-function"
        mock_config.lambda_memory_mb = 2048
        mock_config.lambda_timeout_seconds = 300
        mock_config.iam_role_arn = "arn:aws:iam::123456789012:role/lambda-execution"

        return LambdaManager(aws_client=mock_aws_client, config=mock_config)

    def test_invoke_async_success(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test successful asynchronous prediction invocation."""
        # Arrange
        model_s3_uri = "s3://ml-models/user_123/model.pkl"
        data_s3_uri = "s3://ml-data/user_123/input.csv"
        output_s3_uri = "s3://ml-results/user_123/predictions.csv"
        prediction_column = "predicted_price"

        mock_response = {
            "StatusCode": 202,
            "ResponseMetadata": {"RequestId": "async-request-xyz-789"}
        }
        mock_lambda_client.invoke.return_value = mock_response

        # Act
        request_id = lambda_manager.invoke_async(
            model_s3_uri=model_s3_uri,
            data_s3_uri=data_s3_uri,
            output_s3_uri=output_s3_uri,
            prediction_column_name=prediction_column
        )

        # Assert
        assert request_id == "async-request-xyz-789"

        # Verify Lambda invoke call with Event invocation type
        mock_lambda_client.invoke.assert_called_once()
        call_args = mock_lambda_client.invoke.call_args
        assert call_args[1]["FunctionName"] == "ml-prediction-function"
        assert call_args[1]["InvocationType"] == "Event"

    def test_invoke_async_with_feature_columns(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test async invocation with feature columns specified."""
        # Arrange
        feature_columns = ["feature1", "feature2", "feature3"]
        mock_response = {
            "StatusCode": 202,
            "ResponseMetadata": {"RequestId": "async-request-abc-123"}
        }
        mock_lambda_client.invoke.return_value = mock_response

        # Act
        request_id = lambda_manager.invoke_async(
            model_s3_uri="s3://ml-models/model.pkl",
            data_s3_uri="s3://ml-data/input.csv",
            output_s3_uri="s3://ml-results/output.csv",
            prediction_column_name="prediction",
            feature_columns=feature_columns
        )

        # Assert
        assert request_id == "async-request-abc-123"

        # Verify payload includes feature_columns
        call_args = mock_lambda_client.invoke.call_args
        payload = json.loads(call_args[1]["Payload"])
        assert payload["feature_columns"] == feature_columns

    def test_invoke_async_boto3_error(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test handling of boto3 error during async invocation."""
        # Arrange
        error_response = {
            "Error": {
                "Code": "ServiceException",
                "Message": "Internal service error"
            },
            "ResponseMetadata": {"RequestId": "error-request-456"}
        }
        mock_lambda_client.invoke.side_effect = ClientError(
            error_response,
            "Invoke"
        )

        # Act & Assert
        with pytest.raises(LambdaError) as exc_info:
            lambda_manager.invoke_async(
                model_s3_uri="s3://ml-models/model.pkl",
                data_s3_uri="s3://ml-data/input.csv",
                output_s3_uri="s3://ml-results/output.csv",
                prediction_column_name="prediction"
            )

        assert "Failed to invoke Lambda function asynchronously" in str(exc_info.value)
        assert exc_info.value.error_code == "ServiceException"
        assert exc_info.value.invocation_type == "Event"


class TestDeployFunction:
    """Test Lambda function deployment and updates."""

    @pytest.fixture
    def mock_lambda_client(self) -> Mock:
        """Create mock Lambda client."""
        return Mock()

    @pytest.fixture
    def lambda_manager(self, mock_lambda_client: Mock) -> LambdaManager:
        """Create LambdaManager with mocked dependencies."""
        mock_aws_client = Mock(spec=AWSClient)
        mock_aws_client.get_lambda_client.return_value = mock_lambda_client

        mock_config = Mock(spec=CloudConfig)
        mock_config.lambda_function_name = "ml-prediction-function"
        mock_config.lambda_memory_mb = 2048
        mock_config.lambda_timeout_seconds = 300
        mock_config.iam_role_arn = "arn:aws:iam::123456789012:role/lambda-execution"

        return LambdaManager(aws_client=mock_aws_client, config=mock_config)

    def test_deploy_function_create_new(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test creating a new Lambda function (function does not exist)."""
        # Arrange
        zip_file_path = "/path/to/function.zip"
        layer_arn = "arn:aws:lambda:us-east-1:123456789012:layer:ml-deps:1"

        # Mock get_function to raise ResourceNotFoundException (function doesn't exist)
        error_response = {
            "Error": {"Code": "ResourceNotFoundException"},
            "ResponseMetadata": {"RequestId": "test-req-1"}
        }
        mock_lambda_client.get_function.side_effect = ClientError(
            error_response,
            "GetFunction"
        )

        # Mock successful create_function
        with open("/dev/null", "rb") as f:
            zip_content = f.read()

        mock_create_response = {
            "FunctionName": "ml-prediction-function",
            "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:ml-prediction-function",
            "Runtime": "python3.11",
            "Handler": "prediction_handler.lambda_handler",
            "CodeSize": 1024,
            "MemorySize": 2048,
            "Timeout": 300,
            "State": "Active",
            "LastUpdateStatus": "Successful"
        }
        mock_lambda_client.create_function.return_value = mock_create_response

        # Act
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_zip_data")):
            result = lambda_manager.deploy_function(
                zip_file_path=zip_file_path,
                layer_arn=layer_arn
            )

        # Assert
        assert result["FunctionName"] == "ml-prediction-function"
        assert result["Runtime"] == "python3.11"
        assert result["Handler"] == "prediction_handler.lambda_handler"
        assert result["State"] == "Active"

        # Verify create_function was called with correct parameters
        mock_lambda_client.create_function.assert_called_once()
        create_args = mock_lambda_client.create_function.call_args[1]
        assert create_args["FunctionName"] == "ml-prediction-function"
        assert create_args["Runtime"] == "python3.11"
        assert create_args["Handler"] == "prediction_handler.lambda_handler"
        assert create_args["Role"] == "arn:aws:iam::123456789012:role/lambda-execution"
        assert create_args["MemorySize"] == 2048
        assert create_args["Timeout"] == 300
        assert create_args["Layers"] == [layer_arn]

    def test_deploy_function_update_existing(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test updating an existing Lambda function."""
        # Arrange
        zip_file_path = "/path/to/function.zip"

        # Mock get_function to return existing function (function exists)
        mock_get_response = {
            "Configuration": {
                "FunctionName": "ml-prediction-function",
                "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:ml-prediction-function"
            }
        }
        mock_lambda_client.get_function.return_value = mock_get_response

        # Mock successful update_function_code
        mock_update_response = {
            "FunctionName": "ml-prediction-function",
            "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:ml-prediction-function",
            "Runtime": "python3.11",
            "Handler": "prediction_handler.lambda_handler",
            "CodeSize": 2048,
            "State": "Active",
            "LastUpdateStatus": "Successful"
        }
        mock_lambda_client.update_function_code.return_value = mock_update_response

        # Act
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_zip_data")):
            result = lambda_manager.deploy_function(zip_file_path=zip_file_path)

        # Assert
        assert result["FunctionName"] == "ml-prediction-function"
        assert result["State"] == "Active"

        # Verify update_function_code was called
        mock_lambda_client.update_function_code.assert_called_once()
        update_args = mock_lambda_client.update_function_code.call_args[1]
        assert update_args["FunctionName"] == "ml-prediction-function"
        assert "ZipFile" in update_args

        # Verify create_function was NOT called
        mock_lambda_client.create_function.assert_not_called()

    def test_deploy_function_without_layer(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test function deployment without Lambda layer."""
        # Arrange
        zip_file_path = "/path/to/function.zip"

        # Function doesn't exist
        error_response = {
            "Error": {"Code": "ResourceNotFoundException"},
            "ResponseMetadata": {"RequestId": "test-req-2"}
        }
        mock_lambda_client.get_function.side_effect = ClientError(
            error_response,
            "GetFunction"
        )

        mock_create_response = {
            "FunctionName": "ml-prediction-function",
            "State": "Active"
        }
        mock_lambda_client.create_function.return_value = mock_create_response

        # Act
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_zip_data")):
            result = lambda_manager.deploy_function(zip_file_path=zip_file_path)

        # Assert
        assert result["FunctionName"] == "ml-prediction-function"

        # Verify Layers parameter is not included or is empty
        create_args = mock_lambda_client.create_function.call_args[1]
        assert "Layers" not in create_args or create_args["Layers"] == []

    def test_deploy_function_file_not_found(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test deployment with non-existent zip file."""
        # Arrange
        zip_file_path = "/nonexistent/path/function.zip"

        # Act & Assert
        with pytest.raises(LambdaError) as exc_info:
            lambda_manager.deploy_function(zip_file_path=zip_file_path)

        assert "Zip file not found" in str(exc_info.value) or "Failed to deploy" in str(exc_info.value)

    def test_deploy_function_boto3_error(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test handling of boto3 error during deployment."""
        # Arrange
        zip_file_path = "/path/to/function.zip"

        # Function doesn't exist
        error_response_not_found = {
            "Error": {"Code": "ResourceNotFoundException"},
            "ResponseMetadata": {"RequestId": "test-req-3"}
        }
        mock_lambda_client.get_function.side_effect = ClientError(
            error_response_not_found,
            "GetFunction"
        )

        # create_function fails
        error_response_create = {
            "Error": {
                "Code": "InvalidParameterValueException",
                "Message": "Invalid IAM role"
            },
            "ResponseMetadata": {"RequestId": "test-req-4"}
        }
        mock_lambda_client.create_function.side_effect = ClientError(
            error_response_create,
            "CreateFunction"
        )

        # Act & Assert
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_zip_data")):
            with pytest.raises(LambdaError) as exc_info:
                lambda_manager.deploy_function(zip_file_path=zip_file_path)

        assert "Failed to deploy Lambda function" in str(exc_info.value)
        assert exc_info.value.error_code == "InvalidParameterValueException"


class TestCreateLayer:
    """Test Lambda layer creation."""

    @pytest.fixture
    def mock_lambda_client(self) -> Mock:
        """Create mock Lambda client."""
        return Mock()

    @pytest.fixture
    def lambda_manager(self, mock_lambda_client: Mock) -> LambdaManager:
        """Create LambdaManager with mocked dependencies."""
        mock_aws_client = Mock(spec=AWSClient)
        mock_aws_client.get_lambda_client.return_value = mock_lambda_client

        mock_config = Mock(spec=CloudConfig)
        mock_config.lambda_function_name = "ml-prediction-function"
        mock_config.lambda_memory_mb = 2048
        mock_config.lambda_timeout_seconds = 300
        mock_config.iam_role_arn = "arn:aws:iam::123456789012:role/lambda-execution"

        return LambdaManager(aws_client=mock_aws_client, config=mock_config)

    def test_create_layer_success(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test successful Lambda layer creation."""
        # Arrange
        layer_name = "ml-dependencies"
        zip_file_path = "/path/to/layer.zip"
        description = "ML dependencies including scikit-learn and pandas"

        mock_response = {
            "LayerArn": "arn:aws:lambda:us-east-1:123456789012:layer:ml-dependencies",
            "LayerVersionArn": "arn:aws:lambda:us-east-1:123456789012:layer:ml-dependencies:1",
            "Version": 1,
            "CompatibleRuntimes": ["python3.11"],
            "CreatedDate": "2025-10-24T12:00:00.000+0000"
        }
        mock_lambda_client.publish_layer_version.return_value = mock_response

        # Act
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_layer_zip")):
            layer_arn = lambda_manager.create_layer(
                layer_name=layer_name,
                zip_file_path=zip_file_path,
                description=description
            )

        # Assert
        assert layer_arn == "arn:aws:lambda:us-east-1:123456789012:layer:ml-dependencies:1"

        # Verify publish_layer_version was called correctly
        mock_lambda_client.publish_layer_version.assert_called_once()
        call_args = mock_lambda_client.publish_layer_version.call_args[1]
        assert call_args["LayerName"] == layer_name
        assert call_args["Description"] == description
        assert call_args["CompatibleRuntimes"] == ["python3.11"]
        assert "Content" in call_args
        assert "ZipFile" in call_args["Content"]

    def test_create_layer_default_description(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test layer creation with default description."""
        # Arrange
        layer_name = "ml-deps"
        zip_file_path = "/path/to/layer.zip"

        mock_response = {
            "LayerVersionArn": "arn:aws:lambda:us-east-1:123456789012:layer:ml-deps:1"
        }
        mock_lambda_client.publish_layer_version.return_value = mock_response

        # Act
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_layer_zip")):
            layer_arn = lambda_manager.create_layer(
                layer_name=layer_name,
                zip_file_path=zip_file_path
            )

        # Assert
        assert "ml-deps" in layer_arn

        # Verify default description was used
        call_args = mock_lambda_client.publish_layer_version.call_args[1]
        assert call_args["Description"] == "ML dependencies layer"

    def test_create_layer_file_not_found(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test layer creation with non-existent zip file."""
        # Arrange
        layer_name = "ml-deps"
        zip_file_path = "/nonexistent/layer.zip"

        # Act & Assert
        with pytest.raises(LambdaError) as exc_info:
            lambda_manager.create_layer(
                layer_name=layer_name,
                zip_file_path=zip_file_path
            )

        assert "Zip file not found" in str(exc_info.value) or "Failed to create" in str(exc_info.value)

    def test_create_layer_boto3_error(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test handling of boto3 error during layer creation."""
        # Arrange
        layer_name = "ml-deps"
        zip_file_path = "/path/to/layer.zip"

        error_response = {
            "Error": {
                "Code": "CodeStorageExceededException",
                "Message": "Code storage limit exceeded"
            },
            "ResponseMetadata": {"RequestId": "test-req-5"}
        }
        mock_lambda_client.publish_layer_version.side_effect = ClientError(
            error_response,
            "PublishLayerVersion"
        )

        # Act & Assert
        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_layer_zip")):
            with pytest.raises(LambdaError) as exc_info:
                lambda_manager.create_layer(
                    layer_name=layer_name,
                    zip_file_path=zip_file_path
                )

        assert "Failed to create Lambda layer" in str(exc_info.value)
        assert exc_info.value.error_code == "CodeStorageExceededException"


class TestLambdaManagerIntegration:
    """Integration tests for LambdaManager workflows."""

    @pytest.fixture
    def mock_lambda_client(self) -> Mock:
        """Create mock Lambda client."""
        return Mock()

    @pytest.fixture
    def lambda_manager(self, mock_lambda_client: Mock) -> LambdaManager:
        """Create LambdaManager with mocked dependencies."""
        mock_aws_client = Mock(spec=AWSClient)
        mock_aws_client.get_lambda_client.return_value = mock_lambda_client

        mock_config = Mock(spec=CloudConfig)
        mock_config.lambda_function_name = "ml-prediction-function"
        mock_config.lambda_memory_mb = 2048
        mock_config.lambda_timeout_seconds = 300
        mock_config.iam_role_arn = "arn:aws:iam::123456789012:role/lambda-execution"

        return LambdaManager(aws_client=mock_aws_client, config=mock_config)

    def test_full_deployment_and_invocation_workflow(
        self,
        lambda_manager: LambdaManager,
        mock_lambda_client: Mock
    ) -> None:
        """Test complete workflow: create layer -> deploy function -> invoke prediction."""
        # Step 1: Create layer
        mock_layer_response = {
            "LayerVersionArn": "arn:aws:lambda:us-east-1:123456789012:layer:ml-deps:1"
        }
        mock_lambda_client.publish_layer_version.return_value = mock_layer_response

        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_layer")):
            layer_arn = lambda_manager.create_layer(
                layer_name="ml-deps",
                zip_file_path="/path/to/layer.zip"
            )

        assert "ml-deps" in layer_arn

        # Step 2: Deploy function with layer
        error_response = {
            "Error": {"Code": "ResourceNotFoundException"},
            "ResponseMetadata": {"RequestId": "test-req-6"}
        }
        mock_lambda_client.get_function.side_effect = ClientError(
            error_response,
            "GetFunction"
        )

        mock_deploy_response = {
            "FunctionName": "ml-prediction-function",
            "State": "Active"
        }
        mock_lambda_client.create_function.return_value = mock_deploy_response

        with patch("builtins.open", return_value=Mock(__enter__=lambda s: s, __exit__=lambda *args: None, read=lambda: b"mock_function")):
            deploy_result = lambda_manager.deploy_function(
                zip_file_path="/path/to/function.zip",
                layer_arn=layer_arn
            )

        assert deploy_result["State"] == "Active"

        # Step 3: Invoke prediction
        mock_invoke_response = {
            "StatusCode": 200,
            "Payload": Mock(read=lambda: json.dumps({
                "statusCode": 200,
                "body": {"num_predictions": 50}
            }).encode())
        }
        mock_lambda_client.invoke.return_value = mock_invoke_response

        result = lambda_manager.invoke_prediction(
            model_s3_uri="s3://models/model.pkl",
            data_s3_uri="s3://data/input.csv",
            output_s3_uri="s3://results/output.csv",
            prediction_column_name="prediction"
        )

        assert result["statusCode"] == 200
        assert result["body"]["num_predictions"] == 50
