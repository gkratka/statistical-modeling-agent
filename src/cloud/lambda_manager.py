"""
AWS Lambda Manager for ML Prediction Orchestration.

This module provides Lambda function management and invocation capabilities
for serverless ML predictions, including synchronous/asynchronous invocations,
function deployment, and Lambda layer creation.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.4-4.5: Lambda Manager Implementation)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import LambdaError
from src.cloud.provider_interface import CloudPredictionProvider


class LambdaManager(CloudPredictionProvider):
    """
    AWS Lambda manager for serverless ML prediction orchestration.

    This class provides methods for:
    - Synchronous and asynchronous Lambda function invocation
    - Lambda function deployment (create/update)
    - Lambda layer creation for ML dependencies

    Attributes:
        _aws_client: AWSClient instance for AWS service access
        _config: CloudConfig with Lambda configuration
        _lambda_client: boto3 Lambda client
    """

    def __init__(self, aws_client: AWSClient, config: CloudConfig) -> None:
        """
        Initialize LambdaManager with AWS client and configuration.

        Args:
            aws_client: AWSClient instance for AWS service access
            config: CloudConfig with Lambda function settings

        Raises:
            LambdaError: If Lambda client initialization fails
        """
        self._aws_client = aws_client
        self._config = config
        self._lambda_client = aws_client.get_lambda_client()

    def invoke_prediction(
        self,
        model_s3_uri: str,
        data_s3_uri: str,
        output_s3_uri: str,
        prediction_column_name: str,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Invoke Lambda function synchronously for ML prediction.

        This method performs a synchronous (RequestResponse) invocation of the
        Lambda function, waiting for the prediction to complete and returning
        the result immediately.

        Args:
            model_s3_uri: S3 URI of the trained model (e.g., s3://bucket/model.pkl)
            data_s3_uri: S3 URI of input data for prediction
            output_s3_uri: S3 URI where predictions will be written
            prediction_column_name: Name for the prediction output column
            feature_columns: Optional list of feature column names to use

        Returns:
            dict: Lambda response body containing:
                - statusCode: HTTP status code (200 for success)
                - body: Prediction result details including:
                    - output_s3_uri: Location of prediction results
                    - num_predictions: Number of predictions made
                    - execution_time_seconds: Time taken for prediction

        Raises:
            LambdaError: If invocation fails or Lambda returns error status
        """
        # Construct payload
        payload: Dict[str, Any] = {
            "model_s3_uri": model_s3_uri,
            "data_s3_uri": data_s3_uri,
            "output_s3_uri": output_s3_uri,
            "prediction_column_name": prediction_column_name
        }

        # Add feature columns if provided
        if feature_columns is not None:
            payload["feature_columns"] = feature_columns

        try:
            # Invoke Lambda function synchronously
            response = self._lambda_client.invoke(
                FunctionName=self._config.lambda_function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload)
            )

            # Parse response payload
            response_payload = json.loads(response["Payload"].read())

            # Check Lambda execution status
            if response_payload.get("statusCode") != 200:
                raise LambdaError(
                    message=f"Lambda function returned error status: {response_payload.get('statusCode')}",
                    function_name=self._config.lambda_function_name,
                    invocation_type="RequestResponse",
                    error_code=f"StatusCode{response_payload.get('statusCode')}"
                )

            return response_payload

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            raise LambdaError(
                message=f"Failed to invoke Lambda function: {error_message}",
                function_name=self._config.lambda_function_name,
                invocation_type="RequestResponse",
                error_code=error_code,
                request_id=request_id
            )

        except LambdaError:
            # Re-raise LambdaError without wrapping
            raise

        except Exception as e:
            raise LambdaError(
                message=f"Unexpected error invoking Lambda function: {str(e)}",
                function_name=self._config.lambda_function_name,
                invocation_type="RequestResponse"
            )

    def invoke_async(
        self,
        model_s3_uri: str,
        data_s3_uri: str,
        output_s3_uri: str,
        prediction_column_name: str,
        feature_columns: Optional[List[str]] = None
    ) -> str:
        """
        Invoke Lambda function asynchronously for ML prediction.

        This method performs an asynchronous (Event) invocation, returning
        immediately with a request ID. The prediction runs in the background
        and results are written to S3 when complete.

        Args:
            model_s3_uri: S3 URI of the trained model
            data_s3_uri: S3 URI of input data for prediction
            output_s3_uri: S3 URI where predictions will be written
            prediction_column_name: Name for the prediction output column
            feature_columns: Optional list of feature column names to use

        Returns:
            str: AWS request ID for tracking the async invocation

        Raises:
            LambdaError: If async invocation fails
        """
        # Construct payload (same as synchronous)
        payload: Dict[str, Any] = {
            "model_s3_uri": model_s3_uri,
            "data_s3_uri": data_s3_uri,
            "output_s3_uri": output_s3_uri,
            "prediction_column_name": prediction_column_name
        }

        # Add feature columns if provided
        if feature_columns is not None:
            payload["feature_columns"] = feature_columns

        try:
            # Invoke Lambda function asynchronously
            response = self._lambda_client.invoke(
                FunctionName=self._config.lambda_function_name,
                InvocationType="Event",
                Payload=json.dumps(payload)
            )

            # Return request ID for tracking
            request_id = response["ResponseMetadata"]["RequestId"]
            return request_id

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            raise LambdaError(
                message=f"Failed to invoke Lambda function asynchronously: {error_message}",
                function_name=self._config.lambda_function_name,
                invocation_type="Event",
                error_code=error_code,
                request_id=request_id
            )

        except Exception as e:
            raise LambdaError(
                message=f"Unexpected error invoking Lambda function asynchronously: {str(e)}",
                function_name=self._config.lambda_function_name,
                invocation_type="Event"
            )

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of async prediction job (CloudPredictionProvider interface).

        Note: AWS Lambda async invocations (Event type) don't provide built-in
        status tracking via request ID. This is a limitation - in production,
        you would need to implement custom status tracking via DynamoDB or
        check S3 for output file existence.

        Args:
            job_id: Prediction job identifier (AWS request ID)

        Returns:
            dict: Status dict with job_id, status, progress, result
                Status will be "unknown" since Lambda Event invocations
                don't provide status tracking by default.
        """
        # Lambda async invocations don't provide status tracking
        # This would require custom implementation with DynamoDB or S3 polling
        return {
            "job_id": job_id,
            "status": "unknown",
            "progress": None,
            "result": "Lambda async invocations don't provide built-in status tracking. "
                     "Check S3 output location for results."
        }

    def deploy_function(
        self,
        zip_file_path: str,
        layer_arn: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy or update Lambda function code.

        This method checks if the function exists and either creates a new
        function or updates the code of an existing function.

        Args:
            zip_file_path: Path to deployment package (.zip file)
            layer_arn: Optional Lambda layer ARN for dependencies

        Returns:
            dict: Lambda function configuration with fields:
                - FunctionName: Name of the Lambda function
                - FunctionArn: ARN of the Lambda function
                - Runtime: Runtime environment (python3.11)
                - Handler: Function entry point
                - CodeSize: Size of deployment package in bytes
                - MemorySize: Memory allocation in MB
                - Timeout: Timeout in seconds
                - State: Function state (Active, Pending, etc.)
                - LastUpdateStatus: Status of last update

        Raises:
            LambdaError: If deployment fails or zip file not found
        """
        # Read zip file
        try:
            with open(zip_file_path, "rb") as f:
                zip_content = f.read()
        except FileNotFoundError:
            raise LambdaError(
                message=f"Zip file not found: {zip_file_path}",
                function_name=self._config.lambda_function_name
            )
        except Exception as e:
            raise LambdaError(
                message=f"Failed to read zip file: {str(e)}",
                function_name=self._config.lambda_function_name
            )

        # Check if function exists
        function_exists = self._check_function_exists()

        try:
            if function_exists:
                # Update existing function code
                response = self._lambda_client.update_function_code(
                    FunctionName=self._config.lambda_function_name,
                    ZipFile=zip_content
                )
            else:
                # Create new function
                create_params: Dict[str, Any] = {
                    "FunctionName": self._config.lambda_function_name,
                    "Runtime": "python3.11",
                    "Role": self._config.iam_role_arn,
                    "Handler": "prediction_handler.lambda_handler",
                    "Code": {"ZipFile": zip_content},
                    "MemorySize": self._config.lambda_memory_mb,
                    "Timeout": self._config.lambda_timeout_seconds,
                    "Description": "ML prediction function for statistical modeling agent"
                }

                # Add layer if provided
                if layer_arn:
                    create_params["Layers"] = [layer_arn]

                response = self._lambda_client.create_function(**create_params)

            return response

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            raise LambdaError(
                message=f"Failed to deploy Lambda function: {error_message}",
                function_name=self._config.lambda_function_name,
                error_code=error_code,
                request_id=request_id
            )

        except Exception as e:
            raise LambdaError(
                message=f"Unexpected error deploying Lambda function: {str(e)}",
                function_name=self._config.lambda_function_name
            )

    def create_layer(
        self,
        layer_name: str,
        zip_file_path: str,
        description: str = "ML dependencies layer"
    ) -> str:
        """
        Create a new Lambda layer version for ML dependencies.

        Lambda layers allow you to package dependencies separately from
        function code, enabling code reuse and reducing deployment size.

        Args:
            layer_name: Name for the Lambda layer
            zip_file_path: Path to layer package (.zip file)
            description: Optional description for the layer

        Returns:
            str: Layer version ARN (e.g., arn:aws:lambda:region:account:layer:name:1)

        Raises:
            LambdaError: If layer creation fails or zip file not found
        """
        # Read zip file
        try:
            with open(zip_file_path, "rb") as f:
                zip_content = f.read()
        except FileNotFoundError:
            raise LambdaError(
                message=f"Zip file not found: {zip_file_path}",
                function_name=layer_name
            )
        except Exception as e:
            raise LambdaError(
                message=f"Failed to read zip file: {str(e)}",
                function_name=layer_name
            )

        try:
            # Publish layer version
            response = self._lambda_client.publish_layer_version(
                LayerName=layer_name,
                Description=description,
                Content={"ZipFile": zip_content},
                CompatibleRuntimes=["python3.11"]
            )

            # Return layer version ARN
            return response["LayerVersionArn"]

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            raise LambdaError(
                message=f"Failed to create Lambda layer: {error_message}",
                function_name=layer_name,
                error_code=error_code,
                request_id=request_id
            )

        except Exception as e:
            raise LambdaError(
                message=f"Unexpected error creating Lambda layer: {str(e)}",
                function_name=layer_name
            )

    def _check_function_exists(self) -> bool:
        """
        Check if Lambda function exists.

        Returns:
            bool: True if function exists, False otherwise
        """
        try:
            self._lambda_client.get_function(
                FunctionName=self._config.lambda_function_name
            )
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                return False
            # Re-raise other errors
            raise
