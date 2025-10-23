"""
AWS Client Wrapper for boto3 service clients.

This module provides a unified wrapper around boto3 clients for S3, EC2, and Lambda
services with health checking and error handling capabilities.

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 1.5: AWS Client Wrapper with TDD)
"""

from typing import Any, Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import AWSError


class AWSClient:
    """
    AWS client wrapper providing unified access to S3, EC2, and Lambda services.

    This class initializes and manages boto3 clients for AWS services with
    credential management, health checking, and error handling.
    """

    def __init__(self, config: CloudConfig) -> None:
        """
        Initialize AWS clients for S3, EC2, and Lambda.

        Args:
            config: CloudConfig instance with AWS credentials and settings

        Raises:
            AWSError: If AWS client initialization fails
        """
        self._config = config

        try:
            # Initialize S3 client
            self._s3_client = boto3.client(
                "s3",
                region_name=config.aws_region,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )

            # Initialize EC2 client
            self._ec2_client = boto3.client(
                "ec2",
                region_name=config.aws_region,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )

            # Initialize Lambda client
            self._lambda_client = boto3.client(
                "lambda",
                region_name=config.aws_region,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )

            # Initialize CloudWatch Logs client
            self._logs_client = boto3.client(
                "logs",
                region_name=config.aws_region,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )

        except BotoCoreError as e:
            raise AWSError(
                message=f"Failed to initialize AWS clients: {str(e)}",
                service="aws",
                error_code="ClientInitializationError"
            )
        except Exception as e:
            raise AWSError(
                message=f"Unexpected error during AWS client initialization: {str(e)}",
                service="aws",
                error_code="UnexpectedError"
            )

    def get_s3_client(self) -> Any:
        """
        Get boto3 S3 client.

        Returns:
            boto3 S3 client instance
        """
        return self._s3_client

    def get_ec2_client(self) -> Any:
        """
        Get boto3 EC2 client.

        Returns:
            boto3 EC2 client instance
        """
        return self._ec2_client

    def get_lambda_client(self) -> Any:
        """
        Get boto3 Lambda client.

        Returns:
            boto3 Lambda client instance
        """
        return self._lambda_client

    def get_logs_client(self) -> Any:
        """
        Get boto3 CloudWatch Logs client.

        Returns:
            boto3 CloudWatch Logs client instance
        """
        return self._logs_client

    def health_check(self) -> Dict[str, Any]:
        """
        Verify AWS connectivity and permissions for all services.

        Performs lightweight API calls to verify:
        - S3: List buckets (verify read access)
        - EC2: Describe regions (verify basic access)
        - Lambda: List functions (verify Lambda access)

        Returns:
            dict: Health status for each service with structure:
                {
                    "s3": {
                        "status": "healthy" | "unhealthy",
                        "error_code": str (if unhealthy),
                        "error_message": str (if unhealthy),
                        "request_id": str (if unhealthy)
                    },
                    "ec2": { ... },
                    "lambda": { ... },
                    "overall_status": "healthy" | "unhealthy"
                }
        """
        results: Dict[str, Any] = {}

        # Check S3 health
        results["s3"] = self._check_s3_health()

        # Check EC2 health
        results["ec2"] = self._check_ec2_health()

        # Check Lambda health
        results["lambda"] = self._check_lambda_health()

        # Determine overall status
        all_healthy = all(
            service_result.get("status") == "healthy"
            for service_result in results.values()
        )
        results["overall_status"] = "healthy" if all_healthy else "unhealthy"

        return results

    def _check_s3_health(self) -> Dict[str, Any]:
        """
        Check S3 service health by listing buckets.

        Returns:
            dict: S3 health status with error details if unhealthy
        """
        try:
            # Lightweight operation to verify S3 access
            self._s3_client.list_buckets()
            return {"status": "healthy"}

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            return {
                "status": "unhealthy",
                "error_code": error_code,
                "error_message": error_message,
                "request_id": request_id
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error_message": str(e)
            }

    def _check_ec2_health(self) -> Dict[str, Any]:
        """
        Check EC2 service health by describing regions.

        Returns:
            dict: EC2 health status with error details if unhealthy
        """
        try:
            # Lightweight operation to verify EC2 access
            self._ec2_client.describe_regions()
            return {"status": "healthy"}

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            return {
                "status": "unhealthy",
                "error_code": error_code,
                "error_message": error_message,
                "request_id": request_id
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error_message": str(e)
            }

    def _check_lambda_health(self) -> Dict[str, Any]:
        """
        Check Lambda service health by listing functions.

        Returns:
            dict: Lambda health status with error details if unhealthy
        """
        try:
            # Lightweight operation to verify Lambda access
            self._lambda_client.list_functions(MaxItems=1)
            return {"status": "healthy"}

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            return {
                "status": "unhealthy",
                "error_code": error_code,
                "error_message": error_message,
                "request_id": request_id
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error_message": str(e)
            }
