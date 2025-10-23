"""
Unit tests for AWS client wrapper.

Tests boto3 client initialization, health checks, and error handling
using mocked AWS services.

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 1.5: AWS Client Wrapper with TDD)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import AWSError


@pytest.fixture
def mock_config():
    """Create mock CloudConfig for testing."""
    return CloudConfig(
        # AWS credentials
        aws_region="us-east-1",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        # S3 configuration
        s3_bucket="test-bucket",
        s3_data_prefix="data/",
        s3_models_prefix="models/",
        s3_results_prefix="results/",
        # EC2 configuration
        ec2_instance_type="t3.micro",
        ec2_ami_id="ami-12345678",
        ec2_key_name="test-key",
        ec2_security_group="sg-12345678",
        # Lambda configuration
        lambda_function_name="test-function",
        lambda_memory_mb=512,
        lambda_timeout_seconds=300,
        # Cost limits
        max_training_cost_dollars=10.0,
        max_prediction_cost_dollars=1.0,
        cost_warning_threshold=0.8
    )


class TestAWSClientInitialization:
    """Test AWS client initialization."""

    @patch("boto3.client")
    def test_successful_initialization(self, mock_boto_client, mock_config):
        """Test successful AWS client initialization."""
        # Create mock clients
        mock_s3 = Mock()
        mock_ec2 = Mock()
        mock_lambda = Mock()

        # Configure boto3.client to return appropriate mocks
        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            elif service_name == "ec2":
                return mock_ec2
            elif service_name == "lambda":
                return mock_lambda
            raise ValueError(f"Unexpected service: {service_name}")

        mock_boto_client.side_effect = client_factory

        # Initialize client
        aws_client = AWSClient(mock_config)

        # Verify boto3.client was called for each service
        assert mock_boto_client.call_count == 3

        # Verify clients are accessible
        assert aws_client.get_s3_client() == mock_s3
        assert aws_client.get_ec2_client() == mock_ec2
        assert aws_client.get_lambda_client() == mock_lambda

    @patch("boto3.client")
    def test_initialization_with_credentials(self, mock_boto_client, mock_config):
        """Test that AWS credentials are passed to boto3."""
        mock_boto_client.return_value = Mock()

        # Initialize client
        AWSClient(mock_config)

        # Verify credentials were passed
        for call in mock_boto_client.call_args_list:
            args, kwargs = call
            assert kwargs["region_name"] == "us-east-1"
            assert kwargs["aws_access_key_id"] == "AKIAIOSFODNN7EXAMPLE"
            assert kwargs["aws_secret_access_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

    @patch("boto3.client")
    def test_initialization_boto3_error(self, mock_boto_client, mock_config):
        """Test initialization handles boto3 errors."""
        # Simulate boto3 client creation failure
        mock_boto_client.side_effect = BotoCoreError()

        # Should raise AWSError
        with pytest.raises(AWSError) as exc_info:
            AWSClient(mock_config)

        assert "Failed to initialize AWS clients" in str(exc_info.value)


class TestHealthCheck:
    """Test AWS health check functionality."""

    @patch("boto3.client")
    def test_health_check_all_services_healthy(self, mock_boto_client, mock_config):
        """Test health check when all services are accessible."""
        # Create mock clients with successful responses
        mock_s3 = Mock()
        mock_s3.list_buckets.return_value = {"Buckets": [{"Name": "test-bucket"}]}

        mock_ec2 = Mock()
        mock_ec2.describe_regions.return_value = {
            "Regions": [{"RegionName": "us-east-1"}]
        }

        mock_lambda = Mock()
        mock_lambda.list_functions.return_value = {
            "Functions": [{"FunctionName": "test-function"}]
        }

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            elif service_name == "ec2":
                return mock_ec2
            elif service_name == "lambda":
                return mock_lambda

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # Verify all services report healthy
        assert result["s3"]["status"] == "healthy"
        assert result["ec2"]["status"] == "healthy"
        assert result["lambda"]["status"] == "healthy"
        assert result["overall_status"] == "healthy"

    @patch("boto3.client")
    def test_health_check_s3_access_denied(self, mock_boto_client, mock_config):
        """Test health check when S3 access is denied."""
        # Create mock clients
        mock_s3 = Mock()
        mock_s3.list_buckets.side_effect = ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "Access denied to S3"
                },
                "ResponseMetadata": {"RequestId": "req-123"}
            },
            "ListBuckets"
        )

        mock_ec2 = Mock()
        mock_ec2.describe_regions.return_value = {"Regions": []}

        mock_lambda = Mock()
        mock_lambda.list_functions.return_value = {"Functions": []}

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            elif service_name == "ec2":
                return mock_ec2
            elif service_name == "lambda":
                return mock_lambda

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # Verify S3 reports unhealthy
        assert result["s3"]["status"] == "unhealthy"
        assert "AccessDenied" in result["s3"]["error_code"]
        assert "req-123" in result["s3"]["request_id"]

        # Other services should still be healthy
        assert result["ec2"]["status"] == "healthy"
        assert result["lambda"]["status"] == "healthy"

        # Overall status should be unhealthy
        assert result["overall_status"] == "unhealthy"

    @patch("boto3.client")
    def test_health_check_ec2_network_error(self, mock_boto_client, mock_config):
        """Test health check when EC2 has network issues."""
        mock_s3 = Mock()
        mock_s3.list_buckets.return_value = {"Buckets": []}

        mock_ec2 = Mock()
        mock_ec2.describe_regions.side_effect = ClientError(
            {
                "Error": {
                    "Code": "RequestTimeout",
                    "Message": "Network timeout"
                },
                "ResponseMetadata": {"RequestId": "req-456"}
            },
            "DescribeRegions"
        )

        mock_lambda = Mock()
        mock_lambda.list_functions.return_value = {"Functions": []}

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            elif service_name == "ec2":
                return mock_ec2
            elif service_name == "lambda":
                return mock_lambda

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # Verify EC2 reports unhealthy
        assert result["ec2"]["status"] == "unhealthy"
        assert "RequestTimeout" in result["ec2"]["error_code"]
        assert result["overall_status"] == "unhealthy"

    @patch("boto3.client")
    def test_health_check_lambda_invalid_permissions(self, mock_boto_client, mock_config):
        """Test health check when Lambda permissions are invalid."""
        mock_s3 = Mock()
        mock_s3.list_buckets.return_value = {"Buckets": []}

        mock_ec2 = Mock()
        mock_ec2.describe_regions.return_value = {"Regions": []}

        mock_lambda = Mock()
        mock_lambda.list_functions.side_effect = ClientError(
            {
                "Error": {
                    "Code": "UnauthorizedException",
                    "Message": "Invalid permissions"
                },
                "ResponseMetadata": {"RequestId": "req-789"}
            },
            "ListFunctions"
        )

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            elif service_name == "ec2":
                return mock_ec2
            elif service_name == "lambda":
                return mock_lambda

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # Verify Lambda reports unhealthy
        assert result["lambda"]["status"] == "unhealthy"
        assert "UnauthorizedException" in result["lambda"]["error_code"]
        assert result["overall_status"] == "unhealthy"

    @patch("boto3.client")
    def test_health_check_all_services_failed(self, mock_boto_client, mock_config):
        """Test health check when all services fail."""
        # Create mock clients that all fail
        error = ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "Service down"}},
            "Operation"
        )

        mock_s3 = Mock()
        mock_s3.list_buckets.side_effect = error

        mock_ec2 = Mock()
        mock_ec2.describe_regions.side_effect = error

        mock_lambda = Mock()
        mock_lambda.list_functions.side_effect = error

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            elif service_name == "ec2":
                return mock_ec2
            elif service_name == "lambda":
                return mock_lambda

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # All services should be unhealthy
        assert result["s3"]["status"] == "unhealthy"
        assert result["ec2"]["status"] == "unhealthy"
        assert result["lambda"]["status"] == "unhealthy"
        assert result["overall_status"] == "unhealthy"


class TestClientGetters:
    """Test client getter methods."""

    @patch("boto3.client")
    def test_get_s3_client(self, mock_boto_client, mock_config):
        """Test getting S3 client."""
        mock_s3 = Mock()

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            return Mock()

        mock_boto_client.side_effect = client_factory

        aws_client = AWSClient(mock_config)
        client = aws_client.get_s3_client()

        assert client == mock_s3

    @patch("boto3.client")
    def test_get_ec2_client(self, mock_boto_client, mock_config):
        """Test getting EC2 client."""
        mock_ec2 = Mock()

        def client_factory(service_name, **kwargs):
            if service_name == "ec2":
                return mock_ec2
            return Mock()

        mock_boto_client.side_effect = client_factory

        aws_client = AWSClient(mock_config)
        client = aws_client.get_ec2_client()

        assert client == mock_ec2

    @patch("boto3.client")
    def test_get_lambda_client(self, mock_boto_client, mock_config):
        """Test getting Lambda client."""
        mock_lambda = Mock()

        def client_factory(service_name, **kwargs):
            if service_name == "lambda":
                return mock_lambda
            return Mock()

        mock_boto_client.side_effect = client_factory

        aws_client = AWSClient(mock_config)
        client = aws_client.get_lambda_client()

        assert client == mock_lambda


class TestErrorHandling:
    """Test error handling in AWS client."""

    @patch("boto3.client")
    def test_client_error_with_all_context(self, mock_boto_client, mock_config):
        """Test error handling includes all AWS context."""
        mock_s3 = Mock()
        mock_s3.list_buckets.side_effect = ClientError(
            {
                "Error": {
                    "Code": "InvalidAccessKeyId",
                    "Message": "The AWS Access Key Id you provided does not exist"
                },
                "ResponseMetadata": {"RequestId": "abcd-1234-efgh-5678"}
            },
            "ListBuckets"
        )

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            return Mock()

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # Verify error context is captured
        s3_result = result["s3"]
        assert s3_result["status"] == "unhealthy"
        assert s3_result["error_code"] == "InvalidAccessKeyId"
        assert s3_result["request_id"] == "abcd-1234-efgh-5678"
        assert "AWS Access Key" in s3_result["error_message"]

    @patch("boto3.client")
    def test_non_client_error_handling(self, mock_boto_client, mock_config):
        """Test handling of non-ClientError exceptions."""
        mock_s3 = Mock()
        mock_s3.list_buckets.side_effect = BotoCoreError()

        def client_factory(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            return Mock()

        mock_boto_client.side_effect = client_factory

        # Initialize and run health check
        aws_client = AWSClient(mock_config)
        result = aws_client.health_check()

        # Should handle gracefully
        assert result["s3"]["status"] == "unhealthy"
        assert "error_message" in result["s3"]
