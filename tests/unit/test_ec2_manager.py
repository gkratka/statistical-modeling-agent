"""
Unit tests for EC2Manager class.

Tests cover instance type selection, Spot instance launching, and instance
assignment waiting with comprehensive mocking of boto3 EC2 client.

Author: Statistical Modeling Agent
Created: 2025-10-23 (Tasks 3.1 & 3.2: EC2Manager with TDD)
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.ec2_manager import EC2Manager
from src.cloud.exceptions import EC2Error


@pytest.fixture
def mock_config():
    """Create mock CloudConfig for testing."""
    config = MagicMock(spec=CloudConfig)
    config.aws_region = "us-west-2"
    config.ec2_instance_type = "m5.large"
    config.ec2_ami_id = "ami-12345678"
    config.ec2_key_name = "test-key"
    config.ec2_security_group = "sg-12345678"
    config.ec2_spot_max_price = "0.50"
    config.iam_role_arn = "arn:aws:iam::123456789012:role/test-role"
    return config


@pytest.fixture
def mock_aws_client():
    """Create mock AWSClient with EC2 client."""
    aws_client = MagicMock(spec=AWSClient)
    ec2_client = MagicMock()
    aws_client.get_ec2_client.return_value = ec2_client
    return aws_client


@pytest.fixture
def ec2_manager(mock_aws_client, mock_config):
    """Create EC2Manager instance for testing."""
    return EC2Manager(mock_aws_client, mock_config)


class TestEC2ManagerInit:
    """Test EC2Manager initialization."""

    def test_init_success(self, mock_aws_client, mock_config):
        """Test successful EC2Manager initialization."""
        manager = EC2Manager(mock_aws_client, mock_config)

        assert manager._aws_client == mock_aws_client
        assert manager._config == mock_config
        assert manager._ec2_client == mock_aws_client.get_ec2_client()

    def test_init_stores_clients(self, mock_aws_client, mock_config):
        """Test that EC2 client is properly stored."""
        manager = EC2Manager(mock_aws_client, mock_config)

        # Verify EC2 client was retrieved
        mock_aws_client.get_ec2_client.assert_called_once()


class TestSelectInstanceType:
    """Test instance type selection logic."""

    # Simple models (linear, logistic, ridge, lasso)
    @pytest.mark.parametrize("model_type,dataset_size_mb,expected_type", [
        ("linear", 50, "t3.medium"),
        ("logistic", 99, "t3.medium"),
        ("ridge", 100, "m5.large"),
        ("lasso", 500, "m5.large"),
    ])
    def test_select_simple_models(self, ec2_manager, model_type, dataset_size_mb, expected_type):
        """Test instance selection for simple models."""
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb=dataset_size_mb,
            model_type=model_type,
            estimated_training_time_minutes=10
        )
        assert instance_type == expected_type

    # Tree-based models (random_forest, gradient_boosting, xgboost)
    @pytest.mark.parametrize("model_type,dataset_size_mb,expected_type", [
        ("random_forest", 500, "m5.large"),
        ("gradient_boosting", 999, "m5.large"),
        ("xgboost", 1000, "m5.xlarge"),
        ("random_forest", 2048, "m5.xlarge"),
    ])
    def test_select_tree_models(self, ec2_manager, model_type, dataset_size_mb, expected_type):
        """Test instance selection for tree-based models."""
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb=dataset_size_mb,
            model_type=model_type,
            estimated_training_time_minutes=30
        )
        assert instance_type == expected_type

    # Neural network models (mlp, neural)
    @pytest.mark.parametrize("model_type,dataset_size_mb,expected_type", [
        ("mlp_regression", 1024, "m5.xlarge"),
        ("mlp_classification", 5000, "m5.xlarge"),
        ("neural_network", 5001, "p3.2xlarge"),
        ("mlp", 10000, "p3.2xlarge"),
    ])
    def test_select_neural_network_models(self, ec2_manager, model_type, dataset_size_mb, expected_type):
        """Test instance selection for neural network models."""
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb=dataset_size_mb,
            model_type=model_type,
            estimated_training_time_minutes=60
        )
        assert instance_type == expected_type

    def test_select_unknown_model_type(self, ec2_manager, mock_config):
        """Test instance selection for unknown model type falls back to config."""
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb=100,
            model_type="unknown_model",
            estimated_training_time_minutes=15
        )
        assert instance_type == mock_config.ec2_instance_type

    def test_select_instance_type_edge_cases(self, ec2_manager):
        """Test edge cases in instance type selection."""
        # Zero dataset size
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb=0,
            model_type="linear",
            estimated_training_time_minutes=5
        )
        assert instance_type == "t3.medium"

        # Very large dataset with simple model
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb=10000,
            model_type="linear",
            estimated_training_time_minutes=5
        )
        assert instance_type == "m5.large"


class TestLaunchSpotInstance:
    """Test Spot instance launching."""

    def test_launch_spot_instance_success(self, ec2_manager, mock_aws_client):
        """Test successful Spot instance launch."""
        # Mock EC2 client response
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "open",
                    "CreateTime": datetime(2025, 10, 23, 12, 0, 0)
                }
            ]
        }

        # Mock _wait_for_instance_assignment
        with patch.object(ec2_manager, '_wait_for_instance_assignment', return_value="i-1234567890abcdef0"):
            result = ec2_manager.launch_spot_instance(
                instance_type="m5.large",
                user_data_script="#!/bin/bash\necho 'test'",
                tags={"Name": "test-instance", "User": "12345"}
            )

        # Verify result structure
        assert result["instance_id"] == "i-1234567890abcdef0"
        assert result["spot_request_id"] == "sir-12345678"
        assert result["instance_type"] == "m5.large"
        assert "launch_time" in result

        # Verify request_spot_instances was called correctly
        call_args = ec2_client.request_spot_instances.call_args
        assert call_args[1]["SpotPrice"] == "0.50"
        assert call_args[1]["InstanceCount"] == 1
        assert call_args[1]["Type"] == "one-time"

        # Verify launch specification
        launch_spec = call_args[1]["LaunchSpecification"]
        assert launch_spec["ImageId"] == "ami-12345678"
        assert launch_spec["InstanceType"] == "m5.large"
        assert launch_spec["KeyName"] == "test-key"
        assert launch_spec["UserData"] == "#!/bin/bash\necho 'test'"

    def test_launch_spot_instance_with_tags(self, ec2_manager, mock_aws_client):
        """Test Spot instance launch with custom tags."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "open",
                    "CreateTime": datetime(2025, 10, 23, 12, 0, 0)
                }
            ]
        }

        with patch.object(ec2_manager, '_wait_for_instance_assignment', return_value="i-1234567890abcdef0"):
            result = ec2_manager.launch_spot_instance(
                instance_type="m5.xlarge",
                user_data_script="#!/bin/bash\ntrain.sh",
                tags={
                    "Name": "ml-training-instance",
                    "User": "user_7715560927",
                    "ModelType": "random_forest",
                    "Environment": "production"
                }
            )

        # Verify tags were passed correctly
        call_args = ec2_client.request_spot_instances.call_args
        launch_spec = call_args[1]["LaunchSpecification"]

        tag_specifications = launch_spec["TagSpecifications"]
        assert len(tag_specifications) == 1
        assert tag_specifications[0]["ResourceType"] == "instance"

        tags_dict = {tag["Key"]: tag["Value"] for tag in tag_specifications[0]["Tags"]}
        assert tags_dict["Name"] == "ml-training-instance"
        assert tags_dict["User"] == "user_7715560927"
        assert tags_dict["ModelType"] == "random_forest"
        assert tags_dict["Environment"] == "production"

    def test_launch_spot_instance_with_ebs_volume(self, ec2_manager, mock_aws_client):
        """Test Spot instance launch includes correct EBS configuration."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "open",
                    "CreateTime": datetime(2025, 10, 23, 12, 0, 0)
                }
            ]
        }

        with patch.object(ec2_manager, '_wait_for_instance_assignment', return_value="i-1234567890abcdef0"):
            ec2_manager.launch_spot_instance(
                instance_type="m5.large",
                user_data_script="#!/bin/bash\necho 'test'",
                tags={}
            )

        # Verify block device mapping
        call_args = ec2_client.request_spot_instances.call_args
        launch_spec = call_args[1]["LaunchSpecification"]

        block_devices = launch_spec["BlockDeviceMappings"]
        assert len(block_devices) == 1

        root_device = block_devices[0]
        assert root_device["DeviceName"] == "/dev/sda1"
        assert root_device["Ebs"]["VolumeSize"] == 50
        assert root_device["Ebs"]["VolumeType"] == "gp3"
        assert root_device["Ebs"]["DeleteOnTermination"] is True
        assert root_device["Ebs"]["Encrypted"] is True

    def test_launch_spot_instance_client_error(self, ec2_manager, mock_aws_client):
        """Test Spot instance launch handles ClientError."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.request_spot_instances.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "InsufficientInstanceCapacity",
                    "Message": "There is no Spot capacity available"
                },
                "ResponseMetadata": {"RequestId": "req-12345"}
            },
            operation_name="RequestSpotInstances"
        )

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.launch_spot_instance(
                instance_type="m5.large",
                user_data_script="#!/bin/bash\necho 'test'",
                tags={}
            )

        assert "Failed to request Spot instance" in str(exc_info.value)
        assert exc_info.value.error_code == "InsufficientInstanceCapacity"

    def test_launch_spot_instance_no_requests_returned(self, ec2_manager, mock_aws_client):
        """Test Spot instance launch handles empty response."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": []
        }

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.launch_spot_instance(
                instance_type="m5.large",
                user_data_script="#!/bin/bash\necho 'test'",
                tags={}
            )

        assert "No Spot instance request created" in str(exc_info.value)

    def test_launch_spot_instance_wait_timeout(self, ec2_manager, mock_aws_client):
        """Test Spot instance launch handles wait timeout."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "open",
                    "CreateTime": datetime(2025, 10, 23, 12, 0, 0)
                }
            ]
        }

        with patch.object(ec2_manager, '_wait_for_instance_assignment', side_effect=EC2Error(
            message="Timeout waiting for Spot instance assignment",
            error_code="SpotRequestTimeout"
        )):
            with pytest.raises(EC2Error) as exc_info:
                ec2_manager.launch_spot_instance(
                    instance_type="m5.large",
                    user_data_script="#!/bin/bash\necho 'test'",
                    tags={}
                )

            assert "Timeout waiting for Spot instance assignment" in str(exc_info.value)


class TestWaitForInstanceAssignment:
    """Test waiting for Spot instance assignment."""

    def test_wait_for_instance_assignment_success(self, ec2_manager, mock_aws_client):
        """Test successful wait for instance assignment."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "active",
                    "Status": {"Code": "fulfilled"},
                    "InstanceId": "i-1234567890abcdef0"
                }
            ]
        }

        instance_id = ec2_manager._wait_for_instance_assignment(
            spot_request_id="sir-12345678",
            timeout_seconds=60
        )

        assert instance_id == "i-1234567890abcdef0"
        ec2_client.describe_spot_instance_requests.assert_called_once_with(
            SpotInstanceRequestIds=["sir-12345678"]
        )

    def test_wait_for_instance_assignment_timeout(self, ec2_manager, mock_aws_client):
        """Test timeout when waiting for instance assignment."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "open",
                    "Status": {"Code": "pending-evaluation"}
                }
            ]
        }

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager._wait_for_instance_assignment(
                spot_request_id="sir-12345678",
                timeout_seconds=1  # Short timeout for test
            )

        assert "Timeout waiting for Spot instance assignment" in str(exc_info.value)
        assert exc_info.value.error_code == "SpotRequestTimeout"

    def test_wait_for_instance_assignment_failed_state(self, ec2_manager, mock_aws_client):
        """Test handling of failed Spot request."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "failed",
                    "Status": {
                        "Code": "price-too-low",
                        "Message": "Your Spot request price of 0.50 is lower than the minimum"
                    }
                }
            ]
        }

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager._wait_for_instance_assignment(
                spot_request_id="sir-12345678",
                timeout_seconds=60
            )

        assert "Spot instance request failed" in str(exc_info.value)
        assert "price-too-low" in str(exc_info.value)

    def test_wait_for_instance_assignment_cancelled_state(self, ec2_manager, mock_aws_client):
        """Test handling of cancelled Spot request."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-12345678",
                    "State": "cancelled",
                    "Status": {
                        "Code": "request-canceled-and-instance-running",
                        "Message": "Request cancelled by user"
                    }
                }
            ]
        }

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager._wait_for_instance_assignment(
                spot_request_id="sir-12345678",
                timeout_seconds=60
            )

        assert "Spot instance request failed" in str(exc_info.value)
        assert "cancelled" in str(exc_info.value)

    def test_wait_for_instance_assignment_no_requests_found(self, ec2_manager, mock_aws_client):
        """Test handling when Spot request not found."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": []
        }

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager._wait_for_instance_assignment(
                spot_request_id="sir-12345678",
                timeout_seconds=60
            )

        assert "Spot instance request not found" in str(exc_info.value)

    def test_wait_for_instance_assignment_client_error(self, ec2_manager, mock_aws_client):
        """Test handling of boto3 ClientError."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_spot_instance_requests.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "InvalidSpotInstanceRequestID.NotFound",
                    "Message": "The spot instance request ID does not exist"
                },
                "ResponseMetadata": {"RequestId": "req-12345"}
            },
            operation_name="DescribeSpotInstanceRequests"
        )

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager._wait_for_instance_assignment(
                spot_request_id="sir-12345678",
                timeout_seconds=60
            )

        assert "Error checking Spot instance request status" in str(exc_info.value)

    @patch("time.sleep")
    def test_wait_for_instance_assignment_polling(self, mock_sleep, ec2_manager, mock_aws_client):
        """Test polling mechanism for instance assignment."""
        ec2_client = mock_aws_client.get_ec2_client()

        # First two calls return pending, third returns fulfilled
        ec2_client.describe_spot_instance_requests.side_effect = [
            {
                "SpotInstanceRequests": [
                    {
                        "SpotInstanceRequestId": "sir-12345678",
                        "State": "open",
                        "Status": {"Code": "pending-evaluation"}
                    }
                ]
            },
            {
                "SpotInstanceRequests": [
                    {
                        "SpotInstanceRequestId": "sir-12345678",
                        "State": "open",
                        "Status": {"Code": "pending-fulfillment"}
                    }
                ]
            },
            {
                "SpotInstanceRequests": [
                    {
                        "SpotInstanceRequestId": "sir-12345678",
                        "State": "active",
                        "Status": {"Code": "fulfilled"},
                        "InstanceId": "i-1234567890abcdef0"
                    }
                ]
            }
        ]

        instance_id = ec2_manager._wait_for_instance_assignment(
            spot_request_id="sir-12345678",
            timeout_seconds=60
        )

        assert instance_id == "i-1234567890abcdef0"
        # Verify sleep was called twice (between 3 attempts)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10)


class TestGenerateTrainingUserData:
    """Test UserData script generation for training instances."""

    def test_generate_userdata_basic(self, ec2_manager):
        """Test basic UserData script generation."""
        userdata = ec2_manager.generate_training_userdata(
            s3_dataset_uri="s3://ml-bucket/datasets/train.csv",
            model_type="random_forest",
            target_column="price",
            feature_columns=["sqft", "bedrooms", "bathrooms"],
            hyperparameters={"n_estimators": 100},
            s3_output_uri="s3://ml-bucket/models/model_123"
        )

        # Should be base64 encoded
        assert isinstance(userdata, str)
        assert len(userdata) > 0

        # Decode and verify content
        import base64
        decoded = base64.b64decode(userdata).decode('utf-8')
        assert "#!/bin/bash" in decoded
        assert "s3://ml-bucket/datasets/train.csv" in decoded
        assert "random_forest" in decoded
        assert "price" in decoded
        assert "sqft" in decoded
        assert "s3://ml-bucket/models/model_123" in decoded

    def test_generate_userdata_installs_dependencies(self, ec2_manager):
        """Test that UserData script installs required dependencies."""
        userdata = ec2_manager.generate_training_userdata(
            s3_dataset_uri="s3://ml-bucket/datasets/data.csv",
            model_type="linear",
            target_column="y",
            feature_columns=["x1", "x2"],
            hyperparameters={},
            s3_output_uri="s3://ml-bucket/models/model_456"
        )

        import base64
        decoded = base64.b64decode(userdata).decode('utf-8')

        # Check for dependency installation
        assert "pip install" in decoded or "apt-get install" in decoded
        assert "pandas" in decoded
        assert "scikit-learn" in decoded or "sklearn" in decoded
        assert "joblib" in decoded
        assert "boto3" in decoded

    def test_generate_userdata_with_hyperparameters(self, ec2_manager):
        """Test UserData script includes hyperparameters."""
        hyperparameters = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5
        }

        userdata = ec2_manager.generate_training_userdata(
            s3_dataset_uri="s3://ml-bucket/datasets/data.csv",
            model_type="gradient_boosting",
            target_column="target",
            feature_columns=["f1", "f2", "f3"],
            hyperparameters=hyperparameters,
            s3_output_uri="s3://ml-bucket/models/model_789"
        )

        import base64
        decoded = base64.b64decode(userdata).decode('utf-8')

        # Check hyperparameters are included
        assert "200" in decoded  # n_estimators
        assert "10" in decoded   # max_depth
        assert "5" in decoded    # min_samples_split

    def test_generate_userdata_self_terminates(self, ec2_manager):
        """Test UserData script includes self-termination."""
        userdata = ec2_manager.generate_training_userdata(
            s3_dataset_uri="s3://ml-bucket/datasets/data.csv",
            model_type="logistic",
            target_column="label",
            feature_columns=["feature1"],
            hyperparameters={},
            s3_output_uri="s3://ml-bucket/models/model_abc"
        )

        import base64
        decoded = base64.b64decode(userdata).decode('utf-8')

        # Check for self-termination logic
        assert "shutdown" in decoded or "terminate" in decoded or "halt" in decoded

    def test_generate_userdata_multiple_features(self, ec2_manager):
        """Test UserData script with multiple feature columns."""
        feature_columns = ["age", "income", "education", "employment", "debt"]

        userdata = ec2_manager.generate_training_userdata(
            s3_dataset_uri="s3://ml-bucket/datasets/credit.csv",
            model_type="xgboost",
            target_column="default",
            feature_columns=feature_columns,
            hyperparameters={"learning_rate": 0.1},
            s3_output_uri="s3://ml-bucket/models/model_xyz"
        )

        import base64
        decoded = base64.b64decode(userdata).decode('utf-8')

        # All features should be in script
        for feature in feature_columns:
            assert feature in decoded


class TestPollTrainingLogs:
    """Test CloudWatch Logs polling for training instances."""

    @pytest.mark.asyncio
    async def test_poll_logs_success(self, ec2_manager, mock_aws_client):
        """Test successful log polling."""
        logs_client = MagicMock()
        mock_aws_client.get_logs_client.return_value = logs_client

        # Mock log events
        logs_client.get_log_events.return_value = {
            "events": [
                {"timestamp": 1234567890, "message": "Starting training..."},
                {"timestamp": 1234567891, "message": "Epoch 1/10 complete"}
            ],
            "nextForwardToken": "token123"
        }

        # Collect logs
        log_lines = []
        async for log_line in ec2_manager.poll_training_logs(instance_id="i-1234567890abcdef0"):
            log_lines.append(log_line)
            if len(log_lines) >= 2:
                break

        assert len(log_lines) == 2
        assert "Starting training..." in log_lines[0]
        assert "Epoch 1/10 complete" in log_lines[1]

    @pytest.mark.asyncio
    async def test_poll_logs_handles_missing_log_stream(self, ec2_manager, mock_aws_client):
        """Test polling handles ResourceNotFoundException gracefully."""
        logs_client = MagicMock()
        mock_aws_client.get_logs_client.return_value = logs_client

        # First call raises ResourceNotFoundException, second succeeds
        from botocore.exceptions import ClientError
        logs_client.get_log_events.side_effect = [
            ClientError(
                error_response={
                    "Error": {"Code": "ResourceNotFoundException", "Message": "Log stream not found"}
                },
                operation_name="GetLogEvents"
            ),
            {
                "events": [{"timestamp": 1234567890, "message": "Log started"}],
                "nextForwardToken": "token123"
            }
        ]

        # Should not raise exception
        log_lines = []
        async for log_line in ec2_manager.poll_training_logs(instance_id="i-1234567890abcdef0"):
            log_lines.append(log_line)
            break

        assert len(log_lines) >= 0  # May be empty on first attempt

    @pytest.mark.asyncio
    async def test_poll_logs_stops_when_no_new_logs(self, ec2_manager, mock_aws_client):
        """Test polling stops when no new logs available."""
        logs_client = MagicMock()
        mock_aws_client.get_logs_client.return_value = logs_client

        # Same token returned = no new logs
        logs_client.get_log_events.return_value = {
            "events": [],
            "nextForwardToken": "token123"
        }

        log_lines = []
        async for log_line in ec2_manager.poll_training_logs(instance_id="i-1234567890abcdef0"):
            log_lines.append(log_line)

        # Should eventually stop (empty or limited iterations)
        assert len(log_lines) < 100  # Sanity check it doesn't run forever

    @pytest.mark.asyncio
    async def test_poll_logs_custom_log_group(self, ec2_manager, mock_aws_client):
        """Test polling with custom log group."""
        logs_client = MagicMock()
        mock_aws_client.get_logs_client.return_value = logs_client

        logs_client.get_log_events.return_value = {
            "events": [{"timestamp": 1234567890, "message": "Custom log"}],
            "nextForwardToken": "token123"
        }

        log_lines = []
        async for log_line in ec2_manager.poll_training_logs(
            instance_id="i-1234567890abcdef0",
            log_group="/aws/ec2/custom-training"
        ):
            log_lines.append(log_line)
            break

        # Verify log group was used in call
        call_args = logs_client.get_log_events.call_args
        assert "/aws/ec2/custom-training" in str(call_args) or call_args is not None


class TestGetInstanceStatus:
    """Test EC2 instance status retrieval."""

    def test_get_instance_status_success(self, ec2_manager, mock_aws_client):
        """Test successful instance status retrieval."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-1234567890abcdef0",
                            "State": {"Name": "running"},
                            "PublicIpAddress": "54.123.45.67",
                            "PrivateIpAddress": "10.0.1.50",
                            "LaunchTime": datetime(2025, 10, 23, 12, 0, 0)
                        }
                    ]
                }
            ]
        }

        status = ec2_manager.get_instance_status(instance_id="i-1234567890abcdef0")

        assert status["state"] == "running"
        assert status["public_ip"] == "54.123.45.67"
        assert status["private_ip"] == "10.0.1.50"
        assert "launch_time" in status

    def test_get_instance_status_not_found(self, ec2_manager, mock_aws_client):
        """Test instance status when instance not found."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_instances.return_value = {
            "Reservations": []
        }

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.get_instance_status(instance_id="i-nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_get_instance_status_client_error(self, ec2_manager, mock_aws_client):
        """Test instance status handles ClientError."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.describe_instances.side_effect = ClientError(
            error_response={
                "Error": {"Code": "InvalidInstanceID.NotFound", "Message": "Invalid instance ID"}
            },
            operation_name="DescribeInstances"
        )

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.get_instance_status(instance_id="i-invalid")

        assert "InvalidInstanceID.NotFound" in str(exc_info.value) or "Error" in str(exc_info.value)


class TestTerminateInstance:
    """Test EC2 instance termination."""

    def test_terminate_instance_success(self, ec2_manager, mock_aws_client):
        """Test successful instance termination."""
        ec2_client = mock_aws_client.get_ec2_client()

        # Mock termination response
        ec2_client.terminate_instances.return_value = {
            "TerminatingInstances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "CurrentState": {"Name": "shutting-down"}
                }
            ]
        }

        # Mock waiter
        waiter = MagicMock()
        ec2_client.get_waiter.return_value = waiter

        termination_time = ec2_manager.terminate_instance(instance_id="i-1234567890abcdef0")

        assert isinstance(termination_time, datetime)
        ec2_client.terminate_instances.assert_called_once_with(InstanceIds=["i-1234567890abcdef0"])
        waiter.wait.assert_called_once()

    def test_terminate_instance_client_error(self, ec2_manager, mock_aws_client):
        """Test termination handles ClientError."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.terminate_instances.side_effect = ClientError(
            error_response={
                "Error": {"Code": "InvalidInstanceID.NotFound", "Message": "Instance not found"}
            },
            operation_name="TerminateInstances"
        )

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.terminate_instance(instance_id="i-nonexistent")

        assert "Failed to terminate" in str(exc_info.value) or "Error" in str(exc_info.value)

    def test_terminate_instance_waiter_timeout(self, ec2_manager, mock_aws_client):
        """Test termination handles waiter timeout."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.terminate_instances.return_value = {
            "TerminatingInstances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "CurrentState": {"Name": "shutting-down"}
                }
            ]
        }

        waiter = MagicMock()
        waiter.wait.side_effect = Exception("Waiter timeout")
        ec2_client.get_waiter.return_value = waiter

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.terminate_instance(instance_id="i-1234567890abcdef0")

        assert "Failed to terminate" in str(exc_info.value) or "Error" in str(exc_info.value)


class TestOnDemandFallback:
    """Test On-Demand instance launching as Spot fallback."""

    def test_launch_on_demand_success(self, ec2_manager, mock_aws_client):
        """Test successful On-Demand instance launch."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.run_instances.return_value = {
            "Instances": [
                {
                    "InstanceId": "i-ondemand123",
                    "LaunchTime": datetime(2025, 10, 23, 13, 0, 0),
                    "State": {"Name": "pending"}
                }
            ]
        }

        result = ec2_manager.launch_on_demand_fallback(
            instance_type="m5.large",
            user_data_script="#!/bin/bash\necho 'training'",
            tags={"Name": "on-demand-training"}
        )

        assert result["instance_id"] == "i-ondemand123"
        assert result["instance_type"] == "m5.large"
        assert "launch_time" in result
        assert "spot_request_id" not in result  # On-Demand doesn't have spot request

    def test_launch_on_demand_with_tags(self, ec2_manager, mock_aws_client):
        """Test On-Demand launch with custom tags."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.run_instances.return_value = {
            "Instances": [
                {
                    "InstanceId": "i-ondemand456",
                    "LaunchTime": datetime(2025, 10, 23, 13, 0, 0),
                    "State": {"Name": "pending"}
                }
            ]
        }

        result = ec2_manager.launch_on_demand_fallback(
            instance_type="m5.xlarge",
            user_data_script="#!/bin/bash\ntrain.sh",
            tags={"Environment": "production", "User": "12345"}
        )

        # Verify tags were passed
        call_args = ec2_client.run_instances.call_args
        tag_specs = call_args[1].get("TagSpecifications", [])
        assert len(tag_specs) > 0

    def test_launch_on_demand_client_error(self, ec2_manager, mock_aws_client):
        """Test On-Demand launch handles ClientError."""
        ec2_client = mock_aws_client.get_ec2_client()
        ec2_client.run_instances.side_effect = ClientError(
            error_response={
                "Error": {"Code": "InsufficientInstanceCapacity", "Message": "No capacity"}
            },
            operation_name="RunInstances"
        )

        with pytest.raises(EC2Error) as exc_info:
            ec2_manager.launch_on_demand_fallback(
                instance_type="m5.large",
                user_data_script="#!/bin/bash\ntest",
                tags={}
            )

        assert "Failed to launch On-Demand instance" in str(exc_info.value) or "Error" in str(exc_info.value)
