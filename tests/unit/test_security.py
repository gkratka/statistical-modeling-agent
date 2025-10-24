"""
Unit tests for SecurityManager.

Tests security policy generation, encryption configuration, user isolation,
and audit logging functionality.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 7.7: Security Tests)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest
from botocore.exceptions import ClientError

from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import CloudConfigurationError, S3Error
from src.cloud.security import SecurityManager


@pytest.fixture
def cloud_config() -> CloudConfig:
    """Create CloudConfig instance for testing."""
    return CloudConfig(
        aws_region="us-west-2",
        aws_access_key_id="test_access_key",
        aws_secret_access_key="test_secret_key",
        s3_bucket="ml-agent-test-bucket",
        s3_data_prefix="datasets",
        s3_models_prefix="models",
        s3_results_prefix="predictions",
        ec2_instance_type="c5.xlarge",
        ec2_ami_id="ami-12345678",
        ec2_key_name="test-key",
        ec2_security_group="sg-12345678",
        lambda_function_name="ml-agent-predict",
        lambda_memory_mb=1024,
        lambda_timeout_seconds=300,
        max_training_cost_dollars=10.0,
        max_prediction_cost_dollars=1.0,
        cost_warning_threshold=0.8,
        iam_role_arn="arn:aws:iam::123456789012:role/ml-agent-role"
    )


@pytest.fixture
def cloud_config_with_kms(cloud_config: CloudConfig) -> CloudConfig:
    """Create CloudConfig with KMS encryption key."""
    cloud_config.kms_key_id = "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
    return cloud_config


@pytest.fixture
def security_manager(cloud_config: CloudConfig) -> SecurityManager:
    """Create SecurityManager instance for testing."""
    return SecurityManager(cloud_config)


class TestS3BucketPolicyGeneration:
    """Test S3 bucket policy generation."""

    def test_generate_s3_bucket_policy_denies_unencrypted(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test bucket policy denies unencrypted object uploads."""
        account_id = "123456789012"
        policy = security_manager.generate_s3_bucket_policy(account_id)

        # Verify policy structure
        assert "Version" in policy
        assert policy["Version"] == "2012-10-17"
        assert "Statement" in policy
        assert isinstance(policy["Statement"], list)
        assert len(policy["Statement"]) >= 1

        # Find DenyUnencryptedObjectUploads statement
        deny_unencrypted = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "DenyUnencryptedObjectUploads":
                deny_unencrypted = statement
                break

        # Verify deny unencrypted statement exists
        assert deny_unencrypted is not None
        assert deny_unencrypted["Effect"] == "Deny"
        assert deny_unencrypted["Action"] == "s3:PutObject"
        assert deny_unencrypted["Principal"] == "*"
        assert f"arn:aws:s3:::{cloud_config.s3_bucket}/*" in deny_unencrypted["Resource"]

        # Verify encryption condition
        assert "Condition" in deny_unencrypted
        assert "StringNotEquals" in deny_unencrypted["Condition"]
        assert "s3:x-amz-server-side-encryption" in deny_unencrypted["Condition"]["StringNotEquals"]
        assert deny_unencrypted["Condition"]["StringNotEquals"]["s3:x-amz-server-side-encryption"] == "AES256"

    def test_generate_s3_bucket_policy_denies_public_access(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test bucket policy denies public access."""
        account_id = "123456789012"
        policy = security_manager.generate_s3_bucket_policy(account_id)

        # Find DenyPublicAccess statement
        deny_public = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "DenyPublicAccess":
                deny_public = statement
                break

        # Verify deny public access statement
        assert deny_public is not None
        assert deny_public["Effect"] == "Deny"
        assert deny_public["Principal"] == "*"
        assert "s3:*" in deny_public["Action"]

    def test_generate_s3_bucket_policy_allows_bot_role(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test bucket policy allows bot role full access."""
        account_id = "123456789012"
        policy = security_manager.generate_s3_bucket_policy(account_id)

        # Find AllowBotRoleFullAccess statement
        allow_bot = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "AllowBotRoleFullAccess":
                allow_bot = statement
                break

        # Verify allow bot role statement
        assert allow_bot is not None
        assert allow_bot["Effect"] == "Allow"
        assert allow_bot["Principal"]["AWS"] == cloud_config.iam_role_arn
        assert allow_bot["Action"] == "s3:*"


class TestEC2IAMPolicyGeneration:
    """Test EC2 IAM policy generation."""

    def test_generate_ec2_iam_policy_allows_s3_read(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test EC2 IAM policy allows S3 dataset read."""
        policy = security_manager.generate_ec2_iam_role_policy()

        # Verify policy structure
        assert "Version" in policy
        assert policy["Version"] == "2012-10-17"
        assert "Statement" in policy

        # Find S3DatasetRead statement
        s3_read = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "S3DatasetRead":
                s3_read = statement
                break

        # Verify S3 read statement
        assert s3_read is not None
        assert s3_read["Effect"] == "Allow"
        assert "s3:GetObject" in s3_read["Action"]
        assert "s3:ListBucket" in s3_read["Action"]

        # Verify resources include datasets prefix
        assert any(
            f"{cloud_config.s3_data_prefix}" in resource
            for resource in s3_read["Resource"]
        )

    def test_generate_ec2_iam_policy_allows_s3_model_write(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test EC2 IAM policy allows S3 model write."""
        policy = security_manager.generate_ec2_iam_role_policy()

        # Find S3ModelWrite statement
        s3_write = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "S3ModelWrite":
                s3_write = statement
                break

        # Verify S3 write statement
        assert s3_write is not None
        assert s3_write["Effect"] == "Allow"
        assert "s3:PutObject" in s3_write["Action"]
        assert "s3:PutObjectAcl" in s3_write["Action"]

        # Verify resources include models prefix
        assert any(
            f"{cloud_config.s3_models_prefix}" in resource
            for resource in s3_write["Resource"]
        )

    def test_generate_ec2_iam_policy_denies_s3_write_datasets(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test EC2 IAM policy does not allow writing to datasets prefix."""
        policy = security_manager.generate_ec2_iam_role_policy()

        # Find S3ModelWrite statement
        s3_write = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "S3ModelWrite":
                s3_write = statement
                break

        # Verify datasets prefix is NOT in write resources
        assert s3_write is not None
        assert not any(
            f"{cloud_config.s3_data_prefix}" in resource
            for resource in s3_write["Resource"]
        )

    def test_generate_ec2_iam_policy_allows_cloudwatch_logs(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test EC2 IAM policy allows CloudWatch logs."""
        policy = security_manager.generate_ec2_iam_role_policy()

        # Find CloudWatchLogs statement
        cloudwatch = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "CloudWatchLogs":
                cloudwatch = statement
                break

        # Verify CloudWatch logs statement
        assert cloudwatch is not None
        assert cloudwatch["Effect"] == "Allow"
        assert "logs:CreateLogGroup" in cloudwatch["Action"]
        assert "logs:CreateLogStream" in cloudwatch["Action"]
        assert "logs:PutLogEvents" in cloudwatch["Action"]

    def test_generate_ec2_iam_policy_allows_self_terminate(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test EC2 IAM policy allows instance self-termination."""
        policy = security_manager.generate_ec2_iam_role_policy()

        # Find EC2SelfTerminate statement
        terminate = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "EC2SelfTerminate":
                terminate = statement
                break

        # Verify self-terminate statement
        assert terminate is not None
        assert terminate["Effect"] == "Allow"
        assert "ec2:TerminateInstances" in terminate["Action"]

        # Verify condition restricts to workflow tag
        assert "Condition" in terminate
        assert "StringEquals" in terminate["Condition"]
        assert "ec2:ResourceTag/workflow" in terminate["Condition"]["StringEquals"]
        assert terminate["Condition"]["StringEquals"]["ec2:ResourceTag/workflow"] == "cloud_training"


class TestLambdaIAMPolicyGeneration:
    """Test Lambda IAM policy generation."""

    def test_generate_lambda_iam_policy_allows_s3_read(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test Lambda IAM policy allows S3 read for models and datasets."""
        policy = security_manager.generate_lambda_iam_role_policy()

        # Find S3ReadModelAndData statement
        s3_read = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "S3ReadModelAndData":
                s3_read = statement
                break

        # Verify S3 read statement
        assert s3_read is not None
        assert s3_read["Effect"] == "Allow"
        assert "s3:GetObject" in s3_read["Action"]

        # Verify resources include both models and datasets
        assert any(
            f"{cloud_config.s3_models_prefix}" in resource
            for resource in s3_read["Resource"]
        )
        assert any(
            f"{cloud_config.s3_data_prefix}" in resource
            for resource in s3_read["Resource"]
        )

    def test_generate_lambda_iam_policy_allows_prediction_write(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test Lambda IAM policy allows writing predictions to S3."""
        policy = security_manager.generate_lambda_iam_role_policy()

        # Find S3WritePredictions statement
        s3_write = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "S3WritePredictions":
                s3_write = statement
                break

        # Verify S3 write predictions statement
        assert s3_write is not None
        assert s3_write["Effect"] == "Allow"
        assert "s3:PutObject" in s3_write["Action"]

        # Verify resources include predictions prefix
        assert any(
            "predictions" in resource
            for resource in s3_write["Resource"]
        )

    def test_generate_lambda_iam_policy_allows_cloudwatch_logs(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test Lambda IAM policy allows CloudWatch logs."""
        policy = security_manager.generate_lambda_iam_role_policy()

        # Find CloudWatchLogs statement
        cloudwatch = None
        for statement in policy["Statement"]:
            if statement.get("Sid") == "CloudWatchLogs":
                cloudwatch = statement
                break

        # Verify CloudWatch logs statement
        assert cloudwatch is not None
        assert cloudwatch["Effect"] == "Allow"
        assert "logs:CreateLogGroup" in cloudwatch["Action"]
        assert "logs:CreateLogStream" in cloudwatch["Action"]
        assert "logs:PutLogEvents" in cloudwatch["Action"]


class TestEncryptionConfiguration:
    """Test encryption configuration methods."""

    def test_configure_bucket_encryption_aes256(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test bucket encryption configuration with AES256."""
        # Create mock S3 client
        mock_s3_client = MagicMock()

        # Configure encryption
        security_manager.configure_bucket_encryption(mock_s3_client)

        # Verify put_bucket_encryption was called
        mock_s3_client.put_bucket_encryption.assert_called_once()

        # Extract call arguments
        call_args = mock_s3_client.put_bucket_encryption.call_args

        # Verify bucket name
        assert call_args[1]["Bucket"] == cloud_config.s3_bucket

        # Verify encryption configuration
        encryption_config = call_args[1]["ServerSideEncryptionConfiguration"]
        assert "Rules" in encryption_config
        assert len(encryption_config["Rules"]) == 1

        rule = encryption_config["Rules"][0]
        assert "ApplyServerSideEncryptionByDefault" in rule
        assert rule["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"
        assert rule["BucketKeyEnabled"] is True

    def test_configure_bucket_encryption_kms(
        self,
        cloud_config_with_kms: CloudConfig
    ) -> None:
        """Test bucket encryption configuration with KMS key."""
        security_manager = SecurityManager(cloud_config_with_kms)

        # Create mock S3 client
        mock_s3_client = MagicMock()

        # Configure encryption
        security_manager.configure_bucket_encryption(mock_s3_client)

        # Verify put_bucket_encryption was called
        mock_s3_client.put_bucket_encryption.assert_called_once()

        # Extract encryption configuration
        call_args = mock_s3_client.put_bucket_encryption.call_args
        encryption_config = call_args[1]["ServerSideEncryptionConfiguration"]

        # Verify KMS encryption
        rule = encryption_config["Rules"][0]
        assert rule["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "aws:kms"
        assert rule["ApplyServerSideEncryptionByDefault"]["KMSMasterKeyID"] == cloud_config_with_kms.kms_key_id

    def test_configure_bucket_encryption_client_error(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test bucket encryption configuration handles ClientError."""
        # Create mock S3 client that raises error
        mock_s3_client = MagicMock()
        mock_s3_client.put_bucket_encryption.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutBucketEncryption"
        )

        # Verify raises CloudConfigurationError
        with pytest.raises(CloudConfigurationError) as exc_info:
            security_manager.configure_bucket_encryption(mock_s3_client)

        assert "Failed to configure encryption" in str(exc_info.value)

    def test_configure_bucket_versioning(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test bucket versioning configuration."""
        # Create mock S3 client
        mock_s3_client = MagicMock()

        # Configure versioning
        security_manager.configure_bucket_versioning(mock_s3_client)

        # Verify put_bucket_versioning was called
        mock_s3_client.put_bucket_versioning.assert_called_once()

        # Extract call arguments
        call_args = mock_s3_client.put_bucket_versioning.call_args

        # Verify bucket name and versioning config
        assert call_args[1]["Bucket"] == cloud_config.s3_bucket
        assert call_args[1]["VersioningConfiguration"]["Status"] == "Enabled"

    def test_configure_bucket_versioning_client_error(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test bucket versioning configuration handles ClientError."""
        # Create mock S3 client that raises error
        mock_s3_client = MagicMock()
        mock_s3_client.put_bucket_versioning.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutBucketVersioning"
        )

        # Verify raises CloudConfigurationError
        with pytest.raises(CloudConfigurationError) as exc_info:
            security_manager.configure_bucket_versioning(mock_s3_client)

        assert "Failed to enable versioning" in str(exc_info.value)


class TestUserIsolationValidation:
    """Test user S3 access validation."""

    def test_validate_user_s3_access_own_prefix_write(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test user can write to their own prefix."""
        user_id = 12345
        s3_key = f"{cloud_config.s3_data_prefix}/user_{user_id}/dataset.csv"

        # Should not raise exception
        result = security_manager.validate_user_s3_access(
            s3_key=s3_key,
            user_id=user_id,
            operation="write"
        )

        assert result is True

    def test_validate_user_s3_access_own_prefix_read(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test user can read from their own prefix."""
        user_id = 12345
        s3_key = f"{cloud_config.s3_data_prefix}/user_{user_id}/dataset.csv"

        # Should not raise exception
        result = security_manager.validate_user_s3_access(
            s3_key=s3_key,
            user_id=user_id,
            operation="read"
        )

        assert result is True

    def test_validate_user_s3_access_other_user_write(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test user cannot write to another user's prefix."""
        user_id = 12345
        other_user_id = 67890
        s3_key = f"{cloud_config.s3_data_prefix}/user_{other_user_id}/dataset.csv"

        # Should raise S3Error
        with pytest.raises(S3Error) as exc_info:
            security_manager.validate_user_s3_access(
                s3_key=s3_key,
                user_id=user_id,
                operation="write"
            )

        assert "Write access denied" in str(exc_info.value)
        assert f"user {user_id}" in str(exc_info.value)

    def test_validate_user_s3_access_other_user_read(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test user cannot read another user's data."""
        user_id = 12345
        other_user_id = 67890
        s3_key = f"{cloud_config.s3_data_prefix}/user_{other_user_id}/dataset.csv"

        # Should raise S3Error
        with pytest.raises(S3Error) as exc_info:
            security_manager.validate_user_s3_access(
                s3_key=s3_key,
                user_id=user_id,
                operation="read"
            )

        assert "Read access denied" in str(exc_info.value)
        assert "belongs to another user" in str(exc_info.value)

    def test_validate_user_s3_access_model_prefix_write(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test user can write to models prefix with their user_id."""
        user_id = 12345
        s3_key = f"{cloud_config.s3_models_prefix}/user_{user_id}/model.pkl"

        # Should not raise exception
        result = security_manager.validate_user_s3_access(
            s3_key=s3_key,
            user_id=user_id,
            operation="write"
        )

        assert result is True

    def test_validate_user_s3_access_predictions_prefix_write(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test user can write to predictions prefix with their user_id."""
        user_id = 12345
        s3_key = f"predictions/user_{user_id}/predictions.csv"

        # Should not raise exception
        result = security_manager.validate_user_s3_access(
            s3_key=s3_key,
            user_id=user_id,
            operation="write"
        )

        assert result is True


class TestAuditLogging:
    """Test audit logging functionality."""

    def test_enable_s3_access_logging(
        self,
        security_manager: SecurityManager,
        cloud_config: CloudConfig
    ) -> None:
        """Test enabling S3 access logging."""
        # Create mock S3 client
        mock_s3_client = MagicMock()
        log_bucket = "ml-agent-logs"

        # Enable logging
        security_manager.enable_s3_access_logging(mock_s3_client, log_bucket)

        # Verify put_bucket_logging was called
        mock_s3_client.put_bucket_logging.assert_called_once()

        # Extract call arguments
        call_args = mock_s3_client.put_bucket_logging.call_args

        # Verify bucket and logging configuration
        assert call_args[1]["Bucket"] == cloud_config.s3_bucket
        assert call_args[1]["BucketLoggingStatus"]["LoggingEnabled"]["TargetBucket"] == log_bucket
        assert f"access-logs/{cloud_config.s3_bucket}/" in call_args[1]["BucketLoggingStatus"]["LoggingEnabled"]["TargetPrefix"]

    def test_enable_s3_access_logging_client_error(
        self,
        security_manager: SecurityManager
    ) -> None:
        """Test S3 access logging handles ClientError."""
        # Create mock S3 client that raises error
        mock_s3_client = MagicMock()
        mock_s3_client.put_bucket_logging.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutBucketLogging"
        )

        # Verify raises CloudConfigurationError
        with pytest.raises(CloudConfigurationError) as exc_info:
            security_manager.enable_s3_access_logging(mock_s3_client, "log-bucket")

        assert "Failed to enable access logging" in str(exc_info.value)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_audit_log_operation_creates_entry(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_file: Mock,
        security_manager: SecurityManager
    ) -> None:
        """Test audit log operation creates log entry."""
        # Mock file doesn't exist
        mock_exists.return_value = False

        # Create audit log entry
        user_id = 12345
        operation = "cloud_training"
        resource = "i-1234567890abcdef0"
        success = True
        metadata = {
            "instance_type": "c5.xlarge",
            "cost": 0.17
        }

        security_manager.audit_log_operation(
            user_id=user_id,
            operation=operation,
            resource=resource,
            success=success,
            **metadata
        )

        # Verify directory was created
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file was written
        assert mock_file.call_count >= 1

        # Get the write call
        write_calls = [call for call in mock_file().write.call_args_list]
        assert len(write_calls) > 0

        # Parse written JSON
        written_data = "".join([call[0][0] for call in write_calls])
        logs = json.loads(written_data)

        # Verify log entry structure
        assert isinstance(logs, list)
        assert len(logs) == 1

        log_entry = logs[0]
        assert log_entry["user_id"] == user_id
        assert log_entry["operation"] == operation
        assert log_entry["resource"] == resource
        assert log_entry["success"] is success
        assert log_entry["instance_type"] == "c5.xlarge"
        assert log_entry["cost"] == 0.17
        assert "timestamp" in log_entry

    @patch("builtins.open", new_callable=mock_open, read_data='[{"user_id": 999, "operation": "test"}]')
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_audit_log_operation_appends_to_existing(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_file: Mock,
        security_manager: SecurityManager
    ) -> None:
        """Test audit log operation appends to existing log file."""
        # Mock file exists
        mock_exists.return_value = True

        # Create audit log entry
        security_manager.audit_log_operation(
            user_id=12345,
            operation="cloud_prediction",
            resource="lambda-function",
            success=True
        )

        # Verify file was read and written
        assert mock_file.call_count >= 2  # Read and write

        # Get the write call
        write_calls = [call for call in mock_file().write.call_args_list]
        written_data = "".join([call[0][0] for call in write_calls])
        logs = json.loads(written_data)

        # Verify new entry was appended
        assert len(logs) == 2
        assert logs[0]["user_id"] == 999
        assert logs[1]["user_id"] == 12345
