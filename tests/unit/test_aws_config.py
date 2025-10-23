"""
Unit tests for AWS CloudConfig dataclass.

This test suite validates cloud configuration loading, validation,
and error handling for AWS infrastructure setup.

Test Coverage:
- Valid configuration from YAML
- Valid configuration from environment variables
- Missing required fields validation
- Invalid value validation
- Field type validation
- File loading error handling

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 1.2: Cloud Configuration)
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import CloudConfigurationError


class TestCloudConfigValidation:
    """Test CloudConfig validation logic."""

    def test_valid_minimal_config(self):
        """Minimal valid configuration should pass validation."""
        config = CloudConfig(
            # AWS Credentials
            aws_region="us-east-1",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",

            # S3 Configuration
            s3_bucket="ml-models-bucket",
            s3_data_prefix="data/",
            s3_models_prefix="models/",
            s3_results_prefix="results/",

            # EC2 Configuration
            ec2_instance_type="t3.medium",
            ec2_ami_id="ami-0c55b159cbfafe1f0",
            ec2_key_name="ml-training-key",
            ec2_security_group="sg-0123456789abcdef0",

            # Lambda Configuration
            lambda_function_name="ml-prediction-handler",
            lambda_memory_mb=512,
            lambda_timeout_seconds=300,

            # Cost Limits
            max_training_cost_dollars=10.0,
            max_prediction_cost_dollars=1.0,
            cost_warning_threshold=0.8
        )

        # Should not raise any exceptions
        config.validate()

    def test_valid_full_config(self):
        """Full configuration with all optional fields should pass."""
        config = CloudConfig(
            # AWS Credentials
            aws_region="us-west-2",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",

            # S3 Configuration
            s3_bucket="ml-models-bucket",
            s3_data_prefix="data/",
            s3_models_prefix="models/",
            s3_results_prefix="results/",
            s3_lifecycle_days=30,

            # EC2 Configuration
            ec2_instance_type="c5.2xlarge",
            ec2_ami_id="ami-0c55b159cbfafe1f0",
            ec2_key_name="ml-training-key",
            ec2_security_group="sg-0123456789abcdef0",
            ec2_spot_max_price=0.50,

            # Lambda Configuration
            lambda_function_name="ml-prediction-handler",
            lambda_memory_mb=1024,
            lambda_timeout_seconds=600,
            lambda_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:ml-libs:1",

            # Cost Limits
            max_training_cost_dollars=100.0,
            max_prediction_cost_dollars=10.0,
            cost_warning_threshold=0.75,

            # Security
            kms_key_id="arn:aws:kms:us-west-2:123456789012:key/12345678-1234",
            iam_role_arn="arn:aws:iam::123456789012:role/MLTrainingRole"
        )

        config.validate()

    def test_missing_aws_region_raises_error(self):
        """Missing AWS region should raise CloudConfigurationError."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="",  # Empty region
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_bucket="ml-models-bucket",
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=512,
                lambda_timeout_seconds=300,
                max_training_cost_dollars=10.0,
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=0.8
            )
            config.validate()

        assert "aws_region" in str(exc_info.value).lower()

    def test_missing_aws_credentials_raises_error(self):
        """Missing AWS credentials should raise CloudConfigurationError."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="us-east-1",
                aws_access_key_id="",  # Empty
                aws_secret_access_key="",  # Empty
                s3_bucket="ml-models-bucket",
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=512,
                lambda_timeout_seconds=300,
                max_training_cost_dollars=10.0,
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=0.8
            )
            config.validate()

        error_msg = str(exc_info.value).lower()
        assert "access_key_id" in error_msg or "secret_access_key" in error_msg

    def test_missing_s3_bucket_raises_error(self):
        """Missing S3 bucket should raise CloudConfigurationError."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="us-east-1",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_bucket="",  # Empty bucket
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=512,
                lambda_timeout_seconds=300,
                max_training_cost_dollars=10.0,
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=0.8
            )
            config.validate()

        assert "s3_bucket" in str(exc_info.value).lower()

    def test_invalid_lambda_memory_raises_error(self):
        """Lambda memory below 128MB should raise error."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="us-east-1",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_bucket="ml-models-bucket",
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=64,  # Below AWS minimum of 128MB
                lambda_timeout_seconds=300,
                max_training_cost_dollars=10.0,
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=0.8
            )
            config.validate()

        assert "lambda_memory_mb" in str(exc_info.value).lower()

    def test_invalid_lambda_timeout_raises_error(self):
        """Lambda timeout above 900 seconds should raise error."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="us-east-1",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_bucket="ml-models-bucket",
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=512,
                lambda_timeout_seconds=1000,  # Above AWS maximum of 900s
                max_training_cost_dollars=10.0,
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=0.8
            )
            config.validate()

        assert "lambda_timeout_seconds" in str(exc_info.value).lower()

    def test_invalid_cost_threshold_raises_error(self):
        """Cost warning threshold outside 0-1 range should raise error."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="us-east-1",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_bucket="ml-models-bucket",
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=512,
                lambda_timeout_seconds=300,
                max_training_cost_dollars=10.0,
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=1.5  # Invalid: must be 0.0-1.0
            )
            config.validate()

        assert "cost_warning_threshold" in str(exc_info.value).lower()

    def test_negative_cost_limit_raises_error(self):
        """Negative cost limits should raise error."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            config = CloudConfig(
                aws_region="us-east-1",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_bucket="ml-models-bucket",
                s3_data_prefix="data/",
                s3_models_prefix="models/",
                s3_results_prefix="results/",
                ec2_instance_type="t3.medium",
                ec2_ami_id="ami-0c55b159cbfafe1f0",
                ec2_key_name="ml-training-key",
                ec2_security_group="sg-0123456789abcdef0",
                lambda_function_name="ml-prediction-handler",
                lambda_memory_mb=512,
                lambda_timeout_seconds=300,
                max_training_cost_dollars=-1.0,  # Negative cost
                max_prediction_cost_dollars=1.0,
                cost_warning_threshold=0.8
            )
            config.validate()

        assert "max_training_cost_dollars" in str(exc_info.value).lower()


class TestCloudConfigFromYAML:
    """Test CloudConfig.from_yaml() method."""

    @pytest.fixture
    def valid_yaml_config(self) -> str:
        """Create valid YAML configuration content."""
        return """
aws:
  region: us-east-1
  access_key_id: AKIAIOSFODNN7EXAMPLE
  secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

s3:
  bucket: ml-models-bucket
  data_prefix: data/
  models_prefix: models/
  results_prefix: results/
  lifecycle_days: 30

ec2:
  instance_type: t3.medium
  ami_id: ami-0c55b159cbfafe1f0
  key_name: ml-training-key
  security_group: sg-0123456789abcdef0
  spot_max_price: 0.25

lambda:
  function_name: ml-prediction-handler
  memory_mb: 512
  timeout_seconds: 300
  layer_arn: arn:aws:lambda:us-east-1:123456789012:layer:ml-libs:1

cost_limits:
  max_training_cost: 10.0
  max_prediction_cost: 1.0
  warning_threshold: 0.8

security:
  kms_key_id: arn:aws:kms:us-east-1:123456789012:key/12345678-1234
  iam_role_arn: arn:aws:iam::123456789012:role/MLTrainingRole
"""

    def test_load_from_valid_yaml_file(self, valid_yaml_config):
        """Loading from valid YAML file should succeed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_yaml_config)
            config_path = f.name

        try:
            config = CloudConfig.from_yaml(config_path)

            # Verify AWS settings
            assert config.aws_region == "us-east-1"
            assert config.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
            assert config.aws_secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

            # Verify S3 settings
            assert config.s3_bucket == "ml-models-bucket"
            assert config.s3_data_prefix == "data/"
            assert config.s3_lifecycle_days == 30

            # Verify EC2 settings
            assert config.ec2_instance_type == "t3.medium"
            assert config.ec2_spot_max_price == 0.25

            # Verify Lambda settings
            assert config.lambda_function_name == "ml-prediction-handler"
            assert config.lambda_memory_mb == 512

            # Verify cost limits
            assert config.max_training_cost_dollars == 10.0
            assert config.cost_warning_threshold == 0.8

            # Verify security settings
            assert config.kms_key_id == "arn:aws:kms:us-east-1:123456789012:key/12345678-1234"
            assert config.iam_role_arn == "arn:aws:iam::123456789012:role/MLTrainingRole"
        finally:
            os.unlink(config_path)

    def test_load_from_nonexistent_file_raises_error(self):
        """Loading from nonexistent file should raise CloudConfigurationError."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            CloudConfig.from_yaml("/nonexistent/path/config.yaml")

        assert "not found" in str(exc_info.value).lower()

    def test_load_from_invalid_yaml_raises_error(self):
        """Loading from invalid YAML should raise CloudConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unmatched")
            config_path = f.name

        try:
            with pytest.raises(CloudConfigurationError) as exc_info:
                CloudConfig.from_yaml(config_path)

            assert "yaml" in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)

    def test_load_from_missing_required_field_raises_error(self):
        """Loading YAML missing required fields should raise error."""
        incomplete_yaml = """
aws:
  region: us-east-1
  # Missing credentials
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(incomplete_yaml)
            config_path = f.name

        try:
            with pytest.raises(CloudConfigurationError) as exc_info:
                CloudConfig.from_yaml(config_path)

            # Should mention missing field
            error_msg = str(exc_info.value).lower()
            assert "access_key_id" in error_msg or "required" in error_msg
        finally:
            os.unlink(config_path)


class TestCloudConfigFromEnv:
    """Test CloudConfig.from_env() method."""

    def test_load_from_environment_variables(self, monkeypatch):
        """Loading from environment variables should succeed."""
        # Set environment variables
        env_vars = {
            "AWS_REGION": "us-west-2",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "S3_BUCKET": "ml-models-bucket",
            "S3_DATA_PREFIX": "data/",
            "S3_MODELS_PREFIX": "models/",
            "S3_RESULTS_PREFIX": "results/",
            "EC2_INSTANCE_TYPE": "c5.xlarge",
            "EC2_AMI_ID": "ami-0c55b159cbfafe1f0",
            "EC2_KEY_NAME": "ml-training-key",
            "EC2_SECURITY_GROUP": "sg-0123456789abcdef0",
            "LAMBDA_FUNCTION_NAME": "ml-prediction-handler",
            "LAMBDA_MEMORY_MB": "1024",
            "LAMBDA_TIMEOUT_SECONDS": "600",
            "MAX_TRAINING_COST": "50.0",
            "MAX_PREDICTION_COST": "5.0",
            "COST_WARNING_THRESHOLD": "0.75",
        }

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = CloudConfig.from_env()

        # Verify loaded values
        assert config.aws_region == "us-west-2"
        assert config.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert config.s3_bucket == "ml-models-bucket"
        assert config.ec2_instance_type == "c5.xlarge"
        assert config.lambda_memory_mb == 1024
        assert config.max_training_cost_dollars == 50.0

    def test_load_from_env_with_defaults(self, monkeypatch):
        """Loading from env with missing optional fields should use defaults."""
        # Set only required fields
        env_vars = {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "S3_BUCKET": "ml-models-bucket",
            "S3_DATA_PREFIX": "data/",
            "S3_MODELS_PREFIX": "models/",
            "S3_RESULTS_PREFIX": "results/",
            "EC2_INSTANCE_TYPE": "t3.medium",
            "EC2_AMI_ID": "ami-0c55b159cbfafe1f0",
            "EC2_KEY_NAME": "ml-training-key",
            "EC2_SECURITY_GROUP": "sg-0123456789abcdef0",
            "LAMBDA_FUNCTION_NAME": "ml-prediction-handler",
            "LAMBDA_MEMORY_MB": "512",
            "LAMBDA_TIMEOUT_SECONDS": "300",
            "MAX_TRAINING_COST": "10.0",
            "MAX_PREDICTION_COST": "1.0",
            "COST_WARNING_THRESHOLD": "0.8",
        }

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = CloudConfig.from_env()

        # Optional fields should be None
        assert config.s3_lifecycle_days is None
        assert config.ec2_spot_max_price is None
        assert config.lambda_layer_arn is None
        assert config.kms_key_id is None
        assert config.iam_role_arn is None

    def test_load_from_env_missing_required_raises_error(self, monkeypatch):
        """Loading from env with missing required field should raise error."""
        # Set incomplete environment
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        # Missing AWS credentials and other required fields

        with pytest.raises(CloudConfigurationError):
            CloudConfig.from_env()
