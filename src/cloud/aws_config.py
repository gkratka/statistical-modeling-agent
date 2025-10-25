"""
AWS Cloud Configuration Management.

This module provides configuration loading, validation, and management
for AWS cloud infrastructure including S3, EC2, Lambda, and cost tracking.

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 1.2: Cloud Configuration)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.cloud.exceptions import CloudConfigurationError


@dataclass
class CloudConfig:
    """Configuration for cloud infrastructure (AWS/RunPod)."""

    # Cloud Provider Selection
    provider: str = "aws"  # "aws" or "runpod"

    # AWS Credentials
    aws_region: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # S3 Configuration
    s3_bucket: str = ""
    s3_data_prefix: str = ""
    s3_models_prefix: str = ""
    s3_results_prefix: str = ""

    # EC2 Configuration
    ec2_instance_type: str = ""
    ec2_ami_id: str = ""
    ec2_key_name: str = ""
    ec2_security_group: str = ""

    # Lambda Configuration
    lambda_function_name: str = ""
    lambda_memory_mb: int = 128
    lambda_timeout_seconds: int = 300

    # Cost Limits
    max_training_cost_dollars: float = 0.0
    max_prediction_cost_dollars: float = 0.0
    cost_warning_threshold: float = 0.8

    # S3 Configuration (optional)
    s3_lifecycle_days: Optional[int] = None

    # EC2 Configuration (optional)
    ec2_spot_max_price: Optional[float] = None

    # Lambda Configuration (optional)
    lambda_layer_arn: Optional[str] = None

    # Security (optional)
    kms_key_id: Optional[str] = None
    iam_role_arn: Optional[str] = None

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            CloudConfigurationError: If any configuration value is invalid
        """
        # Validate provider selection
        if self.provider not in ("aws", "runpod"):
            raise CloudConfigurationError(
                f"Invalid cloud provider: {self.provider}. Must be 'aws' or 'runpod'",
                config_key="provider",
                config_value=self.provider
            )

        # Validate AWS credentials
        if not self.aws_region or not self.aws_region.strip():
            raise CloudConfigurationError(
                "AWS region is required",
                config_key="aws_region"
            )

        if not self.aws_access_key_id or not self.aws_access_key_id.strip():
            raise CloudConfigurationError(
                "AWS access key ID is required",
                config_key="aws_access_key_id"
            )

        if not self.aws_secret_access_key or not self.aws_secret_access_key.strip():
            raise CloudConfigurationError(
                "AWS secret access key is required",
                config_key="aws_secret_access_key"
            )

        # Validate S3 configuration
        if not self.s3_bucket or not self.s3_bucket.strip():
            raise CloudConfigurationError(
                "S3 bucket is required",
                config_key="s3_bucket"
            )

        if not self.s3_data_prefix:
            raise CloudConfigurationError(
                "S3 data prefix is required",
                config_key="s3_data_prefix"
            )

        if not self.s3_models_prefix:
            raise CloudConfigurationError(
                "S3 models prefix is required",
                config_key="s3_models_prefix"
            )

        if not self.s3_results_prefix:
            raise CloudConfigurationError(
                "S3 results prefix is required",
                config_key="s3_results_prefix"
            )

        if self.s3_lifecycle_days is not None and self.s3_lifecycle_days <= 0:
            raise CloudConfigurationError(
                "S3 lifecycle days must be positive",
                config_key="s3_lifecycle_days",
                config_value=str(self.s3_lifecycle_days)
            )

        # Validate EC2 configuration
        if not self.ec2_instance_type or not self.ec2_instance_type.strip():
            raise CloudConfigurationError(
                "EC2 instance type is required",
                config_key="ec2_instance_type"
            )

        if not self.ec2_ami_id or not self.ec2_ami_id.strip():
            raise CloudConfigurationError(
                "EC2 AMI ID is required",
                config_key="ec2_ami_id"
            )

        if not self.ec2_key_name or not self.ec2_key_name.strip():
            raise CloudConfigurationError(
                "EC2 key name is required",
                config_key="ec2_key_name"
            )

        if not self.ec2_security_group or not self.ec2_security_group.strip():
            raise CloudConfigurationError(
                "EC2 security group is required",
                config_key="ec2_security_group"
            )

        if self.ec2_spot_max_price is not None and self.ec2_spot_max_price <= 0:
            raise CloudConfigurationError(
                "EC2 spot max price must be positive",
                config_key="ec2_spot_max_price",
                config_value=str(self.ec2_spot_max_price)
            )

        # Validate Lambda configuration
        if not self.lambda_function_name or not self.lambda_function_name.strip():
            raise CloudConfigurationError(
                "Lambda function name is required",
                config_key="lambda_function_name"
            )

        # AWS Lambda memory constraints: 128MB - 10240MB
        if self.lambda_memory_mb < 128 or self.lambda_memory_mb > 10240:
            raise CloudConfigurationError(
                "Lambda memory must be between 128MB and 10240MB",
                config_key="lambda_memory_mb",
                config_value=str(self.lambda_memory_mb)
            )

        # AWS Lambda timeout constraints: 1 - 900 seconds
        if self.lambda_timeout_seconds < 1 or self.lambda_timeout_seconds > 900:
            raise CloudConfigurationError(
                "Lambda timeout must be between 1 and 900 seconds",
                config_key="lambda_timeout_seconds",
                config_value=str(self.lambda_timeout_seconds)
            )

        # Validate cost limits
        if self.max_training_cost_dollars < 0:
            raise CloudConfigurationError(
                "Max training cost must be non-negative",
                config_key="max_training_cost_dollars",
                config_value=str(self.max_training_cost_dollars)
            )

        if self.max_prediction_cost_dollars < 0:
            raise CloudConfigurationError(
                "Max prediction cost must be non-negative",
                config_key="max_prediction_cost_dollars",
                config_value=str(self.max_prediction_cost_dollars)
            )

        if not (0.0 <= self.cost_warning_threshold <= 1.0):
            raise CloudConfigurationError(
                "Cost warning threshold must be between 0.0 and 1.0",
                config_key="cost_warning_threshold",
                config_value=str(self.cost_warning_threshold)
            )

    @classmethod
    def from_yaml(cls, config_path: str) -> "CloudConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            CloudConfig instance

        Raises:
            CloudConfigurationError: If config file is invalid or missing
        """
        path = Path(config_path)

        # Check file exists
        if not path.exists():
            raise CloudConfigurationError(
                f"Configuration file not found: {config_path}",
                config_key="config_path"
            )

        # Load YAML
        try:
            with open(path) as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise CloudConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_key="config_path"
            )
        except Exception as e:
            raise CloudConfigurationError(
                f"Error reading configuration file: {e}",
                config_key="config_path"
            )

        # Validate structure
        if not isinstance(config_data, dict):
            raise CloudConfigurationError(
                "Configuration file must contain a dictionary",
                config_key="config_structure"
            )

        # Extract sections (support both nested "cloud" and top-level structure)
        try:
            # Check if using nested cloud structure (config.yaml style)
            if "cloud" in config_data:
                cloud_section = config_data["cloud"]
                provider = cloud_section.get("provider", "aws")
                aws_config = cloud_section.get("aws", {})
                s3_config = cloud_section.get("s3", {})
                ec2_config = cloud_section.get("ec2", {})
                lambda_config = cloud_section.get("lambda", {})
                cost_config = cloud_section.get("cost_limits", {})
                security_config = cloud_section.get("security", {})
            else:
                # Top-level structure (test/legacy style)
                provider = "aws"
                aws_config = config_data.get("aws", {})
                s3_config = config_data.get("s3", {})
                ec2_config = config_data.get("ec2", {})
                lambda_config = config_data.get("lambda", {})
                cost_config = config_data.get("cost_limits", {})
                security_config = config_data.get("security", {})

            # Create CloudConfig instance
            config = cls(
                # Provider
                provider=provider,
                # AWS credentials
                aws_region=aws_config.get("region", ""),
                aws_access_key_id=aws_config.get("access_key_id", ""),
                aws_secret_access_key=aws_config.get("secret_access_key", ""),

                # S3 configuration
                s3_bucket=s3_config.get("bucket", ""),
                s3_data_prefix=s3_config.get("data_prefix", ""),
                s3_models_prefix=s3_config.get("models_prefix", ""),
                s3_results_prefix=s3_config.get("results_prefix", ""),
                s3_lifecycle_days=s3_config.get("lifecycle_days"),

                # EC2 configuration
                ec2_instance_type=ec2_config.get("instance_type", ""),
                ec2_ami_id=ec2_config.get("ami_id", ""),
                ec2_key_name=ec2_config.get("key_name", ""),
                ec2_security_group=ec2_config.get("security_group", ""),
                ec2_spot_max_price=ec2_config.get("spot_max_price"),

                # Lambda configuration
                lambda_function_name=lambda_config.get("function_name", ""),
                lambda_memory_mb=lambda_config.get("memory_mb", 128),
                lambda_timeout_seconds=lambda_config.get("timeout_seconds", 300),
                lambda_layer_arn=lambda_config.get("layer_arn"),

                # Cost limits
                max_training_cost_dollars=cost_config.get("max_training_cost", 0.0),
                max_prediction_cost_dollars=cost_config.get("max_prediction_cost", 0.0),
                cost_warning_threshold=cost_config.get("warning_threshold", 0.8),

                # Security
                kms_key_id=security_config.get("kms_key_id"),
                iam_role_arn=security_config.get("iam_role_arn")
            )

            # Validate configuration
            config.validate()

            return config

        except KeyError as e:
            raise CloudConfigurationError(
                f"Missing required configuration key: {e}",
                config_key=str(e)
            )
        except (TypeError, ValueError) as e:
            raise CloudConfigurationError(
                f"Invalid configuration value: {e}",
                config_key="config_value"
            )

    @classmethod
    def from_env(cls) -> "CloudConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            CLOUD_PROVIDER: Cloud provider (aws or runpod), defaults to "aws"
            AWS_REGION: AWS region
            AWS_ACCESS_KEY_ID: AWS access key ID
            AWS_SECRET_ACCESS_KEY: AWS secret access key
            S3_BUCKET: S3 bucket name
            S3_DATA_PREFIX: S3 data prefix
            S3_MODELS_PREFIX: S3 models prefix
            S3_RESULTS_PREFIX: S3 results prefix
            S3_LIFECYCLE_DAYS: Optional S3 lifecycle days
            EC2_INSTANCE_TYPE: EC2 instance type
            EC2_AMI_ID: EC2 AMI ID
            EC2_KEY_NAME: EC2 key pair name
            EC2_SECURITY_GROUP: EC2 security group ID
            EC2_SPOT_MAX_PRICE: Optional EC2 spot max price
            LAMBDA_FUNCTION_NAME: Lambda function name
            LAMBDA_MEMORY_MB: Lambda memory in MB
            LAMBDA_TIMEOUT_SECONDS: Lambda timeout in seconds
            LAMBDA_LAYER_ARN: Optional Lambda layer ARN
            MAX_TRAINING_COST: Max training cost in dollars
            MAX_PREDICTION_COST: Max prediction cost in dollars
            COST_WARNING_THRESHOLD: Cost warning threshold (0.0-1.0)
            KMS_KEY_ID: Optional KMS key ID
            IAM_ROLE_ARN: Optional IAM role ARN

        Returns:
            CloudConfig instance

        Raises:
            CloudConfigurationError: If required environment variables are missing
        """
        try:
            # Parse optional numeric fields
            s3_lifecycle_days = None
            if os.getenv("S3_LIFECYCLE_DAYS"):
                s3_lifecycle_days = int(os.getenv("S3_LIFECYCLE_DAYS"))

            ec2_spot_max_price = None
            if os.getenv("EC2_SPOT_MAX_PRICE"):
                ec2_spot_max_price = float(os.getenv("EC2_SPOT_MAX_PRICE"))

            config = cls(
                # Provider
                provider=os.getenv("CLOUD_PROVIDER", "aws"),
                # AWS credentials
                aws_region=os.getenv("AWS_REGION", ""),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),

                # S3 configuration
                s3_bucket=os.getenv("S3_BUCKET", ""),
                s3_data_prefix=os.getenv("S3_DATA_PREFIX", ""),
                s3_models_prefix=os.getenv("S3_MODELS_PREFIX", ""),
                s3_results_prefix=os.getenv("S3_RESULTS_PREFIX", ""),
                s3_lifecycle_days=s3_lifecycle_days,

                # EC2 configuration
                ec2_instance_type=os.getenv("EC2_INSTANCE_TYPE", ""),
                ec2_ami_id=os.getenv("EC2_AMI_ID", ""),
                ec2_key_name=os.getenv("EC2_KEY_NAME", ""),
                ec2_security_group=os.getenv("EC2_SECURITY_GROUP", ""),
                ec2_spot_max_price=ec2_spot_max_price,

                # Lambda configuration
                lambda_function_name=os.getenv("LAMBDA_FUNCTION_NAME", ""),
                lambda_memory_mb=int(os.getenv("LAMBDA_MEMORY_MB", "128")),
                lambda_timeout_seconds=int(os.getenv("LAMBDA_TIMEOUT_SECONDS", "300")),
                lambda_layer_arn=os.getenv("LAMBDA_LAYER_ARN"),

                # Cost limits
                max_training_cost_dollars=float(os.getenv("MAX_TRAINING_COST", "0.0")),
                max_prediction_cost_dollars=float(os.getenv("MAX_PREDICTION_COST", "0.0")),
                cost_warning_threshold=float(os.getenv("COST_WARNING_THRESHOLD", "0.8")),

                # Security
                kms_key_id=os.getenv("KMS_KEY_ID"),
                iam_role_arn=os.getenv("IAM_ROLE_ARN")
            )

            # Validate configuration
            config.validate()

            return config

        except ValueError as e:
            raise CloudConfigurationError(
                f"Invalid environment variable value: {e}",
                config_key="environment_variable"
            )
        except Exception as e:
            raise CloudConfigurationError(
                f"Error loading configuration from environment: {e}",
                config_key="environment"
            )
