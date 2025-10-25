"""
AWS Security Manager for IAM policies, S3 encryption, and user isolation.

This module provides SecurityManager class for generating IAM policies,
configuring S3 encryption, validating user access, and audit logging.

Key Features:
- S3 bucket policy generation with encryption enforcement
- IAM role policies for EC2 training instances and Lambda functions
- S3 encryption configuration (AES256 or KMS)
- User isolation validation (user_{user_id} prefix enforcement)
- Audit logging for security-relevant operations

Security Design:
- Least privilege access (minimal permissions)
- User isolation (users can only access their own data)
- Encryption at rest (AES256 or KMS)
- Audit trail (all operations logged)

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 7.1-7.5: Security & User Isolation)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from botocore.exceptions import ClientError

from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import CloudConfigurationError, S3Error


class SecurityManager:
    """
    Manage AWS security policies, encryption, and user isolation.

    Provides methods for generating IAM policies, configuring S3 encryption,
    validating user access to S3 paths, and logging security operations.
    """

    def __init__(self, config: CloudConfig) -> None:
        """
        Initialize SecurityManager with cloud configuration.

        Args:
            config: CloudConfig instance with AWS settings

        Example:
            >>> config = CloudConfig.from_yaml("config.yaml")
            >>> security_manager = SecurityManager(config)
        """
        self.config = config

    def generate_s3_bucket_policy(self, account_id: str) -> Dict[str, Any]:
        """
        Generate S3 bucket policy with user isolation and encryption.

        Policy enforces:
        - All uploads must be encrypted (AES256)
        - Public access blocked
        - Bot role has full access

        Args:
            account_id: AWS account ID for public access restriction

        Returns:
            dict: S3 bucket policy JSON

        Example:
            >>> policy = security_manager.generate_s3_bucket_policy("123456789012")
            >>> print(json.dumps(policy, indent=2))
        """
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DenyUnencryptedObjectUploads",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": "s3:PutObject",
                    "Resource": f"arn:aws:s3:::{self.config.s3_bucket}/*",
                    "Condition": {
                        "StringNotEquals": {
                            "s3:x-amz-server-side-encryption": "AES256"
                        }
                    }
                },
                {
                    "Sid": "DenyPublicAccess",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": "s3:*",
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket}",
                        f"arn:aws:s3:::{self.config.s3_bucket}/*"
                    ],
                    "Condition": {
                        "StringNotEquals": {
                            "aws:PrincipalAccount": account_id
                        }
                    }
                },
                {
                    "Sid": "AllowBotRoleFullAccess",
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": self.config.iam_role_arn
                    },
                    "Action": "s3:*",
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket}",
                        f"arn:aws:s3:::{self.config.s3_bucket}/*"
                    ]
                }
            ]
        }

    def generate_ec2_iam_role_policy(self) -> Dict[str, Any]:
        """
        Generate IAM policy for EC2 training instances.

        Permissions granted:
        - Read from S3 datasets prefix
        - Write to S3 models prefix
        - Write CloudWatch logs
        - Terminate self (with workflow tag condition)

        Returns:
            dict: IAM policy JSON

        Example:
            >>> policy = security_manager.generate_ec2_iam_role_policy()
            >>> print(json.dumps(policy, indent=2))
        """
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "S3DatasetRead",
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket}",
                        f"arn:aws:s3:::{self.config.s3_bucket}/{self.config.s3_data_prefix}/*"
                    ]
                },
                {
                    "Sid": "S3ModelWrite",
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject",
                        "s3:PutObjectAcl"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket}/{self.config.s3_models_prefix}/*"
                    ]
                },
                {
                    "Sid": "CloudWatchLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:*:*:*"
                },
                {
                    "Sid": "EC2SelfTerminate",
                    "Effect": "Allow",
                    "Action": [
                        "ec2:TerminateInstances"
                    ],
                    "Resource": "*",
                    "Condition": {
                        "StringEquals": {
                            "ec2:ResourceTag/workflow": "cloud_training"
                        }
                    }
                }
            ]
        }

    def generate_lambda_iam_role_policy(self) -> Dict[str, Any]:
        """
        Generate IAM policy for Lambda prediction function.

        Permissions granted:
        - Read models and datasets from S3
        - Write predictions to S3
        - Write CloudWatch logs

        Returns:
            dict: IAM policy JSON

        Example:
            >>> policy = security_manager.generate_lambda_iam_role_policy()
            >>> print(json.dumps(policy, indent=2))
        """
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "S3ReadModelAndData",
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket}/{self.config.s3_models_prefix}/*",
                        f"arn:aws:s3:::{self.config.s3_bucket}/{self.config.s3_data_prefix}/*"
                    ]
                },
                {
                    "Sid": "S3WritePredictions",
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket}/predictions/*"
                    ]
                },
                {
                    "Sid": "CloudWatchLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:*:*:*"
                }
            ]
        }

    def configure_bucket_encryption(self, s3_client: Any) -> None:
        """
        Enable default encryption for S3 bucket.

        Uses AES256 server-side encryption by default.
        If KMS key configured, uses KMS encryption instead.

        Args:
            s3_client: boto3 S3 client

        Raises:
            CloudConfigurationError: If encryption configuration fails

        Example:
            >>> import boto3
            >>> s3_client = boto3.client('s3')
            >>> security_manager.configure_bucket_encryption(s3_client)
        """
        # Build encryption configuration
        if self.config.kms_key_id:
            # Use KMS encryption if key configured
            encryption_config = {
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'aws:kms',
                            'KMSMasterKeyID': self.config.kms_key_id
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }
        else:
            # Use AES256 encryption by default
            encryption_config = {
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }

        # Apply encryption configuration
        try:
            s3_client.put_bucket_encryption(
                Bucket=self.config.s3_bucket,
                ServerSideEncryptionConfiguration=encryption_config
            )
        except ClientError as e:
            raise CloudConfigurationError(
                f"Failed to configure encryption: {e}",
                config_key="s3_encryption"
            )

    def configure_bucket_versioning(self, s3_client: Any) -> None:
        """
        Enable versioning for S3 bucket.

        Versioning provides:
        - Protection against accidental deletion
        - Audit trail of object changes
        - Ability to restore previous versions

        Args:
            s3_client: boto3 S3 client

        Raises:
            CloudConfigurationError: If versioning configuration fails

        Example:
            >>> import boto3
            >>> s3_client = boto3.client('s3')
            >>> security_manager.configure_bucket_versioning(s3_client)
        """
        try:
            s3_client.put_bucket_versioning(
                Bucket=self.config.s3_bucket,
                VersioningConfiguration={'Status': 'Enabled'}
            )
        except ClientError as e:
            raise CloudConfigurationError(
                f"Failed to enable versioning: {e}",
                config_key="s3_versioning"
            )

    def validate_user_s3_access(
        self,
        s3_key: str,
        user_id: int,
        operation: str
    ) -> bool:
        """
        Validate user can access S3 path.

        Enforces user isolation:
        - Users can only access paths with user_{user_id} prefix
        - Read operations: can read own data, blocked from other users
        - Write operations: only allowed for user's own prefix

        Args:
            s3_key: S3 object key to validate
            user_id: User ID requesting access
            operation: 'read' or 'write'

        Returns:
            bool: True if access allowed

        Raises:
            S3Error: If access denied

        Example:
            >>> security_manager.validate_user_s3_access(
            ...     s3_key="datasets/user_12345/data.csv",
            ...     user_id=12345,
            ...     operation="write"
            ... )
            True
        """
        user_prefix = f"user_{user_id}"

        # Validate write operations
        if operation == 'write':
            # Write: must be in user's prefix
            allowed_prefixes = [
                f"{self.config.s3_data_prefix}/{user_prefix}/",
                f"{self.config.s3_models_prefix}/{user_prefix}/",
                f"predictions/{user_prefix}/"
            ]

            if not any(s3_key.startswith(prefix) for prefix in allowed_prefixes):
                raise S3Error(
                    message=f"Write access denied: {s3_key} does not belong to user {user_id}",
                    bucket=self.config.s3_bucket,
                    key=s3_key,
                    error_code="AccessDenied"
                )

        # Validate read operations
        elif operation == 'read':
            # Read: can read own data, but not other users' data
            if user_prefix not in s3_key:
                # Check if trying to read another user's data
                # Pattern: user_{digits}/
                other_user_pattern = r'user_(\d+)/'
                match = re.search(other_user_pattern, s3_key)

                if match:
                    other_user_id = int(match.group(1))
                    if other_user_id != user_id:
                        raise S3Error(
                            message=f"Read access denied: {s3_key} belongs to another user",
                            bucket=self.config.s3_bucket,
                            key=s3_key,
                            error_code="AccessDenied"
                        )

        return True

    def enable_s3_access_logging(
        self,
        s3_client: Any,
        log_bucket: str
    ) -> None:
        """
        Enable S3 access logging for audit trail.

        Access logs provide:
        - Complete record of all S3 requests
        - Requester information
        - Operation details
        - Request time

        Args:
            s3_client: boto3 S3 client
            log_bucket: Bucket to store access logs

        Raises:
            CloudConfigurationError: If access logging configuration fails

        Example:
            >>> import boto3
            >>> s3_client = boto3.client('s3')
            >>> security_manager.enable_s3_access_logging(
            ...     s3_client,
            ...     log_bucket="ml-agent-logs"
            ... )
        """
        try:
            s3_client.put_bucket_logging(
                Bucket=self.config.s3_bucket,
                BucketLoggingStatus={
                    'LoggingEnabled': {
                        'TargetBucket': log_bucket,
                        'TargetPrefix': f'access-logs/{self.config.s3_bucket}/'
                    }
                }
            )
        except ClientError as e:
            raise CloudConfigurationError(
                f"Failed to enable access logging: {e}",
                config_key="s3_access_logging"
            )

    def audit_log_operation(
        self,
        user_id: int,
        operation: str,
        resource: str,
        success: bool,
        **metadata: Any
    ) -> None:
        """
        Log security-relevant operation to audit trail.

        Creates structured log entry with:
        - Timestamp
        - User ID
        - Operation type
        - Resource identifier
        - Success status
        - Additional metadata

        Logs are stored in data/logs/cloud_audit.json

        Args:
            user_id: User performing operation
            operation: Operation type (cloud_training, cloud_prediction, s3_upload, etc.)
            resource: Resource identifier (instance_id, s3_key, etc.)
            success: Whether operation succeeded
            **metadata: Additional context (cost, instance_type, duration, etc.)

        Example:
            >>> security_manager.audit_log_operation(
            ...     user_id=12345,
            ...     operation="cloud_training",
            ...     resource="i-1234567890abcdef0",
            ...     success=True,
            ...     instance_type="c5.xlarge",
            ...     cost=0.17,
            ...     duration=300
            ... )
        """
        # Build log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'operation': operation,
            'resource': resource,
            'success': success,
            **metadata
        }

        # Ensure audit log directory exists
        audit_log_file = Path("data/logs/cloud_audit.json")
        audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing logs or create new list
        if audit_log_file.exists():
            with open(audit_log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        # Append new log entry
        logs.append(log_entry)

        # Write updated logs
        with open(audit_log_file, 'w') as f:
            json.dump(logs, f, indent=2)


class RunPodSecurityManager:
    """
    Simplified security manager for RunPod (no IAM policies needed).

    RunPod provides simpler security model:
    - No IAM roles or policies (API key-based authentication)
    - No bucket policies (network volumes are private by default)
    - No encryption configuration (RunPod handles encryption)
    - Path-based user isolation (similar to AWS but simpler)
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize RunPodSecurityManager.

        Args:
            config: RunPodConfig instance

        Example:
            >>> from src.cloud.runpod_config import RunPodConfig
            >>> config = RunPodConfig.from_yaml("config.yaml")
            >>> security_manager = RunPodSecurityManager(config)
        """
        self.config = config

    def validate_user_storage_access(
        self,
        storage_key: str,
        user_id: int,
        operation: str
    ) -> bool:
        """
        Validate user can access storage path.

        Enforces user isolation:
        - Users can only access paths with user_{user_id} prefix
        - Read operations: can read own data, blocked from other users
        - Write operations: only allowed for user's own prefix

        Args:
            storage_key: Storage object key to validate
            user_id: User ID requesting access
            operation: 'read' or 'write'

        Returns:
            bool: True if access allowed

        Raises:
            ValueError: If access denied

        Example:
            >>> security_manager.validate_user_storage_access(
            ...     storage_key="datasets/user_12345/data.csv",
            ...     user_id=12345,
            ...     operation="write"
            ... )
            True
        """
        user_prefix = f"user_{user_id}"

        # Validate write operations
        if operation == 'write':
            # Write: must be in user's prefix
            allowed_prefixes = [
                f"{self.config.data_prefix}/{user_prefix}/",
                f"{self.config.models_prefix}/{user_prefix}/",
                f"predictions/{user_prefix}/",
                f"{self.config.results_prefix}/{user_prefix}/"
            ]

            if not any(storage_key.startswith(prefix) for prefix in allowed_prefixes):
                raise ValueError(
                    f"Write access denied: {storage_key} does not belong to user {user_id}. "
                    f"Allowed prefixes: {allowed_prefixes}"
                )

        # Validate read operations
        elif operation == 'read':
            # Read: can read own data, but not other users' data
            if user_prefix not in storage_key:
                # Check if trying to read another user's data
                # Pattern: user_{digits}/
                other_user_pattern = r'user_(\d+)/'
                match = re.search(other_user_pattern, storage_key)

                if match:
                    other_user_id = int(match.group(1))
                    if other_user_id != user_id:
                        raise ValueError(
                            f"Read access denied: {storage_key} belongs to user {other_user_id}, "
                            f"not user {user_id}"
                        )

        return True

    def audit_log_operation(
        self,
        user_id: int,
        operation: str,
        resource: str,
        success: bool,
        **metadata: Any
    ) -> None:
        """
        Log security-relevant operation to audit trail.

        Creates structured log entry with:
        - Timestamp
        - User ID
        - Operation type
        - Resource identifier
        - Success status
        - Provider (RunPod)
        - Additional metadata

        Logs are stored in data/logs/cloud_audit.json

        Args:
            user_id: User performing operation
            operation: Operation type (runpod_training, runpod_prediction, storage_upload, etc.)
            resource: Resource identifier (pod_id, storage_key, endpoint_id, etc.)
            success: Whether operation succeeded
            **metadata: Additional context (cost, gpu_type, duration, etc.)

        Example:
            >>> security_manager.audit_log_operation(
            ...     user_id=12345,
            ...     operation="runpod_training",
            ...     resource="pod-abc123",
            ...     success=True,
            ...     gpu_type="NVIDIA RTX A5000",
            ...     cost=0.145,
            ...     duration=1800
            ... )
        """
        # Build log entry with provider tag
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'operation': operation,
            'resource': resource,
            'success': success,
            'provider': 'runpod',  # Tag RunPod operations
            **metadata
        }

        # Ensure audit log directory exists
        audit_log_file = Path("data/logs/cloud_audit.json")
        audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing logs or create new list
        if audit_log_file.exists():
            with open(audit_log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        # Append new log entry
        logs.append(log_entry)

        # Write updated logs
        with open(audit_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
