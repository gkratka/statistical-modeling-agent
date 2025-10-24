# Cloud ML Workflows - Detailed Sub-Tasks

**Phase 1: Cloud ML Implementation**
**Date**: 2025-10-23
**Status**: Planning Complete - Ready for Implementation

---

## Overview

This document provides actionable sub-tasks for implementing cloud-based ML training and prediction using AWS infrastructure. Each sub-task is designed to be implementable by a junior developer with clear file references and testing steps.

---

## Relevant Files

### New Files to Create

**AWS Infrastructure Layer**:
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/aws_client.py` - AWS client initialization and health checks
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/aws_config.py` - Cloud configuration dataclasses
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py` - S3 operations (upload, download, list)
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py` - EC2 Spot instance management
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/lambda_manager.py` - Lambda prediction service
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py` - Cost calculation and tracking

**Exception Handling**:
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/exceptions.py` - Cloud-specific exception hierarchy

**Telegram Integration**:
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_training_handlers.py` - Cloud training workflow
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_prediction_handlers.py` - Cloud prediction workflow
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/messages/cloud_messages.py` - Cloud workflow messages

**Security**:
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py` - IAM policies, bucket policies, encryption

**Scripts**:
- `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/package_lambda.sh` - Lambda deployment packaging
- `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/setup_aws.py` - AWS infrastructure setup script

**Lambda Functions**:
- `/Users/gkratka/Documents/statistical-modeling-agent/lambda/prediction_handler.py` - Lambda prediction entry point
- `/Users/gkratka/Documents/statistical-modeling-agent/lambda/requirements.txt` - Lambda dependencies

**Tests**:
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_aws_client.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_s3_manager.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_ec2_manager.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_lambda_manager.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_cost_tracker.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/integration/test_cloud_training_workflow.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/integration/test_cloud_prediction_workflow.py`

### Files to Modify

- `/Users/gkratka/Documents/statistical-modeling-agent/config/config.yaml` - Add cloud configuration section
- `/Users/gkratka/Documents/statistical-modeling-agent/.env.example` - Add AWS credential placeholders
- `/Users/gkratka/Documents/statistical-modeling-agent/src/core/state_manager.py` - Add cloud workflow states
- `/Users/gkratka/Documents/statistical-modeling-agent/src/utils/exceptions.py` - Import cloud exceptions
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/telegram_bot.py` - Register cloud handlers
- `/Users/gkratka/Documents/statistical-modeling-agent/requirements.txt` - Add boto3, watchtower

---

## 1.0 AWS Infrastructure Foundation

**Goal**: Establish AWS client initialization, configuration management, and health checks.

### Sub-Tasks

- [x] **1.1 Create cloud exception hierarchy**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/exceptions.py`
  - Define exception classes:
    - `CloudError(AgentError)` - Base cloud exception
    - `AWSError(CloudError)` - General AWS errors
    - `S3Error(AWSError)` - S3-specific errors
    - `EC2Error(AWSError)` - EC2-specific errors
    - `LambdaError(AWSError)` - Lambda-specific errors
    - `CostTrackingError(CloudError)` - Cost tracking errors
    - `CloudConfigurationError(CloudError)` - Configuration errors
  - Each exception should include:
    - `message: str` - Human-readable error
    - `service: str` - AWS service name (s3, ec2, lambda)
    - `error_code: str` - AWS error code (if available)
    - `request_id: str` - AWS request ID for debugging
  - Import in `/Users/gkratka/Documents/statistical-modeling-agent/src/utils/exceptions.py`:
    ```python
    from src.cloud.exceptions import (
        CloudError, AWSError, S3Error, EC2Error, LambdaError,
        CostTrackingError, CloudConfigurationError
    )
    ```

- [x] **1.2 Create cloud configuration dataclasses**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/aws_config.py`
  - Define `CloudConfig` dataclass with fields:
    ```python
    @dataclass
    class CloudConfig:
        # AWS Credentials
        aws_region: str = "us-east-1"
        aws_access_key_id: str = ""  # Load from env
        aws_secret_access_key: str = ""  # Load from env

        # S3 Configuration
        s3_bucket_name: str = ""  # e.g., "ml-agent-datasets-{account_id}"
        s3_model_prefix: str = "models"
        s3_dataset_prefix: str = "datasets"
        s3_lifecycle_days: int = 90  # Auto-delete after 90 days

        # EC2 Configuration
        ec2_instance_type: str = "m5.xlarge"  # Default instance type
        ec2_spot_max_price: float = 0.5  # Max $/hour for Spot instances
        ec2_ami_id: str = ""  # Deep Learning AMI ID
        ec2_key_pair_name: str = "ml-agent-key"
        ec2_security_group_id: str = ""

        # Lambda Configuration
        lambda_function_name: str = "ml-agent-prediction"
        lambda_memory_mb: int = 3008  # Max Lambda memory
        lambda_timeout_seconds: int = 900  # 15 minutes max
        lambda_layer_arn: str = ""  # ARN for dependencies layer

        # Cost Limits
        max_training_cost_usd: float = 10.0
        max_prediction_cost_usd: float = 1.0
        warn_threshold_percent: float = 0.8  # Warn at 80% of limit

        # Security
        encryption_key_id: str = ""  # KMS key ID for encryption
        iam_role_arn: str = ""  # IAM role for EC2/Lambda

        @classmethod
        def from_yaml(cls, config_path: str) -> "CloudConfig":
            """Load configuration from YAML file."""
            pass

        def validate(self) -> None:
            """Validate configuration completeness."""
            pass
    ```
  - Add validation method to check required fields
  - Add `from_env()` class method to load from environment variables

- [x] **1.3 Update config.yaml with cloud section**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/config/config.yaml`
  - Add cloud configuration section:
    ```yaml
    # Cloud ML Configuration (AWS)
    cloud:
      enabled: false  # Feature flag: enable/disable cloud workflows
      provider: aws   # Currently only AWS supported

      aws:
        region: us-east-1
        account_id: ${AWS_ACCOUNT_ID}  # From environment

        s3:
          bucket_name: ml-agent-${AWS_ACCOUNT_ID}
          model_prefix: models
          dataset_prefix: datasets
          lifecycle_days: 90
          versioning_enabled: true
          encryption: AES256

        ec2:
          default_instance_type: m5.xlarge
          spot_max_price: 0.50
          ami_id: ${EC2_AMI_ID}  # Deep Learning AMI
          key_pair_name: ml-agent-key
          security_group_id: ${EC2_SECURITY_GROUP_ID}
          subnet_id: ${EC2_SUBNET_ID}

        lambda:
          function_name: ml-agent-prediction
          memory_mb: 3008
          timeout_seconds: 900
          runtime: python3.11
          layer_arn: ${LAMBDA_LAYER_ARN}

        cost_limits:
          max_training_cost_usd: 10.0
          max_prediction_cost_usd: 1.0
          warn_threshold_percent: 0.8
          monthly_budget_usd: 100.0

        security:
          encryption_key_id: ${KMS_KEY_ID}
          iam_role_arn: ${IAM_ROLE_ARN}
          enable_access_logging: true
    ```

- [x] **1.4 Update .env.example with AWS variables**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/.env.example`
  - Add cloud-specific environment variables:
    ```bash
    # AWS Configuration (optional - only if cloud workflows enabled)
    AWS_ACCESS_KEY_ID=your_aws_access_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret_key
    AWS_REGION=us-east-1
    AWS_ACCOUNT_ID=123456789012

    # EC2 Configuration
    EC2_AMI_ID=ami-0123456789abcdef0  # Deep Learning AMI
    EC2_SECURITY_GROUP_ID=sg-0123456789abcdef0
    EC2_SUBNET_ID=subnet-0123456789abcdef0

    # Lambda Configuration
    LAMBDA_LAYER_ARN=arn:aws:lambda:us-east-1:123456789012:layer:ml-dependencies:1

    # Security
    KMS_KEY_ID=alias/ml-agent-encryption
    IAM_ROLE_ARN=arn:aws:iam::123456789012:role/ml-agent-role
    ```

- [x] **1.5 Create AWS client initialization module**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/aws_client.py`
  - Implement `AWSClient` class:
    ```python
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    from typing import Optional

    class AWSClient:
        """Centralized AWS client management."""

        def __init__(self, config: CloudConfig):
            self.config = config
            self._s3_client: Optional[boto3.client] = None
            self._ec2_client: Optional[boto3.client] = None
            self._lambda_client: Optional[boto3.client] = None
            self._cloudwatch_client: Optional[boto3.client] = None

        @property
        def s3(self) -> boto3.client:
            """Lazy-load S3 client."""
            if self._s3_client is None:
                self._s3_client = boto3.client(
                    's3',
                    region_name=self.config.aws_region,
                    aws_access_key_id=self.config.aws_access_key_id,
                    aws_secret_access_key=self.config.aws_secret_access_key
                )
            return self._s3_client

        @property
        def ec2(self) -> boto3.client:
            """Lazy-load EC2 client."""
            # Similar to S3
            pass

        @property
        def lambda_(self) -> boto3.client:
            """Lazy-load Lambda client."""
            # Similar to S3
            pass

        @property
        def cloudwatch(self) -> boto3.client:
            """Lazy-load CloudWatch Logs client."""
            # Similar to S3
            pass

        def health_check(self) -> dict[str, bool]:
            """Check connectivity to all AWS services."""
            results = {}

            # S3 health check
            try:
                self.s3.list_buckets()
                results['s3'] = True
            except (ClientError, BotoCoreError):
                results['s3'] = False

            # EC2 health check
            try:
                self.ec2.describe_instances(MaxResults=5)
                results['ec2'] = True
            except (ClientError, BotoCoreError):
                results['ec2'] = False

            # Lambda health check
            try:
                self.lambda_.list_functions(MaxItems=1)
                results['lambda'] = True
            except (ClientError, BotoCoreError):
                results['lambda'] = False

            return results

        def validate_configuration(self) -> None:
            """Validate AWS configuration and credentials."""
            # Check bucket exists
            # Check IAM role exists
            # Check AMI exists
            # Check security group exists
            pass
    ```

- [x] **1.6 Write unit tests for AWS client**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_aws_client.py`
  - Test cases:
    - `test_aws_client_initialization` - Client creates successfully with valid config
    - `test_lazy_loading_clients` - Clients only created when accessed
    - `test_health_check_all_services_up` - All services return True
    - `test_health_check_service_down` - Service failure returns False
    - `test_validate_configuration_success` - Valid config passes
    - `test_validate_configuration_missing_bucket` - Missing bucket raises error
    - `test_credential_error_handling` - Invalid credentials raise AWSError
  - Use `moto` library for AWS mocking:
    ```python
    from moto import mock_s3, mock_ec2, mock_lambda

    @mock_s3
    def test_s3_health_check():
        # Test implementation
        pass
    ```

---

## 2.0 S3 Dataset & Model Storage System

**Goal**: Implement secure S3 operations for dataset upload, model storage, and retrieval.

### Sub-Tasks

- [x] **2.1 Create S3Manager class with upload functionality**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py`
  - Implement upload methods:
    ```python
    class S3Manager:
        """S3 operations for datasets and models."""

        def __init__(self, aws_client: AWSClient, config: CloudConfig):
            self.client = aws_client.s3
            self.config = config
            self.bucket = config.s3_bucket_name

        def upload_dataset(
            self,
            user_id: int,
            file_path: str,
            dataset_name: Optional[str] = None
        ) -> str:
            """
            Upload dataset to S3 with user isolation.

            Args:
                user_id: User ID for path isolation
                file_path: Local path to dataset file
                dataset_name: Optional custom name (default: filename)

            Returns:
                S3 URI: s3://bucket/datasets/user_{user_id}/dataset_name

            Raises:
                S3Error: Upload failed
            """
            # Generate S3 key with user prefix
            s3_key = self._generate_dataset_key(user_id, dataset_name or Path(file_path).name)

            # Validate path
            if not Path(file_path).exists():
                raise S3Error(f"File not found: {file_path}")

            # Get file size
            file_size = Path(file_path).stat().st_size

            # Use multipart upload for files >5MB
            if file_size > 5 * 1024 * 1024:
                return self._multipart_upload(file_path, s3_key)
            else:
                return self._simple_upload(file_path, s3_key)

        def _simple_upload(self, file_path: str, s3_key: str) -> str:
            """Upload file <5MB using simple upload."""
            try:
                self.client.upload_file(
                    Filename=file_path,
                    Bucket=self.bucket,
                    Key=s3_key,
                    ExtraArgs={
                        'ServerSideEncryption': 'AES256',
                        'Metadata': {
                            'uploaded_at': datetime.now().isoformat(),
                            'original_filename': Path(file_path).name
                        }
                    }
                )
                return f"s3://{self.bucket}/{s3_key}"
            except ClientError as e:
                raise S3Error(f"Upload failed: {e}", error_code=e.response['Error']['Code'])

        def _multipart_upload(self, file_path: str, s3_key: str) -> str:
            """Upload large file using multipart upload."""
            # Implementation: boto3.s3.transfer.TransferConfig
            pass

        def _generate_dataset_key(self, user_id: int, filename: str) -> str:
            """Generate S3 key with user isolation."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.config.s3_dataset_prefix}/user_{user_id}/{timestamp}_{filename}"
    ```

- [x] **2.2 Implement S3 model save/load with versioning**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py`
  - Add model operations to `S3Manager`:
    ```python
    def save_model(
        self,
        user_id: int,
        model_id: str,
        model_dir: Path
    ) -> str:
        """
        Upload trained model directory to S3.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            model_dir: Local directory containing model files

        Returns:
            S3 URI to model directory
        """
        # Generate S3 prefix
        s3_prefix = f"{self.config.s3_model_prefix}/user_{user_id}/{model_id}"

        # Upload all files in model directory
        uploaded_files = []
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                self._simple_upload(str(file_path), s3_key)
                uploaded_files.append(s3_key)

        # Save manifest file listing all model files
        manifest = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': datetime.now().isoformat(),
            'files': uploaded_files,
            'version': 1
        }
        self._upload_json(f"{s3_prefix}/manifest.json", manifest)

        return f"s3://{self.bucket}/{s3_prefix}"

    def load_model(
        self,
        user_id: int,
        model_id: str,
        local_dir: Path
    ) -> Path:
        """
        Download model from S3 to local directory.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            local_dir: Local directory to download to

        Returns:
            Path to downloaded model directory
        """
        s3_prefix = f"{self.config.s3_model_prefix}/user_{user_id}/{model_id}"

        # Download manifest first
        manifest_key = f"{s3_prefix}/manifest.json"
        manifest = self._download_json(manifest_key)

        # Download all files listed in manifest
        local_model_dir = local_dir / model_id
        local_model_dir.mkdir(parents=True, exist_ok=True)

        for s3_key in manifest['files']:
            relative_path = s3_key.replace(f"{s3_prefix}/", "")
            local_path = local_model_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=str(local_path)
            )

        return local_model_dir
    ```

- [x] **2.3 Implement S3 path validation and security**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py`
  - Add validation methods:
    ```python
    def validate_s3_path(self, s3_uri: str, user_id: int) -> bool:
        """
        Validate S3 path belongs to user.

        Args:
            s3_uri: S3 URI to validate (s3://bucket/key)
            user_id: User ID attempting access

        Returns:
            True if user owns the path

        Raises:
            S3Error: Path validation failed
        """
        # Parse S3 URI using regex
        pattern = r"s3://([^/]+)/(.+)"
        match = re.match(pattern, s3_uri)
        if not match:
            raise S3Error(f"Invalid S3 URI format: {s3_uri}")

        bucket, key = match.groups()

        # Check bucket matches config
        if bucket != self.bucket:
            raise S3Error(f"Bucket mismatch: {bucket} != {self.bucket}")

        # Check user prefix isolation
        expected_prefix = f"user_{user_id}"
        if not key.startswith(f"{self.config.s3_dataset_prefix}/{expected_prefix}") and \
           not key.startswith(f"{self.config.s3_model_prefix}/{expected_prefix}"):
            raise S3Error(f"Access denied: path does not belong to user {user_id}")

        return True

    def list_user_datasets(self, user_id: int) -> list[dict[str, Any]]:
        """List all datasets for a user."""
        prefix = f"{self.config.s3_dataset_prefix}/user_{user_id}/"

        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        datasets = []
        for obj in response.get('Contents', []):
            datasets.append({
                'key': obj['Key'],
                's3_uri': f"s3://{self.bucket}/{obj['Key']}",
                'size_mb': obj['Size'] / (1024 * 1024),
                'last_modified': obj['LastModified'].isoformat(),
                'filename': Path(obj['Key']).name
            })

        return datasets

    def list_user_models(self, user_id: int) -> list[dict[str, Any]]:
        """List all models for a user."""
        # Similar to list_user_datasets
        pass
    ```

- [x] **2.4 Implement pre-signed URL generation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py`
  - Add pre-signed URL methods:
    ```python
    def generate_presigned_download_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate pre-signed URL for downloading file.

        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default: 1 hour)

        Returns:
            Pre-signed URL string
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise S3Error(f"Failed to generate presigned URL: {e}")

    def generate_presigned_upload_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """Generate pre-signed URL for uploading file."""
        # Similar to download URL
        pass
    ```

- [x] **2.5 Implement S3 lifecycle policy setup**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py`
  - Add lifecycle management:
    ```python
    def configure_bucket_lifecycle(self) -> None:
        """
        Configure S3 bucket lifecycle policy for auto-deletion.

        Policy:
        - Delete datasets after 90 days
        - Transition models to Glacier after 30 days
        - Delete models after 180 days
        """
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'delete-old-datasets',
                    'Filter': {'Prefix': f"{self.config.s3_dataset_prefix}/"},
                    'Status': 'Enabled',
                    'Expiration': {'Days': self.config.s3_lifecycle_days}
                },
                {
                    'Id': 'archive-old-models',
                    'Filter': {'Prefix': f"{self.config.s3_model_prefix}/"},
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'GLACIER'
                        }
                    ],
                    'Expiration': {'Days': 180}
                }
            ]
        }

        try:
            self.client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration=lifecycle_config
            )
        except ClientError as e:
            raise S3Error(f"Failed to configure lifecycle: {e}")
    ```

- [x] **2.6 Write unit tests for S3Manager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_s3_manager.py`
  - Test cases:
    - `test_upload_dataset_small_file` - Upload file <5MB successfully
    - `test_upload_dataset_large_file` - Multipart upload for file >5MB
    - `test_upload_dataset_file_not_found` - Raises S3Error for missing file
    - `test_save_model_all_files` - All model files uploaded with manifest
    - `test_load_model_downloads_all_files` - All files downloaded correctly
    - `test_validate_s3_path_valid_user` - Valid user path passes
    - `test_validate_s3_path_wrong_user` - Different user path raises error
    - `test_validate_s3_path_invalid_format` - Malformed URI raises error
    - `test_list_user_datasets` - Returns correct dataset list
    - `test_generate_presigned_url` - URL generated with correct expiration
    - `test_configure_lifecycle_policy` - Lifecycle rules applied correctly

---

## 3.0 Cloud Training Workflow with EC2 Spot Instances

**Goal**: Implement EC2 Spot instance management for cost-effective cloud training.

### Sub-Tasks

- [x] **3.1 Create EC2Manager class with instance selection**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Implement instance type selection logic:
    ```python
    class EC2Manager:
        """EC2 Spot instance management for ML training."""

        def __init__(self, aws_client: AWSClient, config: CloudConfig):
            self.client = aws_client.ec2
            self.cloudwatch = aws_client.cloudwatch
            self.config = config

        def select_instance_type(
            self,
            dataset_size_mb: float,
            model_type: str,
            estimated_training_time_minutes: int
        ) -> str:
            """
            Select optimal instance type based on workload.

            Decision Matrix:
            - Dataset <100MB, simple model (linear/logistic): t3.medium
            - Dataset <1GB, tree models: m5.large
            - Dataset >1GB, neural networks: m5.xlarge or p3.2xlarge (GPU)

            Args:
                dataset_size_mb: Dataset size in megabytes
                model_type: Type of model (linear, random_forest, etc.)
                estimated_training_time_minutes: Expected training duration

            Returns:
                EC2 instance type string
            """
            # Simple models
            if model_type in ['linear', 'logistic', 'ridge', 'lasso']:
                if dataset_size_mb < 100:
                    return 't3.medium'
                else:
                    return 'm5.large'

            # Tree-based models
            elif model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
                if dataset_size_mb < 1000:
                    return 'm5.large'
                else:
                    return 'm5.xlarge'

            # Neural networks - consider GPU
            elif 'neural' in model_type or 'mlp' in model_type:
                if dataset_size_mb > 5000:
                    return 'p3.2xlarge'  # GPU instance
                else:
                    return 'm5.xlarge'

            # Default fallback
            return self.config.ec2_instance_type
    ```

- [x] **3.2 Implement Spot Instance launch configuration**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Add Spot request methods:
    ```python
    def launch_spot_instance(
        self,
        instance_type: str,
        user_data_script: str,
        tags: dict[str, str]
    ) -> dict[str, Any]:
        """
        Launch EC2 Spot instance with training script.

        Args:
            instance_type: EC2 instance type
            user_data_script: Bash script to run on instance startup
            tags: Tags for instance identification

        Returns:
            Dict with instance_id, spot_request_id, launch_time

        Raises:
            EC2Error: Launch failed
        """
        try:
            response = self.client.request_spot_instances(
                SpotPrice=str(self.config.ec2_spot_max_price),
                InstanceCount=1,
                Type='one-time',
                LaunchSpecification={
                    'ImageId': self.config.ec2_ami_id,
                    'InstanceType': instance_type,
                    'KeyName': self.config.ec2_key_pair_name,
                    'SecurityGroupIds': [self.config.ec2_security_group_id],
                    'SubnetId': self.config.ec2_subnet_id,
                    'UserData': user_data_script,
                    'IamInstanceProfile': {
                        'Arn': self.config.iam_role_arn
                    },
                    'BlockDeviceMappings': [
                        {
                            'DeviceName': '/dev/xvda',
                            'Ebs': {
                                'VolumeSize': 50,  # 50GB root volume
                                'VolumeType': 'gp3',
                                'DeleteOnTermination': True,
                                'Encrypted': True
                            }
                        }
                    ]
                },
                TagSpecifications=[
                    {
                        'ResourceType': 'spot-instances-request',
                        'Tags': [{'Key': k, 'Value': v} for k, v in tags.items()]
                    }
                ]
            )

            spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']

            # Wait for instance to be assigned
            instance_id = self._wait_for_instance_assignment(spot_request_id)

            return {
                'instance_id': instance_id,
                'spot_request_id': spot_request_id,
                'instance_type': instance_type,
                'launch_time': datetime.now().isoformat()
            }

        except ClientError as e:
            raise EC2Error(f"Failed to launch Spot instance: {e}")

    def _wait_for_instance_assignment(
        self,
        spot_request_id: str,
        timeout_seconds: int = 300
    ) -> str:
        """Wait for Spot request to be fulfilled."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            response = self.client.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )

            request = response['SpotInstanceRequests'][0]

            if request['State'] == 'active' and 'InstanceId' in request:
                return request['InstanceId']
            elif request['State'] == 'failed':
                raise EC2Error(f"Spot request failed: {request.get('Fault', {}).get('Message', 'Unknown')}")

            time.sleep(5)

        raise EC2Error(f"Spot request timeout after {timeout_seconds}s")
    ```

- [x] **3.3 Generate UserData training script**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Add script generation method:
    ```python
    def generate_training_userdata(
        self,
        s3_dataset_uri: str,
        model_type: str,
        target_column: str,
        feature_columns: list[str],
        hyperparameters: dict[str, Any],
        s3_output_uri: str
    ) -> str:
        """
        Generate bash UserData script for EC2 training.

        The script will:
        1. Download dataset from S3
        2. Install Python dependencies
        3. Run training script
        4. Upload trained model to S3
        5. Send CloudWatch logs
        6. Terminate instance

        Returns:
            Base64-encoded bash script
        """
        script = f'''#!/bin/bash
set -e  # Exit on error

# CloudWatch logging setup
LOG_GROUP="/aws/ec2/ml-training"
LOG_STREAM="instance-$(ec2-metadata --instance-id | cut -d' ' -f2)"

log_to_cloudwatch() {{
    echo "$1" | tee -a /var/log/training.log
    # Send to CloudWatch (requires awslogs agent)
}}

log_to_cloudwatch "Starting ML training job..."

# Install Python dependencies
log_to_cloudwatch "Installing dependencies..."
pip install pandas scikit-learn joblib boto3

# Download dataset from S3
log_to_cloudwatch "Downloading dataset from {s3_dataset_uri}..."
aws s3 cp {s3_dataset_uri} /tmp/dataset.csv

# Create training script
cat > /tmp/train.py << 'PYTHON_SCRIPT'
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Import based on model_type
import boto3
import sys

# Load data
df = pd.read_csv('/tmp/dataset.csv')

# Split features and target
X = df[{feature_columns}]
y = df['{target_column}']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(**{hyperparameters})
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R2 Score: {{score}}")

# Save model
joblib.dump(model, '/tmp/model.pkl')

# Upload to S3
s3 = boto3.client('s3')
bucket, key = '{s3_output_uri}'.replace('s3://', '').split('/', 1)
s3.upload_file('/tmp/model.pkl', bucket, f'{{key}}/model.pkl')

print("Training complete!")
PYTHON_SCRIPT

# Run training
log_to_cloudwatch "Running training script..."
python3 /tmp/train.py 2>&1 | tee -a /var/log/training.log

# Upload logs to S3
log_to_cloudwatch "Uploading logs..."
aws s3 cp /var/log/training.log {s3_output_uri}/training.log

# Self-terminate instance
log_to_cloudwatch "Terminating instance..."
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
'''

        # Base64 encode for UserData
        import base64
        return base64.b64encode(script.encode()).decode()
    ```

- [x] **3.4 Implement CloudWatch log polling**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Add log streaming methods:
    ```python
    def poll_training_logs(
        self,
        instance_id: str,
        log_group: str = "/aws/ec2/ml-training"
    ) -> AsyncIterator[str]:
        """
        Poll CloudWatch logs for training progress.

        Yields:
            Log lines as they become available
        """
        log_stream = f"instance-{instance_id}"
        next_token = None

        while True:
            try:
                params = {
                    'logGroupName': log_group,
                    'logStreamName': log_stream,
                    'startFromHead': True
                }
                if next_token:
                    params['nextToken'] = next_token

                response = self.cloudwatch.get_log_events(**params)

                for event in response['events']:
                    yield event['message']

                # Check if more logs available
                if response['nextForwardToken'] == next_token:
                    break  # No new logs

                next_token = response['nextForwardToken']

                # Wait before next poll
                await asyncio.sleep(5)

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Log stream not created yet
                    await asyncio.sleep(10)
                else:
                    raise EC2Error(f"Failed to poll logs: {e}")

    def parse_training_progress(self, log_line: str) -> Optional[dict[str, Any]]:
        """
        Parse training progress from log line.

        Expected formats:
        - "Epoch 10/100, Loss: 0.234"
        - "Training progress: 45%"
        - "Model R2 Score: 0.85"

        Returns:
            Dict with progress info, or None if not a progress line
        """
        # Regex patterns for different progress indicators
        patterns = {
            'epoch': r'Epoch (\d+)/(\d+)',
            'loss': r'Loss: ([\d.]+)',
            'score': r'Score: ([\d.]+)',
            'percent': r'progress: (\d+)%'
        }

        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, log_line)
            if match:
                result[key] = match.group(1)

        return result if result else None
    ```

- [x] **3.5 Implement Spot interruption handling**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Add interruption detection and checkpoint logic:
    ```python
    def monitor_spot_interruption(self, instance_id: str) -> bool:
        """
        Check for Spot instance interruption notice.

        AWS gives 2-minute warning before terminating Spot instances.

        Returns:
            True if interruption notice detected
        """
        try:
            # Check instance metadata for interruption notice
            response = self.client.describe_instances(InstanceIds=[instance_id])

            if not response['Reservations']:
                return True  # Instance already terminated

            instance = response['Reservations'][0]['Instances'][0]

            # Check for spot interruption tag or state
            if instance['State']['Name'] in ['shutting-down', 'terminated']:
                return True

            # Check for interruption notice (typically in instance metadata)
            # In production, the EC2 instance itself would check the metadata endpoint
            # http://169.254.169.254/latest/meta-data/spot/instance-action

            return False

        except ClientError as e:
            raise EC2Error(f"Failed to check interruption status: {e}")

    def handle_spot_interruption(
        self,
        instance_id: str,
        s3_checkpoint_uri: str
    ) -> None:
        """
        Handle Spot interruption by saving checkpoint.

        This method would be called from within the EC2 instance's training script
        when it detects an interruption notice.
        """
        # Signal instance to save checkpoint to S3
        # Then gracefully terminate
        pass
    ```

- [x] **3.6 Implement auto-termination logic**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Add instance cleanup methods:
    ```python
    def terminate_instance(self, instance_id: str) -> None:
        """Terminate EC2 instance."""
        try:
            self.client.terminate_instances(InstanceIds=[instance_id])
        except ClientError as e:
            raise EC2Error(f"Failed to terminate instance {instance_id}: {e}")

    def auto_terminate_on_completion(
        self,
        instance_id: str,
        log_group: str = "/aws/ec2/ml-training"
    ) -> None:
        """
        Monitor training logs and auto-terminate when complete.

        Termination triggers:
        - Log contains "Training complete!"
        - Instance idle for >10 minutes
        - Training exceeds max time (from config)
        """
        start_time = time.time()
        last_log_time = start_time

        async for log_line in self.poll_training_logs(instance_id, log_group):
            last_log_time = time.time()

            # Check for completion message
            if "Training complete!" in log_line or "Model saved to S3" in log_line:
                await asyncio.sleep(60)  # Wait 1 minute for cleanup
                self.terminate_instance(instance_id)
                return

            # Check for timeout
            if time.time() - start_time > self.config.max_training_time_seconds:
                self.terminate_instance(instance_id)
                raise EC2Error("Training timeout exceeded")

            # Check for idle timeout (no logs for 10 minutes)
            if time.time() - last_log_time > 600:
                self.terminate_instance(instance_id)
                raise EC2Error("Instance idle timeout")
    ```

- [x] **3.7 Write unit tests for EC2Manager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_ec2_manager.py`
  - Test cases:
    - `test_select_instance_type_small_dataset` - Returns t3.medium
    - `test_select_instance_type_large_neural_net` - Returns p3.2xlarge
    - `test_launch_spot_instance_success` - Instance launched with correct config
    - `test_launch_spot_instance_timeout` - Raises EC2Error after timeout
    - `test_generate_training_userdata` - Script contains all required commands
    - `test_poll_training_logs` - Yields log lines correctly
    - `test_parse_training_progress_epoch` - Parses epoch progress
    - `test_parse_training_progress_score` - Parses score metrics
    - `test_monitor_spot_interruption_active` - Returns False for running instance
    - `test_monitor_spot_interruption_terminating` - Returns True for terminating
    - `test_auto_terminate_on_completion` - Terminates after completion log
    - `test_auto_terminate_on_timeout` - Terminates after max time

---

## 4.0 Lambda-Based Cloud Prediction Service

**Goal**: Implement serverless prediction using AWS Lambda for cost-effective inference.

### Sub-Tasks

- [x] **4.1 Create Lambda prediction handler function**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/lambda/prediction_handler.py`
  - Implement Lambda entry point:
    ```python
    import json
    import boto3
    import pandas as pd
    import joblib
    from pathlib import Path
    from typing import Any

    s3_client = boto3.client('s3')

    def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
        """
        AWS Lambda handler for ML predictions.

        Event payload:
        {
            "model_s3_uri": "s3://bucket/models/user_123/model_id/model.pkl",
            "data_s3_uri": "s3://bucket/datasets/user_123/data.csv",
            "output_s3_uri": "s3://bucket/predictions/user_123/output.csv",
            "prediction_column_name": "prediction",
            "feature_columns": ["col1", "col2", "col3"]
        }

        Returns:
        {
            "statusCode": 200,
            "body": {
                "success": true,
                "predictions_s3_uri": "s3://...",
                "num_predictions": 1000,
                "execution_time_ms": 1234
            }
        }
        """
        import time
        start_time = time.time()

        try:
            # Parse input
            model_s3_uri = event['model_s3_uri']
            data_s3_uri = event['data_s3_uri']
            output_s3_uri = event['output_s3_uri']
            prediction_column = event.get('prediction_column_name', 'prediction')
            feature_columns = event.get('feature_columns')

            # Download model from S3
            model_local_path = '/tmp/model.pkl'
            bucket, key = parse_s3_uri(model_s3_uri)
            s3_client.download_file(bucket, key, model_local_path)
            model = joblib.load(model_local_path)

            # Download data from S3
            data_local_path = '/tmp/data.csv'
            bucket, key = parse_s3_uri(data_s3_uri)
            s3_client.download_file(bucket, key, data_local_path)
            df = pd.read_csv(data_local_path)

            # Select features
            if feature_columns:
                X = df[feature_columns]
            else:
                X = df

            # Make predictions
            predictions = model.predict(X)

            # Add predictions to dataframe
            df[prediction_column] = predictions

            # Save results to S3
            output_local_path = '/tmp/output.csv'
            df.to_csv(output_local_path, index=False)

            bucket, key = parse_s3_uri(output_s3_uri)
            s3_client.upload_file(output_local_path, bucket, key)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'predictions_s3_uri': output_s3_uri,
                    'num_predictions': len(predictions),
                    'execution_time_ms': execution_time_ms
                })
            }

        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
            }

    def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        parts = s3_uri.replace('s3://', '').split('/', 1)
        return parts[0], parts[1]
    ```

- [x] **4.2 Create Lambda requirements.txt**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/lambda/requirements.txt`
  - List Lambda dependencies:
    ```
    pandas==2.0.3
    scikit-learn==1.3.0
    joblib==1.3.2
    numpy==1.24.3
    boto3==1.28.25
    ```

- [x] **4.3 Create Lambda packaging script**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/package_lambda.sh`
  - Bash script to package Lambda deployment:
    ```bash
    #!/bin/bash
    set -e

    echo "Packaging Lambda function..."

    # Create clean directory
    rm -rf lambda_package
    mkdir -p lambda_package

    # Copy handler
    cp lambda/prediction_handler.py lambda_package/

    # Install dependencies to package directory
    pip install -r lambda/requirements.txt -t lambda_package/

    # Create deployment zip
    cd lambda_package
    zip -r ../lambda_deployment.zip .
    cd ..

    echo "Lambda package created: lambda_deployment.zip"
    echo "Size: $(du -h lambda_deployment.zip | cut -f1)"

    # If size >50MB, need to create Layer instead
    SIZE_MB=$(du -m lambda_deployment.zip | cut -f1)
    if [ $SIZE_MB -gt 50 ]; then
        echo "WARNING: Package >50MB. Creating Lambda Layer instead..."

        # Create layer package (dependencies only)
        mkdir -p lambda_layer/python
        pip install -r lambda/requirements.txt -t lambda_layer/python/
        cd lambda_layer
        zip -r ../lambda_layer.zip .
        cd ..

        # Create function package (handler only)
        cd lambda
        zip ../lambda_function.zip prediction_handler.py
        cd ..

        echo "Lambda Layer created: lambda_layer.zip ($(du -h lambda_layer.zip | cut -f1))"
        echo "Lambda Function created: lambda_function.zip ($(du -h lambda_function.zip | cut -f1))"
    fi
    ```
  - Make script executable: `chmod +x scripts/cloud/package_lambda.sh`

- [x] **4.4 Create LambdaManager class with invocation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/lambda_manager.py`
  - Implement Lambda operations:
    ```python
    class LambdaManager:
        """AWS Lambda management for ML predictions."""

        def __init__(self, aws_client: AWSClient, config: CloudConfig):
            self.client = aws_client.lambda_
            self.config = config

        def invoke_prediction(
            self,
            model_s3_uri: str,
            data_s3_uri: str,
            output_s3_uri: str,
            prediction_column_name: str,
            feature_columns: Optional[list[str]] = None
        ) -> dict[str, Any]:
            """
            Invoke Lambda function for prediction.

            Args:
                model_s3_uri: S3 URI to model file
                data_s3_uri: S3 URI to prediction data
                output_s3_uri: S3 URI for output file
                prediction_column_name: Name for prediction column
                feature_columns: Optional feature subset

            Returns:
                Prediction results dict

            Raises:
                LambdaError: Invocation failed
            """
            payload = {
                'model_s3_uri': model_s3_uri,
                'data_s3_uri': data_s3_uri,
                'output_s3_uri': output_s3_uri,
                'prediction_column_name': prediction_column_name,
                'feature_columns': feature_columns
            }

            try:
                response = self.client.invoke(
                    FunctionName=self.config.lambda_function_name,
                    InvocationType='RequestResponse',  # Synchronous
                    Payload=json.dumps(payload)
                )

                # Parse response
                result = json.loads(response['Payload'].read())

                if result['statusCode'] != 200:
                    error_body = json.loads(result['body'])
                    raise LambdaError(
                        f"Lambda prediction failed: {error_body.get('error', 'Unknown')}",
                        error_code=error_body.get('error_type', 'UnknownError')
                    )

                return json.loads(result['body'])

            except ClientError as e:
                raise LambdaError(f"Failed to invoke Lambda: {e}")

        def invoke_async(
            self,
            model_s3_uri: str,
            data_s3_uri: str,
            output_s3_uri: str,
            prediction_column_name: str,
            feature_columns: Optional[list[str]] = None
        ) -> str:
            """
            Invoke Lambda asynchronously for large predictions.

            Returns:
                Request ID for tracking
            """
            payload = {
                'model_s3_uri': model_s3_uri,
                'data_s3_uri': data_s3_uri,
                'output_s3_uri': output_s3_uri,
                'prediction_column_name': prediction_column_name,
                'feature_columns': feature_columns
            }

            try:
                response = self.client.invoke(
                    FunctionName=self.config.lambda_function_name,
                    InvocationType='Event',  # Asynchronous
                    Payload=json.dumps(payload)
                )

                return response['ResponseMetadata']['RequestId']

            except ClientError as e:
                raise LambdaError(f"Failed to invoke Lambda async: {e}")
    ```

- [x] **4.5 Implement Lambda deployment method**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/lambda_manager.py`
  - Add deployment methods:
    ```python
    def deploy_function(
        self,
        zip_file_path: str,
        layer_arn: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Deploy or update Lambda function.

        Args:
            zip_file_path: Path to deployment zip
            layer_arn: Optional Lambda Layer ARN

        Returns:
            Function configuration dict
        """
        with open(zip_file_path, 'rb') as f:
            zip_content = f.read()

        try:
            # Check if function exists
            try:
                self.client.get_function(FunctionName=self.config.lambda_function_name)
                function_exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    function_exists = False
                else:
                    raise

            if function_exists:
                # Update existing function
                response = self.client.update_function_code(
                    FunctionName=self.config.lambda_function_name,
                    ZipFile=zip_content
                )
            else:
                # Create new function
                create_params = {
                    'FunctionName': self.config.lambda_function_name,
                    'Runtime': 'python3.11',
                    'Role': self.config.iam_role_arn,
                    'Handler': 'prediction_handler.lambda_handler',
                    'Code': {'ZipFile': zip_content},
                    'Timeout': self.config.lambda_timeout_seconds,
                    'MemorySize': self.config.lambda_memory_mb
                }

                if layer_arn or self.config.lambda_layer_arn:
                    create_params['Layers'] = [layer_arn or self.config.lambda_layer_arn]

                response = self.client.create_function(**create_params)

            return response

        except ClientError as e:
            raise LambdaError(f"Failed to deploy Lambda function: {e}")

    def create_layer(
        self,
        layer_name: str,
        zip_file_path: str,
        description: str = "ML dependencies layer"
    ) -> str:
        """
        Create Lambda Layer for dependencies.

        Returns:
            Layer ARN
        """
        with open(zip_file_path, 'rb') as f:
            zip_content = f.read()

        try:
            response = self.client.publish_layer_version(
                LayerName=layer_name,
                Description=description,
                Content={'ZipFile': zip_content},
                CompatibleRuntimes=['python3.11']
            )

            return response['LayerVersionArn']

        except ClientError as e:
            raise LambdaError(f"Failed to create Lambda layer: {e}")
    ```

- [x] **4.6 Implement batch processing for large datasets**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/lambda_manager.py`
  - Add batch prediction logic:
    ```python
    def batch_predict(
        self,
        model_s3_uri: str,
        data_s3_uri: str,
        output_s3_uri: str,
        prediction_column_name: str,
        feature_columns: Optional[list[str]] = None,
        chunk_size: int = 10000
    ) -> dict[str, Any]:
        """
        Process large datasets in chunks via Lambda.

        For datasets >10k rows, split into chunks and invoke Lambda
        multiple times to avoid timeout.

        Args:
            model_s3_uri: S3 URI to model
            data_s3_uri: S3 URI to data (will be split)
            output_s3_uri: S3 URI for final output
            prediction_column_name: Column name for predictions
            feature_columns: Optional feature subset
            chunk_size: Rows per Lambda invocation

        Returns:
            Aggregated prediction results
        """
        # Download data to check size
        s3_manager = S3Manager(self.aws_client, self.config)
        local_data_path = s3_manager.download_file(data_s3_uri, '/tmp/data.csv')

        df = pd.read_csv(local_data_path)
        total_rows = len(df)

        if total_rows <= chunk_size:
            # Single invocation
            return self.invoke_prediction(
                model_s3_uri,
                data_s3_uri,
                output_s3_uri,
                prediction_column_name,
                feature_columns
            )

        # Split into chunks
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        chunk_results = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)

            # Create chunk file
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_path = f'/tmp/chunk_{i}.csv'
            chunk_df.to_csv(chunk_path, index=False)

            # Upload chunk to S3
            chunk_s3_uri = f"{data_s3_uri}_chunk_{i}"
            s3_manager.upload_file(chunk_path, chunk_s3_uri)

            # Invoke Lambda for chunk
            chunk_output_uri = f"{output_s3_uri}_chunk_{i}"
            result = self.invoke_prediction(
                model_s3_uri,
                chunk_s3_uri,
                chunk_output_uri,
                prediction_column_name,
                feature_columns
            )

            chunk_results.append({
                'chunk_id': i,
                'rows': end_idx - start_idx,
                'output_uri': chunk_output_uri,
                'execution_time_ms': result['execution_time_ms']
            })

        # Merge chunk results
        merged_output = self._merge_prediction_chunks(chunk_results, output_s3_uri)

        return {
            'success': True,
            'predictions_s3_uri': output_s3_uri,
            'num_predictions': total_rows,
            'num_chunks': num_chunks,
            'total_execution_time_ms': sum(r['execution_time_ms'] for r in chunk_results)
        }
    ```

- [ ] **4.7 Write unit tests for LambdaManager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_lambda_manager.py`
  - Test cases:
    - `test_invoke_prediction_success` - Returns prediction results
    - `test_invoke_prediction_lambda_error` - Handles Lambda function error
    - `test_invoke_async_returns_request_id` - Async invocation returns ID
    - `test_deploy_function_create_new` - Creates new function
    - `test_deploy_function_update_existing` - Updates existing function
    - `test_create_layer` - Layer created with ARN returned
    - `test_batch_predict_single_chunk` - Small dataset uses single invocation
    - `test_batch_predict_multiple_chunks` - Large dataset split correctly
    - `test_batch_predict_merge_results` - Chunks merged into single output

---

## 5.0 Cloud Workflow Telegram Integration

**Goal**: Integrate cloud workflows into Telegram bot with user-friendly messages and state management.

### Sub-Tasks

- [x] **5.1 Add cloud workflow states to StateManager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/core/state_manager.py`
  - Add new workflow enum:
    ```python
    class CloudTrainingState(Enum):
        """States for cloud-based ML training workflow."""
        CHOOSING_CLOUD_LOCAL = "choosing_cloud_local"      # Choose cloud vs local training
        AWAITING_S3_DATASET = "awaiting_s3_dataset"        # User provides S3 URI or uploads
        SELECTING_TARGET = "selecting_target"              # Same as local workflow
        SELECTING_FEATURES = "selecting_features"          # Same as local workflow
        CONFIRMING_MODEL = "confirming_model"              # Same as local workflow
        CONFIRMING_INSTANCE_TYPE = "confirming_instance_type"  # Review instance selection
        LAUNCHING_TRAINING = "launching_training"          # EC2 instance launching
        MONITORING_TRAINING = "monitoring_training"        # Streaming CloudWatch logs
        TRAINING_COMPLETE = "training_complete"            # Training finished
        COMPLETE = "complete"

    class CloudPredictionState(Enum):
        """States for cloud-based prediction workflow."""
        CHOOSING_CLOUD_LOCAL = "choosing_cloud_local"      # Choose cloud vs local prediction
        AWAITING_S3_DATASET = "awaiting_s3_dataset"        # User provides S3 URI
        SELECTING_MODEL = "selecting_model"                # Choose model (local or S3)
        CONFIRMING_PREDICTION_COLUMN = "confirming_prediction_column"
        LAUNCHING_PREDICTION = "launching_prediction"      # Lambda invoking
        PREDICTION_COMPLETE = "prediction_complete"        # Results ready
        COMPLETE = "complete"
    ```
  - Add state transitions:
    ```python
    CLOUD_TRAINING_TRANSITIONS: Dict[Optional[str], Set[str]] = {
        None: {CloudTrainingState.CHOOSING_CLOUD_LOCAL.value},
        CloudTrainingState.CHOOSING_CLOUD_LOCAL.value: {
            CloudTrainingState.AWAITING_S3_DATASET.value,
            MLTrainingState.CHOOSING_DATA_SOURCE.value  # Fall back to local
        },
        CloudTrainingState.AWAITING_S3_DATASET.value: {
            CloudTrainingState.SELECTING_TARGET.value
        },
        # ... rest of transitions
    }
    ```

- [x] **5.2 Create cloud training message templates**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/messages/cloud_messages.py`
  - Define message templates:
    ```python
    # Cloud training messages

    CHOOSE_CLOUD_LOCAL_MESSAGE = """
    🌩️ **Training Location**

    Where would you like to train this model?

    💻 **Local Training** (Free)
    - Runs on this server
    - Limited resources
    - Best for: Small datasets (<1GB), quick experiments

    ☁️ **Cloud Training** (Paid - AWS)
    - Runs on AWS EC2 Spot Instances
    - Scalable resources (up to 64GB RAM, GPU available)
    - Best for: Large datasets (>1GB), neural networks
    - **Cost**: $0.10 - $2.00 per training run

    Choose your training environment:
    """

    def cloud_instance_confirmation_message(
        instance_type: str,
        estimated_cost_usd: float,
        estimated_time_minutes: int,
        dataset_size_mb: float
    ) -> str:
        return f"""
    ☁️ **Cloud Training Configuration**

    📊 Dataset: {dataset_size_mb:.1f} MB
    🖥️ Instance Type: {instance_type}
    ⏱️ Estimated Time: {estimated_time_minutes} minutes
    💰 Estimated Cost: ${estimated_cost_usd:.2f}

    ⚠️ **Important**:
    - You will be charged for actual usage
    - Training logs will stream in real-time
    - Instance will auto-terminate when complete

    Ready to launch cloud training?
    """

    def cloud_training_launched_message(
        instance_id: str,
        instance_type: str
    ) -> str:
        return f"""
    🚀 **Cloud Training Launched**

    Instance ID: `{instance_id}`
    Instance Type: {instance_type}
    Status: Launching...

    ⏳ Waiting for instance to start (typically 1-2 minutes)...

    I'll stream the training logs here as they become available.
    """

    def cloud_training_log_message(log_line: str) -> str:
        return f"📝 `{log_line}`"

    def cloud_training_complete_message(
        model_id: str,
        s3_model_uri: str,
        training_time_minutes: float,
        actual_cost_usd: float,
        metrics: dict[str, float]
    ) -> str:
        metrics_str = "\n".join([f"  - {k}: {v:.4f}" for k, v in metrics.items()])

        return f"""
    ✅ **Cloud Training Complete!**

    🎯 Model ID: `{model_id}`
    📦 S3 Location: `{s3_model_uri}`
    ⏱️ Training Time: {training_time_minutes:.1f} minutes
    💰 Actual Cost: ${actual_cost_usd:.2f}

    📊 **Metrics**:
    {metrics_str}

    The model has been saved to S3 and can be used for predictions.

    Use /predict to run predictions with this model.
    """

    # Cloud prediction messages

    def cloud_prediction_launched_message(
        request_id: str,
        num_rows: int
    ) -> str:
        return f"""
    🚀 **Cloud Prediction Launched**

    Request ID: `{request_id}`
    Rows: {num_rows:,}
    Status: Processing...

    ⏳ Lambda function is running...

    This should complete in 1-2 minutes.
    """

    def cloud_prediction_complete_message(
        s3_output_uri: str,
        num_predictions: int,
        execution_time_ms: int,
        cost_usd: float,
        presigned_url: str
    ) -> str:
        return f"""
    ✅ **Cloud Prediction Complete!**

    📦 Output S3 URI: `{s3_output_uri}`
    📊 Predictions: {num_predictions:,} rows
    ⏱️ Execution Time: {execution_time_ms}ms
    💰 Cost: ${cost_usd:.4f}

    📥 **Download Results**:
    [Click here to download]({presigned_url})
    (Link expires in 1 hour)

    Or access via S3 URI in your AWS account.
    """

    # Error messages

    def cloud_error_message(error_type: str, error_details: str) -> str:
        return f"""
    ❌ **Cloud Operation Failed**

    Error Type: {error_type}
    Details: {error_details}

    No charges were incurred for failed operations.

    Please check your AWS configuration or try again.
    """
    ```

- [x] **5.3 Create cloud training handlers**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_training_handlers.py`
  - Implement workflow handlers:
    ```python
    async def handle_choose_cloud_local(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cloud vs local training choice."""
        user_id = update.effective_user.id
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        # Show choice buttons
        keyboard = [
            [InlineKeyboardButton("💻 Local Training", callback_data="training_local")],
            [InlineKeyboardButton("☁️ Cloud Training (AWS)", callback_data="training_cloud")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            CHOOSE_CLOUD_LOCAL_MESSAGE,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )

    async def handle_cloud_training_start(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Start cloud training workflow."""
        user_id = update.effective_user.id
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        # Transition to cloud training state
        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)
        await state_manager.transition_state(session, CloudTrainingState.AWAITING_S3_DATASET.value)

        await update.callback_query.message.reply_text(
            "☁️ Please provide your dataset:\n\n"
            "1️⃣ Upload a CSV file directly, OR\n"
            "2️⃣ Provide an S3 URI (e.g., s3://my-bucket/data.csv)\n\n"
            "If using S3 URI, ensure the bot's IAM role has read access."
        )

    async def handle_cloud_dataset_input(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle dataset upload or S3 URI."""
        user_id = update.effective_user.id
        session = await state_manager.get_session(user_id, str(user_id))

        if update.message.document:
            # User uploaded file - upload to S3
            file = await update.message.document.get_file()
            local_path = f"/tmp/{file.file_id}.csv"
            await file.download_to_drive(local_path)

            # Upload to S3
            s3_manager = S3Manager(aws_client, cloud_config)
            s3_uri = s3_manager.upload_dataset(user_id, local_path)

            session.selections['s3_dataset_uri'] = s3_uri

            await update.message.reply_text(
                f"✅ Dataset uploaded to S3: `{s3_uri}`",
                parse_mode=ParseMode.MARKDOWN
            )

        elif update.message.text and update.message.text.startswith('s3://'):
            # User provided S3 URI
            s3_uri = update.message.text.strip()

            # Validate S3 path
            s3_manager = S3Manager(aws_client, cloud_config)
            if not s3_manager.validate_s3_path(s3_uri, user_id):
                await update.message.reply_text(
                    "❌ Invalid S3 URI or access denied.\n\n"
                    "Ensure the URI is correct and the bot has read access."
                )
                return

            session.selections['s3_dataset_uri'] = s3_uri

            await update.message.reply_text(
                f"✅ Using S3 dataset: `{s3_uri}`",
                parse_mode=ParseMode.MARKDOWN
            )

        else:
            await update.message.reply_text(
                "❌ Invalid input. Please upload a CSV file or provide an S3 URI."
            )
            return

        # Download and load dataset for schema detection
        # (Continue with existing ML training workflow for target/feature selection)
        await transition_to_target_selection(update, context, session)

    async def handle_cloud_training_launch(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Launch EC2 instance for cloud training."""
        user_id = update.effective_user.id
        session = await state_manager.get_session(user_id, str(user_id))

        # Get training configuration from session
        model_type = session.selections['model_type']
        target_column = session.selections['target_column']
        feature_columns = session.selections['feature_columns']
        hyperparameters = session.selections.get('hyperparameters', {})
        s3_dataset_uri = session.selections['s3_dataset_uri']

        # Select instance type
        ec2_manager = EC2Manager(aws_client, cloud_config)
        dataset_size_mb = session.uploaded_data.memory_usage(deep=True).sum() / (1024 * 1024)
        instance_type = ec2_manager.select_instance_type(
            dataset_size_mb,
            model_type,
            estimated_training_time_minutes=10  # TODO: Better estimation
        )

        # Estimate cost
        cost_tracker = CostTracker(cloud_config)
        estimated_cost = cost_tracker.estimate_training_cost(instance_type, 10)

        # Show confirmation
        confirmation_msg = cloud_instance_confirmation_message(
            instance_type,
            estimated_cost,
            10,  # estimated time
            dataset_size_mb
        )

        keyboard = [
            [InlineKeyboardButton("✅ Launch Training", callback_data="confirm_cloud_launch")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_cloud_training")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            confirmation_msg,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )

    async def handle_cloud_training_confirmed(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Actually launch EC2 instance after confirmation."""
        user_id = update.effective_user.id
        session = await state_manager.get_session(user_id, str(user_id))

        # Generate output S3 URI
        model_id = f"model_{user_id}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        s3_output_uri = f"s3://{cloud_config.s3_bucket_name}/{cloud_config.s3_model_prefix}/user_{user_id}/{model_id}"

        # Generate UserData script
        ec2_manager = EC2Manager(aws_client, cloud_config)
        userdata_script = ec2_manager.generate_training_userdata(
            s3_dataset_uri=session.selections['s3_dataset_uri'],
            model_type=session.selections['model_type'],
            target_column=session.selections['target_column'],
            feature_columns=session.selections['feature_columns'],
            hyperparameters=session.selections.get('hyperparameters', {}),
            s3_output_uri=s3_output_uri
        )

        # Launch instance
        instance_info = ec2_manager.launch_spot_instance(
            instance_type=session.selections['instance_type'],
            user_data_script=userdata_script,
            tags={
                'user_id': str(user_id),
                'model_id': model_id,
                'workflow': 'cloud_training'
            }
        )

        # Save instance info to session
        session.selections['instance_id'] = instance_info['instance_id']
        session.selections['model_id'] = model_id
        session.selections['s3_output_uri'] = s3_output_uri
        await state_manager.update_session(session)

        # Send launch message
        await update.callback_query.message.reply_text(
            cloud_training_launched_message(
                instance_info['instance_id'],
                instance_info['instance_type']
            ),
            parse_mode=ParseMode.MARKDOWN
        )

        # Start log streaming in background
        asyncio.create_task(stream_training_logs(update, context, session, instance_info['instance_id']))

    async def stream_training_logs(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession,
        instance_id: str
    ) -> None:
        """Stream CloudWatch logs to Telegram."""
        ec2_manager = EC2Manager(aws_client, cloud_config)

        async for log_line in ec2_manager.poll_training_logs(instance_id):
            # Send log to user
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=cloud_training_log_message(log_line),
                parse_mode=ParseMode.MARKDOWN
            )

            # Parse progress
            progress = ec2_manager.parse_training_progress(log_line)
            if progress:
                # Update user with progress indicator
                pass

            # Check for completion
            if "Training complete!" in log_line:
                await handle_cloud_training_completion(update, context, session)
                break

    async def handle_cloud_training_completion(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Handle training completion."""
        # Calculate actual cost
        cost_tracker = CostTracker(cloud_config)
        actual_cost = cost_tracker.calculate_training_cost(
            session.selections['instance_id'],
            session.selections['instance_type']
        )

        # Get model metrics from S3 (uploaded by training script)
        # TODO: Download metrics.json from S3

        # Send completion message
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=cloud_training_complete_message(
                session.selections['model_id'],
                session.selections['s3_output_uri'],
                10.0,  # TODO: actual training time
                actual_cost,
                {'r2': 0.85, 'mse': 0.12}  # TODO: actual metrics
            ),
            parse_mode=ParseMode.MARKDOWN
        )

        # Complete workflow
        await state_manager.complete_workflow(session.user_id)
    ```

- [x] **5.4 Create cloud prediction handlers**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_prediction_handlers.py`
  - Implement prediction workflow handlers (similar structure to training)

- [x] **5.5 Register cloud handlers in telegram_bot.py**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/telegram_bot.py`
  - Add handler registrations:
    ```python
    # Cloud training handlers
    from src.bot.handlers.cloud_training_handlers import (
        handle_choose_cloud_local,
        handle_cloud_training_start,
        handle_cloud_dataset_input,
        handle_cloud_training_launch,
        handle_cloud_training_confirmed
    )

    # Register handlers
    application.add_handler(CallbackQueryHandler(handle_cloud_training_start, pattern="^training_cloud$"))
    application.add_handler(MessageHandler(filters.Document.ALL | filters.TEXT, handle_cloud_dataset_input))
    application.add_handler(CallbackQueryHandler(handle_cloud_training_confirmed, pattern="^confirm_cloud_launch$"))
    ```

- [x] **5.6 Update requirements.txt with boto3**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/requirements.txt`
  - Add cloud dependencies:
    ```
    boto3==1.28.85
    botocore==1.31.85
    watchtower==3.0.1  # CloudWatch logging integration
    moto==4.2.0  # For AWS mocking in tests
    ```

---

## 6.0 Cost Tracking & Display System

**Goal**: Implement cost calculation, tracking, and user-facing cost displays.

### Sub-Tasks

- [x] **6.1 Create CostTracker class with pricing lookups**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Implement cost calculation logic:
    ```python
    class CostTracker:
        """Track and calculate AWS usage costs."""

        # EC2 Spot pricing (approximate, varies by region/availability)
        EC2_SPOT_PRICING = {
            't3.medium': 0.0125,   # $/hour
            'm5.large': 0.030,
            'm5.xlarge': 0.060,
            'm5.2xlarge': 0.120,
            'p3.2xlarge': 0.918    # GPU instance
        }

        # Lambda pricing
        LAMBDA_PRICE_PER_GB_SECOND = 0.0000166667
        LAMBDA_PRICE_PER_REQUEST = 0.0000002

        # S3 pricing
        S3_STORAGE_PRICE_PER_GB_MONTH = 0.023
        S3_REQUEST_PRICE_PUT = 0.000005  # per 1000 requests
        S3_REQUEST_PRICE_GET = 0.0000004  # per 1000 requests

        def __init__(self, config: CloudConfig):
            self.config = config
            self.cost_log_file = Path("data/logs/cloud_costs.json")
            self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)

        def estimate_training_cost(
            self,
            instance_type: str,
            estimated_time_minutes: int
        ) -> float:
            """
            Estimate EC2 training cost.

            Args:
                instance_type: EC2 instance type
                estimated_time_minutes: Expected training duration

            Returns:
                Estimated cost in USD
            """
            hourly_rate = self.EC2_SPOT_PRICING.get(instance_type, 0.10)
            hours = estimated_time_minutes / 60.0
            return hourly_rate * hours

        def calculate_training_cost(
            self,
            instance_id: str,
            instance_type: str
        ) -> float:
            """
            Calculate actual EC2 training cost.

            Args:
                instance_id: EC2 instance ID
                instance_type: EC2 instance type

            Returns:
                Actual cost in USD based on runtime
            """
            # Get instance runtime from CloudWatch or EC2 API
            runtime_minutes = self._get_instance_runtime(instance_id)

            hourly_rate = self.EC2_SPOT_PRICING.get(instance_type, 0.10)
            hours = runtime_minutes / 60.0
            cost = hourly_rate * hours

            # Log cost
            self._log_cost(
                service='ec2',
                operation='training',
                instance_id=instance_id,
                instance_type=instance_type,
                runtime_minutes=runtime_minutes,
                cost_usd=cost
            )

            return cost

        def estimate_prediction_cost(
            self,
            num_rows: int,
            lambda_memory_mb: int = 3008,
            estimated_time_seconds: int = 60
        ) -> float:
            """
            Estimate Lambda prediction cost.

            Args:
                num_rows: Number of prediction rows
                lambda_memory_mb: Lambda memory allocation
                estimated_time_seconds: Expected execution time

            Returns:
                Estimated cost in USD
            """
            gb_seconds = (lambda_memory_mb / 1024.0) * estimated_time_seconds
            compute_cost = gb_seconds * self.LAMBDA_PRICE_PER_GB_SECOND
            request_cost = self.LAMBDA_PRICE_PER_REQUEST

            return compute_cost + request_cost

        def calculate_prediction_cost(
            self,
            execution_time_ms: int,
            lambda_memory_mb: int = 3008
        ) -> float:
            """
            Calculate actual Lambda prediction cost.

            Args:
                execution_time_ms: Actual execution time in milliseconds
                lambda_memory_mb: Lambda memory allocation

            Returns:
                Actual cost in USD
            """
            execution_time_seconds = execution_time_ms / 1000.0
            gb_seconds = (lambda_memory_mb / 1024.0) * execution_time_seconds
            compute_cost = gb_seconds * self.LAMBDA_PRICE_PER_GB_SECOND
            request_cost = self.LAMBDA_PRICE_PER_REQUEST

            total_cost = compute_cost + request_cost

            # Log cost
            self._log_cost(
                service='lambda',
                operation='prediction',
                execution_time_ms=execution_time_ms,
                memory_mb=lambda_memory_mb,
                cost_usd=total_cost
            )

            return total_cost
    ```

- [x] **6.2 Implement S3 storage cost calculation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add S3 cost methods:
    ```python
    def calculate_s3_storage_cost(
        self,
        user_id: int,
        s3_manager: S3Manager
    ) -> dict[str, Any]:
        """
        Calculate monthly S3 storage cost for user.

        Args:
            user_id: User ID
            s3_manager: S3Manager instance

        Returns:
            Dict with storage breakdown and cost
        """
        # List all user's datasets and models
        datasets = s3_manager.list_user_datasets(user_id)
        models = s3_manager.list_user_models(user_id)

        # Calculate total storage
        total_dataset_mb = sum(d['size_mb'] for d in datasets)
        total_model_mb = sum(m['size_mb'] for m in models)
        total_gb = (total_dataset_mb + total_model_mb) / 1024.0

        # Calculate monthly cost
        monthly_cost = total_gb * self.S3_STORAGE_PRICE_PER_GB_MONTH

        return {
            'total_storage_gb': total_gb,
            'dataset_storage_mb': total_dataset_mb,
            'model_storage_mb': total_model_mb,
            'num_datasets': len(datasets),
            'num_models': len(models),
            'monthly_cost_usd': monthly_cost
        }
    ```

- [x] **6.3 Implement cost logging and persistence**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add logging methods:
    ```python
    def _log_cost(
        self,
        service: str,
        operation: str,
        cost_usd: float,
        **metadata: Any
    ) -> None:
        """
        Log cost entry to file.

        Args:
            service: AWS service (ec2, lambda, s3)
            operation: Operation type (training, prediction, storage)
            cost_usd: Cost in USD
            **metadata: Additional metadata
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'service': service,
            'operation': operation,
            'cost_usd': cost_usd,
            **metadata
        }

        # Append to log file
        if self.cost_log_file.exists():
            with open(self.cost_log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(entry)

        with open(self.cost_log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def get_user_costs(
        self,
        user_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve cost logs for user and date range.

        Args:
            user_id: Optional user ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of cost log entries
        """
        if not self.cost_log_file.exists():
            return []

        with open(self.cost_log_file, 'r') as f:
            logs = json.load(f)

        # Apply filters
        filtered = logs

        if user_id is not None:
            filtered = [log for log in filtered if log.get('user_id') == user_id]

        if start_date:
            filtered = [log for log in filtered if datetime.fromisoformat(log['timestamp']) >= start_date]

        if end_date:
            filtered = [log for log in filtered if datetime.fromisoformat(log['timestamp']) <= end_date]

        return filtered
    ```

- [x] **6.4 Implement monthly cost summary generation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add summary methods:
    ```python
    def generate_monthly_summary(
        self,
        user_id: int,
        month: Optional[int] = None,
        year: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Generate monthly cost summary for user.

        Args:
            user_id: User ID
            month: Month (1-12), defaults to current month
            year: Year, defaults to current year

        Returns:
            Summary dict with costs by service and operation
        """
        now = datetime.now()
        month = month or now.month
        year = year or now.year

        # Get costs for month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        logs = self.get_user_costs(user_id, start_date, end_date)

        # Aggregate by service and operation
        summary = {
            'user_id': user_id,
            'month': month,
            'year': year,
            'total_cost_usd': 0.0,
            'by_service': {},
            'by_operation': {},
            'num_operations': len(logs)
        }

        for log in logs:
            service = log['service']
            operation = log['operation']
            cost = log['cost_usd']

            summary['total_cost_usd'] += cost

            # By service
            if service not in summary['by_service']:
                summary['by_service'][service] = 0.0
            summary['by_service'][service] += cost

            # By operation
            if operation not in summary['by_operation']:
                summary['by_operation'][operation] = 0.0
            summary['by_operation'][operation] += cost

        return summary

    def format_cost_summary_message(self, summary: dict[str, Any]) -> str:
        """Format cost summary for Telegram display."""
        month_name = calendar.month_name[summary['month']]

        by_service = "\n".join([
            f"  - {service}: ${cost:.2f}"
            for service, cost in summary['by_service'].items()
        ])

        by_operation = "\n".join([
            f"  - {operation}: ${cost:.2f}"
            for operation, cost in summary['by_operation'].items()
        ])

        return f"""
    💰 **Cloud Cost Summary**

    📅 Period: {month_name} {summary['year']}
    👤 User ID: {summary['user_id']}

    💵 **Total Cost**: ${summary['total_cost_usd']:.2f}

    📊 **By Service**:
    {by_service}

    🔧 **By Operation**:
    {by_operation}

    🔢 Total Operations: {summary['num_operations']}
    """
    ```

- [x] **6.5 Implement cost limit validation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add validation methods:
    ```python
    def validate_training_cost(
        self,
        estimated_cost_usd: float,
        user_id: int
    ) -> tuple[bool, Optional[str]]:
        """
        Validate training cost against limits.

        Args:
            estimated_cost_usd: Estimated training cost
            user_id: User ID

        Returns:
            (is_valid, error_message)
        """
        # Check single operation limit
        if estimated_cost_usd > self.config.max_training_cost_usd:
            return False, (
                f"❌ Estimated cost (${estimated_cost_usd:.2f}) exceeds "
                f"single training limit (${self.config.max_training_cost_usd:.2f})."
            )

        # Check monthly budget
        summary = self.generate_monthly_summary(user_id)
        projected_total = summary['total_cost_usd'] + estimated_cost_usd

        if projected_total > self.config.monthly_budget_usd:
            return False, (
                f"❌ This operation would exceed your monthly budget.\n\n"
                f"Current spend: ${summary['total_cost_usd']:.2f}\n"
                f"Estimated cost: ${estimated_cost_usd:.2f}\n"
                f"Monthly budget: ${self.config.monthly_budget_usd:.2f}"
            )

        # Check warning threshold
        warn_threshold = self.config.monthly_budget_usd * self.config.warn_threshold_percent
        if projected_total > warn_threshold:
            return True, (
                f"⚠️ Warning: You've used {(projected_total / self.config.monthly_budget_usd * 100):.0f}% "
                f"of your monthly budget (${self.config.monthly_budget_usd:.2f})."
            )

        return True, None
    ```

- [x] **6.6 Write unit tests for CostTracker**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_cost_tracker.py`
  - Test cases:
    - `test_estimate_training_cost_t3_medium` - Calculates correct cost
    - `test_estimate_training_cost_gpu_instance` - GPU pricing correct
    - `test_calculate_prediction_cost` - Lambda cost calculated correctly
    - `test_calculate_s3_storage_cost` - S3 monthly cost accurate
    - `test_log_cost_creates_file` - Creates log file on first log
    - `test_log_cost_appends_entries` - Appends to existing log
    - `test_get_user_costs_filters_by_user` - Returns only user's costs
    - `test_get_user_costs_filters_by_date` - Date range filter works
    - `test_generate_monthly_summary_aggregates` - Aggregation correct
    - `test_validate_training_cost_within_limit` - Valid cost passes
    - `test_validate_training_cost_exceeds_limit` - Over limit rejected
    - `test_validate_training_cost_budget_warning` - Warning at 80%

---

## 7.0 Security & User Isolation Implementation

**Goal**: Implement S3 bucket policies, IAM roles, encryption, and user isolation.

### Sub-Tasks

- [ ] **7.1 Create S3 bucket policy JSON templates**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py`
  - Define bucket policies:
    ```python
    class SecurityManager:
        """Manage AWS security policies and encryption."""

        def __init__(self, config: CloudConfig):
            self.config = config

        def generate_s3_bucket_policy(self, account_id: str) -> dict[str, Any]:
            """
            Generate S3 bucket policy with user isolation.

            Policy enforces:
            - Users can only access their own prefix
            - All uploads must be encrypted
            - Public access blocked

            Returns:
                Bucket policy JSON
            """
            return {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "DenyUnencryptedObjectUploads",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:PutObject",
                        "Resource": f"arn:aws:s3:::{self.config.s3_bucket_name}/*",
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
                            f"arn:aws:s3:::{self.config.s3_bucket_name}",
                            f"arn:aws:s3:::{self.config.s3_bucket_name}/*"
                        ],
                        "Condition": {
                            "StringEquals": {
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
                            f"arn:aws:s3:::{self.config.s3_bucket_name}",
                            f"arn:aws:s3:::{self.config.s3_bucket_name}/*"
                        ]
                    }
                ]
            }
    ```

- [ ] **7.2 Create IAM role policy JSON templates**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py`
  - Define IAM policies:
    ```python
    def generate_ec2_iam_role_policy(self) -> dict[str, Any]:
        """
        Generate IAM policy for EC2 training instances.

        Permissions:
        - Read from S3 datasets prefix
        - Write to S3 models prefix
        - Write CloudWatch logs
        - Terminate self

        Returns:
            IAM policy JSON
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
                        f"arn:aws:s3:::{self.config.s3_bucket_name}",
                        f"arn:aws:s3:::{self.config.s3_bucket_name}/{self.config.s3_dataset_prefix}/*"
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
                        f"arn:aws:s3:::{self.config.s3_bucket_name}/{self.config.s3_model_prefix}/*"
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

    def generate_lambda_iam_role_policy(self) -> dict[str, Any]:
        """
        Generate IAM policy for Lambda prediction function.

        Permissions:
        - Read models and datasets from S3
        - Write predictions to S3
        - Write CloudWatch logs

        Returns:
            IAM policy JSON
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
                        f"arn:aws:s3:::{self.config.s3_bucket_name}/{self.config.s3_model_prefix}/*",
                        f"arn:aws:s3:::{self.config.s3_bucket_name}/{self.config.s3_dataset_prefix}/*"
                    ]
                },
                {
                    "Sid": "S3WritePredictions",
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.config.s3_bucket_name}/predictions/*"
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
    ```

- [ ] **7.3 Implement encryption configuration**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py`
  - Add encryption methods:
    ```python
    def configure_bucket_encryption(
        self,
        s3_client: boto3.client
    ) -> None:
        """
        Enable default encryption for S3 bucket.

        Uses AES256 server-side encryption by default.
        If KMS key configured, use KMS encryption instead.
        """
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

        # Use KMS if configured
        if self.config.encryption_key_id:
            encryption_config['Rules'][0]['ApplyServerSideEncryptionByDefault'] = {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': self.config.encryption_key_id
            }

        try:
            s3_client.put_bucket_encryption(
                Bucket=self.config.s3_bucket_name,
                ServerSideEncryptionConfiguration=encryption_config
            )
        except ClientError as e:
            raise CloudConfigurationError(f"Failed to configure encryption: {e}")

    def configure_bucket_versioning(
        self,
        s3_client: boto3.client
    ) -> None:
        """Enable versioning for S3 bucket."""
        try:
            s3_client.put_bucket_versioning(
                Bucket=self.config.s3_bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
        except ClientError as e:
            raise CloudConfigurationError(f"Failed to enable versioning: {e}")
    ```

- [ ] **7.4 Implement user path isolation validation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py`
  - Add isolation validation:
    ```python
    def validate_user_s3_access(
        self,
        s3_key: str,
        user_id: int,
        operation: str  # 'read' or 'write'
    ) -> bool:
        """
        Validate user can access S3 path.

        Enforces user isolation:
        - Users can only access paths with user_{user_id} prefix
        - Read operations allowed for datasets and models
        - Write operations only allowed for user's own prefix

        Args:
            s3_key: S3 object key to validate
            user_id: User ID requesting access
            operation: 'read' or 'write'

        Returns:
            True if access allowed

        Raises:
            S3Error: Access denied
        """
        user_prefix = f"user_{user_id}"

        # Check key belongs to user
        if operation == 'write':
            # Write: must be in user's prefix
            allowed_prefixes = [
                f"{self.config.s3_dataset_prefix}/{user_prefix}/",
                f"{self.config.s3_model_prefix}/{user_prefix}/",
                f"predictions/{user_prefix}/"
            ]

            if not any(s3_key.startswith(prefix) for prefix in allowed_prefixes):
                raise S3Error(
                    f"Write access denied: {s3_key} does not belong to user {user_id}"
                )

        elif operation == 'read':
            # Read: can read own data, but not other users' data
            if user_prefix not in s3_key:
                # Check if trying to read another user's data
                if any(f"user_{uid}/" in s3_key for uid in range(1, 1000000) if uid != user_id):
                    raise S3Error(
                        f"Read access denied: {s3_key} belongs to another user"
                    )

        return True
    ```

- [ ] **7.5 Implement audit logging setup**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py`
  - Add audit logging:
    ```python
    def enable_s3_access_logging(
        self,
        s3_client: boto3.client,
        log_bucket: str
    ) -> None:
        """
        Enable S3 access logging for audit trail.

        Args:
            s3_client: S3 client
            log_bucket: Bucket to store access logs
        """
        try:
            s3_client.put_bucket_logging(
                Bucket=self.config.s3_bucket_name,
                BucketLoggingStatus={
                    'LoggingEnabled': {
                        'TargetBucket': log_bucket,
                        'TargetPrefix': f'access-logs/{self.config.s3_bucket_name}/'
                    }
                }
            )
        except ClientError as e:
            raise CloudConfigurationError(f"Failed to enable access logging: {e}")

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

        Args:
            user_id: User performing operation
            operation: Operation type (training, prediction, s3_upload, etc.)
            resource: Resource identifier (instance_id, s3_key, etc.)
            success: Whether operation succeeded
            **metadata: Additional context
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'operation': operation,
            'resource': resource,
            'success': success,
            **metadata
        }

        audit_log_file = Path("data/logs/cloud_audit.json")
        audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        if audit_log_file.exists():
            with open(audit_log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(audit_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    ```

- [ ] **7.6 Create AWS setup automation script**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/setup_aws.py`
  - Python script to automate AWS infrastructure setup:
    ```python
    #!/usr/bin/env python3
    """
    AWS Infrastructure Setup Script

    This script automates the setup of AWS resources for cloud ML workflows:
    - Creates S3 bucket with encryption and policies
    - Creates IAM roles with minimal permissions
    - Configures security groups
    - Deploys Lambda function
    - Sets up CloudWatch log groups

    Usage:
        python scripts/cloud/setup_aws.py --config config/config.yaml
    """

    import argparse
    import boto3
    import json
    from pathlib import Path

    from src.cloud.aws_config import CloudConfig
    from src.cloud.security import SecurityManager

    def create_s3_bucket(s3_client, bucket_name: str, region: str) -> None:
        """Create S3 bucket with best practices."""
        print(f"Creating S3 bucket: {bucket_name}")

        try:
            if region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )

            print(f"✅ Bucket created: {bucket_name}")
        except s3_client.exceptions.BucketAlreadyOwnedByYou:
            print(f"⚠️  Bucket already exists: {bucket_name}")

    def configure_bucket_security(
        s3_client,
        bucket_name: str,
        security_manager: SecurityManager,
        account_id: str
    ) -> None:
        """Apply security configurations to bucket."""
        print(f"Configuring bucket security...")

        # Block public access
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        print("✅ Public access blocked")

        # Enable encryption
        security_manager.configure_bucket_encryption(s3_client)
        print("✅ Encryption enabled")

        # Enable versioning
        security_manager.configure_bucket_versioning(s3_client)
        print("✅ Versioning enabled")

        # Apply bucket policy
        bucket_policy = security_manager.generate_s3_bucket_policy(account_id)
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(bucket_policy)
        )
        print("✅ Bucket policy applied")

    def create_iam_roles(
        iam_client,
        security_manager: SecurityManager
    ) -> dict[str, str]:
        """Create IAM roles for EC2 and Lambda."""
        print("Creating IAM roles...")

        # EC2 role
        ec2_role_name = "ml-agent-ec2-training-role"
        try:
            ec2_role = iam_client.create_role(
                RoleName=ec2_role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": "ec2.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }]
                })
            )

            # Attach policy
            policy = security_manager.generate_ec2_iam_role_policy()
            iam_client.put_role_policy(
                RoleName=ec2_role_name,
                PolicyName="EC2TrainingPolicy",
                PolicyDocument=json.dumps(policy)
            )

            print(f"✅ EC2 role created: {ec2_role_name}")
            ec2_role_arn = ec2_role['Role']['Arn']
        except iam_client.exceptions.EntityAlreadyExistsException:
            print(f"⚠️  EC2 role already exists: {ec2_role_name}")
            ec2_role_arn = iam_client.get_role(RoleName=ec2_role_name)['Role']['Arn']

        # Lambda role (similar)
        # ...

        return {
            'ec2_role_arn': ec2_role_arn,
            'lambda_role_arn': lambda_role_arn
        }

    def main():
        parser = argparse.ArgumentParser(description="Setup AWS infrastructure")
        parser.add_argument('--config', required=True, help="Path to config.yaml")
        args = parser.parse_args()

        # Load configuration
        config = CloudConfig.from_yaml(args.config)
        config.validate()

        # Initialize clients
        s3_client = boto3.client('s3', region_name=config.aws_region)
        iam_client = boto3.client('iam', region_name=config.aws_region)

        # Get account ID
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']

        # Initialize security manager
        security_manager = SecurityManager(config)

        # Setup S3
        create_s3_bucket(s3_client, config.s3_bucket_name, config.aws_region)
        configure_bucket_security(s3_client, config.s3_bucket_name, security_manager, account_id)

        # Setup IAM
        role_arns = create_iam_roles(iam_client, security_manager)

        # Print summary
        print("\n" + "="*60)
        print("AWS Infrastructure Setup Complete!")
        print("="*60)
        print(f"S3 Bucket: {config.s3_bucket_name}")
        print(f"EC2 Role ARN: {role_arns['ec2_role_arn']}")
        print(f"Lambda Role ARN: {role_arns['lambda_role_arn']}")
        print("\nUpdate your .env file with these values:")
        print(f"IAM_ROLE_ARN={role_arns['ec2_role_arn']}")

    if __name__ == '__main__':
        main()
    ```
  - Make script executable: `chmod +x scripts/cloud/setup_aws.py`

- [ ] **7.7 Write security tests**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_security.py`
  - Test cases:
    - `test_generate_s3_bucket_policy_denies_unencrypted` - Policy blocks unencrypted
    - `test_generate_ec2_iam_policy_allows_s3_read` - Policy allows dataset read
    - `test_generate_ec2_iam_policy_denies_s3_write_datasets` - Can't write to datasets
    - `test_generate_lambda_iam_policy_allows_prediction_write` - Can write predictions
    - `test_validate_user_s3_access_own_prefix_write` - User can write own prefix
    - `test_validate_user_s3_access_other_user_write` - Blocks writing other user
    - `test_validate_user_s3_access_other_user_read` - Blocks reading other user
    - `test_configure_bucket_encryption_aes256` - AES256 configured
    - `test_configure_bucket_encryption_kms` - KMS configured when key present
    - `test_audit_log_operation_creates_entry` - Audit log created

---

## Testing Notes

### Integration Testing Strategy

**Test Order**:
1. Unit tests for each component (S3Manager, EC2Manager, LambdaManager, CostTracker)
2. Integration tests for AWS operations (requires AWS account or LocalStack)
3. End-to-end Telegram workflow tests (mock Telegram API)

**Test Environment Setup**:
```bash
# Option 1: Use LocalStack for local AWS testing
pip install localstack localstack-client
localstack start

# Option 2: Use AWS test account with separate resources
export AWS_PROFILE=ml-agent-test
```

**Critical Test Scenarios**:
1. **Cloud Training Flow**: Upload dataset → Launch EC2 → Stream logs → Save model → Terminate
2. **Cloud Prediction Flow**: Select model → Upload data → Invoke Lambda → Download results
3. **Cost Tracking**: Verify cost calculations match actual AWS billing
4. **Security Isolation**: Attempt cross-user access (should fail)
5. **Error Handling**: Spot interruption, Lambda timeout, S3 access denied

### Manual Testing Checklist

Before production deployment:
- [ ] AWS credentials configured correctly
- [ ] S3 bucket created with encryption
- [ ] IAM roles have minimal required permissions
- [ ] EC2 Spot instances launch successfully
- [ ] CloudWatch logs stream to Telegram
- [ ] Lambda function deploys and invokes
- [ ] Cost tracking logs match AWS billing
- [ ] User isolation prevents cross-user access
- [ ] Audit logs capture all operations
- [ ] Error messages are user-friendly

---

## Implementation Order Recommendation

**Week 1: Foundation**
- Complete tasks 1.1-1.6 (AWS Infrastructure)
- Complete tasks 2.1-2.6 (S3 Management)

**Week 2: Training Infrastructure**
- Complete tasks 3.1-3.7 (EC2 Training)
- Complete tasks 6.1-6.6 (Cost Tracking)

**Week 3: Prediction Service**
- Complete tasks 4.1-4.7 (Lambda Prediction)
- Complete tasks 7.1-7.7 (Security)

**Week 4: Integration & Testing**
- Complete tasks 5.1-5.6 (Telegram Integration)
- Complete all integration tests
- Manual testing and bug fixes

---

## Success Criteria

Implementation is complete when:
1. ✅ All unit tests pass (>90% coverage)
2. ✅ Integration tests pass with real AWS services
3. ✅ Manual Telegram workflow test completes end-to-end
4. ✅ Cost tracking matches AWS billing (within 5%)
5. ✅ Security audit shows no cross-user access possible
6. ✅ Documentation updated with cloud workflow instructions
7. ✅ `.env.example` and `config.yaml` have complete cloud sections

---

**Document Version**: 1.0
**Last Updated**: 2025-10-23
**Next Review**: After Phase 1 completion
