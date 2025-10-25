"""
S3 Manager for dataset and model storage operations.

This module provides S3Manager class for uploading datasets and models to S3
with user isolation, multipart upload support, encryption, and metadata tracking.

Key Features:
- User isolation: datasets/user_{user_id}/{timestamp}_{filename}
- Automatic multipart upload for files >5MB
- Server-side encryption (AES256)
- Metadata: uploaded_at, original_filename
- S3 URI return format: s3://bucket/key
- Lifecycle policies for automatic cleanup
- Presigned URLs for temporary access

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 2.1: S3Manager with TDD)
Updated: 2025-10-23 (Task 2.3: Add download and listing operations)
Updated: 2025-10-23 (Task 2.4 & 2.5: Add lifecycle policies and presigned URLs)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from botocore.exceptions import ClientError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import S3Error
from src.cloud.provider_interface import CloudStorageProvider


class S3Manager(CloudStorageProvider):
    """
    S3 storage manager for datasets and models.

    Provides high-level interface for uploading datasets with user isolation,
    automatic multipart upload for large files, encryption, and metadata tracking.
    """

    # Multipart upload threshold (5MB)
    MULTIPART_THRESHOLD_BYTES = 5 * 1024 * 1024
    # Multipart chunk size (5MB)
    MULTIPART_CHUNK_SIZE = 5 * 1024 * 1024

    def __init__(self, aws_client: AWSClient, config: CloudConfig) -> None:
        """
        Initialize S3Manager with AWS client and configuration.

        Args:
            aws_client: AWSClient instance providing S3 client access
            config: CloudConfig instance with S3 bucket and prefix settings

        Example:
            >>> config = CloudConfig.from_yaml("config.yaml")
            >>> aws_client = AWSClient(config)
            >>> s3_manager = S3Manager(aws_client, config)
        """
        self._aws_client = aws_client
        self._config = config
        self._s3_client = aws_client.get_s3_client()

    def upload_dataset(
        self,
        user_id: int,
        file_path: Union[str, Path],
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Upload dataset to S3 with user isolation.

        Automatically chooses simple upload (<5MB) or multipart upload (>5MB).
        Generates S3 key with user prefix and timestamp for isolation.

        Args:
            user_id: User ID for isolation (used in S3 key prefix)
            file_path: Path to dataset file to upload
            dataset_name: Optional custom dataset name (defaults to filename)

        Returns:
            str: S3 URI in format s3://bucket/key

        Raises:
            S3Error: If file doesn't exist or upload fails

        Example:
            >>> uri = s3_manager.upload_dataset(
            ...     user_id=12345,
            ...     file_path="/path/to/housing.csv",
            ...     dataset_name="housing_data.csv"
            ... )
            >>> print(uri)
            s3://my-bucket/datasets/user_12345/20251023_143045_housing_data.csv
        """
        # Convert to Path object
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise S3Error(
                message=f"Dataset file not found: {file_path}",
                bucket=self._config.s3_bucket,
                error_code="FileNotFound"
            )

        # Determine filename to use
        filename = dataset_name if dataset_name else file_path.name

        # Generate S3 key with user isolation
        s3_key = self._generate_dataset_key(user_id=user_id, filename=filename)

        # Get file size to choose upload method
        file_size = file_path.stat().st_size

        # Choose upload method based on file size
        if file_size < self.MULTIPART_THRESHOLD_BYTES:
            return self._simple_upload(file_path=file_path, s3_key=s3_key)
        else:
            return self._multipart_upload(file_path=file_path, s3_key=s3_key)

    def _simple_upload(self, file_path: Path, s3_key: str) -> str:
        """
        Upload small file (<5MB) using simple put_object.

        Args:
            file_path: Path to file to upload
            s3_key: S3 object key (path within bucket)

        Returns:
            str: S3 URI (s3://bucket/key)

        Raises:
            S3Error: If upload fails
        """
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Prepare metadata
            metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'original_filename': file_path.name
            }

            # Upload to S3 with encryption
            self._s3_client.put_object(
                Bucket=self._config.s3_bucket,
                Key=s3_key,
                Body=file_content,
                ServerSideEncryption='AES256',
                Metadata=metadata
            )

            # Return S3 URI
            return f"s3://{self._config.s3_bucket}/{s3_key}"

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to upload file to S3: {error_message}",
                bucket=self._config.s3_bucket,
                key=s3_key,
                error_code=error_code,
                request_id=request_id
            )

    def _multipart_upload(self, file_path: Path, s3_key: str) -> str:
        """
        Upload large file (>5MB) using multipart upload.

        Splits file into chunks and uploads in parallel for better performance
        and reliability. Automatically aborts upload on error.

        Args:
            file_path: Path to file to upload
            s3_key: S3 object key (path within bucket)

        Returns:
            str: S3 URI (s3://bucket/key)

        Raises:
            S3Error: If multipart upload fails
        """
        upload_id = None

        try:
            # Prepare metadata
            metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'original_filename': file_path.name
            }

            # Create multipart upload
            response = self._s3_client.create_multipart_upload(
                Bucket=self._config.s3_bucket,
                Key=s3_key,
                ServerSideEncryption='AES256',
                Metadata=metadata
            )
            upload_id = response['UploadId']

            # Upload parts
            parts = []
            part_number = 1

            with open(file_path, 'rb') as f:
                while True:
                    # Read chunk
                    chunk = f.read(self.MULTIPART_CHUNK_SIZE)
                    if not chunk:
                        break

                    # Upload part
                    part_response = self._s3_client.upload_part(
                        Bucket=self._config.s3_bucket,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk
                    )

                    # Store part info
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': part_response['ETag']
                    })

                    part_number += 1

            # Complete multipart upload
            self._s3_client.complete_multipart_upload(
                Bucket=self._config.s3_bucket,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )

            # Return S3 URI
            return f"s3://{self._config.s3_bucket}/{s3_key}"

        except ClientError as e:
            # Abort multipart upload on error
            if upload_id:
                try:
                    self._s3_client.abort_multipart_upload(
                        Bucket=self._config.s3_bucket,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                except Exception:
                    pass  # Ignore abort errors

            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to upload file with multipart upload: {error_message}",
                bucket=self._config.s3_bucket,
                key=s3_key,
                error_code=error_code,
                request_id=request_id
            )

    def _generate_dataset_key(self, user_id: int, filename: str) -> str:
        """
        Generate S3 key for dataset with user isolation and timestamp.

        Format: datasets/user_{user_id}/{timestamp}_{filename}

        Args:
            user_id: User ID for isolation
            filename: Dataset filename

        Returns:
            str: S3 object key

        Example:
            >>> key = s3_manager._generate_dataset_key(12345, "housing.csv")
            >>> print(key)
            datasets/user_12345/20251023_143045_housing.csv
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build S3 key with user isolation
        key = f"{self._config.s3_data_prefix}/user_{user_id}/{timestamp}_{filename}"

        return key

    def save_model(
        self,
        user_id: int,
        model_id: str,
        model_dir: Path
    ) -> str:
        """
        Save model directory to S3 with versioning.

        Uploads all files in model directory to S3, preserving directory structure.
        Creates manifest.json with metadata for model tracking.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            model_dir: Path to model directory to upload

        Returns:
            str: S3 URI in format s3://bucket/models/user_{user_id}/{model_id}

        Raises:
            S3Error: If directory doesn't exist, is empty, or upload fails

        Example:
            >>> uri = s3_manager.save_model(
            ...     user_id=12345,
            ...     model_id="model_12345_random_forest",
            ...     model_dir=Path("/models/model_12345_random_forest")
            ... )
            >>> print(uri)
            s3://my-bucket/models/user_12345/model_12345_random_forest
        """
        # Convert to Path object
        model_dir = Path(model_dir)

        # Validate directory exists
        if not model_dir.exists():
            raise S3Error(
                message=f"Model directory not found: {model_dir}",
                bucket=self._config.s3_bucket,
                error_code="DirectoryNotFound"
            )

        # Get all files in directory (recursive)
        all_files = list(model_dir.rglob("*"))
        # Filter out directories, keep only files
        file_list = [f for f in all_files if f.is_file()]

        # Validate directory is not empty
        if not file_list:
            raise S3Error(
                message=f"Model directory is empty: {model_dir}",
                bucket=self._config.s3_bucket,
                error_code="EmptyDirectory"
            )

        # Build S3 key prefix
        s3_prefix = f"{self._config.s3_models_prefix}/user_{user_id}/{model_id}/"

        # Upload all files
        uploaded_files = []
        for file_path in file_list:
            # Get relative path from model_dir
            relative_path = file_path.relative_to(model_dir)

            # Build S3 key
            s3_key = f"{s3_prefix}{relative_path}"

            # Upload file using existing _simple_upload
            self._simple_upload(file_path=file_path, s3_key=s3_key)

            # Track uploaded file
            uploaded_files.append(str(relative_path))

        # Create manifest
        manifest_data = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': datetime.now().isoformat(),
            'files': uploaded_files,
            'version': '1.0'
        }

        # Upload manifest
        manifest_key = f"{s3_prefix}manifest.json"
        self._upload_json(s3_key=manifest_key, data=manifest_data)

        # Return S3 URI
        return f"s3://{self._config.s3_bucket}/{self._config.s3_models_prefix}/user_{user_id}/{model_id}"

    def load_model(
        self,
        user_id: int,
        model_id: str,
        local_dir: Path
    ) -> Path:
        """
        Load model from S3 to local directory.

        Downloads all model files from S3 based on manifest, preserving directory structure.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            local_dir: Local directory to download model to

        Returns:
            Path: Path to downloaded model directory (local_dir/model_id)

        Raises:
            S3Error: If manifest not found or file download fails

        Example:
            >>> model_path = s3_manager.load_model(
            ...     user_id=12345,
            ...     model_id="model_12345_random_forest",
            ...     local_dir=Path("/tmp/downloads")
            ... )
            >>> print(model_path)
            /tmp/downloads/model_12345_random_forest
        """
        # Convert to Path object
        local_dir = Path(local_dir)

        # Build S3 key prefix
        s3_prefix = f"{self._config.s3_models_prefix}/user_{user_id}/{model_id}/"

        # Download manifest first
        manifest_key = f"{s3_prefix}manifest.json"
        manifest_data = self._download_json(s3_key=manifest_key)

        # Create model directory
        model_local_dir = local_dir / model_id
        model_local_dir.mkdir(parents=True, exist_ok=True)

        # Download all files from manifest
        for file_relative_path in manifest_data['files']:
            # Build S3 key
            s3_key = f"{s3_prefix}{file_relative_path}"

            # Build local file path
            local_file_path = model_local_dir / file_relative_path

            # Create parent directories if needed
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            try:
                response = self._s3_client.get_object(
                    Bucket=self._config.s3_bucket,
                    Key=s3_key
                )

                # Write file content
                with open(local_file_path, 'wb') as f:
                    f.write(response['Body'].read())

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

                raise S3Error(
                    message=f"Failed to download model file from S3: {error_message}",
                    bucket=self._config.s3_bucket,
                    key=s3_key,
                    error_code=error_code,
                    request_id=request_id
                )

        # Return path to model directory
        return model_local_dir

    def _upload_json(self, s3_key: str, data: Dict[str, Any]) -> None:
        """
        Upload JSON data to S3.

        Helper method to serialize dict to JSON and upload to S3.

        Args:
            s3_key: S3 object key
            data: Dictionary to serialize and upload

        Raises:
            S3Error: If upload fails
        """
        try:
            # Serialize to JSON
            json_content = json.dumps(data)

            # Upload to S3
            self._s3_client.put_object(
                Bucket=self._config.s3_bucket,
                Key=s3_key,
                Body=json_content,
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to upload JSON to S3: {error_message}",
                bucket=self._config.s3_bucket,
                key=s3_key,
                error_code=error_code,
                request_id=request_id
            )

    def _download_json(self, s3_key: str) -> Dict[str, Any]:
        """
        Download and deserialize JSON from S3.

        Helper method to download JSON file from S3 and deserialize to dict.

        Args:
            s3_key: S3 object key

        Returns:
            dict: Deserialized JSON data

        Raises:
            S3Error: If download fails or JSON is invalid
        """
        try:
            # Download from S3
            response = self._s3_client.get_object(
                Bucket=self._config.s3_bucket,
                Key=s3_key
            )

            # Read and deserialize JSON
            json_content = response['Body'].read()
            data = json.loads(json_content)

            return data

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to download JSON from S3: {error_message}",
                bucket=self._config.s3_bucket,
                key=s3_key,
                error_code=error_code,
                request_id=request_id
            )

    def validate_s3_path(self, s3_uri: str, user_id: int) -> bool:
        """
        Validate S3 path belongs to user.

        Validates that S3 URI:
        1. Has valid format: s3://bucket/key
        2. Matches configured bucket
        3. Path starts with datasets/user_{user_id}/ OR models/user_{user_id}/

        Args:
            s3_uri: S3 URI to validate (e.g., s3://bucket/datasets/user_123/file.csv)
            user_id: User ID for ownership validation

        Returns:
            bool: True if path is valid and belongs to user

        Raises:
            S3Error: If URI format is invalid, bucket doesn't match, or access denied

        Example:
            >>> s3_manager.validate_s3_path(
            ...     s3_uri="s3://bucket/datasets/user_12345/data.csv",
            ...     user_id=12345
            ... )
            True
        """
        # Parse S3 URI using regex: s3://bucket/key
        uri_pattern = r'^s3://([^/]+)/(.+)$'
        match = re.match(uri_pattern, s3_uri)

        if not match:
            raise S3Error(
                message=f"Invalid S3 URI format: {s3_uri}. Expected: s3://bucket/key",
                bucket=self._config.s3_bucket,
                error_code="InvalidURI"
            )

        bucket = match.group(1)
        key = match.group(2)

        # Validate bucket matches config
        if bucket != self._config.s3_bucket:
            raise S3Error(
                message=f"Invalid bucket: {bucket}. Expected: {self._config.s3_bucket}",
                bucket=self._config.s3_bucket,
                error_code="InvalidBucket"
            )

        # Validate key starts with user-specific prefix
        dataset_prefix = f"{self._config.s3_data_prefix}/user_{user_id}/"
        model_prefix = f"{self._config.s3_models_prefix}/user_{user_id}/"

        if not (key.startswith(dataset_prefix) or key.startswith(model_prefix)):
            raise S3Error(
                message=f"Access denied: Path does not belong to user {user_id}",
                bucket=self._config.s3_bucket,
                key=key,
                error_code="AccessDenied"
            )

        return True

    def download_dataset(
        self,
        s3_uri: str,
        local_path: Path,
        user_id: int
    ) -> Path:
        """
        Download dataset from S3 to local path.

        Validates path ownership before downloading.

        Args:
            s3_uri: S3 URI of dataset to download
            local_path: Local path to save downloaded file
            user_id: User ID for ownership validation

        Returns:
            Path: Path to downloaded file (same as local_path)

        Raises:
            S3Error: If validation fails or download fails

        Example:
            >>> path = s3_manager.download_dataset(
            ...     s3_uri="s3://bucket/datasets/user_12345/data.csv",
            ...     local_path=Path("/tmp/data.csv"),
            ...     user_id=12345
            ... )
            >>> print(path)
            /tmp/data.csv
        """
        # Convert to Path object
        local_path = Path(local_path)

        # Validate path belongs to user
        self.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        # Parse S3 URI to extract bucket and key
        uri_pattern = r'^s3://([^/]+)/(.+)$'
        match = re.match(uri_pattern, s3_uri)
        bucket = match.group(1)
        key = match.group(2)

        # Download file
        try:
            response = self._s3_client.get_object(
                Bucket=bucket,
                Key=key
            )

            # Write file content to local path
            with open(local_path, 'wb') as f:
                f.write(response['Body'].read())

            return local_path

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to download dataset from S3: {error_message}",
                bucket=bucket,
                key=key,
                error_code=error_code,
                request_id=request_id
            )

    def list_user_datasets(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List all datasets for user.

        Args:
            user_id: User ID to list datasets for

        Returns:
            list[dict]: List of dataset dicts with keys:
                - key: S3 object key
                - s3_uri: Full S3 URI
                - size_mb: File size in MB
                - last_modified: Last modified timestamp
                - filename: Dataset filename

        Raises:
            S3Error: If list operation fails

        Example:
            >>> datasets = s3_manager.list_user_datasets(user_id=12345)
            >>> print(datasets[0])
            {
                'key': 'datasets/user_12345/20251023_143045_housing.csv',
                's3_uri': 's3://bucket/datasets/user_12345/20251023_143045_housing.csv',
                'size_mb': 1.5,
                'last_modified': '2025-10-23T14:30:45',
                'filename': '20251023_143045_housing.csv'
            }
        """
        # Build prefix for user's datasets
        prefix = f"{self._config.s3_data_prefix}/user_{user_id}/"

        try:
            # List objects with user prefix
            response = self._s3_client.list_objects_v2(
                Bucket=self._config.s3_bucket,
                Prefix=prefix
            )

            # Handle empty results
            if 'Contents' not in response:
                return []

            # Format results
            datasets = []
            for obj in response['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]

                datasets.append({
                    'key': key,
                    's3_uri': f"s3://{self._config.s3_bucket}/{key}",
                    'size_mb': obj['Size'] / (1024 * 1024),  # Convert bytes to MB
                    'last_modified': obj['LastModified'].isoformat(),
                    'filename': filename
                })

            return datasets

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to list user datasets: {error_message}",
                bucket=self._config.s3_bucket,
                error_code=error_code,
                request_id=request_id
            )

    def list_user_models(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List all models for user.

        Filters out manifest.json files to show only model files.

        Args:
            user_id: User ID to list models for

        Returns:
            list[dict]: List of model dicts with keys:
                - key: S3 object key
                - s3_uri: Full S3 URI
                - size_mb: File size in MB
                - last_modified: Last modified timestamp
                - filename: Model filename

        Raises:
            S3Error: If list operation fails

        Example:
            >>> models = s3_manager.list_user_models(user_id=12345)
            >>> print(models[0])
            {
                'key': 'models/user_12345/model_12345_random_forest/model.pkl',
                's3_uri': 's3://bucket/models/user_12345/model_12345_random_forest/model.pkl',
                'size_mb': 5.2,
                'last_modified': '2025-10-23T14:30:45',
                'filename': 'model.pkl'
            }
        """
        # Build prefix for user's models
        prefix = f"{self._config.s3_models_prefix}/user_{user_id}/"

        try:
            # List objects with user prefix
            response = self._s3_client.list_objects_v2(
                Bucket=self._config.s3_bucket,
                Prefix=prefix
            )

            # Handle empty results
            if 'Contents' not in response:
                return []

            # Format results (filter out manifest.json)
            models = []
            for obj in response['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]

                # Skip manifest.json files
                if filename == 'manifest.json':
                    continue

                models.append({
                    'key': key,
                    's3_uri': f"s3://{self._config.s3_bucket}/{key}",
                    'size_mb': obj['Size'] / (1024 * 1024),  # Convert bytes to MB
                    'last_modified': obj['LastModified'].isoformat(),
                    'filename': filename
                })

            return models

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to list user models: {error_message}",
                bucket=self._config.s3_bucket,
                error_code=error_code,
                request_id=request_id
            )

    def configure_bucket_lifecycle(self) -> None:
        """
        Configure S3 bucket lifecycle policy for automatic data cleanup.

        Creates two lifecycle rules:
        1. Dataset rule: Delete datasets after s3_lifecycle_days (default 90 days)
        2. Model rule: Transition to GLACIER after 30 days, delete after 180 days

        The lifecycle policy helps manage storage costs by automatically:
        - Deleting old datasets that are no longer needed
        - Moving old models to cheaper GLACIER storage
        - Eventually deleting very old models

        Raises:
            S3Error: If lifecycle configuration fails

        Example:
            >>> s3_manager.configure_bucket_lifecycle()
            # Lifecycle policy configured successfully
        """
        try:
            # Get lifecycle days from config or use default
            lifecycle_days = self._config.s3_lifecycle_days or 90

            # Build lifecycle configuration
            lifecycle_config = {
                'Rules': [
                    # Rule 1: Delete datasets after lifecycle_days
                    {
                        'ID': 'Dataset-Cleanup',
                        'Status': 'Enabled',
                        'Filter': {
                            'Prefix': f'{self._config.s3_data_prefix}/'
                        },
                        'Expiration': {
                            'Days': lifecycle_days
                        }
                    },
                    # Rule 2: Transition models to GLACIER after 30 days, delete after 180 days
                    {
                        'ID': 'Model-Archive-And-Cleanup',
                        'Status': 'Enabled',
                        'Filter': {
                            'Prefix': f'{self._config.s3_models_prefix}/'
                        },
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'GLACIER'
                            }
                        ],
                        'Expiration': {
                            'Days': 180
                        }
                    }
                ]
            }

            # Apply lifecycle configuration to bucket
            self._s3_client.put_bucket_lifecycle_configuration(
                Bucket=self._config.s3_bucket,
                LifecycleConfiguration=lifecycle_config
            )

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to configure bucket lifecycle policy: {error_message}",
                bucket=self._config.s3_bucket,
                error_code=error_code,
                request_id=request_id
            )

    def generate_presigned_download_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for downloading file from S3.

        Presigned URLs allow temporary access to S3 objects without AWS credentials.
        Useful for sharing datasets or model files with users or external systems.

        Args:
            s3_key: S3 object key to generate URL for
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)

        Returns:
            str: Presigned URL for downloading the file

        Raises:
            S3Error: If URL generation fails

        Example:
            >>> url = s3_manager.generate_presigned_download_url(
            ...     s3_key="datasets/user_12345/housing.csv",
            ...     expiration=7200
            ... )
            >>> print(url)
            https://my-bucket.s3.amazonaws.com/datasets/user_12345/housing.csv?X-Amz-Algorithm=...
        """
        try:
            # Generate presigned URL using boto3
            url = self._s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self._config.s3_bucket,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )

            return url

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to generate presigned download URL: {error_message}",
                bucket=self._config.s3_bucket,
                key=s3_key,
                error_code=error_code,
                request_id=request_id
            )

    def generate_presigned_upload_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for uploading file to S3.

        Presigned URLs allow temporary upload access to S3 without AWS credentials.
        Useful for allowing users to upload datasets directly to S3 from browser.

        Args:
            s3_key: S3 object key to generate URL for
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)

        Returns:
            str: Presigned URL for uploading the file

        Raises:
            S3Error: If URL generation fails

        Example:
            >>> url = s3_manager.generate_presigned_upload_url(
            ...     s3_key="datasets/user_12345/new_data.csv",
            ...     expiration=1800
            ... )
            >>> # Client can now PUT file to this URL
        """
        try:
            # Generate presigned URL using boto3
            url = self._s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self._config.s3_bucket,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )

            return url

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to generate presigned upload URL: {error_message}",
                bucket=self._config.s3_bucket,
                key=s3_key,
                error_code=error_code,
                request_id=request_id
            )

    def delete_dataset(self, s3_uri: str, user_id: int) -> None:
        """
        Delete dataset from S3.

        Validates that the dataset belongs to the user before deletion.
        Useful for manual cleanup or testing lifecycle policies.

        Args:
            s3_uri: S3 URI of dataset to delete
            user_id: User ID for ownership validation

        Raises:
            S3Error: If validation fails or deletion fails

        Example:
            >>> s3_manager.delete_dataset(
            ...     s3_uri="s3://bucket/datasets/user_12345/old_data.csv",
            ...     user_id=12345
            ... )
            # Dataset deleted successfully
        """
        # Validate path belongs to user
        self.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        # Parse S3 URI to extract bucket and key
        uri_pattern = r'^s3://([^/]+)/(.+)$'
        match = re.match(uri_pattern, s3_uri)
        bucket = match.group(1)
        key = match.group(2)

        # Delete object
        try:
            self._s3_client.delete_object(
                Bucket=bucket,
                Key=key
            )

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            request_id = e.response.get('ResponseMetadata', {}).get('RequestId', '')

            raise S3Error(
                message=f"Failed to delete dataset from S3: {error_message}",
                bucket=bucket,
                key=key,
                error_code=error_code,
                request_id=request_id
            )
