"""
RunPod Network Volume Storage Manager.

This module provides storage operations using RunPod's S3-compatible network volumes.
It extends S3Manager to work with RunPod's storage endpoint while maintaining the
same interface for uploading datasets, saving/loading models, and managing files.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 2.1: RunPod Storage Manager)
"""

import boto3
from pathlib import Path
from typing import Optional

from src.cloud.s3_manager import S3Manager
from src.cloud.runpod_config import RunPodConfig
from src.cloud.provider_interface import CloudStorageProvider
from src.cloud.exceptions import S3Error


class RunPodStorageManager(S3Manager, CloudStorageProvider):
    """
    RunPod network volume manager using S3-compatible API.

    RunPod provides S3-compatible network volumes that can be accessed using
    the standard boto3 S3 client with a custom endpoint. This class extends
    S3Manager and overrides the client initialization to point to RunPod's
    storage endpoint.

    All S3Manager methods (upload_dataset, save_model, load_model, etc.) work
    unchanged since RunPod's storage is S3-compatible.
    """

    def __init__(self, config: RunPodConfig):
        """
        Initialize RunPod storage manager.

        Args:
            config: RunPodConfig instance with storage credentials

        Example:
            >>> config = RunPodConfig.from_yaml("config.yaml")
            >>> storage = RunPodStorageManager(config)
            >>> uri = storage.upload_dataset(12345, "data.csv")
        """
        self.config = config

        # Create S3-compatible client with RunPod endpoint
        self._s3_client = boto3.client(
            's3',
            endpoint_url=config.storage_endpoint,
            aws_access_key_id=config.storage_access_key,
            aws_secret_access_key=config.storage_secret_key,
            # Disable signature verification for S3-compatible endpoints
            config=boto3.session.Config(signature_version='s3v4')
        )

        # Store RunPod-specific configuration
        self._volume_id = config.network_volume_id
        self._data_prefix = config.data_prefix
        self._models_prefix = config.models_prefix
        self._results_prefix = config.results_prefix

    def validate_s3_path(self, s3_uri: str, user_id: int) -> bool:
        """
        Validate RunPod network volume path.

        RunPod uses network volume IDs as bucket names in S3-compatible URIs.
        URIs can use either:
        - runpod://volume_id/path format
        - s3://volume_id/path format (S3-compatible)

        Args:
            s3_uri: RunPod storage URI (runpod://... or s3://...)
            user_id: User ID for ownership validation

        Returns:
            True if path is valid for user

        Raises:
            S3Error: If path is invalid or user doesn't have access

        Example:
            >>> storage.validate_s3_path("runpod://vol-123/datasets/user_12345/data.csv", 12345)
            True
        """
        # Accept both runpod:// and s3:// prefixes
        if s3_uri.startswith('runpod://'):
            # Convert runpod:// to s3:// for compatibility
            s3_uri = s3_uri.replace('runpod://', 's3://', 1)

        if not s3_uri.startswith('s3://'):
            raise S3Error(
                f"Invalid RunPod URI format: {s3_uri}. Must start with 'runpod://' or 's3://'",
                bucket=None,
                key=None
            )

        # Parse URI: s3://volume_id/prefix/user_id/file
        parts = s3_uri.replace('s3://', '').split('/', 1)
        if len(parts) < 2:
            raise S3Error(
                f"Invalid RunPod URI format: {s3_uri}",
                bucket=parts[0] if parts else None,
                key=None
            )

        volume_id, key = parts

        # Validate volume ID matches configured volume
        if volume_id != self._volume_id:
            raise S3Error(
                f"Volume ID mismatch: expected {self._volume_id}, got {volume_id}",
                bucket=volume_id,
                key=key
            )

        # Validate user isolation: path must contain user_id
        user_id_str = f"user_{user_id}"
        if user_id_str not in key:
            raise S3Error(
                f"Access denied: path does not belong to user {user_id}",
                bucket=volume_id,
                key=key
            )

        # Validate path is within allowed prefixes
        allowed_prefixes = (
            f"{self._data_prefix}/",
            f"{self._models_prefix}/",
            f"{self._results_prefix}/"
        )

        if not any(key.startswith(prefix) for prefix in allowed_prefixes):
            raise S3Error(
                f"Access denied: path must start with one of {allowed_prefixes}",
                bucket=volume_id,
                key=key
            )

        return True

    def _generate_s3_uri(self, key: str) -> str:
        """
        Generate RunPod storage URI from key.

        Args:
            key: S3 object key

        Returns:
            RunPod storage URI (runpod://volume_id/key)
        """
        return f"runpod://{self._volume_id}/{key}"

    def upload_dataset(
        self,
        user_id: int,
        file_path: str,
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Upload dataset to RunPod network volume.

        Inherits behavior from S3Manager but returns runpod:// URI.

        Args:
            user_id: User ID for isolation
            file_path: Local file path to upload
            dataset_name: Optional dataset name (defaults to filename)

        Returns:
            RunPod storage URI (runpod://volume_id/datasets/user_X/file)

        Example:
            >>> uri = storage.upload_dataset(12345, "data.csv")
            >>> print(uri)
            runpod://vol-abc123/datasets/user_12345/data.csv
        """
        # Call parent S3Manager method
        s3_uri = super().upload_dataset(user_id, file_path, dataset_name)

        # Convert s3:// to runpod:// for RunPod-specific URI format
        return s3_uri.replace('s3://', 'runpod://', 1)

    def save_model(
        self,
        user_id: int,
        model_id: str,
        model_dir: Path
    ) -> str:
        """
        Save model directory to RunPod network volume.

        Inherits behavior from S3Manager but returns runpod:// URI.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            model_dir: Local directory containing model files

        Returns:
            RunPod storage URI for model

        Example:
            >>> uri = storage.save_model(12345, "model_abc", Path("./models/"))
            >>> print(uri)
            runpod://vol-abc123/models/user_12345/model_abc/
        """
        # Call parent S3Manager method
        s3_uri = super().save_model(user_id, model_id, model_dir)

        # Convert s3:// to runpod:// for RunPod-specific URI format
        return s3_uri.replace('s3://', 'runpod://', 1)

    def load_model(
        self,
        user_id: int,
        model_id: str,
        local_dir: Path
    ) -> Path:
        """
        Load model from RunPod network volume to local directory.

        Accepts both runpod:// and s3:// URI formats for backward compatibility.
        Inherits behavior from S3Manager.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            local_dir: Local directory to download to

        Returns:
            Path to downloaded model directory

        Example:
            >>> path = storage.load_model(12345, "model_abc", Path("./temp/"))
            >>> print(path)
            /tmp/temp/model_abc
        """
        # Parent S3Manager.load_model expects s3:// URIs internally
        # No conversion needed here since it works with keys directly
        return super().load_model(user_id, model_id, local_dir)

    @property
    def bucket(self) -> str:
        """Return RunPod network volume ID (acts as S3 bucket)."""
        return self._volume_id
