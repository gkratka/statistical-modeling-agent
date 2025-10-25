"""
RunPod SDK client wrapper.

This module provides a wrapper around the RunPod SDK for managing
API connectivity and S3-compatible storage access.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.2: RunPod Client Wrapper)
"""

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    runpod = None  # type: ignore
    RUNPOD_AVAILABLE = False

import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError

from src.cloud.runpod_config import RunPodConfig


class RunPodClient:
    """RunPod SDK client wrapper for API and storage operations."""

    def __init__(self, config: RunPodConfig):
        """
        Initialize RunPod client.

        Args:
            config: RunPod configuration instance

        Raises:
            ImportError: If runpod package is not installed
        """
        if not RUNPOD_AVAILABLE:
            raise ImportError(
                "runpod package is not installed. "
                "Install it with: pip install runpod>=1.0.0"
            )

        self.config = config

        # Set RunPod API key globally
        if runpod:
            runpod.api_key = config.runpod_api_key

        # S3-compatible storage client (lazy-initialized)
        self._storage_client: Optional[Any] = None

    def get_storage_client(self) -> Any:
        """
        Get S3-compatible storage client for RunPod network volumes.

        Returns:
            boto3 S3 client configured for RunPod storage endpoint

        Raises:
            ValueError: If storage credentials are missing
        """
        if self._storage_client is None:
            if not self.config.storage_access_key or not self.config.storage_secret_key:
                raise ValueError(
                    "Storage credentials (storage_access_key and storage_secret_key) "
                    "are required for storage operations"
                )

            self._storage_client = boto3.client(
                's3',
                endpoint_url=self.config.storage_endpoint,
                aws_access_key_id=self.config.storage_access_key,
                aws_secret_access_key=self.config.storage_secret_key
            )

        return self._storage_client

    def health_check(self) -> Dict[str, Any]:
        """
        Check RunPod API and storage connectivity.

        Returns:
            Health check results with API status, storage status, and pod count

        Example:
            >>> client = RunPodClient(config)
            >>> health = client.health_check()
            >>> print(health)
            {
                'api': True,
                'storage': True,
                'pod_count': 0,
                'volume_accessible': True
            }
        """
        result: Dict[str, Any] = {
            'api': False,
            'storage': False,
            'pod_count': 0,
            'volume_accessible': False,
            'error': None
        }

        try:
            # Test RunPod API connectivity
            pods = runpod.get_pods()
            result['api'] = True
            result['pod_count'] = len(pods) if pods else 0

        except Exception as e:
            result['api'] = False
            result['error'] = f"API error: {str(e)}"

        # Test storage endpoint accessibility
        try:
            storage_healthy = self._test_storage_access()
            result['storage'] = storage_healthy
            result['volume_accessible'] = storage_healthy

        except Exception as e:
            result['storage'] = False
            result['volume_accessible'] = False
            if not result['error']:
                result['error'] = f"Storage error: {str(e)}"
            else:
                result['error'] += f"; Storage error: {str(e)}"

        return result

    def _test_storage_access(self) -> bool:
        """
        Test storage endpoint accessibility.

        Returns:
            True if storage is accessible, False otherwise

        Raises:
            Exception: If storage credentials are invalid or endpoint is unreachable
        """
        try:
            s3_client = self.get_storage_client()

            # Attempt to list objects in network volume
            # This verifies:
            # 1. Endpoint is reachable
            # 2. Credentials are valid
            # 3. Network volume exists and is accessible
            s3_client.list_objects_v2(
                Bucket=self.config.network_volume_id,
                MaxKeys=1  # Only check existence, don't fetch data
            )

            return True

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')

            # NoSuchBucket means volume doesn't exist
            if error_code == 'NoSuchBucket':
                raise ValueError(
                    f"Network volume '{self.config.network_volume_id}' does not exist. "
                    "Please create a network volume first."
                )

            # AccessDenied means credentials are invalid
            if error_code == 'AccessDenied':
                raise ValueError(
                    "Storage access denied. Please verify storage_access_key "
                    "and storage_secret_key are correct."
                )

            # Other errors
            raise

        except Exception as e:
            # Connection errors, endpoint unreachable, etc.
            raise Exception(f"Failed to access storage endpoint: {str(e)}")

    def verify_api_key(self) -> bool:
        """
        Verify RunPod API key is valid.

        Returns:
            True if API key is valid, False otherwise

        Example:
            >>> client = RunPodClient(config)
            >>> if client.verify_api_key():
            ...     print("API key is valid")
        """
        try:
            runpod.get_pods()
            return True
        except Exception:
            return False

    def get_pod_count(self) -> int:
        """
        Get count of active pods.

        Returns:
            Number of pods (0 if API call fails)
        """
        try:
            pods = runpod.get_pods()
            return len(pods) if pods else 0
        except Exception:
            return 0

    def test_network_volume_access(self) -> Dict[str, Any]:
        """
        Test network volume access and return detailed results.

        Returns:
            Dictionary with volume access details:
                - accessible: bool
                - volume_id: str
                - error: Optional[str]

        Example:
            >>> client = RunPodClient(config)
            >>> result = client.test_network_volume_access()
            >>> if result['accessible']:
            ...     print(f"Volume {result['volume_id']} is accessible")
        """
        result = {
            'accessible': False,
            'volume_id': self.config.network_volume_id,
            'error': None
        }

        try:
            self._test_storage_access()
            result['accessible'] = True
        except Exception as e:
            result['error'] = str(e)

        return result
