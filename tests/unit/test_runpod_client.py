"""
Tests for RunPodClient.

This module tests the RunPod client wrapper including API connectivity,
storage access, and health checks.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.6: RunPod Configuration Tests)
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

# Mock runpod module before importing RunPodClient
sys.modules['runpod'] = MagicMock()

from src.cloud.runpod_client import RunPodClient
from src.cloud.runpod_config import RunPodConfig
import runpod


@pytest.fixture
def runpod_config():
    """Create valid RunPod configuration for testing."""
    return RunPodConfig(
        runpod_api_key="test-api-key-12345",
        storage_endpoint="https://storage.runpod.io",
        network_volume_id="vol-test123",
        default_gpu_type="NVIDIA RTX A5000",
        cloud_type="COMMUNITY",
        storage_access_key="AKIAIOSFODNN7EXAMPLE",
        storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        data_prefix="datasets",
        models_prefix="models",
        results_prefix="results",
        max_training_cost_dollars=10.0,
        max_prediction_cost_dollars=1.0
    )


@pytest.fixture
def runpod_client(runpod_config):
    """Create RunPodClient instance with mocked dependencies."""
    with patch('runpod.api_key'):
        client = RunPodClient(runpod_config)
        return client


class TestRunPodClientInitialization:
    """Test RunPodClient initialization."""

    @patch('runpod.api_key')
    def test_init_sets_runpod_api_key(self, mock_api_key, runpod_config):
        """Initialization should set runpod.api_key globally."""
        client = RunPodClient(runpod_config)

        assert client.config == runpod_config
        # runpod.api_key should be set to config value
        assert runpod.api_key == runpod_config.runpod_api_key

    @patch('runpod.api_key')
    def test_init_storage_client_lazy_initialized(self, mock_api_key, runpod_config):
        """Storage client should be lazy-initialized (not created in __init__)."""
        client = RunPodClient(runpod_config)

        assert client._storage_client is None


class TestRunPodClientStorageAccess:
    """Test RunPod storage client access."""

    @patch('runpod.api_key')
    @patch('boto3.client')
    def test_get_storage_client_creates_s3_client(self, mock_boto3, mock_api_key, runpod_config):
        """get_storage_client should create boto3 S3 client with RunPod endpoint."""
        mock_s3_client = Mock()
        mock_boto3.return_value = mock_s3_client

        client = RunPodClient(runpod_config)
        storage_client = client.get_storage_client()

        # Verify boto3.client was called with correct parameters
        mock_boto3.assert_called_once_with(
            's3',
            endpoint_url="https://storage.runpod.io",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        assert storage_client == mock_s3_client

    @patch('runpod.api_key')
    @patch('boto3.client')
    def test_get_storage_client_cached_on_second_call(self, mock_boto3, mock_api_key, runpod_config):
        """get_storage_client should return cached client on subsequent calls."""
        mock_s3_client = Mock()
        mock_boto3.return_value = mock_s3_client

        client = RunPodClient(runpod_config)

        # First call creates client
        storage_client1 = client.get_storage_client()
        # Second call returns cached client
        storage_client2 = client.get_storage_client()

        # boto3.client should only be called once
        mock_boto3.assert_called_once()

        # Both calls return same instance
        assert storage_client1 is storage_client2

    @patch('runpod.api_key')
    def test_get_storage_client_missing_credentials_raises_error(self, mock_api_key):
        """get_storage_client should raise ValueError if credentials missing."""
        config = RunPodConfig(
            runpod_api_key="test-key",
            storage_endpoint="https://storage.runpod.io",
            network_volume_id="vol-123",
            default_gpu_type="NVIDIA RTX A5000",
            cloud_type="COMMUNITY",
            storage_access_key="",  # Missing
            storage_secret_key="",  # Missing
            data_prefix="datasets",
            models_prefix="models",
            results_prefix="results"
        )

        client = RunPodClient(config)

        with pytest.raises(ValueError) as exc_info:
            client.get_storage_client()

        assert "storage credentials" in str(exc_info.value).lower()


class TestRunPodClientHealthCheck:
    """Test RunPod health check functionality."""

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_health_check_success_with_pods(self, mock_api_key, mock_get_pods, runpod_client):
        """Health check should return success when API accessible and pods exist."""
        # Mock API response with 2 pods
        mock_get_pods.return_value = [{'id': 'pod1'}, {'id': 'pod2'}]

        # Mock storage access test
        with patch.object(runpod_client, '_test_storage_access', return_value=True):
            health = runpod_client.health_check()

        assert health['api'] is True
        assert health['storage'] is True
        assert health['pod_count'] == 2
        assert health['volume_accessible'] is True
        assert health['error'] is None

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_health_check_success_with_no_pods(self, mock_api_key, mock_get_pods, runpod_client):
        """Health check should return success even with 0 pods."""
        mock_get_pods.return_value = []

        with patch.object(runpod_client, '_test_storage_access', return_value=True):
            health = runpod_client.health_check()

        assert health['api'] is True
        assert health['storage'] is True
        assert health['pod_count'] == 0

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_health_check_api_failure(self, mock_api_key, mock_get_pods, runpod_client):
        """Health check should handle API errors gracefully."""
        mock_get_pods.side_effect = Exception("API connection failed")

        with patch.object(runpod_client, '_test_storage_access', return_value=True):
            health = runpod_client.health_check()

        assert health['api'] is False
        assert health['pod_count'] == 0
        assert "API error" in health['error']

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_health_check_storage_failure(self, mock_api_key, mock_get_pods, runpod_client):
        """Health check should handle storage errors gracefully."""
        mock_get_pods.return_value = []

        with patch.object(runpod_client, '_test_storage_access', side_effect=Exception("Storage unavailable")):
            health = runpod_client.health_check()

        assert health['storage'] is False
        assert health['volume_accessible'] is False
        assert "Storage error" in health['error']


class TestRunPodClientStorageTest:
    """Test storage access testing."""

    @patch('runpod.api_key')
    @patch('boto3.client')
    def test_test_storage_access_success(self, mock_boto3, mock_api_key, runpod_config):
        """_test_storage_access should return True when volume accessible."""
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {}
        mock_boto3.return_value = mock_s3_client

        client = RunPodClient(runpod_config)
        result = client._test_storage_access()

        assert result is True
        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket="vol-test123",
            MaxKeys=1
        )

    @patch('runpod.api_key')
    @patch('boto3.client')
    def test_test_storage_access_no_such_bucket(self, mock_boto3, mock_api_key, runpod_config):
        """_test_storage_access should raise ValueError for non-existent volume."""
        mock_s3_client = Mock()
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_s3_client.list_objects_v2.side_effect = ClientError(error_response, 'ListObjectsV2')
        mock_boto3.return_value = mock_s3_client

        client = RunPodClient(runpod_config)

        with pytest.raises(ValueError) as exc_info:
            client._test_storage_access()

        assert "does not exist" in str(exc_info.value)
        assert "vol-test123" in str(exc_info.value)

    @patch('runpod.api_key')
    @patch('boto3.client')
    def test_test_storage_access_access_denied(self, mock_boto3, mock_api_key, runpod_config):
        """_test_storage_access should raise ValueError for invalid credentials."""
        mock_s3_client = Mock()
        error_response = {'Error': {'Code': 'AccessDenied'}}
        mock_s3_client.list_objects_v2.side_effect = ClientError(error_response, 'ListObjectsV2')
        mock_boto3.return_value = mock_s3_client

        client = RunPodClient(runpod_config)

        with pytest.raises(ValueError) as exc_info:
            client._test_storage_access()

        assert "access denied" in str(exc_info.value).lower()
        assert "storage_access_key" in str(exc_info.value).lower()


class TestRunPodClientUtilityMethods:
    """Test utility methods."""

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_verify_api_key_valid(self, mock_api_key, mock_get_pods, runpod_client):
        """verify_api_key should return True for valid API key."""
        mock_get_pods.return_value = []

        assert runpod_client.verify_api_key() is True

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_verify_api_key_invalid(self, mock_api_key, mock_get_pods, runpod_client):
        """verify_api_key should return False for invalid API key."""
        mock_get_pods.side_effect = Exception("Invalid API key")

        assert runpod_client.verify_api_key() is False

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_get_pod_count_success(self, mock_api_key, mock_get_pods, runpod_client):
        """get_pod_count should return number of pods."""
        mock_get_pods.return_value = [{'id': 'pod1'}, {'id': 'pod2'}, {'id': 'pod3'}]

        assert runpod_client.get_pod_count() == 3

    @patch('runpod.get_pods')
    @patch('runpod.api_key')
    def test_get_pod_count_api_failure(self, mock_api_key, mock_get_pods, runpod_client):
        """get_pod_count should return 0 on API failure."""
        mock_get_pods.side_effect = Exception("API error")

        assert runpod_client.get_pod_count() == 0

    @patch('runpod.api_key')
    def test_test_network_volume_access_success(self, mock_api_key, runpod_client):
        """test_network_volume_access should return detailed success result."""
        with patch.object(runpod_client, '_test_storage_access', return_value=True):
            result = runpod_client.test_network_volume_access()

        assert result['accessible'] is True
        assert result['volume_id'] == "vol-test123"
        assert result['error'] is None

    @patch('runpod.api_key')
    def test_test_network_volume_access_failure(self, mock_api_key, runpod_client):
        """test_network_volume_access should return detailed failure result."""
        with patch.object(runpod_client, '_test_storage_access', side_effect=ValueError("Volume not found")):
            result = runpod_client.test_network_volume_access()

        assert result['accessible'] is False
        assert result['volume_id'] == "vol-test123"
        assert "Volume not found" in result['error']
