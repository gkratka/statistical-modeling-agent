"""
Tests for RunPod Storage Manager.

This module tests the RunPodStorageManager including S3-compatible operations,
path validation, and URI format conversion.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 2.1: RunPod Storage Tests)
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.cloud.runpod_storage_manager import RunPodStorageManager
from src.cloud.runpod_config import RunPodConfig
from src.cloud.exceptions import S3Error


@pytest.fixture
def runpod_config():
    """Create valid RunPod configuration."""
    return RunPodConfig(
        runpod_api_key="test-api-key",
        storage_endpoint="https://storage.runpod.io",
        network_volume_id="vol-abc123",
        storage_access_key="AKIAIOSFODNN7EXAMPLE",
        storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        data_prefix="datasets",
        models_prefix="models",
        results_prefix="results"
    )


@pytest.fixture
def storage_manager(runpod_config):
    """Create RunPodStorageManager instance with mocked boto3 client."""
    with patch('boto3.client') as mock_boto3:
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        manager = RunPodStorageManager(runpod_config)
        manager._s3_client = mock_client

        yield manager


class TestRunPodStorageManagerInitialization:
    """Test RunPodStorageManager initialization."""

    @patch('boto3.client')
    def test_init_creates_s3_client_with_runpod_endpoint(self, mock_boto3, runpod_config):
        """Initialization should create S3 client with RunPod endpoint."""
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        manager = RunPodStorageManager(runpod_config)

        # Verify boto3.client was called with RunPod endpoint
        mock_boto3.assert_called_once()
        call_args = mock_boto3.call_args

        assert call_args[0][0] == 's3'
        assert call_args[1]['endpoint_url'] == "https://storage.runpod.io"
        assert call_args[1]['aws_access_key_id'] == "AKIAIOSFODNN7EXAMPLE"
        assert call_args[1]['aws_secret_access_key'] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

    @patch('boto3.client')
    def test_init_stores_volume_id(self, mock_boto3, runpod_config):
        """Initialization should store network volume ID."""
        mock_boto3.return_value = Mock()
        manager = RunPodStorageManager(runpod_config)

        assert manager._volume_id == "vol-abc123"
        assert manager.bucket == "vol-abc123"

    @patch('boto3.client')
    def test_init_stores_prefixes(self, mock_boto3, runpod_config):
        """Initialization should store data, models, and results prefixes."""
        mock_boto3.return_value = Mock()
        manager = RunPodStorageManager(runpod_config)

        assert manager._data_prefix == "datasets"
        assert manager._models_prefix == "models"
        assert manager._results_prefix == "results"


class TestRunPodPathValidation:
    """Test RunPod storage path validation."""

    def test_validate_runpod_uri_format(self, storage_manager):
        """Valid runpod:// URI should pass validation."""
        uri = "runpod://vol-abc123/datasets/user_12345/data.csv"
        assert storage_manager.validate_s3_path(uri, 12345) is True

    def test_validate_s3_uri_format(self, storage_manager):
        """Valid s3:// URI should pass validation."""
        uri = "s3://vol-abc123/datasets/user_12345/data.csv"
        assert storage_manager.validate_s3_path(uri, 12345) is True

    def test_validate_invalid_uri_prefix_raises_error(self, storage_manager):
        """Invalid URI prefix should raise S3Error."""
        uri = "http://vol-abc123/datasets/user_12345/data.csv"

        with pytest.raises(S3Error) as exc_info:
            storage_manager.validate_s3_path(uri, 12345)

        assert "runpod://" in str(exc_info.value).lower() or "s3://" in str(exc_info.value).lower()

    def test_validate_wrong_volume_id_raises_error(self, storage_manager):
        """Wrong volume ID should raise S3Error."""
        uri = "runpod://vol-wrong/datasets/user_12345/data.csv"

        with pytest.raises(S3Error) as exc_info:
            storage_manager.validate_s3_path(uri, 12345)

        assert "volume" in str(exc_info.value).lower()
        assert "vol-abc123" in str(exc_info.value)

    def test_validate_wrong_user_id_raises_error(self, storage_manager):
        """Path for different user should raise S3Error."""
        uri = "runpod://vol-abc123/datasets/user_99999/data.csv"

        with pytest.raises(S3Error) as exc_info:
            storage_manager.validate_s3_path(uri, 12345)

        assert "access denied" in str(exc_info.value).lower()
        assert "12345" in str(exc_info.value)

    def test_validate_invalid_prefix_raises_error(self, storage_manager):
        """Path outside allowed prefixes should raise S3Error."""
        uri = "runpod://vol-abc123/unauthorized/user_12345/data.csv"

        with pytest.raises(S3Error) as exc_info:
            storage_manager.validate_s3_path(uri, 12345)

        assert "access denied" in str(exc_info.value).lower()

    def test_validate_models_prefix_allowed(self, storage_manager):
        """Path with models prefix should be allowed."""
        uri = "runpod://vol-abc123/models/user_12345/model.pkl"
        assert storage_manager.validate_s3_path(uri, 12345) is True

    def test_validate_results_prefix_allowed(self, storage_manager):
        """Path with results prefix should be allowed."""
        uri = "runpod://vol-abc123/results/user_12345/predictions.csv"
        assert storage_manager.validate_s3_path(uri, 12345) is True


class TestRunPodUploadDataset:
    """Test RunPod dataset upload operations."""

    def test_upload_dataset_returns_runpod_uri(self, storage_manager):
        """Upload dataset should return runpod:// URI."""
        # Mock S3Manager's upload_dataset to return s3:// URI
        with patch.object(storage_manager.__class__.__bases__[0], 'upload_dataset', return_value='s3://vol-abc123/datasets/user_12345/test.csv'):
            uri = storage_manager.upload_dataset(12345, "/path/to/test.csv")

            assert uri.startswith("runpod://")
            assert "vol-abc123" in uri
            assert "user_12345" in uri

    def test_upload_dataset_calls_parent_method(self, storage_manager):
        """Upload dataset should call S3Manager's upload method."""
        with patch.object(storage_manager.__class__.__bases__[0], 'upload_dataset', return_value='s3://vol-abc123/datasets/user_12345/test.csv') as mock_upload:
            storage_manager.upload_dataset(12345, "/path/to/test.csv", "test_dataset")

            mock_upload.assert_called_once_with(12345, "/path/to/test.csv", "test_dataset")

    def test_upload_dataset_uri_format(self, storage_manager):
        """Upload dataset URI should have correct RunPod format."""
        with patch.object(storage_manager.__class__.__bases__[0], 'upload_dataset', return_value='s3://vol-abc123/datasets/user_12345/data.csv'):
            uri = storage_manager.upload_dataset(12345, "/path/to/data.csv")

            assert uri == "runpod://vol-abc123/datasets/user_12345/data.csv"


class TestRunPodSaveModel:
    """Test RunPod model save operations."""

    def test_save_model_returns_runpod_uri(self, storage_manager):
        """Save model should return runpod:// URI."""
        with patch.object(storage_manager.__class__.__bases__[0], 'save_model', return_value='s3://vol-abc123/models/user_12345/model_abc/'):
            uri = storage_manager.save_model(12345, "model_abc", Path("/path/to/model/"))

            assert uri.startswith("runpod://")
            assert "vol-abc123" in uri
            assert "user_12345" in uri
            assert "model_abc" in uri

    def test_save_model_calls_parent_method(self, storage_manager):
        """Save model should call S3Manager's save method."""
        with patch.object(storage_manager.__class__.__bases__[0], 'save_model', return_value='s3://vol-abc123/models/user_12345/model_abc/') as mock_save:
            model_dir = Path("/path/to/model/")
            storage_manager.save_model(12345, "model_abc", model_dir)

            mock_save.assert_called_once_with(12345, "model_abc", model_dir)

    def test_save_model_uri_format(self, storage_manager):
        """Save model URI should have correct RunPod format."""
        with patch.object(storage_manager.__class__.__bases__[0], 'save_model', return_value='s3://vol-abc123/models/user_12345/model_123/'):
            uri = storage_manager.save_model(12345, "model_123", Path("/path/to/model/"))

            assert uri == "runpod://vol-abc123/models/user_12345/model_123/"


class TestRunPodLoadModel:
    """Test RunPod model load operations."""

    def test_load_model_calls_parent_method(self, storage_manager):
        """Load model should call S3Manager's load method."""
        with patch.object(storage_manager.__class__.__bases__[0], 'load_model', return_value=Path("/tmp/model_abc")) as mock_load:
            local_dir = Path("/tmp/models/")
            result = storage_manager.load_model(12345, "model_abc", local_dir)

            mock_load.assert_called_once_with(12345, "model_abc", local_dir)
            assert result == Path("/tmp/model_abc")

    def test_load_model_returns_path(self, storage_manager):
        """Load model should return Path object."""
        with patch.object(storage_manager.__class__.__bases__[0], 'load_model', return_value=Path("/tmp/model")):
            result = storage_manager.load_model(12345, "model_123", Path("/tmp/"))

            assert isinstance(result, Path)


class TestRunPodURIGeneration:
    """Test RunPod URI generation."""

    def test_generate_s3_uri_format(self, storage_manager):
        """Generated URI should have runpod:// format."""
        uri = storage_manager._generate_s3_uri("datasets/user_12345/data.csv")

        assert uri == "runpod://vol-abc123/datasets/user_12345/data.csv"

    def test_generate_s3_uri_with_volume_id(self, storage_manager):
        """Generated URI should include volume ID."""
        uri = storage_manager._generate_s3_uri("models/user_123/model.pkl")

        assert "vol-abc123" in uri
        assert uri.startswith("runpod://vol-abc123/")


class TestRunPodStorageIntegration:
    """Integration tests for RunPod storage operations."""

    @patch('boto3.client')
    def test_full_upload_workflow(self, mock_boto3, runpod_config):
        """Full upload workflow should work end-to-end."""
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n")
            test_file = f.name

        try:
            manager = RunPodStorageManager(runpod_config)
            manager._s3_client = mock_client

            # Mock S3 upload
            mock_client.upload_file.return_value = None

            # Mock parent class behavior
            with patch.object(manager.__class__.__bases__[0], 'upload_dataset', return_value='s3://vol-abc123/datasets/user_12345/test.csv'):
                uri = manager.upload_dataset(12345, test_file, "test_data")

            assert uri.startswith("runpod://")
            assert "vol-abc123" in uri
        finally:
            import os
            os.unlink(test_file)

    @patch('boto3.client')
    def test_bucket_property(self, mock_boto3, runpod_config):
        """Bucket property should return volume ID."""
        mock_boto3.return_value = Mock()
        manager = RunPodStorageManager(runpod_config)

        assert manager.bucket == "vol-abc123"
