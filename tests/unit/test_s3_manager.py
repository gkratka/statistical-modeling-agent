"""
Unit tests for S3Manager class following TDD methodology.

Tests S3 upload operations including:
- Dataset upload with user isolation
- Simple upload (<5MB)
- Multipart upload (>5MB)
- Error handling and validation
- S3 key generation
- Metadata attachment

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 2.1: S3Manager with TDD)
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import S3Error
from src.cloud.s3_manager import S3Manager


@pytest.fixture
def mock_cloud_config():
    """Create mock CloudConfig for testing."""
    config = Mock(spec=CloudConfig)
    config.s3_bucket = "test-bucket"
    config.s3_data_prefix = "datasets"
    config.s3_models_prefix = "models"
    config.s3_results_prefix = "results"
    config.aws_region = "us-east-1"
    return config


@pytest.fixture
def mock_s3_client():
    """Create mock boto3 S3 client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_aws_client(mock_s3_client):
    """Create mock AWSClient with S3 client."""
    aws_client = Mock(spec=AWSClient)
    aws_client.get_s3_client.return_value = mock_s3_client
    return aws_client


@pytest.fixture
def s3_manager(mock_aws_client, mock_cloud_config):
    """Create S3Manager instance with mocked dependencies."""
    return S3Manager(aws_client=mock_aws_client, config=mock_cloud_config)


@pytest.fixture
def temp_small_file(tmp_path):
    """Create temporary small file (<5MB)."""
    file_path = tmp_path / "small_data.csv"
    content = "column1,column2\n" + "data,123\n" * 1000  # ~15KB
    file_path.write_text(content)
    return file_path


@pytest.fixture
def temp_large_file(tmp_path):
    """Create temporary large file (>5MB)."""
    file_path = tmp_path / "large_data.csv"
    # Create 6MB file
    content = "column1,column2\n" + ("x" * 1000 + "\n") * 6000
    file_path.write_text(content)
    return file_path


class TestS3ManagerInitialization:
    """Test S3Manager initialization."""

    def test_init_stores_dependencies(self, mock_aws_client, mock_cloud_config):
        """Test initialization stores AWS client and config."""
        manager = S3Manager(aws_client=mock_aws_client, config=mock_cloud_config)

        assert manager._aws_client == mock_aws_client
        assert manager._config == mock_cloud_config
        assert manager._s3_client == mock_aws_client.get_s3_client()

    def test_init_gets_s3_client(self, mock_aws_client, mock_cloud_config):
        """Test initialization retrieves S3 client from AWSClient."""
        manager = S3Manager(aws_client=mock_aws_client, config=mock_cloud_config)

        mock_aws_client.get_s3_client.assert_called_once()


class TestGenerateDatasetKey:
    """Test S3 key generation for datasets."""

    def test_generate_key_with_user_id(self, s3_manager):
        """Test key generation includes user ID prefix."""
        key = s3_manager._generate_dataset_key(user_id=12345, filename="data.csv")

        assert key.startswith("datasets/user_12345/")
        assert key.endswith("_data.csv")

    def test_generate_key_includes_timestamp(self, s3_manager):
        """Test key generation includes timestamp."""
        with patch('src.cloud.s3_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 10, 23, 14, 30, 45)

            key = s3_manager._generate_dataset_key(user_id=12345, filename="data.csv")

            assert "20251023_143045" in key

    def test_generate_key_format(self, s3_manager):
        """Test key follows format: datasets/user_{user_id}/{timestamp}_{filename}."""
        with patch('src.cloud.s3_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 10, 23, 14, 30, 45)

            key = s3_manager._generate_dataset_key(user_id=12345, filename="housing.csv")

            expected = "datasets/user_12345/20251023_143045_housing.csv"
            assert key == expected

    def test_generate_key_different_users(self, s3_manager):
        """Test key generation isolates different users."""
        key1 = s3_manager._generate_dataset_key(user_id=100, filename="data.csv")
        key2 = s3_manager._generate_dataset_key(user_id=200, filename="data.csv")

        assert "user_100" in key1
        assert "user_200" in key2
        assert key1 != key2

    def test_generate_key_preserves_filename(self, s3_manager):
        """Test key generation preserves original filename."""
        filenames = ["data.csv", "housing_prices.xlsx", "credit_data.parquet"]

        for filename in filenames:
            key = s3_manager._generate_dataset_key(user_id=12345, filename=filename)
            assert key.endswith(f"_{filename}")


class TestSimpleUpload:
    """Test simple upload for small files (<5MB)."""

    def test_simple_upload_calls_put_object(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload uses S3 put_object API."""
        s3_key = "datasets/user_12345/test.csv"

        s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        mock_s3_client.put_object.assert_called_once()

    def test_simple_upload_includes_bucket(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload specifies correct bucket."""
        s3_key = "datasets/user_12345/test.csv"

        s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs['Bucket'] == "test-bucket"

    def test_simple_upload_includes_key(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload uses provided S3 key."""
        s3_key = "datasets/user_12345/test.csv"

        s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs['Key'] == s3_key

    def test_simple_upload_includes_file_content(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload includes file content as Body."""
        s3_key = "datasets/user_12345/test.csv"

        s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert 'Body' in call_kwargs
        # Body should be file content
        assert len(call_kwargs['Body']) > 0

    def test_simple_upload_includes_encryption(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload enables server-side encryption (AES256)."""
        s3_key = "datasets/user_12345/test.csv"

        s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs['ServerSideEncryption'] == 'AES256'

    def test_simple_upload_includes_metadata(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload includes metadata: uploaded_at, original_filename."""
        s3_key = "datasets/user_12345/data.csv"

        s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        metadata = call_kwargs['Metadata']

        assert 'uploaded_at' in metadata
        assert 'original_filename' in metadata
        assert metadata['original_filename'] == temp_small_file.name

    def test_simple_upload_returns_s3_uri(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload returns S3 URI format: s3://bucket/key."""
        s3_key = "datasets/user_12345/test.csv"

        result = s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        assert result == "s3://test-bucket/datasets/user_12345/test.csv"

    def test_simple_upload_handles_client_error(self, s3_manager, temp_small_file, mock_s3_client):
        """Test simple upload raises S3Error on ClientError."""
        s3_key = "datasets/user_12345/test.csv"

        # Mock ClientError
        error_response = {
            'Error': {
                'Code': 'AccessDenied',
                'Message': 'Access Denied'
            }
        }
        mock_s3_client.put_object.side_effect = ClientError(error_response, 'PutObject')

        with pytest.raises(S3Error) as exc_info:
            s3_manager._simple_upload(file_path=temp_small_file, s3_key=s3_key)

        assert exc_info.value.error_code == 'AccessDenied'
        assert exc_info.value.bucket == "test-bucket"
        assert exc_info.value.key == s3_key


class TestMultipartUpload:
    """Test multipart upload for large files (>5MB)."""

    def test_multipart_upload_creates_multipart(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload creates multipart upload."""
        s3_key = "datasets/user_12345/large.csv"
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}

        s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        mock_s3_client.create_multipart_upload.assert_called_once()

    def test_multipart_upload_includes_encryption(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload enables server-side encryption."""
        s3_key = "datasets/user_12345/large.csv"
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}

        s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.create_multipart_upload.call_args[1]
        assert call_kwargs['ServerSideEncryption'] == 'AES256'

    def test_multipart_upload_includes_metadata(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload includes metadata."""
        s3_key = "datasets/user_12345/large.csv"
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}

        s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        call_kwargs = mock_s3_client.create_multipart_upload.call_args[1]
        metadata = call_kwargs['Metadata']

        assert 'uploaded_at' in metadata
        assert 'original_filename' in metadata

    def test_multipart_upload_uploads_parts(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload uploads file in parts."""
        s3_key = "datasets/user_12345/large.csv"
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}
        mock_s3_client.upload_part.return_value = {'ETag': 'test-etag'}

        s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        # Should upload at least one part
        assert mock_s3_client.upload_part.call_count >= 1

    def test_multipart_upload_completes_upload(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload completes the upload."""
        s3_key = "datasets/user_12345/large.csv"
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}
        mock_s3_client.upload_part.return_value = {'ETag': 'test-etag'}

        s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        mock_s3_client.complete_multipart_upload.assert_called_once()

    def test_multipart_upload_returns_s3_uri(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload returns S3 URI."""
        s3_key = "datasets/user_12345/large.csv"
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}
        mock_s3_client.upload_part.return_value = {'ETag': 'test-etag'}

        result = s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        assert result == "s3://test-bucket/datasets/user_12345/large.csv"

    def test_multipart_upload_aborts_on_error(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload aborts on error."""
        s3_key = "datasets/user_12345/large.csv"
        upload_id = 'test-upload-id'
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': upload_id}

        # Mock upload_part error
        error_response = {'Error': {'Code': 'InternalError', 'Message': 'Internal Error'}}
        mock_s3_client.upload_part.side_effect = ClientError(error_response, 'UploadPart')

        with pytest.raises(S3Error):
            s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        # Should abort the upload
        mock_s3_client.abort_multipart_upload.assert_called_once_with(
            Bucket="test-bucket",
            Key=s3_key,
            UploadId=upload_id
        )

    def test_multipart_upload_handles_client_error(self, s3_manager, temp_large_file, mock_s3_client):
        """Test multipart upload raises S3Error on ClientError."""
        s3_key = "datasets/user_12345/large.csv"

        error_response = {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket not found'}}
        mock_s3_client.create_multipart_upload.side_effect = ClientError(error_response, 'CreateMultipartUpload')

        with pytest.raises(S3Error) as exc_info:
            s3_manager._multipart_upload(file_path=temp_large_file, s3_key=s3_key)

        assert exc_info.value.error_code == 'NoSuchBucket'


class TestUploadDataset:
    """Test high-level dataset upload method."""

    def test_upload_dataset_validates_file_exists(self, s3_manager):
        """Test upload_dataset raises S3Error if file doesn't exist."""
        with pytest.raises(S3Error) as exc_info:
            s3_manager.upload_dataset(
                user_id=12345,
                file_path="/nonexistent/file.csv"
            )

        assert "not found" in str(exc_info.value).lower()

    def test_upload_dataset_uses_simple_upload_for_small_files(
        self, s3_manager, temp_small_file, mock_s3_client
    ):
        """Test upload_dataset uses simple upload for files <5MB."""
        s3_manager.upload_dataset(user_id=12345, file_path=str(temp_small_file))

        # Should call put_object, not multipart upload
        mock_s3_client.put_object.assert_called_once()
        mock_s3_client.create_multipart_upload.assert_not_called()

    def test_upload_dataset_uses_multipart_for_large_files(
        self, s3_manager, temp_large_file, mock_s3_client
    ):
        """Test upload_dataset uses multipart upload for files >5MB."""
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}
        mock_s3_client.upload_part.return_value = {'ETag': 'test-etag'}

        s3_manager.upload_dataset(user_id=12345, file_path=str(temp_large_file))

        # Should call multipart upload
        mock_s3_client.create_multipart_upload.assert_called_once()
        mock_s3_client.put_object.assert_not_called()

    def test_upload_dataset_generates_key_with_user_id(
        self, s3_manager, temp_small_file, mock_s3_client
    ):
        """Test upload_dataset generates key with user isolation."""
        s3_manager.upload_dataset(user_id=12345, file_path=str(temp_small_file))

        call_kwargs = mock_s3_client.put_object.call_args[1]
        key = call_kwargs['Key']

        assert key.startswith("datasets/user_12345/")

    def test_upload_dataset_uses_custom_dataset_name(
        self, s3_manager, temp_small_file, mock_s3_client
    ):
        """Test upload_dataset uses custom dataset_name if provided."""
        s3_manager.upload_dataset(
            user_id=12345,
            file_path=str(temp_small_file),
            dataset_name="custom_housing_data.csv"
        )

        call_kwargs = mock_s3_client.put_object.call_args[1]
        key = call_kwargs['Key']

        assert key.endswith("_custom_housing_data.csv")

    def test_upload_dataset_uses_filename_by_default(
        self, s3_manager, temp_small_file, mock_s3_client
    ):
        """Test upload_dataset uses file's name if dataset_name not provided."""
        s3_manager.upload_dataset(user_id=12345, file_path=str(temp_small_file))

        call_kwargs = mock_s3_client.put_object.call_args[1]
        key = call_kwargs['Key']

        assert temp_small_file.name in key

    def test_upload_dataset_returns_s3_uri(self, s3_manager, temp_small_file, mock_s3_client):
        """Test upload_dataset returns S3 URI."""
        result = s3_manager.upload_dataset(user_id=12345, file_path=str(temp_small_file))

        assert result.startswith("s3://test-bucket/datasets/user_12345/")
        assert temp_small_file.name in result

    def test_upload_dataset_with_path_object(self, s3_manager, temp_small_file, mock_s3_client):
        """Test upload_dataset accepts Path objects."""
        result = s3_manager.upload_dataset(user_id=12345, file_path=temp_small_file)

        assert result.startswith("s3://")
        mock_s3_client.put_object.assert_called_once()

    def test_upload_dataset_handles_upload_errors(self, s3_manager, temp_small_file, mock_s3_client):
        """Test upload_dataset propagates S3Error from upload methods."""
        error_response = {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket not found'}}
        mock_s3_client.put_object.side_effect = ClientError(error_response, 'PutObject')

        with pytest.raises(S3Error) as exc_info:
            s3_manager.upload_dataset(user_id=12345, file_path=str(temp_small_file))

        assert exc_info.value.error_code == 'NoSuchBucket'


class TestUploadDatasetIntegration:
    """Integration tests for complete upload workflow."""

    def test_upload_small_dataset_complete_workflow(
        self, s3_manager, temp_small_file, mock_s3_client
    ):
        """Test complete workflow for small dataset upload."""
        user_id = 12345

        result = s3_manager.upload_dataset(
            user_id=user_id,
            file_path=str(temp_small_file),
            dataset_name="housing_data.csv"
        )

        # Verify S3 URI returned
        assert result.startswith(f"s3://test-bucket/datasets/user_{user_id}/")
        assert "housing_data.csv" in result

        # Verify put_object called with correct parameters
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs['Bucket'] == "test-bucket"
        assert call_kwargs['ServerSideEncryption'] == 'AES256'
        assert 'uploaded_at' in call_kwargs['Metadata']

    def test_upload_large_dataset_complete_workflow(
        self, s3_manager, temp_large_file, mock_s3_client
    ):
        """Test complete workflow for large dataset upload."""
        user_id = 67890
        mock_s3_client.create_multipart_upload.return_value = {'UploadId': 'test-upload-id'}
        mock_s3_client.upload_part.return_value = {'ETag': 'test-etag'}

        result = s3_manager.upload_dataset(
            user_id=user_id,
            file_path=str(temp_large_file)
        )

        # Verify S3 URI returned
        assert result.startswith(f"s3://test-bucket/datasets/user_{user_id}/")
        assert temp_large_file.name in result

        # Verify multipart upload workflow
        mock_s3_client.create_multipart_upload.assert_called_once()
        assert mock_s3_client.upload_part.call_count >= 1
        mock_s3_client.complete_multipart_upload.assert_called_once()

    def test_upload_with_different_users_isolated(
        self, s3_manager, temp_small_file, mock_s3_client
    ):
        """Test uploads from different users are isolated."""
        result1 = s3_manager.upload_dataset(user_id=100, file_path=str(temp_small_file))
        result2 = s3_manager.upload_dataset(user_id=200, file_path=str(temp_small_file))

        assert "user_100" in result1
        assert "user_200" in result2
        assert result1 != result2


class TestS3ManagerModelOperations:
    """Test S3Manager model save/load operations with versioning."""

    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create temporary model directory with files."""
        model_dir = tmp_path / "model_12345_random_forest"
        model_dir.mkdir()

        # Create model files
        (model_dir / "model.pkl").write_bytes(b"model_binary_data" * 100)
        (model_dir / "metadata.json").write_text('{"version": "1.0", "type": "random_forest"}')
        (model_dir / "preprocessor.pkl").write_bytes(b"preprocessor_data" * 50)

        # Create subdirectory with files
        subdir = model_dir / "checkpoints"
        subdir.mkdir()
        (subdir / "checkpoint_1.pkl").write_bytes(b"checkpoint_data")

        return model_dir

    @pytest.fixture
    def empty_model_dir(self, tmp_path):
        """Create empty model directory."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        return model_dir

    def test_save_model_uploads_all_files(self, s3_manager, temp_model_dir, mock_s3_client):
        """Test save_model uploads all files in model directory."""
        user_id = 12345
        model_id = "model_12345_random_forest"

        result = s3_manager.save_model(
            user_id=user_id,
            model_id=model_id,
            model_dir=temp_model_dir
        )

        # Should call put_object for each file + manifest
        # 3 files in root + 1 file in checkpoints subdir + 1 manifest = 5 total
        assert mock_s3_client.put_object.call_count == 5

    def test_save_model_preserves_directory_structure(self, s3_manager, temp_model_dir, mock_s3_client):
        """Test save_model preserves directory structure in S3."""
        user_id = 12345
        model_id = "model_12345_random_forest"

        s3_manager.save_model(
            user_id=user_id,
            model_id=model_id,
            model_dir=temp_model_dir
        )

        # Get all uploaded keys
        uploaded_keys = [call[1]['Key'] for call in mock_s3_client.put_object.call_args_list]

        # Should have checkpoint file with subdir path
        checkpoint_keys = [k for k in uploaded_keys if 'checkpoints/checkpoint_1.pkl' in k]
        assert len(checkpoint_keys) == 1

    def test_save_model_creates_manifest(self, s3_manager, temp_model_dir, mock_s3_client):
        """Test save_model creates manifest.json with metadata."""
        user_id = 12345
        model_id = "model_12345_random_forest"

        s3_manager.save_model(
            user_id=user_id,
            model_id=model_id,
            model_dir=temp_model_dir
        )

        # Find manifest upload call
        manifest_calls = [
            call for call in mock_s3_client.put_object.call_args_list
            if 'manifest.json' in call[1]['Key']
        ]
        assert len(manifest_calls) == 1

        # Verify manifest content
        import json
        manifest_body = manifest_calls[0][1]['Body']
        manifest_data = json.loads(manifest_body)

        assert manifest_data['model_id'] == model_id
        assert manifest_data['user_id'] == user_id
        assert 'uploaded_at' in manifest_data
        assert 'files' in manifest_data
        assert 'version' in manifest_data
        assert len(manifest_data['files']) == 4  # 3 root files + 1 checkpoint file

    def test_save_model_uses_correct_s3_key_format(self, s3_manager, temp_model_dir, mock_s3_client):
        """Test save_model uses format: models/user_{user_id}/{model_id}/"""
        user_id = 12345
        model_id = "model_12345_random_forest"

        s3_manager.save_model(
            user_id=user_id,
            model_id=model_id,
            model_dir=temp_model_dir
        )

        # All keys should start with models/user_12345/model_12345_random_forest/
        for call in mock_s3_client.put_object.call_args_list:
            key = call[1]['Key']
            assert key.startswith(f"models/user_{user_id}/{model_id}/")

    def test_save_model_returns_s3_uri(self, s3_manager, temp_model_dir, mock_s3_client):
        """Test save_model returns S3 URI."""
        user_id = 12345
        model_id = "model_12345_random_forest"

        result = s3_manager.save_model(
            user_id=user_id,
            model_id=model_id,
            model_dir=temp_model_dir
        )

        expected_uri = f"s3://test-bucket/models/user_{user_id}/{model_id}"
        assert result == expected_uri

    def test_save_model_raises_error_on_empty_directory(self, s3_manager, empty_model_dir):
        """Test save_model raises S3Error if model directory is empty."""
        with pytest.raises(S3Error) as exc_info:
            s3_manager.save_model(
                user_id=12345,
                model_id="model_12345_test",
                model_dir=empty_model_dir
            )

        assert "empty" in str(exc_info.value).lower()

    def test_save_model_raises_error_if_directory_not_exists(self, s3_manager, tmp_path):
        """Test save_model raises S3Error if directory doesn't exist."""
        nonexistent_dir = tmp_path / "nonexistent"

        with pytest.raises(S3Error) as exc_info:
            s3_manager.save_model(
                user_id=12345,
                model_id="model_12345_test",
                model_dir=nonexistent_dir
            )

        assert "not found" in str(exc_info.value).lower()

    def test_load_model_downloads_manifest_first(self, s3_manager, tmp_path, mock_s3_client):
        """Test load_model downloads manifest.json first."""
        user_id = 12345
        model_id = "model_12345_random_forest"
        local_dir = tmp_path / "downloads"
        local_dir.mkdir()

        # Mock manifest response
        import json
        manifest_data = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': '2025-10-23T10:00:00',
            'files': ['model.pkl', 'metadata.json'],
            'version': '1.0'
        }

        def get_object_side_effect(*args, **kwargs):
            key = kwargs['Key']
            response = {'Body': MagicMock()}
            if 'manifest.json' in key:
                response['Body'].read.return_value = json.dumps(manifest_data).encode()
            else:
                response['Body'].read.return_value = b"model_data"
            return response

        mock_s3_client.get_object.side_effect = get_object_side_effect

        s3_manager.load_model(user_id=user_id, model_id=model_id, local_dir=local_dir)

        # First get_object call should be for manifest
        first_call_key = mock_s3_client.get_object.call_args_list[0][1]['Key']
        assert 'manifest.json' in first_call_key

    def test_load_model_downloads_all_files_from_manifest(self, s3_manager, tmp_path, mock_s3_client):
        """Test load_model downloads all files listed in manifest."""
        user_id = 12345
        model_id = "model_12345_random_forest"
        local_dir = tmp_path / "downloads"
        local_dir.mkdir()

        # Mock manifest with multiple files
        import json
        manifest_data = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': '2025-10-23T10:00:00',
            'files': ['model.pkl', 'metadata.json', 'preprocessor.pkl'],
            'version': '1.0'
        }

        def get_object_side_effect(*args, **kwargs):
            response = {'Body': MagicMock()}
            key = kwargs['Key']
            if 'manifest.json' in key:
                response['Body'].read.return_value = json.dumps(manifest_data).encode()
            else:
                response['Body'].read.return_value = b"file_data"
            return response

        mock_s3_client.get_object.side_effect = get_object_side_effect

        s3_manager.load_model(user_id=user_id, model_id=model_id, local_dir=local_dir)

        # Should download manifest + 3 model files = 4 total
        assert mock_s3_client.get_object.call_count == 4

    def test_load_model_preserves_directory_structure(self, s3_manager, tmp_path, mock_s3_client):
        """Test load_model preserves directory structure when downloading."""
        user_id = 12345
        model_id = "model_12345_random_forest"
        local_dir = tmp_path / "downloads"
        local_dir.mkdir()

        # Mock manifest with nested file
        import json
        manifest_data = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': '2025-10-23T10:00:00',
            'files': ['model.pkl', 'checkpoints/checkpoint_1.pkl'],
            'version': '1.0'
        }

        def get_object_side_effect(*args, **kwargs):
            response = {'Body': MagicMock()}
            key = kwargs['Key']
            if 'manifest.json' in key:
                response['Body'].read.return_value = json.dumps(manifest_data).encode()
            else:
                response['Body'].read.return_value = b"file_data"
            return response

        mock_s3_client.get_object.side_effect = get_object_side_effect

        result_path = s3_manager.load_model(user_id=user_id, model_id=model_id, local_dir=local_dir)

        # Verify directory structure was created
        expected_model_dir = local_dir / model_id
        assert expected_model_dir.exists()
        assert (expected_model_dir / "model.pkl").exists()
        assert (expected_model_dir / "checkpoints").exists()
        assert (expected_model_dir / "checkpoints" / "checkpoint_1.pkl").exists()

    def test_load_model_returns_model_directory_path(self, s3_manager, tmp_path, mock_s3_client):
        """Test load_model returns Path to downloaded model directory."""
        user_id = 12345
        model_id = "model_12345_random_forest"
        local_dir = tmp_path / "downloads"
        local_dir.mkdir()

        import json
        manifest_data = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': '2025-10-23T10:00:00',
            'files': ['model.pkl'],
            'version': '1.0'
        }

        def get_object_side_effect(*args, **kwargs):
            response = {'Body': MagicMock()}
            key = kwargs['Key']
            if 'manifest.json' in key:
                response['Body'].read.return_value = json.dumps(manifest_data).encode()
            else:
                response['Body'].read.return_value = b"file_data"
            return response

        mock_s3_client.get_object.side_effect = get_object_side_effect

        result = s3_manager.load_model(user_id=user_id, model_id=model_id, local_dir=local_dir)

        expected_path = local_dir / model_id
        assert result == expected_path
        assert isinstance(result, Path)

    def test_load_model_raises_error_if_manifest_not_found(self, s3_manager, tmp_path, mock_s3_client):
        """Test load_model raises S3Error if manifest doesn't exist."""
        user_id = 12345
        model_id = "model_12345_nonexistent"
        local_dir = tmp_path / "downloads"
        local_dir.mkdir()

        # Mock manifest not found error
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        with pytest.raises(S3Error) as exc_info:
            s3_manager.load_model(user_id=user_id, model_id=model_id, local_dir=local_dir)

        assert exc_info.value.error_code == 'NoSuchKey'

    def test_load_model_raises_error_if_file_missing(self, s3_manager, tmp_path, mock_s3_client):
        """Test load_model raises S3Error if file from manifest is missing."""
        user_id = 12345
        model_id = "model_12345_random_forest"
        local_dir = tmp_path / "downloads"
        local_dir.mkdir()

        import json
        manifest_data = {
            'model_id': model_id,
            'user_id': user_id,
            'uploaded_at': '2025-10-23T10:00:00',
            'files': ['model.pkl', 'missing_file.pkl'],
            'version': '1.0'
        }

        call_count = 0
        def get_object_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            response = {'Body': MagicMock()}
            key = kwargs['Key']

            if 'manifest.json' in key:
                response['Body'].read.return_value = json.dumps(manifest_data).encode()
                return response
            elif call_count == 2:  # First file succeeds
                response['Body'].read.return_value = b"file_data"
                return response
            else:  # Second file fails
                error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'File not found'}}
                raise ClientError(error_response, 'GetObject')

        mock_s3_client.get_object.side_effect = get_object_side_effect

        with pytest.raises(S3Error) as exc_info:
            s3_manager.load_model(user_id=user_id, model_id=model_id, local_dir=local_dir)

        assert exc_info.value.error_code == 'NoSuchKey'

    def test_upload_json_serializes_dict_to_json(self, s3_manager, mock_s3_client):
        """Test _upload_json serializes dict to JSON and uploads."""
        s3_key = "models/user_12345/model_123/metadata.json"
        data = {
            'model_id': 'model_123',
            'version': '1.0',
            'metrics': {'accuracy': 0.95}
        }

        s3_manager._upload_json(s3_key=s3_key, data=data)

        # Verify put_object called
        mock_s3_client.put_object.assert_called_once()

        # Verify JSON serialization
        call_kwargs = mock_s3_client.put_object.call_args[1]
        import json
        uploaded_data = json.loads(call_kwargs['Body'])

        assert uploaded_data == data
        assert call_kwargs['Key'] == s3_key
        assert call_kwargs['ContentType'] == 'application/json'

    def test_download_json_deserializes_json_to_dict(self, s3_manager, mock_s3_client):
        """Test _download_json downloads and deserializes JSON to dict."""
        s3_key = "models/user_12345/model_123/metadata.json"
        expected_data = {
            'model_id': 'model_123',
            'version': '1.0',
            'metrics': {'accuracy': 0.95}
        }

        # Mock S3 response
        import json
        mock_response = {'Body': MagicMock()}
        mock_response['Body'].read.return_value = json.dumps(expected_data).encode()
        mock_s3_client.get_object.return_value = mock_response

        result = s3_manager._download_json(s3_key=s3_key)

        # Verify get_object called
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key=s3_key
        )

        # Verify deserialization
        assert result == expected_data

    def test_upload_json_handles_nested_structures(self, s3_manager, mock_s3_client):
        """Test _upload_json handles complex nested JSON structures."""
        s3_key = "models/user_12345/manifest.json"
        data = {
            'files': [
                {'name': 'model.pkl', 'size': 1024},
                {'name': 'preprocessor.pkl', 'size': 512}
            ],
            'metadata': {
                'nested': {
                    'deeply': {
                        'value': 42
                    }
                }
            }
        }

        s3_manager._upload_json(s3_key=s3_key, data=data)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        import json
        uploaded_data = json.loads(call_kwargs['Body'])

        assert uploaded_data == data
        assert uploaded_data['files'][0]['name'] == 'model.pkl'
        assert uploaded_data['metadata']['nested']['deeply']['value'] == 42


class TestS3ManagerListingAndDownload:
    """Test S3Manager listing and download operations."""

    def test_validate_s3_path_valid_dataset_path(self, s3_manager):
        """Test validate_s3_path accepts valid dataset S3 URI."""
        user_id = 12345
        s3_uri = "s3://test-bucket/datasets/user_12345/20251023_143045_housing.csv"

        # Should return True for valid path
        result = s3_manager.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        assert result is True

    def test_validate_s3_path_valid_model_path(self, s3_manager):
        """Test validate_s3_path accepts valid model S3 URI."""
        user_id = 12345
        s3_uri = "s3://test-bucket/models/user_12345/model_12345_random_forest/model.pkl"

        # Should return True for valid path
        result = s3_manager.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        assert result is True

    def test_validate_s3_path_invalid_bucket(self, s3_manager):
        """Test validate_s3_path rejects wrong bucket."""
        user_id = 12345
        s3_uri = "s3://wrong-bucket/datasets/user_12345/data.csv"

        # Should raise S3Error for wrong bucket
        with pytest.raises(S3Error) as exc_info:
            s3_manager.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        assert "bucket" in str(exc_info.value).lower()

    def test_validate_s3_path_wrong_user_id(self, s3_manager):
        """Test validate_s3_path rejects path with different user_id."""
        user_id = 12345
        s3_uri = "s3://test-bucket/datasets/user_67890/data.csv"

        # Should raise S3Error for access denied
        with pytest.raises(S3Error) as exc_info:
            s3_manager.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        assert "access denied" in str(exc_info.value).lower()

    def test_validate_s3_path_invalid_uri_format(self, s3_manager):
        """Test validate_s3_path rejects invalid URI format."""
        user_id = 12345
        invalid_uris = [
            "http://test-bucket/datasets/user_12345/data.csv",
            "s3:/test-bucket/datasets/user_12345/data.csv",
            "test-bucket/datasets/user_12345/data.csv",
            "s3://",
            ""
        ]

        for invalid_uri in invalid_uris:
            with pytest.raises(S3Error) as exc_info:
                s3_manager.validate_s3_path(s3_uri=invalid_uri, user_id=user_id)

            assert "invalid" in str(exc_info.value).lower()

    def test_validate_s3_path_not_dataset_or_model_prefix(self, s3_manager):
        """Test validate_s3_path rejects paths not in datasets/ or models/."""
        user_id = 12345
        s3_uri = "s3://test-bucket/results/user_12345/output.csv"

        # Should raise S3Error for invalid path
        with pytest.raises(S3Error) as exc_info:
            s3_manager.validate_s3_path(s3_uri=s3_uri, user_id=user_id)

        assert "access denied" in str(exc_info.value).lower()

    def test_download_dataset_successful(self, s3_manager, tmp_path, mock_s3_client):
        """Test download_dataset successfully downloads file from S3."""
        user_id = 12345
        s3_uri = "s3://test-bucket/datasets/user_12345/20251023_143045_housing.csv"
        local_path = tmp_path / "downloaded_housing.csv"

        # Mock S3 response
        mock_response = {'Body': MagicMock()}
        mock_response['Body'].read.return_value = b"column1,column2\ndata1,data2\n"
        mock_s3_client.get_object.return_value = mock_response

        result = s3_manager.download_dataset(
            s3_uri=s3_uri,
            local_path=local_path,
            user_id=user_id
        )

        # Should return Path to downloaded file
        assert result == local_path
        assert local_path.exists()
        assert local_path.read_bytes() == b"column1,column2\ndata1,data2\n"

    def test_download_dataset_validates_path_ownership(self, s3_manager, tmp_path, mock_s3_client):
        """Test download_dataset validates path belongs to user."""
        user_id = 12345
        # S3 URI with different user_id
        s3_uri = "s3://test-bucket/datasets/user_67890/data.csv"
        local_path = tmp_path / "downloaded.csv"

        # Should raise S3Error for access denied
        with pytest.raises(S3Error) as exc_info:
            s3_manager.download_dataset(
                s3_uri=s3_uri,
                local_path=local_path,
                user_id=user_id
            )

        assert "access denied" in str(exc_info.value).lower()

    def test_download_dataset_invalid_uri_format(self, s3_manager, tmp_path):
        """Test download_dataset rejects invalid S3 URI."""
        user_id = 12345
        s3_uri = "invalid-uri"
        local_path = tmp_path / "downloaded.csv"

        with pytest.raises(S3Error) as exc_info:
            s3_manager.download_dataset(
                s3_uri=s3_uri,
                local_path=local_path,
                user_id=user_id
            )

        assert "invalid" in str(exc_info.value).lower()

    def test_download_dataset_file_not_found(self, s3_manager, tmp_path, mock_s3_client):
        """Test download_dataset handles file not found in S3."""
        user_id = 12345
        s3_uri = "s3://test-bucket/datasets/user_12345/nonexistent.csv"
        local_path = tmp_path / "downloaded.csv"

        # Mock NoSuchKey error
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        with pytest.raises(S3Error) as exc_info:
            s3_manager.download_dataset(
                s3_uri=s3_uri,
                local_path=local_path,
                user_id=user_id
            )

        assert exc_info.value.error_code == 'NoSuchKey'

    def test_download_dataset_wrong_bucket(self, s3_manager, tmp_path):
        """Test download_dataset rejects wrong bucket."""
        user_id = 12345
        s3_uri = "s3://wrong-bucket/datasets/user_12345/data.csv"
        local_path = tmp_path / "downloaded.csv"

        with pytest.raises(S3Error) as exc_info:
            s3_manager.download_dataset(
                s3_uri=s3_uri,
                local_path=local_path,
                user_id=user_id
            )

        assert "bucket" in str(exc_info.value).lower()

    def test_list_user_datasets_returns_formatted_list(self, s3_manager, mock_s3_client):
        """Test list_user_datasets returns list of dataset dicts."""
        user_id = 12345

        # Mock S3 list_objects_v2 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'datasets/user_12345/20251023_143045_housing.csv',
                    'Size': 1024000,  # 1MB
                    'LastModified': datetime(2025, 10, 23, 14, 30, 45)
                },
                {
                    'Key': 'datasets/user_12345/20251024_101520_credit.csv',
                    'Size': 2048000,  # 2MB
                    'LastModified': datetime(2025, 10, 24, 10, 15, 20)
                }
            ]
        }

        result = s3_manager.list_user_datasets(user_id=user_id)

        # Should return list with 2 datasets
        assert len(result) == 2

        # Check first dataset structure
        dataset1 = result[0]
        assert dataset1['key'] == 'datasets/user_12345/20251023_143045_housing.csv'
        assert dataset1['s3_uri'] == 's3://test-bucket/datasets/user_12345/20251023_143045_housing.csv'
        assert dataset1['size_mb'] == pytest.approx(1.0, rel=0.1)
        assert dataset1['filename'] == '20251023_143045_housing.csv'
        assert 'last_modified' in dataset1

    def test_list_user_datasets_empty_results(self, s3_manager, mock_s3_client):
        """Test list_user_datasets handles empty results."""
        user_id = 12345

        # Mock empty response
        mock_s3_client.list_objects_v2.return_value = {}

        result = s3_manager.list_user_datasets(user_id=user_id)

        # Should return empty list
        assert result == []

    def test_list_user_datasets_uses_correct_prefix(self, s3_manager, mock_s3_client):
        """Test list_user_datasets uses correct S3 prefix."""
        user_id = 12345

        mock_s3_client.list_objects_v2.return_value = {}

        s3_manager.list_user_datasets(user_id=user_id)

        # Verify list_objects_v2 called with correct prefix
        mock_s3_client.list_objects_v2.assert_called_once()
        call_kwargs = mock_s3_client.list_objects_v2.call_args[1]

        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['Prefix'] == 'datasets/user_12345/'

    def test_list_user_datasets_handles_client_error(self, s3_manager, mock_s3_client):
        """Test list_user_datasets handles S3 client errors."""
        user_id = 12345

        # Mock error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3_client.list_objects_v2.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(S3Error) as exc_info:
            s3_manager.list_user_datasets(user_id=user_id)

        assert exc_info.value.error_code == 'AccessDenied'

    def test_list_user_models_returns_formatted_list(self, s3_manager, mock_s3_client):
        """Test list_user_models returns list of model dicts."""
        user_id = 12345

        # Mock S3 response with model directories
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'models/user_12345/model_12345_random_forest/model.pkl',
                    'Size': 5242880,  # 5MB
                    'LastModified': datetime(2025, 10, 23, 14, 30, 45)
                },
                {
                    'Key': 'models/user_12345/model_12345_random_forest/metadata.json',
                    'Size': 1024,
                    'LastModified': datetime(2025, 10, 23, 14, 30, 45)
                },
                {
                    'Key': 'models/user_12345/model_12345_random_forest/manifest.json',
                    'Size': 512,
                    'LastModified': datetime(2025, 10, 23, 14, 30, 45)
                },
                {
                    'Key': 'models/user_12345/model_67890_logistic/model.pkl',
                    'Size': 2097152,  # 2MB
                    'LastModified': datetime(2025, 10, 24, 10, 15, 20)
                }
            ]
        }

        result = s3_manager.list_user_models(user_id=user_id)

        # Should return list with 3 files (manifest.json filtered, 2 .pkl + 1 metadata.json)
        assert len(result) == 3

        # Check structure
        assert result[0]['key'] == 'models/user_12345/model_12345_random_forest/model.pkl'
        assert result[0]['s3_uri'] == 's3://test-bucket/models/user_12345/model_12345_random_forest/model.pkl'
        assert 'size_mb' in result[0]
        assert 'filename' in result[0]

    def test_list_user_models_filters_manifest_files(self, s3_manager, mock_s3_client):
        """Test list_user_models filters out manifest.json files."""
        user_id = 12345

        # Mock response with manifest.json
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'models/user_12345/model_123/manifest.json',
                    'Size': 512,
                    'LastModified': datetime(2025, 10, 23, 14, 30, 45)
                },
                {
                    'Key': 'models/user_12345/model_123/model.pkl',
                    'Size': 1024,
                    'LastModified': datetime(2025, 10, 23, 14, 30, 45)
                }
            ]
        }

        result = s3_manager.list_user_models(user_id=user_id)

        # Should only have 1 item (manifest filtered out)
        assert len(result) == 1
        assert 'manifest.json' not in result[0]['key']

    def test_list_user_models_empty_results(self, s3_manager, mock_s3_client):
        """Test list_user_models handles empty results."""
        user_id = 12345

        # Mock empty response
        mock_s3_client.list_objects_v2.return_value = {}

        result = s3_manager.list_user_models(user_id=user_id)

        # Should return empty list
        assert result == []

    def test_list_user_models_uses_correct_prefix(self, s3_manager, mock_s3_client):
        """Test list_user_models uses correct S3 prefix."""
        user_id = 12345

        mock_s3_client.list_objects_v2.return_value = {}

        s3_manager.list_user_models(user_id=user_id)

        # Verify list_objects_v2 called with correct prefix
        mock_s3_client.list_objects_v2.assert_called_once()
        call_kwargs = mock_s3_client.list_objects_v2.call_args[1]

        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['Prefix'] == 'models/user_12345/'

    def test_list_user_models_handles_client_error(self, s3_manager, mock_s3_client):
        """Test list_user_models handles S3 client errors."""
        user_id = 12345

        # Mock error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3_client.list_objects_v2.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(S3Error) as exc_info:
            s3_manager.list_user_models(user_id=user_id)

        assert exc_info.value.error_code == 'AccessDenied'


class TestS3ManagerLifecycleAndPresigned:
    """Test S3Manager lifecycle policies and presigned URL generation."""

    def test_configure_bucket_lifecycle_creates_correct_rules(self, s3_manager, mock_s3_client, mock_cloud_config):
        """Test configure_bucket_lifecycle creates rules for datasets and models."""
        # Set lifecycle days in config
        mock_cloud_config.s3_lifecycle_days = 90

        s3_manager.configure_bucket_lifecycle()

        # Verify put_bucket_lifecycle_configuration called
        mock_s3_client.put_bucket_lifecycle_configuration.assert_called_once()

        # Extract lifecycle configuration
        call_kwargs = mock_s3_client.put_bucket_lifecycle_configuration.call_args[1]
        assert call_kwargs['Bucket'] == 'test-bucket'

        # Verify rules structure
        rules = call_kwargs['LifecycleConfiguration']['Rules']
        assert len(rules) == 2

        # Find dataset rule
        dataset_rule = next(r for r in rules if 'Dataset' in r['ID'])
        assert dataset_rule['Status'] == 'Enabled'
        assert dataset_rule['Filter']['Prefix'] == 'datasets/'
        assert dataset_rule['Expiration']['Days'] == 90

        # Find model rule
        model_rule = next(r for r in rules if 'Model' in r['ID'])
        assert model_rule['Status'] == 'Enabled'
        assert model_rule['Filter']['Prefix'] == 'models/'
        # Model rule should have transition to GLACIER after 30 days
        assert model_rule['Transitions'][0]['Days'] == 30
        assert model_rule['Transitions'][0]['StorageClass'] == 'GLACIER'
        # And expiration after 180 days
        assert model_rule['Expiration']['Days'] == 180

    def test_configure_bucket_lifecycle_uses_default_lifecycle_days(self, s3_manager, mock_s3_client, mock_cloud_config):
        """Test configure_bucket_lifecycle uses 90 days default."""
        # s3_lifecycle_days is None (not configured)
        mock_cloud_config.s3_lifecycle_days = None

        s3_manager.configure_bucket_lifecycle()

        call_kwargs = mock_s3_client.put_bucket_lifecycle_configuration.call_args[1]
        rules = call_kwargs['LifecycleConfiguration']['Rules']

        dataset_rule = next(r for r in rules if 'Dataset' in r['ID'])
        assert dataset_rule['Expiration']['Days'] == 90

    def test_configure_bucket_lifecycle_custom_lifecycle_days(self, s3_manager, mock_s3_client, mock_cloud_config):
        """Test configure_bucket_lifecycle respects custom lifecycle days."""
        # Custom lifecycle days
        mock_cloud_config.s3_lifecycle_days = 120

        s3_manager.configure_bucket_lifecycle()

        call_kwargs = mock_s3_client.put_bucket_lifecycle_configuration.call_args[1]
        rules = call_kwargs['LifecycleConfiguration']['Rules']

        dataset_rule = next(r for r in rules if 'Dataset' in r['ID'])
        assert dataset_rule['Expiration']['Days'] == 120

    def test_configure_bucket_lifecycle_error_handling(self, s3_manager, mock_s3_client):
        """Test configure_bucket_lifecycle raises S3Error on failure."""
        # Mock ClientError
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3_client.put_bucket_lifecycle_configuration.side_effect = ClientError(
            error_response, 'PutBucketLifecycleConfiguration'
        )

        with pytest.raises(S3Error) as exc_info:
            s3_manager.configure_bucket_lifecycle()

        assert exc_info.value.error_code == 'AccessDenied'
        assert 'lifecycle' in str(exc_info.value).lower()

    def test_generate_presigned_download_url_creates_valid_url(self, s3_manager, mock_s3_client):
        """Test generate_presigned_download_url creates URL for downloading."""
        s3_key = "datasets/user_12345/20251023_143045_housing.csv"
        expected_url = "https://test-bucket.s3.amazonaws.com/datasets/user_12345/20251023_143045_housing.csv?signature=..."

        # Mock presigned URL generation
        mock_s3_client.generate_presigned_url.return_value = expected_url

        result = s3_manager.generate_presigned_download_url(s3_key=s3_key)

        # Verify generate_presigned_url called with correct parameters
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            'get_object',
            Params={
                'Bucket': 'test-bucket',
                'Key': s3_key
            },
            ExpiresIn=3600
        )

        # Verify URL returned
        assert result == expected_url

    def test_generate_presigned_download_url_custom_expiration(self, s3_manager, mock_s3_client):
        """Test generate_presigned_download_url respects custom expiration."""
        s3_key = "datasets/user_12345/data.csv"
        custom_expiration = 7200  # 2 hours

        mock_s3_client.generate_presigned_url.return_value = "https://example.com/presigned"

        s3_manager.generate_presigned_download_url(s3_key=s3_key, expiration=custom_expiration)

        # Verify expiration parameter passed
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs['ExpiresIn'] == 7200

    def test_generate_presigned_download_url_error_handling(self, s3_manager, mock_s3_client):
        """Test generate_presigned_download_url handles errors."""
        s3_key = "datasets/user_12345/data.csv"

        # Mock ClientError
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}}
        mock_s3_client.generate_presigned_url.side_effect = ClientError(
            error_response, 'GeneratePresignedUrl'
        )

        with pytest.raises(S3Error) as exc_info:
            s3_manager.generate_presigned_download_url(s3_key=s3_key)

        assert exc_info.value.error_code == 'NoSuchKey'
        assert 'presigned' in str(exc_info.value).lower()

    def test_generate_presigned_upload_url_creates_valid_url(self, s3_manager, mock_s3_client):
        """Test generate_presigned_upload_url creates URL for uploading."""
        s3_key = "datasets/user_12345/new_upload.csv"
        expected_url = "https://test-bucket.s3.amazonaws.com/datasets/user_12345/new_upload.csv?signature=..."

        # Mock presigned URL generation
        mock_s3_client.generate_presigned_url.return_value = expected_url

        result = s3_manager.generate_presigned_upload_url(s3_key=s3_key)

        # Verify generate_presigned_url called with put_object
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            'put_object',
            Params={
                'Bucket': 'test-bucket',
                'Key': s3_key
            },
            ExpiresIn=3600
        )

        # Verify URL returned
        assert result == expected_url

    def test_generate_presigned_upload_url_custom_expiration(self, s3_manager, mock_s3_client):
        """Test generate_presigned_upload_url respects custom expiration."""
        s3_key = "datasets/user_12345/data.csv"
        custom_expiration = 1800  # 30 minutes

        mock_s3_client.generate_presigned_url.return_value = "https://example.com/presigned"

        s3_manager.generate_presigned_upload_url(s3_key=s3_key, expiration=custom_expiration)

        # Verify expiration parameter passed
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs['ExpiresIn'] == 1800

    def test_generate_presigned_upload_url_error_handling(self, s3_manager, mock_s3_client):
        """Test generate_presigned_upload_url handles errors."""
        s3_key = "datasets/user_12345/data.csv"

        # Mock ClientError
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3_client.generate_presigned_url.side_effect = ClientError(
            error_response, 'GeneratePresignedUrl'
        )

        with pytest.raises(S3Error) as exc_info:
            s3_manager.generate_presigned_upload_url(s3_key=s3_key)

        assert exc_info.value.error_code == 'AccessDenied'

    def test_delete_dataset_successful_deletion(self, s3_manager, mock_s3_client):
        """Test delete_dataset successfully deletes dataset."""
        user_id = 12345
        s3_uri = "s3://test-bucket/datasets/user_12345/20251023_143045_housing.csv"

        s3_manager.delete_dataset(s3_uri=s3_uri, user_id=user_id)

        # Verify delete_object called
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='datasets/user_12345/20251023_143045_housing.csv'
        )

    def test_delete_dataset_validates_path_ownership(self, s3_manager, mock_s3_client):
        """Test delete_dataset validates path belongs to user."""
        user_id = 12345
        # S3 URI with different user_id
        s3_uri = "s3://test-bucket/datasets/user_67890/data.csv"

        # Should raise S3Error for access denied
        with pytest.raises(S3Error) as exc_info:
            s3_manager.delete_dataset(s3_uri=s3_uri, user_id=user_id)

        assert "access denied" in str(exc_info.value).lower()

        # delete_object should NOT be called
        mock_s3_client.delete_object.assert_not_called()

    def test_delete_dataset_invalid_uri_format(self, s3_manager):
        """Test delete_dataset rejects invalid S3 URI."""
        user_id = 12345
        s3_uri = "invalid-uri"

        with pytest.raises(S3Error) as exc_info:
            s3_manager.delete_dataset(s3_uri=s3_uri, user_id=user_id)

        assert "invalid" in str(exc_info.value).lower()

    def test_delete_dataset_error_handling(self, s3_manager, mock_s3_client):
        """Test delete_dataset handles S3 deletion errors."""
        user_id = 12345
        s3_uri = "s3://test-bucket/datasets/user_12345/data.csv"

        # Mock deletion error
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}}
        mock_s3_client.delete_object.side_effect = ClientError(error_response, 'DeleteObject')

        with pytest.raises(S3Error) as exc_info:
            s3_manager.delete_dataset(s3_uri=s3_uri, user_id=user_id)

        assert exc_info.value.error_code == 'NoSuchKey'
        assert 'delete' in str(exc_info.value).lower()
