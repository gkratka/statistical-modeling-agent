"""
Unit tests for CloudProviderFactory.

Tests factory pattern for creating cloud provider instances (AWS, RunPod).

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 1.2: Provider Factory Implementation)
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.cloud.provider_factory import CloudProviderFactory
from src.cloud.provider_interface import (
    CloudStorageProvider,
    CloudTrainingProvider,
    CloudPredictionProvider
)
from src.cloud.s3_manager import S3Manager
from src.cloud.ec2_manager import EC2Manager
from src.cloud.lambda_manager import LambdaManager
from src.cloud.aws_config import CloudConfig
from src.cloud.aws_client import AWSClient


class TestCloudProviderFactory:
    """Test CloudProviderFactory creation methods."""

    @pytest.fixture
    def mock_aws_client(self):
        """Create mock AWSClient."""
        mock_client = Mock(spec=AWSClient)
        mock_client.get_s3_client.return_value = MagicMock()
        mock_client.get_ec2_client.return_value = MagicMock()
        mock_client.get_lambda_client.return_value = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_config(self):
        """Create mock CloudConfig."""
        return Mock(spec=CloudConfig)

    def test_create_aws_storage_provider(self, mock_aws_client, mock_config):
        """Test creating AWS storage provider returns S3Manager instance."""
        # Act
        provider = CloudProviderFactory.create_storage_provider(
            provider="aws",
            aws_client=mock_aws_client,
            config=mock_config
        )

        # Assert
        assert isinstance(provider, S3Manager)
        assert isinstance(provider, CloudStorageProvider)

    def test_create_aws_training_provider(self, mock_aws_client, mock_config):
        """Test creating AWS training provider returns EC2Manager instance."""
        # Act
        provider = CloudProviderFactory.create_training_provider(
            provider="aws",
            aws_client=mock_aws_client,
            config=mock_config
        )

        # Assert
        assert isinstance(provider, EC2Manager)
        assert isinstance(provider, CloudTrainingProvider)

    def test_create_aws_prediction_provider(self, mock_aws_client, mock_config):
        """Test creating AWS prediction provider returns LambdaManager instance."""
        # Act
        provider = CloudProviderFactory.create_prediction_provider(
            provider="aws",
            aws_client=mock_aws_client,
            config=mock_config
        )

        # Assert
        assert isinstance(provider, LambdaManager)
        assert isinstance(provider, CloudPredictionProvider)

    def test_create_runpod_storage_provider_raises_not_implemented(
        self, mock_aws_client, mock_config
    ):
        """Test creating RunPod storage provider raises NotImplementedError."""
        # Act & Assert
        with pytest.raises(NotImplementedError) as exc_info:
            CloudProviderFactory.create_storage_provider(
                provider="runpod",
                aws_client=mock_aws_client,
                config=mock_config
            )

        assert "RunPod storage provider not implemented yet" in str(exc_info.value)

    def test_create_runpod_training_provider_raises_not_implemented(
        self, mock_aws_client, mock_config
    ):
        """Test creating RunPod training provider raises NotImplementedError."""
        # Act & Assert
        with pytest.raises(NotImplementedError) as exc_info:
            CloudProviderFactory.create_training_provider(
                provider="runpod",
                aws_client=mock_aws_client,
                config=mock_config
            )

        assert "RunPod training provider not implemented yet" in str(exc_info.value)

    def test_create_runpod_prediction_provider_raises_not_implemented(
        self, mock_aws_client, mock_config
    ):
        """Test creating RunPod prediction provider raises NotImplementedError."""
        # Act & Assert
        with pytest.raises(NotImplementedError) as exc_info:
            CloudProviderFactory.create_prediction_provider(
                provider="runpod",
                aws_client=mock_aws_client,
                config=mock_config
            )

        assert "RunPod prediction provider not implemented yet" in str(exc_info.value)

    def test_invalid_storage_provider_raises_value_error(
        self, mock_aws_client, mock_config
    ):
        """Test creating storage provider with invalid provider name raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CloudProviderFactory.create_storage_provider(
                provider="invalid",  # type: ignore
                aws_client=mock_aws_client,
                config=mock_config
            )

        assert "Unsupported storage provider: invalid" in str(exc_info.value)
        assert "Supported providers: aws, runpod" in str(exc_info.value)

    def test_invalid_training_provider_raises_value_error(
        self, mock_aws_client, mock_config
    ):
        """Test creating training provider with invalid provider name raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CloudProviderFactory.create_training_provider(
                provider="gcp",  # type: ignore
                aws_client=mock_aws_client,
                config=mock_config
            )

        assert "Unsupported training provider: gcp" in str(exc_info.value)
        assert "Supported providers: aws, runpod" in str(exc_info.value)

    def test_invalid_prediction_provider_raises_value_error(
        self, mock_aws_client, mock_config
    ):
        """Test creating prediction provider with invalid provider name raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CloudProviderFactory.create_prediction_provider(
                provider="azure",  # type: ignore
                aws_client=mock_aws_client,
                config=mock_config
            )

        assert "Unsupported prediction provider: azure" in str(exc_info.value)
        assert "Supported providers: aws, runpod" in str(exc_info.value)

    def test_aws_storage_provider_has_correct_config(
        self, mock_aws_client, mock_config
    ):
        """Test AWS storage provider receives correct configuration."""
        # Act
        provider = CloudProviderFactory.create_storage_provider(
            provider="aws",
            aws_client=mock_aws_client,
            config=mock_config
        )

        # Assert - verify provider has access to config
        assert provider._config is mock_config
        assert provider._aws_client is mock_aws_client

    def test_aws_training_provider_has_correct_config(
        self, mock_aws_client, mock_config
    ):
        """Test AWS training provider receives correct configuration."""
        # Act
        provider = CloudProviderFactory.create_training_provider(
            provider="aws",
            aws_client=mock_aws_client,
            config=mock_config
        )

        # Assert - verify provider has access to config
        assert provider._config is mock_config
        assert provider._aws_client is mock_aws_client

    def test_aws_prediction_provider_has_correct_config(
        self, mock_aws_client, mock_config
    ):
        """Test AWS prediction provider receives correct configuration."""
        # Act
        provider = CloudProviderFactory.create_prediction_provider(
            provider="aws",
            aws_client=mock_aws_client,
            config=mock_config
        )

        # Assert - verify provider has access to config
        assert provider._config is mock_config
        assert provider._aws_client is mock_aws_client
