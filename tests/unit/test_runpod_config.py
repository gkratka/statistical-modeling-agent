"""
Tests for RunPod configuration management.

This module tests the RunPodConfig dataclass including YAML loading,
environment variable loading, and configuration validation.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.1: RunPod Configuration Tests)
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.cloud.exceptions import CloudConfigurationError
from src.cloud.runpod_config import RunPodConfig


class TestRunPodConfigValidation:
    """Test RunPodConfig validation logic."""

    def test_valid_minimal_config(self):
        """Valid minimal configuration should pass validation."""
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        config.validate()  # Should not raise

    def test_valid_full_config(self):
        """Valid full configuration should pass validation."""
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            storage_endpoint="https://storage.runpod.io",
            network_volume_id="vol-123abc",
            default_gpu_type="NVIDIA RTX A5000",
            cloud_type="COMMUNITY",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            data_prefix="datasets",
            models_prefix="models",
            results_prefix="results",
            max_training_cost_dollars=10.0,
            max_prediction_cost_dollars=1.0,
            cost_warning_threshold=0.8,
            docker_registry="myregistry.io/ml",
            serverless_endpoint_id="endpoint-123"
        )

        config.validate()  # Should not raise

    def test_missing_api_key_raises_error(self):
        """Missing API key should raise CloudConfigurationError."""
        config = RunPodConfig(
            runpod_api_key="",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "api key" in str(exc_info.value).lower()

    def test_missing_network_volume_raises_error(self):
        """Missing network volume ID should raise CloudConfigurationError."""
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "volume" in str(exc_info.value).lower()

    def test_missing_storage_credentials_raises_error(self):
        """Missing storage credentials should raise CloudConfigurationError."""
        # Missing access key
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "access key" in str(exc_info.value).lower()

        # Missing secret key
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key=""
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "secret key" in str(exc_info.value).lower()

    def test_invalid_cloud_type_raises_error(self):
        """Invalid cloud type should raise CloudConfigurationError."""
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            cloud_type="INVALID"
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "cloud type" in str(exc_info.value).lower()
        assert "COMMUNITY" in str(exc_info.value) or "SECURE" in str(exc_info.value)

    def test_invalid_storage_endpoint_raises_error(self):
        """Invalid storage endpoint should raise CloudConfigurationError."""
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            storage_endpoint="http://insecure.endpoint.com"
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "https" in str(exc_info.value).lower()

    def test_invalid_cost_threshold_raises_error(self):
        """Invalid cost threshold should raise CloudConfigurationError."""
        # Threshold too high
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            cost_warning_threshold=1.5
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "threshold" in str(exc_info.value).lower()

    def test_negative_cost_limit_raises_error(self):
        """Negative cost limits should raise CloudConfigurationError."""
        config = RunPodConfig(
            runpod_api_key="test-api-key",
            network_volume_id="vol-123abc",
            storage_access_key="AKIAIOSFODNN7EXAMPLE",
            storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            max_training_cost_dollars=-1.0
        )

        with pytest.raises(CloudConfigurationError) as exc_info:
            config.validate()

        assert "training cost" in str(exc_info.value).lower()


class TestRunPodConfigFromYAML:
    """Test RunPodConfig.from_yaml() method."""

    @pytest.fixture
    def valid_yaml_config(self) -> str:
        """Create valid YAML configuration content."""
        return """
runpod:
  api_key: test-api-key-12345
  storage_endpoint: https://storage.runpod.io
  network_volume_id: vol-abc123
  default_gpu_type: NVIDIA RTX A5000
  cloud_type: COMMUNITY
  storage_access_key: AKIAIOSFODNN7EXAMPLE
  storage_secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  data_prefix: datasets
  models_prefix: models
  results_prefix: results
  max_training_cost_dollars: 10.0
  max_prediction_cost_dollars: 1.0
  cost_warning_threshold: 0.8
  docker_registry: myregistry.io/ml
  serverless_endpoint_id: endpoint-123
"""

    @pytest.fixture
    def valid_nested_yaml_config(self) -> str:
        """Create valid nested YAML configuration (cloud.runpod structure)."""
        return """
cloud:
  provider: runpod
  runpod:
    api_key: test-api-key-12345
    storage_endpoint: https://storage.runpod.io
    network_volume_id: vol-abc123
    default_gpu_type: NVIDIA RTX A5000
    cloud_type: COMMUNITY
    storage_access_key: AKIAIOSFODNN7EXAMPLE
    storage_secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    data_prefix: datasets
    models_prefix: models
    results_prefix: results
    max_training_cost_dollars: 10.0
    max_prediction_cost_dollars: 1.0
    cost_warning_threshold: 0.8
"""

    def test_load_from_valid_yaml_file(self, valid_yaml_config):
        """Loading from valid YAML file should succeed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_yaml_config)
            config_path = f.name

        try:
            config = RunPodConfig.from_yaml(config_path)

            # Verify RunPod settings
            assert config.runpod_api_key == "test-api-key-12345"
            assert config.network_volume_id == "vol-abc123"
            assert config.default_gpu_type == "NVIDIA RTX A5000"
            assert config.cloud_type == "COMMUNITY"

            # Verify storage settings
            assert config.storage_access_key == "AKIAIOSFODNN7EXAMPLE"
            assert config.storage_secret_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            assert config.data_prefix == "datasets"
            assert config.models_prefix == "models"

            # Verify cost limits
            assert config.max_training_cost_dollars == 10.0
            assert config.cost_warning_threshold == 0.8

            # Verify optional settings
            assert config.docker_registry == "myregistry.io/ml"
            assert config.serverless_endpoint_id == "endpoint-123"
        finally:
            os.unlink(config_path)

    def test_load_from_nested_yaml_file(self, valid_nested_yaml_config):
        """Loading from valid nested YAML file should succeed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_nested_yaml_config)
            config_path = f.name

        try:
            config = RunPodConfig.from_yaml(config_path)

            # Verify RunPod settings
            assert config.runpod_api_key == "test-api-key-12345"
            assert config.network_volume_id == "vol-abc123"
            assert config.default_gpu_type == "NVIDIA RTX A5000"
        finally:
            os.unlink(config_path)

    def test_load_from_nonexistent_file_raises_error(self):
        """Loading from nonexistent file should raise CloudConfigurationError."""
        with pytest.raises(CloudConfigurationError) as exc_info:
            RunPodConfig.from_yaml("/nonexistent/path/config.yaml")

        assert "not found" in str(exc_info.value).lower()

    def test_load_from_invalid_yaml_raises_error(self):
        """Loading from invalid YAML should raise CloudConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(CloudConfigurationError) as exc_info:
                RunPodConfig.from_yaml(config_path)

            assert "yaml" in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)

    def test_load_from_missing_runpod_section_raises_error(self):
        """Loading from YAML without runpod section should raise error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("other_config:\n  key: value\n")
            config_path = f.name

        try:
            with pytest.raises(CloudConfigurationError) as exc_info:
                RunPodConfig.from_yaml(config_path)

            assert "runpod" in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)


class TestRunPodConfigFromEnv:
    """Test RunPodConfig.from_env() method."""

    def test_load_from_environment_variables(self, monkeypatch):
        """Loading from environment variables should succeed."""
        env_vars = {
            "RUNPOD_API_KEY": "test-api-key-12345",
            "RUNPOD_STORAGE_ENDPOINT": "https://storage.runpod.io",
            "RUNPOD_NETWORK_VOLUME_ID": "vol-abc123",
            "RUNPOD_DEFAULT_GPU_TYPE": "NVIDIA RTX A5000",
            "RUNPOD_CLOUD_TYPE": "COMMUNITY",
            "RUNPOD_STORAGE_ACCESS_KEY": "AKIAIOSFODNN7EXAMPLE",
            "RUNPOD_STORAGE_SECRET_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "RUNPOD_DATA_PREFIX": "datasets",
            "RUNPOD_MODELS_PREFIX": "models",
            "RUNPOD_RESULTS_PREFIX": "results",
            "RUNPOD_MAX_TRAINING_COST": "10.0",
            "RUNPOD_MAX_PREDICTION_COST": "1.0",
            "RUNPOD_COST_WARNING_THRESHOLD": "0.8",
            "RUNPOD_DOCKER_REGISTRY": "myregistry.io/ml",
            "RUNPOD_SERVERLESS_ENDPOINT_ID": "endpoint-123"
        }

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = RunPodConfig.from_env()

        # Verify loaded values
        assert config.runpod_api_key == "test-api-key-12345"
        assert config.network_volume_id == "vol-abc123"
        assert config.default_gpu_type == "NVIDIA RTX A5000"
        assert config.cloud_type == "COMMUNITY"
        assert config.storage_access_key == "AKIAIOSFODNN7EXAMPLE"
        assert config.data_prefix == "datasets"
        assert config.max_training_cost_dollars == 10.0
        assert config.docker_registry == "myregistry.io/ml"

    def test_load_from_env_with_defaults(self, monkeypatch):
        """Loading from environment with defaults should succeed."""
        # Set only required fields
        monkeypatch.setenv("RUNPOD_API_KEY", "test-api-key")
        monkeypatch.setenv("RUNPOD_NETWORK_VOLUME_ID", "vol-123")
        monkeypatch.setenv("RUNPOD_STORAGE_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("RUNPOD_STORAGE_SECRET_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        config = RunPodConfig.from_env()

        # Verify defaults
        assert config.default_gpu_type == "NVIDIA RTX A5000"
        assert config.cloud_type == "COMMUNITY"
        assert config.data_prefix == "datasets"
        assert config.max_training_cost_dollars == 10.0
        assert config.docker_registry is None

    def test_load_from_env_missing_required_raises_error(self):
        """Loading from environment with missing required fields should raise error."""
        # Don't set any environment variables
        with pytest.raises(CloudConfigurationError):
            RunPodConfig.from_env()
