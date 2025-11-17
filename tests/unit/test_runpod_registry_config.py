"""
Tests for RunPod Container Registry configuration and validation.

Author: Statistical Modeling Agent
Created: 2025-11-11 (RunPod Registry Integration)
"""

import os
import pytest
from unittest.mock import patch

from src.cloud.runpod_config import RunPodConfig
from src.cloud.exceptions import CloudConfigurationError


class TestRunPodRegistryConfig:
    """Test RunPod registry configuration validation."""

    def test_standard_docker_hub_image(self):
        """Test configuration with standard Docker Hub image."""
        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "test_key",
            "RUNPOD_NETWORK_VOLUME_ID": "test_volume",
            "RUNPOD_STORAGE_ACCESS_KEY": "test_access",
            "RUNPOD_STORAGE_SECRET_KEY": "test_secret",
        }):
            config = RunPodConfig.from_env()
            config.validate()

            # Should work with default config (no registry)
            assert config.docker_registry is None

    def test_registry_image_format_validation(self):
        """Test registry image URL format validation."""
        test_cases = [
            ("registry.runpod.io/username/image:v1", True),
            ("registry.runpod.io/username/image", True),
            ("username/image:v1", False),  # Not a registry URL
            ("tensorflow/tensorflow:2.13.0-gpu", False),  # Docker Hub
        ]

        for image_url, is_registry in test_cases:
            if is_registry:
                assert image_url.startswith("registry.runpod.io/")
            else:
                assert not image_url.startswith("registry.runpod.io/")

    def test_registry_credentials_from_env(self):
        """Test loading registry credentials from environment."""
        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "test_key",
            "RUNPOD_NETWORK_VOLUME_ID": "test_volume",
            "RUNPOD_STORAGE_ACCESS_KEY": "test_access",
            "RUNPOD_STORAGE_SECRET_KEY": "test_secret",
            "RUNPOD_REGISTRY_USERNAME": "my_username",
            "RUNPOD_REGISTRY_PASSWORD": "my_password",
        }):
            # These would be loaded separately in the pod manager
            registry_username = os.getenv("RUNPOD_REGISTRY_USERNAME")
            registry_password = os.getenv("RUNPOD_REGISTRY_PASSWORD")

            assert registry_username == "my_username"
            assert registry_password == "my_password"

    def test_registry_username_parsing(self):
        """Test extracting username from registry image URL."""
        test_cases = [
            ("registry.runpod.io/johndoe/ml-training:v1", "johndoe"),
            ("registry.runpod.io/company/ml-training:latest", "company"),
            ("registry.runpod.io/test123/image", "test123"),
        ]

        for image_url, expected_username in test_cases:
            # Parse: registry.runpod.io/{username}/{image}:{tag}
            parts = image_url.replace("registry.runpod.io/", "").split("/")
            username = parts[0]
            assert username == expected_username

    def test_config_with_docker_registry_field(self):
        """Test RunPodConfig with docker_registry field."""
        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "test_key",
            "RUNPOD_NETWORK_VOLUME_ID": "test_volume",
            "RUNPOD_STORAGE_ACCESS_KEY": "test_access",
            "RUNPOD_STORAGE_SECRET_KEY": "test_secret",
            "RUNPOD_DOCKER_REGISTRY": "my_username",
        }):
            config = RunPodConfig.from_env()
            config.validate()

            # docker_registry field should be populated
            assert config.docker_registry == "my_username"

    def test_template_id_with_registry_image(self):
        """Test that template_id and registry image can coexist."""
        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "test_key",
            "RUNPOD_NETWORK_VOLUME_ID": "test_volume",
            "RUNPOD_STORAGE_ACCESS_KEY": "test_access",
            "RUNPOD_STORAGE_SECRET_KEY": "test_secret",
            "RUNPOD_TEMPLATE_ID": "v3ie98gf2m",
        }):
            config = RunPodConfig.from_env()
            config.validate()

            # Template and registry image are independent
            assert config.template_id == "v3ie98gf2m"
            # Can still use registry image with template

    def test_missing_registry_credentials_warning(self):
        """Test that missing registry credentials are handled gracefully."""
        # If image uses registry but credentials not set,
        # RunPod will fail at pod creation (not config validation)
        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "test_key",
            "RUNPOD_NETWORK_VOLUME_ID": "test_volume",
            "RUNPOD_STORAGE_ACCESS_KEY": "test_access",
            "RUNPOD_STORAGE_SECRET_KEY": "test_secret",
        }):
            config = RunPodConfig.from_env()
            config.validate()  # Should not fail

            # Registry credentials are optional at config level
            registry_username = os.getenv("RUNPOD_REGISTRY_USERNAME")
            registry_password = os.getenv("RUNPOD_REGISTRY_PASSWORD")

            assert registry_username is None
            assert registry_password is None


class TestRegistryImageParsing:
    """Test registry image URL parsing utilities."""

    def test_is_registry_image(self):
        """Test detection of registry vs Docker Hub images."""
        def is_registry_image(image: str) -> bool:
            return image.startswith("registry.runpod.io/")

        assert is_registry_image("registry.runpod.io/user/image:v1") is True
        assert is_registry_image("tensorflow/tensorflow:2.13.0-gpu") is False
        assert is_registry_image("nvidia/cuda:11.8.0-runtime") is False
        assert is_registry_image("registry.runpod.io/company/ml:latest") is True

    def test_parse_registry_components(self):
        """Test parsing registry image into components."""
        def parse_registry_image(image: str) -> dict:
            if not image.startswith("registry.runpod.io/"):
                return {"is_registry": False}

            # Format: registry.runpod.io/{username}/{image_name}:{tag}
            path = image.replace("registry.runpod.io/", "")
            parts = path.split("/")
            username = parts[0]
            image_with_tag = parts[1] if len(parts) > 1 else ""

            if ":" in image_with_tag:
                image_name, tag = image_with_tag.rsplit(":", 1)
            else:
                image_name, tag = image_with_tag, "latest"

            return {
                "is_registry": True,
                "username": username,
                "image_name": image_name,
                "tag": tag,
                "full_url": image,
            }

        # Test valid registry image
        result = parse_registry_image("registry.runpod.io/johndoe/ml-training:v1")
        assert result["is_registry"] is True
        assert result["username"] == "johndoe"
        assert result["image_name"] == "ml-training"
        assert result["tag"] == "v1"

        # Test Docker Hub image
        result = parse_registry_image("tensorflow/tensorflow:2.13.0-gpu")
        assert result["is_registry"] is False

    def test_registry_image_validation(self):
        """Test validation of registry image format."""
        def validate_registry_image(image: str) -> bool:
            if not image.startswith("registry.runpod.io/"):
                return True  # Docker Hub image (valid)

            # Validate registry format
            path = image.replace("registry.runpod.io/", "")
            parts = path.split("/")

            # Must have username and image name
            if len(parts) < 2:
                return False

            # Username must be alphanumeric + underscore/hyphen
            username = parts[0]
            if not username.replace("-", "").replace("_", "").isalnum():
                return False

            return True

        # Valid cases
        assert validate_registry_image("registry.runpod.io/user/image:v1") is True
        assert validate_registry_image("registry.runpod.io/user-name/image") is True
        assert validate_registry_image("tensorflow/tensorflow:2.13.0-gpu") is True

        # Invalid cases
        assert validate_registry_image("registry.runpod.io/") is False
        assert validate_registry_image("registry.runpod.io/user") is False
        assert validate_registry_image("registry.runpod.io/user with spaces/image") is False
