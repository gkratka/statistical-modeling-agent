"""
RunPod Cloud Configuration Management.

This module provides configuration loading, validation, and management
for RunPod cloud infrastructure including GPU pods, network volumes,
and serverless endpoints.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.1: RunPod Configuration)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from src.cloud.exceptions import CloudConfigurationError


@dataclass
class RunPodConfig:
    """Configuration for RunPod cloud infrastructure."""

    # RunPod API Configuration
    runpod_api_key: str = ""
    storage_endpoint: str = "https://storage.runpod.io"
    network_volume_id: str = ""

    # GPU Configuration
    default_gpu_type: str = "NVIDIA RTX A5000"
    cloud_type: str = "COMMUNITY"  # 'COMMUNITY' or 'SECURE'

    # Storage Configuration
    storage_access_key: str = ""
    storage_secret_key: str = ""
    data_prefix: str = "datasets"
    models_prefix: str = "models"
    results_prefix: str = "results"

    # Cost Limits
    max_training_cost_dollars: float = 10.0
    max_prediction_cost_dollars: float = 1.0
    cost_warning_threshold: float = 0.8

    # Optional Configuration
    docker_registry: Optional[str] = None
    serverless_endpoint_id: Optional[str] = None

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            CloudConfigurationError: If any configuration value is invalid
        """
        # Validate RunPod API key
        if not self.runpod_api_key or not self.runpod_api_key.strip():
            raise CloudConfigurationError(
                "RunPod API key is required",
                config_key="runpod_api_key"
            )

        # Validate network volume ID
        if not self.network_volume_id or not self.network_volume_id.strip():
            raise CloudConfigurationError(
                "RunPod network volume ID is required",
                config_key="network_volume_id"
            )

        # Validate storage credentials
        if not self.storage_access_key or not self.storage_access_key.strip():
            raise CloudConfigurationError(
                "RunPod storage access key is required",
                config_key="storage_access_key"
            )

        if not self.storage_secret_key or not self.storage_secret_key.strip():
            raise CloudConfigurationError(
                "RunPod storage secret key is required",
                config_key="storage_secret_key"
            )

        # Validate cloud type
        if self.cloud_type not in ('COMMUNITY', 'SECURE'):
            raise CloudConfigurationError(
                f"RunPod cloud type must be 'COMMUNITY' or 'SECURE', got: {self.cloud_type}",
                config_key="cloud_type",
                config_value=self.cloud_type
            )

        # Validate storage endpoint
        if not self.storage_endpoint.startswith('https://'):
            raise CloudConfigurationError(
                "RunPod storage endpoint must start with 'https://'",
                config_key="storage_endpoint",
                config_value=self.storage_endpoint
            )

        # Validate prefixes
        if not self.data_prefix:
            raise CloudConfigurationError(
                "RunPod data prefix is required",
                config_key="data_prefix"
            )

        if not self.models_prefix:
            raise CloudConfigurationError(
                "RunPod models prefix is required",
                config_key="models_prefix"
            )

        if not self.results_prefix:
            raise CloudConfigurationError(
                "RunPod results prefix is required",
                config_key="results_prefix"
            )

        # Validate cost limits
        if self.max_training_cost_dollars < 0:
            raise CloudConfigurationError(
                "Max training cost must be non-negative",
                config_key="max_training_cost_dollars",
                config_value=str(self.max_training_cost_dollars)
            )

        if self.max_prediction_cost_dollars < 0:
            raise CloudConfigurationError(
                "Max prediction cost must be non-negative",
                config_key="max_prediction_cost_dollars",
                config_value=str(self.max_prediction_cost_dollars)
            )

        if not (0.0 <= self.cost_warning_threshold <= 1.0):
            raise CloudConfigurationError(
                "Cost warning threshold must be between 0.0 and 1.0",
                config_key="cost_warning_threshold",
                config_value=str(self.cost_warning_threshold)
            )

    @classmethod
    def from_yaml(cls, config_path: str) -> "RunPodConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            RunPodConfig instance

        Raises:
            CloudConfigurationError: If config file is invalid or missing
        """
        path = Path(config_path)

        # Check file exists
        if not path.exists():
            raise CloudConfigurationError(
                f"Configuration file not found: {config_path}",
                config_key="config_path"
            )

        # Load YAML
        try:
            with open(path) as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise CloudConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_key="config_path"
            )
        except Exception as e:
            raise CloudConfigurationError(
                f"Error reading configuration file: {e}",
                config_key="config_path"
            )

        # Validate structure
        if not isinstance(config_data, dict):
            raise CloudConfigurationError(
                "Configuration file must contain a dictionary",
                config_key="config_structure"
            )

        # Extract RunPod section (support both nested "cloud.runpod" and top-level "runpod")
        try:
            if "cloud" in config_data and "runpod" in config_data["cloud"]:
                runpod_config = config_data["cloud"]["runpod"]
            elif "runpod" in config_data:
                runpod_config = config_data["runpod"]
            else:
                raise CloudConfigurationError(
                    "No RunPod configuration section found",
                    config_key="runpod"
                )

            # Create RunPodConfig instance
            config = cls(
                # RunPod API
                runpod_api_key=runpod_config.get("api_key", ""),
                storage_endpoint=runpod_config.get("storage_endpoint", "https://storage.runpod.io"),
                network_volume_id=runpod_config.get("network_volume_id", ""),

                # GPU configuration
                default_gpu_type=runpod_config.get("default_gpu_type", "NVIDIA RTX A5000"),
                cloud_type=runpod_config.get("cloud_type", "COMMUNITY"),

                # Storage configuration
                storage_access_key=runpod_config.get("storage_access_key", ""),
                storage_secret_key=runpod_config.get("storage_secret_key", ""),
                data_prefix=runpod_config.get("data_prefix", "datasets"),
                models_prefix=runpod_config.get("models_prefix", "models"),
                results_prefix=runpod_config.get("results_prefix", "results"),

                # Cost limits
                max_training_cost_dollars=runpod_config.get("max_training_cost_dollars", 10.0),
                max_prediction_cost_dollars=runpod_config.get("max_prediction_cost_dollars", 1.0),
                cost_warning_threshold=runpod_config.get("cost_warning_threshold", 0.8),

                # Optional
                docker_registry=runpod_config.get("docker_registry"),
                serverless_endpoint_id=runpod_config.get("serverless_endpoint_id")
            )

            # Validate configuration
            config.validate()

            return config

        except KeyError as e:
            raise CloudConfigurationError(
                f"Missing required configuration key: {e}",
                config_key=str(e)
            )
        except (TypeError, ValueError) as e:
            raise CloudConfigurationError(
                f"Invalid configuration value: {e}",
                config_key="config_value"
            )

    @classmethod
    def from_env(cls) -> "RunPodConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            RUNPOD_API_KEY: RunPod API key
            RUNPOD_STORAGE_ENDPOINT: Storage endpoint URL
            RUNPOD_NETWORK_VOLUME_ID: Network volume ID
            RUNPOD_DEFAULT_GPU_TYPE: Default GPU type
            RUNPOD_CLOUD_TYPE: Cloud type (COMMUNITY or SECURE)
            RUNPOD_STORAGE_ACCESS_KEY: Storage access key
            RUNPOD_STORAGE_SECRET_KEY: Storage secret key
            RUNPOD_DATA_PREFIX: Data prefix
            RUNPOD_MODELS_PREFIX: Models prefix
            RUNPOD_RESULTS_PREFIX: Results prefix
            RUNPOD_MAX_TRAINING_COST: Max training cost in dollars
            RUNPOD_MAX_PREDICTION_COST: Max prediction cost in dollars
            RUNPOD_COST_WARNING_THRESHOLD: Cost warning threshold (0.0-1.0)
            RUNPOD_DOCKER_REGISTRY: Optional Docker registry
            RUNPOD_SERVERLESS_ENDPOINT_ID: Optional serverless endpoint ID

        Returns:
            RunPodConfig instance

        Raises:
            CloudConfigurationError: If required environment variables are missing
        """
        try:
            config = cls(
                # RunPod API
                runpod_api_key=os.getenv("RUNPOD_API_KEY", ""),
                storage_endpoint=os.getenv("RUNPOD_STORAGE_ENDPOINT", "https://storage.runpod.io"),
                network_volume_id=os.getenv("RUNPOD_NETWORK_VOLUME_ID", ""),

                # GPU configuration
                default_gpu_type=os.getenv("RUNPOD_DEFAULT_GPU_TYPE", "NVIDIA RTX A5000"),
                cloud_type=os.getenv("RUNPOD_CLOUD_TYPE", "COMMUNITY"),

                # Storage configuration
                storage_access_key=os.getenv("RUNPOD_STORAGE_ACCESS_KEY", ""),
                storage_secret_key=os.getenv("RUNPOD_STORAGE_SECRET_KEY", ""),
                data_prefix=os.getenv("RUNPOD_DATA_PREFIX", "datasets"),
                models_prefix=os.getenv("RUNPOD_MODELS_PREFIX", "models"),
                results_prefix=os.getenv("RUNPOD_RESULTS_PREFIX", "results"),

                # Cost limits
                max_training_cost_dollars=float(os.getenv("RUNPOD_MAX_TRAINING_COST", "10.0")),
                max_prediction_cost_dollars=float(os.getenv("RUNPOD_MAX_PREDICTION_COST", "1.0")),
                cost_warning_threshold=float(os.getenv("RUNPOD_COST_WARNING_THRESHOLD", "0.8")),

                # Optional
                docker_registry=os.getenv("RUNPOD_DOCKER_REGISTRY"),
                serverless_endpoint_id=os.getenv("RUNPOD_SERVERLESS_ENDPOINT_ID")
            )

            # Validate configuration
            config.validate()

            return config

        except ValueError as e:
            raise CloudConfigurationError(
                f"Invalid environment variable value: {e}",
                config_key="environment_variable"
            )
        except Exception as e:
            raise CloudConfigurationError(
                f"Error loading configuration from environment: {e}",
                config_key="environment"
            )
