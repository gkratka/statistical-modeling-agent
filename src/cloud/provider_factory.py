"""
Cloud Provider Factory for creating cloud provider instances.

This module implements the factory pattern for instantiating cloud providers
(AWS, RunPod, etc.) based on provider name, abstracting provider-specific
implementation details.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 1.2: Provider Factory Implementation)
"""

from typing import Literal, Optional, Union

from src.cloud.provider_interface import (
    CloudStorageProvider,
    CloudTrainingProvider,
    CloudPredictionProvider
)
from src.cloud.s3_manager import S3Manager
from src.cloud.ec2_manager import EC2Manager
from src.cloud.lambda_manager import LambdaManager
from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.runpod_config import RunPodConfig
from src.cloud.runpod_storage_manager import RunPodStorageManager


class CloudProviderFactory:
    """
    Factory for creating cloud provider instances.

    This factory implements the factory pattern for creating cloud provider
    instances (storage, training, prediction) based on provider name.
    Currently supports AWS, with RunPod support planned for future implementation.
    """

    @staticmethod
    def create_storage_provider(
        provider: Literal["aws", "runpod"],
        config: Union[CloudConfig, RunPodConfig],
        aws_client: Optional[AWSClient] = None
    ) -> CloudStorageProvider:
        """
        Create cloud storage provider instance.

        Factory method for creating storage providers that implement the
        CloudStorageProvider interface. Supports AWS S3Manager and
        RunPod storage manager.

        Args:
            provider: Provider name ("aws" or "runpod")
            config: CloudConfig for AWS or RunPodConfig for RunPod
            aws_client: AWSClient instance (required for AWS, unused for RunPod)

        Returns:
            CloudStorageProvider: Storage provider instance

        Raises:
            ValueError: If provider is not supported or required arguments missing

        Example:
            >>> # AWS
            >>> aws_client = AWSClient(aws_config)
            >>> storage = CloudProviderFactory.create_storage_provider(
            ...     provider="aws",
            ...     config=aws_config,
            ...     aws_client=aws_client
            ... )
            >>> isinstance(storage, S3Manager)
            True

            >>> # RunPod
            >>> storage = CloudProviderFactory.create_storage_provider(
            ...     provider="runpod",
            ...     config=runpod_config
            ... )
            >>> isinstance(storage, RunPodStorageManager)
            True
        """
        if provider == "aws":
            if aws_client is None:
                raise ValueError("aws_client is required for AWS storage provider")
            return S3Manager(aws_client=aws_client, config=config)
        elif provider == "runpod":
            if not isinstance(config, RunPodConfig):
                raise ValueError("RunPodConfig is required for RunPod storage provider")
            return RunPodStorageManager(config=config)
        else:
            raise ValueError(
                f"Unsupported storage provider: {provider}. "
                f"Supported providers: aws, runpod"
            )

    @staticmethod
    def create_training_provider(
        provider: Literal["aws", "runpod"],
        aws_client: AWSClient,
        config: CloudConfig
    ) -> CloudTrainingProvider:
        """
        Create cloud training provider instance.

        Factory method for creating training providers that implement the
        CloudTrainingProvider interface. Currently supports AWS EC2Manager,
        with RunPod compute support planned.

        Args:
            provider: Provider name ("aws" or "runpod")
            aws_client: AWSClient instance for AWS service access
            config: CloudConfig with provider configuration

        Returns:
            CloudTrainingProvider: Training provider instance (EC2Manager for AWS)

        Raises:
            ValueError: If provider is not supported
            NotImplementedError: If RunPod provider requested (not yet implemented)

        Example:
            >>> aws_client = AWSClient(config)
            >>> training = CloudProviderFactory.create_training_provider(
            ...     provider="aws",
            ...     aws_client=aws_client,
            ...     config=config
            ... )
            >>> isinstance(training, EC2Manager)
            True
        """
        if provider == "aws":
            return EC2Manager(aws_client=aws_client, config=config)
        elif provider == "runpod":
            raise NotImplementedError(
                "RunPod training provider not implemented yet. "
                "AWS EC2Manager is currently the only supported training provider."
            )
        else:
            raise ValueError(
                f"Unsupported training provider: {provider}. "
                f"Supported providers: aws, runpod"
            )

    @staticmethod
    def create_prediction_provider(
        provider: Literal["aws", "runpod"],
        aws_client: AWSClient,
        config: CloudConfig
    ) -> CloudPredictionProvider:
        """
        Create cloud prediction provider instance.

        Factory method for creating prediction providers that implement the
        CloudPredictionProvider interface. Currently supports AWS LambdaManager,
        with RunPod serverless support planned.

        Args:
            provider: Provider name ("aws" or "runpod")
            aws_client: AWSClient instance for AWS service access
            config: CloudConfig with provider configuration

        Returns:
            CloudPredictionProvider: Prediction provider instance (LambdaManager for AWS)

        Raises:
            ValueError: If provider is not supported
            NotImplementedError: If RunPod provider requested (not yet implemented)

        Example:
            >>> aws_client = AWSClient(config)
            >>> prediction = CloudProviderFactory.create_prediction_provider(
            ...     provider="aws",
            ...     aws_client=aws_client,
            ...     config=config
            ... )
            >>> isinstance(prediction, LambdaManager)
            True
        """
        if provider == "aws":
            return LambdaManager(aws_client=aws_client, config=config)
        elif provider == "runpod":
            raise NotImplementedError(
                "RunPod prediction provider not implemented yet. "
                "AWS LambdaManager is currently the only supported prediction provider."
            )
        else:
            raise ValueError(
                f"Unsupported prediction provider: {provider}. "
                f"Supported providers: aws, runpod"
            )
