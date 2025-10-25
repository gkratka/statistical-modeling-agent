"""
Abstract interfaces for cloud provider operations.

This module defines the contract that all cloud providers (AWS, RunPod, etc.)
must implement to support storage, training, and prediction operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncIterator
from pathlib import Path


class CloudStorageProvider(ABC):
    """Abstract interface for cloud storage operations."""

    @abstractmethod
    def upload_dataset(
        self,
        user_id: int,
        file_path: str,
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Upload dataset to cloud storage.

        Args:
            user_id: User ID for isolation
            file_path: Local file path to upload
            dataset_name: Optional dataset name

        Returns:
            Storage URI (e.g., s3://bucket/key or runpod://volume/key)
        """
        pass

    @abstractmethod
    def save_model(
        self,
        user_id: int,
        model_id: str,
        model_dir: Path
    ) -> str:
        """
        Save model directory to cloud storage.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            model_dir: Local directory containing model files

        Returns:
            Storage URI for model
        """
        pass

    @abstractmethod
    def load_model(
        self,
        user_id: int,
        model_id: str,
        local_dir: Path
    ) -> Path:
        """
        Load model from cloud storage to local directory.

        Args:
            user_id: User ID for isolation
            model_id: Model identifier
            local_dir: Local directory to download to

        Returns:
            Path to downloaded model directory
        """
        pass

    @abstractmethod
    def list_user_datasets(self, user_id: int) -> list[Dict[str, Any]]:
        """
        List all datasets for user.

        Args:
            user_id: User ID

        Returns:
            List of dataset metadata dicts
        """
        pass

    @abstractmethod
    def list_user_models(self, user_id: int) -> list[Dict[str, Any]]:
        """
        List all models for user.

        Args:
            user_id: User ID

        Returns:
            List of model metadata dicts
        """
        pass


class CloudTrainingProvider(ABC):
    """Abstract interface for cloud training operations."""

    @abstractmethod
    def select_compute_type(
        self,
        dataset_size_mb: float,
        model_type: str,
        estimated_training_time_minutes: int = 0
    ) -> str:
        """
        Select optimal compute resource for training.

        Args:
            dataset_size_mb: Dataset size in megabytes
            model_type: Type of ML model (linear, random_forest, etc.)
            estimated_training_time_minutes: Estimated training duration

        Returns:
            Compute resource identifier (e.g., 't3.medium', 'NVIDIA RTX A5000')
        """
        pass

    @abstractmethod
    def launch_training(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Launch training job on cloud compute.

        Args:
            config: Training configuration dict containing:
                - compute_type: Resource identifier
                - dataset_uri: Storage URI for dataset
                - model_id: Model identifier
                - user_id: User ID
                - model_type: ML model type
                - target_column: Target variable name
                - feature_columns: List of feature names
                - hyperparameters: Model hyperparameters

        Returns:
            Job details dict with job_id, status, launch_time
        """
        pass

    @abstractmethod
    def monitor_training(self, job_id: str) -> Dict[str, Any]:
        """
        Monitor training job status.

        Args:
            job_id: Training job identifier

        Returns:
            Status dict with job_id, status, runtime, progress
        """
        pass

    @abstractmethod
    def terminate_training(self, job_id: str) -> str:
        """
        Terminate training job.

        Args:
            job_id: Training job identifier

        Returns:
            Job ID of terminated job
        """
        pass


class CloudPredictionProvider(ABC):
    """Abstract interface for cloud prediction operations."""

    @abstractmethod
    def invoke_prediction(
        self,
        model_uri: str,
        data_uri: str,
        output_uri: str,
        prediction_column_name: str = 'prediction',
        feature_columns: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Invoke prediction service.

        Args:
            model_uri: Storage URI for trained model
            data_uri: Storage URI for input data
            output_uri: Storage URI for output results
            prediction_column_name: Name for prediction column
            feature_columns: List of feature column names to use

        Returns:
            Prediction results dict with success, output_uri, num_predictions
        """
        pass

    @abstractmethod
    def invoke_async(
        self,
        model_uri: str,
        data_uri: str,
        output_uri: str,
        prediction_column_name: str = 'prediction',
        feature_columns: Optional[list] = None
    ) -> str:
        """
        Invoke prediction service asynchronously.

        Args:
            model_uri: Storage URI for trained model
            data_uri: Storage URI for input data
            output_uri: Storage URI for output results
            prediction_column_name: Name for prediction column
            feature_columns: List of feature column names to use

        Returns:
            Job ID for status checking
        """
        pass

    @abstractmethod
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of async prediction job.

        Args:
            job_id: Prediction job identifier

        Returns:
            Status dict with job_id, status, progress, result
        """
        pass
