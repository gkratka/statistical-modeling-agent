"""
RunPod Pod Manager for ML Training.

This module manages RunPod GPU pods for batch ML training,
replacing AWS EC2 Spot Instances with RunPod's GPU-powered infrastructure.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 6.1: RunPod Pod Manager)
"""

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    runpod = None  # type: ignore
    RUNPOD_AVAILABLE = False

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from src.cloud.provider_interface import CloudTrainingProvider
from src.cloud.runpod_config import RunPodConfig


class RunPodPodManager(CloudTrainingProvider):
    """
    Manage RunPod GPU pods for ML training.

    Provides methods for launching training pods, monitoring progress,
    streaming logs, and automatic termination.
    """

    def __init__(self, config: RunPodConfig):
        """
        Initialize RunPodPodManager.

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

    def select_compute_type(
        self,
        dataset_size_mb: float,
        model_type: str,
        estimated_training_time_minutes: int = 0
    ) -> str:
        """
        Select optimal GPU type based on dataset size and model.

        Decision matrix:
        - <1GB: RTX A5000 (24GB VRAM, $0.29/hr)
        - 1-5GB: RTX A40 (48GB VRAM, $0.39/hr)
        - >5GB or neural networks: A100 40GB ($0.79/hr)

        Args:
            dataset_size_mb: Dataset size in megabytes
            model_type: Type of ML model (e.g., 'linear', 'random_forest', 'mlp_regression')
            estimated_training_time_minutes: Optional estimated training time

        Returns:
            GPU type string (e.g., 'NVIDIA A100 PCIe 40GB')

        Example:
            >>> manager = RunPodPodManager(config)
            >>> gpu = manager.select_compute_type(2048, 'random_forest')
            >>> print(gpu)
            'NVIDIA RTX A40'
        """
        dataset_size_gb = dataset_size_mb / 1024

        if model_type in ['mlp_regression', 'mlp_classification', 'neural_network']:
            # Neural networks benefit from A100
            return 'NVIDIA A100 PCIe 40GB'
        elif dataset_size_gb > 5:
            return 'NVIDIA A100 PCIe 40GB'
        elif dataset_size_gb > 1:
            return 'NVIDIA RTX A40'
        else:
            return 'NVIDIA RTX A5000'

    def launch_training(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Launch RunPod pod for training.

        Args:
            config: Training configuration dictionary with keys:
                - gpu_type: str (GPU type ID)
                - dataset_key: str (storage key for dataset)
                - model_id: str (unique model identifier)
                - user_id: int (user identifier)
                - model_type: str (e.g., 'linear', 'random_forest')
                - target_column: str (target variable name)
                - feature_columns: List[str] (feature variable names)
                - hyperparameters: Dict[str, Any] (model hyperparameters)
                - docker_image: str (optional, Docker image to use)
                - volume_size_gb: int (optional, ephemeral volume size)

        Returns:
            Pod launch details dictionary with keys:
                - pod_id: str (RunPod pod identifier)
                - gpu_type: str (GPU type used)
                - launch_time: float (timestamp)
                - status: str ('launching')

        Raises:
            ValueError: If required config keys are missing
            RuntimeError: If pod creation fails

        Example:
            >>> result = manager.launch_training({
            ...     'gpu_type': 'NVIDIA RTX A5000',
            ...     'dataset_key': 'datasets/user_123/data.csv',
            ...     'model_id': 'model_456',
            ...     'user_id': 123,
            ...     'model_type': 'random_forest',
            ...     'target_column': 'price',
            ...     'feature_columns': ['sqft', 'bedrooms'],
            ...     'hyperparameters': {'n_estimators': 100}
            ... })
            >>> print(result['pod_id'])
            'pod-xyz123'
        """
        # Validate required keys
        required_keys = [
            'gpu_type', 'dataset_key', 'model_id', 'user_id',
            'model_type', 'target_column', 'feature_columns'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        gpu_type = config['gpu_type']

        # Environment variables for training script
        env_vars = {
            'STORAGE_ACCESS_KEY': self.config.storage_access_key,
            'STORAGE_SECRET_KEY': self.config.storage_secret_key,
            'STORAGE_ENDPOINT': self.config.storage_endpoint,
            'VOLUME_ID': self.config.network_volume_id,
            'DATASET_KEY': config['dataset_key'],
            'MODEL_ID': config['model_id'],
            'MODEL_TYPE': config['model_type'],
            'TARGET_COLUMN': config['target_column'],
            'FEATURE_COLUMNS': ','.join(config['feature_columns']),
            'HYPERPARAMETERS': json.dumps(config.get('hyperparameters', {}))
        }

        # Default Docker image (PyTorch with CUDA)
        default_image = 'runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel'
        docker_image = config.get('docker_image', default_image)

        # Launch pod with training configuration
        try:
            pod = runpod.create_pod(
                name=f"training_{config['user_id']}_{config['model_id']}",
                image_name=docker_image,
                gpu_type_id=gpu_type,
                cloud_type=self.config.cloud_type,
                volume_in_gb=config.get('volume_size_gb', 50),
                volume_mount_path='/workspace',
                env=env_vars,
                # Training script runs on pod start
                docker_args="python /workspace/train.py"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create RunPod pod: {e}")

        return {
            'pod_id': pod['id'],
            'gpu_type': gpu_type,
            'launch_time': time.time(),
            'status': 'launching'
        }

    def monitor_training(self, pod_id: str) -> Dict[str, Any]:
        """
        Monitor training pod status.

        Args:
            pod_id: RunPod pod identifier

        Returns:
            Pod status dictionary with keys:
                - pod_id: str
                - status: str ('RUNNING', 'EXITED', 'FAILED', etc.)
                - runtime_seconds: Optional[int]
                - gpu_utilization: int (0-100)
                - machine_id: Optional[str]

        Example:
            >>> status = manager.monitor_training('pod-xyz123')
            >>> print(status['status'])
            'RUNNING'
        """
        try:
            pod = runpod.get_pod(pod_id)
        except Exception as e:
            return {
                'pod_id': pod_id,
                'status': 'UNKNOWN',
                'error': str(e)
            }

        return {
            'pod_id': pod_id,
            'status': pod.get('desiredStatus', 'UNKNOWN'),
            'runtime_seconds': pod.get('runtime'),
            'gpu_utilization': pod.get('gpuUtilization', 0),
            'machine_id': pod.get('machineId')
        }

    def get_pod_logs(self, pod_id: str, lines: int = 100) -> List[str]:
        """
        Retrieve pod logs for progress monitoring.

        Args:
            pod_id: RunPod pod identifier
            lines: Number of log lines to retrieve (tail)

        Returns:
            List of log line strings

        Example:
            >>> logs = manager.get_pod_logs('pod-xyz123', lines=50)
            >>> for line in logs:
            ...     print(line)
        """
        try:
            # RunPod SDK method for logs
            logs = runpod.get_pod_logs(pod_id, tail=lines)
            return logs.split('\n') if logs else []
        except Exception as e:
            return [f"Error retrieving logs: {e}"]

    def terminate_pod(self, pod_id: str) -> str:
        """
        Terminate training pod.

        Args:
            pod_id: RunPod pod identifier

        Returns:
            Pod ID that was terminated

        Example:
            >>> terminated_id = manager.terminate_pod('pod-xyz123')
            >>> print(f"Terminated {terminated_id}")
        """
        try:
            runpod.terminate_pod(pod_id)
        except Exception as e:
            raise RuntimeError(f"Failed to terminate pod {pod_id}: {e}")

        return pod_id

    def terminate_training(self, job_id: str) -> str:
        """
        Terminate training job (implements CloudTrainingProvider interface).

        This is an alias for terminate_pod to comply with the CloudTrainingProvider interface.

        Args:
            job_id: Training job identifier (pod ID)

        Returns:
            Job ID of terminated job

        Example:
            >>> terminated_id = manager.terminate_training('pod-xyz123')
            >>> print(f"Terminated {terminated_id}")
        """
        return self.terminate_pod(job_id)

    def estimate_training_time(
        self,
        dataset_size_mb: float,
        model_type: str
    ) -> int:
        """
        Estimate training time in seconds.

        Rough heuristics:
        - Linear models: ~0.1 sec per MB
        - Tree models: ~0.5 sec per MB
        - Neural networks: ~2 sec per MB

        Args:
            dataset_size_mb: Dataset size in megabytes
            model_type: Type of ML model

        Returns:
            Estimated training time in seconds

        Example:
            >>> time_sec = manager.estimate_training_time(100, 'random_forest')
            >>> print(f"Estimated: {time_sec} seconds")
            50
        """
        if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            return int(dataset_size_mb * 0.1)
        elif model_type in ['decision_tree', 'random_forest', 'gradient_boosting',
                           'xgboost', 'lightgbm', 'catboost']:
            return int(dataset_size_mb * 0.5)
        else:  # Neural networks
            return int(dataset_size_mb * 2.0)

    async def poll_training_logs(
        self,
        pod_id: str,
        poll_interval_seconds: int = 5
    ) -> AsyncIterator[str]:
        """
        Poll pod logs for training progress (async generator).

        Yields log lines as they become available.

        Args:
            pod_id: RunPod pod identifier
            poll_interval_seconds: Seconds between polling

        Yields:
            Log line strings

        Example:
            >>> async for log_line in manager.poll_training_logs('pod-xyz123'):
            ...     print(log_line)
        """
        last_line_count = 0

        while True:
            # Get pod status
            try:
                pod = runpod.get_pod(pod_id)
                status = pod.get('desiredStatus', 'UNKNOWN')
            except Exception:
                break

            # Get logs
            try:
                logs = runpod.get_pod_logs(pod_id)
                if logs:
                    lines = logs.split('\n')

                    # Yield new lines only
                    for line in lines[last_line_count:]:
                        if line.strip():
                            yield line

                    last_line_count = len(lines)
            except Exception as e:
                yield f"Error retrieving logs: {e}"
                break

            # Stop if pod completed or failed
            if status in ['EXITED', 'FAILED', 'TERMINATED']:
                break

            await asyncio.sleep(poll_interval_seconds)

    def wait_for_completion_and_terminate(
        self,
        pod_id: str,
        max_wait_seconds: int = 7200  # 2 hours
    ) -> Dict[str, Any]:
        """
        Wait for pod to complete training, then terminate.

        Args:
            pod_id: RunPod pod identifier
            max_wait_seconds: Maximum wait time (default 2 hours)

        Returns:
            Final pod status dictionary with keys:
                - success: bool
                - runtime_seconds: float
                - pod_id: str
                - error: Optional[str]

        Raises:
            None (returns error in dict instead)

        Example:
            >>> result = manager.wait_for_completion_and_terminate('pod-xyz123')
            >>> if result['success']:
            ...     print(f"Training completed in {result['runtime_seconds']}s")
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > max_wait_seconds:
                # Timeout - force terminate
                try:
                    self.terminate_pod(pod_id)
                except Exception:
                    pass

                return {
                    'success': False,
                    'error': 'Training timeout exceeded',
                    'runtime_seconds': elapsed,
                    'pod_id': pod_id
                }

            # Check pod status
            pod_status = self.monitor_training(pod_id)

            if pod_status['status'] == 'EXITED':
                # Training complete - terminate pod
                try:
                    self.terminate_pod(pod_id)
                except Exception:
                    pass

                return {
                    'success': True,
                    'runtime_seconds': elapsed,
                    'pod_id': pod_id
                }
            elif pod_status['status'] == 'FAILED':
                # Training failed - terminate pod
                try:
                    self.terminate_pod(pod_id)
                except Exception:
                    pass

                return {
                    'success': False,
                    'error': 'Pod failed during training',
                    'runtime_seconds': elapsed,
                    'pod_id': pod_id
                }

            time.sleep(10)  # Poll every 10 seconds
