"""
RunPod Serverless Manager for ML Predictions.

This module manages RunPod serverless endpoints for batch predictions,
replacing AWS Lambda with RunPod's GPU-powered serverless infrastructure.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.4: RunPod Serverless Manager)
"""

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    runpod = None  # type: ignore
    RUNPOD_AVAILABLE = False

from typing import Dict, Any, Optional, List
import time

from src.cloud.runpod_config import RunPodConfig
from src.cloud.provider_interface import CloudPredictionProvider


class RunPodServerlessManager(CloudPredictionProvider):
    """
    Manage RunPod serverless endpoints for ML predictions.

    Provides methods for invoking serverless predictions (sync/async),
    checking job status, and creating/managing endpoints.
    """

    def __init__(self, config: RunPodConfig):
        """
        Initialize RunPodServerlessManager.

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

    def invoke_prediction(
        self,
        model_uri: str,
        data_uri: str,
        output_uri: str,
        endpoint_id: Optional[str] = None,
        prediction_column_name: str = 'prediction',
        feature_columns: Optional[List[str]] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Invoke RunPod serverless endpoint for predictions (synchronous).

        Args:
            model_uri: Storage URI for model (e.g., 'models/user_123/model_456')
            data_uri: Storage URI for input data
            output_uri: Storage URI for output results
            endpoint_id: RunPod endpoint ID (uses config default if not specified)
            prediction_column_name: Name for prediction column
            feature_columns: List of feature column names
            timeout: Maximum wait time in seconds

        Returns:
            Prediction results dictionary

        Raises:
            ValueError: If endpoint_id not provided and not in config
            RuntimeError: If prediction fails or times out

        Example:
            >>> manager = RunPodServerlessManager(config)
            >>> result = manager.invoke_prediction(
            ...     model_uri="models/user_12345/model_abc",
            ...     data_uri="datasets/user_12345/test_data.csv",
            ...     output_uri="predictions/user_12345/predictions.csv"
            ... )
            >>> print(result['success'])
            True
        """
        # Use endpoint from config if not specified
        if endpoint_id is None:
            if not self.config.endpoint_id:
                raise ValueError(
                    "endpoint_id must be provided or configured in RunPodConfig.endpoint_id"
                )
            endpoint_id = self.config.endpoint_id

        # Create endpoint instance
        endpoint = runpod.Endpoint(endpoint_id)

        # Prepare input payload
        input_data = {
            "input": {
                "model_key": model_uri,
                "data_key": data_uri,
                "output_key": output_uri,
                "volume_id": self.config.network_volume_id,
                "prediction_column_name": prediction_column_name
            }
        }

        if feature_columns:
            input_data["input"]["feature_columns"] = feature_columns

        # Synchronous invocation
        result = endpoint.run_sync(input_data, timeout=timeout)

        return result

    def invoke_async(
        self,
        model_uri: str,
        data_uri: str,
        output_uri: str,
        endpoint_id: Optional[str] = None,
        prediction_column_name: str = 'prediction',
        feature_columns: Optional[List[str]] = None
    ) -> str:
        """
        Invoke RunPod serverless endpoint asynchronously.

        Returns immediately with a job ID that can be used to check status.

        Args:
            model_uri: Storage URI for model
            data_uri: Storage URI for input data
            output_uri: Storage URI for output results
            endpoint_id: RunPod endpoint ID (uses config default if not specified)
            prediction_column_name: Name for prediction column
            feature_columns: List of feature column names

        Returns:
            Job ID string for status checking

        Example:
            >>> manager = RunPodServerlessManager(config)
            >>> job_id = manager.invoke_async(
            ...     model_uri="models/user_12345/model_abc",
            ...     data_uri="datasets/user_12345/large_data.csv",
            ...     output_uri="predictions/user_12345/predictions.csv"
            ... )
            >>> # Check status later
            >>> status = manager.check_job_status(endpoint_id, job_id)
        """
        # Use endpoint from config if not specified
        if endpoint_id is None:
            if not self.config.endpoint_id:
                raise ValueError(
                    "endpoint_id must be provided or configured in RunPodConfig.endpoint_id"
                )
            endpoint_id = self.config.endpoint_id

        # Create endpoint instance
        endpoint = runpod.Endpoint(endpoint_id)

        # Prepare input payload
        input_data = {
            "input": {
                "model_key": model_uri,
                "data_key": data_uri,
                "output_key": output_uri,
                "volume_id": self.config.network_volume_id,
                "prediction_column_name": prediction_column_name
            }
        }

        if feature_columns:
            input_data["input"]["feature_columns"] = feature_columns

        # Asynchronous invocation
        job = endpoint.run(input_data)
        return job['id']

    def check_job_status(self, endpoint_id: str, job_id: str) -> Dict[str, Any]:
        """
        Check status of async prediction job.

        Args:
            endpoint_id: RunPod endpoint ID
            job_id: Job ID returned from invoke_async

        Returns:
            Job status dictionary with keys:
                - status: str (IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED)
                - output: Optional[Dict] (results if completed)
                - error: Optional[str] (error message if failed)

        Example:
            >>> status = manager.check_job_status(endpoint_id, job_id)
            >>> if status['status'] == 'COMPLETED':
            ...     print(status['output'])
        """
        endpoint = runpod.Endpoint(endpoint_id)
        status = endpoint.status(job_id)
        return status

    def wait_for_completion(
        self,
        endpoint_id: str,
        job_id: str,
        poll_interval: int = 5,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Wait for async job to complete with polling.

        Args:
            endpoint_id: RunPod endpoint ID
            job_id: Job ID returned from invoke_async
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Final job status dictionary

        Raises:
            TimeoutError: If job doesn't complete within timeout

        Example:
            >>> job_id = manager.invoke_async(...)
            >>> result = manager.wait_for_completion(endpoint_id, job_id)
            >>> print(result['output'])
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )

            # Check status
            status = self.check_job_status(endpoint_id, job_id)

            # Check if completed or failed
            if status['status'] == 'COMPLETED':
                return status
            elif status['status'] == 'FAILED':
                raise RuntimeError(
                    f"Job {job_id} failed: {status.get('error', 'Unknown error')}"
                )

            # Wait before next check
            time.sleep(poll_interval)

    def create_endpoint(
        self,
        name: str,
        docker_image: str,
        gpu_type: Optional[str] = None,
        active_workers: int = 0,
        max_workers: int = 3,
        gpu_count: int = 1
    ) -> Dict[str, Any]:
        """
        Create a new RunPod serverless endpoint.

        Args:
            name: Endpoint name
            docker_image: Docker image URL (e.g., username/image:tag)
            gpu_type: GPU type (uses config default if not specified)
            active_workers: Number of always-on workers (0 for autoscaling)
            max_workers: Maximum workers for scaling
            gpu_count: Number of GPUs per worker

        Returns:
            Endpoint details including ID

        Note:
            This method uses RunPod GraphQL API directly as the SDK
            may not have a create_endpoint method yet.

        Example:
            >>> endpoint = manager.create_endpoint(
            ...     name="ml-predictions",
            ...     docker_image="username/ml-agent-prediction:latest",
            ...     gpu_type="NVIDIA RTX A5000"
            ... )
            >>> print(endpoint['id'])
        """
        import requests

        # Use GPU type from config if not specified
        if gpu_type is None:
            gpu_type = self.config.default_gpu_type

        # GraphQL mutation for creating endpoint
        query = """
        mutation CreateEndpoint($input: EndpointInput!) {
            createEndpoint(input: $input) {
                id
                name
                idleTimeout
                scaleType
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "dockerImage": docker_image,
                "gpuTypeId": gpu_type,
                "activeWorkers": active_workers,
                "maxWorkers": max_workers,
                "gpuCount": gpu_count,
                "volumeId": self.config.network_volume_id
            }
        }

        # Make GraphQL request
        response = requests.post(
            'https://api.runpod.io/graphql',
            json={'query': query, 'variables': variables},
            headers={'Authorization': f'Bearer {self.config.runpod_api_key}'}
        )

        result = response.json()

        if 'errors' in result:
            raise RuntimeError(
                f"Failed to create endpoint: {result['errors']}"
            )

        return result['data']['createEndpoint']

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all RunPod serverless endpoints.

        Returns:
            List of endpoint dictionaries

        Example:
            >>> endpoints = manager.list_endpoints()
            >>> for endpoint in endpoints:
            ...     print(endpoint['name'], endpoint['id'])
        """
        import requests

        query = """
        query GetEndpoints {
            myself {
                endpoints {
                    id
                    name
                    gpuIds
                    scaleType
                    activeWorkers
                }
            }
        }
        """

        response = requests.post(
            'https://api.runpod.io/graphql',
            json={'query': query},
            headers={'Authorization': f'Bearer {self.config.runpod_api_key}'}
        )

        result = response.json()

        if 'errors' in result:
            raise RuntimeError(
                f"Failed to list endpoints: {result['errors']}"
            )

        return result['data']['myself']['endpoints']
