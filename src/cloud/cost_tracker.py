"""
CostTracker for AWS cost estimation and tracking.

This module provides cost estimation and tracking for AWS services:
- EC2 spot instance costs (training)
- Lambda function costs (predictions)
- S3 storage costs
- User spending limits and enforcement
- Cost logging to JSON file

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 6.0: CostTracker with TDD)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import CostTrackingError


class CostTracker:
    """
    AWS cost estimation and tracking manager.

    Provides cost estimation before operations and actual cost calculation
    after operations complete. Tracks spending per user and enforces limits.
    """

    # EC2 Spot Instance Pricing (per hour in dollars)
    EC2_SPOT_PRICING = {
        't3.medium': 0.01,
        'm5.large': 0.05,
        'm5.xlarge': 0.10,
        'p3.2xlarge': 1.20,
    }

    # Lambda Pricing Constants
    LAMBDA_PRICE_PER_GB_SECOND = 0.0000166667  # Per GB-second
    LAMBDA_PRICE_PER_REQUEST = 0.0000002  # Per request

    # S3 Pricing Constants
    S3_STORAGE_PRICE_PER_GB_MONTH = 0.023  # Per GB per month

    # RunPod GPU Pricing (Community Cloud - per hour in dollars)
    RUNPOD_GPU_PRICING = {
        'NVIDIA RTX A5000': 0.29,
        'NVIDIA RTX A40': 0.39,
        'NVIDIA A100 PCIe 40GB': 0.79,
        'NVIDIA A100 PCIe 80GB': 1.19,
        'NVIDIA H100 PCIe': 1.99,
    }

    # RunPod Serverless Pricing (per second during execution)
    RUNPOD_SERVERLESS_PRICING = {
        'NVIDIA RTX A5000': 0.29 / 3600,  # Convert hourly to per-second
        'NVIDIA RTX A40': 0.39 / 3600,
        'NVIDIA A100 PCIe 40GB': 0.79 / 3600,
    }

    # RunPod Storage Pricing (per GB per month)
    RUNPOD_STORAGE_PRICE_PER_GB_MONTH = 0.07  # Network volume (stopped)
    RUNPOD_STORAGE_PRICE_PER_GB_MONTH_RUNNING = 0.10  # Network volume (running)

    def __init__(self, config: CloudConfig) -> None:
        """
        Initialize CostTracker with configuration.

        Args:
            config: CloudConfig instance with cost limits and AWS settings
        """
        self._config = config
        self._log_path = Path("data/logs/cloud_costs.json")

    def estimate_training_cost(
        self,
        instance_type: str,
        estimated_time_minutes: float
    ) -> float:
        """
        Estimate EC2 training cost before starting.

        Args:
            instance_type: EC2 instance type (e.g., 'm5.large')
            estimated_time_minutes: Estimated training time in minutes

        Returns:
            float: Estimated cost in dollars

        Raises:
            CostTrackingError: If instance type is unknown

        Example:
            >>> cost = tracker.estimate_training_cost('m5.large', 120)
            >>> print(cost)
            0.10
        """
        # Validate instance type
        if instance_type not in self.EC2_SPOT_PRICING:
            raise CostTrackingError(
                f"Unknown instance type: {instance_type}. "
                f"Supported types: {list(self.EC2_SPOT_PRICING.keys())}",
                operation="estimate_training_cost"
            )

        # Get hourly rate
        hourly_rate = self.EC2_SPOT_PRICING[instance_type]

        # Convert minutes to hours
        hours = estimated_time_minutes / 60.0

        # Calculate cost
        cost = hourly_rate * hours

        return cost

    def calculate_training_cost(
        self,
        instance_id: str,
        instance_type: str
    ) -> float:
        """
        Calculate actual EC2 training cost after completion.

        Queries EC2 to get actual runtime and calculates cost based on
        spot pricing for the instance type.

        Args:
            instance_id: EC2 instance ID
            instance_type: EC2 instance type (e.g., 'm5.large')

        Returns:
            float: Actual cost in dollars

        Raises:
            CostTrackingError: If instance not found or runtime cannot be determined

        Example:
            >>> cost = tracker.calculate_training_cost('i-12345', 'm5.large')
            >>> print(cost)
            0.085
        """
        # Get instance runtime
        runtime_minutes = self._get_instance_runtime(instance_id)

        # Calculate cost using estimate function
        cost = self.estimate_training_cost(instance_type, runtime_minutes)

        return cost

    def estimate_prediction_cost(
        self,
        num_rows: int,
        lambda_memory_mb: int = 3008,
        estimated_time_seconds: float = 60
    ) -> float:
        """
        Estimate Lambda prediction cost before execution.

        Args:
            num_rows: Number of rows to predict
            lambda_memory_mb: Lambda memory allocation in MB
            estimated_time_seconds: Estimated execution time in seconds

        Returns:
            float: Estimated cost in dollars

        Example:
            >>> cost = tracker.estimate_prediction_cost(1000, 3008, 60)
            >>> print(cost)
            0.00293
        """
        # Convert memory to GB
        memory_gb = lambda_memory_mb / 1024.0

        # Calculate compute cost: GB-seconds * price per GB-second
        compute_cost = memory_gb * estimated_time_seconds * self.LAMBDA_PRICE_PER_GB_SECOND

        # Calculate request cost: 1 request
        request_cost = self.LAMBDA_PRICE_PER_REQUEST

        # Total cost
        total_cost = compute_cost + request_cost

        return total_cost

    def calculate_prediction_cost(
        self,
        execution_time_ms: float,
        lambda_memory_mb: int = 3008
    ) -> float:
        """
        Calculate actual Lambda prediction cost after execution.

        Args:
            execution_time_ms: Actual execution time in milliseconds
            lambda_memory_mb: Lambda memory allocation in MB

        Returns:
            float: Actual cost in dollars

        Example:
            >>> cost = tracker.calculate_prediction_cost(5000, 3008)
            >>> print(cost)
            0.000244
        """
        # Convert milliseconds to seconds
        execution_time_seconds = execution_time_ms / 1000.0

        # Convert memory to GB
        memory_gb = lambda_memory_mb / 1024.0

        # Calculate compute cost: GB-seconds * price per GB-second
        compute_cost = memory_gb * execution_time_seconds * self.LAMBDA_PRICE_PER_GB_SECOND

        # Calculate request cost: 1 request
        request_cost = self.LAMBDA_PRICE_PER_REQUEST

        # Total cost
        total_cost = compute_cost + request_cost

        return total_cost

    def calculate_s3_storage_cost(
        self,
        user_id: int,
        s3_manager: Any
    ) -> Dict[str, float]:
        """
        Calculate S3 storage cost for user.

        Args:
            user_id: User ID to calculate storage for
            s3_manager: S3Manager instance for listing files

        Returns:
            dict: Storage breakdown with keys:
                - total_storage_gb: Total storage in GB
                - monthly_cost: Monthly cost in dollars
                - datasets_gb: Dataset storage in GB
                - models_gb: Model storage in GB

        Example:
            >>> result = tracker.calculate_s3_storage_cost(12345, s3_manager)
            >>> print(result)
            {
                'total_storage_gb': 1.5,
                'monthly_cost': 0.0345,
                'datasets_gb': 0.5,
                'models_gb': 1.0
            }
        """
        # Get datasets
        datasets = s3_manager.list_user_datasets(user_id=user_id)
        datasets_size_mb = sum(d['size_mb'] for d in datasets)

        # Get models
        models = s3_manager.list_user_models(user_id=user_id)
        models_size_mb = sum(m['size_mb'] for m in models)

        # Convert to GB
        datasets_gb = datasets_size_mb / 1024.0
        models_gb = models_size_mb / 1024.0
        total_gb = datasets_gb + models_gb

        # Calculate monthly cost
        monthly_cost = total_gb * self.S3_STORAGE_PRICE_PER_GB_MONTH

        return {
            'total_storage_gb': total_gb,
            'monthly_cost': monthly_cost,
            'datasets_gb': datasets_gb,
            'models_gb': models_gb
        }

    def get_user_total_spend(self, user_id: int) -> float:
        """
        Get total spending for user from cost logs.

        Args:
            user_id: User ID to get spending for

        Returns:
            float: Total spending in dollars

        Example:
            >>> total = tracker.get_user_total_spend(12345)
            >>> print(total)
            5.67
        """
        # Check if log file exists
        if not self._log_path.exists():
            return 0.0

        # Read log file and sum costs for user
        total_spend = 0.0

        try:
            with open(self._log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('user_id') == user_id:
                        total_spend += entry.get('cost', 0.0)
        except (IOError, json.JSONDecodeError) as e:
            raise CostTrackingError(
                f"Error reading cost log: {e}",
                operation="get_user_total_spend"
            )

        return total_spend

    def check_cost_limit(
        self,
        user_id: int,
        operation_cost: float,
        limit_type: str
    ) -> bool:
        """
        Check if operation would exceed user's cost limit.

        Args:
            user_id: User ID to check limit for
            operation_cost: Cost of proposed operation in dollars
            limit_type: Type of limit ('training' or 'prediction')

        Returns:
            bool: True if within limit, False if would exceed

        Raises:
            CostTrackingError: If limit_type is invalid

        Example:
            >>> can_proceed = tracker.check_cost_limit(12345, 0.50, 'training')
            >>> print(can_proceed)
            True
        """
        # Validate limit type
        if limit_type not in ['training', 'prediction']:
            raise CostTrackingError(
                f"Invalid limit type: {limit_type}. Must be 'training' or 'prediction'",
                operation="check_cost_limit"
            )

        # Get cost limit
        if limit_type == 'training':
            limit = self._config.max_training_cost_dollars
        else:  # prediction
            limit = self._config.max_prediction_cost_dollars

        # Get current spending
        current_spend = self.get_user_total_spend(user_id=user_id)

        # Check if proposed cost would exceed limit
        would_exceed = (current_spend + operation_cost) > limit

        return not would_exceed

    # RunPod Cost Estimation Methods (Task 3.2-3.4)

    def estimate_runpod_training_cost(
        self,
        gpu_type: str,
        estimated_time_seconds: int
    ) -> float:
        """
        Estimate RunPod GPU training cost with per-second billing.

        RunPod bills per-second for GPU usage, unlike AWS EC2 which bills
        hourly. This provides more accurate cost estimation for shorter jobs.

        Args:
            gpu_type: GPU type (e.g., 'NVIDIA RTX A5000')
            estimated_time_seconds: Estimated training time in seconds

        Returns:
            float: Estimated cost in dollars

        Raises:
            CostTrackingError: If GPU type is unknown

        Example:
            >>> cost = tracker.estimate_runpod_training_cost('NVIDIA RTX A5000', 1800)
            >>> print(cost)
            0.145
        """
        # Validate GPU type
        if gpu_type not in self.RUNPOD_GPU_PRICING:
            raise CostTrackingError(
                f"Unknown GPU type: {gpu_type}. "
                f"Supported types: {list(self.RUNPOD_GPU_PRICING.keys())}",
                operation="estimate_runpod_training_cost"
            )

        # Get hourly rate
        hourly_rate = self.RUNPOD_GPU_PRICING[gpu_type]

        # Convert seconds to hours
        hours = estimated_time_seconds / 3600.0

        # Calculate cost
        cost = hourly_rate * hours

        return cost

    def estimate_runpod_prediction_cost(
        self,
        num_rows: int,
        gpu_type: str = 'NVIDIA RTX A5000',
        estimated_time_seconds: float = 10
    ) -> float:
        """
        Estimate RunPod serverless prediction cost.

        RunPod serverless bills per-second only during execution (no idle costs).
        Much cheaper than AWS Lambda for GPU inference.

        Args:
            num_rows: Number of rows to predict
            gpu_type: GPU type for serverless endpoint
            estimated_time_seconds: Estimated execution time in seconds

        Returns:
            float: Estimated cost in dollars

        Example:
            >>> cost = tracker.estimate_runpod_prediction_cost(1000, 'NVIDIA RTX A5000', 10)
            >>> print(cost)
            0.0008
        """
        # Validate GPU type
        if gpu_type not in self.RUNPOD_SERVERLESS_PRICING:
            raise CostTrackingError(
                f"Unknown serverless GPU type: {gpu_type}. "
                f"Supported types: {list(self.RUNPOD_SERVERLESS_PRICING.keys())}",
                operation="estimate_runpod_prediction_cost"
            )

        # Get per-second rate
        per_second_rate = self.RUNPOD_SERVERLESS_PRICING[gpu_type]

        # Calculate cost: per-second rate * execution time
        cost = per_second_rate * estimated_time_seconds

        return cost

    def calculate_runpod_storage_cost(
        self,
        user_id: int,
        storage_manager: Any,
        is_running: bool = False
    ) -> Dict[str, float]:
        """
        Calculate RunPod network volume storage cost.

        RunPod has different rates for stopped vs running volumes.
        Cheaper than AWS S3 ($0.07/GB vs $0.023/GB) but limited to
        one volume per region.

        Args:
            user_id: User ID to calculate storage for
            storage_manager: RunPodStorageManager instance
            is_running: Whether volume is attached to running pod

        Returns:
            dict: Storage breakdown with keys:
                - total_storage_gb: Total storage in GB
                - monthly_cost: Monthly cost in dollars
                - datasets_gb: Dataset storage in GB
                - models_gb: Model storage in GB

        Example:
            >>> result = tracker.calculate_runpod_storage_cost(12345, storage_mgr)
            >>> print(result['monthly_cost'])
            3.50
        """
        # Get storage breakdown from storage manager
        datasets = storage_manager.list_user_datasets(user_id)
        models = storage_manager.list_user_models(user_id)

        # Calculate total storage in GB
        datasets_gb = sum(d.get('size_bytes', 0) for d in datasets) / (1024**3)
        models_gb = sum(m.get('size_bytes', 0) for m in models) / (1024**3)
        total_storage_gb = datasets_gb + models_gb

        # Select pricing based on running state
        price_per_gb = (
            self.RUNPOD_STORAGE_PRICE_PER_GB_MONTH_RUNNING
            if is_running
            else self.RUNPOD_STORAGE_PRICE_PER_GB_MONTH
        )

        # Calculate monthly cost
        monthly_cost = total_storage_gb * price_per_gb

        return {
            'total_storage_gb': total_storage_gb,
            'monthly_cost': monthly_cost,
            'datasets_gb': datasets_gb,
            'models_gb': models_gb,
            'price_per_gb': price_per_gb
        }

    def _get_instance_runtime(self, instance_id: str) -> float:
        """
        Get EC2 instance runtime in minutes.

        Queries EC2 to get launch time and calculates runtime based on
        current time (for running instances) or termination time (for
        terminated instances).

        Args:
            instance_id: EC2 instance ID

        Returns:
            float: Runtime in minutes

        Raises:
            CostTrackingError: If instance not found or runtime cannot be determined
        """
        try:
            # Create EC2 client
            ec2_client = boto3.client(
                'ec2',
                region_name=self._config.aws_region,
                aws_access_key_id=self._config.aws_access_key_id,
                aws_secret_access_key=self._config.aws_secret_access_key
            )

            # Describe instance
            response = ec2_client.describe_instances(InstanceIds=[instance_id])

            # Check if instance exists
            if not response['Reservations']:
                raise CostTrackingError(
                    f"Instance not found: {instance_id}",
                    operation="_get_instance_runtime"
                )

            instance = response['Reservations'][0]['Instances'][0]
            launch_time = instance['LaunchTime']
            state = instance['State']['Name']

            # Calculate runtime based on state
            if state == 'running':
                # Use current time
                end_time = datetime.now()
            else:
                # Extract termination time from StateTransitionReason
                # Format: "User initiated (2025-10-24T11:30:00.000Z)"
                reason = instance.get('StateTransitionReason', '')
                try:
                    # Extract ISO timestamp
                    start = reason.index('(') + 1
                    end = reason.index(')')
                    time_str = reason[start:end]
                    end_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                except (ValueError, IndexError):
                    # If parsing fails, use current time
                    end_time = datetime.now()

            # Make launch_time timezone-naive for comparison
            if launch_time.tzinfo is not None:
                launch_time = launch_time.replace(tzinfo=None)
            if end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)

            # Calculate runtime in minutes
            runtime_seconds = (end_time - launch_time).total_seconds()
            runtime_minutes = runtime_seconds / 60.0

            return runtime_minutes

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))

            raise CostTrackingError(
                f"Failed to get instance runtime: {error_message}",
                operation="_get_instance_runtime"
            )

    def _log_cost(
        self,
        service: str,
        operation: str,
        **kwargs
    ) -> None:
        """
        Log cost entry to JSON file.

        Args:
            service: Cloud service name (ec2, lambda, s3, runpod_gpu, runpod_serverless, runpod_storage)
            operation: Operation performed (training, prediction, storage)
            **kwargs: Additional fields (user_id, cost, instance_type, gpu_type, etc.)

        Example:
            >>> tracker._log_cost(
            ...     service='ec2',
            ...     operation='training',
            ...     user_id=12345,
            ...     cost=0.50,
            ...     instance_type='m5.large',
            ...     duration_minutes=60
            ... )
        """
        # Create parent directory if needed
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare log entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'service': service,
            'operation': operation,
            **kwargs
        }

        # Append to log file
        try:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except IOError as e:
            raise CostTrackingError(
                f"Error writing to cost log: {e}",
                operation="_log_cost"
            )
