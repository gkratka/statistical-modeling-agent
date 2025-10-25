# RunPod Migration - Detailed Sub-Tasks

**Project**: Migrate Cloud ML Workflows from AWS to RunPod
**Date**: 2025-10-24
**Status**: Planning
**Estimated Effort**: 46-62 hours (~1.5-2 weeks)

---

## Overview

This document provides actionable sub-tasks for migrating the cloud-based ML training and prediction infrastructure from AWS to RunPod. The migration maintains existing functionality while leveraging RunPod's simpler architecture, lower costs, and GPU-focused platform.

**Key Benefits of RunPod**:
- 40-80% lower GPU costs compared to AWS
- Per-second billing (not per-hour)
- No data transfer fees
- Simpler configuration (API keys vs IAM roles)
- Actual GPU models (RTX A5000, A40, A100) vs instance types

---

## Relevant Files

### New Files to Create

**RunPod Infrastructure Layer**:
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_config.py` - RunPod configuration dataclass
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_client.py` - RunPod SDK client wrapper
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_pod_manager.py` - GPU pod management (replaces EC2Manager)
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_serverless_manager.py` - Serverless endpoint management (replaces LambdaManager)

**RunPod Serverless**:
- `/Users/gkratka/Documents/statistical-modeling-agent/runpod/prediction_handler.py` - RunPod serverless handler
- `/Users/gkratka/Documents/statistical-modeling-agent/runpod/Dockerfile` - Container for serverless predictions
- `/Users/gkratka/Documents/statistical-modeling-agent/runpod/requirements.txt` - RunPod handler dependencies

**Scripts**:
- `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/setup_runpod.py` - RunPod infrastructure setup script
- `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/package_runpod.sh` - RunPod deployment packaging

**Abstraction Layer**:
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/provider_interface.py` - Cloud provider interface definitions
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/provider_factory.py` - Factory for cloud provider selection

### Files to Modify

- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py` - Update S3 endpoint to RunPod storage
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py` - Update pricing for RunPod GPUs
- `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py` - Simplify for RunPod (remove IAM policies)
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_training_handlers.py` - Update to use RunPod managers
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_prediction_handlers.py` - Update to use RunPod managers
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/messages/cloud_messages.py` - Update terminology (AWS→RunPod)
- `/Users/gkratka/Documents/statistical-modeling-agent/config/config.yaml` - Add RunPod configuration section
- `/Users/gkratka/Documents/statistical-modeling-agent/.env.example` - Add RunPod credentials
- `/Users/gkratka/Documents/statistical-modeling-agent/requirements.txt` - Add `runpod` SDK

### Test Files

- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_config.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_client.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_pod_manager.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_serverless_manager.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_provider_factory.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/tests/integration/test_runpod_workflows.py`

---

## 1.0 Create Cloud Provider Abstraction Layer

**Goal**: Abstract cloud operations to support multiple providers (AWS and RunPod) side-by-side.

**Complexity**: 🟡 Medium (6-9 hours)

### Sub-Tasks

- [x] **1.1 Define cloud provider interfaces**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/provider_interface.py`
  - Create abstract base classes:
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Dict, Optional
    from pathlib import Path

    class CloudStorageProvider(ABC):
        """Abstract interface for cloud storage operations."""

        @abstractmethod
        def upload_dataset(self, user_id: int, file_path: str, dataset_name: Optional[str] = None) -> str:
            """Upload dataset to cloud storage."""
            pass

        @abstractmethod
        def save_model(self, user_id: int, model_id: str, model_dir: Path) -> str:
            """Save model to cloud storage."""
            pass

        @abstractmethod
        def load_model(self, user_id: int, model_id: str, local_dir: Path) -> Path:
            """Load model from cloud storage."""
            pass

    class CloudTrainingProvider(ABC):
        """Abstract interface for cloud training operations."""

        @abstractmethod
        def select_compute_type(self, dataset_size_mb: float, model_type: str) -> str:
            """Select optimal compute resource for training."""
            pass

        @abstractmethod
        def launch_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
            """Launch training job on cloud compute."""
            pass

        @abstractmethod
        def monitor_training(self, job_id: str) -> Dict[str, Any]:
            """Monitor training job status."""
            pass

    class CloudPredictionProvider(ABC):
        """Abstract interface for cloud prediction operations."""

        @abstractmethod
        def invoke_prediction(self, model_uri: str, data_uri: str, output_uri: str) -> Dict[str, Any]:
            """Invoke prediction service."""
            pass
    ```
  - Test coverage: Interface definition tests

- [x] **1.2 Create provider factory**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/provider_factory.py`
  - Implement factory pattern:
    ```python
    from typing import Literal
    from src.cloud.provider_interface import (
        CloudStorageProvider,
        CloudTrainingProvider,
        CloudPredictionProvider
    )
    from src.cloud.aws_config import CloudConfig
    from src.cloud.runpod_config import RunPodConfig

    ProviderType = Literal["aws", "runpod"]

    class CloudProviderFactory:
        """Factory for creating cloud provider instances."""

        @staticmethod
        def create_storage_provider(
            provider: ProviderType,
            config: CloudConfig | RunPodConfig
        ) -> CloudStorageProvider:
            if provider == "aws":
                from src.cloud.s3_manager import S3Manager
                return S3Manager(config)
            elif provider == "runpod":
                from src.cloud.runpod_storage_manager import RunPodStorageManager
                return RunPodStorageManager(config)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        @staticmethod
        def create_training_provider(
            provider: ProviderType,
            config: CloudConfig | RunPodConfig
        ) -> CloudTrainingProvider:
            if provider == "aws":
                from src.cloud.ec2_manager import EC2Manager
                return EC2Manager(config)
            elif provider == "runpod":
                from src.cloud.runpod_pod_manager import RunPodPodManager
                return RunPodPodManager(config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
    ```
  - Test cases:
    - `test_create_aws_storage_provider`
    - `test_create_runpod_storage_provider`
    - `test_invalid_provider_raises_error`

- [x] **1.3 Update existing AWS managers to implement interfaces**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/s3_manager.py`
  - Make S3Manager inherit from CloudStorageProvider
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/ec2_manager.py`
  - Make EC2Manager inherit from CloudTrainingProvider
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/lambda_manager.py`
  - Make LambdaManager inherit from CloudPredictionProvider
  - Test: Ensure existing tests still pass

- [x] **1.4 Add provider selection to config**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/config/config.yaml`
  - Add `cloud_provider` field:
    ```yaml
    cloud:
      provider: "aws"  # or "runpod"
      # ... rest of config
    ```
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/.env.example`
  - Add `CLOUD_PROVIDER=aws` example

---

## 2.0 Migrate Storage Layer to RunPod Network Volumes

**Goal**: Update S3Manager to support RunPod's S3-compatible storage endpoint.

**Complexity**: 🟢 Low (2-3 hours)

### Sub-Tasks

- [x] **2.1 Create RunPodStorageManager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_storage_manager.py`
  - Extend S3Manager with RunPod-specific endpoint:
    ```python
    from src.cloud.s3_manager import S3Manager
    from src.cloud.runpod_config import RunPodConfig
    import boto3

    class RunPodStorageManager(S3Manager):
        """RunPod network volume manager using S3-compatible API."""

        def __init__(self, config: RunPodConfig):
            self.config = config
            # Override S3 client with RunPod endpoint
            self._s3_client = boto3.client(
                's3',
                endpoint_url='https://storage.runpod.io',
                aws_access_key_id=config.runpod_api_key,
                aws_secret_access_key=config.runpod_secret_key
            )

        # All other methods inherited from S3Manager work unchanged
    ```
  - Test: Upload and download with RunPod network volume

- [x] **2.2 Update S3 path validation for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_storage_manager.py`
  - Override `validate_s3_path` to handle RunPod volume naming:
    ```python
    def validate_s3_path(self, s3_uri: str, user_id: int) -> bool:
        """Validate RunPod network volume path."""
        # RunPod uses volume IDs instead of bucket names
        # Format: runpod://volume_id/path
        if not s3_uri.startswith('runpod://'):
            raise ValueError("RunPod URIs must start with runpod://")
        # ... validation logic
    ```
  - Test: Path validation for RunPod volumes

- [x] **2.3 Test multipart upload with RunPod**
  - Use existing `_multipart_upload()` method from S3Manager
  - Test with large files (>5MB) to ensure compatibility
  - Test cases:
    - `test_runpod_multipart_upload_success`
    - `test_runpod_upload_large_dataset`

---

## 3.0 Migrate Cost Tracking to RunPod Pricing

**Goal**: Replace AWS pricing with RunPod GPU pricing and per-second billing.

**Complexity**: 🟡 Medium (6-8 hours)

### Sub-Tasks

- [x] **3.1 Create RunPod pricing constants**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add RunPod pricing:
    ```python
    # RunPod GPU Pricing (Community Cloud - per hour)
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

    # RunPod Storage Pricing
    RUNPOD_STORAGE_PRICE_PER_GB_MONTH = 0.07  # Network volume (stopped)
    RUNPOD_STORAGE_PRICE_PER_GB_MONTH_RUNNING = 0.10  # Network volume (running)
    ```

- [x] **3.2 Implement per-second billing calculations**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add methods:
    ```python
    def estimate_runpod_training_cost(
        self,
        gpu_type: str,
        estimated_time_seconds: int
    ) -> float:
        """Estimate RunPod training cost (per-second billing)."""
        hourly_rate = RUNPOD_GPU_PRICING.get(gpu_type, 0.39)
        per_second_rate = hourly_rate / 3600
        return per_second_rate * estimated_time_seconds

    def calculate_runpod_training_cost(
        self,
        pod_id: str,
        gpu_type: str,
        start_time: float,
        end_time: float
    ) -> float:
        """Calculate actual RunPod training cost."""
        duration_seconds = end_time - start_time
        return self.estimate_runpod_training_cost(gpu_type, int(duration_seconds))
    ```
  - Test cases:
    - `test_estimate_runpod_training_cost_a5000`
    - `test_estimate_runpod_training_cost_a100`
    - `test_calculate_runpod_training_cost_accuracy`

- [x] **3.3 Implement RunPod prediction cost estimation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add method:
    ```python
    def estimate_runpod_prediction_cost(
        self,
        num_rows: int,
        gpu_type: str = 'NVIDIA RTX A5000',
        estimated_time_seconds: int = 60
    ) -> float:
        """Estimate RunPod serverless prediction cost."""
        per_second_rate = RUNPOD_SERVERLESS_PRICING.get(gpu_type)

        # RunPod serverless has no request fees (unlike Lambda)
        execution_cost = per_second_rate * estimated_time_seconds

        return execution_cost
    ```
  - Test: Prediction cost calculation

- [x] **3.4 Implement RunPod storage cost calculation**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Add method:
    ```python
    def calculate_runpod_storage_cost(
        self,
        user_id: int,
        storage_manager
    ) -> Dict[str, float]:
        """Calculate monthly RunPod storage cost."""
        # List user's datasets and models
        datasets = storage_manager.list_user_datasets(user_id)
        models = storage_manager.list_user_models(user_id)

        total_size_gb = 0
        for dataset in datasets:
            total_size_gb += dataset['size_mb'] / 1024
        for model in models:
            total_size_gb += model['size_mb'] / 1024

        # RunPod network volume pricing (stopped)
        monthly_cost = total_size_gb * RUNPOD_STORAGE_PRICE_PER_GB_MONTH

        return {
            'total_size_gb': total_size_gb,
            'monthly_cost': monthly_cost,
            'datasets_cost': (sum(d['size_mb'] for d in datasets) / 1024) * RUNPOD_STORAGE_PRICE_PER_GB_MONTH,
            'models_cost': (sum(m['size_mb'] for m in models) / 1024) * RUNPOD_STORAGE_PRICE_PER_GB_MONTH
        }
    ```
  - Test: Storage cost calculation

- [x] **3.5 Update cost logging for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/cost_tracker.py`
  - Update `_log_cost` to support RunPod:
    ```python
    def _log_cost(
        self,
        service: str,  # 'runpod_pod', 'runpod_serverless', 'runpod_storage'
        operation: str,
        user_id: int,
        cost_usd: float,
        **kwargs
    ) -> None:
        """Log cost to data/logs/cloud_costs.json."""
        # Same implementation, just different service names
    ```

- [x] **3.6 Write tests for RunPod cost tracking**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_cost_tracking.py`
  - Test cases:
    - `test_estimate_runpod_training_cost_per_second`
    - `test_runpod_prediction_cost_no_request_fees`
    - `test_runpod_storage_cost_calculation`
    - `test_cost_logging_runpod_format`

---

## 4.0 Migrate Security & Configuration to RunPod

**Goal**: Create RunPod configuration and simplify security (no IAM policies).

**Complexity**: 🟢 Low (5-7 hours)

### Sub-Tasks

- [x] **4.1 Create RunPodConfig dataclass**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_config.py`
  - Define configuration:
    ```python
    from dataclasses import dataclass
    from typing import Optional
    from pathlib import Path
    import yaml
    import os

    @dataclass
    class RunPodConfig:
        """RunPod configuration."""

        # Required fields
        runpod_api_key: str
        storage_endpoint: str  # https://storage.runpod.io
        network_volume_id: str

        # GPU configuration
        default_gpu_type: str  # e.g., 'NVIDIA RTX A5000'
        cloud_type: str  # 'COMMUNITY' or 'SECURE'

        # Storage configuration
        storage_access_key: str
        storage_secret_key: str
        data_prefix: str  # 'datasets'
        models_prefix: str  # 'models'

        # Cost limits
        max_training_cost_dollars: float = 10.0
        max_prediction_cost_dollars: float = 1.0

        # Optional
        docker_registry: Optional[str] = None  # For custom images

        @classmethod
        def from_yaml(cls, config_path: str) -> "RunPodConfig":
            """Load configuration from YAML file."""
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            runpod_config = config_data.get('runpod', {})
            return cls(
                runpod_api_key=runpod_config['api_key'],
                storage_endpoint=runpod_config['storage_endpoint'],
                network_volume_id=runpod_config['network_volume_id'],
                default_gpu_type=runpod_config.get('default_gpu_type', 'NVIDIA RTX A5000'),
                cloud_type=runpod_config.get('cloud_type', 'COMMUNITY'),
                storage_access_key=runpod_config['storage_access_key'],
                storage_secret_key=runpod_config['storage_secret_key'],
                data_prefix=runpod_config.get('data_prefix', 'datasets'),
                models_prefix=runpod_config.get('models_prefix', 'models'),
                max_training_cost_dollars=runpod_config.get('max_training_cost_dollars', 10.0),
                max_prediction_cost_dollars=runpod_config.get('max_prediction_cost_dollars', 1.0),
                docker_registry=runpod_config.get('docker_registry')
            )

        @classmethod
        def from_env(cls) -> "RunPodConfig":
            """Load configuration from environment variables."""
            return cls(
                runpod_api_key=os.getenv('RUNPOD_API_KEY', ''),
                storage_endpoint=os.getenv('RUNPOD_STORAGE_ENDPOINT', 'https://storage.runpod.io'),
                network_volume_id=os.getenv('RUNPOD_NETWORK_VOLUME_ID', ''),
                default_gpu_type=os.getenv('RUNPOD_DEFAULT_GPU_TYPE', 'NVIDIA RTX A5000'),
                cloud_type=os.getenv('RUNPOD_CLOUD_TYPE', 'COMMUNITY'),
                storage_access_key=os.getenv('RUNPOD_STORAGE_ACCESS_KEY', ''),
                storage_secret_key=os.getenv('RUNPOD_STORAGE_SECRET_KEY', ''),
                data_prefix=os.getenv('RUNPOD_DATA_PREFIX', 'datasets'),
                models_prefix=os.getenv('RUNPOD_MODELS_PREFIX', 'models'),
                max_training_cost_dollars=float(os.getenv('RUNPOD_MAX_TRAINING_COST', '10.0')),
                max_prediction_cost_dollars=float(os.getenv('RUNPOD_MAX_PREDICTION_COST', '1.0')),
                docker_registry=os.getenv('RUNPOD_DOCKER_REGISTRY')
            )

        def validate(self) -> None:
            """Validate required configuration fields."""
            if not self.runpod_api_key:
                raise ValueError("runpod_api_key is required")
            if not self.network_volume_id:
                raise ValueError("network_volume_id is required")
            if not self.storage_access_key or not self.storage_secret_key:
                raise ValueError("storage credentials are required")
            if self.cloud_type not in ['COMMUNITY', 'SECURE']:
                raise ValueError("cloud_type must be COMMUNITY or SECURE")
    ```
  - Test cases:
    - `test_runpod_config_from_yaml`
    - `test_runpod_config_from_env`
    - `test_runpod_config_validation`

- [ ] **4.2 Create RunPodClient wrapper**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_client.py`
  - Implement client:
    ```python
    import runpod
    import boto3
    from src.cloud.runpod_config import RunPodConfig

    class RunPodClient:
        """RunPod SDK client wrapper."""

        def __init__(self, config: RunPodConfig):
            self.config = config
            runpod.api_key = config.runpod_api_key

            # S3-compatible storage client
            self._storage_client = None

        def get_storage_client(self) -> boto3.client:
            """Get S3-compatible storage client for RunPod network volumes."""
            if self._storage_client is None:
                self._storage_client = boto3.client(
                    's3',
                    endpoint_url=self.config.storage_endpoint,
                    aws_access_key_id=self.config.storage_access_key,
                    aws_secret_access_key=self.config.storage_secret_key
                )
            return self._storage_client

        def health_check(self) -> dict:
            """Check RunPod API connectivity."""
            try:
                # Test API key validity
                pods = runpod.get_pods()
                storage_healthy = self._test_storage_access()

                return {
                    'api': True,
                    'storage': storage_healthy,
                    'pod_count': len(pods)
                }
            except Exception as e:
                return {
                    'api': False,
                    'storage': False,
                    'error': str(e)
                }

        def _test_storage_access(self) -> bool:
            """Test storage endpoint accessibility."""
            try:
                s3 = self.get_storage_client()
                # List objects in volume (should not error)
                s3.list_objects_v2(
                    Bucket=self.config.network_volume_id,
                    MaxKeys=1
                )
                return True
            except Exception:
                return False
    ```
  - Test: Health check and storage access

- [ ] **4.3 Update config.yaml for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/config/config.yaml`
  - Add RunPod section:
    ```yaml
    # Cloud provider selection
    cloud_provider: "aws"  # Options: "aws", "runpod"

    # RunPod Configuration
    runpod:
      api_key: "${RUNPOD_API_KEY}"
      storage_endpoint: "https://storage.runpod.io"
      network_volume_id: "${RUNPOD_NETWORK_VOLUME_ID}"

      # GPU Configuration
      default_gpu_type: "NVIDIA RTX A5000"
      cloud_type: "COMMUNITY"  # or "SECURE"

      # Storage Configuration
      storage_access_key: "${RUNPOD_STORAGE_ACCESS_KEY}"
      storage_secret_key: "${RUNPOD_STORAGE_SECRET_KEY}"
      data_prefix: "datasets"
      models_prefix: "models"

      # Cost Limits
      max_training_cost_dollars: 10.0
      max_prediction_cost_dollars: 1.0

      # Optional
      docker_registry: null  # e.g., "username" for Docker Hub
    ```

- [ ] **4.4 Update .env.example for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/.env.example`
  - Add RunPod variables:
    ```bash
    # Cloud Provider Selection
    CLOUD_PROVIDER=aws  # Options: aws, runpod

    # RunPod Credentials
    RUNPOD_API_KEY=your_runpod_api_key_here
    RUNPOD_NETWORK_VOLUME_ID=your_network_volume_id_here
    RUNPOD_STORAGE_ACCESS_KEY=your_storage_access_key_here
    RUNPOD_STORAGE_SECRET_KEY=your_storage_secret_key_here

    # RunPod Configuration
    RUNPOD_STORAGE_ENDPOINT=https://storage.runpod.io
    RUNPOD_DEFAULT_GPU_TYPE=NVIDIA RTX A5000
    RUNPOD_CLOUD_TYPE=COMMUNITY
    RUNPOD_DATA_PREFIX=datasets
    RUNPOD_MODELS_PREFIX=models
    RUNPOD_MAX_TRAINING_COST=10.0
    RUNPOD_MAX_PREDICTION_COST=1.0
    ```

- [ ] **4.5 Simplify security for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/security.py`
  - Add RunPod security manager (much simpler - no IAM):
    ```python
    class RunPodSecurityManager:
        """Simplified security for RunPod (no IAM policies needed)."""

        def __init__(self, config: RunPodConfig):
            self.config = config

        def validate_user_storage_access(
            self,
            storage_key: str,
            user_id: int,
            operation: str
        ) -> bool:
            """Validate user can access storage path."""
            user_prefix = f"user_{user_id}"

            if operation == 'write':
                allowed_prefixes = [
                    f"{self.config.data_prefix}/{user_prefix}/",
                    f"{self.config.models_prefix}/{user_prefix}/",
                    f"predictions/{user_prefix}/"
                ]
                if not any(storage_key.startswith(prefix) for prefix in allowed_prefixes):
                    raise ValueError(f"Write access denied: {storage_key}")

            return True

        def audit_log_operation(
            self,
            user_id: int,
            operation: str,
            resource: str,
            success: bool,
            **metadata
        ) -> None:
            """Log security-relevant operation."""
            # Same implementation as AWS version
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'operation': operation,
                'resource': resource,
                'success': success,
                'provider': 'runpod',
                **metadata
            }
            # ... log to file
    ```
  - No bucket policies, IAM roles, or encryption methods needed (RunPod handles)

- [ ] **4.6 Write tests for RunPod configuration**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_config.py`
  - Test cases:
    - `test_runpod_config_from_yaml_valid`
    - `test_runpod_config_validation_missing_api_key`
    - `test_runpod_config_validation_invalid_cloud_type`
    - `test_runpod_client_health_check`

---

## 5.0 Migrate Prediction Service to RunPod Serverless

**Goal**: Replace Lambda with RunPod Serverless GPU endpoints.

**Complexity**: 🟡 Medium (8-10 hours)

### Sub-Tasks

- [ ] **5.1 Create RunPod serverless handler**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/runpod/prediction_handler.py`
  - Implement handler:
    ```python
    import runpod
    import joblib
    import pandas as pd
    import boto3
    from io import BytesIO
    import json

    # Initialize storage client (RunPod S3-compatible)
    s3_client = boto3.client(
        's3',
        endpoint_url='https://storage.runpod.io',
        aws_access_key_id=os.getenv('STORAGE_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('STORAGE_SECRET_KEY')
    )

    def download_from_storage(volume_id: str, key: str) -> bytes:
        """Download file from RunPod network volume."""
        obj = s3_client.get_object(Bucket=volume_id, Key=key)
        return obj['Body'].read()

    def upload_to_storage(volume_id: str, key: str, data: bytes) -> None:
        """Upload file to RunPod network volume."""
        s3_client.put_object(Bucket=volume_id, Key=key, Body=data)

    def handler(event):
        """RunPod serverless handler for ML predictions."""
        try:
            # Parse input
            input_data = event['input']
            model_key = input_data['model_key']
            data_key = input_data['data_key']
            output_key = input_data['output_key']
            volume_id = input_data['volume_id']
            prediction_column_name = input_data.get('prediction_column_name', 'prediction')
            feature_columns = input_data.get('feature_columns')

            # Download model
            model_bytes = download_from_storage(volume_id, f"{model_key}/model.pkl")
            model = joblib.load(BytesIO(model_bytes))

            # Download preprocessor
            prep_bytes = download_from_storage(volume_id, f"{model_key}/preprocessor.pkl")
            preprocessor = joblib.load(BytesIO(prep_bytes))

            # Download data
            data_bytes = download_from_storage(volume_id, data_key)
            df = pd.read_csv(BytesIO(data_bytes))

            # Select features if specified
            if feature_columns:
                X = df[feature_columns]
            else:
                X = df

            # Preprocess and predict
            X_processed = preprocessor.transform(X)
            predictions = model.predict(X_processed)

            # Add predictions to dataframe
            result_df = df.copy()
            result_df[prediction_column_name] = predictions

            # Upload results
            result_csv = result_df.to_csv(index=False)
            upload_to_storage(volume_id, output_key, result_csv.encode())

            return {
                "output": {
                    "success": True,
                    "output_key": output_key,
                    "num_predictions": len(predictions),
                    "volume_id": volume_id
                }
            }

        except Exception as e:
            return {
                "output": {
                    "success": False,
                    "error": str(e)
                }
            }

    # Start RunPod serverless worker
    if __name__ == '__main__':
        runpod.serverless.start({'handler': handler})
    ```

- [ ] **5.2 Create Dockerfile for RunPod serverless**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/runpod/Dockerfile`
  - Create container:
    ```dockerfile
    FROM python:3.10-slim

    WORKDIR /app

    # Install dependencies
    RUN pip install --no-cache-dir \
        runpod \
        pandas \
        numpy \
        scikit-learn \
        joblib \
        boto3

    # Copy handler
    COPY prediction_handler.py /app/

    # Start worker
    CMD ["python", "-u", "prediction_handler.py"]
    ```

- [ ] **5.3 Create RunPod requirements.txt**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/runpod/requirements.txt`
  - List dependencies:
    ```
    runpod>=1.0.0
    pandas>=2.0.0
    numpy>=1.24.0
    scikit-learn>=1.3.0
    joblib>=1.3.0
    boto3>=1.28.0
    ```

- [ ] **5.4 Create RunPodServerlessManager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_serverless_manager.py`
  - Implement manager:
    ```python
    import runpod
    from typing import Dict, Any, Optional
    from src.cloud.runpod_config import RunPodConfig
    from src.cloud.provider_interface import CloudPredictionProvider

    class RunPodServerlessManager(CloudPredictionProvider):
        """Manage RunPod serverless endpoints for predictions."""

        def __init__(self, config: RunPodConfig):
            self.config = config
            runpod.api_key = config.runpod_api_key

        def invoke_prediction(
            self,
            model_key: str,
            data_key: str,
            output_key: str,
            endpoint_id: str,
            prediction_column_name: str = 'prediction',
            feature_columns: Optional[list] = None
        ) -> Dict[str, Any]:
            """
            Invoke RunPod serverless endpoint for predictions.

            Args:
                model_key: Storage key for model (e.g., 'models/user_123/model_456')
                data_key: Storage key for input data
                output_key: Storage key for output results
                endpoint_id: RunPod endpoint ID
                prediction_column_name: Name for prediction column
                feature_columns: List of feature column names

            Returns:
                Prediction results
            """
            endpoint = runpod.Endpoint(endpoint_id)

            # Prepare input
            input_data = {
                "input": {
                    "model_key": model_key,
                    "data_key": data_key,
                    "output_key": output_key,
                    "volume_id": self.config.network_volume_id,
                    "prediction_column_name": prediction_column_name,
                    "feature_columns": feature_columns
                }
            }

            # Synchronous invocation
            result = endpoint.run_sync(input_data, timeout=300)

            return result

        def invoke_async(
            self,
            model_key: str,
            data_key: str,
            output_key: str,
            endpoint_id: str,
            prediction_column_name: str = 'prediction',
            feature_columns: Optional[list] = None
        ) -> str:
            """
            Invoke RunPod serverless endpoint asynchronously.

            Returns:
                Job ID for status checking
            """
            endpoint = runpod.Endpoint(endpoint_id)

            input_data = {
                "input": {
                    "model_key": model_key,
                    "data_key": data_key,
                    "output_key": output_key,
                    "volume_id": self.config.network_volume_id,
                    "prediction_column_name": prediction_column_name,
                    "feature_columns": feature_columns
                }
            }

            # Asynchronous invocation
            job = endpoint.run(input_data)
            return job['id']

        def check_job_status(self, endpoint_id: str, job_id: str) -> Dict[str, Any]:
            """Check status of async prediction job."""
            endpoint = runpod.Endpoint(endpoint_id)
            status = endpoint.status(job_id)
            return status

        def create_endpoint(
            self,
            name: str,
            docker_image: str,
            gpu_type: str = 'NVIDIA RTX A5000',
            active_workers: int = 0,
            max_workers: int = 3
        ) -> Dict[str, Any]:
            """
            Create a new RunPod serverless endpoint.

            Args:
                name: Endpoint name
                docker_image: Docker image URL (e.g., username/image:tag)
                gpu_type: GPU type for workers
                active_workers: Number of always-on workers (0 for flex)
                max_workers: Maximum workers for scaling

            Returns:
                Endpoint details including ID
            """
            # Use GraphQL API to create endpoint
            # (RunPod Python SDK may not have direct create_endpoint method yet)
            import requests

            query = """
            mutation CreateEndpoint($input: EndpointInput!) {
                createEndpoint(input: $input) {
                    id
                    name
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
                    "volumeId": self.config.network_volume_id
                }
            }

            response = requests.post(
                'https://api.runpod.io/graphql',
                json={'query': query, 'variables': variables},
                headers={'Authorization': f'Bearer {self.config.runpod_api_key}'}
            )

            return response.json()
    ```
  - Test: Endpoint invocation (sync and async)

- [ ] **5.5 Create deployment script for RunPod serverless**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/package_runpod.sh`
  - Create script:
    ```bash
    #!/bin/bash
    # Package and deploy RunPod serverless endpoint

    set -e

    echo "🚀 Building RunPod serverless container..."

    # Build Docker image (must be linux/amd64)
    docker build --platform linux/amd64 \
        -t ${DOCKER_USERNAME}/ml-agent-prediction:latest \
        -f runpod/Dockerfile \
        runpod/

    echo "✅ Container built successfully"

    # Push to Docker Hub
    echo "📤 Pushing to Docker Hub..."
    docker push ${DOCKER_USERNAME}/ml-agent-prediction:latest

    echo "✅ Deployment package ready"
    echo ""
    echo "Next steps:"
    echo "1. Go to console.runpod.io → Serverless"
    echo "2. Create new endpoint with image: ${DOCKER_USERNAME}/ml-agent-prediction:latest"
    echo "3. Select GPU type and worker configuration"
    echo "4. Copy endpoint ID and add to .env as RUNPOD_ENDPOINT_ID"
    ```
  - Make executable: `chmod +x scripts/cloud/package_runpod.sh`

- [ ] **5.6 Write tests for RunPod serverless**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_serverless_manager.py`
  - Test cases:
    - `test_invoke_prediction_sync`
    - `test_invoke_prediction_async`
    - `test_check_job_status`
    - `test_handler_processes_predictions` (unit test for handler logic)

---

## 6.0 Migrate Training Workflow to RunPod Pods

**Goal**: Replace EC2 Spot Instances with RunPod GPU pods for training.

**Complexity**: 🔴 High (12-16 hours)

### Sub-Tasks

- [ ] **6.1 Create RunPodPodManager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_pod_manager.py`
  - Implement pod manager:
    ```python
    import runpod
    import time
    from typing import Dict, Any, Optional
    from src.cloud.runpod_config import RunPodConfig
    from src.cloud.provider_interface import CloudTrainingProvider

    class RunPodPodManager(CloudTrainingProvider):
        """Manage RunPod GPU pods for ML training."""

        def __init__(self, config: RunPodConfig):
            self.config = config
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
                config: Training configuration
                    - gpu_type: str
                    - dataset_key: str (storage key)
                    - model_id: str
                    - user_id: int
                    - model_type: str
                    - target_column: str
                    - feature_columns: list
                    - hyperparameters: dict

            Returns:
                Pod launch details
            """
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

            # Launch pod with training image
            pod = runpod.create_pod(
                name=f"training_{config['user_id']}_{config['model_id']}",
                image_name=config.get('docker_image', 'runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel'),
                gpu_type_id=gpu_type,
                cloud_type=self.config.cloud_type,
                volume_in_gb=config.get('volume_size_gb', 50),
                volume_mount_path='/workspace',
                env=env_vars,
                # Training script runs on pod start
                docker_args=f"python /workspace/train.py"
            )

            return {
                'pod_id': pod['id'],
                'gpu_type': gpu_type,
                'launch_time': time.time(),
                'status': 'launching'
            }

        def monitor_training(self, pod_id: str) -> Dict[str, Any]:
            """
            Monitor training pod status.

            Returns:
                Pod status and runtime info
            """
            pod = runpod.get_pod(pod_id)

            return {
                'pod_id': pod_id,
                'status': pod['desiredStatus'],  # 'RUNNING', 'EXITED', etc.
                'runtime_seconds': pod.get('runtime'),
                'gpu_utilization': pod.get('gpuUtilization', 0),
                'machine_id': pod.get('machineId')
            }

        def get_pod_logs(self, pod_id: str, lines: int = 100) -> list[str]:
            """Retrieve pod logs for progress monitoring."""
            # RunPod SDK method for logs
            logs = runpod.get_pod_logs(pod_id, tail=lines)
            return logs.split('\n') if logs else []

        def terminate_pod(self, pod_id: str) -> str:
            """Terminate training pod."""
            runpod.terminate_pod(pod_id)
            return pod_id

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
            """
            if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
                return int(dataset_size_mb * 0.1)
            elif model_type in ['decision_tree', 'random_forest', 'gradient_boosting']:
                return int(dataset_size_mb * 0.5)
            else:  # Neural networks
                return int(dataset_size_mb * 2.0)
    ```

- [ ] **6.2 Create training Docker image for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/runpod/training/Dockerfile`
  - Create Dockerfile:
    ```dockerfile
    FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

    WORKDIR /workspace

    # Install ML dependencies
    RUN pip install --no-cache-dir \
        pandas \
        numpy \
        scikit-learn \
        joblib \
        boto3 \
        xgboost \
        lightgbm

    # Copy training script
    COPY train.py /workspace/

    # Training script runs on container start
    CMD ["python", "/workspace/train.py"]
    ```

- [ ] **6.3 Create training script for RunPod pods**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/runpod/training/train.py`
  - Implement training:
    ```python
    import os
    import json
    import boto3
    import pandas as pd
    from io import BytesIO
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import sys

    def download_from_storage(s3_client, volume_id: str, key: str) -> bytes:
        """Download from RunPod network volume."""
        obj = s3_client.get_object(Bucket=volume_id, Key=key)
        return obj['Body'].read()

    def upload_to_storage(s3_client, volume_id: str, key: str, data: bytes) -> None:
        """Upload to RunPod network volume."""
        s3_client.put_object(Bucket=volume_id, Key=key, Body=data)

    def main():
        # Parse environment variables
        storage_endpoint = os.getenv('STORAGE_ENDPOINT')
        volume_id = os.getenv('VOLUME_ID')
        dataset_key = os.getenv('DATASET_KEY')
        model_id = os.getenv('MODEL_ID')
        model_type = os.getenv('MODEL_TYPE')
        target_column = os.getenv('TARGET_COLUMN')
        feature_columns = os.getenv('FEATURE_COLUMNS').split(',')
        hyperparameters = json.loads(os.getenv('HYPERPARAMETERS', '{}'))

        print("🚀 Starting training on RunPod pod...")
        print(f"Model type: {model_type}")
        print(f"Dataset: {dataset_key}")

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=storage_endpoint,
            aws_access_key_id=os.getenv('STORAGE_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('STORAGE_SECRET_KEY')
        )

        # Download dataset
        print("📥 Downloading dataset...")
        data_bytes = download_from_storage(s3_client, volume_id, dataset_key)
        df = pd.read_csv(BytesIO(data_bytes))
        print(f"✅ Loaded {len(df)} rows")

        # Prepare data
        X = df[feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocessing
        print("🔧 Preprocessing data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        print(f"🎯 Training {model_type} model...")

        if model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**hyperparameters)
        # ... other model types

        model.fit(X_train_scaled, y_train)

        # Evaluate
        score = model.score(X_test_scaled, y_test)
        print(f"✅ Training complete! R² score: {score:.4f}")

        # Save model
        print("💾 Saving model to storage...")
        model_bytes = BytesIO()
        joblib.dump(model, model_bytes)
        upload_to_storage(
            s3_client,
            volume_id,
            f"models/{model_id}/model.pkl",
            model_bytes.getvalue()
        )

        # Save preprocessor
        prep_bytes = BytesIO()
        joblib.dump(scaler, prep_bytes)
        upload_to_storage(
            s3_client,
            volume_id,
            f"models/{model_id}/preprocessor.pkl",
            prep_bytes.getvalue()
        )

        # Save metadata
        metadata = {
            'model_type': model_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'hyperparameters': hyperparameters,
            'score': score,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        upload_to_storage(
            s3_client,
            volume_id,
            f"models/{model_id}/metadata.json",
            json.dumps(metadata).encode()
        )

        print("✅ Model saved successfully!")
        print(f"Model ID: {model_id}")

        # Training complete - pod will auto-terminate
        sys.exit(0)

    if __name__ == '__main__':
        main()
    ```

- [ ] **6.4 Implement pod monitoring and log streaming**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_pod_manager.py`
  - Add async log streaming:
    ```python
    import asyncio
    from typing import AsyncIterator

    async def poll_training_logs(
        self,
        pod_id: str,
        poll_interval_seconds: int = 5
    ) -> AsyncIterator[str]:
        """
        Poll pod logs for training progress (async generator).

        Yields log lines as they become available.
        """
        last_line_count = 0

        while True:
            # Get pod status
            pod = runpod.get_pod(pod_id)
            status = pod['desiredStatus']

            # Get logs
            logs = runpod.get_pod_logs(pod_id)
            if logs:
                lines = logs.split('\n')

                # Yield new lines only
                for line in lines[last_line_count:]:
                    if line.strip():
                        yield line

                last_line_count = len(lines)

            # Stop if pod completed or failed
            if status in ['EXITED', 'FAILED', 'TERMINATED']:
                break

            await asyncio.sleep(poll_interval_seconds)
    ```

- [ ] **6.5 Implement auto-termination logic**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_pod_manager.py`
  - Add termination after training:
    ```python
    def wait_for_completion_and_terminate(
        self,
        pod_id: str,
        max_wait_seconds: int = 7200  # 2 hours
    ) -> Dict[str, Any]:
        """
        Wait for pod to complete training, then terminate.

        Returns:
            Final pod status and training results
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > max_wait_seconds:
                # Timeout - force terminate
                self.terminate_pod(pod_id)
                return {
                    'success': False,
                    'error': 'Training timeout exceeded',
                    'runtime_seconds': elapsed
                }

            # Check pod status
            pod_status = self.monitor_training(pod_id)

            if pod_status['status'] == 'EXITED':
                # Training complete - terminate pod
                self.terminate_pod(pod_id)
                return {
                    'success': True,
                    'runtime_seconds': elapsed,
                    'pod_id': pod_id
                }
            elif pod_status['status'] == 'FAILED':
                # Training failed - terminate pod
                self.terminate_pod(pod_id)
                return {
                    'success': False,
                    'error': 'Pod failed during training',
                    'runtime_seconds': elapsed
                }

            time.sleep(10)  # Poll every 10 seconds
    ```

- [ ] **6.6 Write tests for RunPod pod manager**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_pod_manager.py`
  - Test cases:
    - `test_select_compute_type_small_dataset`
    - `test_select_compute_type_large_dataset`
    - `test_select_compute_type_neural_network`
    - `test_launch_training_creates_pod`
    - `test_monitor_training_returns_status`
    - `test_get_pod_logs`
    - `test_terminate_pod`

---

## 7.0 Update Telegram Integration for RunPod

**Goal**: Update Telegram handlers and messages for RunPod terminology.

**Complexity**: 🟢 Low (4-5 hours)

### Sub-Tasks

- [ ] **7.1 Update cloud messages for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/messages/cloud_messages.py`
  - Replace AWS terminology:
    ```python
    # Before (AWS)
    CHOOSE_CLOUD_LOCAL_MESSAGE = """
    ☁️ **Cloud Training** (Paid - AWS)
    Instance: EC2 Spot
    """

    # After (RunPod)
    CHOOSE_CLOUD_LOCAL_MESSAGE = """
    ☁️ **Cloud Training** (Paid - RunPod GPU)
    Compute: GPU Pod
    """

    # Update instance type messages
    def cloud_instance_confirmation_message(
        gpu_type: str,  # Changed from instance_type
        estimated_cost_usd: float,
        estimated_time_minutes: int,
        dataset_size_mb: float
    ) -> str:
        return f"""
    🎯 **Training Configuration**

    GPU: {gpu_type}
    Dataset: {dataset_size_mb:.1f} MB
    Est. Time: {estimated_time_minutes} minutes
    Est. Cost: ${estimated_cost_usd:.2f}

    Proceed with cloud training? (yes/no)
    """
    ```

- [ ] **7.2 Update cloud training handlers for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_training_handlers.py`
  - Replace EC2Manager with RunPodPodManager:
    ```python
    from src.cloud.runpod_pod_manager import RunPodPodManager
    from src.cloud.provider_factory import CloudProviderFactory

    class CloudTrainingHandlers:
        def __init__(self, config):
            # Use factory to get correct provider
            provider_type = config.get('cloud_provider', 'aws')
            if provider_type == 'runpod':
                from src.cloud.runpod_config import RunPodConfig
                runpod_config = RunPodConfig.from_yaml('config/config.yaml')
                self.training_manager = RunPodPodManager(runpod_config)
            else:
                # AWS fallback
                from src.cloud.ec2_manager import EC2Manager
                self.training_manager = EC2Manager(config)

        async def handle_instance_confirmation(self, update, context):
            # Get GPU type (not instance type)
            gpu_type = session.data.get('gpu_type')

            # Launch training with RunPod
            result = self.training_manager.launch_training({
                'gpu_type': gpu_type,
                'dataset_key': session.data['dataset_key'],
                # ... other config
            })

            pod_id = result['pod_id']
            # Store pod_id instead of instance_id
            session.data['pod_id'] = pod_id
    ```

- [ ] **7.3 Update cloud prediction handlers for RunPod**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers/cloud_prediction_handlers.py`
  - Replace LambdaManager with RunPodServerlessManager:
    ```python
    from src.cloud.runpod_serverless_manager import RunPodServerlessManager

    class CloudPredictionHandlers:
        def __init__(self, config):
            provider_type = config.get('cloud_provider', 'aws')
            if provider_type == 'runpod':
                from src.cloud.runpod_config import RunPodConfig
                runpod_config = RunPodConfig.from_yaml('config/config.yaml')
                self.prediction_manager = RunPodServerlessManager(runpod_config)
            else:
                from src.cloud.lambda_manager import LambdaManager
                self.prediction_manager = LambdaManager(config)

        async def launch_cloud_prediction(self, update, context):
            # Get endpoint ID from config (not function name)
            endpoint_id = self.config.runpod_endpoint_id

            # Invoke RunPod serverless
            result = self.prediction_manager.invoke_prediction(
                model_key=model_key,
                data_key=data_key,
                output_key=output_key,
                endpoint_id=endpoint_id
            )
    ```

- [ ] **7.4 Update message templates for GPU types**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/messages/cloud_messages.py`
  - Add GPU type descriptions:
    ```python
    GPU_TYPE_DESCRIPTIONS = {
        'NVIDIA RTX A5000': '24GB VRAM - $0.29/hr - Best for small datasets',
        'NVIDIA RTX A40': '48GB VRAM - $0.39/hr - Good for medium datasets',
        'NVIDIA A100 PCIe 40GB': '40GB VRAM - $0.79/hr - Best for large datasets & neural networks',
        'NVIDIA A100 PCIe 80GB': '80GB VRAM - $1.19/hr - Very large models',
    }

    def gpu_selection_message(recommended_gpu: str, dataset_size_mb: float) -> str:
        """Display GPU selection with recommendations."""
        return f"""
    🎯 **Recommended GPU**: {recommended_gpu}
    Dataset size: {dataset_size_mb:.1f} MB

    {GPU_TYPE_DESCRIPTIONS[recommended_gpu]}

    Use recommended GPU? (yes to use recommended, or type GPU name to override)
    """
    ```

---

## 8.0 Create RunPod Setup Automation

**Goal**: Create setup script for RunPod infrastructure (network volume, etc.).

**Complexity**: 🟡 Medium (3-4 hours)

### Sub-Tasks

- [ ] **8.1 Create RunPod setup script**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/scripts/cloud/setup_runpod.py`
  - Implement setup:
    ```python
    #!/usr/bin/env python3
    """
    RunPod Infrastructure Setup Script

    This script automates the setup of RunPod resources for cloud ML workflows:
    - Creates network volume for dataset and model storage
    - Configures storage access keys
    - Tests connectivity

    Usage:
        python scripts/cloud/setup_runpod.py --config config/config.yaml
    """

    import argparse
    import runpod
    import sys
    from src.cloud.runpod_config import RunPodConfig
    from src.cloud.runpod_client import RunPodClient

    def create_network_volume(
        name: str,
        size_gb: int,
        data_center_id: str = 'us-west'
    ) -> str:
        """Create RunPod network volume."""
        print(f"Creating network volume: {name} ({size_gb}GB)...")

        # Use GraphQL API to create volume
        import requests

        query = """
        mutation CreateNetworkVolume($input: NetworkVolumeInput!) {
            createNetworkVolume(input: $input) {
                id
                name
                size
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "size": size_gb,
                "dataCenterId": data_center_id
            }
        }

        response = requests.post(
            'https://api.runpod.io/graphql',
            json={'query': query, 'variables': variables},
            headers={'Authorization': f'Bearer {runpod.api_key}'}
        )

        result = response.json()
        if 'errors' in result:
            print(f"❌ Failed to create volume: {result['errors']}")
            return None

        volume_id = result['data']['createNetworkVolume']['id']
        print(f"✅ Volume created: {volume_id}")
        return volume_id

    def test_connectivity(config: RunPodConfig) -> None:
        """Test RunPod API and storage connectivity."""
        print("\n🔍 Testing connectivity...")

        client = RunPodClient(config)
        health = client.health_check()

        if health['api']:
            print("✅ RunPod API: Connected")
        else:
            print("❌ RunPod API: Failed")
            print(f"   Error: {health.get('error')}")

        if health['storage']:
            print("✅ Storage endpoint: Accessible")
        else:
            print("❌ Storage endpoint: Failed")

    def main():
        parser = argparse.ArgumentParser(description="Setup RunPod infrastructure")
        parser.add_argument('--config', required=True, help="Path to config.yaml")
        parser.add_argument('--create-volume', action='store_true', help="Create new network volume")
        parser.add_argument('--volume-size', type=int, default=100, help="Volume size in GB")
        args = parser.parse_args()

        # Load configuration
        config = RunPodConfig.from_yaml(args.config)
        runpod.api_key = config.runpod_api_key

        print("="*60)
        print("RunPod Infrastructure Setup")
        print("="*60)

        # Create network volume if requested
        if args.create_volume:
            volume_id = create_network_volume(
                name="ml-agent-storage",
                size_gb=args.volume_size
            )
            if volume_id:
                print(f"\n⚠️  Update your .env file:")
                print(f"RUNPOD_NETWORK_VOLUME_ID={volume_id}")

        # Test connectivity
        test_connectivity(config)

        print("\n" + "="*60)
        print("RunPod Infrastructure Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update .env with RUNPOD_NETWORK_VOLUME_ID if created new volume")
        print("2. Build and push serverless prediction container:")
        print("   ./scripts/cloud/package_runpod.sh")
        print("3. Create serverless endpoint at console.runpod.io")
        print("4. Update .env with RUNPOD_ENDPOINT_ID")

    if __name__ == '__main__':
        main()
    ```
  - Make executable: `chmod +x scripts/cloud/setup_runpod.py`

- [ ] **8.2 Add RunPod to requirements.txt**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/requirements.txt`
  - Add RunPod SDK:
    ```
    runpod>=1.0.0
    ```

- [ ] **8.3 Create RunPod testing guide**
  - File: `/Users/gkratka/Documents/statistical-modeling-agent/docs/runpod-testing-guide.md`
  - Document testing steps (similar to AWS testing guide from Task 0001)

---

## Testing Notes

### Integration Testing Strategy

**Test Order**:
1. Unit tests for each RunPod component
2. Integration tests with RunPod API (requires account)
3. End-to-end Telegram workflow tests

**Test Environment Setup**:
```bash
# Set up RunPod credentials
export RUNPOD_API_KEY="your_key"
export RUNPOD_NETWORK_VOLUME_ID="your_volume_id"
export RUNPOD_STORAGE_ACCESS_KEY="your_storage_key"
export RUNPOD_STORAGE_SECRET_KEY="your_storage_secret"

# Or configure via .env
cp .env.example .env
# Edit .env with real RunPod credentials
```

**Critical Test Scenarios**:
1. **Storage**: Upload dataset → List → Download → Delete
2. **Training**: Launch pod → Monitor logs → Wait for completion → Terminate
3. **Prediction**: Deploy serverless → Invoke → Check results
4. **Cost Tracking**: Verify cost calculations match RunPod billing
5. **Provider Switching**: Test switching between AWS and RunPod

### Manual Testing Checklist

Before production deployment:
- [ ] RunPod API key valid
- [ ] Network volume created and accessible
- [ ] Storage upload/download works
- [ ] GPU pods launch successfully
- [ ] Training completes end-to-end
- [ ] Serverless endpoint deploys
- [ ] Predictions work correctly
- [ ] Cost tracking accurate (±5% of RunPod billing)
- [ ] Provider factory switches correctly
- [ ] Telegram workflows work for both providers

---

## Implementation Order Recommendation

**Week 1: Foundation & Storage**
- Complete tasks 1.0, 2.0, 4.0
- Get storage working with RunPod
- Test provider abstraction

**Week 2: Prediction Service**
- Complete tasks 3.0, 5.0
- Deploy RunPod serverless
- Test end-to-end predictions

**Week 3: Training Service**
- Complete task 6.0
- Build training Docker image
- Test end-to-end training

**Week 4: Integration & Polish**
- Complete tasks 7.0, 8.0
- End-to-end testing
- Cost validation
- Documentation

---

## Success Criteria

Migration is complete when:
1. ✅ All unit tests pass (target >90% coverage)
2. ✅ Integration tests pass with real RunPod services
3. ✅ Manual Telegram workflow test completes for RunPod
4. ✅ Can switch between AWS and RunPod via config
5. ✅ Cost tracking matches RunPod billing (within 5%)
6. ✅ All original AWS functionality preserved
7. ✅ Documentation updated with RunPod instructions

---

**Total Estimated Effort**: 46-62 hours (~1.5-2 weeks)
