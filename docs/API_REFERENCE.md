# Cloud API Reference

**Version:** 1.0
**Last Updated:** 2025-11-08
**Target Audience:** Software Developers

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Provider Interfaces](#2-provider-interfaces)
3. [Cloud Storage API](#3-cloud-storage-api)
4. [Cloud Training API](#4-cloud-training-api)
5. [Cloud Prediction API](#5-cloud-prediction-api)
6. [Provider Factory API](#6-provider-factory-api)
7. [State Manager API](#7-state-manager-api)
8. [Error Codes](#8-error-codes)
9. [Code Examples](#9-code-examples)

---

## 1. Introduction

### Purpose

This document provides complete API reference for the Statistical Modeling Agent's cloud infrastructure. All cloud operations are abstracted through provider interfaces, enabling support for multiple cloud providers (AWS, RunPod) with consistent APIs.

### Design Philosophy

- **Provider Agnostic:** All interfaces work identically across AWS and RunPod
- **Type Safe:** Full type annotations with mypy strict mode compliance
- **Error Transparent:** Detailed error codes and messages for debugging
- **Async First:** All I/O operations are async-compatible

### Module Structure

```
src/cloud/
├── provider_interface.py     # Abstract base classes
├── provider_factory.py        # Factory for creating providers
├── aws_config.py              # AWS configuration
├── runpod_config.py           # RunPod configuration
├── s3_manager.py              # AWS S3 storage implementation
├── ec2_manager.py             # AWS EC2 training implementation
├── lambda_manager.py          # AWS Lambda prediction implementation
├── runpod_storage_manager.py  # RunPod storage implementation
├── runpod_pod_manager.py      # RunPod training implementation
└── runpod_serverless_manager.py  # RunPod prediction implementation
```

---

## 2. Provider Interfaces

### CloudStorageProvider

**File:** `src/cloud/provider_interface.py`

Abstract interface for all storage operations (S3, RunPod Network Volumes, etc.).

#### Methods

##### upload_dataset

```python
def upload_dataset(
    self,
    user_id: int,
    file_path: str,
    dataset_name: Optional[str] = None
) -> str:
    """
    Upload dataset to cloud storage.

    Args:
        user_id: User ID for isolation (used in storage path)
        file_path: Local file path to upload
        dataset_name: Optional name for dataset (defaults to filename)

    Returns:
        Storage URI string (e.g., "s3://bucket/datasets/user_123/data.csv"
        or "runpod://volume_id/datasets/user_123/data.csv")

    Raises:
        CloudStorageError: If upload fails
        FileNotFoundError: If file_path does not exist
        PermissionError: If insufficient storage permissions

    Example:
        >>> storage = factory.create_storage_provider('runpod', config)
        >>> uri = storage.upload_dataset(
        ...     user_id=12345,
        ...     file_path='/tmp/housing.csv',
        ...     dataset_name='housing_2024'
        ... )
        >>> print(uri)
        'runpod://v3zskt9gvb/datasets/user_12345/housing_2024.csv'
    """
```

##### save_model

```python
def save_model(
    self,
    user_id: int,
    model_id: str,
    model_dir: Path
) -> str:
    """
    Save model directory to cloud storage.

    Uploads all files in model_dir (model.pkl, scaler.pkl, metadata.json)
    to cloud storage under user's model directory.

    Args:
        user_id: User ID for isolation
        model_id: Unique model identifier
        model_dir: Local directory containing model files

    Returns:
        Storage URI for model directory

    Raises:
        CloudStorageError: If upload fails
        FileNotFoundError: If model_dir does not exist
        ValueError: If model_dir is empty

    Example:
        >>> model_dir = Path('./models/temp_model')
        >>> model_dir.mkdir(exist_ok=True)
        >>> # Save model files to model_dir...
        >>> uri = storage.save_model(
        ...     user_id=12345,
        ...     model_id='model_12345_rf',
        ...     model_dir=model_dir
        ... )
        >>> print(uri)
        's3://bucket/models/user_12345/model_12345_rf/'
    """
```

##### load_model

```python
def load_model(
    self,
    user_id: int,
    model_id: str,
    local_dir: Path
) -> Path:
    """
    Load model from cloud storage to local directory.

    Downloads all model files from cloud storage to local_dir.

    Args:
        user_id: User ID for isolation
        model_id: Model identifier
        local_dir: Local directory to download to (created if not exists)

    Returns:
        Path to downloaded model directory

    Raises:
        CloudStorageError: If download fails
        FileNotFoundError: If model not found in cloud storage
        PermissionError: If insufficient permissions

    Example:
        >>> local_dir = Path('./temp/model_download')
        >>> model_path = storage.load_model(
        ...     user_id=12345,
        ...     model_id='model_12345_rf',
        ...     local_dir=local_dir
        ... )
        >>> assert (model_path / 'model.pkl').exists()
        >>> assert (model_path / 'scaler.pkl').exists()
    """
```

##### list_user_datasets

```python
def list_user_datasets(self, user_id: int) -> list[Dict[str, Any]]:
    """
    List all datasets for user.

    Args:
        user_id: User ID

    Returns:
        List of dataset metadata dictionaries with keys:
        - name: Dataset name
        - uri: Storage URI
        - size: File size in bytes
        - uploaded_at: Upload timestamp
        - format: File format (csv, xlsx, parquet)

    Raises:
        CloudStorageError: If listing fails

    Example:
        >>> datasets = storage.list_user_datasets(user_id=12345)
        >>> for ds in datasets:
        ...     print(f"{ds['name']}: {ds['size']} bytes")
        housing_2024.csv: 52428 bytes
        census_data.parquet: 1048576 bytes
    """
```

##### list_user_models

```python
def list_user_models(self, user_id: int) -> list[Dict[str, Any]]:
    """
    List all models for user.

    Args:
        user_id: User ID

    Returns:
        List of model metadata dictionaries with keys:
        - model_id: Model identifier
        - uri: Storage URI
        - model_type: Model type (random_forest, xgboost, etc.)
        - task_type: Task type (regression, classification)
        - trained_at: Training timestamp
        - metrics: Training metrics dict

    Raises:
        CloudStorageError: If listing fails

    Example:
        >>> models = storage.list_user_models(user_id=12345)
        >>> for model in models:
        ...     print(f"{model['model_id']}: R²={model['metrics']['r2']}")
        model_12345_rf: R²=0.85
        model_12346_xgb: R²=0.89
    """
```

---

### CloudTrainingProvider

**File:** `src/cloud/provider_interface.py`

Abstract interface for all training operations (EC2, RunPod GPU Pods, etc.).

#### Methods

##### select_compute_type

```python
def select_compute_type(
    self,
    dataset_size_mb: float,
    model_type: str,
    estimated_training_time_minutes: int = 0
) -> str:
    """
    Select optimal compute resource for training.

    Uses dataset size, model complexity, and estimated time to recommend
    the most cost-effective compute resource.

    Args:
        dataset_size_mb: Dataset size in megabytes
        model_type: Model type (linear, random_forest, neural_network, etc.)
        estimated_training_time_minutes: Estimated training duration

    Returns:
        Compute resource identifier:
        - AWS: Instance type (e.g., 'p3.2xlarge', 'm5.xlarge')
        - RunPod: GPU type (e.g., 'NVIDIA RTX A5000', 'NVIDIA A100')

    Raises:
        ValueError: If model_type not recognized

    Example:
        >>> training = factory.create_training_provider('runpod', config)
        >>> gpu = training.select_compute_type(
        ...     dataset_size_mb=500,
        ...     model_type='random_forest',
        ...     estimated_training_time_minutes=5
        ... )
        >>> print(gpu)
        'NVIDIA RTX A5000'
    """
```

##### launch_training

```python
def launch_training(
    self,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Launch training job on cloud compute.

    Creates cloud compute instance (EC2 or GPU pod) and starts training.

    Args:
        config: Training configuration dictionary with keys:
            - compute_type: Resource identifier (from select_compute_type)
            - dataset_uri: Storage URI for dataset
            - model_id: Model identifier
            - user_id: User ID
            - model_type: ML model type
            - target_column: Target variable name
            - feature_columns: List of feature names
            - hyperparameters: Model hyperparameters dict
            - test_size: Train/test split ratio (default: 0.2)
            - preprocessing: Preprocessing config dict (optional)

    Returns:
        Job details dictionary with keys:
        - job_id: Unique job identifier (instance ID or pod ID)
        - status: Current status ('launching', 'running', 'complete', 'failed')
        - launch_time: Job launch timestamp
        - compute_type: Compute resource used
        - estimated_cost: Estimated cost in dollars

    Raises:
        CloudTrainingError: If launch fails
        BudgetExceededError: If estimated cost exceeds budget
        PodCreationTimeoutError: If resource allocation times out (5 min)

    Example:
        >>> config = {
        ...     'compute_type': 'NVIDIA RTX A5000',
        ...     'dataset_uri': 'runpod://vol/datasets/user_123/data.csv',
        ...     'model_id': 'model_123_rf',
        ...     'user_id': 123,
        ...     'model_type': 'random_forest',
        ...     'target_column': 'price',
        ...     'feature_columns': ['sqft', 'bedrooms'],
        ...     'hyperparameters': {'n_estimators': 100}
        ... }
        >>> result = training.launch_training(config)
        >>> print(result['job_id'])
        'pod-xyz123abc'
    """
```

##### monitor_training

```python
def monitor_training(self, job_id: str) -> Dict[str, Any]:
    """
    Monitor training job status.

    Args:
        job_id: Training job identifier

    Returns:
        Status dictionary with keys:
        - job_id: Job identifier
        - status: Current status
        - runtime_seconds: Elapsed time
        - progress: Progress percentage (0-100)
        - current_epoch: Current training epoch (if applicable)
        - logs: Recent log lines (last 10 lines)

    Raises:
        CloudTrainingError: If job not found

    Example:
        >>> status = training.monitor_training('pod-xyz123')
        >>> print(f"Progress: {status['progress']}%")
        Progress: 75%
    """
```

##### terminate_training

```python
def terminate_training(self, job_id: str) -> str:
    """
    Terminate training job.

    CRITICAL: Always call after training completes to stop costs.

    Args:
        job_id: Training job identifier

    Returns:
        Terminated job ID

    Raises:
        CloudTrainingError: If termination fails

    Example:
        >>> training.terminate_training('pod-xyz123')
        'pod-xyz123'
    """
```

---

### CloudPredictionProvider

**File:** `src/cloud/provider_interface.py`

Abstract interface for all prediction operations (Lambda, RunPod Serverless, etc.).

#### Methods

##### invoke_prediction

```python
def invoke_prediction(
    self,
    model_uri: str,
    data_uri: str,
    output_uri: str,
    prediction_column_name: str = 'prediction',
    feature_columns: Optional[list] = None
) -> Dict[str, Any]:
    """
    Invoke prediction service (synchronous).

    Args:
        model_uri: Storage URI for trained model
        data_uri: Storage URI for input data
        output_uri: Storage URI for output results
        prediction_column_name: Name for prediction column (default: 'prediction')
        feature_columns: List of feature names (uses all if None)

    Returns:
        Prediction results dictionary with keys:
        - success: Boolean success flag
        - output_uri: URI where predictions were saved
        - num_predictions: Number of predictions made
        - execution_time: Execution time in seconds
        - cost: Prediction cost in dollars

    Raises:
        CloudPredictionError: If prediction fails
        SchemaMismatchError: If features don't match model
        LambdaTimeoutError: If execution exceeds timeout (AWS only)

    Example:
        >>> prediction = factory.create_prediction_provider('aws', ...)
        >>> result = prediction.invoke_prediction(
        ...     model_uri='s3://bucket/models/user_123/model.pkl',
        ...     data_uri='s3://bucket/datasets/user_123/test.csv',
        ...     output_uri='s3://bucket/results/user_123/pred.csv'
        ... )
        >>> print(f"Made {result['num_predictions']} predictions")
        Made 5000 predictions
    """
```

##### invoke_async

```python
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

    Use for large datasets (>10,000 rows) to avoid timeouts.

    Args:
        model_uri: Storage URI for trained model
        data_uri: Storage URI for input data
        output_uri: Storage URI for output results
        prediction_column_name: Name for prediction column
        feature_columns: List of feature names (uses all if None)

    Returns:
        Job ID for status checking

    Raises:
        CloudPredictionError: If invocation fails

    Example:
        >>> job_id = prediction.invoke_async(
        ...     model_uri='s3://bucket/models/user_123/model.pkl',
        ...     data_uri='s3://bucket/datasets/user_123/large.csv',
        ...     output_uri='s3://bucket/results/user_123/pred.csv'
        ... )
        >>> # Poll for completion
        >>> while True:
        ...     status = prediction.check_job_status(job_id)
        ...     if status['status'] in ['complete', 'failed']:
        ...         break
        ...     time.sleep(5)
    """
```

##### check_job_status

```python
def check_job_status(self, job_id: str) -> Dict[str, Any]:
    """
    Check status of async prediction job.

    Args:
        job_id: Prediction job identifier

    Returns:
        Status dictionary with keys:
        - job_id: Job identifier
        - status: Job status ('pending', 'running', 'complete', 'failed')
        - progress: Progress percentage (0-100)
        - result: Prediction result dict (if complete)

    Raises:
        CloudPredictionError: If job not found

    Example:
        >>> status = prediction.check_job_status('job-abc123')
        >>> if status['status'] == 'complete':
        ...     print(f"Predictions: {status['result']['num_predictions']}")
    """
```

---

## 3. Cloud Storage API

### S3Manager (AWS)

**File:** `src/cloud/s3_manager.py`

AWS S3 implementation of CloudStorageProvider.

#### Additional Methods

##### download_file

```python
def download_file(self, storage_uri: str, local_path: str) -> None:
    """
    Download file from S3 to local path.

    Args:
        storage_uri: S3 URI (e.g., 's3://bucket/path/file.csv')
        local_path: Local file path to save to

    Raises:
        CloudStorageError: If download fails
        ValueError: If URI format invalid

    Example:
        >>> s3 = S3Manager(aws_client, config)
        >>> s3.download_file(
        ...     's3://ml-data/datasets/data.csv',
        ...     '/tmp/data.csv'
        ... )
    """
```

##### upload_file

```python
def upload_file(self, local_path: str, storage_uri: str) -> str:
    """
    Upload file to S3.

    Args:
        local_path: Local file path
        storage_uri: S3 URI destination

    Returns:
        S3 URI where file was uploaded

    Raises:
        CloudStorageError: If upload fails
        FileNotFoundError: If local_path doesn't exist

    Example:
        >>> uri = s3.upload_file(
        ...     '/tmp/model.pkl',
        ...     's3://ml-data/models/model.pkl'
        ... )
    """
```

---

### RunPodStorageManager (RunPod)

**File:** `src/cloud/runpod_storage_manager.py`

RunPod Network Volume implementation of CloudStorageProvider.

#### Configuration

```python
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig(
    runpod_api_key='runpod-api-xxx',
    network_volume_id='v3zskt9gvb',
    storage_access_key='xxx',
    storage_secret_key='xxx',
    data_prefix='datasets',
    models_prefix='models'
)

storage = RunPodStorageManager(config)
```

#### Additional Methods

##### upload_to_volume

```python
def upload_to_volume(
    self,
    local_path: str,
    remote_path: str
) -> str:
    """
    Upload file directly to RunPod network volume.

    Args:
        local_path: Local file path
        remote_path: Remote path on volume (relative to volume root)

    Returns:
        RunPod URI (runpod://volume_id/remote_path)

    Example:
        >>> uri = storage.upload_to_volume(
        ...     '/tmp/data.csv',
        ...     'datasets/user_123/data.csv'
        ... )
        >>> print(uri)
        'runpod://v3zskt9gvb/datasets/user_123/data.csv'
    """
```

---

## 4. Cloud Training API

### EC2Manager (AWS)

**File:** `src/cloud/ec2_manager.py`

AWS EC2 implementation of CloudTrainingProvider.

#### Configuration

```python
from src.cloud.aws_config import CloudConfig
from src.cloud.aws_client import AWSClient

config = CloudConfig(
    aws_region='us-east-1',
    aws_access_key_id='AKIAXXXXXXXX',
    aws_secret_access_key='xxxxxxxx',
    ec2_instance_type='p3.2xlarge',
    ec2_ami_id='ami-xxxxxxxx',
    ec2_key_name='ml-agent-key',
    ec2_security_group='sg-xxxxxxxx',
    ec2_spot_max_price=0.5
)

aws_client = AWSClient(config)
training = EC2Manager(aws_client, config)
```

#### Additional Methods

##### get_instance_logs

```python
def get_instance_logs(self, instance_id: str) -> str:
    """
    Get CloudWatch logs for training instance.

    Args:
        instance_id: EC2 instance ID

    Returns:
        Log content as string

    Example:
        >>> logs = training.get_instance_logs('i-abc123')
        >>> print(logs[-500:])  # Last 500 characters
    """
```

---

### RunPodPodManager (RunPod)

**File:** `src/cloud/runpod_pod_manager.py`

RunPod GPU Pod implementation of CloudTrainingProvider.

#### Configuration

```python
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig(
    runpod_api_key='runpod-api-xxx',
    network_volume_id='v3zskt9gvb',
    default_gpu_type='NVIDIA RTX A5000',
    cloud_type='COMMUNITY'
)

training = RunPodPodManager(config)
```

#### Additional Methods

##### poll_training_logs

```python
async def poll_training_logs(self, pod_id: str) -> AsyncIterator[str]:
    """
    Stream training logs asynchronously.

    Args:
        pod_id: RunPod pod ID

    Yields:
        Log lines as they become available

    Example:
        >>> async for log_line in training.poll_training_logs('pod-xyz'):
        ...     print(log_line)
        Loading dataset...
        Training epoch 1/100...
        Training epoch 2/100...
    """
```

##### get_pod_metrics

```python
def get_pod_metrics(self, pod_id: str) -> Dict[str, Any]:
    """
    Get GPU utilization and metrics for pod.

    Args:
        pod_id: RunPod pod ID

    Returns:
        Metrics dictionary with keys:
        - gpu_utilization: GPU usage percentage
        - vram_used: VRAM used in GB
        - vram_total: Total VRAM in GB
        - cpu_utilization: CPU usage percentage
        - uptime_seconds: Pod uptime

    Example:
        >>> metrics = training.get_pod_metrics('pod-xyz')
        >>> print(f"GPU: {metrics['gpu_utilization']}%")
        GPU: 85%
    """
```

---

## 5. Cloud Prediction API

### LambdaManager (AWS)

**File:** `src/cloud/lambda_manager.py`

AWS Lambda implementation of CloudPredictionProvider.

#### Configuration

```python
config = CloudConfig(
    lambda_function_name='ml-agent-prediction',
    lambda_memory_mb=3008,
    lambda_timeout_seconds=900,
    lambda_layer_arn='arn:aws:lambda:...'
)

prediction = LambdaManager(aws_client, config)
```

#### Additional Methods

##### update_function_code

```python
def update_function_code(self, zip_file_path: str) -> None:
    """
    Update Lambda function code.

    Args:
        zip_file_path: Path to ZIP file with new code

    Example:
        >>> prediction.update_function_code('lambda_function.zip')
    """
```

---

### RunPodServerlessManager (RunPod)

**File:** `src/cloud/runpod_serverless_manager.py`

RunPod Serverless implementation of CloudPredictionProvider.

#### Configuration

```python
config = RunPodConfig(
    runpod_api_key='runpod-api-xxx',
    serverless_endpoint_id='abc123xyz'
)

prediction = RunPodServerlessManager(config)
```

---

## 6. Provider Factory API

### CloudProviderFactory

**File:** `src/cloud/provider_factory.py`

Factory for creating and managing cloud provider instances.

#### Methods

##### create_storage_provider

```python
@staticmethod
def create_storage_provider(
    provider: Literal["aws", "runpod"],
    config: Union[CloudConfig, RunPodConfig],
    aws_client: Optional[AWSClient] = None
) -> CloudStorageProvider:
    """
    Create cloud storage provider instance.

    Args:
        provider: Provider name ("aws" or "runpod")
        config: CloudConfig for AWS or RunPodConfig for RunPod
        aws_client: AWSClient instance (required for AWS)

    Returns:
        CloudStorageProvider instance

    Raises:
        ValueError: If provider unsupported or missing arguments

    Example:
        >>> from src.cloud.provider_factory import CloudProviderFactory
        >>> from src.cloud.aws_config import CloudConfig
        >>> from src.cloud.aws_client import AWSClient
        >>>
        >>> config = CloudConfig.from_env()
        >>> aws_client = AWSClient(config)
        >>> storage = CloudProviderFactory.create_storage_provider(
        ...     provider="aws",
        ...     config=config,
        ...     aws_client=aws_client
        ... )
    """
```

##### detect_available_providers

```python
def detect_available_providers(self) -> List[str]:
    """
    Auto-detect available cloud providers.

    Checks environment variables for API keys and credentials.

    Returns:
        List of available provider names (e.g., ['runpod', 'aws'])

    Example:
        >>> factory = CloudProviderFactory()
        >>> providers = factory.detect_available_providers()
        >>> print(providers)
        ['runpod', 'aws']
    """
```

##### check_provider_health

```python
def check_provider_health(
    self,
    provider: str,
    timeout: int = 5,
    **credentials
) -> Dict[str, Any]:
    """
    Check provider health and connectivity.

    Args:
        provider: Provider name ('runpod', 'aws', or 'local')
        timeout: Timeout in seconds (default: 5)
        **credentials: Provider-specific credentials

    Returns:
        Health status dictionary:
        {
            'healthy': bool,
            'errors': List[str]  # Empty if healthy
        }

    Example:
        >>> health = factory.check_provider_health(
        ...     'runpod',
        ...     api_key='runpod-api-xxx'
        ... )
        >>> if health['healthy']:
        ...     print("RunPod is healthy")
    """
```

##### get_best_provider

```python
def get_best_provider(self) -> str:
    """
    Get best available provider with fallback logic.

    Priority: RunPod > AWS > Local

    Returns:
        Provider name ('runpod', 'aws', or 'local')

    Example:
        >>> provider = factory.get_best_provider()
        >>> print(f"Using {provider}")
        Using runpod
    """
```

##### create_provider_with_fallback

```python
def create_provider_with_fallback(
    self,
    provider_type: Literal["storage", "training", "prediction"],
    config: Optional[Union[CloudConfig, RunPodConfig]] = None
) -> Optional[Union[CloudStorageProvider, CloudTrainingProvider, CloudPredictionProvider]]:
    """
    Create provider with automatic fallback on failure.

    Tries providers in priority order with health checks.

    Args:
        provider_type: Type of provider ('storage', 'training', 'prediction')
        config: Optional config (auto-created if not provided)

    Returns:
        Provider instance, or None if all failed (use local)

    Example:
        >>> provider = factory.create_provider_with_fallback('training')
        >>> if provider:
        ...     result = provider.launch_training(config)
        ... else:
        ...     # Use local training
        ...     pass
    """
```

---

## 7. State Manager API

### StateManager

**File:** `src/core/state_manager.py`

Manages workflow state transitions with transaction safety.

#### Methods

##### get_state

```python
def get_state(self, user_id: int, conversation_id: str) -> Dict[str, Any]:
    """
    Get current workflow state.

    Args:
        user_id: User ID
        conversation_id: Conversation ID

    Returns:
        State dictionary with keys:
        - state: Current state name
        - workflow_type: Workflow type
        - data: State-specific data
        - history: List of previous states

    Example:
        >>> state = state_manager.get_state(123, 'conv_456')
        >>> print(state['state'])
        'CONFIRMING_INSTANCE_TYPE'
    """
```

##### transition

```python
@transaction
def transition(
    self,
    user_id: int,
    conversation_id: str,
    new_state: str,
    data: Dict[str, Any]
) -> None:
    """
    Transition to new state (transaction-safe).

    Args:
        user_id: User ID
        conversation_id: Conversation ID
        new_state: New state name
        data: State data

    Raises:
        InvalidTransitionError: If transition not allowed
        StateCorruptionError: If data validation fails

    Example:
        >>> state_manager.transition(
        ...     user_id=123,
        ...     conversation_id='conv_456',
        ...     new_state='LAUNCHING_TRAINING',
        ...     data={'job_id': 'pod-xyz', 'gpu_type': 'A5000'}
        ... )
    """
```

##### validate_transition

```python
def validate_transition(
    self,
    from_state: str,
    to_state: str
) -> bool:
    """
    Check if state transition is valid.

    Args:
        from_state: Current state
        to_state: Target state

    Returns:
        True if transition allowed

    Example:
        >>> valid = state_manager.validate_transition(
        ...     'CONFIRMING_MODEL',
        ...     'CONFIRMING_INSTANCE_TYPE'
        ... )
        >>> assert valid
    """
```

---

## 8. Error Codes

### Error Code Reference

| Code | Error Class | Description | Recovery |
|------|-------------|-------------|----------|
| **1000** | `CloudConfigurationError` | Missing or invalid configuration | Check .env and config.yaml |
| **1001** | `MissingCredentialError` | API key or credentials not found | Set environment variables |
| **1002** | `InvalidCredentialError` | API key invalid or expired | Regenerate API key |
| **2000** | `CloudStorageError` | Generic storage error | Retry operation |
| **2001** | `UploadError` | File upload failed | Check network, retry |
| **2002** | `DownloadError` | File download failed | Check permissions |
| **2003** | `BucketNotFoundError` | S3 bucket or volume not found | Verify bucket name |
| **3000** | `CloudTrainingError` | Generic training error | Check logs |
| **3001** | `PodCreationTimeoutError` | Pod allocation timeout (5 min) | Retry with different GPU |
| **3002** | `InstanceLaunchError` | EC2 launch failed | Check quotas, AMI |
| **3003** | `SpotInterruptionError` | Spot instance reclaimed | Automatic retry |
| **3004** | `BudgetExceededError` | Cost exceeded limit | Increase budget |
| **4000** | `CloudPredictionError` | Generic prediction error | Check logs |
| **4001** | `LambdaTimeoutError` | Lambda timeout (15 min max) | Use async invocation |
| **4002** | `SchemaMismatchError` | Features don't match model | Verify feature names |
| **5000** | `StateManagementError` | State transition error | Contact support |
| **5001** | `InvalidTransitionError` | Transition not allowed | Review workflow |

### Error Response Format

All errors include:

```python
{
    "error_code": 3001,
    "error_class": "PodCreationTimeoutError",
    "message": "RunPod did not allocate GPU within 5 minutes",
    "details": {
        "provider": "runpod",
        "gpu_type": "NVIDIA RTX A5000",
        "timeout_seconds": 300
    },
    "recovery_suggestion": "Retry with different GPU type or try AWS",
    "timestamp": "2025-11-08T14:30:00Z"
}
```

---

## 9. Code Examples

### Example 1: Complete Training Workflow

```python
from src.cloud.provider_factory import CloudProviderFactory
from src.cloud.runpod_config import RunPodConfig

# Initialize factory
factory = CloudProviderFactory()

# Auto-select best provider
provider_name = factory.get_best_provider()
print(f"Using provider: {provider_name}")

# Create providers
config = RunPodConfig.from_env()
storage = factory.create_storage_provider('runpod', config)
training = factory.create_training_provider('runpod', config)

# Upload dataset
dataset_uri = storage.upload_dataset(
    user_id=12345,
    file_path='/tmp/housing.csv',
    dataset_name='housing_2024'
)
print(f"Dataset uploaded: {dataset_uri}")

# Select compute type
gpu_type = training.select_compute_type(
    dataset_size_mb=50,
    model_type='random_forest',
    estimated_training_time_minutes=5
)
print(f"Selected GPU: {gpu_type}")

# Launch training
training_config = {
    'compute_type': gpu_type,
    'dataset_uri': dataset_uri,
    'model_id': 'model_12345_rf',
    'user_id': 12345,
    'model_type': 'random_forest',
    'target_column': 'price',
    'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
    'hyperparameters': {'n_estimators': 100, 'max_depth': 10}
}

result = training.launch_training(training_config)
job_id = result['job_id']
print(f"Training launched: {job_id}")

# Monitor training
import time
while True:
    status = training.monitor_training(job_id)
    print(f"Progress: {status['progress']}%")

    if status['status'] == 'complete':
        print("Training complete!")
        break
    elif status['status'] == 'failed':
        print(f"Training failed: {status['error']}")
        break

    time.sleep(10)

# CRITICAL: Terminate pod to stop costs
training.terminate_training(job_id)
print(f"Pod terminated: {job_id}")

# List models
models = storage.list_user_models(user_id=12345)
for model in models:
    print(f"Model: {model['model_id']}, R²: {model['metrics']['r2']}")
```

### Example 2: Cloud Prediction Workflow

```python
from src.cloud.provider_factory import CloudProviderFactory
from src.cloud.aws_config import CloudConfig
from src.cloud.aws_client import AWSClient

# Initialize providers
config = CloudConfig.from_env()
aws_client = AWSClient(config)
factory = CloudProviderFactory()

storage = factory.create_storage_provider('aws', config, aws_client)
prediction = factory.create_prediction_provider('aws', aws_client, config)

# Upload prediction data
data_uri = storage.upload_dataset(
    user_id=12345,
    file_path='/tmp/prediction_data.csv',
    dataset_name='predictions_batch_1'
)

# Define model and output URIs
model_uri = 's3://ml-data/models/user_12345/model_12345_rf/model.pkl'
output_uri = 's3://ml-data/results/user_12345/predictions.csv'

# Invoke prediction (async for large dataset)
job_id = prediction.invoke_async(
    model_uri=model_uri,
    data_uri=data_uri,
    output_uri=output_uri,
    prediction_column_name='predicted_price'
)

# Poll for completion
while True:
    status = prediction.check_job_status(job_id)
    print(f"Status: {status['status']}, Progress: {status['progress']}%")

    if status['status'] in ['complete', 'failed']:
        break

    time.sleep(5)

# Download results
if status['status'] == 'complete':
    storage.download_file(output_uri, '/tmp/predictions.csv')
    print(f"Predictions saved to /tmp/predictions.csv")
    print(f"Made {status['result']['num_predictions']} predictions")
```

### Example 3: Provider Fallback

```python
from src.cloud.provider_factory import CloudProviderFactory

factory = CloudProviderFactory()

# Try to create training provider with fallback
training = factory.create_provider_with_fallback('training')

if training:
    # Cloud provider available
    print("Using cloud training")
    result = training.launch_training(config)
else:
    # All cloud providers failed, use local
    print("Using local training")
    from src.engines.ml_engine import MLEngine
    engine = MLEngine()
    result = engine.train_model(...)
```

### Example 4: Health Checks Before Operation

```python
from src.cloud.provider_factory import CloudProviderFactory

factory = CloudProviderFactory()

# Check RunPod health
health = factory.check_provider_health(
    'runpod',
    api_key=os.getenv('RUNPOD_API_KEY')
)

if health['healthy']:
    print("RunPod is healthy, proceeding with training")
    # Create provider and train...
else:
    print(f"RunPod unhealthy: {health['errors']}")
    print("Falling back to AWS")
    # Try AWS instead...
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Related:** CLOUD_ARCHITECTURE.md, DEPLOYMENT_GUIDE.md, CLOUD_TRAINING_GUIDE.md
