# URL-Based Cloud Training Implementation

**Date**: 2025-11-02
**Feature**: Online File Path Cloud Training Support
**Status**: Complete

## Overview

Extended the RunPod cloud training infrastructure to support URL-based dataset loading, enabling training on publicly accessible datasets without uploading to RunPod storage.

## Changes

### 1. Training Script Generator (`src/cloud/training_script_generator.py`)

**Modifications**:
- Added optional `dataset_url` parameter to `generate_training_script()` method
- Implemented mutual exclusivity validation (exactly one of `dataset_key` or `dataset_url` must be provided)
- Updated `_prepare_substitutions()` to inject either `DATASET_URL` or `DATASET_KEY` into templates
- Maintained full backward compatibility with existing `dataset_key` approach

**Key Changes**:
```python
def generate_training_script(
    self,
    session: UserSession,
    model_id: str,
    storage_config: Dict[str, str],
    dataset_url: Optional[str] = None  # NEW PARAMETER
) -> str:
    # Validation: exactly one dataset source required
    has_dataset_key = 'dataset_key' in storage_config and storage_config['dataset_key']
    has_dataset_url = dataset_url is not None and dataset_url.strip()

    if not has_dataset_key and not has_dataset_url:
        raise ValidationError("Must provide either dataset_key or dataset_url")

    if has_dataset_key and has_dataset_url:
        raise ValidationError("Cannot provide both dataset_key and dataset_url")
```

**Template Variables**:
- `{{DATASET_URL}}`: URL to download dataset from (empty string if using storage)
- `{{DATASET_KEY}}`: Storage path to dataset (empty string if using URL)

### 2. RunPod Pod Manager (`src/cloud/runpod_pod_manager.py`)

**Modifications**:
- Updated `launch_training()` to accept `dataset_url` in config dictionary
- Implemented mutual exclusivity validation matching script generator
- Modified environment variable injection to set either `DATASET_URL` or `DATASET_KEY`
- Added `requests` library to pod startup script for URL downloading
- Maintained full backward compatibility with existing workflows

**Key Changes**:
```python
# Validate exactly one dataset source
has_dataset_key = 'dataset_key' in config and config['dataset_key']
has_dataset_url = 'dataset_url' in config and config['dataset_url']

if not has_dataset_key and not has_dataset_url:
    raise ValueError("Must provide either 'dataset_key' or 'dataset_url' in config")

if has_dataset_key and has_dataset_url:
    raise ValueError("Cannot provide both 'dataset_key' and 'dataset_url'")

# Set environment variables based on dataset source
if has_dataset_url:
    dataset_key_value = ''
    dataset_url_value = config['dataset_url']
else:
    dataset_key_value = config['dataset_key']
    dataset_url_value = ''
```

**Dependency Addition**:
```bash
# Updated startup script to include requests library
pip install --no-cache-dir pandas scikit-learn xgboost lightgbm catboost boto3 joblib requests
```

### 3. Training Templates (Already Supported)

The training templates (`src/cloud/templates/*.py`) already had URL loading support implemented:

```python
# Template logic (already existed)
if DATASET_URL:
    df = load_dataset_from_url(DATASET_URL)
else:
    df = load_dataset_from_storage(DATASET_KEY)
```

**URL Loading Function** (already implemented in templates):
- Streams dataset from URL with timeout (300s)
- Auto-detects format from URL extension (`.csv`, `.xlsx`, `.parquet`)
- Downloads to temporary file, loads into DataFrame, cleans up
- Supports CSV, Excel, and Parquet formats

## Test Coverage

### Training Script Generator Tests (`tests/unit/test_training_script_generator_url.py`)

**27 tests covering**:
- Dataset source validation (5 tests)
  - Reject both dataset_key and dataset_url
  - Reject neither dataset_key nor dataset_url
  - Accept dataset_key only
  - Accept dataset_url only
  - Reject empty/whitespace dataset_url
- Template variable injection (2 tests)
  - URL injection with empty dataset_key
  - Key injection with empty dataset_url (backward compatibility)
- URL format support (6 tests)
  - Various URL formats (HTTP, HTTPS, localhost, cloud storage)
  - Special characters and query parameters
- All model types (12 tests)
  - XGBoost (regression, binary, multiclass)
  - LightGBM (regression, binary, multiclass)
  - CatBoost (regression, binary, multiclass)
  - Keras (regression, binary, multiclass)
- Backward compatibility (2 tests)
  - Existing code without dataset_url parameter
  - Default parameter value behavior

### RunPod Pod Manager Tests (`tests/unit/test_runpod_pod_manager_url.py`)

**25 tests covering**:
- Dataset source validation (6 tests)
  - Reject both sources
  - Reject neither source
  - Reject empty dataset_key
  - Reject empty dataset_url
  - Accept dataset_key only
  - Accept dataset_url only
- Environment variable injection (3 tests)
  - Env vars with dataset_key
  - Env vars with dataset_url
  - Other env vars preserved
- URL format support (6 tests)
  - Various URL formats and protocols
  - Query parameters and special characters
- Backward compatibility (2 tests)
  - Existing dataset_key workflow
  - Required keys still validated
- Pod creation parameters (4 tests)
  - Pod name format
  - GPU type passing
  - Docker image default/custom
- Error handling (2 tests)
  - RunPod API failures
  - Missing required keys
- Integration scenarios (2 tests)
  - Complete URL-based training flow
  - Complete storage-based training flow (backward compatibility)

**Total: 52 tests, all passing**

## Usage Examples

### Option 1: URL-Based Dataset (NEW)

```python
from src.cloud.training_script_generator import TrainingScriptGenerator

generator = TrainingScriptGenerator()

# Generate script with URL
script = generator.generate_training_script(
    session=session,
    model_id='model_123',
    storage_config={
        'storage_endpoint': 'https://storage.runpod.io',
        'storage_access_key': 'key',
        'storage_secret_key': 'secret',
        'volume_id': 'vol_123'
        # Note: dataset_key NOT provided
    },
    dataset_url='https://example.com/datasets/housing_data.csv'  # NEW PARAMETER
)

# Launch pod with URL
from src.cloud.runpod_pod_manager import RunPodPodManager

manager = RunPodPodManager(config)
result = manager.launch_training({
    'gpu_type': 'NVIDIA RTX A5000',
    'dataset_url': 'https://example.com/datasets/housing_data.csv',  # NEW FIELD
    'model_id': 'model_123',
    'user_id': 12345,
    'model_type': 'xgboost_regression',
    'target_column': 'price',
    'feature_columns': ['sqft', 'bedrooms'],
    'hyperparameters': {'n_estimators': 100},
    'training_script_b64': encoded_script
})
```

### Option 2: Storage-Based Dataset (Backward Compatible)

```python
# Existing code continues to work unchanged
script = generator.generate_training_script(
    session=session,
    model_id='model_456',
    storage_config={
        'storage_endpoint': 'https://storage.runpod.io',
        'storage_access_key': 'key',
        'storage_secret_key': 'secret',
        'volume_id': 'vol_123',
        'dataset_key': 'datasets/user_123/data.csv'  # Traditional approach
    }
    # No dataset_url parameter - defaults to None
)

result = manager.launch_training({
    'gpu_type': 'NVIDIA A100 PCIe 40GB',
    'dataset_key': 'datasets/user_123/data.csv',  # Traditional field
    'model_id': 'model_456',
    'user_id': 12345,
    'model_type': 'lightgbm_binary_classification',
    'target_column': 'default',
    'feature_columns': ['income', 'debt_ratio'],
    'hyperparameters': {},
    'training_script_b64': encoded_script
})
```

## Validation Rules

**Mutual Exclusivity**:
1. Exactly one of `dataset_key` or `dataset_url` must be provided
2. Empty strings and whitespace-only values are treated as not provided
3. Both script generator and pod manager enforce the same validation

**Error Messages**:
- Both sources: "Cannot provide both dataset_key and dataset_url. Only one dataset source is allowed."
- Neither source: "Must provide either dataset_key or dataset_url. Exactly one dataset source is required."

## Backward Compatibility

**Guaranteed compatibility**:
- Existing code using `dataset_key` continues to work without changes
- The `dataset_url` parameter is optional with default value `None`
- No changes to method signatures (only added optional parameter)
- All existing tests pass without modification
- Training templates support both loading methods

**Migration path**:
- Users can switch to URL-based datasets incrementally
- No breaking changes to existing workflows
- Both approaches can coexist in the same codebase

## Security Considerations

**URL Download Security** (handled by training script templates):
- 300-second timeout prevents hanging on slow/malicious URLs
- Streaming download prevents memory exhaustion
- Temporary file cleanup prevents disk space issues
- Format validation from URL extension
- No authentication token exposure (URLs can include query params for tokens)

**No Additional Validation Needed**:
- URL validation is delegated to the training pod
- Pod has network access and can download from any URL
- Validation would require predicting pod's network environment

## Benefits

1. **No Upload Required**: Train on public datasets without uploading to RunPod storage
2. **Cost Savings**: Avoid storage costs for publicly available datasets
3. **Faster Iteration**: Skip dataset upload step in development workflow
4. **Flexibility**: Support both private (storage) and public (URL) datasets
5. **Backward Compatible**: Existing code continues to work unchanged

## Files Modified

1. `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/training_script_generator.py` (31 lines changed)
2. `/Users/gkratka/Documents/statistical-modeling-agent/src/cloud/runpod_pod_manager.py` (59 lines changed)

## Files Created

1. `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_training_script_generator_url.py` (370 lines, 27 tests)
2. `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_runpod_pod_manager_url.py` (600 lines, 25 tests)

## Next Steps

**Immediate**: Update cloud training handlers to pass `dataset_url` when online file path is selected

**Integration Points**:
- `src/bot/cloud_handlers/cloud_training_handlers.py`: Modify `launch_cloud_training()` to check for online URL in session and pass to pod manager
- State management: Add logic to detect URL vs. storage dataset source
- User messaging: Update prompts to inform users about URL option

## Testing

Run all URL-related tests:
```bash
python3 -m pytest \
  tests/unit/test_training_script_generator_url.py \
  tests/unit/test_runpod_pod_manager_url.py \
  -v
```

Expected result: 52 tests passed
