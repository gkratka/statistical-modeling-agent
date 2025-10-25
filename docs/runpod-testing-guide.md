# RunPod Cloud Integration Testing Guide

**Author:** Statistical Modeling Agent
**Created:** 2025-10-24 (Task 8.3: RunPod Testing Guide)
**Status:** Production Ready

---

## Overview

This guide provides comprehensive testing procedures for RunPod cloud integration, covering environment setup, infrastructure testing, ML workflow validation, and cost tracking verification.

### Testing Scope

- **Infrastructure**: Network volumes, API connectivity, storage access
- **ML Training**: GPU pod provisioning, model training, artifact storage
- **ML Prediction**: Serverless endpoints, inference execution, result retrieval
- **Cost Tracking**: Real-time cost monitoring, budget enforcement, usage reporting
- **Error Handling**: Network failures, GPU unavailability, timeout scenarios

---

## Prerequisites

### 1. RunPod Account Setup

1. Create account at [console.runpod.io](https://console.runpod.io)
2. Add credits to account (minimum $10 recommended for testing)
3. Generate API key:
   - Go to Settings → API Keys
   - Create new API key
   - Save securely (cannot be retrieved later)

### 2. Environment Configuration

**File: `.env`**
```bash
# RunPod API Configuration
RUNPOD_API_KEY=your_api_key_here

# Network Volume (created during setup or manually)
RUNPOD_NETWORK_VOLUME_ID=your_volume_id

# Storage Access (S3-compatible API)
RUNPOD_STORAGE_ACCESS_KEY=your_storage_access_key
RUNPOD_STORAGE_SECRET_KEY=your_storage_secret_key

# Serverless Endpoint (created after Docker deployment)
RUNPOD_ENDPOINT_ID=your_endpoint_id

# Optional: Override defaults
RUNPOD_DEFAULT_GPU_TYPE=NVIDIA RTX A5000
RUNPOD_CLOUD_TYPE=COMMUNITY
```

**File: `config/config.yaml`**
```yaml
cloud:
  runpod:
    # API Configuration
    runpod_api_key: ${RUNPOD_API_KEY}
    storage_endpoint: https://storage.runpod.io
    network_volume_id: ${RUNPOD_NETWORK_VOLUME_ID}

    # GPU Settings
    default_gpu_type: "NVIDIA RTX A5000"
    cloud_type: "COMMUNITY"  # or "SECURE"

    # Storage Configuration
    storage_access_key: ${RUNPOD_STORAGE_ACCESS_KEY}
    storage_secret_key: ${RUNPOD_STORAGE_SECRET_KEY}
    data_prefix: "datasets"
    models_prefix: "models"
    results_prefix: "results"

    # Cost Limits
    max_training_cost_dollars: 10.0
    max_prediction_cost_dollars: 1.0
    cost_warning_threshold: 0.8

    # Optional
    docker_registry: null
    serverless_endpoint_id: ${RUNPOD_ENDPOINT_ID}
```

### 3. Install Dependencies

```bash
# Install all requirements including runpod SDK
pip install -r requirements.txt

# Verify runpod installation
python -c "import runpod; print(f'RunPod SDK v{runpod.__version__}')"
```

---

## Phase 1: Infrastructure Setup Testing

### Test 1.1: RunPod API Connectivity

**Objective:** Verify API key is valid and RunPod API is accessible.

**Steps:**
```bash
# Run setup script (dry-run mode - no volume creation)
python scripts/cloud/setup_runpod.py --config config/config.yaml

# Expected output:
# ✅ RunPod API: Connected
# ✅ Storage endpoint: Accessible
```

**Success Criteria:**
- API connection succeeds
- No authentication errors
- Pod count reported (may be 0)

**Troubleshooting:**
```bash
# Invalid API key
Error: 401 Unauthorized
Fix: Verify RUNPOD_API_KEY in .env

# Network issues
Error: Connection timeout
Fix: Check firewall/proxy settings

# Missing dependencies
Error: ModuleNotFoundError: runpod
Fix: pip install runpod>=1.0.0
```

### Test 1.2: Network Volume Creation

**Objective:** Create persistent storage volume for datasets and models.

**Steps:**
```bash
# Create 100GB network volume
python scripts/cloud/setup_runpod.py \
  --config config/config.yaml \
  --create-volume \
  --volume-size 100

# Expected output:
# ✅ Volume created successfully!
#    Volume ID:     abc123-xyz789
#    Volume Name:   ml-agent-storage
#    Size:          100 GB
#    Data Center:   us-west
```

**Success Criteria:**
- Volume created without errors
- Volume ID returned
- Volume visible in RunPod console (Storage → Network Volumes)

**Post-Test Actions:**
```bash
# Update .env with returned volume ID
echo "RUNPOD_NETWORK_VOLUME_ID=abc123-xyz789" >> .env

# Verify update
grep RUNPOD_NETWORK_VOLUME_ID .env
```

**Cost Impact:** ~$7/month for 100GB volume

### Test 1.3: Storage Access (S3-Compatible API)

**Objective:** Verify S3-compatible storage credentials and file operations.

**Steps:**
```bash
# Test using boto3 directly
python -c "
from src.cloud.runpod_client import RunPodClient
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig.from_yaml('config/config.yaml')
client = RunPodClient(config)

# Get storage client
s3 = client.get_storage_client()

# List objects in volume
response = s3.list_objects_v2(Bucket=config.network_volume_id)
print(f'Objects in volume: {response.get(\"KeyCount\", 0)}')
"
```

**Success Criteria:**
- Storage client initializes without errors
- `list_objects_v2()` returns response (may be empty)
- No credential errors

**Troubleshooting:**
```bash
# Invalid credentials
Error: botocore.exceptions.NoCredentialsError
Fix: Set RUNPOD_STORAGE_ACCESS_KEY and RUNPOD_STORAGE_SECRET_KEY

# Wrong endpoint
Error: EndpointConnectionError
Fix: Verify storage_endpoint in config.yaml
```

### Test 1.4: Unit Tests for RunPod Components

**Objective:** Verify all RunPod modules pass unit tests.

**Steps:**
```bash
# Run all RunPod unit tests
pytest tests/unit/test_runpod_*.py -v

# Test specific modules
pytest tests/unit/test_runpod_config.py -v
pytest tests/unit/test_runpod_client.py -v
pytest tests/unit/test_runpod_storage_manager.py -v
pytest tests/unit/test_runpod_cost_tracking.py -v
```

**Success Criteria:**
- All tests pass
- No import errors
- Mock objects behave correctly

**Expected Test Count:**
- `test_runpod_config.py`: 15+ tests
- `test_runpod_client.py`: 10+ tests
- `test_runpod_storage_manager.py`: 12+ tests
- `test_runpod_cost_tracking.py`: 8+ tests

---

## Phase 2: ML Training Workflow Testing

### Test 2.1: Pod Provisioning (Manual)

**Objective:** Verify GPU pod can be provisioned with network volume.

**Steps:**
1. Go to [console.runpod.io](https://console.runpod.io) → GPU Cloud → Pods
2. Click "Deploy" → Select "PyTorch" template
3. Choose GPU type: `NVIDIA RTX A5000` (cost-effective)
4. Attach network volume created in Test 1.2
5. Deploy pod

**Success Criteria:**
- Pod provisions within 60 seconds
- Pod status changes to "Running"
- Network volume mounted at `/workspace`
- SSH/Jupyter access available

**Cost Impact:** ~$0.29/hour (stop when done testing)

### Test 2.2: Training Data Upload

**Objective:** Upload dataset to network volume via storage API.

**Steps:**
```bash
# Test with sample dataset
python -c "
import pandas as pd
from src.cloud.runpod_storage_manager import RunPodStorageManager
from src.cloud.runpod_config import RunPodConfig

# Create sample dataset
df = pd.DataFrame({
    'feature1': range(100),
    'feature2': range(100, 200),
    'target': range(200, 300)
})
df.to_csv('/tmp/test_dataset.csv', index=False)

# Upload to RunPod
config = RunPodConfig.from_yaml('config/config.yaml')
storage = RunPodStorageManager(config)

result = storage.upload_data(
    user_id='test_user',
    data_path='/tmp/test_dataset.csv',
    dataset_id='test_dataset_001'
)

print(f'Upload successful: {result.success}')
print(f'S3 key: {result.s3_key}')
"
```

**Success Criteria:**
- Upload completes without errors
- File appears in network volume (check via console or API)
- S3 key returned matches expected path

**Verification:**
```bash
# List uploaded files
aws s3 ls s3://${RUNPOD_NETWORK_VOLUME_ID}/datasets/ \
  --endpoint-url https://storage.runpod.io
```

### Test 2.3: Model Training (Integration Test)

**Objective:** Execute full training workflow using RunPod GPU pod.

**Steps:**
```bash
# Run integration test
pytest tests/integration/test_runpod_training_workflow.py -v -s

# Or test manually via Telegram bot:
# 1. Start bot: python src/bot/telegram_bot.py
# 2. Send /train command
# 3. Select "Cloud Training (RunPod)"
# 4. Upload dataset
# 5. Configure model (e.g., Random Forest)
# 6. Monitor training progress
```

**Expected Workflow:**
1. Dataset uploaded to network volume
2. GPU pod provisioned (60-90 seconds)
3. Training container starts
4. Model trains on GPU
5. Model artifacts saved to network volume
6. Pod auto-terminates after training
7. Cost tracked and reported

**Success Criteria:**
- Training completes without errors
- Model file saved to `models/` prefix in volume
- Training metrics returned (accuracy, loss, etc.)
- Pod auto-terminates to stop charges
- Cost logged correctly

**Sample Output:**
```
Training started on RunPod...
✅ Pod provisioned: pod-abc123
✅ Dataset loaded (100 rows, 3 columns)
✅ Training Random Forest (100 estimators)
⏱  Training time: 45.2 seconds
✅ Model saved: models/user_123/model_rf_20251024_143022.pkl
💰 Training cost: $0.004 (0.29/hour × 0.0126 hours)
✅ Pod terminated
```

### Test 2.4: Cost Tracking During Training

**Objective:** Verify cost tracking calculates accurate training costs.

**Steps:**
```python
# Test cost tracking
python -c "
from src.cloud.cost_tracker import CostTracker
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig.from_yaml('config/config.yaml')
tracker = CostTracker(config, provider='runpod')

# Simulate training session
tracker.start_operation('training', gpu_type='NVIDIA RTX A5000')

# Simulate 5 minutes of training
import time
time.sleep(5)  # In real test, training would run here

cost = tracker.end_operation()

print(f'Estimated cost: ${cost.estimated_cost:.4f}')
print(f'GPU hours: {cost.gpu_hours:.4f}')
print(f'Within budget: {cost.within_budget}')
"
```

**Success Criteria:**
- Cost calculated based on GPU type and duration
- Budget enforcement works (stops if exceeds limit)
- Cost warning triggered at 80% threshold

---

## Phase 3: ML Prediction Workflow Testing

### Test 3.1: Serverless Endpoint Deployment

**Objective:** Deploy prediction container to RunPod Serverless.

**Steps:**
```bash
# 1. Build prediction Docker image
./scripts/cloud/package_runpod.sh

# 2. Push to Docker Hub (or private registry)
docker push yourusername/ml-agent-runpod-prediction:latest

# 3. Create serverless endpoint via console:
#    - Go to console.runpod.io → Serverless → New Endpoint
#    - Enter Docker image: yourusername/ml-agent-runpod-prediction:latest
#    - Select GPU: NVIDIA RTX A5000
#    - Configure autoscaling: Min=0, Max=3
#    - Advanced: Enable FlashBoot for faster cold starts
#    - Create endpoint

# 4. Save endpoint ID to .env
echo "RUNPOD_ENDPOINT_ID=your_endpoint_id" >> .env
```

**Success Criteria:**
- Docker build completes without errors
- Image pushed successfully
- Endpoint created and shows "Active" status
- Endpoint ID saved to configuration

**Cost Impact:** $0.00 when idle (pay-per-request pricing)

### Test 3.2: Prediction Execution (Cold Start)

**Objective:** Test serverless prediction with cold start latency.

**Steps:**
```bash
# Test prediction via API
python -c "
from src.cloud.runpod_serverless_manager import RunPodServerlessManager
from src.cloud.runpod_config import RunPodConfig
import pandas as pd
import time

config = RunPodConfig.from_yaml('config/config.yaml')
manager = RunPodServerlessManager(config)

# Create test data
test_data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [10, 20, 30]
})

# Execute prediction
start = time.time()
result = manager.predict(
    model_id='test_model_rf',
    data=test_data,
    user_id='test_user'
)
duration = time.time() - start

print(f'Prediction result: {result.predictions}')
print(f'Cold start latency: {duration:.2f}s')
print(f'Status: {result.status}')
"
```

**Success Criteria:**
- Prediction completes successfully
- Cold start latency < 3 seconds (with FlashBoot)
- Predictions returned as expected
- No timeout errors

**Expected Cold Start Times:**
- Without FlashBoot: 3-10 seconds
- With FlashBoot: 0.5-2 seconds

### Test 3.3: Prediction Execution (Warm Instances)

**Objective:** Test prediction latency with active workers.

**Steps:**
```bash
# Run multiple predictions rapidly to keep workers warm
for i in {1..5}; do
  python -c "
from src.cloud.runpod_serverless_manager import RunPodServerlessManager
from src.cloud.runpod_config import RunPodConfig
import pandas as pd
import time

config = RunPodConfig.from_yaml('config/config.yaml')
manager = RunPodServerlessManager(config)

test_data = pd.DataFrame({'feature1': [1], 'feature2': [10]})

start = time.time()
result = manager.predict(
    model_id='test_model_rf',
    data=test_data,
    user_id='test_user'
)
duration = time.time() - start

print(f'Request $i: {duration:.3f}s')
"
done
```

**Success Criteria:**
- First request: 0.5-2 seconds (warm start)
- Subsequent requests: <0.5 seconds (active worker)
- All predictions succeed
- Consistent latency across requests

### Test 3.4: Concurrent Prediction Load Test

**Objective:** Verify autoscaling handles concurrent requests.

**Steps:**
```bash
# Test with concurrent requests
pytest tests/integration/test_runpod_prediction_load.py -v

# Or manual load test:
python -c "
import asyncio
from src.cloud.runpod_serverless_manager import RunPodServerlessManager
from src.cloud.runpod_config import RunPodConfig
import pandas as pd

async def predict_async(manager, i):
    test_data = pd.DataFrame({'feature1': [i], 'feature2': [i*10]})
    result = await manager.predict_async(
        model_id='test_model_rf',
        data=test_data,
        user_id='test_user'
    )
    return result.predictions

async def load_test():
    config = RunPodConfig.from_yaml('config/config.yaml')
    manager = RunPodServerlessManager(config)

    # Send 10 concurrent requests
    tasks = [predict_async(manager, i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    print(f'Completed {len(results)} predictions')
    print(f'Success rate: {sum(1 for r in results if r) / len(results) * 100}%')

asyncio.run(load_test())
"
```

**Success Criteria:**
- All 10 requests complete successfully
- Autoscaling provisions additional workers
- No rate limiting errors
- Average latency < 2 seconds

---

## Phase 4: Error Handling & Edge Cases

### Test 4.1: Network Failure Recovery

**Objective:** Verify graceful handling of network interruptions.

**Steps:**
```bash
# Simulate network failure
pytest tests/integration/test_runpod_network_failures.py -v

# Manual test: Disconnect network during training/prediction
```

**Expected Behavior:**
- Retry logic attempts 3 times with exponential backoff
- Clear error message returned to user
- Resources cleaned up (pods terminated)
- Partial results saved if possible

### Test 4.2: GPU Unavailability

**Objective:** Test behavior when requested GPU type is unavailable.

**Steps:**
```bash
# Test with rare GPU type (likely unavailable)
python -c "
from src.cloud.runpod_pod_manager import RunPodPodManager
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig.from_yaml('config/config.yaml')
manager = RunPodPodManager(config)

# Try to provision H100 (may be unavailable)
result = manager.provision_training_pod(
    gpu_type='NVIDIA H100 PCIe',
    dataset_id='test_dataset',
    user_id='test_user'
)

print(f'Result: {result}')
"
```

**Expected Behavior:**
- Fallback to alternative GPU type
- User notified of GPU substitution
- Training continues with available hardware
- Or: Clear error message if no GPUs available

### Test 4.3: Budget Limit Enforcement

**Objective:** Verify cost limits prevent overspending.

**Steps:**
```bash
# Set very low budget limit
python -c "
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig.from_yaml('config/config.yaml')
config.max_training_cost_dollars = 0.01  # $0.01 limit

# Try to train (should fail due to cost estimate)
from src.cloud.runpod_pod_manager import RunPodPodManager
manager = RunPodPodManager(config)

result = manager.provision_training_pod(
    gpu_type='NVIDIA A100 PCIe 80GB',  # Expensive GPU
    dataset_id='test_dataset',
    user_id='test_user'
)

print(f'Result: {result}')  # Should show budget error
"
```

**Expected Behavior:**
- Training rejected before pod provision
- Clear budget error message
- No charges incurred
- User directed to increase budget

### Test 4.4: Timeout Scenarios

**Objective:** Test timeout handling for long-running operations.

**Steps:**
```bash
# Test with artificially low timeout
pytest tests/integration/test_runpod_timeouts.py -v
```

**Expected Behavior:**
- Operations timeout after configured duration
- Partial results returned if available
- Resources cleaned up (pods terminated)
- Timeout error logged clearly

---

## Phase 5: End-to-End Telegram Bot Testing

### Test 5.1: Full Training Workflow via Bot

**Manual Test Checklist:**

1. **Start Bot**
   ```bash
   python src/bot/telegram_bot.py
   ```

2. **Initiate Training**
   - Send: `/train`
   - Select: "Cloud Training (RunPod)"
   - Expected: Provider confirmation message

3. **Upload Dataset**
   - Upload CSV file (e.g., housing_data.csv)
   - Expected: "Dataset uploaded successfully (X rows, Y columns)"

4. **Configure Model**
   - Select target column
   - Select features
   - Choose model type: "Random Forest"
   - Expected: Training confirmation

5. **Monitor Training**
   - Expected: Real-time updates
   - Expected: Cost estimates
   - Expected: Training progress

6. **Receive Results**
   - Expected: Model ID
   - Expected: Performance metrics
   - Expected: Final cost
   - Expected: "Model saved successfully"

**Success Criteria:**
- All steps complete without errors
- Clear status messages at each step
- Accurate cost reporting
- Model accessible for predictions

### Test 5.2: Full Prediction Workflow via Bot

**Manual Test Checklist:**

1. **List Models**
   - Send: `/models`
   - Expected: List of trained models with IDs

2. **Initiate Prediction**
   - Send: `/predict`
   - Select model from list
   - Expected: "Upload dataset for prediction"

3. **Upload Prediction Data**
   - Upload CSV file with same schema
   - Expected: Schema validation confirmation

4. **Execute Prediction**
   - Expected: "Prediction in progress..."
   - Expected: Cost estimate
   - Expected: Results preview

5. **Download Results**
   - Expected: Full predictions CSV download
   - Expected: Final cost reported

**Success Criteria:**
- Predictions match expected values
- Cost calculated correctly
- Results downloadable
- No schema validation errors

---

## Phase 6: Cost Analysis & Reporting

### Test 6.1: Cost Tracking Accuracy

**Objective:** Verify cost calculations match RunPod billing.

**Steps:**
```bash
# Run test training session
python -c "
from src.cloud.cost_tracker import CostTracker
from src.cloud.runpod_config import RunPodConfig
import time

config = RunPodConfig.from_yaml('config/config.yaml')
tracker = CostTracker(config, provider='runpod')

# Track 10-minute session
tracker.start_operation('training', gpu_type='NVIDIA RTX A5000')
time.sleep(600)  # 10 minutes
cost = tracker.end_operation()

print(f'Calculated cost: ${cost.estimated_cost:.4f}')
print(f'Expected cost: $0.0483 (0.29/hour × 0.1667 hours)')
print(f'Variance: {abs(cost.estimated_cost - 0.0483) / 0.0483 * 100:.2f}%')
"

# Compare with RunPod console billing
# Go to console.runpod.io → Billing → Usage
# Verify charges match calculated costs within 5%
```

**Success Criteria:**
- Calculated cost within 5% of actual charges
- GPU hours tracked accurately
- Per-second billing reflected correctly

### Test 6.2: Cost Reporting

**Objective:** Verify cost reports contain accurate information.

**Steps:**
```bash
# Generate cost report
python -c "
from src.cloud.cost_tracker import CostTracker
from src.cloud.runpod_config import RunPodConfig

config = RunPodConfig.from_yaml('config/config.yaml')
tracker = CostTracker(config, provider='runpod')

# Get report for user
report = tracker.get_cost_report(user_id='test_user')

print(f'Total cost: ${report.total_cost:.4f}')
print(f'Operations: {report.operation_count}')
print(f'Average cost: ${report.average_cost:.4f}')
print(f'GPU hours: {report.total_gpu_hours:.4f}')
"
```

**Success Criteria:**
- All operations logged
- Costs aggregated correctly
- GPU hours summed accurately
- Report includes timestamps

---

## Troubleshooting Guide

### Common Issues

#### Issue: "API key invalid"
```
Error: 401 Unauthorized
```
**Solution:**
1. Verify API key in .env: `grep RUNPOD_API_KEY .env`
2. Check key is active in RunPod console
3. Ensure no extra whitespace in key

#### Issue: "Network volume not found"
```
Error: Volume abc123 not accessible
```
**Solution:**
1. Verify volume ID: `grep RUNPOD_NETWORK_VOLUME_ID .env`
2. Check volume exists in console: Storage → Network Volumes
3. Ensure volume is in correct data center

#### Issue: "Out of credits"
```
Error: Insufficient balance to launch pod
```
**Solution:**
1. Add credits at console.runpod.io → Billing
2. Minimum 1 hour credit balance required
3. Check current balance before large operations

#### Issue: "GPU type unavailable"
```
Error: No NVIDIA RTX A5000 available in us-west
```
**Solution:**
1. Try different data center
2. Use alternative GPU type
3. Enable "Secure Cloud" for better availability
4. Wait and retry (community cloud availability varies)

#### Issue: "Slow cold starts"
```
Warning: Cold start took 8 seconds
```
**Solution:**
1. Enable FlashBoot in endpoint settings
2. Use active workers for latency-sensitive apps
3. Optimize Docker image size (<2GB recommended)
4. Pre-load models in global scope

#### Issue: "Prediction timeout"
```
Error: Request timeout after 30 seconds
```
**Solution:**
1. Increase timeout in config.yaml
2. Check model size (large models take longer to load)
3. Verify endpoint is running (console.runpod.io)
4. Test with smaller dataset first

---

## Performance Benchmarks

### Expected Latencies

| Operation | Expected Duration | Acceptable Range |
|-----------|------------------|------------------|
| API health check | 0.5s | 0.3-2s |
| Volume creation | 10s | 5-30s |
| Pod provisioning | 60s | 30-120s |
| Data upload (10MB) | 5s | 2-15s |
| Model training (1000 rows) | 2-5 min | 1-10 min |
| Cold start prediction | 2s | 0.5-5s |
| Warm prediction | 0.3s | 0.1-1s |

### Cost Estimates

| Operation | GPU Type | Duration | Cost |
|-----------|----------|----------|------|
| Training (small dataset) | RTX A5000 | 5 min | $0.024 |
| Training (medium dataset) | RTX A5000 | 30 min | $0.145 |
| Training (large dataset) | A100 80GB | 2 hours | $2.38 |
| Prediction (100 rows) | RTX A5000 | 2s | $0.0002 |
| Storage (100GB/month) | N/A | N/A | $7.00 |

---

## Test Report Template

```markdown
# RunPod Integration Test Report

**Date:** YYYY-MM-DD
**Tester:** [Name]
**Environment:** [Production/Staging]

## Test Summary
- Total Tests: XX
- Passed: XX
- Failed: XX
- Skipped: XX

## Phase Results

### Phase 1: Infrastructure (X/X passed)
- [✅] API connectivity
- [✅] Volume creation
- [⚠️] Storage access (warnings noted)
- [✅] Unit tests

### Phase 2: Training (X/X passed)
- [✅] Pod provisioning
- [✅] Data upload
- [❌] Model training (see issue #123)
- [✅] Cost tracking

### Phase 3: Prediction (X/X passed)
- [✅] Endpoint deployment
- [✅] Cold start prediction
- [✅] Warm prediction
- [✅] Load testing

### Phase 4: Error Handling (X/X passed)
- [✅] Network failures
- [✅] GPU unavailability
- [✅] Budget enforcement
- [✅] Timeout handling

### Phase 5: End-to-End (X/X passed)
- [✅] Training workflow
- [✅] Prediction workflow

### Phase 6: Cost Analysis (X/X passed)
- [✅] Cost accuracy
- [✅] Cost reporting

## Issues Found
1. [Issue description]
   - Severity: [High/Medium/Low]
   - Steps to reproduce: [...]
   - Workaround: [...]

## Recommendations
- [Recommendation 1]
- [Recommendation 2]

## Next Steps
- [ ] Fix identified issues
- [ ] Retest failed scenarios
- [ ] Update documentation
```

---

## Continuous Integration Testing

### Automated Test Suite

```bash
# Run full test suite
pytest tests/ -v --cov=src/cloud --cov-report=html

# Run only RunPod tests
pytest tests/unit/test_runpod_*.py tests/integration/test_runpod_*.py -v

# Run with cost tracking disabled (for CI/CD)
DISABLE_CLOUD_TESTS=1 pytest tests/
```

### CI/CD Pipeline Integration

**File: `.github/workflows/runpod-tests.yml`**
```yaml
name: RunPod Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-runpod:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: pytest tests/unit/test_runpod_*.py -v

      - name: Run integration tests (if secrets available)
        if: github.event_name == 'push'
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          RUNPOD_NETWORK_VOLUME_ID: ${{ secrets.RUNPOD_NETWORK_VOLUME_ID }}
        run: |
          pytest tests/integration/test_runpod_*.py -v
```

---

## Additional Resources

- **RunPod Documentation:** https://docs.runpod.io
- **RunPod API Reference:** https://docs.runpod.io/api-reference
- **RunPod Python SDK:** https://github.com/runpod/runpod-python
- **RunPod Discord:** Active community support
- **Project MRD:** `/Users/gkratka/Documents/statistical-modeling-agent/docs/runpod-mrd`

---

## Summary

This testing guide provides comprehensive coverage for RunPod cloud integration. Follow each phase sequentially for first-time setup, or focus on specific phases for ongoing validation.

**Key Testing Priorities:**
1. Infrastructure connectivity (Phase 1)
2. Training workflow (Phase 2)
3. Prediction workflow (Phase 3)
4. Cost tracking accuracy (Phase 6)

**Before Production:**
- ✅ All unit tests passing
- ✅ Integration tests verified
- ✅ Cost tracking validated
- ✅ Error handling tested
- ✅ Performance benchmarks met

**Maintenance:**
- Re-run Phase 1 monthly (infrastructure health)
- Run Phase 2-3 before each release
- Monitor cost reports weekly
- Update benchmarks quarterly
