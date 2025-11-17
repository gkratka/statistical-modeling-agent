# Cloud Prediction User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Model Selection](#model-selection)
5. [Data Sources](#data-sources)
6. [Performance Guidance](#performance-guidance)
7. [Result Formats](#result-formats)
8. [Cost Examples](#cost-examples)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Cloud Prediction?

Cloud prediction enables you to generate predictions using trained machine learning models through serverless cloud infrastructure (AWS Lambda or RunPod Serverless). This approach offers significant advantages over local predictions for production workloads.

### When to Use Cloud Prediction vs Local

**Use Cloud Prediction When:**
- Dataset size exceeds local memory constraints (>1GB)
- You need guaranteed availability and uptime
- Processing large batches (>10,000 rows)
- Sharing predictions with distributed teams
- Requiring audit trails and version control
- Cost efficiency matters for intermittent workloads

**Use Local Prediction When:**
- Dataset is small (<1,000 rows)
- Low latency is critical (<100ms)
- Working with sensitive data requiring air-gapped processing
- Prototyping and experimentation
- Network connectivity is unreliable

### Cloud Providers Supported

**AWS Lambda** (Default)
- Synchronous predictions: <30 seconds
- Memory: Up to 3GB
- Best for: Small to medium datasets (<10K rows)
- Cost: Pay-per-invocation ($0.0000166667 per GB-second)

**RunPod Serverless** (GPU-accelerated)
- Asynchronous predictions with job tracking
- GPU support for deep learning models
- Best for: Large datasets, neural networks
- Cost: Pay-per-second GPU usage

---

## Prerequisites

### 1. Trained Cloud Model

You must first train a model using cloud infrastructure:
```
/train → Cloud Training → Model saved to cloud storage
```

Verify your models exist:
```
/list_models
```

### 2. Cloud Provider Configuration

**AWS Setup:**
- S3 bucket configured for model storage
- Lambda function deployed with ML dependencies
- IAM roles with appropriate permissions

**RunPod Setup:**
- Network volume created
- Serverless endpoint deployed
- API key configured

See [RUNPOD_SERVERLESS_DEPLOYMENT_GUIDE.md](./RUNPOD_SERVERLESS_DEPLOYMENT_GUIDE.md) for detailed setup.

### 3. Data Requirements

Your prediction data must:
- Contain **all features** used during training
- Match training data schema (column names, types)
- Be in supported format (CSV, Excel, Parquet)
- Not exceed provider limits:
  - Telegram upload: 100MB
  - Local path: 1GB (configurable)
  - S3/Storage: No practical limit

---

## Quick Start

### First Prediction Walkthrough

**Step 1: Start Cloud Prediction**
```
/predict_cloud
```

**Step 2: Choose Data Source**

Bot displays three options:
- 📤 **Upload File (Telegram)** - Send CSV directly
- 📁 **Local File Path** - Provide absolute path
- ☁️ **S3 URI** - Use existing cloud dataset

**Step 3: Select Model**

Bot shows your available models with:
- Model type (random_forest, xgboost, keras, etc.)
- Training date
- Performance metrics (MSE, accuracy, R2)
- Number of features

Example:
```
1. random_forest
   ID: model_12345_random_forest_20251107
   Trained: 2025-11-07T10:00:00
   Metrics: mse=0.15, r2=0.85
   Features: 5
```

**Step 4: Feature Validation**

Bot automatically validates that your data contains required features:
- ✅ Match: Prediction proceeds
- ❌ Mismatch: Error shows missing features

**Step 5: Receive Results**

Bot sends:
1. Summary statistics (mean, min, max, std dev)
2. Inline preview (first 5 rows, if <100 total rows)
3. CSV download (all predictions)

---

## Model Selection

### Listing Your Cloud Models

Cloud models are automatically discovered from your cloud storage:

**AWS:** S3 bucket under `models/user_{user_id}/`
**RunPod:** Network volume under `models/user_{user_id}/`

Each model includes metadata:
```json
{
  "model_id": "model_12345_random_forest_20251107",
  "model_type": "random_forest",
  "task_type": "regression",
  "feature_columns": ["feature_1", "feature_2", ...],
  "target_column": "price",
  "training_date": "2025-11-07T10:00:00",
  "metrics": {
    "mse": 0.15,
    "r2": 0.85,
    "mae": 0.32
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

### Filtering Models

**By Task Type:**
- Regression models: Predict continuous values
- Binary classification: Predict 0/1, yes/no
- Multiclass classification: Predict 3+ categories

**By Performance:**
- Sort by R² (regression) or accuracy (classification)
- Review training metrics before selection

**By Date:**
- Most recent training typically has best performance
- Older models remain available for comparison

### Model Metadata Display

Each model shows:
- **ID**: Unique identifier for tracking
- **Type**: Algorithm (random_forest, xgboost, keras_binary_classification)
- **Trained**: Date and time of training
- **Metrics**: Top 2 performance metrics
- **Features**: Count of required input features

---

## Data Sources

### Option 1: Telegram Upload

**Best For:** Quick predictions, datasets <100MB

**Usage:**
1. Select "Upload File (Telegram)"
2. Upload CSV file through Telegram
3. Bot validates and confirms upload

**Limitations:**
- 100MB file size limit (Telegram restriction)
- Synchronous upload (wait for completion)

**Example:**
```
User: /predict_cloud
Bot: Choose Data Source: [Upload File] [Local Path] [S3 URI]
User: [Upload File]
Bot: Please upload your CSV file
User: [Uploads housing_data.csv]
Bot: ✅ Dataset Uploaded
     Rows: 5,000
     Columns: 10
     Loading available models...
```

### Option 2: Local File Path

**Best For:** Large datasets on server, repeated predictions

**Usage:**
1. Select "Local File Path"
2. Provide absolute path to CSV file
3. Bot validates path security and loads data

**Security:**
- Path must be in allowed directories (configured in `config.yaml`)
- 8-layer validation prevents directory traversal
- Symlinks resolved to actual paths
- Extension whitelist enforced (.csv, .xlsx, .parquet)

**Configuration:**
```yaml
local_data:
  enabled: true
  allowed_directories:
    - /home/user/datasets
    - /Users/username/Documents/data
    - ./data  # Relative to bot directory
  max_file_size_mb: 1000  # 1GB
  allowed_extensions: [.csv, .xlsx, .xls, .parquet]
```

**Example:**
```
User: /predict_cloud
Bot: Choose Data Source: [Upload File] [Local Path] [S3 URI]
User: [Local Path]
Bot: Please send the absolute path to your CSV file
User: /home/user/datasets/test_predictions.csv
Bot: ✅ Dataset Loaded
     Path: /home/user/datasets/test_predictions.csv
     Rows: 50,000
     Columns: 15
     Loading available models...
```

### Option 3: S3 URI (AWS) / Storage Path (RunPod)

**Best For:** Large datasets, distributed teams, production workflows

**Usage:**
1. Select "S3 URI"
2. Provide S3 path or RunPod storage path
3. Bot validates access and registers dataset

**AWS S3 Format:**
```
s3://bucket-name/path/to/dataset.csv
```

**RunPod Storage Format:**
```
datasets/user_12345/prediction_data.csv
```

**Validation:**
- Read permissions verified
- File existence confirmed
- Path format validated

**Example:**
```
User: /predict_cloud
Bot: Choose Data Source: [Upload File] [Local Path] [S3 URI]
User: [S3 URI]
Bot: Please send your S3 dataset URI
User: s3://ml-data-bucket/predictions/housing_test.csv
Bot: ✅ S3 Dataset Registered
     URI: s3://ml-data-bucket/predictions/housing_test.csv
     Loading available models...
```

---

## Performance Guidance

### Dataset Size Recommendations

#### Small Datasets (<1,000 rows)

**Provider:** AWS Lambda (Synchronous)

**Performance:**
- Execution time: <5 seconds
- Memory usage: <512MB
- Invocation type: RequestResponse (synchronous)

**Cost:** ~$0.001 per prediction batch

**Use Case:** Real-time predictions, interactive workflows

#### Medium Datasets (1,000 - 10,000 rows)

**Provider:** AWS Lambda (Batched)

**Performance:**
- Execution time: 5-30 seconds
- Memory usage: 512MB - 2GB
- Batch size: 5,000 rows per invocation

**Cost:** ~$0.01 per prediction batch

**Use Case:** Batch processing, scheduled jobs

#### Large Datasets (10,000 - 100,000 rows)

**Provider:** AWS Lambda (Async) or RunPod Serverless

**Performance:**
- Execution time: 30 seconds - 5 minutes
- Memory usage: 2GB - 3GB
- Job tracking: Async with status polling

**Cost:** $0.05 - $0.10 per prediction batch

**Use Case:** Bulk processing, ETL pipelines

#### Very Large Datasets (>100,000 rows)

**Provider:** RunPod Serverless (GPU-accelerated)

**Performance:**
- Execution time: 5-30 minutes
- GPU acceleration available
- Chunked processing (10,000 rows per chunk)

**Cost:** $0.10 - $0.50+ per prediction batch (GPU-dependent)

**Use Case:** Production workloads, deep learning models

### Optimization Tips

**1. Batch Size Selection**
- Small datasets (<1K): Single batch, synchronous
- Medium datasets (1K-10K): 5,000-row batches
- Large datasets (>10K): 10,000-row batches with async

**2. Feature Selection**
- Reduce feature count if possible (faster serialization)
- Remove unnecessary columns before upload
- Use feature importance from training to prune

**3. Data Format**
- Parquet: Fastest for columnar data (3-5x faster than CSV)
- CSV: Universal compatibility, slower for large files
- Excel: Avoid for >10,000 rows (slow parsing)

**4. Provider Selection**
- Lambda: Best for <10K rows, CPU-only models
- RunPod: Best for >10K rows, GPU models, neural networks

---

## Result Formats

### Summary Statistics

Always included in completion message:

**Regression Predictions:**
```
✅ Prediction Complete

Summary Statistics:
• Mean: 245.67
• Min: 123.45
• Max: 456.78
• Std Dev: 45.23

Total predictions: 5,000 rows
```

**Classification Predictions:**
```
✅ Prediction Complete

Summary Statistics:
• Class Distribution:
  - Class 0: 2,345 (46.9%)
  - Class 1: 2,655 (53.1%)

Total predictions: 5,000 rows
```

### Inline Table Preview

Included for **small datasets only** (<100 rows):

```
Preview (First 5 Rows):
┌─────────┬────────────┬────────────┐
│ Row     │ feature_1  │ prediction │
├─────────┼────────────┼────────────┤
│ 0       │ 45.2       │ 234.5      │
│ 1       │ 67.8       │ 312.1      │
│ 2       │ 23.4       │ 189.7      │
│ 3       │ 89.1       │ 401.2      │
│ 4       │ 34.5       │ 212.8      │
└─────────┴────────────┴────────────┘
```

For larger datasets (≥100 rows), download CSV for full results.

### CSV Download

Always available for all prediction sizes:

**Format:**
```csv
feature_1,feature_2,feature_3,prediction
45.2,67.8,23.4,234.5
67.8,89.1,34.5,312.1
...
```

**Features:**
- All original columns preserved
- Prediction column appended
- UTF-8 encoding
- Comma-separated format

**File Naming:**
```
predictions_{model_id}_{timestamp}.csv
```

Example:
```
predictions_model_12345_random_forest_20251107_143022.csv
```

---

## Cost Examples

### AWS Lambda Pricing

**Formula:**
```
Cost = (Memory_GB × Duration_seconds × $0.0000166667) + (Requests × $0.20 per 1M)
```

#### Small Dataset Example (500 rows)

**Configuration:**
- Memory: 1GB
- Duration: 3 seconds
- Requests: 1

**Cost Calculation:**
```
Compute: 1 GB × 3s × $0.0000166667 = $0.00005
Requests: 1 × ($0.20 / 1,000,000) = $0.0000002
Total: ~$0.00005 ($0.05 per 1,000 predictions)
```

#### Medium Dataset Example (5,000 rows)

**Configuration:**
- Memory: 2GB
- Duration: 15 seconds
- Requests: 1

**Cost Calculation:**
```
Compute: 2 GB × 15s × $0.0000166667 = $0.0005
Requests: 1 × ($0.20 / 1,000,000) = $0.0000002
Total: ~$0.0005 ($0.10 per 10,000 predictions)
```

#### Large Dataset Example (50,000 rows)

**Configuration:**
- Memory: 3GB
- Duration: 180 seconds (3 minutes)
- Requests: 5 (batched)

**Cost Calculation:**
```
Compute: 3 GB × 180s × $0.0000166667 × 5 batches = $0.045
Requests: 5 × ($0.20 / 1,000,000) = $0.000001
Total: ~$0.045 ($0.90 per million predictions)
```

### RunPod Serverless Pricing

**Formula:**
```
Cost = GPU_Type_Rate × Duration_seconds
```

**GPU Rates (approximate):**
- RTX A4000: $0.00034/second ($1.22/hour)
- RTX A5000: $0.00044/second ($1.58/hour)
- RTX 4090: $0.00054/second ($1.94/hour)

#### Medium Dataset Example (10,000 rows, CPU model)

**Configuration:**
- GPU: RTX A4000
- Duration: 45 seconds

**Cost:**
```
$0.00034/s × 45s = $0.0153 (~$1.53 per 100K predictions)
```

#### Large Dataset Example (100,000 rows, Neural Network)

**Configuration:**
- GPU: RTX A5000
- Duration: 300 seconds (5 minutes)

**Cost:**
```
$0.00044/s × 300s = $0.132 (~$1.32 per million predictions)
```

### Cost Optimization Strategies

1. **Batch Processing:** Combine multiple prediction requests
2. **Right-size Memory:** Use minimum Lambda memory for workload
3. **Choose Appropriate Provider:** Lambda for CPU, RunPod for GPU
4. **Cache Models:** Reduce cold start overhead
5. **Use Spot Instances:** RunPod spot pricing (30-50% cheaper)

---

## Troubleshooting

### Schema Mismatch Errors

**Error Message:**
```
❌ Schema Mismatch

Missing features: feature_3, feature_5

Please ensure your data contains all required features.
```

**Cause:** Prediction data missing columns that were present during training

**Solution:**
1. Check training metadata: `/list_models` → View model details
2. Compare your data columns with required features
3. Add missing columns or retrain model with current features

**Prevention:**
```python
# Save feature list during training
training_features = ['feature_1', 'feature_2', 'feature_3']

# Before prediction, validate
prediction_features = prediction_df.columns.tolist()
missing = set(training_features) - set(prediction_features)
if missing:
    print(f"Missing features: {missing}")
```

### Missing Features

**Error Message:**
```
❌ Schema Mismatch

Missing features: age, income

Please ensure your data contains all required features.
```

**Cause:** Prediction dataset doesn't have all required input features

**Solution:**
1. Add missing columns to prediction data
2. Use same column names as training data (case-sensitive)
3. Ensure correct data types (numeric vs categorical)

**Example Fix:**
```python
# Training features
required_features = ['age', 'income', 'education_years']

# Prediction data missing 'income'
prediction_df = pd.DataFrame({
    'age': [25, 30, 35],
    'education_years': [12, 16, 14]
})

# Add missing feature
prediction_df['income'] = 0  # Or use appropriate default/imputation

# Now prediction will succeed
```

### Timeout Errors

**Error Message:**
```
❌ Prediction failed

Lambda timeout after 30 seconds
```

**Cause:** Dataset too large for synchronous Lambda invocation

**Solution:**

**Option 1: Use Async Invocation** (Large datasets)
- Automatically enabled for >10,000 rows
- Job status polling every 5 seconds
- Up to 15-minute timeout

**Option 2: Switch to RunPod** (Very large datasets)
- GPU acceleration for neural networks
- Longer timeout limits (30+ minutes)
- Chunked processing for datasets >100K rows

**Option 3: Reduce Dataset Size**
- Process in smaller batches
- Sample data for prototyping
- Increase Lambda memory (faster execution)

**Configuration Adjustment:**
```yaml
# config/config.yaml
cloud:
  lambda:
    timeout_seconds: 900  # Increase to 15 minutes
    memory_mb: 3008  # Maximum Lambda memory
```

### Invalid Model ID

**Error Message:**
```
❌ Model not found. Please try again.
```

**Cause:** Selected model doesn't exist in cloud storage or was deleted

**Solution:**
1. List available models: `/list_models`
2. Verify model was successfully trained and uploaded
3. Check cloud storage directly:
   - AWS: S3 bucket `models/user_{user_id}/`
   - RunPod: Network volume `models/user_{user_id}/`

**Debugging Steps:**
```bash
# AWS: Check S3 bucket
aws s3 ls s3://your-bucket/models/user_12345/

# RunPod: Check network volume (via API)
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"query": "{ myself { networkVolumes { id name } } }"}'
```

### S3 Access Denied

**Error Message:**
```
❌ Invalid S3 URI

Cannot access: s3://bucket/data.csv

Please verify:
• URI format: s3://bucket/path
• File exists
• Read permissions granted
```

**Cause:** Insufficient S3 permissions or incorrect URI

**Solution:**

**1. Verify IAM Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/*",
        "arn:aws:s3:::your-bucket"
      ]
    }
  ]
}
```

**2. Check URI Format:**
```
✅ Correct: s3://bucket-name/path/to/file.csv
❌ Wrong: bucket-name/path/to/file.csv
❌ Wrong: https://bucket-name.s3.amazonaws.com/file.csv
```

**3. Verify File Exists:**
```bash
aws s3 ls s3://your-bucket/path/to/file.csv
```

### Row Count Mismatch

**Error Message:**
```
❌ Validation Error

Row count mismatch: Expected 1000 rows, got 850 predictions
```

**Cause:** Internal processing error or data corruption

**Solution:**
1. Retry prediction (may be transient error)
2. Verify input data integrity:
   ```python
   # Check for NaN rows
   print(f"Rows with NaN: {df.isna().any(axis=1).sum()}")

   # Remove NaN rows
   df_clean = df.dropna()
   ```
3. Check Lambda logs for detailed error
4. Contact support with model_id and error timestamp

### Empty Predictions

**Error Message:**
```
❌ Validation Error

Empty predictions returned
```

**Cause:** Model failed to generate predictions or data processing error

**Solution:**
1. Verify model is valid: Test with small known-good dataset
2. Check data types match training schema
3. Review Lambda/RunPod logs for errors
4. Retrain model if corrupted

**Debugging:**
```python
# Test model locally first
import joblib
model = joblib.load('model.pkl')

# Verify model works
test_data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
prediction = model.predict(test_data)
print(f"Local prediction: {prediction}")
```

---

## Additional Resources

- [Cloud Training Guide](./RUNPOD_SERVERLESS_DEPLOYMENT_GUIDE.md) - Train models for cloud prediction
- [ML Engine Documentation](../CLAUDE.md) - Local training and prediction
- [Configuration Reference](../config/config.yaml) - Cloud provider settings
- [API Documentation](../src/cloud/) - Cloud manager implementation details

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Author:** Statistical Modeling Agent
**Related:** Task 7.2 - Cloud Prediction User Guide
