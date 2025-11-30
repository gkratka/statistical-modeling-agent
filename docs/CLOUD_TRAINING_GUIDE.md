# Cloud Training User Guide

**Last Updated:** 2025-11-08
**Version:** 1.0
**Target Audience:** Statistical Modeling Agent users seeking GPU-accelerated model training

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Getting Started](#3-getting-started)
4. [Data Source Options](#4-data-source-options)
5. [Model Selection](#5-model-selection)
6. [Workflow States](#6-workflow-states)
7. [Cost Management](#7-cost-management)
8. [Troubleshooting](#8-troubleshooting)
9. [Advanced Topics](#9-advanced-topics)

---

## 1. Introduction

### What is Cloud Training?

Cloud training enables you to train machine learning models on high-performance GPU infrastructure instead of local hardware. This provides:

- **Scalability**: Access to powerful GPUs (24-80GB VRAM)
- **Speed**: Significantly faster training times for large datasets
- **Cost Efficiency**: Pay-per-second billing with no idle costs
- **No Hardware Limits**: Train models regardless of local machine capabilities

### Why Use Cloud Training?

**Choose Cloud Training When:**
- Dataset size exceeds 1GB
- Training neural networks (Keras MLP models)
- Processing image/text data requiring GPU acceleration
- Local training exceeds 10 minutes
- Experimenting with large gradient boosting models (XGBoost, CatBoost, LightGBM)

**Use Local Training When:**
- Dataset size under 500MB
- Simple linear/logistic regression models
- Budget constraints (local is free)
- Quick prototyping and experimentation

### Benefits

| Feature | Local Training | Cloud Training (RunPod) |
|---------|---------------|------------------------|
| **Cost** | Free | $0.29-$1.19/hour (billed per second) |
| **Speed** | CPU-limited | GPU-accelerated (5-50x faster) |
| **Dataset Size** | Up to 1GB | Up to 100GB+ |
| **VRAM** | N/A | 24-80GB |
| **Model Types** | All 13 models | All 13 models (optimized for GPU) |
| **Training Time** | Minutes to hours | Seconds to minutes |

---

## 2. Prerequisites

### Required Credentials

#### RunPod Setup (Recommended)

1. **Create RunPod Account**
   - Visit: https://www.runpod.io
   - Sign up with email or GitHub
   - Verify email address

2. **Generate API Key**
   - Navigate to Settings → API Keys
   - Click "Create API Key"
   - Copy key (starts with `runpod-api-...`)
   - **Important**: Save this key securely - it won't be shown again

3. **Add Credits**
   - Minimum: $5 USD
   - Recommended: $20-50 for experimentation
   - Go to Billing → Add Credits
   - Accepts credit cards, crypto, or bank transfer

4. **Create Network Volume** (Optional but Recommended)
   - Navigate to Storage → Network Volumes
   - Click "Create Network Volume"
   - Region: `us-east-1` (recommended for speed)
   - Size: 50GB minimum, 100GB recommended
   - Note the Volume ID (format: `v3zskt9gvb`)

#### AWS Setup (Alternative)

AWS support is available but RunPod is recommended for simplicity and cost.

**Requirements:**
- AWS Account with billing enabled
- IAM user with permissions: EC2, S3, Lambda, CloudWatch
- AWS Access Key ID and Secret Access Key
- S3 bucket for data/model storage

See [AWS Configuration Guide](AWS_SETUP.md) for detailed setup.

### Configuration

Add credentials to your `.env` file:

```bash
# RunPod Configuration
RUNPOD_API_KEY=runpod-api-xxxxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_NETWORK_VOLUME_ID=v3zskt9gvb  # Optional: your network volume ID
RUNPOD_STORAGE_ACCESS_KEY=xxxxxx     # Optional: S3-compatible storage
RUNPOD_STORAGE_SECRET_KEY=xxxxxx     # Optional: S3-compatible storage

# AWS Configuration (Alternative)
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION=us-east-1
```

**Security Best Practices:**
- Never commit `.env` file to version control
- Rotate API keys every 90 days
- Use separate keys for development/production
- Enable 2FA on RunPod/AWS accounts

---

## 3. Getting Started

### Your First Cloud Training Job

This walkthrough demonstrates training a binary classification model on the cloud using the Iris dataset.

#### Step 1: Prepare Dataset

**Option A: Upload via Telegram**
```
1. Start training: /train
2. Bot: "🌩️ Training Location - Choose: Local or Cloud"
3. Reply: "Cloud"
4. Bot: "☁️ Cloud Dataset Input - Upload CSV or provide path"
5. Upload: iris.csv (150 rows, 5 columns, ~4KB)
```

**Option B: Use Local File Path**
```
1. Start training: /train
2. Reply: "Cloud"
3. Bot: "Provide dataset path"
4. Reply: "/Users/yourname/datasets/iris.csv"
5. Bot validates and loads dataset
```

#### Step 2: Review Auto-Detected Schema

Bot analyzes dataset and suggests configuration:

```
✅ Dataset Loaded Successfully

📊 Dataset Info:
  - Rows: 150
  - Columns: 5
  - Size: 4.2 KB

🎯 Detected Target: species (categorical, 3 classes)
📐 Suggested Features: sepal_length, sepal_width, petal_length, petal_width
🤖 Recommended Task: Binary Classification

Accept this configuration? (yes/no/edit)
```

Reply: **yes**

#### Step 3: Choose Model Type

Bot presents model categories:

```
📚 Model Categories

1️⃣ Regression (5 models)
   - Predict continuous values (price, temperature, etc.)

2️⃣ Classification (6 models)
   - Predict categories (spam/not spam, species, etc.)

3️⃣ Neural Networks (2 models)
   - Deep learning for complex patterns

Which category matches your task?
```

Reply: **2** (Classification)

#### Step 4: Select Specific Model

```
🎯 Classification Models

1. Logistic Regression - Fast, interpretable baseline
2. Decision Tree - Visual rules, handles non-linear patterns
3. Random Forest - Ensemble of trees, robust
4. Gradient Boosting - State-of-art accuracy, slower
5. SVM (Support Vector Machine) - Kernel-based, good for small datasets
6. Naive Bayes - Probabilistic, fast

Choose model (1-6):
```

Reply: **3** (Random Forest)

#### Step 5: GPU Selection and Cost Estimate

```
🎯 Recommended GPU: NVIDIA RTX A5000
📊 Dataset Size: 4.2 KB

💡 24GB VRAM - $0.29/hr - Best for small datasets

Available GPU Types:
  • NVIDIA RTX A5000: 24GB VRAM - $0.29/hr - Best for small datasets
  • NVIDIA RTX A40: 48GB VRAM - $0.39/hr - Good for medium datasets
  • NVIDIA A100 PCIe 40GB: 40GB VRAM - $0.79/hr - Best for large datasets
  • NVIDIA A100 PCIe 80GB: 80GB VRAM - $1.19/hr - Very large models

☁️ Cloud Training Configuration

📊 Dataset: 4.2 KB
🎯 GPU Type: NVIDIA RTX A5000
⏱️ Estimated Time: ~2 minutes
💰 Estimated Cost: $0.01

Ready to launch cloud training?
```

Reply: **yes**

#### Step 6: Monitor Training Progress

Real-time log streaming:

```
🚀 Cloud Training Launched

Pod ID: pod-xyz123abc
GPU Type: NVIDIA RTX A5000
Status: Launching...

⏳ Waiting for pod to start...

📝 Loading dataset...
📝 Preprocessing features (StandardScaler)...
📝 Training Random Forest (n_estimators=100)...
📝 Epoch 1/100 - Accuracy: 0.8333
📝 Epoch 25/100 - Accuracy: 0.9333
📝 Epoch 50/100 - Accuracy: 0.9667
📝 Epoch 75/100 - Accuracy: 0.9667
📝 Epoch 100/100 - Accuracy: 0.9667
📝 Evaluating model...
📝 Saving to RunPod storage...
```

#### Step 7: Review Results

```
✅ Cloud Training Complete!

🎯 Model ID: model_7715560927_random_forest_binary_classification_20251108_143022
📦 Storage: runpod://v3zskt9gvb/models/user_7715560927/...
⏱️ Training Time: 1.8 minutes
💰 Actual Cost: $0.009

📊 Metrics:
  • Accuracy: 96.67%
  • Precision: 96.55%
  • Recall: 96.67%
  • F1 Score: 96.58%

The model has been saved and can be used for predictions.

Use /predict to run predictions with this model.
```

**Success!** You've trained your first cloud model for less than $0.01.

---

## 4. Data Source Options

Cloud training supports four data input methods, each optimized for different scenarios.

### 4.1 Telegram File Upload

**Best For:** Small to medium datasets (up to 100MB)

**Workflow:**
1. Start `/train` → Select "Cloud"
2. Bot prompts: "Upload CSV file or provide path"
3. Drag-and-drop file in Telegram chat
4. Bot uploads to RunPod storage automatically
5. Training begins

**Advantages:**
- Simple, no file path management
- Automatic upload and validation
- Works from any device

**Limitations:**
- Telegram file size limit: 100MB
- Slower for large files (network dependent)

**Supported Formats:** CSV, Excel (.xlsx, .xls), Parquet

---

### 4.2 Local File Path

**Best For:** Large datasets stored on server (100MB - 10GB)

**Workflow:**
1. Start `/train` → Select "Cloud"
2. Bot prompts: "Provide dataset path"
3. Enter absolute path: `/home/user/datasets/large_data.csv`
4. Bot validates path against whitelist
5. Auto-upload to RunPod storage
6. Training begins

**Advantages:**
- No Telegram upload bottleneck
- Direct access to server datasets
- Supports very large files (10GB+)

**Limitations:**
- Requires file path configuration
- Server-side storage needed

**Security:** Paths must be in allowed directories (configured in `config.yaml`):

```yaml
local_data:
  allowed_directories:
    - /home/user/datasets
    - /data/ml_projects
    - ./data
```

**Example:**
```
User: /train
Bot: 🌩️ Training Location - Choose: Local or Cloud
User: Cloud
Bot: ☁️ Cloud Dataset Input - Upload CSV or provide path
User: /home/user/datasets/housing_data.csv
Bot: ✅ Dataset Loaded Successfully (250 MB, 500,000 rows)
```

---

### 4.3 RunPod Storage URI

**Best For:** Pre-uploaded datasets on RunPod Network Volumes

**Workflow:**
1. Pre-upload dataset to RunPod Network Volume using RunPodCTL or Web UI
2. Start `/train` → Select "Cloud"
3. Provide RunPod URI: `runpod://v3zskt9gvb/datasets/user_123/data.csv`
4. Bot accesses data directly (no upload needed)
5. Training begins

**Advantages:**
- Fastest startup (data already on GPU infrastructure)
- Reuse datasets across multiple training jobs
- No redundant uploads

**URI Format:**
```
runpod://<volume_id>/<path>

Examples:
runpod://v3zskt9gvb/datasets/housing.csv
runpod://v3zskt9gvb/users/john/census_data.parquet
```

**Pre-Upload Methods:**

**Method 1: RunPodCTL (Command Line)**
```bash
# Install RunPodCTL
pip install runpodctl

# Configure
runpodctl config

# Upload dataset
runpodctl send <volume_id> /local/path/data.csv /remote/datasets/data.csv
```

**Method 2: Web UI**
1. Navigate to https://www.runpod.io/console/storage
2. Select your Network Volume
3. Click "Upload File"
4. Choose file and destination path

---

### 4.4 AWS S3 URI (AWS Provider Only)

**Best For:** Datasets already stored in AWS S3

**Workflow:**
1. Ensure S3 bucket permissions configured
2. Start `/train` → Select "Cloud"
3. Provide S3 URI: `s3://my-bucket/datasets/data.csv`
4. Bot downloads to training instance
5. Training begins

**URI Format:**
```
s3://<bucket>/<key>

Examples:
s3://ml-datasets/housing/train.csv
s3://company-data/2024/customer_churn.parquet
```

**Required Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket/*",
        "arn:aws:s3:::my-bucket"
      ]
    }
  ]
}
```

---

### Data Source Comparison

| Data Source | Max Size | Upload Speed | Best Use Case | Cost |
|-------------|----------|--------------|---------------|------|
| **Telegram Upload** | 100MB | Slow (network) | Quick experiments, small datasets | Free upload |
| **Local Path** | 10GB+ | Fast (local disk) | Large server datasets | Free upload |
| **RunPod URI** | Unlimited | Instant (no upload) | Reusable datasets, fastest training | Storage: $0.07/GB/month |
| **S3 URI** | Unlimited | Medium (S3 download) | Existing AWS infrastructure | S3 storage + transfer costs |

---

## 5. Model Selection

The bot supports 13 ML models across three categories. Each model has specific GPU requirements and cost implications.

### 5.1 Regression Models (5 Models)

**Use Case:** Predict continuous numerical values (prices, temperatures, stock values, etc.)

#### 1. Linear Regression

**Description:** Simple linear relationship between features and target

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000 (24GB VRAM)
- Minimal GPU usage (primarily CPU-based)

**Cost Estimate (per 1M rows):**
- Training Time: 1-2 minutes
- GPU Cost: $0.01 - $0.02
- Best for: Baseline models, interpretability

**When to Use:**
- Linear relationships expected
- Need model interpretability
- Quick baseline performance
- Small to medium datasets (<1M rows)

**Hyperparameters:**
- None required (uses default OLS solver)

**Example Output:**
```
📊 Metrics:
  • R²: 0.7234
  • RMSE: 12,450.22
  • MAE: 8,332.15
```

---

#### 2. Ridge Regression (L2 Regularization)

**Description:** Linear regression with L2 penalty to prevent overfitting

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000
- CPU-based, minimal GPU usage

**Cost Estimate (per 1M rows):**
- Training Time: 1-3 minutes
- GPU Cost: $0.01 - $0.02

**When to Use:**
- High multicollinearity in features
- Need to prevent overfitting
- Many correlated features

**Hyperparameters:**
- `alpha`: Regularization strength (default: 1.0)
  - Higher = more regularization
  - Range: 0.01 to 100

**Example:**
```
User: Configure alpha
Bot: Enter alpha value (0.01-100, default 1.0):
User: 10.0
Bot: ✅ Ridge configured with alpha=10.0
```

---

#### 3. Lasso Regression (L1 Regularization)

**Description:** Linear regression with L1 penalty for feature selection

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000
- CPU-based optimization

**Cost Estimate (per 1M rows):**
- Training Time: 2-4 minutes
- GPU Cost: $0.02 - $0.03

**When to Use:**
- Need automatic feature selection
- Many irrelevant features
- Sparse model desired

**Hyperparameters:**
- `alpha`: Regularization strength (default: 1.0)

**Feature Selection:**
- Bot reports which features had coefficients zeroed out
- Useful for identifying important predictors

---

#### 4. ElasticNet (L1 + L2 Regularization)

**Description:** Combines Ridge and Lasso penalties

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000

**Cost Estimate (per 1M rows):**
- Training Time: 2-5 minutes
- GPU Cost: $0.02 - $0.04

**When to Use:**
- Best of both Ridge and Lasso
- Grouped feature selection
- Correlated features with sparsity needs

**Hyperparameters:**
- `alpha`: Overall regularization (default: 1.0)
- `l1_ratio`: L1 vs L2 mix (default: 0.5)
  - 0.0 = pure Ridge
  - 1.0 = pure Lasso
  - 0.5 = equal mix

---

#### 5. Polynomial Regression

**Description:** Linear regression with polynomial feature expansion

**GPU Requirements:**
- Recommended: NVIDIA RTX A40 (48GB VRAM)
- Higher memory for polynomial features

**Cost Estimate (per 1M rows):**
- Training Time: 5-10 minutes
- GPU Cost: $0.03 - $0.07

**When to Use:**
- Non-linear relationships
- Curved patterns in data
- Need interpretable non-linear model

**Hyperparameters:**
- `degree`: Polynomial degree (default: 2)
  - Degree 2: quadratic (x²)
  - Degree 3: cubic (x³)
  - **Warning:** Degree >3 risks overfitting

**Memory Considerations:**
- Polynomial features grow exponentially
- Degree 2 + 10 features = 55 features
- Degree 3 + 10 features = 220 features

---

### 5.2 Classification Models (6 Models)

**Use Case:** Predict categorical outcomes (spam/not spam, species, customer churn, etc.)

#### 1. Logistic Regression

**Description:** Linear model for binary/multiclass classification

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000

**Cost Estimate (per 1M rows):**
- Training Time: 1-2 minutes
- GPU Cost: $0.01 - $0.02

**When to Use:**
- Binary or multiclass classification
- Need probability predictions
- Baseline model
- Interpretable coefficients

**Hyperparameters:**
- `C`: Inverse regularization (default: 1.0)
  - Smaller = stronger regularization
- `max_iter`: Solver iterations (default: 100)

**Example Output:**
```
📊 Metrics:
  • Accuracy: 87.32%
  • Precision: 85.67%
  • Recall: 88.45%
  • F1 Score: 87.04%
  • AUC-ROC: 0.9234
```

---

#### 2. Decision Tree Classifier

**Description:** Tree-based model with if-then-else rules

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000

**Cost Estimate (per 1M rows):**
- Training Time: 3-5 minutes
- GPU Cost: $0.02 - $0.04

**When to Use:**
- Need visualizable decision rules
- Non-linear patterns
- Mixed numerical/categorical features
- No scaling required

**Hyperparameters:**
- `max_depth`: Tree depth limit (default: None)
  - Deeper = more complex, risk overfitting
  - Recommended: 5-15
- `min_samples_split`: Minimum samples to split (default: 2)
- `min_samples_leaf`: Minimum samples per leaf (default: 1)

**Interpretability:**
- Bot provides feature importance ranking
- Exportable to visual tree diagram

---

#### 3. Random Forest Classifier

**Description:** Ensemble of decision trees (bagging)

**GPU Requirements:**
- Recommended: NVIDIA RTX A40 (48GB VRAM)
- Higher memory for ensemble storage

**Cost Estimate (per 1M rows):**
- Training Time: 5-10 minutes
- GPU Cost: $0.04 - $0.08

**When to Use:**
- High accuracy needed
- Robust to overfitting
- Feature importance analysis
- Handle missing values well

**Hyperparameters:**
- `n_estimators`: Number of trees (default: 100)
  - More trees = better accuracy, slower
  - Recommended: 100-500
- `max_depth`: Tree depth (default: None)
- `max_features`: Features per tree (default: 'sqrt')

**Performance:**
- Generally 2-5% accuracy improvement over single tree
- Excellent for tabular data

**Example:**
```
🏋️ Training Random Forest (n_estimators=300)

📊 Tree Building Progress:
  - Trees 1-100: Complete
  - Trees 101-200: Complete
  - Trees 201-300: Complete

✅ Training Complete!
  • Accuracy: 94.23%
  • Feature Importance: [provided in report]
```

---

#### 4. Gradient Boosting Classifier

**Description:** Ensemble of sequential trees (boosting)

**GPU Requirements:**
- Recommended: NVIDIA A100 PCIe 40GB
- GPU-accelerated boosting algorithms

**Cost Estimate (per 1M rows):**
- Training Time: 10-20 minutes
- GPU Cost: $0.13 - $0.26

**When to Use:**
- State-of-art accuracy required
- Structured/tabular data
- Kaggle-level competition performance
- Have GPU budget

**Hyperparameters:**
- `n_estimators`: Boosting rounds (default: 100)
- `learning_rate`: Step size (default: 0.1)
  - Smaller = slower but more accurate
  - Range: 0.01 - 0.3
- `max_depth`: Tree depth (default: 3)

**GPU Acceleration:**
- Uses GPU for tree building (5-10x faster than CPU)
- Especially beneficial for large datasets (>100K rows)

**Best Practices:**
- Start with learning_rate=0.1, n_estimators=100
- Lower learning_rate + increase n_estimators = better accuracy
- Monitor for overfitting via validation metrics

---

#### 5. SVM (Support Vector Machine)

**Description:** Kernel-based classifier for non-linear boundaries

**GPU Requirements:**
- Recommended: NVIDIA RTX A40
- Memory-intensive for large datasets

**Cost Estimate (per 100K rows):**
- Training Time: 5-15 minutes
- GPU Cost: $0.03 - $0.10

**When to Use:**
- Small to medium datasets (<100K rows)
- Non-linear decision boundaries
- High-dimensional data
- Need maximum-margin classifier

**Hyperparameters:**
- `C`: Regularization (default: 1.0)
- `kernel`: Kernel function (default: 'rbf')
  - 'linear': Linear boundary
  - 'rbf': Radial basis (non-linear)
  - 'poly': Polynomial

**Limitations:**
- Slow for large datasets (>100K rows)
- Memory scales quadratically
- Consider Random Forest for large data

---

#### 6. Naive Bayes Classifier

**Description:** Probabilistic classifier based on Bayes' theorem

**GPU Requirements:**
- Recommended: NVIDIA RTX A5000

**Cost Estimate (per 1M rows):**
- Training Time: 1-2 minutes
- GPU Cost: $0.01 - $0.02

**When to Use:**
- Text classification (spam detection)
- Fast baseline needed
- Independent features
- Real-time prediction latency critical

**Hyperparameters:**
- Minimal configuration required
- Uses Gaussian Naive Bayes by default

**Advantages:**
- Extremely fast training and prediction
- Works well with small datasets
- Handles missing data gracefully

---

### 5.3 Neural Network Models (2 Models)

**Use Case:** Deep learning for complex patterns, image/text data, non-linear relationships

#### 1. MLP Regression (Multi-Layer Perceptron)

**Description:** Feedforward neural network for regression tasks

**GPU Requirements:**
- Recommended: NVIDIA A100 PCIe 40GB (40GB VRAM)
- GPU essential for training speed

**Cost Estimate (per 1M rows):**
- Training Time: 15-30 minutes
- GPU Cost: $0.20 - $0.40

**When to Use:**
- Complex non-linear patterns
- Large datasets (>100K rows)
- Feature interactions not captured by linear models
- Have GPU budget

**Architecture Configuration:**

Bot guides through network design:

```
🏗️ Neural Network Architecture

Layer 1 (Input): [auto-detected from features]
Hidden Layers: Configure now

How many hidden layers? (1-5, default 2):
User: 2

Layer 1 neurons (16-512, default 64):
User: 128

Layer 2 neurons (16-512, default 32):
User: 64

Activation function (relu/tanh/sigmoid, default relu):
User: relu

Dropout rate (0.0-0.5, default 0.2):
User: 0.3
```

**Hyperparameters:**
- `hidden_layers`: List of layer sizes (e.g., [128, 64])
- `activation`: Activation function ('relu', 'tanh', 'sigmoid')
- `dropout`: Dropout rate for regularization (0.0-0.5)
- `learning_rate`: Optimizer learning rate (default: 0.001)
- `batch_size`: Mini-batch size (default: 32)
- `epochs`: Training epochs (default: 100)

**Training Monitoring:**
```
🏋️ Training MLP Regression

Epoch 1/100 - Loss: 0.8234 - Val Loss: 0.7892
Epoch 10/100 - Loss: 0.3421 - Val Loss: 0.3567
Epoch 25/100 - Loss: 0.1234 - Val Loss: 0.1456
...
Epoch 100/100 - Loss: 0.0456 - Val Loss: 0.0523

✅ Training Complete!
  • Final Loss: 0.0456
  • Validation Loss: 0.0523
  • R²: 0.8876
```

**Best Practices:**
- Start simple (2 layers, 64-128 neurons)
- Monitor validation loss for overfitting
- Use dropout if overfitting occurs
- GPU acceleration provides 10-50x speedup

---

#### 2. MLP Classification (Multi-Layer Perceptron)

**Description:** Feedforward neural network for classification tasks

**GPU Requirements:**
- Recommended: NVIDIA A100 PCIe 40GB

**Cost Estimate (per 1M rows):**
- Training Time: 15-30 minutes
- GPU Cost: $0.20 - $0.40

**When to Use:**
- Complex decision boundaries
- Image or text classification
- Large datasets with many features
- Pattern recognition tasks

**Architecture Configuration:**

Same as MLP Regression, with additional output layer configuration:

```
Output layer activation:
  • Binary classification: sigmoid
  • Multiclass (>2 classes): softmax

Auto-configured based on target variable.
```

**Hyperparameters:**
- Same as MLP Regression
- Output activation auto-selected

**Class Imbalance Handling:**
- Bot automatically detects class imbalance
- Applies class weights if ratio >10:1

**Example Output:**
```
📊 Metrics:
  • Accuracy: 95.67%
  • Precision (weighted): 95.34%
  • Recall (weighted): 95.67%
  • F1 Score (weighted): 95.50%

Per-Class Metrics:
  Class 0: Precision=96.2%, Recall=94.8%
  Class 1: Precision=94.5%, Recall=96.5%
```

---

### 5.4 Model Selection Decision Tree

Use this flowchart to choose the right model:

```
START
  |
  v
Predict continuous or categorical?
  |
  ├─ Continuous (Regression)
  |    |
  |    v
  |  Dataset size?
  |    |
  |    ├─ Small (<10K rows)
  |    |    └─> Linear Regression (baseline) or Polynomial (non-linear)
  |    |
  |    ├─ Medium (10K-100K rows)
  |    |    └─> Ridge/Lasso (if many features) or Random Forest
  |    |
  |    └─ Large (>100K rows)
  |         └─> Gradient Boosting or MLP Regression (GPU)
  |
  └─ Categorical (Classification)
       |
       v
     Dataset size?
       |
       ├─ Small (<10K rows)
       |    └─> Logistic Regression or Decision Tree
       |
       ├─ Medium (10K-100K rows)
       |    └─> Random Forest or SVM
       |
       └─ Large (>100K rows)
            └─> Gradient Boosting or MLP Classification (GPU)
```

---

### 5.5 GPU Requirement Summary

| Model Type | Min GPU | Recommended GPU | Dataset Size Limit |
|------------|---------|-----------------|-------------------|
| **Linear, Ridge, Lasso, ElasticNet** | RTX A5000 (24GB) | RTX A5000 | 10M rows |
| **Polynomial Regression** | RTX A40 (48GB) | RTX A40 | 1M rows |
| **Logistic, Decision Tree, Naive Bayes** | RTX A5000 | RTX A5000 | 10M rows |
| **Random Forest** | RTX A40 (48GB) | RTX A40 | 5M rows |
| **Gradient Boosting** | RTX A40 (48GB) | A100 40GB | 10M rows |
| **SVM** | RTX A40 (48GB) | RTX A40 | 100K rows |
| **MLP Regression/Classification** | A100 40GB | A100 80GB | 50M rows |

**Notes:**
- Dataset limits assume ~50 features
- Neural networks benefit most from GPU acceleration
- Gradient boosting uses GPU-accelerated tree building
- All other models are CPU-optimized but run on GPU instances

---

## 6. Workflow States

Understanding the workflow states helps you navigate the training process and troubleshoot issues.

### 6.1 Complete State Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CLOUD TRAINING WORKFLOW                  │
└─────────────────────────────────────────────────────────────┘

START (/train command)
  │
  v
┌──────────────────────────────┐
│ CHOOSING_TRAINING_LOCATION   │ ◄─── Default entry point
│ Prompt: "Local or Cloud?"    │
└──────────────────────────────┘
  │ User: "Cloud"
  v
┌──────────────────────────────┐
│ AWAITING_S3_DATASET          │ ◄─── Data source selection
│ Prompt: "Upload CSV or       │
│          provide path"       │
└──────────────────────────────┘
  │ Options:
  ├─ Upload file via Telegram
  ├─ Provide local path
  ├─ Provide RunPod URI
  └─ Provide S3 URI (AWS)
  │
  v
┌──────────────────────────────┐
│ CHOOSING_LOAD_STRATEGY       │ ◄─── Schema detection
│ Prompt: "Auto-detect schema  │
│          or manual?"         │
└──────────────────────────────┘
  │ User: "Auto-detect"
  v
┌──────────────────────────────┐
│ CONFIRMING_SCHEMA            │ ◄─── Review detected schema
│ Display: Dataset stats,      │
│          suggested target,   │
│          features, task type │
└──────────────────────────────┘
  │ User: "Accept"
  v
┌──────────────────────────────┐
│ SELECTING_MODEL_CATEGORY     │ ◄─── Choose model category
│ Prompt: "Regression (1),     │
│          Classification (2), │
│          Neural Networks (3)"│
└──────────────────────────────┘
  │ User: Select category
  v
┌──────────────────────────────┐
│ SELECTING_MODEL              │ ◄─── Choose specific model
│ Display: List of models in   │
│          selected category   │
└──────────────────────────────┘
  │ User: Select model
  v
┌──────────────────────────────┐
│ CONFIRMING_MODEL             │ ◄─── Confirm model choice
│ Display: Model summary,      │
│          hyperparameters     │
└──────────────────────────────┘
  │ User: "Confirm"
  v
┌──────────────────────────────┐
│ CONFIRMING_INSTANCE_TYPE     │ ◄─── GPU selection & cost
│ Display: Recommended GPU,    │
│          cost estimate,      │
│          time estimate       │
└──────────────────────────────┘
  │ User: "Launch"
  v
┌──────────────────────────────┐
│ LAUNCHING_TRAINING           │ ◄─── Pod creation
│ Status: Creating RunPod GPU  │
│         instance...          │
└──────────────────────────────┘
  │ Pod started
  v
┌──────────────────────────────┐
│ MONITORING_TRAINING          │ ◄─── Real-time log streaming
│ Display: Live training logs, │
│          progress updates    │
└──────────────────────────────┘
  │ Training complete
  v
┌──────────────────────────────┐
│ TRAINING_COMPLETE            │ ◄─── Results summary
│ Display: Model ID, metrics,  │
│          cost, storage URI   │
└──────────────────────────────┘
  │
  v
COMPLETE
```

---

### 6.2 State Descriptions and Expected Prompts

#### State: CHOOSING_TRAINING_LOCATION

**Purpose:** Select training environment (local CPU vs cloud GPU)

**Bot Prompt:**
```
🌩️ Training Location

Where would you like to train this model?

💻 Local Training (Free)
  • Runs on this server
  • Limited resources (CPU/RAM)
  • Best for: Small datasets (<1GB), quick experiments

☁️ Cloud Training (Paid - RunPod GPU)
  • Runs on RunPod GPU Pods
  • Scalable GPU resources (24-80GB VRAM)
  • Best for: Large datasets (>1GB), neural networks
  • Cost: $0.29 - $1.19 per hour (billed per second)

Choose: Local or Cloud
```

**User Actions:**
- Reply: "Local" → Switches to LOCAL_TRAINING workflow
- Reply: "Cloud" → Continues to AWAITING_S3_DATASET
- Reply: "/cancel" → Aborts workflow

**Validation:**
- Must be "Local" or "Cloud" (case-insensitive)

---

#### State: AWAITING_S3_DATASET

**Purpose:** Collect dataset from user (upload or path)

**Bot Prompt:**
```
☁️ Cloud Dataset Input

Please provide your dataset:

1️⃣ Upload CSV File
  • Send file directly via Telegram
  • Will be automatically uploaded to RunPod storage
  • Max size: 100MB

2️⃣ Provide Dataset Path
  • Local file: /home/user/data/housing.csv
  • RunPod storage: runpod://v3zskt9gvb/datasets/data.csv
  • S3 URI: s3://bucket/data.csv

Which option would you like to use?
```

**User Actions:**
- Upload file → Bot validates and uploads to RunPod
- Provide local path → Bot validates path and uploads
- Provide RunPod URI → Bot validates URI
- Provide S3 URI → Bot downloads from S3 (AWS only)

**Validation:**
- File format: CSV, Excel, Parquet
- File size: Under 100MB (Telegram) or 10GB (local path)
- Path security: Must be in allowed directories

**Error Handling:**
- Invalid format → Re-prompt with error message
- File too large → Suggest local path or S3 option
- Path not allowed → Show allowed directories

---

#### State: CHOOSING_LOAD_STRATEGY

**Purpose:** Select schema detection method

**Bot Prompt:**
```
🔍 Schema Detection

How would you like to configure features?

1️⃣ Auto-Detect (Recommended)
  • Bot analyzes dataset
  • Suggests target column
  • Suggests feature columns
  • Recommends task type
  • You review and confirm

2️⃣ Manual Entry
  • You specify target column
  • You specify feature columns
  • You specify task type

Choose: Auto or Manual
```

**User Actions:**
- Reply: "Auto" → Continues to CONFIRMING_SCHEMA
- Reply: "Manual" → Goes to AWAITING_MANUAL_SCHEMA

**Auto-Detection Logic:**
- Analyzes column types (numeric, categorical, datetime)
- Detects target column (last column or most categorical)
- Suggests features (all numeric/categorical except target)
- Recommends task type based on target cardinality:
  - Binary classification: 2 unique values
  - Multiclass classification: 3-20 unique values
  - Regression: >20 unique values or continuous

---

#### State: CONFIRMING_SCHEMA

**Purpose:** Review and approve auto-detected configuration

**Bot Prompt:**
```
✅ Dataset Loaded Successfully

📊 Dataset Info:
  - Rows: 506
  - Columns: 14
  - Size: 52.3 KB
  - Missing values: 0

🎯 Detected Target: price (continuous, range: 5.0-50.0)
📐 Suggested Features (13):
  - crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat

🤖 Recommended Task: Regression

Schema Summary:
  • Numeric features: 13
  • Categorical features: 0
  • Target type: Continuous

Accept this configuration? (yes/no/edit)
```

**User Actions:**
- Reply: "yes" → Continues to SELECTING_MODEL_CATEGORY
- Reply: "no" → Returns to CHOOSING_LOAD_STRATEGY (manual mode)
- Reply: "edit" → Allows manual override of target/features

**Schema Display Format:**
- Clear statistics (rows, columns, size)
- Target column with value range/cardinality
- Feature list (comma-separated, truncated if >20)
- Task type recommendation with reasoning

---

#### State: SELECTING_MODEL_CATEGORY

**Purpose:** Choose model category (regression, classification, neural)

**Bot Prompt:**
```
📚 Model Categories

Based on your task (Regression), here are available model types:

1️⃣ Regression (5 models)
   - Predict continuous values
   - Examples: Linear, Ridge, Lasso, ElasticNet, Polynomial
   - Best for: Price prediction, forecasting

2️⃣ Classification (6 models)
   - Predict categories
   - Examples: Logistic, Random Forest, Gradient Boosting
   - Best for: Spam detection, image classification

3️⃣ Neural Networks (2 models)
   - Deep learning
   - Examples: MLP Regression, MLP Classification
   - Best for: Complex patterns, large datasets

Which category matches your task?
```

**User Actions:**
- Reply: "1" (Regression) → Shows regression models
- Reply: "2" (Classification) → Shows classification models
- Reply: "3" (Neural) → Shows neural network models

**Task Type Validation:**
- Bot warns if category mismatch with detected task type
- Example: "You selected Regression but detected task is Classification. Continue anyway? (yes/no)"

---

#### State: SELECTING_MODEL

**Purpose:** Select specific model within category

**Bot Prompt (Regression Example):**
```
🎯 Regression Models

1. Linear Regression
   - Fast baseline model
   - Interpretable coefficients
   - Best for: Simple linear relationships

2. Ridge Regression
   - Linear with L2 regularization
   - Prevents overfitting
   - Best for: Many correlated features

3. Lasso Regression
   - Linear with L1 regularization
   - Automatic feature selection
   - Best for: Sparse models

4. ElasticNet
   - Combines Ridge and Lasso
   - Balanced regularization
   - Best for: Large feature sets

5. Polynomial Regression
   - Non-linear relationships
   - Feature expansion
   - Best for: Curved patterns

Choose model (1-5):
```

**User Actions:**
- Reply: Model number (1-5)
- Reply: "/back" → Returns to category selection

**Model Recommendation:**
- Bot highlights recommended model with ⭐ icon
- Based on dataset size and feature count

---

#### State: CONFIRMING_MODEL

**Purpose:** Confirm model selection and configure hyperparameters

**Bot Prompt:**
```
🤖 Model Configuration

Selected Model: Random Forest Regression

📊 Default Hyperparameters:
  - n_estimators: 100 (number of trees)
  - max_depth: None (unlimited depth)
  - min_samples_split: 2
  - min_samples_leaf: 1

💡 Recommendations for your dataset (506 rows):
  - n_estimators: 100-200 (good balance)
  - max_depth: 10-15 (prevent overfitting)

Options:
1️⃣ Use defaults (recommended for first run)
2️⃣ Customize hyperparameters

Choose: Default or Customize
```

**User Actions:**
- Reply: "Default" → Uses default hyperparameters
- Reply: "Customize" → Enters hyperparameter configuration flow
- Reply: "/back" → Returns to model selection

**Hyperparameter Configuration Flow (if Customize):**
```
Bot: Enter n_estimators (50-500, default 100):
User: 200
Bot: ✅ n_estimators set to 200

Bot: Enter max_depth (5-50, default None):
User: 15
Bot: ✅ max_depth set to 15

Bot: Configuration complete. Continue? (yes/back)
User: yes
```

---

#### State: CONFIRMING_INSTANCE_TYPE

**Purpose:** GPU selection and final cost confirmation

**Bot Prompt:**
```
☁️ Cloud Training Configuration

📊 Dataset: 52.3 KB (506 rows, 13 features)
🎯 Model: Random Forest Regression (n_estimators=200)
🎯 GPU Type: NVIDIA RTX A5000 (Recommended)
💡 GPU Info: 24GB VRAM - $0.29/hr - Best for small datasets

⏱️ Estimated Time: ~3 minutes
💰 Estimated Cost: $0.015

⚠️ Important:
  • You will be charged for actual usage (billed per second)
  • Training logs will stream in real-time
  • Pod will auto-terminate when complete
  • RunPod may reclaim GPU (rare, will retry)

Ready to launch cloud training?

Options:
1️⃣ Yes, launch training
2️⃣ Change GPU type
3️⃣ Back to model selection
```

**User Actions:**
- Reply: "1" or "yes" → Launches training
- Reply: "2" → Shows GPU selection menu
- Reply: "3" or "/back" → Returns to model confirmation

**GPU Selection Menu (if user chooses "Change GPU"):**
```
Available GPUs:

1. NVIDIA RTX A5000
   - 24GB VRAM
   - $0.29/hour
   - Best for: Small-medium datasets (<1M rows)

2. NVIDIA RTX A40
   - 48GB VRAM
   - $0.39/hour
   - Best for: Medium datasets (1-5M rows)

3. NVIDIA A100 PCIe 40GB
   - 40GB VRAM
   - $0.79/hour
   - Best for: Large datasets, neural networks

4. NVIDIA A100 PCIe 80GB
   - 80GB VRAM
   - $1.19/hour
   - Best for: Very large models

Select GPU (1-4):
```

---

#### State: LAUNCHING_TRAINING

**Purpose:** RunPod instance creation and initialization

**Bot Messages:**
```
🚀 Cloud Training Launched

Pod ID: pod-xyz123abc456
GPU Type: NVIDIA RTX A5000
Status: Launching...

⏳ Waiting for pod to start (typically 1-2 minutes)...

[30 seconds later]
✅ Pod started successfully!
📦 Uploading dataset to pod...

[10 seconds later]
✅ Dataset uploaded
🔧 Installing dependencies...

[20 seconds later]
✅ Environment ready
🏋️ Starting training...
```

**What Happens:**
1. Bot creates RunPod GPU pod via API
2. Waits for pod to enter "RUNNING" state
3. Uploads dataset to pod storage
4. Installs Python dependencies (scikit-learn, pandas, etc.)
5. Starts training script execution

**Timeout Handling:**
- Pod creation timeout: 5 minutes
- If timeout: Bot retries with different GPU type
- User notified of any delays

---

#### State: MONITORING_TRAINING

**Purpose:** Real-time training progress streaming

**Bot Messages (Example - Random Forest):**
```
📝 [00:00:05] Loading dataset...
📝 [00:00:08] Dataset loaded: 506 rows, 13 features
📝 [00:00:10] Splitting train/test (80/20)...
📝 [00:00:12] Preprocessing features (StandardScaler)...
📝 [00:00:15] Training Random Forest (n_estimators=200)...

🏋️ Training Progress: 25%
████░░░░░░
  - Trees 1-50: Complete

🏋️ Training Progress: 50%
████████░░
  - Trees 51-100: Complete

🏋️ Training Progress: 75%
████████████
  - Trees 101-150: Complete

🏋️ Training Progress: 100%
██████████
  - Trees 151-200: Complete

📝 [00:02:45] Evaluating model on test set...
📝 [00:02:48] Calculating metrics...
📝 [00:02:50] Saving model to RunPod storage...
📝 [00:02:55] Upload complete!
```

**Log Streaming:**
- Bot polls pod logs every 5 seconds
- Progress updates sent every 10% completion
- Full logs stored for debugging

**User Interruption:**
- Reply: "/cancel" → Stops training and terminates pod
- Pod cost still charged for actual runtime

---

#### State: TRAINING_COMPLETE

**Purpose:** Display final results and model information

**Bot Prompt:**
```
✅ Cloud Training Complete!

🎯 Model ID: model_7715560927_random_forest_regression_20251108_143022
📦 Storage URI: runpod://v3zskt9gvb/models/user_7715560927/...
⏱️ Training Time: 2.9 minutes
💰 Actual Cost: $0.014

📊 Metrics:
  • R²: 0.8234
  • RMSE: 3.4567
  • MAE: 2.1234
  • MSE: 11.9487

📈 Feature Importance (Top 5):
  1. rm (room count): 45.2%
  2. lstat (% lower status): 23.7%
  3. ptratio (pupil-teacher ratio): 12.3%
  4. dis (distance to employment): 8.9%
  5. crim (crime rate): 5.4%

💾 Model Size: 4.2 MB
📁 Files Saved:
  - model.pkl (trained model)
  - scaler.pkl (feature scaler)
  - metadata.json (training config)
  - metrics.json (performance metrics)

The model has been saved to RunPod storage and can be used for predictions.

Next Steps:
  • /predict - Run predictions with this model
  • /models - View all your trained models
  • /train - Train another model
```

**Cost Comparison:**
```
💰 Cost Breakdown:
  • Estimated: $0.015
  • Actual: $0.014
  • Variance: -6.7% (under budget)

📊 Breakdown:
  - Compute (GPU): $0.0120
  - Storage: $0.0015
  - Data Transfer: $0.0005
```

---

### 6.3 Error State Handling

All workflows include an ERROR_STATE for graceful recovery.

**Entry Conditions:**
- Dataset upload failure
- GPU pod creation failure
- Training script crash
- Network timeout
- Budget exceeded

**Bot Prompt (Example):**
```
❌ Training Error

Error Type: Pod Creation Timeout
Details: RunPod did not allocate GPU within 5 minutes

Possible Causes:
  • High demand for selected GPU type
  • Temporary service interruption
  • Regional capacity constraints

Recovery Options:
1️⃣ Retry with same GPU
2️⃣ Retry with different GPU (NVIDIA RTX A40)
3️⃣ Cancel and return to model selection
4️⃣ Contact support

Choose option (1-4):
```

**User Actions:**
- Reply: "1" → Retries pod creation (max 3 attempts)
- Reply: "2" → Tries alternative GPU type
- Reply: "3" → Returns to SELECTING_MODEL state
- Reply: "4" → Escalates to human support

**Error Context Saved:**
- Error message and stack trace
- Previous state for recovery
- Retry count (max 3)
- Timestamp for debugging

---

### 6.4 State Transition Timing

| State | Typical Duration | Timeout |
|-------|-----------------|---------|
| CHOOSING_TRAINING_LOCATION | User input | None |
| AWAITING_S3_DATASET | User input | 30 min session timeout |
| CHOOSING_LOAD_STRATEGY | User input | None |
| CONFIRMING_SCHEMA | User input | None |
| SELECTING_MODEL_CATEGORY | User input | None |
| SELECTING_MODEL | User input | None |
| CONFIRMING_MODEL | User input | None |
| CONFIRMING_INSTANCE_TYPE | User input | None |
| LAUNCHING_TRAINING | 1-3 minutes | 5 min |
| MONITORING_TRAINING | 2-30 minutes | 60 min |
| TRAINING_COMPLETE | User input | None |

**Session Timeout:**
- Inactive sessions expire after 30 minutes
- Bot sends warning at 25 minutes
- User can extend session with any message

---

## 7. Cost Management

### 7.1 Cost Estimation

The bot provides detailed cost estimates before launching training.

**Estimation Factors:**

1. **Dataset Size**
   - Small (<1MB): Minimal cost ($0.01-$0.05)
   - Medium (1-100MB): Low cost ($0.05-$0.20)
   - Large (100MB-1GB): Moderate cost ($0.20-$1.00)
   - Very Large (>1GB): High cost ($1.00+)

2. **Model Complexity**
   - Linear models: Fast (1-2 min)
   - Tree ensembles: Medium (5-10 min)
   - Neural networks: Slow (15-30 min)

3. **GPU Type**
   - RTX A5000: $0.29/hour = $0.0048/min
   - RTX A40: $0.39/hour = $0.0065/min
   - A100 40GB: $0.79/hour = $0.0132/min
   - A100 80GB: $1.19/hour = $0.0198/min

**Estimation Formula:**
```
Estimated Cost = (Training Time Minutes × GPU Rate per Minute) + Storage + Transfer
```

**Example Estimates:**

| Dataset | Model | GPU | Time | Cost |
|---------|-------|-----|------|------|
| Iris (4KB) | Random Forest | RTX A5000 | 2 min | $0.01 |
| Housing (50KB) | Gradient Boosting | RTX A40 | 5 min | $0.03 |
| Census (5MB) | MLP Classification | A100 40GB | 15 min | $0.20 |
| ImageNet (50MB) | MLP + Custom Layers | A100 80GB | 30 min | $0.60 |

---

### 7.2 Cost Tracking

The bot tracks actual costs in real-time during training.

**Real-Time Cost Display:**
```
🏋️ Training Progress: 45% - Training (Epoch 45/100)

█████░░░░░

📉 Loss: 0.324
🎯 Accuracy: 0.876

⏱️ ETA: 12 minutes
💰 Cost So Far: $0.087
```

**Cost Components:**

1. **Compute (GPU)**: Primary cost, billed per second
2. **Storage**: RunPod Network Volume ($0.07/GB/month)
3. **Data Transfer**: Upload/download costs (negligible for most jobs)

**Cost Alerts:**

Bot sends alerts at spending thresholds:

- 50% of budget: ⚠️ Warning
- 75% of budget: ⚠️ High usage alert
- 90% of budget: 🚨 Critical alert
- 100% of budget: 🚨 Budget exceeded, training continues

**Example Alert:**
```
⚠️ Cost Alert: Training has consumed $0.45 of your $0.50 budget (90%)

📊 Breakdown:
  - Compute (GPU): $0.40
  - Storage: $0.03
  - Data Transfer: $0.02

💡 Projected Final Cost: $0.52 (over budget by $0.02)
```

---

### 7.3 Budget Configuration

Set spending limits in `config.yaml`:

```yaml
cloud:
  cost_limits:
    max_training_cost_dollars: 10.0      # Per-job limit
    max_prediction_cost_dollars: 1.0     # Per-prediction limit
    cost_warning_threshold: 0.8          # Warn at 80%
```

**User-Specific Budgets:**

Admins can set per-user budgets:

```python
# In bot configuration
USER_BUDGETS = {
    123456: 50.0,  # User ID → Monthly budget
    789012: 100.0,
}
```

**Budget Enforcement:**

- Hard limit: Training stops if budget exceeded (optional)
- Soft limit: Warning only, training continues (default)

---

### 7.4 Cost Optimization Tips

**1. Choose the Right GPU**

- Don't overpay for unused VRAM
- RTX A5000 sufficient for most tabular data (<1M rows)
- A100 only needed for large neural networks

**2. Use Efficient Models**

| Model Type | Relative Cost | When to Use |
|------------|---------------|-------------|
| Linear/Logistic | 1x (baseline) | Interpretability needed |
| Random Forest | 2-3x | Accuracy boost worth cost |
| Gradient Boosting | 5-10x | Competition-level accuracy |
| Neural Networks | 10-20x | Complex patterns, large data |

**3. Optimize Dataset**

- Remove unnecessary columns before upload
- Sample large datasets for prototyping
- Use Parquet format (smaller than CSV)

**4. Batch Training Jobs**

- Train multiple models in single pod session
- Reuse uploaded datasets
- Avoid repeated pod initialization costs

**5. Use Spot/Interruptible Instances**

- RunPod Community Cloud uses spot pricing automatically
- 50-70% cheaper than on-demand
- Rare interruptions (<5% of jobs)

**6. Monitor and Terminate**

- Watch training logs for completion
- Terminate pod immediately after training
- Avoid idle pod costs (charged while running)

---

### 7.5 Cost Comparison: Cloud vs Local

**Scenario 1: Small Dataset (Iris, 150 rows)**

| Environment | Time | Cost |
|-------------|------|------|
| Local (CPU) | 5 seconds | $0 (free) |
| Cloud (RTX A5000) | 2 minutes | $0.01 |

**Verdict:** Use local for quick experiments

---

**Scenario 2: Medium Dataset (Housing, 500K rows, Random Forest)**

| Environment | Time | Cost |
|-------------|------|------|
| Local (CPU 4-core) | 15 minutes | $0 (free) |
| Cloud (RTX A40) | 3 minutes | $0.02 |

**Verdict:** Cloud provides 5x speedup for $0.02

---

**Scenario 3: Large Dataset (1M rows, Gradient Boosting)**

| Environment | Time | Cost |
|-------------|------|------|
| Local (CPU 8-core) | 2 hours | $0 (free, but blocks server) |
| Cloud (A100 40GB) | 15 minutes | $0.20 |

**Verdict:** Cloud dramatically faster, worth the cost

---

**Scenario 4: Neural Network (1M rows, MLP 3 hidden layers)**

| Environment | Time | Cost |
|-------------|------|------|
| Local (CPU 8-core) | 8 hours | $0 (free, but impractical) |
| Cloud (A100 40GB GPU) | 20 minutes | $0.26 |

**Verdict:** Cloud essential for neural networks

---

### 7.6 Monthly Cost Planning

**Light Usage (Experimentation)**
- 10 training jobs/month
- Average: $0.05/job
- **Total: $0.50/month**

**Moderate Usage (Active Development)**
- 50 training jobs/month
- Mix: 30 small ($0.02), 15 medium ($0.10), 5 large ($0.50)
- **Total: $8.60/month**

**Heavy Usage (Production/Research)**
- 200 training jobs/month
- Mix: 100 small ($0.02), 70 medium ($0.10), 30 large ($0.50)
- **Total: $24.00/month**

**Storage Costs:**
- 10GB stored models: $0.70/month
- 50GB dataset cache: $3.50/month
- **Total Storage: $4.20/month**

**Grand Total Estimates:**
- Light: $0.50 + $0.70 = **$1.20/month**
- Moderate: $8.60 + $2.10 = **$10.70/month**
- Heavy: $24.00 + $4.20 = **$28.20/month**

---

## 8. Troubleshooting

### 8.1 Common Errors and Solutions

#### Error: "Pod Creation Timeout"

**Symptoms:**
```
❌ Training Error

Error Type: Pod Creation Timeout
Details: RunPod did not allocate GPU within 5 minutes
```

**Causes:**
- High demand for selected GPU type
- Regional capacity constraints
- Temporary RunPod service issue

**Solutions:**
1. **Retry with same GPU** (often succeeds on 2nd attempt)
2. **Choose different GPU type** (e.g., switch A100 → RTX A40)
3. **Try different region** (configure in settings)
4. **Wait 5-10 minutes** and retry (demand fluctuates)

**Prevention:**
- Use RTX A5000 for most jobs (highest availability)
- Avoid peak hours (weekday mornings US time)
- Keep RunPod account funded ($5+ balance)

---

#### Error: "Dataset Upload Failed"

**Symptoms:**
```
❌ Upload Error

Failed to upload dataset to RunPod storage
Details: Connection timeout after 60 seconds
```

**Causes:**
- Large file size (>100MB) over slow network
- Temporary network interruption
- RunPod storage service issue

**Solutions:**
1. **Retry upload** (temporary network issues resolve)
2. **Use smaller dataset** (sample rows for prototyping)
3. **Pre-upload to RunPod** (use RunPodCTL or Web UI)
4. **Check network connection** (stable WiFi/Ethernet needed)

**Prevention:**
- For large files, use RunPod URI instead of upload
- Compress datasets (Parquet format vs CSV)
- Upload during off-peak hours

---

#### Error: "Training Script Crashed"

**Symptoms:**
```
❌ Training Error

Error Type: Script Execution Failed
Details: ValueError: Input contains NaN

Traceback:
  File "train.py", line 42, in <module>
    model.fit(X_train, y_train)
```

**Causes:**
- Missing values (NaN) in dataset
- Invalid feature types (strings in numeric columns)
- Memory overflow (dataset too large for GPU)

**Solutions:**

**For Missing Values:**
1. Check dataset for NaN: `df.isnull().sum()`
2. Preprocessing options:
   - Drop rows: `df.dropna()`
   - Impute with mean: `df.fillna(df.mean())`
   - Bot can handle this: Reply "yes" to preprocessing prompt

**For Invalid Types:**
1. Check column types: `df.dtypes`
2. Convert to numeric: `pd.to_numeric(df['column'], errors='coerce')`
3. Ensure target column is numeric (for regression) or categorical (for classification)

**For Memory Issues:**
1. Reduce dataset size (sample rows)
2. Select fewer features
3. Use smaller GPU (RTX A5000 → A40 → A100 40GB)
4. Batch processing (split dataset into chunks)

---

#### Error: "Model Training Diverged"

**Symptoms:**
```
📝 Epoch 10/100 - Loss: 0.5234
📝 Epoch 20/100 - Loss: 1.2345
📝 Epoch 30/100 - Loss: NaN

❌ Training stopped: Loss diverged (NaN detected)
```

**Causes:**
- Learning rate too high (neural networks)
- Exploding gradients
- Numerical instability (extreme feature values)

**Solutions:**

**For Neural Networks:**
1. **Lower learning rate**: 0.001 → 0.0001
2. **Add gradient clipping**: max_grad_norm=1.0
3. **Normalize features**: StandardScaler or MinMaxScaler
4. **Use batch normalization**: Add to architecture

**For Gradient Boosting:**
1. **Lower learning rate**: 0.1 → 0.01
2. **Increase regularization**: min_child_weight, lambda
3. **Reduce max_depth**: 10 → 5

**Prevention:**
- Always normalize features (bot does this automatically if enabled)
- Start with conservative hyperparameters
- Monitor loss during training (early stopping if loss increases)

---

#### Error: "Budget Exceeded"

**Symptoms:**
```
🚨 Budget Exceeded

Job ID: job-123
Budget: $5.00
Final Cost: $5.50
Over Budget: $0.50

Training completed but exceeded your budget limit.
```

**Causes:**
- Training took longer than estimated
- Larger dataset than expected
- Complex model required more epochs

**Solutions:**
1. **Increase budget** in config: `max_training_cost_dollars: 10.0`
2. **Use faster model** (Random Forest → Logistic Regression)
3. **Reduce dataset size** (sample or filter rows)
4. **Set hard budget limit** (stop training at limit)

**Prevention:**
- Review cost estimate before confirming
- Monitor real-time cost during training
- Use cost alerts (90% threshold)

---

#### Error: "Invalid Credentials"

**Symptoms:**
```
❌ Authentication Error

RunPod API key invalid or expired
Details: 401 Unauthorized
```

**Causes:**
- API key not set in `.env`
- API key expired or revoked
- Typo in API key

**Solutions:**
1. **Check `.env` file:**
   ```bash
   cat .env | grep RUNPOD_API_KEY
   ```
2. **Regenerate API key:**
   - Visit https://www.runpod.io/console/settings
   - Create new API key
   - Update `.env` file
3. **Restart bot** to reload environment variables

**Prevention:**
- Store API key securely (password manager)
- Don't commit `.env` to Git
- Rotate keys quarterly

---

### 8.2 Debugging Techniques

#### Enable Debug Logging

**In `.env`:**
```bash
LOG_LEVEL=DEBUG
```

**View detailed logs:**
```bash
tail -f data/logs/agent.log
```

**Debug output includes:**
- API requests/responses
- State transitions
- Dataset validation details
- Cost calculations
- Pod creation status

---

#### Check RunPod Pod Status

**Via Web UI:**
1. Navigate to https://www.runpod.io/console/pods
2. Find your pod by ID (shown in bot message)
3. View logs, metrics, status

**Via RunPodCTL:**
```bash
runpodctl list pods
runpodctl logs <pod_id>
runpodctl exec <pod_id> nvidia-smi  # Check GPU usage
```

---

#### Validate Dataset Locally

**Before uploading, test dataset:**

```python
import pandas as pd

# Load dataset
df = pd.read_csv('your_data.csv')

# Check shape
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# Check for missing values
print(df.isnull().sum())

# Check column types
print(df.dtypes)

# Check target distribution
print(df['target_column'].value_counts())
```

---

#### Test Training Locally First

**Run local training to validate dataset:**

```bash
# Start bot and select "Local" training
/train
> Local

# If local training succeeds, cloud training will work
# If local fails, fix dataset issues before cloud attempt
```

---

### 8.3 Performance Issues

#### Slow Training (Longer than Estimate)

**Symptoms:**
- Training takes 2-3x longer than estimated
- GPU utilization low (<50%)

**Causes:**
- CPU bottleneck (data loading)
- Inefficient hyperparameters
- Large number of epochs

**Solutions:**
1. **Check GPU utilization:**
   ```bash
   runpodctl exec <pod_id> nvidia-smi
   ```
2. **Optimize data loading:**
   - Use Parquet format (faster I/O)
   - Increase batch size (neural networks)
3. **Adjust hyperparameters:**
   - Reduce n_estimators (tree models)
   - Reduce epochs (neural networks)
   - Increase learning rate (neural networks)

---

#### Out of Memory (OOM)

**Symptoms:**
```
❌ Training Error

Error Type: CUDA Out of Memory
Details: GPU ran out of VRAM (24GB exceeded)
```

**Causes:**
- Dataset too large for GPU VRAM
- Model too complex (too many parameters)
- Batch size too large (neural networks)

**Solutions:**
1. **Upgrade GPU:**
   - RTX A5000 (24GB) → RTX A40 (48GB) → A100 (40/80GB)
2. **Reduce batch size:**
   - 64 → 32 → 16
3. **Simplify model:**
   - Reduce hidden layers (neural networks)
   - Reduce n_estimators (tree models)
4. **Sample dataset:**
   - Use 50% of rows for prototyping

---

### 8.4 Getting Help

#### Bot Support Commands

```bash
/help           # Show all available commands
/status         # Check current session status
/models         # View trained models
/cancel         # Abort current workflow
```

#### Community Support

- **GitHub Issues:** https://github.com/your-repo/issues
- **Discord Server:** [invite link]
- **Documentation:** https://docs.yourproject.com

#### Reporting Bugs

**Include in bug report:**
1. Error message (screenshot or copy-paste)
2. Dataset size and format
3. Selected model and hyperparameters
4. Pod ID (if training launched)
5. Bot logs (if accessible)

**Example:**
```
Bug Report: Training Timeout

Error: Pod creation timeout after 5 minutes
Dataset: housing.csv (500KB, 506 rows, 14 columns)
Model: Random Forest Regression (n_estimators=100)
GPU: NVIDIA RTX A5000
Pod ID: N/A (failed to create)
Timestamp: 2025-11-08 14:30 UTC
```

---

## 9. Advanced Topics

### 9.1 Custom Hyperparameters

For advanced users, customize hyperparameters for optimal performance.

#### Gradient Boosting Tuning

**Key Hyperparameters:**

1. **Learning Rate (`learning_rate`)**
   - Lower = slower but more accurate
   - Recommended: 0.01 - 0.3
   - Default: 0.1

2. **Number of Estimators (`n_estimators`)**
   - More trees = better accuracy, slower training
   - Recommended: 100 - 1000
   - Default: 100

3. **Max Depth (`max_depth`)**
   - Tree complexity limit
   - Recommended: 3 - 10
   - Default: 3

4. **Min Samples Split (`min_samples_split`)**
   - Minimum samples to split node
   - Prevents overfitting
   - Recommended: 2 - 20
   - Default: 2

**Example Configuration:**
```
Bot: Configure Gradient Boosting hyperparameters

learning_rate (0.01-0.3, default 0.1):
User: 0.05

n_estimators (50-1000, default 100):
User: 500

max_depth (3-10, default 3):
User: 5

min_samples_split (2-20, default 2):
User: 10

✅ Configuration complete
Estimated training time: 15-20 minutes
Estimated cost: $0.20 (A100 40GB GPU)
```

**Tuning Strategy:**
1. Start with defaults
2. Lower learning_rate → increase n_estimators (2-3x)
3. Adjust max_depth based on validation performance
4. Use early stopping to prevent overfitting

---

#### Neural Network Architecture Design

**Architecture Components:**

1. **Hidden Layers**
   - More layers = deeper network
   - Recommended: 2-4 layers for tabular data
   - Too many layers = vanishing gradients

2. **Neurons per Layer**
   - Decreasing pattern: 128 → 64 → 32
   - Power of 2: 16, 32, 64, 128, 256, 512
   - Rule of thumb: 2/3 input size for first layer

3. **Activation Functions**
   - `relu`: Default, works well for most cases
   - `tanh`: Alternative for hidden layers
   - `sigmoid`: Output layer (binary classification)
   - `softmax`: Output layer (multiclass)

4. **Dropout**
   - Regularization technique
   - Prevents overfitting
   - Recommended: 0.2 - 0.5

**Example Advanced Architecture:**
```
Bot: Design Neural Network

Number of input features: 50 (auto-detected)
Number of output classes: 3 (multiclass)

Hidden Layer 1 - Neurons (16-512, default 64):
User: 256

Hidden Layer 1 - Activation (relu/tanh, default relu):
User: relu

Hidden Layer 1 - Dropout (0.0-0.5, default 0.2):
User: 0.3

Add another hidden layer? (yes/no):
User: yes

Hidden Layer 2 - Neurons:
User: 128

Hidden Layer 2 - Activation:
User: relu

Hidden Layer 2 - Dropout:
User: 0.3

Add another hidden layer? (yes/no):
User: yes

Hidden Layer 3 - Neurons:
User: 64

Hidden Layer 3 - Activation:
User: relu

Hidden Layer 3 - Dropout:
User: 0.2

Add another hidden layer? (yes/no):
User: no

Output Layer (auto-configured):
  - Neurons: 3 (multiclass)
  - Activation: softmax

Final Architecture:
  Input(50) → Dense(256, relu, dropout=0.3) → Dense(128, relu, dropout=0.3) → Dense(64, relu, dropout=0.2) → Output(3, softmax)

Total Parameters: ~50,000
Estimated training time: 20-25 minutes
Estimated cost: $0.26 (A100 40GB GPU)

Confirm? (yes/no/back)
```

---

### 9.2 Using RunPod Spot Instances

**What are Spot Instances?**
- Unused GPU capacity sold at discounted rates (50-70% off)
- Subject to reclamation if demand increases (rare)
- RunPod Community Cloud uses spot pricing by default

**Spot vs On-Demand Pricing:**

| GPU Type | On-Demand | Spot (Community) | Savings |
|----------|-----------|------------------|---------|
| RTX A5000 | $0.49/hr | $0.29/hr | 41% |
| RTX A40 | $0.69/hr | $0.39/hr | 43% |
| A100 40GB | $1.39/hr | $0.79/hr | 43% |
| A100 80GB | $2.09/hr | $1.19/hr | 43% |

**Interruption Handling:**

Bot automatically handles spot interruptions:

```
⚠️ Spot Instance Interruption Warning

Instance ID: pod-xyz123
Time Remaining: 120 seconds

AWS is reclaiming this Spot instance. The system will:
1. Attempt to save partial progress to storage
2. Automatically retry training with new instance
3. Send you an update once training completes

No action needed from you. Training will continue automatically.
```

**When to Use Spot:**
- All training jobs (default for RunPod Community Cloud)
- Cost savings important
- Training can tolerate occasional restarts

**When to Use On-Demand:**
- Critical deadlines
- Cannot tolerate interruptions
- Very long training jobs (>6 hours)

**Enable On-Demand (RunPod Secure Cloud):**

In `config.yaml`:
```yaml
cloud:
  runpod:
    cloud_type: "SECURE"  # On-demand pricing
```

---

### 9.3 Batch Training Workflows

Train multiple models in a single session to save on pod initialization costs.

**Workflow:**

1. **Start batch training:**
   ```
   User: /train_batch
   Bot: Batch Training Mode
        How many models to train? (2-10):
   User: 3
   ```

2. **Configure each model:**
   ```
   Bot: Model 1/3 - Upload dataset or provide path
   User: /data/housing.csv

   Bot: Select model type
   User: Linear Regression

   Bot: ✅ Model 1 configured

   Bot: Model 2/3 - Upload dataset or provide path
   User: /data/housing.csv (reuse dataset)

   Bot: Select model type
   User: Random Forest

   Bot: ✅ Model 2 configured

   Bot: Model 3/3 - Upload dataset or provide path
   User: /data/housing.csv (reuse dataset)

   Bot: Select model type
   User: Gradient Boosting

   Bot: ✅ Model 3 configured
   ```

3. **Launch batch training:**
   ```
   Bot: Batch Training Summary
        - 3 models configured
        - Dataset: housing.csv (shared)
        - GPU: NVIDIA A100 40GB (recommended)
        - Estimated time: 25 minutes total
        - Estimated cost: $0.33

   Launch batch training? (yes/no)
   User: yes
   ```

4. **Monitor progress:**
   ```
   🚀 Batch Training Started

   Pod ID: pod-batch-xyz123

   📊 Model 1/3: Linear Regression - Training...
   ✅ Model 1/3: Complete (R²=0.72, Time=2min, Cost=$0.03)

   📊 Model 2/3: Random Forest - Training...
   ✅ Model 2/3: Complete (R²=0.85, Time=8min, Cost=$0.11)

   📊 Model 3/3: Gradient Boosting - Training...
   ✅ Model 3/3: Complete (R²=0.88, Time=15min, Cost=$0.20)

   🎉 Batch Training Complete!

   Total time: 25 minutes
   Total cost: $0.34
   Savings vs individual jobs: $0.12 (26%)
   ```

**Cost Savings:**
- Individual jobs: 3 × (pod initialization + training) = ~$0.46
- Batch job: 1 × pod initialization + 3 × training = ~$0.34
- **Savings: ~26%**

---

### 9.4 Model Comparison and Selection

**Automated Model Comparison:**

Train multiple models and compare results:

```
User: /compare_models

Bot: Model Comparison Mode
     Dataset: housing.csv

     Select models to compare (comma-separated):
     1. Linear Regression
     2. Ridge Regression
     3. Random Forest
     4. Gradient Boosting
     5. MLP Regression

User: 1,3,4

Bot: Training 3 models for comparison...

[Training completes]

📊 Model Comparison Results

| Model | R² | RMSE | MAE | Time | Cost |
|-------|-----|------|-----|------|------|
| Linear Regression | 0.72 | 4.32 | 3.21 | 2min | $0.03 |
| Random Forest | 0.85 | 3.15 | 2.34 | 8min | $0.11 |
| Gradient Boosting | 0.88 | 2.89 | 2.12 | 15min | $0.20 |

🏆 Best Model: Gradient Boosting
   - Highest R² (0.88)
   - Lowest RMSE (2.89)
   - Worth the extra cost? Trade-off: +13% accuracy for +$0.17

Recommended: Gradient Boosting for production, Random Forest for cost-efficiency

Save which model? (1-3 or all)
```

---

### 9.5 Model Deployment Preparation

**Export Model for Production:**

After training, export model for deployment:

```
User: /export_model model_7715560927_random_forest_regression_20251108_143022

Bot: Export Options

1️⃣ Download as ZIP
   - model.pkl (trained model)
   - scaler.pkl (feature scaler)
   - metadata.json (config)
   - requirements.txt (dependencies)

2️⃣ Deploy to RunPod Serverless
   - Auto-creates inference endpoint
   - REST API for predictions
   - Auto-scaling
   - Cost: $0.0001/second (only during inference)

3️⃣ Deploy to AWS Lambda
   - Serverless inference
   - Pay per request
   - Cost: $0.0000002/request

Choose option (1-3):
User: 2

Bot: 🚀 Deploying to RunPod Serverless...

[30 seconds later]

✅ Deployment Complete!

Endpoint ID: abc123xyz
Endpoint URL: https://api.runpod.ai/v2/abc123xyz/run

Test your model:
curl -X POST https://api.runpod.ai/v2/abc123xyz/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"features": [1.5, 2.0, 3.5, ...]}}'

Response:
{"prediction": 245.67, "confidence": 0.95}

Monthly cost estimate: $0.50 (based on 5000 requests/month)
```

---

### 9.6 Monitoring and Logging

**Access Full Training Logs:**

```
User: /logs model_7715560927_random_forest_regression_20251108_143022

Bot: Training Logs

📁 Log Files:
  - stdout.log (training output)
  - stderr.log (error output)
  - metrics.json (training metrics)
  - system.log (GPU/CPU usage)

Download all logs? (yes/no)
User: yes

[Bot sends ZIP file with all logs]
```

**GPU Utilization Report:**

```
User: /gpu_stats model_7715560927_random_forest_regression_20251108_143022

Bot: GPU Utilization Report

📊 NVIDIA A100 40GB

Average GPU Utilization: 78%
Peak GPU Utilization: 95%
Average VRAM Usage: 12.3 GB (31%)
Peak VRAM Usage: 18.7 GB (47%)

Training Efficiency: Good ✅
  - High GPU utilization (>75%)
  - VRAM not bottleneck (<80%)
  - No memory overflow warnings

Optimization Suggestions:
  - Current GPU appropriate for this job
  - Could handle 2x larger dataset
  - Consider batch training for cost savings
```

---

## Conclusion

This guide covered:

1. **Introduction**: What cloud training is and why to use it
2. **Prerequisites**: RunPod/AWS setup and credentials
3. **Getting Started**: First training job walkthrough
4. **Data Sources**: Telegram upload, local paths, RunPod URI, S3 URI
5. **Model Selection**: All 13 models with GPU requirements and costs
6. **Workflow States**: Complete state flow with expected prompts
7. **Cost Management**: Estimation, tracking, optimization, budgets
8. **Troubleshooting**: Common errors and debugging techniques
9. **Advanced Topics**: Custom hyperparameters, spot instances, batch training

**Next Steps:**

- Try your first cloud training job with a small dataset
- Experiment with different models to understand cost/accuracy trade-offs
- Join the community for support and best practices

**Additional Resources:**

- [RunPod Documentation](https://docs.runpod.io)
- [AWS ML Documentation](https://docs.aws.amazon.com/machine-learning/)
- [Model Selection Guide](MODEL_SELECTION_GUIDE.md)
- [Cost Optimization Best Practices](COST_OPTIMIZATION.md)

**Support:**

- GitHub Issues: https://github.com/your-repo/issues
- Discord: [invite link]
- Email: support@yourproject.com

---

**Document Metadata:**
- Sections: 9 major sections, 40+ subsections
- Word Count: ~12,500 words
- Target Reading Time: 45-60 minutes
- Last Updated: 2025-11-08
- Version: 1.0
