# Statistical Modeling Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests/)
[![Cloud Ready](https://img.shields.io/badge/cloud-AWS%20%7C%20RunPod-blue.svg)](./docs/)

A conversational AI-powered Telegram bot for statistical analysis and machine learning. Train models through natural language, deploy to the cloud, and generate predictions at scale—all via simple chat commands.

## Table of Contents

1. [Features](#features)
2. [Cloud Training](#cloud-training)
3. [Cloud Prediction](#cloud-prediction)
4. [Quick Start](#quick-start)
5. [Cloud Provider Comparison](#cloud-provider-comparison)
6. [Architecture](#architecture)
7. [Documentation](#documentation)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [License](#license)

---

## Features

### Core Capabilities

- **Natural Language Interface**: Train models and run analyses using conversational commands via Telegram
- **13 ML Model Types**: Complete toolkit from linear regression to deep neural networks
- **Statistical Analysis**: Descriptive stats, correlation matrices, hypothesis testing, distribution analysis
- **Dual Execution Modes**: Local CPU training or cloud GPU-accelerated training
- **Multi-Source Data Loading**: Telegram uploads, local paths, S3/cloud storage, HTTP URLs
- **Auto-Schema Detection**: ML-powered target and feature suggestions with confidence scoring
- **Model Persistence**: Save, version, and reuse trained models with full metadata tracking
- **Result Visualization**: Inline tables, CSV downloads, performance metrics, feature importance
- **Cost Transparency**: Real-time cost estimates and tracking for cloud operations

### Statistical Analysis Tools

- **Descriptive Statistics**: Mean, median, mode, standard deviation, quartiles, skewness, kurtosis
- **Correlation Analysis**: Pearson, Spearman, Kendall correlation with p-values
- **Hypothesis Testing**: t-tests, ANOVA, chi-square tests with interpretation
- **Distribution Analysis**: Normality tests, Q-Q plots, distribution fitting
- **Data Quality Checks**: Missing value detection, outlier identification, data type validation

---

## Cloud Training

Train machine learning models on GPU-accelerated cloud infrastructure (AWS EC2 or RunPod Pods) for faster training, larger datasets, and scalable compute.

### Key Features

- **GPU Acceleration**: 5-50x faster training for neural networks and gradient boosting models
- **Scalable Resources**: 24GB to 80GB VRAM GPUs, handle datasets up to 100GB+
- **Pay-Per-Second Billing**: Only pay for actual training time (typical cost: $0.01 - $0.50 per job)
- **Spot Instance Support**: 50-70% cost savings with automatic failover to on-demand
- **Real-Time Monitoring**: Live training logs streamed to Telegram with progress updates
- **Multi-Source Data**: Upload via Telegram, local file paths, S3/RunPod URIs, HTTP URLs
- **Auto-Schema Detection**: ML-powered target/feature suggestions with task type classification

### Supported Model Types (13 Total)

**Regression Models (5)**
- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet (L1 + L2)
- Polynomial Regression

**Classification Models (6)**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- SVM (Support Vector Machine)
- Naive Bayes Classifier

**Neural Networks (2)**
- MLP Regression (Multi-Layer Perceptron)
- MLP Classification (Multi-Layer Perceptron)

### Data Source Options

1. **Telegram Upload** - Send CSV files directly (up to 100MB)
2. **Local File Path** - Reference server-side datasets (up to 10GB)
3. **RunPod Storage URI** - Access pre-uploaded data (`runpod://volume_id/path`)
4. **S3 URI** - Load from AWS S3 (`s3://bucket/path`)

### Cost Examples

| Dataset Size | Model Type | GPU Type | Training Time | Estimated Cost |
|--------------|------------|----------|---------------|----------------|
| 4KB (Iris) | Random Forest | RTX A5000 | 2 minutes | $0.01 |
| 50KB (Housing) | Gradient Boosting | RTX A40 | 5 minutes | $0.03 |
| 5MB (Census) | MLP Neural Net | A100 40GB | 15 minutes | $0.20 |
| 50MB (ImageNet) | Deep MLP | A100 80GB | 30 minutes | $0.60 |

**Typical Monthly Costs:**
- Light Usage (10 jobs/month): $0.50 - $1.20/month
- Moderate Usage (50 jobs/month): $8.60 - $10.70/month
- Heavy Usage (200 jobs/month): $24.00 - $28.20/month

### Workflow Example

```
User: /train
Bot: 🌩️ Training Location - Choose: Local or Cloud
User: Cloud
Bot: ☁️ Cloud Dataset Input - Upload CSV or provide path
User: [uploads housing_data.csv]
Bot: ✅ Dataset Loaded (506 rows, 14 columns)
     🎯 Detected Target: price (continuous)
     📐 Suggested Features: 13 numeric features
     🤖 Recommended Task: Regression
     Accept this configuration? (yes/no)
User: yes
Bot: 📚 Model Categories: 1) Regression 2) Classification 3) Neural Networks
User: 1
Bot: 🎯 Regression Models: 1) Linear 2) Ridge 3) Lasso 4) ElasticNet 5) Polynomial
User: 3
Bot: ☁️ Cloud Training Configuration
     📊 Dataset: 52.3 KB
     🎯 GPU Type: NVIDIA RTX A5000 (24GB VRAM)
     ⏱️ Estimated Time: ~3 minutes
     💰 Estimated Cost: $0.015
     Ready to launch cloud training?
User: yes
Bot: 🚀 Cloud Training Launched (Pod ID: pod-xyz123)
     📝 [00:00:05] Loading dataset...
     📝 [00:00:10] Preprocessing features...
     📝 [00:00:15] Training Random Forest...
     🏋️ Training Progress: 50% - Trees 1-50: Complete
     🏋️ Training Progress: 100% - Trees 51-100: Complete
     ✅ Cloud Training Complete!
     🎯 Model ID: model_7715560927_random_forest_regression_20251108_143022
     📊 Metrics: R²=0.823, RMSE=3.457, MAE=2.123
     ⏱️ Training Time: 2.9 minutes
     💰 Actual Cost: $0.014
```

See [Cloud Training User Guide](./docs/CLOUD_TRAINING_GUIDE.md) for complete documentation.

---

## Cloud Prediction

Generate predictions using trained models via serverless cloud infrastructure (AWS Lambda or RunPod Serverless) with automatic scaling and cost optimization.

### Key Features

- **Serverless Execution**: No idle costs, pay only for actual prediction time
- **Automatic Scaling**: Handle 1 to 1 million predictions seamlessly
- **Multiple Data Sources**: Telegram upload, local paths, S3 URIs
- **Performance Tiers**: Small datasets (<1K rows) in <5 seconds, large datasets batch processed
- **Result Formats**: Summary statistics, inline tables (<100 rows), full CSV downloads
- **Schema Validation**: Automatic feature matching against training schema with error reporting

### Performance Guidance

| Dataset Size | Provider | Execution Time | Cost per Batch |
|--------------|----------|----------------|----------------|
| <1,000 rows | AWS Lambda (sync) | <5 seconds | ~$0.001 |
| 1K-10K rows | AWS Lambda (batched) | 5-30 seconds | ~$0.01 |
| 10K-100K rows | AWS Lambda (async) | 30s-5 minutes | $0.05-$0.10 |
| >100K rows | RunPod Serverless (GPU) | 5-30 minutes | $0.10-$0.50+ |

### Result Formats

**Summary Statistics** (Always Included)
```
✅ Prediction Complete

Summary Statistics:
• Mean: 245.67
• Min: 123.45
• Max: 456.78
• Std Dev: 45.23

Total predictions: 5,000 rows
```

**Inline Table Preview** (<100 rows)
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

**CSV Download** (All Sizes)
- Original features preserved
- Prediction column appended
- Timestamped filename: `predictions_model_12345_random_forest_20251108_143022.csv`

See [Cloud Prediction User Guide](./docs/CLOUD_PREDICTION_GUIDE.md) for complete documentation.

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Telegram Bot Token ([create via BotFather](https://core.telegram.org/bots#6-botfather))
- Anthropic API Key ([get from console](https://console.anthropic.com/))
- (Optional) AWS Account with configured credentials
- (Optional) RunPod API Key ([sign up](https://www.runpod.io/))

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/statistical-modeling-agent.git
cd statistical-modeling-agent
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
```bash
cp .env.example .env
# Edit .env with your credentials:
# - TELEGRAM_BOT_TOKEN=your_telegram_bot_token
# - ANTHROPIC_API_KEY=your_anthropic_api_key
# (Optional cloud credentials - see Cloud Setup section)
```

5. **Run the Bot**
```bash
python src/bot/telegram_bot.py
```

### First Training Job (Local)

```
1. Start chat with your bot on Telegram
2. Send: /train
3. Bot: "Training Location - Choose: Local or Cloud"
4. Reply: "Local"
5. Upload a CSV file (e.g., housing_data.csv)
6. Follow the interactive prompts to select target/features/model
7. Receive trained model with metrics and download link
```

### First Cloud Training Job

**Prerequisites:**
- AWS credentials configured OR RunPod API key set
- See [Cloud Setup](#cloud-setup) section below

```
1. Send: /train
2. Reply: "Cloud"
3. Upload dataset or provide path/URI
4. Accept auto-detected schema or customize
5. Select model type (e.g., Random Forest)
6. Confirm GPU selection and cost estimate
7. Monitor real-time training logs
8. Receive model with metrics and cost summary
```

### Cloud Setup

#### AWS Configuration

1. **Install AWS CLI**
```bash
pip install awscli
aws configure
```

2. **Set Environment Variables**
```bash
# Add to .env file:
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
```

3. **Update Configuration**
```yaml
# config/config.yaml
cloud:
  enabled: true
  provider: "aws"
  s3:
    bucket: "ml-agent-data-{your-account-id}"
```

4. **Deploy Infrastructure** (see [Deployment Guide](./docs/DEPLOYMENT_GUIDE.md))

#### RunPod Configuration

1. **Create Account and API Key**
   - Sign up at [runpod.io](https://www.runpod.io)
   - Navigate to Settings → API Keys
   - Generate new API key

2. **Create Network Volume** (Optional but Recommended)
   - Go to Storage → Network Volumes
   - Click "Create Network Volume"
   - Region: `us-east-1`, Size: 50-100GB
   - Note the Volume ID (e.g., `v3zskt9gvb`)

3. **Set Environment Variables**
```bash
# Add to .env file:
RUNPOD_API_KEY=runpod-api-xxxxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_NETWORK_VOLUME_ID=v3zskt9gvb  # Optional
```

4. **Update Configuration**
```yaml
# config/config.yaml
cloud:
  enabled: true
  provider: "runpod"
  runpod:
    default_gpu_type: "NVIDIA RTX A5000"
    cloud_type: "COMMUNITY"  # Spot pricing
```

5. **Add Credits**
   - Minimum: $5 USD
   - Recommended: $20-50 for experimentation

---

## Cloud Provider Comparison

| Feature | AWS EC2 + Lambda | RunPod Pods + Serverless | Recommendation |
|---------|------------------|--------------------------|----------------|
| **GPU Availability** | Limited (requires quota) | High (community cloud) | RunPod for quick start |
| **Spot Pricing** | Yes (50-70% savings) | Yes (default in community) | Both excellent |
| **Setup Complexity** | High (IAM, VPC, AMI) | Low (API key only) | RunPod easier |
| **GPU Options** | 4 types (P3, P4, G4, G5) | 10+ types (RTX, A100, etc.) | RunPod more options |
| **Cost (Training)** | $0.02-$0.60 per job | $0.01-$0.50 per job | RunPod slightly cheaper |
| **Cost (Prediction)** | $0.001-$0.10 per batch | $0.01-$0.20 per batch | AWS cheaper for predictions |
| **Data Transfer** | Free (within region) | Charged ($0.01/GB) | AWS better for large data |
| **Monitoring** | CloudWatch (detailed) | Basic logs | AWS better monitoring |
| **Reliability** | 99.99% SLA | 99.5% (community) | AWS more reliable |
| **Scalability** | High (Auto Scaling) | Manual pod management | AWS better at scale |
| **Storage** | S3 ($0.023/GB/month) | Network Volume ($0.07/GB/month) | AWS cheaper storage |

**Recommendations:**

- **Quick Experimentation**: RunPod (easier setup, good GPU availability)
- **Production Workloads**: AWS (better reliability, monitoring, and SLA)
- **Cost-Sensitive Projects**: RunPod for training, AWS for predictions
- **Large-Scale Deployments**: AWS (auto-scaling, enterprise features)
- **GPU-Heavy Workloads**: RunPod (more GPU options, faster provisioning)

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         TELEGRAM BOT                             │
│  (Natural Language Interface - User Interaction Layer)          │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│                     WORKFLOW ORCHESTRATOR                       │
│  • Data Source Detection (Telegram/Local/S3/URL)               │
│  • Training Location Routing (Local vs Cloud)                  │
│  • State Management & Transitions                              │
│  • Schema Auto-Detection & Validation                          │
└────┬────────────────────────────────────────────────┬──────────┘
     │                                                 │
     ▼                                                 ▼
┌──────────────────────────┐              ┌────────────────────────────┐
│   LOCAL TRAINING PATH    │              │    CLOUD TRAINING PATH     │
│  • CPU-based execution   │              │  • GPU-accelerated exec    │
│  • In-memory processing  │              │  • Cloud provider factory  │
│  • Direct file access    │              │  • Cost tracking & limits  │
│  • Free (no cloud cost)  │              │  • Real-time monitoring    │
└────────┬─────────────────┘              └──────────┬─────────────────┘
         │                                           │
         ▼                                           ▼
┌──────────────────────────┐              ┌────────────────────────────┐
│    ML ENGINE (Local)     │              │  CLOUD PROVIDER FACTORY    │
│  • 13 Model Types        │              │  • AWS (EC2 + Lambda)      │
│  • scikit-learn          │              │  • RunPod (Pods + Serverless)
│  • Preprocessing         │              │  • Auto-detection          │
│  • Model persistence     │              │  • Fallback logic          │
└────────┬─────────────────┘              └──────────┬─────────────────┘
         │                                           │
         │                                           ▼
         │                              ┌────────────────────────────┐
         │                              │   CLOUD INFRASTRUCTURE     │
         │                              │  ┌──────────────────────┐  │
         │                              │  │  AWS EC2 Training    │  │
         │                              │  │  • Spot Instances    │  │
         │                              │  │  • CloudWatch Logs   │  │
         │                              │  │  • S3 Storage        │  │
         │                              │  └──────────────────────┘  │
         │                              │  ┌──────────────────────┐  │
         │                              │  │  RunPod Pods         │  │
         │                              │  │  • GPU Pods          │  │
         │                              │  │  • Real-time Logs    │  │
         │                              │  │  • Network Storage   │  │
         │                              │  └──────────────────────┘  │
         │                              └────────────────────────────┘
         │                                           │
         ▼                                           ▼
┌────────────────────────────────────────────────────────────────┐
│                    MODEL STORAGE & RETRIEVAL                    │
│  • Local: ./models/user_{id}/model_{id}/                       │
│  • AWS: s3://bucket/models/user_{id}/model_{id}/               │
│  • RunPod: volume://models/user_{id}/model_{id}/               │
│  • Metadata: JSON (metrics, hyperparams, schema)               │
│  • Artifacts: model.pkl, preprocessor.pkl, training_log.txt    │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                    PREDICTION WORKFLOWS                         │
│  • Local: In-process scikit-learn                              │
│  • Cloud (AWS): Lambda functions (serverless)                  │
│  • Cloud (RunPod): Serverless endpoints (GPU)                  │
│  • Auto-scaling based on dataset size                          │
│  • Result formatting and delivery via Telegram                 │
└────────────────────────────────────────────────────────────────┘
```

### Key Components

**Workflow Orchestrator** (`src/bot/workflow_handlers.py`)
- Detects data source from user input (file, path, URI, URL)
- Routes to local or cloud training based on user selection
- Manages state transitions and validation
- Implements auto-schema detection with ML confidence scoring

**Provider Factory** (`src/cloud/provider_factory.py`)
- Auto-detects available cloud providers (checks API keys)
- Implements fallback logic: RunPod → AWS → Local
- Performs health checks before workflow start
- Returns provider-specific implementation via interface

**ML Engine** (`src/engines/ml_engine.py`)
- Unified interface for local and cloud training
- Implements all 13 model types with identical logic
- Handles preprocessing (scaling, missing values, encoding)
- Manages model persistence and metadata

**State Manager** (`src/core/state_manager.py`)
- Transaction-safe state updates with rollback
- Workflow recovery after bot restart
- State history tracking for analytics
- Validation of state transitions

**Cloud Managers**
- `ec2_manager.py`: AWS training orchestration
- `s3_manager.py`: AWS storage operations
- `lambda_manager.py`: AWS prediction functions
- `runpod_pod_manager.py`: RunPod training pods
- `runpod_storage_manager.py`: RunPod storage
- `runpod_serverless_manager.py`: RunPod predictions

**Cost Tracker** (`src/cloud/cost_tracker.py`)
- Real-time cost estimation before training
- Live cost tracking during execution
- Cost breakdown by component (compute, storage, transfer)
- Budget alerts and limit enforcement

---

## Documentation

### User Guides

- [Cloud Training User Guide](./docs/CLOUD_TRAINING_GUIDE.md) - Complete cloud training workflow (12,500 words)
- [Cloud Prediction User Guide](./docs/CLOUD_PREDICTION_GUIDE.md) - Cloud prediction workflows (2,449 words)
- [RunPod Serverless Deployment Guide](./docs/RUNPOD_SERVERLESS_DEPLOYMENT_GUIDE.md) - RunPod setup

### Developer Documentation

- [CLAUDE.md](./CLAUDE.md) - Project overview and coding standards (1,813 lines)
- [Architecture Documentation](./docs/CLOUD_ARCHITECTURE.md) - System design (coming soon)
- [API Reference](./docs/API_REFERENCE.md) - Cloud interfaces (coming soon)
- [Deployment Guide](./docs/DEPLOYMENT_GUIDE.md) - AWS/RunPod deployment (coming soon)

### Configuration

- [config.yaml](./config/config.yaml) - Main configuration file
- [.env.example](./.env.example) - Environment variables template

---

## Testing

### Test Coverage

- **Unit Tests**: 600+ tests covering all modules
- **Integration Tests**: 82 tests for end-to-end workflows
- **Cloud Tests**: Complete coverage for AWS and RunPod providers
- **Test Isolation**: All external APIs mocked (boto3, RunPod SDK)

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run cloud tests only
pytest tests/ -k cloud

# Run specific test file
pytest tests/unit/test_ml_engine.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "cloud_training or cloud_prediction"
```

### Test Structure

```
tests/
├── unit/                               # Unit tests (isolated components)
│   ├── test_cloud_training_handlers.py
│   ├── test_cloud_prediction_handlers.py
│   ├── test_provider_factory.py
│   ├── test_state_manager.py
│   ├── test_ml_engine.py
│   └── ...
├── integration/                        # Integration tests (full workflows)
│   ├── test_cloud_training_e2e.py      # 27 tests
│   ├── test_cloud_prediction_e2e.py    # 24 tests
│   ├── test_all_model_types_cloud.py   # 31 tests
│   └── test_cloud_workflows.py
└── conftest.py                         # Shared fixtures and mocks
```

### Key Test Fixtures

- `mock_boto3_client`: Mocked AWS SDK (S3, EC2, Lambda)
- `mock_runpod_client`: Mocked RunPod API
- `sample_datasets`: Small, medium, large test datasets
- `trained_models`: Pre-trained model fixtures
- `state_db`: In-memory state persistence for testing

---

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`pip install -r requirements-dev.txt`)
4. Make your changes
5. Run tests (`pytest`)
6. Run linters (`black src/ tests/ && flake8 src/ tests/`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Code Standards

- **Python Version**: 3.9+
- **Formatting**: Black (line length 88)
- **Linting**: Flake8, MyPy (strict mode)
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style for all modules, classes, functions
- **Test Coverage**: Minimum 85% for new code

### Project Structure

```
statistical-modeling-agent/
├── src/
│   ├── bot/                    # Telegram bot handlers
│   │   ├── handlers.py         # Main entry point
│   │   ├── workflow_handlers.py  # Workflow orchestration
│   │   ├── cloud_handlers/     # Cloud-specific handlers
│   │   └── ml_handlers/        # Local training handlers
│   ├── cloud/                  # Cloud provider integrations
│   │   ├── provider_factory.py
│   │   ├── ec2_manager.py
│   │   ├── runpod_pod_manager.py
│   │   └── cost_tracker.py
│   ├── core/                   # Core business logic
│   │   ├── state_manager.py
│   │   └── orchestrator.py
│   ├── engines/                # ML engines
│   │   ├── ml_engine.py
│   │   └── stats_engine.py
│   ├── processors/             # Data processing
│   │   └── data_loader.py
│   └── utils/                  # Utilities
│       ├── schema_detector.py
│       └── path_validator.py
├── tests/                      # Test suite
├── docs/                       # Documentation
├── config/                     # Configuration files
└── templates/                  # Training templates
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/statistical-modeling-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/statistical-modeling-agent/discussions)
- **Email**: support@yourproject.com

---

## Acknowledgments

- Built with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Powered by [Anthropic Claude](https://www.anthropic.com/) for natural language understanding
- ML framework: [scikit-learn](https://scikit-learn.org/)
- Cloud providers: [AWS](https://aws.amazon.com/) and [RunPod](https://www.runpod.io/)

---

**Last Updated**: 2025-11-08
**Version**: 2.0 (Cloud Training & Prediction Release)
**Contributors**: 1
**Stars**: Give us a star if you find this project useful!
