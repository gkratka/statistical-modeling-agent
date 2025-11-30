# Cloud Deployment Guide

**Version:** 1.0
**Last Updated:** 2025-11-08
**Target Audience:** DevOps Engineers, System Administrators

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [AWS Deployment](#2-aws-deployment)
3. [RunPod Deployment](#3-runpod-deployment)
4. [Environment Configuration](#4-environment-configuration)
5. [Monitoring Setup](#5-monitoring-setup)
6. [Cost Optimization](#6-cost-optimization)
7. [Security Best Practices](#7-security-best-practices)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Introduction

### Overview

This guide provides step-by-step instructions for deploying the Statistical Modeling Agent's cloud infrastructure. The system supports two cloud providers:

- **AWS** (EC2, S3, Lambda) - Enterprise-grade infrastructure with comprehensive service integration
- **RunPod** (GPU Pods, Network Volumes, Serverless) - Cost-effective GPU-accelerated training

### Prerequisites

**Required Tools:**
- AWS CLI v2+ or RunPod CLI (depending on provider)
- Python 3.9+
- Git
- Docker (for Lambda deployment)
- SSH client
- Text editor

**Required Access:**
- Cloud provider account with billing enabled
- Administrator privileges (for IAM role creation)
- API credentials (AWS access keys or RunPod API key)

**Estimated Time:**
- AWS: 45-60 minutes
- RunPod: 15-30 minutes

### Architecture Components

| Component | AWS Service | RunPod Equivalent | Purpose |
|-----------|-------------|-------------------|---------|
| **Storage** | S3 | Network Volumes | Dataset and model storage |
| **Training** | EC2 Spot Instances | GPU Pods | ML model training |
| **Prediction** | Lambda | Serverless Endpoints | Inference execution |
| **Logs** | CloudWatch | Pod Logs | Monitoring and debugging |

---

## 2. AWS Deployment

### 2.1 IAM Configuration

#### Create IAM User

1. **Navigate to IAM Console:**
   - Open AWS Console → Services → IAM

2. **Create User:**
   ```bash
   # Via AWS CLI
   aws iam create-user --user-name ml-agent-user

   # Create access key
   aws iam create-access-key --user-name ml-agent-user
   ```

3. **Save Credentials:**
   ```
   Access Key ID: AKIAIOSFODNN7EXAMPLE
   Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   ```

   **IMPORTANT:** Save these immediately - secret key shown only once.

#### Create IAM Policy

**File:** `iam-policy.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetObjectVersion",
        "s3:PutObjectAcl"
      ],
      "Resource": [
        "arn:aws:s3:::ml-agent-data-*",
        "arn:aws:s3:::ml-agent-data-*/*"
      ]
    },
    {
      "Sid": "EC2SpotAccess",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeSpotInstanceRequests",
        "ec2:RequestSpotInstances",
        "ec2:CancelSpotInstanceRequests",
        "ec2:DescribeImages",
        "ec2:DescribeKeyPairs",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeSubnets",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "LambdaAccess",
      "Effect": "Allow",
      "Action": [
        "lambda:InvokeFunction",
        "lambda:GetFunction",
        "lambda:UpdateFunctionCode",
        "lambda:CreateFunction",
        "lambda:DeleteFunction",
        "lambda:PublishLayerVersion"
      ],
      "Resource": "arn:aws:lambda:*:*:function:ml-agent-*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:GetLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/ml-agent/*"
    },
    {
      "Sid": "IAMPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/ml-agent-*"
    }
  ]
}
```

**Apply Policy:**
```bash
# Create policy
aws iam create-policy \
  --policy-name ml-agent-policy \
  --policy-document file://iam-policy.json

# Attach to user
aws iam attach-user-policy \
  --user-name ml-agent-user \
  --policy-arn arn:aws:iam::ACCOUNT_ID:policy/ml-agent-policy
```

#### Create IAM Role for EC2

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name ml-agent-ec2-role \
  --assume-role-policy-document file://trust-policy.json

# Attach S3 access policy
aws iam attach-role-policy \
  --role-name ml-agent-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name ml-agent-ec2-profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name ml-agent-ec2-profile \
  --role-name ml-agent-ec2-role
```

---

### 2.2 S3 Bucket Setup

#### Create S3 Bucket

```bash
# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create bucket (bucket names must be globally unique)
BUCKET_NAME="ml-agent-data-${ACCOUNT_ID}"

aws s3 mb s3://${BUCKET_NAME} --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_NAME} \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket ${BUCKET_NAME} \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

#### Configure Lifecycle Policy

**File:** `lifecycle-policy.json`

```json
{
  "Rules": [
    {
      "Id": "DeleteOldDatasets",
      "Prefix": "datasets/",
      "Status": "Enabled",
      "Expiration": {
        "Days": 90
      }
    },
    {
      "Id": "DeleteOldResults",
      "Prefix": "results/",
      "Status": "Enabled",
      "Expiration": {
        "Days": 30
      }
    },
    {
      "Id": "VersionCleanup",
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 7
      },
      "NoncurrentVersionTransitions": []
    }
  ]
}
```

**Apply Policy:**
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket ${BUCKET_NAME} \
  --lifecycle-configuration file://lifecycle-policy.json
```

#### Create Folder Structure

```bash
# Create standard prefixes
aws s3api put-object --bucket ${BUCKET_NAME} --key datasets/
aws s3api put-object --bucket ${BUCKET_NAME} --key models/
aws s3api put-object --bucket ${BUCKET_NAME} --key results/
aws s3api put-object --bucket ${BUCKET_NAME} --key checkpoints/

# Verify structure
aws s3 ls s3://${BUCKET_NAME}/
```

---

### 2.3 EC2 AMI Creation

#### Launch Base Instance

```bash
# Create key pair for SSH access
aws ec2 create-key-pair \
  --key-name ml-agent-key \
  --query 'KeyMaterial' \
  --output text > ml-agent-key.pem

chmod 400 ml-agent-key.pem

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name ml-agent-sg \
  --description "Security group for ML Agent training instances" \
  --query 'GroupId' \
  --output text)

# Allow SSH access (restrict to your IP in production)
aws ec2 authorize-security-group-ingress \
  --group-id ${SG_ID} \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Launch base instance (Deep Learning AMI)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --key-name ml-agent-key \
  --security-group-ids ${SG_ID} \
  --iam-instance-profile Name=ml-agent-ec2-profile \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: ${INSTANCE_ID}"

# Wait for instance to run
aws ec2 wait instance-running --instance-ids ${INSTANCE_ID}

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids ${INSTANCE_ID} \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Public IP: ${PUBLIC_IP}"
```

#### Install Dependencies

```bash
# SSH into instance
ssh -i ml-agent-key.pem ubuntu@${PUBLIC_IP}

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python dependencies
sudo apt-get install -y python3.9 python3-pip python3-venv

# Install ML libraries
pip3 install --upgrade pip
pip3 install \
  scikit-learn==1.3.0 \
  pandas==2.1.0 \
  numpy==1.24.3 \
  joblib==1.3.2 \
  xgboost==2.0.0 \
  lightgbm==4.0.0 \
  tensorflow==2.13.0 \
  boto3==1.28.0

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installations
python3 --version
pip3 list | grep scikit-learn
aws --version

# Clean up
sudo apt-get clean
rm -rf ~/.cache/pip
```

#### Create Training Script Template

```bash
# Create training script directory
sudo mkdir -p /opt/ml-agent/scripts
sudo chown ubuntu:ubuntu /opt/ml-agent/scripts

# Create base training script
cat > /opt/ml-agent/scripts/train_template.py <<'EOF'
#!/usr/bin/env python3
"""
ML Training Script Template
This script is populated by the cloud training manager.
"""
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import boto3

# Load configuration (injected by training manager)
config = json.loads(sys.argv[1])

# Download dataset from S3
s3 = boto3.client('s3')
dataset_bucket = config['dataset_bucket']
dataset_key = config['dataset_key']

# Load data
s3.download_file(dataset_bucket, dataset_key, '/tmp/dataset.csv')
df = pd.read_csv('/tmp/dataset.csv')

# Preprocessing
X = df[config['feature_columns']]
y = df[config['target_column']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (model class injected)
model = config['model_class'](**config['hyperparameters'])
model.fit(X_train_scaled, y_train)

# Evaluate
score = model.score(X_test_scaled, y_test)
print(f"Model Score: {score:.4f}")

# Save model
joblib.dump(model, '/tmp/model.pkl')
joblib.dump(scaler, '/tmp/scaler.pkl')

# Upload to S3
model_bucket = config['model_bucket']
model_prefix = config['model_prefix']

s3.upload_file('/tmp/model.pkl', model_bucket, f"{model_prefix}/model.pkl")
s3.upload_file('/tmp/scaler.pkl', model_bucket, f"{model_prefix}/scaler.pkl")

print("Training complete!")
EOF

chmod +x /opt/ml-agent/scripts/train_template.py
```

#### Create AMI

```bash
# Exit SSH session
exit

# Create AMI from instance
AMI_ID=$(aws ec2 create-image \
  --instance-id ${INSTANCE_ID} \
  --name "ml-agent-training-ami-$(date +%Y%m%d)" \
  --description "ML Agent Training AMI with Python 3.9 and ML libraries" \
  --query 'ImageId' \
  --output text)

echo "AMI ID: ${AMI_ID}"

# Wait for AMI to be available
aws ec2 wait image-available --image-ids ${AMI_ID}

echo "AMI ready: ${AMI_ID}"

# Terminate base instance
aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}

# Save AMI ID for configuration
echo "EC2_AMI_ID=${AMI_ID}" >> .env
```

---

### 2.4 Lambda Deployment

#### Create Lambda Layer (ML Dependencies)

```bash
# Create layer directory structure
mkdir -p lambda-layer/python/lib/python3.9/site-packages

cd lambda-layer/python/lib/python3.9/site-packages

# Install dependencies
pip3 install \
  --platform manylinux2014_x86_64 \
  --target . \
  --implementation cp \
  --python 3.9 \
  --only-binary=:all: \
  --upgrade \
  scikit-learn==1.3.0 \
  pandas==2.1.0 \
  numpy==1.24.3 \
  joblib==1.3.2 \
  boto3==1.28.0

cd ../../../..

# Create ZIP (must be <250MB)
zip -r lambda-layer.zip python/

# Verify size
ls -lh lambda-layer.zip
# If > 250MB, remove unnecessary files:
# - Test files: find python -name "tests" -type d -exec rm -rf {} +
# - Cache files: find python -name "__pycache__" -type d -exec rm -rf {} +

# Upload layer
LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name ml-agent-deps \
  --description "ML dependencies for predictions" \
  --zip-file fileb://lambda-layer.zip \
  --compatible-runtimes python3.9 \
  --query 'LayerVersionArn' \
  --output text)

echo "Layer ARN: ${LAYER_ARN}"
echo "LAMBDA_LAYER_ARN=${LAYER_ARN}" >> ../.env
```

#### Create Lambda Function

**File:** `lambda_function.py`

```python
import json
import joblib
import pandas as pd
import boto3
from io import BytesIO

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda handler for ML predictions.

    Event format:
    {
        "model_bucket": "ml-agent-data-123456",
        "model_key": "models/user_123/model_456/model.pkl",
        "scaler_key": "models/user_123/model_456/scaler.pkl",
        "data_bucket": "ml-agent-data-123456",
        "data_key": "datasets/user_123/prediction_data.csv",
        "output_key": "results/user_123/predictions.csv"
    }
    """
    try:
        # Load model
        model_obj = s3.get_object(
            Bucket=event['model_bucket'],
            Key=event['model_key']
        )
        model = joblib.load(BytesIO(model_obj['Body'].read()))

        # Load scaler
        scaler_obj = s3.get_object(
            Bucket=event['model_bucket'],
            Key=event['scaler_key']
        )
        scaler = joblib.load(BytesIO(scaler_obj['Body'].read()))

        # Load data
        data_obj = s3.get_object(
            Bucket=event['data_bucket'],
            Key=event['data_key']
        )
        df = pd.read_csv(BytesIO(data_obj['Body'].read()))

        # Preprocess
        X_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(X_scaled)

        # Add predictions to dataframe
        df['prediction'] = predictions

        # Save results
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)

        s3.put_object(
            Bucket=event['data_bucket'],
            Key=event['output_key'],
            Body=csv_buffer.getvalue()
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Predictions complete',
                'num_predictions': len(predictions),
                'output_key': event['output_key']
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
```

**Deploy Function:**

```bash
# Package function
zip lambda_function.zip lambda_function.py

# Create Lambda execution role
cat > lambda-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
  --role-name ml-agent-lambda-role \
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name ml-agent-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name ml-agent-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Get role ARN
LAMBDA_ROLE_ARN=$(aws iam get-role \
  --role-name ml-agent-lambda-role \
  --query 'Role.Arn' \
  --output text)

# Wait for role to propagate
sleep 10

# Create Lambda function
aws lambda create-function \
  --function-name ml-agent-prediction \
  --runtime python3.9 \
  --role ${LAMBDA_ROLE_ARN} \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 900 \
  --memory-size 3008 \
  --layers ${LAYER_ARN}

echo "Lambda function created: ml-agent-prediction"
```

---

## 3. RunPod Deployment

### 3.1 Account Setup

#### Create RunPod Account

1. **Sign Up:**
   - Visit: https://www.runpod.io
   - Sign up with email or GitHub
   - Verify email address

2. **Add Credits:**
   - Navigate to Billing → Add Credits
   - Minimum: $5 USD
   - Recommended: $20-50 for experimentation
   - Payment methods: Credit card, crypto, bank transfer

#### Generate API Key

1. **Navigate to Settings:**
   - Click profile icon → Settings
   - Select "API Keys" tab

2. **Create API Key:**
   - Click "Create API Key"
   - Name: "ml-agent-production"
   - Copy key immediately (starts with `runpod-api-...`)

3. **Save Securely:**
   ```bash
   # Add to .env file
   echo "RUNPOD_API_KEY=runpod-api-xxxxxxxxxxxxxxxxxxxxxxxxxx" >> .env

   # Verify
   grep RUNPOD_API_KEY .env
   ```

---

### 3.2 Network Volume Setup

#### Create Network Volume via Web UI

1. **Navigate to Storage:**
   - Dashboard → Storage → Network Volumes
   - Click "Create Network Volume"

2. **Configure Volume:**
   - **Name:** ml-agent-storage
   - **Region:** us-east-1 (recommended for speed)
   - **Size:** 100GB (minimum 50GB, expandable later)
   - **Type:** Network Volume (persistent storage)

3. **Note Volume ID:**
   - Format: `v3zskt9gvb` (8-character alphanumeric)
   - Copy and save to .env:
   ```bash
   echo "RUNPOD_NETWORK_VOLUME_ID=v3zskt9gvb" >> .env
   ```

#### Create Network Volume via CLI (Alternative)

```bash
# Install RunPodCTL
pip install runpodctl

# Configure CLI
runpodctl config

# Paste API key when prompted

# Create network volume
runpodctl create volume \
  --name ml-agent-storage \
  --size 100 \
  --region us-east-1

# List volumes to get ID
runpodctl list volumes
```

---

### 3.3 Storage Configuration

#### Initialize Directory Structure

**Option 1: Via RunPodCTL**

```bash
# Connect to volume
runpodctl attach volume v3zskt9gvb

# Create directory structure
mkdir -p /mnt/runpod-volume/datasets
mkdir -p /mnt/runpod-volume/models
mkdir -p /mnt/runpod-volume/results
mkdir -p /mnt/runpod-volume/checkpoints

# Set permissions
chmod -R 755 /mnt/runpod-volume

# Verify structure
ls -la /mnt/runpod-volume/
```

**Option 2: Via Pod**

```bash
# Create temporary pod with volume attached
runpodctl create pod \
  --name setup-pod \
  --image runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04 \
  --gpu-type "NVIDIA RTX A5000" \
  --volume v3zskt9gvb:/workspace

# SSH into pod
runpodctl ssh setup-pod

# Create directories
cd /workspace
mkdir -p datasets models results checkpoints

# Exit and terminate pod
exit
runpodctl stop pod setup-pod
```

#### Configure S3-Compatible Storage (Optional)

RunPod provides S3-compatible storage for network volumes:

```bash
# Get storage credentials from RunPod console:
# Settings → Network Volumes → Click volume → Storage Credentials

# Add to .env
echo "RUNPOD_STORAGE_ACCESS_KEY=your_access_key" >> .env
echo "RUNPOD_STORAGE_SECRET_KEY=your_secret_key" >> .env
echo "RUNPOD_STORAGE_ENDPOINT=https://storage.runpod.io" >> .env
```

**Test Connection:**

```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='https://storage.runpod.io',
    aws_access_key_id='your_access_key',
    aws_secret_access_key='your_secret_key'
)

# List buckets (network volumes appear as buckets)
response = s3.list_buckets()
print(response)
```

---

### 3.4 GPU Pod Configuration

#### Select GPU Type

| GPU Model | VRAM | Cost/Hour | Best For |
|-----------|------|-----------|----------|
| **NVIDIA RTX A5000** | 24GB | $0.29 | Small-medium datasets, most models |
| **NVIDIA RTX A40** | 48GB | $0.39 | Medium-large datasets, ensembles |
| **NVIDIA A100 PCIe 40GB** | 40GB | $0.79 | Neural networks, large datasets |
| **NVIDIA A100 PCIe 80GB** | 80GB | $1.19 | Very large models, batch processing |

#### Test Pod Creation

```bash
# Create test pod (auto-terminates after 1 hour)
runpodctl create pod \
  --name test-training \
  --image runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04 \
  --gpu-type "NVIDIA RTX A5000" \
  --volume v3zskt9gvb:/workspace \
  --cloud-type COMMUNITY \
  --min-download 100 \
  --min-upload 100

# Get pod ID
POD_ID=$(runpodctl list pods --filter name=test-training --format json | jq -r '.[0].id')

# Check status
runpodctl get pod ${POD_ID}

# SSH into pod
runpodctl ssh ${POD_ID}

# Verify GPU
nvidia-smi

# Verify volume mount
ls -la /workspace

# Exit and terminate
exit
runpodctl stop pod ${POD_ID}
```

---

### 3.5 Serverless Endpoint Setup (Optional)

For prediction workloads, RunPod Serverless provides auto-scaling inference:

#### Create Serverless Endpoint

1. **Navigate to Serverless:**
   - Dashboard → Serverless → Create Endpoint

2. **Configure Endpoint:**
   - **Name:** ml-agent-predictions
   - **GPU:** NVIDIA RTX A4000 (lowest cost for inference)
   - **Min Workers:** 0 (auto-scale from zero)
   - **Max Workers:** 3 (limit concurrent requests)
   - **Idle Timeout:** 30 seconds
   - **Docker Image:** runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

3. **Add Handler Code:**

```python
import runpod
import joblib
import pandas as pd
from io import BytesIO

def handler(event):
    """
    RunPod serverless handler for ML predictions.

    Event format:
    {
        "input": {
            "model_path": "/workspace/models/user_123/model.pkl",
            "scaler_path": "/workspace/models/user_123/scaler.pkl",
            "data_path": "/workspace/datasets/user_123/data.csv",
            "output_path": "/workspace/results/user_123/predictions.csv"
        }
    }
    """
    try:
        input_data = event["input"]

        # Load model and scaler
        model = joblib.load(input_data["model_path"])
        scaler = joblib.load(input_data["scaler_path"])

        # Load data
        df = pd.read_csv(input_data["data_path"])

        # Predict
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)

        # Save results
        df['prediction'] = predictions
        df.to_csv(input_data["output_path"], index=False)

        return {
            "success": True,
            "num_predictions": len(predictions),
            "output_path": input_data["output_path"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})
```

4. **Deploy and Get Endpoint ID:**
   - Click "Deploy"
   - Copy Endpoint ID (format: `abc123xyz`)
   - Add to .env:
   ```bash
   echo "RUNPOD_ENDPOINT_ID=abc123xyz" >> .env
   ```

---

## 4. Environment Configuration

### 4.1 Environment Variables

#### AWS Configuration

```bash
# Copy example
cp .env.example .env

# Edit .env with your values
nano .env
```

**Required AWS Variables:**

```bash
# Cloud Provider
CLOUD_PROVIDER=aws

# AWS Credentials
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=us-east-1

# S3 Configuration
S3_BUCKET=ml-agent-data-123456789012
S3_DATA_PREFIX=datasets
S3_MODELS_PREFIX=models
S3_RESULTS_PREFIX=results
S3_LIFECYCLE_DAYS=90

# EC2 Configuration
EC2_INSTANCE_TYPE=p3.2xlarge
EC2_SPOT_MAX_PRICE=0.5
EC2_AMI_ID=ami-0c55b159cbfafe1f0
EC2_KEY_NAME=ml-agent-key
EC2_SECURITY_GROUP=sg-0123456789abcdef0

# Lambda Configuration
LAMBDA_FUNCTION_NAME=ml-agent-prediction
LAMBDA_MEMORY_MB=3008
LAMBDA_TIMEOUT_SECONDS=900
LAMBDA_LAYER_ARN=arn:aws:lambda:us-east-1:123456789012:layer:ml-agent-deps:1

# Cost Limits
MAX_TRAINING_COST_DOLLARS=10.0
MAX_PREDICTION_COST_DOLLARS=1.0
COST_WARNING_THRESHOLD=0.8

# Security (Optional)
KMS_KEY_ID=arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012
IAM_ROLE_ARN=arn:aws:iam::123456789012:role/ml-agent-ec2-role
```

#### RunPod Configuration

```bash
# Copy example
cp .env.example .env

# Edit .env
nano .env
```

**Required RunPod Variables:**

```bash
# Cloud Provider
CLOUD_PROVIDER=runpod

# RunPod Credentials
RUNPOD_API_KEY=runpod-api-xxxxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_NETWORK_VOLUME_ID=v3zskt9gvb

# RunPod Storage (S3-compatible)
RUNPOD_STORAGE_ACCESS_KEY=your_storage_access_key
RUNPOD_STORAGE_SECRET_KEY=your_storage_secret_key
RUNPOD_STORAGE_ENDPOINT=https://storage.runpod.io

# RunPod Configuration
RUNPOD_DEFAULT_GPU_TYPE=NVIDIA RTX A5000
RUNPOD_CLOUD_TYPE=COMMUNITY
RUNPOD_DATA_PREFIX=datasets
RUNPOD_MODELS_PREFIX=models
RUNPOD_RESULTS_PREFIX=results

# RunPod Serverless (Optional)
RUNPOD_ENDPOINT_ID=abc123xyz

# Cost Limits
MAX_TRAINING_COST_DOLLARS=10.0
MAX_PREDICTION_COST_DOLLARS=1.0
COST_WARNING_THRESHOLD=0.8
```

---

### 4.2 Configuration File

#### Update config.yaml

```bash
# Open configuration file
nano config/config.yaml
```

**AWS Configuration Section:**

```yaml
cloud:
  provider: "aws"
  enabled: true

  aws:
    region: "us-east-1"

  s3:
    bucket: "ml-agent-data-123456789012"
    data_prefix: "datasets"
    models_prefix: "models"
    results_prefix: "results"
    lifecycle_days: 90

  ec2:
    instance_type: "p3.2xlarge"
    spot_max_price: 0.5
    ami_id: "ami-0c55b159cbfafe1f0"
    key_name: "ml-agent-key"
    security_group: "sg-0123456789abcdef0"

  lambda:
    function_name: "ml-agent-prediction"
    memory_mb: 3008
    timeout_seconds: 900
    layer_arn: "arn:aws:lambda:us-east-1:123456789012:layer:ml-agent-deps:1"

  cost_limits:
    max_training_cost_dollars: 10.0
    max_prediction_cost_dollars: 1.0
    cost_warning_threshold: 0.8

  security:
    kms_key_id: ""  # Optional
    iam_role_arn: "arn:aws:iam::123456789012:role/ml-agent-ec2-role"
```

**RunPod Configuration Section:**

```yaml
cloud:
  provider: "runpod"
  enabled: true

  runpod:
    api_key: ${RUNPOD_API_KEY}
    storage_endpoint: "https://storage.runpod.io"
    network_volume_id: ${RUNPOD_NETWORK_VOLUME_ID}

    default_gpu_type: "NVIDIA RTX A5000"
    cloud_type: "COMMUNITY"

    storage_access_key: ${RUNPOD_STORAGE_ACCESS_KEY}
    storage_secret_key: ${RUNPOD_STORAGE_SECRET_KEY}
    data_prefix: "datasets"
    models_prefix: "models"
    results_prefix: "results"

    endpoint_id: ${RUNPOD_ENDPOINT_ID}

  cost_limits:
    max_training_cost_dollars: 10.0
    max_prediction_cost_dollars: 1.0
    cost_warning_threshold: 0.8
```

---

### 4.3 Validation

#### Test Configuration

```bash
# Activate virtual environment
source venv/bin/activate

# Run configuration validation
python -c "
from src.cloud.aws_config import CloudConfig
from src.cloud.runpod_config import RunPodConfig

# Test AWS
try:
    aws_config = CloudConfig.from_env()
    aws_config.validate()
    print('✅ AWS configuration valid')
except Exception as e:
    print(f'❌ AWS configuration error: {e}')

# Test RunPod
try:
    runpod_config = RunPodConfig.from_env()
    runpod_config.validate()
    print('✅ RunPod configuration valid')
except Exception as e:
    print(f'❌ RunPod configuration error: {e}')
"
```

#### Test Cloud Connectivity

**AWS:**

```bash
# Test S3 access
aws s3 ls s3://${S3_BUCKET}/

# Test EC2 permissions
aws ec2 describe-instances --max-results 1

# Test Lambda invocation (async)
aws lambda invoke \
  --function-name ml-agent-prediction \
  --invocation-type DryRun \
  response.json
```

**RunPod:**

```bash
# Test API key
python -c "
import runpod
import os

runpod.api_key = os.getenv('RUNPOD_API_KEY')
pods = runpod.get_pods()
print(f'✅ RunPod API connected. Pods: {len(pods)}')
"

# Test network volume access
runpodctl list volumes
```

---

## 5. Monitoring Setup

### 5.1 CloudWatch (AWS)

#### Create Log Groups

```bash
# Create log groups
aws logs create-log-group --log-group-name /aws/ml-agent/training
aws logs create-log-group --log-group-name /aws/ml-agent/prediction
aws logs create-log-group --log-group-name /aws/ml-agent/errors

# Set retention (30 days)
for group in training prediction errors; do
  aws logs put-retention-policy \
    --log-group-name /aws/ml-agent/${group} \
    --retention-in-days 30
done
```

#### Create CloudWatch Dashboard

```bash
# Create dashboard JSON
cat > cloudwatch-dashboard.json <<'EOF'
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/EC2", "CPUUtilization", {"stat": "Average"}],
          [".", "NetworkIn", {"stat": "Sum"}],
          [".", "NetworkOut", {"stat": "Sum"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "EC2 Training Instances"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Invocations", {"stat": "Sum"}],
          [".", "Errors", {"stat": "Sum"}],
          [".", "Duration", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "Lambda Predictions"
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '/aws/ml-agent/errors'\n| fields @timestamp, @message\n| sort @timestamp desc\n| limit 20",
        "region": "us-east-1",
        "title": "Recent Errors"
      }
    }
  ]
}
EOF

# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name ml-agent-dashboard \
  --dashboard-body file://cloudwatch-dashboard.json
```

#### Create Alarms

```bash
# High training cost alarm
aws cloudwatch put-metric-alarm \
  --alarm-name ml-agent-high-cost \
  --alarm-description "Alert when training cost exceeds threshold" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold 50.0 \
  --comparison-operator GreaterThanThreshold

# Lambda error alarm
aws cloudwatch put-metric-alarm \
  --alarm-name ml-agent-lambda-errors \
  --alarm-description "Alert on Lambda prediction errors" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --dimensions Name=FunctionName,Value=ml-agent-prediction \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold
```

---

### 5.2 RunPod Monitoring

#### Enable Pod Logs

```python
import runpod
import os

runpod.api_key = os.getenv('RUNPOD_API_KEY')

# Get pod logs
pod_id = "your-pod-id"
logs = runpod.get_pod_logs(pod_id)

print(logs)
```

#### Create Monitoring Script

**File:** `scripts/monitor_runpod.py`

```python
#!/usr/bin/env python3
"""
RunPod monitoring script - checks pod status and costs.
"""
import runpod
import os
from datetime import datetime

runpod.api_key = os.getenv('RUNPOD_API_KEY')

def monitor_pods():
    pods = runpod.get_pods()

    print(f"\n{'='*60}")
    print(f"RunPod Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    total_cost = 0.0

    for pod in pods:
        status = pod.get('desiredStatus', 'UNKNOWN')
        gpu_type = pod.get('machine', {}).get('gpuType', 'N/A')
        uptime = pod.get('runtime', {}).get('uptimeInSeconds', 0)

        # Estimate cost (example rates)
        rates = {
            'NVIDIA RTX A5000': 0.29,
            'NVIDIA RTX A40': 0.39,
            'NVIDIA A100 PCIe 40GB': 0.79,
            'NVIDIA A100 PCIe 80GB': 1.19
        }

        rate = rates.get(gpu_type, 0.0)
        cost = (uptime / 3600) * rate
        total_cost += cost

        print(f"Pod ID: {pod['id']}")
        print(f"  Status: {status}")
        print(f"  GPU: {gpu_type}")
        print(f"  Uptime: {uptime // 60:.0f} minutes")
        print(f"  Cost: ${cost:.3f}")
        print()

    print(f"{'='*60}")
    print(f"Total Cost: ${total_cost:.3f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    monitor_pods()
```

**Run Monitoring:**

```bash
# Make executable
chmod +x scripts/monitor_runpod.py

# Run manually
python scripts/monitor_runpod.py

# Set up cron job (every 30 minutes)
crontab -e
# Add: */30 * * * * /path/to/venv/bin/python /path/to/scripts/monitor_runpod.py >> /tmp/runpod-monitor.log 2>&1
```

---

## 6. Cost Optimization

### 6.1 AWS Cost Optimization

#### Use Spot Instances

```python
# In src/cloud/ec2_manager.py
def launch_training(...):
    # Use Spot instead of On-Demand
    response = ec2.request_spot_instances(
        SpotPrice=str(spot_max_price),
        InstanceCount=1,
        LaunchSpecification={
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'KeyName': key_name,
            'SecurityGroupIds': [security_group],
            'UserData': training_script
        }
    )
```

**Savings:** 50-70% cheaper than On-Demand

#### S3 Lifecycle Policies

Automatically delete old data:

```bash
# Already configured in Section 2.2
# Deletes datasets after 90 days
# Deletes results after 30 days
```

**Savings:** Reduce storage costs by ~40%

#### Lambda Reserved Concurrency

If running >1000 predictions/day:

```bash
# Reserve concurrency (predictable pricing)
aws lambda put-function-concurrency \
  --function-name ml-agent-prediction \
  --reserved-concurrent-executions 10
```

---

### 6.2 RunPod Cost Optimization

#### Use Community Cloud (Spot Pricing)

```yaml
# In config.yaml
cloud:
  runpod:
    cloud_type: "COMMUNITY"  # 40-50% cheaper than SECURE
```

#### Auto-Terminate Idle Pods

```python
# In cloud training handlers
def _handle_training_completion(...):
    # CRITICAL: Terminate pod immediately after training
    self.training_manager.terminate_pod(job_id)
    logger.info(f"Pod {job_id} terminated to stop costs")
```

#### Choose Right-Sized GPUs

| Dataset Size | Recommended GPU | Cost/Hour |
|--------------|-----------------|-----------|
| <100MB | RTX A5000 (24GB) | $0.29 |
| 100MB-1GB | RTX A40 (48GB) | $0.39 |
| 1GB-10GB | A100 40GB | $0.79 |
| >10GB | A100 80GB | $1.19 |

**Avoid over-provisioning:** RTX A5000 sufficient for 90% of jobs

---

## 7. Security Best Practices

### 7.1 Credential Management

#### Use AWS Secrets Manager

```bash
# Store sensitive credentials
aws secretsmanager create-secret \
  --name ml-agent/credentials \
  --secret-string '{
    "telegram_bot_token": "your_token",
    "anthropic_api_key": "your_key"
  }'

# Retrieve in application
aws secretsmanager get-secret-value \
  --secret-id ml-agent/credentials
```

#### Rotate API Keys

```bash
# AWS: Rotate access keys every 90 days
aws iam create-access-key --user-name ml-agent-user
# Update .env with new keys
aws iam delete-access-key --user-name ml-agent-user --access-key-id OLD_KEY

# RunPod: Regenerate API key
# Web UI → Settings → API Keys → Revoke old key → Create new
```

---

### 7.2 Network Security

#### Restrict S3 Access

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RestrictToVPC",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": "arn:aws:s3:::ml-agent-data-*/*",
      "Condition": {
        "StringNotEquals": {
          "aws:SourceVpce": "vpce-1234567"
        }
      }
    }
  ]
}
```

#### Encrypt Data at Rest

```bash
# S3 encryption (already configured)
aws s3api put-bucket-encryption ...

# EBS encryption for EC2
aws ec2 modify-volume \
  --volume-id vol-1234567 \
  --encrypted \
  --kms-key-id arn:aws:kms:...
```

---

## 8. Troubleshooting

### 8.1 Common AWS Issues

#### Issue: S3 Access Denied

**Error:**
```
botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the PutObject operation
```

**Solution:**
```bash
# Check IAM policy attached
aws iam list-attached-user-policies --user-name ml-agent-user

# Verify bucket policy
aws s3api get-bucket-policy --bucket ml-agent-data-123456
```

#### Issue: EC2 Instance Won't Start

**Error:**
```
InsufficientInstanceCapacity: We currently do not have sufficient capacity in the requested Availability Zone
```

**Solution:**
```bash
# Try different AZ
aws ec2 run-instances --placement AvailabilityZone=us-east-1b ...

# Or use different instance type
--instance-type p3.8xlarge
```

---

### 8.2 Common RunPod Issues

#### Issue: Pod Creation Timeout

**Error:**
```
Pod creation timed out after 5 minutes
```

**Solution:**
- Retry with same GPU (often succeeds on 2nd attempt)
- Choose different GPU type
- Try different region
- Check RunPod status: https://status.runpod.io

#### Issue: Network Volume Not Mounted

**Error:**
```
OSError: [Errno 2] No such file or directory: '/workspace/datasets'
```

**Solution:**
```bash
# Verify volume ID in pod configuration
runpodctl get pod <pod_id>

# Ensure volume attached in pod creation
--volume v3zskt9gvb:/workspace
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Cloud provider account created and funded
- [ ] API credentials generated and saved
- [ ] Required tools installed (AWS CLI / RunPodCTL)
- [ ] .env file configured with all credentials
- [ ] config.yaml updated with provider settings

### AWS Deployment

- [ ] IAM user and policies created
- [ ] S3 bucket created with lifecycle policies
- [ ] EC2 AMI created with ML dependencies
- [ ] Lambda function deployed with layer
- [ ] CloudWatch logging configured
- [ ] Security groups and network access configured

### RunPod Deployment

- [ ] Network volume created (100GB+)
- [ ] Directory structure initialized
- [ ] Storage credentials configured
- [ ] Test pod creation successful
- [ ] Monitoring script deployed

### Post-Deployment

- [ ] Configuration validation passed
- [ ] Cloud connectivity tests successful
- [ ] Test training job completed
- [ ] Test prediction executed
- [ ] Monitoring dashboards created
- [ ] Cost alerts configured
- [ ] Documentation updated with provider-specific details

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Next Review:** 2025-12-08
