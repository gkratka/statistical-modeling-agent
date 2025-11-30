# RunPod ML Training Docker Image
# Optimized for instant pod startup with all dependencies pre-installed
#
# Usage:
#   docker build -f docker/ml-training.Dockerfile -t ml-training:v1 .
#   docker tag ml-training:v1 registry.runpod.io/YOUR_USERNAME/ml-training:v1
#   docker push registry.runpod.io/YOUR_USERNAME/ml-training:v1
#
# Author: Statistical Modeling Agent
# Created: 2025-11-11

FROM tensorflow/tensorflow:2.13.0-gpu

LABEL maintainer="Statistical Modeling Agent"
LABEL description="Pre-configured ML training image for RunPod with TensorFlow and scikit-learn"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install ML dependencies (matches requirements.txt)
RUN pip install --no-cache-dir \
    # Core ML libraries
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    scipy==1.10.1 \
    joblib==1.3.2 \
    # Additional ML tools
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    # Data processing
    openpyxl==3.1.2 \
    pyarrow==12.0.1 \
    # Utilities
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    tqdm==4.65.0

# Create workspace directories
RUN mkdir -p /workspace/data \
    /workspace/models \
    /workspace/results \
    /workspace/scripts

# Set working directory
WORKDIR /workspace

# NOTE: Verification skipped to enable building on Apple Silicon
# Libraries will be verified when container runs on RunPod's x86_64 infrastructure

# Default command
CMD ["/bin/bash"]

# Build info
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/yourusername/statistical-modeling-agent"
