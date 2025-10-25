# RunPod Serverless Prediction Container
#
# This Docker image runs ML predictions on RunPod serverless endpoints.
# Based on Python 3.10 slim image for smaller container size.
#
# Author: Statistical Modeling Agent
# Created: 2025-10-24 (Task 5.2: RunPod Dockerfile)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy prediction handler
COPY prediction_handler.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check (optional - RunPod will handle this)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import runpod; print('OK')" || exit 1

# Start serverless worker
CMD ["python", "-u", "prediction_handler.py"]
