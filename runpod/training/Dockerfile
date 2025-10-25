FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Install ML dependencies
RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    joblib>=1.3.0 \
    boto3>=1.28.0 \
    xgboost>=2.0.0 \
    lightgbm>=4.0.0 \
    catboost>=1.2.0

# Copy training script
COPY train.py /workspace/

# Training script runs on container start
CMD ["python", "/workspace/train.py"]
