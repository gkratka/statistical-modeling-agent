"""
Cloud infrastructure package for AWS-based ML operations.

This package provides AWS service wrappers for cloud-based machine learning
training and prediction workflows.
"""

from src.cloud.exceptions import (
    CloudError,
    AWSError,
    S3Error,
    EC2Error,
    LambdaError,
    CostTrackingError,
    CloudConfigurationError,
)

__all__ = [
    "CloudError",
    "AWSError",
    "S3Error",
    "EC2Error",
    "LambdaError",
    "CostTrackingError",
    "CloudConfigurationError",
]
