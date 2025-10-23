"""
Example usage of S3Manager for dataset uploads.

This script demonstrates how to use S3Manager to upload datasets to S3
with user isolation, automatic multipart upload, and encryption.

Author: Statistical Modeling Agent
Created: 2025-10-23 (Task 2.1: S3Manager Usage Example)
"""

from pathlib import Path

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.s3_manager import S3Manager


def example_upload_small_dataset():
    """Example: Upload small dataset (<5MB) with simple upload."""
    # Load configuration
    config = CloudConfig.from_env()

    # Initialize AWS client
    aws_client = AWSClient(config)

    # Initialize S3Manager
    s3_manager = S3Manager(aws_client=aws_client, config=config)

    # Upload small dataset
    user_id = 12345
    file_path = "/path/to/housing_data.csv"

    s3_uri = s3_manager.upload_dataset(
        user_id=user_id,
        file_path=file_path,
        dataset_name="housing_data.csv"
    )

    print(f"Dataset uploaded successfully to: {s3_uri}")
    # Output: s3://my-bucket/datasets/user_12345/20251023_143045_housing_data.csv


def example_upload_large_dataset():
    """Example: Upload large dataset (>5MB) with multipart upload."""
    # Load configuration from YAML
    config = CloudConfig.from_yaml("config/cloud_config.yaml")

    # Initialize components
    aws_client = AWSClient(config)
    s3_manager = S3Manager(aws_client=aws_client, config=config)

    # Upload large dataset (automatically uses multipart upload)
    user_id = 67890
    file_path = Path("/path/to/large_timeseries_data.parquet")

    s3_uri = s3_manager.upload_dataset(
        user_id=user_id,
        file_path=file_path
    )

    print(f"Large dataset uploaded successfully to: {s3_uri}")
    # Output: s3://my-bucket/datasets/user_67890/20251023_143045_large_timeseries_data.parquet


def example_user_isolation():
    """Example: Demonstrate user isolation in dataset uploads."""
    config = CloudConfig.from_env()
    aws_client = AWSClient(config)
    s3_manager = S3Manager(aws_client=aws_client, config=config)

    # Upload same dataset for different users
    file_path = "/path/to/shared_dataset.csv"

    uri_user1 = s3_manager.upload_dataset(user_id=100, file_path=file_path)
    uri_user2 = s3_manager.upload_dataset(user_id=200, file_path=file_path)

    print(f"User 100 dataset: {uri_user1}")
    print(f"User 200 dataset: {uri_user2}")

    # Output:
    # User 100 dataset: s3://my-bucket/datasets/user_100/20251023_143045_shared_dataset.csv
    # User 200 dataset: s3://my-bucket/datasets/user_200/20251023_143046_shared_dataset.csv


def example_error_handling():
    """Example: Error handling for upload failures."""
    from src.cloud.exceptions import S3Error

    config = CloudConfig.from_env()
    aws_client = AWSClient(config)
    s3_manager = S3Manager(aws_client=aws_client, config=config)

    try:
        # Attempt to upload non-existent file
        s3_uri = s3_manager.upload_dataset(
            user_id=12345,
            file_path="/nonexistent/file.csv"
        )
    except S3Error as e:
        print(f"Upload failed: {e}")
        print(f"Error code: {e.error_code}")
        print(f"Bucket: {e.bucket}")
        # Output:
        # Upload failed: Dataset file not found: /nonexistent/file.csv | Path: s3://my-bucket
        # Error code: FileNotFound
        # Bucket: my-bucket


if __name__ == "__main__":
    print("S3Manager Usage Examples")
    print("=" * 50)

    print("\n1. Upload small dataset:")
    print("-" * 50)
    example_upload_small_dataset()

    print("\n2. Upload large dataset:")
    print("-" * 50)
    example_upload_large_dataset()

    print("\n3. User isolation:")
    print("-" * 50)
    example_user_isolation()

    print("\n4. Error handling:")
    print("-" * 50)
    example_error_handling()
