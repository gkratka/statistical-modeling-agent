"""
AWS Client Wrapper Verification Script.

Demonstrates the AWSClient API and usage patterns.
NOTE: Requires boto3 installation to run.

Usage:
    pip install boto3>=1.28.0
    python dev/implemented/aws_client_verification.py
"""

from src.cloud.aws_config import CloudConfig
from src.cloud.aws_client import AWSClient
from src.cloud.exceptions import AWSError


def verify_aws_client():
    """Verify AWSClient implementation with sample config."""

    print("=" * 70)
    print("AWS Client Wrapper Verification")
    print("=" * 70)

    # Create sample configuration
    config = CloudConfig(
        # AWS credentials (use your actual credentials or IAM role)
        aws_region="us-east-1",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",

        # S3 configuration
        s3_bucket="my-ml-models-bucket",
        s3_data_prefix="data/",
        s3_models_prefix="models/",
        s3_results_prefix="results/",

        # EC2 configuration
        ec2_instance_type="c5.xlarge",
        ec2_ami_id="ami-0c55b159cbfafe1f0",
        ec2_key_name="my-ec2-key",
        ec2_security_group="sg-1234567890abcdef0",

        # Lambda configuration
        lambda_function_name="ml-training-function",
        lambda_memory_mb=3008,
        lambda_timeout_seconds=900,

        # Cost limits
        max_training_cost_dollars=100.0,
        max_prediction_cost_dollars=10.0,
        cost_warning_threshold=0.8
    )

    print("\n1. Initializing AWS Client...")
    try:
        aws_client = AWSClient(config)
        print("   ✓ AWS client initialized successfully")
    except AWSError as e:
        print(f"   ✗ Failed to initialize: {e}")
        return

    print("\n2. Running Health Checks...")
    health = aws_client.health_check()

    print(f"\n   Overall Status: {health['overall_status']}")
    print("\n   Service Status:")

    for service in ["s3", "ec2", "lambda"]:
        service_health = health[service]
        status = service_health["status"]

        if status == "healthy":
            print(f"   ✓ {service.upper()}: {status}")
        else:
            print(f"   ✗ {service.upper()}: {status}")
            print(f"      Error Code: {service_health.get('error_code', 'N/A')}")
            print(f"      Message: {service_health.get('error_message', 'N/A')}")
            print(f"      Request ID: {service_health.get('request_id', 'N/A')}")

    print("\n3. Accessing Service Clients...")
    s3_client = aws_client.get_s3_client()
    ec2_client = aws_client.get_ec2_client()
    lambda_client = aws_client.get_lambda_client()

    print(f"   ✓ S3 Client: {type(s3_client).__name__}")
    print(f"   ✓ EC2 Client: {type(ec2_client).__name__}")
    print(f"   ✓ Lambda Client: {type(lambda_client).__name__}")

    print("\n4. Example Usage with S3 Client...")
    try:
        response = s3_client.list_buckets()
        bucket_count = len(response.get("Buckets", []))
        print(f"   ✓ Found {bucket_count} S3 buckets")

        if bucket_count > 0:
            print("\n   Sample buckets:")
            for bucket in response["Buckets"][:3]:
                print(f"   - {bucket['Name']}")
    except Exception as e:
        print(f"   ✗ S3 operation failed: {e}")

    print("\n" + "=" * 70)
    print("Verification Complete")
    print("=" * 70)


if __name__ == "__main__":
    try:
        verify_aws_client()
    except ImportError as e:
        print("\n" + "=" * 70)
        print("ERROR: Missing Required Dependencies")
        print("=" * 70)
        print(f"\n{e}")
        print("\nPlease install boto3:")
        print("  pip install boto3>=1.28.0")
        print("\n" + "=" * 70)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
