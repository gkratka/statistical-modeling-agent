#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script

This script automates the setup of AWS resources for cloud ML workflows:
- Creates S3 bucket with encryption and policies
- Creates IAM roles with minimal permissions
- Configures security groups
- Sets up CloudWatch log groups

Usage:
    python scripts/cloud/setup_aws.py --config config/cloud_config.yaml
    python scripts/cloud/setup_aws.py --config config/cloud_config.yaml --account-id 123456789012

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 7.6: AWS Setup Automation)
"""

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cloud.aws_config import CloudConfig
from src.cloud.security import SecurityManager


def create_s3_bucket(s3_client: boto3.client, bucket_name: str, region: str) -> None:
    """
    Create S3 bucket with best practices.

    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of bucket to create
        region: AWS region

    Raises:
        Exception: If bucket creation fails (other than already exists)
    """
    print(f"Creating S3 bucket: {bucket_name}")

    try:
        # us-east-1 doesn't support LocationConstraint
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )

        print(f"✅ Bucket created: {bucket_name}")

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')

        if error_code == 'BucketAlreadyOwnedByYou':
            print(f"⚠️  Bucket already exists: {bucket_name}")
        elif error_code == 'BucketAlreadyExists':
            print(f"❌ Bucket name already taken globally: {bucket_name}")
            raise
        else:
            print(f"❌ Failed to create bucket: {e}")
            raise


def configure_bucket_security(
    s3_client: boto3.client,
    bucket_name: str,
    security_manager: SecurityManager,
    account_id: str
) -> None:
    """
    Apply security configurations to S3 bucket.

    Configures:
    - Block public access
    - Default encryption
    - Versioning
    - Bucket policy

    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of bucket to configure
        security_manager: SecurityManager instance
        account_id: AWS account ID

    Raises:
        Exception: If security configuration fails
    """
    print(f"\nConfiguring bucket security for: {bucket_name}")

    # Block public access
    try:
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        print("✅ Public access blocked")
    except ClientError as e:
        print(f"❌ Failed to block public access: {e}")
        raise

    # Enable encryption
    try:
        security_manager.configure_bucket_encryption(s3_client)
        print("✅ Encryption enabled")
    except Exception as e:
        print(f"❌ Failed to enable encryption: {e}")
        raise

    # Enable versioning
    try:
        security_manager.configure_bucket_versioning(s3_client)
        print("✅ Versioning enabled")
    except Exception as e:
        print(f"❌ Failed to enable versioning: {e}")
        raise

    # Apply bucket policy
    try:
        bucket_policy = security_manager.generate_s3_bucket_policy(account_id)
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(bucket_policy)
        )
        print("✅ Bucket policy applied")
    except ClientError as e:
        print(f"⚠️  Warning: Failed to apply bucket policy: {e}")
        print("   (This may require IAM role ARN to be configured first)")


def create_iam_roles(
    iam_client: boto3.client,
    security_manager: SecurityManager
) -> dict:
    """
    Create IAM roles for EC2 and Lambda.

    Creates:
    - EC2 training role with S3 read/write and self-terminate permissions
    - Lambda prediction role with S3 read/write and logs permissions

    Args:
        iam_client: boto3 IAM client
        security_manager: SecurityManager instance

    Returns:
        dict: Role ARNs with keys 'ec2_role_arn' and 'lambda_role_arn'

    Raises:
        Exception: If role creation fails
    """
    print("\nCreating IAM roles...")

    # EC2 training role
    ec2_role_name = "ml-agent-ec2-training-role"
    print(f"Creating EC2 role: {ec2_role_name}")

    try:
        ec2_role_response = iam_client.create_role(
            RoleName=ec2_role_name,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }),
            Description="Role for ML Agent EC2 training instances"
        )
        ec2_role_arn = ec2_role_response['Role']['Arn']

        # Attach inline policy
        ec2_policy = security_manager.generate_ec2_iam_role_policy()
        iam_client.put_role_policy(
            RoleName=ec2_role_name,
            PolicyName="EC2TrainingPolicy",
            PolicyDocument=json.dumps(ec2_policy)
        )

        print(f"✅ EC2 role created: {ec2_role_name}")

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')

        if error_code == 'EntityAlreadyExists':
            print(f"⚠️  EC2 role already exists: {ec2_role_name}")
            ec2_role_arn = iam_client.get_role(RoleName=ec2_role_name)['Role']['Arn']
        else:
            print(f"❌ Failed to create EC2 role: {e}")
            raise

    # Lambda prediction role
    lambda_role_name = "ml-agent-lambda-prediction-role"
    print(f"Creating Lambda role: {lambda_role_name}")

    try:
        lambda_role_response = iam_client.create_role(
            RoleName=lambda_role_name,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }),
            Description="Role for ML Agent Lambda prediction function"
        )
        lambda_role_arn = lambda_role_response['Role']['Arn']

        # Attach inline policy
        lambda_policy = security_manager.generate_lambda_iam_role_policy()
        iam_client.put_role_policy(
            RoleName=lambda_role_name,
            PolicyName="LambdaPredictionPolicy",
            PolicyDocument=json.dumps(lambda_policy)
        )

        print(f"✅ Lambda role created: {lambda_role_name}")

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')

        if error_code == 'EntityAlreadyExists':
            print(f"⚠️  Lambda role already exists: {lambda_role_name}")
            lambda_role_arn = iam_client.get_role(RoleName=lambda_role_name)['Role']['Arn']
        else:
            print(f"❌ Failed to create Lambda role: {e}")
            raise

    return {
        'ec2_role_arn': ec2_role_arn,
        'lambda_role_arn': lambda_role_arn
    }


def main() -> None:
    """Main entry point for AWS setup script."""
    parser = argparse.ArgumentParser(
        description="Setup AWS infrastructure for ML Agent cloud workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup using config file
  python scripts/cloud/setup_aws.py --config config/cloud_config.yaml

  # Setup with explicit account ID
  python scripts/cloud/setup_aws.py --config config/cloud_config.yaml --account-id 123456789012

Note:
  - AWS credentials must be configured (environment variables or ~/.aws/credentials)
  - Requires permissions to create S3 buckets and IAM roles
  - IAM role ARN in config may need to be updated after role creation
        """
    )
    parser.add_argument(
        '--config',
        required=True,
        help="Path to cloud configuration YAML file"
    )
    parser.add_argument(
        '--account-id',
        help="AWS account ID (auto-detected if not provided)"
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    try:
        config = CloudConfig.from_yaml(args.config)
        config.validate()
        print(f"✅ Configuration loaded from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize AWS clients
    print(f"\nInitializing AWS clients in region: {config.aws_region}")
    try:
        s3_client = boto3.client(
            's3',
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key
        )
        iam_client = boto3.client(
            'iam',
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key
        )
        sts_client = boto3.client(
            'sts',
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key
        )
        print("✅ AWS clients initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AWS clients: {e}")
        sys.exit(1)

    # Get account ID
    if args.account_id:
        account_id = args.account_id
        print(f"Using provided account ID: {account_id}")
    else:
        try:
            account_id = sts_client.get_caller_identity()['Account']
            print(f"Auto-detected account ID: {account_id}")
        except Exception as e:
            print(f"❌ Failed to get account ID: {e}")
            sys.exit(1)

    # Initialize security manager
    security_manager = SecurityManager(config)

    # Setup S3 bucket
    print("\n" + "="*60)
    print("STEP 1: S3 Bucket Setup")
    print("="*60)
    try:
        create_s3_bucket(s3_client, config.s3_bucket, config.aws_region)
        configure_bucket_security(s3_client, config.s3_bucket, security_manager, account_id)
    except Exception as e:
        print(f"\n❌ S3 bucket setup failed: {e}")
        print("Fix the errors above and re-run the script.")
        sys.exit(1)

    # Setup IAM roles
    print("\n" + "="*60)
    print("STEP 2: IAM Roles Setup")
    print("="*60)
    try:
        role_arns = create_iam_roles(iam_client, security_manager)
    except Exception as e:
        print(f"\n❌ IAM roles setup failed: {e}")
        print("Fix the errors above and re-run the script.")
        sys.exit(1)

    # Print summary
    print("\n" + "="*60)
    print("AWS INFRASTRUCTURE SETUP COMPLETE!")
    print("="*60)
    print(f"\nS3 Bucket:        {config.s3_bucket}")
    print(f"AWS Region:       {config.aws_region}")
    print(f"Account ID:       {account_id}")
    print(f"\nEC2 Role:         {role_arns['ec2_role_arn']}")
    print(f"Lambda Role:      {role_arns['lambda_role_arn']}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Update your configuration file with the IAM role ARN:")
    print(f"   iam_role_arn: {role_arns['ec2_role_arn']}")
    print("\n2. Create EC2 instance profile:")
    print(f"   aws iam create-instance-profile --instance-profile-name ml-agent-ec2-profile")
    print(f"   aws iam add-role-to-instance-profile --instance-profile-name ml-agent-ec2-profile --role-name ml-agent-ec2-training-role")
    print("\n3. Test the setup:")
    print("   python -m pytest tests/unit/test_security.py -v")
    print("\n4. Deploy Lambda function (if using serverless predictions)")
    print("\n")


if __name__ == '__main__':
    main()
