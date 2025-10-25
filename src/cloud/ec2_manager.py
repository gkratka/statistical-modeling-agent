"""
EC2 Manager for orchestrating cloud-based ML training.

This module provides EC2 instance selection, Spot instance launching,
and instance lifecycle management for cloud-based machine learning training.

Author: Statistical Modeling Agent
Created: 2025-10-23 (Tasks 3.1 & 3.2: EC2Manager with TDD)
Updated: 2025-10-23 (Tasks 3.3-3.6: Additional EC2Manager functionality)
"""

import asyncio
import base64
import json
import time
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from botocore.exceptions import ClientError

from src.cloud.aws_client import AWSClient
from src.cloud.aws_config import CloudConfig
from src.cloud.exceptions import EC2Error
from src.cloud.provider_interface import CloudTrainingProvider


class EC2Manager(CloudTrainingProvider):
    """
    EC2 instance manager for cloud-based ML training.

    This class handles:
    - Optimal instance type selection based on dataset and model characteristics
    - Spot instance launching with proper configuration
    - Instance assignment waiting and status tracking
    """

    def __init__(self, aws_client: AWSClient, config: CloudConfig) -> None:
        """
        Initialize EC2Manager.

        Args:
            aws_client: AWSClient instance for accessing EC2 services
            config: CloudConfig instance with EC2 configuration

        Raises:
            EC2Error: If EC2 client initialization fails
        """
        self._aws_client = aws_client
        self._config = config
        self._ec2_client = aws_client.get_ec2_client()

    def select_instance_type(
        self,
        dataset_size_mb: float,
        model_type: str,
        estimated_training_time_minutes: int
    ) -> str:
        """
        Select optimal EC2 instance type based on workload characteristics.

        Instance selection logic:
        - Simple models (linear, logistic, ridge, lasso):
          * <100MB → t3.medium
          * ≥100MB → m5.large

        - Tree-based models (random_forest, gradient_boosting, xgboost):
          * <1000MB → m5.large
          * ≥1000MB → m5.xlarge

        - Neural networks (mlp, neural, mlp_regression, mlp_classification):
          * >5000MB → p3.2xlarge (GPU)
          * ≤5000MB → m5.xlarge

        - Unknown models → config.ec2_instance_type (fallback)

        Args:
            dataset_size_mb: Dataset size in megabytes
            model_type: ML model type identifier
            estimated_training_time_minutes: Estimated training duration

        Returns:
            str: EC2 instance type identifier (e.g., "m5.large")
        """
        # Normalize model type for matching
        model_type_lower = model_type.lower()

        # Simple models: linear, logistic, ridge, lasso
        simple_models = {"linear", "logistic", "ridge", "lasso"}
        if model_type_lower in simple_models:
            if dataset_size_mb < 100:
                return "t3.medium"
            else:
                return "m5.large"

        # Tree-based models: random_forest, gradient_boosting, xgboost
        tree_models = {"random_forest", "gradient_boosting", "xgboost"}
        if model_type_lower in tree_models:
            if dataset_size_mb >= 1000:  # ≥1GB (using 1000MB as threshold)
                return "m5.xlarge"
            else:
                return "m5.large"

        # Neural network models: mlp, neural, mlp_regression, mlp_classification
        neural_models = {"mlp", "neural", "mlp_regression", "mlp_classification", "neural_network"}
        if model_type_lower in neural_models:
            if dataset_size_mb > 5000:  # >5GB (5000MB threshold)
                return "p3.2xlarge"
            else:
                return "m5.xlarge"

        # Unknown model type - fallback to config default
        return self._config.ec2_instance_type

    # CloudTrainingProvider interface implementation
    def select_compute_type(
        self,
        dataset_size_mb: float,
        model_type: str,
        estimated_training_time_minutes: int = 0
    ) -> str:
        """
        Select optimal compute resource for training (CloudTrainingProvider interface).

        Wrapper for select_instance_type() to implement CloudTrainingProvider interface.

        Args:
            dataset_size_mb: Dataset size in megabytes
            model_type: Type of ML model
            estimated_training_time_minutes: Estimated training duration

        Returns:
            str: Compute resource identifier (EC2 instance type)
        """
        return self.select_instance_type(
            dataset_size_mb=dataset_size_mb,
            model_type=model_type,
            estimated_training_time_minutes=estimated_training_time_minutes
        )

    def launch_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Launch training job on cloud compute (CloudTrainingProvider interface).

        Args:
            config: Training configuration dict containing:
                - compute_type: EC2 instance type
                - dataset_uri: S3 URI for dataset
                - model_id: Model identifier
                - user_id: User ID
                - model_type: ML model type
                - target_column: Target variable name
                - feature_columns: List of feature names
                - hyperparameters: Model hyperparameters

        Returns:
            dict: Job details with job_id, status, launch_time
        """
        # Extract configuration
        compute_type = config.get("compute_type", self._config.ec2_instance_type)
        dataset_uri = config["dataset_uri"]
        model_id = config["model_id"]
        user_id = config["user_id"]
        model_type = config["model_type"]
        target_column = config["target_column"]
        feature_columns = config["feature_columns"]
        hyperparameters = config.get("hyperparameters", {})

        # Generate S3 output URI
        s3_output_uri = f"s3://{self._config.s3_bucket}/{self._config.s3_models_prefix}/user_{user_id}/{model_id}"

        # Generate user data script
        user_data_script = self.generate_training_userdata(
            s3_dataset_uri=dataset_uri,
            model_type=model_type,
            target_column=target_column,
            feature_columns=feature_columns,
            hyperparameters=hyperparameters,
            s3_output_uri=s3_output_uri
        )

        # Prepare tags
        tags = {
            "Name": f"ml-training-{model_id}",
            "user_id": str(user_id),
            "model_id": model_id,
            "model_type": model_type
        }

        # Launch Spot instance
        result = self.launch_spot_instance(
            instance_type=compute_type,
            user_data_script=user_data_script,
            tags=tags
        )

        # Return in interface format
        return {
            "job_id": result["instance_id"],
            "status": "launching",
            "launch_time": result["launch_time"].isoformat() if isinstance(result["launch_time"], datetime) else result["launch_time"]
        }

    def monitor_training(self, job_id: str) -> Dict[str, Any]:
        """
        Monitor training job status (CloudTrainingProvider interface).

        Args:
            job_id: Training job identifier (EC2 instance ID)

        Returns:
            dict: Status dict with job_id, status, runtime, progress
        """
        status = self.get_instance_status(instance_id=job_id)

        return {
            "job_id": job_id,
            "status": status["state"],
            "runtime": None,  # Would need CloudWatch metrics for accurate runtime
            "progress": None  # Would need log parsing for progress tracking
        }

    def terminate_training(self, job_id: str) -> str:
        """
        Terminate training job (CloudTrainingProvider interface).

        Args:
            job_id: Training job identifier (EC2 instance ID)

        Returns:
            str: Job ID of terminated job
        """
        self.terminate_instance(instance_id=job_id)
        return job_id

    def launch_spot_instance(
        self,
        instance_type: str,
        user_data_script: str,
        tags: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Launch EC2 Spot instance with training script.

        Creates a Spot instance request with:
        - Specified instance type
        - UserData script for training
        - 50GB gp3 EBS volume (encrypted, delete on termination)
        - Custom tags for tracking
        - IAM role for AWS access
        - Security group and networking

        Args:
            instance_type: EC2 instance type (e.g., "m5.large")
            user_data_script: Bash script to execute on instance startup
            tags: Dictionary of tags to apply to instance

        Returns:
            dict: Launch result with structure:
                {
                    "instance_id": str,  # EC2 instance ID
                    "spot_request_id": str,  # Spot request ID
                    "instance_type": str,  # Instance type used
                    "launch_time": datetime  # Launch timestamp
                }

        Raises:
            EC2Error: If Spot instance request fails
        """
        try:
            # Prepare tag specifications
            tag_specifications = []
            if tags:
                tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
                tag_specifications = [
                    {
                        "ResourceType": "instance",
                        "Tags": tag_list
                    }
                ]

            # Launch specification
            launch_specification = {
                "ImageId": self._config.ec2_ami_id,
                "InstanceType": instance_type,
                "KeyName": self._config.ec2_key_name,
                "SecurityGroups": [self._config.ec2_security_group],
                "UserData": user_data_script,
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": 50,
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                            "Encrypted": True
                        }
                    }
                ]
            }

            # Add IAM role if configured
            if self._config.iam_role_arn:
                launch_specification["IamInstanceProfile"] = {
                    "Arn": self._config.iam_role_arn
                }

            # Add tag specifications if present
            if tag_specifications:
                launch_specification["TagSpecifications"] = tag_specifications

            # Request Spot instance
            response = self._ec2_client.request_spot_instances(
                SpotPrice=str(self._config.ec2_spot_max_price),
                InstanceCount=1,
                Type="one-time",
                LaunchSpecification=launch_specification
            )

            # Extract Spot request information
            spot_requests = response.get("SpotInstanceRequests", [])
            if not spot_requests:
                raise EC2Error(
                    message="No Spot instance request created",
                    error_code="NoSpotRequestReturned"
                )

            spot_request = spot_requests[0]
            spot_request_id = spot_request["SpotInstanceRequestId"]
            launch_time = spot_request.get("CreateTime", datetime.utcnow())

            # Wait for instance assignment
            instance_id = self._wait_for_instance_assignment(
                spot_request_id=spot_request_id,
                timeout_seconds=300
            )

            return {
                "instance_id": instance_id,
                "spot_request_id": spot_request_id,
                "instance_type": instance_type,
                "launch_time": launch_time
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            raise EC2Error(
                message=f"Failed to request Spot instance: {error_message}",
                instance_type=instance_type,
                error_code=error_code,
                request_id=request_id
            )
        except EC2Error:
            # Re-raise EC2Errors from _wait_for_instance_assignment
            raise
        except Exception as e:
            raise EC2Error(
                message=f"Unexpected error launching Spot instance: {str(e)}",
                instance_type=instance_type,
                error_code="UnexpectedError"
            )

    def _wait_for_instance_assignment(
        self,
        spot_request_id: str,
        timeout_seconds: int = 300
    ) -> str:
        """
        Wait for Spot instance request to be fulfilled.

        Polls Spot instance request status every 10 seconds until:
        - State becomes "active" with status "fulfilled" → return instance_id
        - State becomes "failed" or "cancelled" → raise EC2Error
        - Timeout expires → raise EC2Error

        Args:
            spot_request_id: Spot instance request ID
            timeout_seconds: Maximum wait time in seconds (default: 300)

        Returns:
            str: EC2 instance ID once assigned

        Raises:
            EC2Error: If request fails, is cancelled, or times out
        """
        start_time = time.time()
        poll_interval = 10  # seconds

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                raise EC2Error(
                    message=f"Timeout waiting for Spot instance assignment after {timeout_seconds}s",
                    error_code="SpotRequestTimeout"
                )

            try:
                # Query Spot request status
                response = self._ec2_client.describe_spot_instance_requests(
                    SpotInstanceRequestIds=[spot_request_id]
                )

                spot_requests = response.get("SpotInstanceRequests", [])
                if not spot_requests:
                    raise EC2Error(
                        message=f"Spot instance request not found: {spot_request_id}",
                        error_code="SpotRequestNotFound"
                    )

                spot_request = spot_requests[0]
                state = spot_request.get("State", "unknown")
                status = spot_request.get("Status", {})
                status_code = status.get("Code", "unknown")

                # Check for fulfilled state
                if state == "active" and status_code == "fulfilled":
                    instance_id = spot_request.get("InstanceId")
                    if instance_id:
                        return instance_id
                    else:
                        raise EC2Error(
                            message=f"Spot request fulfilled but no InstanceId found",
                            error_code="NoInstanceIdReturned"
                        )

                # Check for failed or cancelled states
                if state in {"failed", "cancelled", "closed"}:
                    status_message = status.get("Message", "No message provided")
                    raise EC2Error(
                        message=f"Spot instance request failed: {state} - {status_code}: {status_message}",
                        error_code=status_code
                    )

                # Continue polling
                time.sleep(poll_interval)

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                raise EC2Error(
                    message=f"Error checking Spot instance request status: {error_message}",
                    error_code=error_code
                )

    def generate_training_userdata(
        self,
        s3_dataset_uri: str,
        model_type: str,
        target_column: str,
        feature_columns: List[str],
        hyperparameters: Dict[str, Any],
        s3_output_uri: str
    ) -> str:
        """
        Generate UserData script for EC2 training instance.

        Creates a bash script that:
        - Downloads dataset from S3
        - Installs dependencies (pandas, scikit-learn, joblib, boto3)
        - Creates and executes Python training script
        - Uploads trained model to S3
        - Self-terminates instance

        Args:
            s3_dataset_uri: S3 URI of training dataset (e.g., "s3://bucket/data.csv")
            model_type: ML model type identifier
            target_column: Name of target column
            feature_columns: List of feature column names
            hyperparameters: Dictionary of model hyperparameters
            s3_output_uri: S3 URI for model output

        Returns:
            str: Base64-encoded UserData script
        """
        # Convert feature columns to Python list string
        features_str = json.dumps(feature_columns)
        hyperparams_str = json.dumps(hyperparameters)

        # Generate bash script
        script = f'''#!/bin/bash
set -e

# Update system and install dependencies
apt-get update
apt-get install -y python3-pip awscli

# Install Python packages
pip3 install pandas scikit-learn joblib boto3

# Download dataset from S3
aws s3 cp {s3_dataset_uri} /tmp/dataset.csv

# Create Python training script
cat > /tmp/train.py << 'PYTHON_SCRIPT'
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
print("Loading dataset...")
df = pd.read_csv("/tmp/dataset.csv")

# Extract features and target
target_column = "{target_column}"
feature_columns = {features_str}
hyperparameters = {hyperparams_str}

X = df[feature_columns]
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Map model types to sklearn classes
model_map = {{
    "linear": LinearRegression,
    "logistic": LogisticRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": GradientBoostingClassifier,  # Use sklearn as fallback
    "svm": SVC,
    "naive_bayes": GaussianNB,
    "mlp_classification": MLPClassifier,
    "mlp_regression": MLPRegressor,
    "neural_network": MLPClassifier
}}

# Train model
print(f"Training {{model_type}} model...")
model_class = model_map.get("{model_type}", RandomForestClassifier)
model = model_class(**hyperparameters)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model score: {{score}}")

# Save model
print("Saving model...")
joblib.dump(model, "/tmp/model.pkl")

# Save metadata
metadata = {{
    "model_type": "{model_type}",
    "target_column": target_column,
    "feature_columns": feature_columns,
    "hyperparameters": hyperparameters,
    "test_score": score
}}
with open("/tmp/metadata.json", "w") as f:
    json.dump(metadata, f)

print("Training complete!")
PYTHON_SCRIPT

# Execute training script
echo "Starting ML training..."
python3 /tmp/train.py

# Upload results to S3
echo "Uploading model to S3..."
aws s3 cp /tmp/model.pkl {s3_output_uri}/model.pkl
aws s3 cp /tmp/metadata.json {s3_output_uri}/metadata.json

# Self-terminate instance
echo "Training complete, shutting down instance..."
shutdown -h now
'''

        # Base64 encode the script
        encoded = base64.b64encode(script.encode('utf-8')).decode('utf-8')
        return encoded

    async def poll_training_logs(
        self,
        instance_id: str,
        log_group: str = "/aws/ec2/ml-training"
    ) -> AsyncIterator[str]:
        """
        Poll CloudWatch Logs for training instance output.

        Continuously polls CloudWatch Logs and yields log lines as they become available.
        Handles log stream initialization delays and stops when no new logs are detected.

        Args:
            instance_id: EC2 instance ID
            log_group: CloudWatch log group name (default: "/aws/ec2/ml-training")

        Yields:
            str: Log lines from training instance

        Note:
            - Polls every 5 seconds
            - Handles ResourceNotFoundException gracefully (log stream not yet created)
            - Stops when nextForwardToken is unchanged (no new logs)
        """
        logs_client = self._aws_client.get_logs_client()
        log_stream_name = instance_id
        next_token = None
        previous_token = None
        poll_interval = 5  # seconds

        while True:
            try:
                # Get log events
                params = {
                    "logGroupName": log_group,
                    "logStreamName": log_stream_name,
                    "startFromHead": True
                }

                if next_token:
                    params["nextToken"] = next_token

                response = logs_client.get_log_events(**params)

                # Yield log messages
                events = response.get("events", [])
                for event in events:
                    yield event["message"]

                # Check for new logs
                next_token = response.get("nextForwardToken")
                if next_token == previous_token:
                    # No new logs, stop polling
                    break

                previous_token = next_token

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "ResourceNotFoundException":
                    # Log stream doesn't exist yet, wait and retry
                    await asyncio.sleep(poll_interval)
                    continue
                else:
                    # Other errors should be raised
                    raise EC2Error(
                        message=f"Error polling training logs: {str(e)}",
                        error_code=error_code
                    )

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Get EC2 instance status information.

        Args:
            instance_id: EC2 instance ID

        Returns:
            dict: Instance status with structure:
                {
                    "state": str,  # Instance state (running, stopped, etc.)
                    "public_ip": str,  # Public IP address (if assigned)
                    "private_ip": str,  # Private IP address
                    "launch_time": datetime  # Instance launch time
                }

        Raises:
            EC2Error: If instance not found or API call fails
        """
        try:
            response = self._ec2_client.describe_instances(
                InstanceIds=[instance_id]
            )

            reservations = response.get("Reservations", [])
            if not reservations or not reservations[0].get("Instances"):
                raise EC2Error(
                    message=f"Instance not found: {instance_id}",
                    error_code="InstanceNotFound"
                )

            instance = reservations[0]["Instances"][0]

            return {
                "state": instance.get("State", {}).get("Name", "unknown"),
                "public_ip": instance.get("PublicIpAddress", ""),
                "private_ip": instance.get("PrivateIpAddress", ""),
                "launch_time": instance.get("LaunchTime")
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            raise EC2Error(
                message=f"Failed to get instance status: {error_message}",
                error_code=error_code
            )

    def terminate_instance(self, instance_id: str) -> datetime:
        """
        Terminate EC2 instance and wait for termination.

        Args:
            instance_id: EC2 instance ID to terminate

        Returns:
            datetime: Termination timestamp

        Raises:
            EC2Error: If termination fails
        """
        try:
            # Terminate instance
            response = self._ec2_client.terminate_instances(
                InstanceIds=[instance_id]
            )

            # Wait for termination
            waiter = self._ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[instance_id])

            return datetime.utcnow()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            raise EC2Error(
                message=f"Failed to terminate instance: {error_message}",
                error_code=error_code
            )
        except Exception as e:
            raise EC2Error(
                message=f"Failed to terminate instance: {str(e)}",
                error_code="TerminationError"
            )

    def launch_on_demand_fallback(
        self,
        instance_type: str,
        user_data_script: str,
        tags: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Launch On-Demand EC2 instance as Spot fallback.

        Similar to launch_spot_instance() but uses On-Demand pricing.
        Used when Spot instances are unavailable or capacity is insufficient.

        Args:
            instance_type: EC2 instance type (e.g., "m5.large")
            user_data_script: Bash script to execute on instance startup
            tags: Dictionary of tags to apply to instance

        Returns:
            dict: Launch result with structure:
                {
                    "instance_id": str,  # EC2 instance ID
                    "instance_type": str,  # Instance type used
                    "launch_time": datetime  # Launch timestamp
                }

        Raises:
            EC2Error: If On-Demand instance launch fails
        """
        try:
            # Prepare tag specifications
            tag_specifications = []
            if tags:
                tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
                tag_specifications = [
                    {
                        "ResourceType": "instance",
                        "Tags": tag_list
                    }
                ]

            # Launch parameters
            launch_params = {
                "ImageId": self._config.ec2_ami_id,
                "InstanceType": instance_type,
                "KeyName": self._config.ec2_key_name,
                "SecurityGroups": [self._config.ec2_security_group],
                "UserData": user_data_script,
                "MinCount": 1,
                "MaxCount": 1,
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": 50,
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                            "Encrypted": True
                        }
                    }
                ]
            }

            # Add IAM role if configured
            if self._config.iam_role_arn:
                launch_params["IamInstanceProfile"] = {
                    "Arn": self._config.iam_role_arn
                }

            # Add tag specifications if present
            if tag_specifications:
                launch_params["TagSpecifications"] = tag_specifications

            # Launch On-Demand instance
            response = self._ec2_client.run_instances(**launch_params)

            # Extract instance information
            instances = response.get("Instances", [])
            if not instances:
                raise EC2Error(
                    message="No On-Demand instance created",
                    error_code="NoInstanceReturned"
                )

            instance = instances[0]
            instance_id = instance["InstanceId"]
            launch_time = instance.get("LaunchTime", datetime.utcnow())

            return {
                "instance_id": instance_id,
                "instance_type": instance_type,
                "launch_time": launch_time
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")

            raise EC2Error(
                message=f"Failed to launch On-Demand instance: {error_message}",
                instance_type=instance_type,
                error_code=error_code,
                request_id=request_id
            )
        except EC2Error:
            raise
        except Exception as e:
            raise EC2Error(
                message=f"Unexpected error launching On-Demand instance: {str(e)}",
                instance_type=instance_type,
                error_code="UnexpectedError"
            )
