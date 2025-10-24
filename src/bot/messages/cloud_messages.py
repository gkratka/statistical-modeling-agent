"""
Message templates for cloud-based ML workflows (AWS).

This module provides user-facing message templates for cloud training and prediction
workflows using EC2, Lambda, and S3. Messages follow Telegram Markdown formatting.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.0: Cloud Workflow Telegram Integration)
"""

from typing import Dict

# =============================================================================
# Cloud Training Messages
# =============================================================================

CHOOSE_CLOUD_LOCAL_MESSAGE = """
🌩️ **Training Location**

Where would you like to train this model?

💻 **Local Training** (Free)
  • Runs on this server
  • Limited resources (CPU/RAM)
  • Best for: Small datasets (<1GB), quick experiments
  • No additional costs

☁️ **Cloud Training** (Paid - AWS)
  • Runs on AWS EC2 Spot Instances
  • Scalable resources (up to 64GB RAM, GPU available)
  • Best for: Large datasets (>1GB), neural networks, production models
  • **Cost**: $0.10 - $2.00 per training run

Choose your training environment:
"""


def cloud_instance_confirmation_message(
    instance_type: str,
    estimated_cost_usd: float,
    estimated_time_minutes: int,
    dataset_size_mb: float
) -> str:
    """
    Generate confirmation message before launching cloud training.

    Args:
        instance_type: EC2 instance type (e.g., "m5.large")
        estimated_cost_usd: Estimated training cost in USD
        estimated_time_minutes: Estimated training duration
        dataset_size_mb: Dataset size in megabytes

    Returns:
        str: Formatted message with instance details and cost estimate
    """
    return f"""
☁️ **Cloud Training Configuration**

📊 **Dataset**: {dataset_size_mb:.1f} MB
🖥️ **Instance Type**: {instance_type}
⏱️ **Estimated Time**: ~{estimated_time_minutes} minutes
💰 **Estimated Cost**: ${estimated_cost_usd:.2f}

⚠️ **Important**:
  • You will be charged for actual usage (billed per second)
  • Training logs will stream in real-time
  • Instance will auto-terminate when complete
  • Spot instances may be interrupted (rare, will retry)

Ready to launch cloud training?
"""


def cloud_training_launched_message(
    instance_id: str,
    instance_type: str
) -> str:
    """
    Generate message when EC2 instance successfully launched.

    Args:
        instance_id: EC2 instance ID (e.g., "i-1234567890abcdef0")
        instance_type: Instance type (e.g., "m5.large")

    Returns:
        str: Launch confirmation message
    """
    return f"""
🚀 **Cloud Training Launched**

Instance ID: `{instance_id}`
Instance Type: {instance_type}
Status: Launching...

⏳ Waiting for instance to start (typically 1-2 minutes)...

I'll stream the training logs here as they become available.
"""


def cloud_training_log_message(log_line: str) -> str:
    """
    Format a single log line for Telegram.

    Args:
        log_line: Raw log line from CloudWatch

    Returns:
        str: Formatted log message with icon
    """
    return f"📝 `{log_line}`"


def cloud_training_complete_message(
    model_id: str,
    s3_model_uri: str,
    training_time_minutes: float,
    actual_cost_usd: float,
    metrics: Dict[str, float]
) -> str:
    """
    Generate completion message after successful training.

    Args:
        model_id: Unique model identifier
        s3_model_uri: S3 URI where model is stored
        training_time_minutes: Actual training duration
        actual_cost_usd: Actual cost incurred
        metrics: Dictionary of training metrics (e.g., {"r2": 0.85, "mse": 0.12})

    Returns:
        str: Training completion message with metrics and cost
    """
    metrics_str = "\n".join([f"  • {k}: {v:.4f}" for k, v in metrics.items()])

    return f"""
✅ **Cloud Training Complete!**

🎯 **Model ID**: `{model_id}`
📦 **S3 Location**: `{s3_model_uri}`
⏱️ **Training Time**: {training_time_minutes:.1f} minutes
💰 **Actual Cost**: ${actual_cost_usd:.2f}

📊 **Metrics**:
{metrics_str}

The model has been saved to S3 and can be used for predictions.

Use /predict to run predictions with this model.
"""


def cloud_training_progress_message(
    progress_percentage: int,
    current_step: str
) -> str:
    """
    Generate progress update message during training.

    Args:
        progress_percentage: Training progress (0-100)
        current_step: Current training step description

    Returns:
        str: Progress message
    """
    # Create progress bar
    filled = int(progress_percentage / 10)
    empty = 10 - filled
    progress_bar = "█" * filled + "░" * empty

    return f"""
⏳ **Training Progress**: {progress_percentage}%

{progress_bar}

{current_step}
"""


# =============================================================================
# Cloud Prediction Messages
# =============================================================================

def cloud_prediction_launched_message(
    request_id: str,
    num_rows: int
) -> str:
    """
    Generate message when Lambda prediction launched.

    Args:
        request_id: Lambda request ID
        num_rows: Number of rows in prediction dataset

    Returns:
        str: Launch confirmation message
    """
    return f"""
🚀 **Cloud Prediction Launched**

Request ID: `{request_id}`
Rows: {num_rows:,}
Status: Processing...

⏳ Lambda function is running...

This should complete in 1-2 minutes.
"""


def cloud_prediction_complete_message(
    s3_output_uri: str,
    num_predictions: int,
    execution_time_ms: int,
    cost_usd: float,
    presigned_url: str
) -> str:
    """
    Generate completion message after successful predictions.

    Args:
        s3_output_uri: S3 URI where results are stored
        num_predictions: Number of predictions generated
        execution_time_ms: Lambda execution time in milliseconds
        cost_usd: Cost of Lambda invocation
        presigned_url: Pre-signed download URL (expires in 1 hour)

    Returns:
        str: Prediction completion message with download link
    """
    return f"""
✅ **Cloud Prediction Complete!**

📦 **Output S3 URI**: `{s3_output_uri}`
📊 **Predictions**: {num_predictions:,} rows
⏱️ **Execution Time**: {execution_time_ms}ms
💰 **Cost**: ${cost_usd:.4f}

📥 **Download Results**:
[Click here to download]({presigned_url})
(Link expires in 1 hour)

Or access via S3 URI in your AWS account.
"""


# =============================================================================
# Error Messages
# =============================================================================

def cloud_error_message(error_type: str, error_details: str) -> str:
    """
    Generate error message for cloud operation failures.

    Args:
        error_type: Type of error (e.g., "EC2LaunchError", "LambdaTimeout")
        error_details: Detailed error description

    Returns:
        str: Formatted error message
    """
    return f"""
❌ **Cloud Operation Failed**

**Error Type**: {error_type}
**Details**: {error_details}

No charges were incurred for failed operations.

Please check your AWS configuration or try again.
"""


def s3_validation_error_message(s3_uri: str, reason: str) -> str:
    """
    Generate error message for S3 path validation failures.

    Args:
        s3_uri: Invalid S3 URI provided by user
        reason: Reason for validation failure

    Returns:
        str: Formatted validation error message
    """
    return f"""
❌ **Cannot Access S3 Path**

**S3 URI**: `{s3_uri}`
**Reason**: {reason}

**Possible Issues**:
  • Bucket doesn't exist or is private
  • File path is incorrect
  • Missing read permissions
  • Region mismatch

**Next Steps**:
1. Verify the S3 URI is correct
2. Check bucket permissions
3. Ensure the bot's IAM role has S3 read access

Try again or use /cancel to exit.
"""


def ec2_spot_interruption_message(instance_id: str, warning_time_seconds: int) -> str:
    """
    Generate message when EC2 Spot instance receives interruption warning.

    Args:
        instance_id: EC2 instance ID being interrupted
        warning_time_seconds: Seconds until termination

    Returns:
        str: Spot interruption warning message
    """
    return f"""
⚠️ **Spot Instance Interruption Warning**

Instance ID: `{instance_id}`
Time Remaining: {warning_time_seconds} seconds

AWS is reclaiming this Spot instance. The system will:
1. Attempt to save partial progress to S3
2. Automatically retry training with On-Demand instance
3. Send you an update once training completes

No action needed from you. Training will continue automatically.
"""


def cost_estimate_message(
    operation_type: str,
    compute_cost: float,
    storage_cost: float,
    transfer_cost: float
) -> str:
    """
    Generate detailed cost breakdown message.

    Args:
        operation_type: "Training" or "Prediction"
        compute_cost: Compute cost (EC2 or Lambda)
        storage_cost: S3 storage cost
        transfer_cost: Data transfer cost

    Returns:
        str: Detailed cost estimate
    """
    total_cost = compute_cost + storage_cost + transfer_cost

    return f"""
💰 **{operation_type} Cost Estimate**

**Compute**: ${compute_cost:.4f}
**Storage**: ${storage_cost:.4f}
**Data Transfer**: ${transfer_cost:.4f}

**Total Estimated Cost**: ${total_cost:.2f}

This is an estimate. Actual costs may vary based on:
  • Actual execution time
  • Resource usage
  • AWS pricing changes

Proceed with {operation_type.lower()}?
"""


# =============================================================================
# Dataset Upload Messages
# =============================================================================

AWAITING_S3_DATASET_MESSAGE = """
☁️ **Cloud Dataset Input**

Please provide your dataset:

1️⃣ **Upload CSV File**
  • Send file directly via Telegram
  • Will be automatically uploaded to S3
  • Max size: 10GB

2️⃣ **Provide S3 URI**
  • Reply with S3 path (e.g., `s3://my-bucket/data.csv`)
  • Ensure the bot's IAM role has read access
  • Supported formats: CSV, Parquet, Excel

Which option would you like to use?
"""


def telegram_file_upload_progress_message(
    filename: str,
    size_mb: float,
    progress_percentage: int
) -> str:
    """
    Generate progress message during Telegram file upload to S3.

    Args:
        filename: Name of file being uploaded
        size_mb: File size in megabytes
        progress_percentage: Upload progress (0-100)

    Returns:
        str: Upload progress message
    """
    filled = int(progress_percentage / 10)
    empty = 10 - filled
    progress_bar = "█" * filled + "░" * empty

    return f"""
📤 **Uploading to S3**

File: `{filename}`
Size: {size_mb:.1f} MB
Progress: {progress_percentage}%

{progress_bar}

Please wait...
"""


def s3_upload_complete_message(s3_uri: str, dataset_size_mb: float, row_count: int, column_count: int) -> str:
    """
    Generate confirmation message after successful S3 upload.

    Args:
        s3_uri: S3 URI where dataset was uploaded
        dataset_size_mb: Dataset size in megabytes
        row_count: Number of rows in dataset
        column_count: Number of columns in dataset

    Returns:
        str: Upload confirmation message
    """
    return f"""
✅ **Dataset Uploaded to S3**

**S3 URI**: `{s3_uri}`
**Size**: {dataset_size_mb:.1f} MB
**Rows**: {row_count:,}
**Columns**: {column_count}

Dataset is ready for training!

Proceeding to next step...
"""
