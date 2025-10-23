"""
Cloud-specific exception hierarchy for AWS operations.

This module defines exceptions for cloud infrastructure errors including
S3 storage, EC2 compute, Lambda functions, and cost tracking operations.
"""

from typing import Optional


class CloudError(Exception):
    """Base exception for all cloud infrastructure errors."""

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize cloud error.

        Args:
            message: Human-readable error description
            service: AWS service name (s3, ec2, lambda, etc.)
            error_code: AWS error code if available
            request_id: AWS request ID for debugging
            **kwargs: Additional error context
        """
        super().__init__(message)
        self.message = message
        self.service = service
        self.error_code = error_code
        self.request_id = request_id

    def __str__(self) -> str:
        """Format error message with AWS context."""
        parts = [self.message]
        if self.service:
            parts.append(f"Service: {self.service}")
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)


class AWSError(CloudError):
    """General AWS service errors not specific to one service."""

    def __init__(
        self,
        message: str,
        service: Optional[str] = "aws",
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AWS error.

        Args:
            message: Human-readable error description
            service: AWS service name (defaults to 'aws')
            error_code: AWS error code if available
            request_id: AWS request ID for debugging
            **kwargs: Additional error context
        """
        super().__init__(
            message=message,
            service=service,
            error_code=error_code,
            request_id=request_id,
            **kwargs
        )


class S3Error(AWSError):
    """S3-specific errors for storage operations."""

    def __init__(
        self,
        message: str,
        bucket: Optional[str] = None,
        key: Optional[str] = None,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize S3 error.

        Args:
            message: Human-readable error description
            bucket: S3 bucket name
            key: S3 object key
            error_code: AWS S3 error code
            request_id: AWS request ID for debugging
            **kwargs: Additional error context
        """
        super().__init__(
            message=message,
            service="s3",
            error_code=error_code,
            request_id=request_id,
            **kwargs
        )
        self.bucket = bucket
        self.key = key

    def __str__(self) -> str:
        """Format S3 error with bucket and key context."""
        parts = [self.message]
        if self.bucket:
            s3_path = f"s3://{self.bucket}"
            if self.key:
                s3_path += f"/{self.key}"
            parts.append(f"Path: {s3_path}")
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)


class EC2Error(AWSError):
    """EC2-specific errors for compute instances."""

    def __init__(
        self,
        message: str,
        instance_id: Optional[str] = None,
        instance_type: Optional[str] = None,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize EC2 error.

        Args:
            message: Human-readable error description
            instance_id: EC2 instance ID
            instance_type: EC2 instance type (e.g., c5.xlarge)
            error_code: AWS EC2 error code
            request_id: AWS request ID for debugging
            **kwargs: Additional error context
        """
        super().__init__(
            message=message,
            service="ec2",
            error_code=error_code,
            request_id=request_id,
            **kwargs
        )
        self.instance_id = instance_id
        self.instance_type = instance_type

    def __str__(self) -> str:
        """Format EC2 error with instance context."""
        parts = [self.message]
        if self.instance_id:
            parts.append(f"Instance: {self.instance_id}")
        if self.instance_type:
            parts.append(f"Type: {self.instance_type}")
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)


class LambdaError(AWSError):
    """Lambda-specific errors for serverless functions."""

    def __init__(
        self,
        message: str,
        function_name: Optional[str] = None,
        invocation_type: Optional[str] = None,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Lambda error.

        Args:
            message: Human-readable error description
            function_name: Lambda function name
            invocation_type: Invocation type (RequestResponse, Event)
            error_code: AWS Lambda error code
            request_id: AWS request ID for debugging
            **kwargs: Additional error context
        """
        super().__init__(
            message=message,
            service="lambda",
            error_code=error_code,
            request_id=request_id,
            **kwargs
        )
        self.function_name = function_name
        self.invocation_type = invocation_type

    def __str__(self) -> str:
        """Format Lambda error with function context."""
        parts = [self.message]
        if self.function_name:
            parts.append(f"Function: {self.function_name}")
        if self.invocation_type:
            parts.append(f"Invocation: {self.invocation_type}")
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)


class CostTrackingError(CloudError):
    """Errors in cost calculation and tracking operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize cost tracking error.

        Args:
            message: Human-readable error description
            operation: Cost tracking operation (estimate, track, report)
            **kwargs: Additional error context
        """
        super().__init__(message=message, service="cost-tracking", **kwargs)
        self.operation = operation

    def __str__(self) -> str:
        """Format cost tracking error with operation context."""
        parts = [self.message]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        return " | ".join(parts)


class CloudConfigurationError(CloudError):
    """Configuration errors for cloud infrastructure setup."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize cloud configuration error.

        Args:
            message: Human-readable error description
            config_key: Configuration key that failed
            config_value: Invalid configuration value
            **kwargs: Additional error context
        """
        super().__init__(message=message, service="configuration", **kwargs)
        self.config_key = config_key
        self.config_value = config_value

    def __str__(self) -> str:
        """Format configuration error with key/value context."""
        parts = [self.message]
        if self.config_key:
            parts.append(f"Key: {self.config_key}")
        if self.config_value:
            parts.append(f"Value: {self.config_value}")
        return " | ".join(parts)
