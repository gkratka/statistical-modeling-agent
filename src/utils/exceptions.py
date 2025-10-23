"""
Custom exception hierarchy for the Statistical Modeling Agent.

This module defines the exception hierarchy as specified in CLAUDE.md
to enable proper error handling and propagation throughout the system.
"""

from typing import List, Tuple, Any, Optional


class AgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR") -> None:
        """
        Initialize agent error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for API responses
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ValidationError(AgentError):
    """Input validation failures."""

    def __init__(self, message: str, field: str = "", value: str = "") -> None:
        """
        Initialize validation error.

        Args:
            message: Human-readable error message
            field: Name of the field that failed validation
            value: Value that failed validation (for logging)
        """
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value


class ParseError(AgentError):
    """Natural language parsing failures."""

    def __init__(self, message: str, raw_input: str = "") -> None:
        """
        Initialize parse error.

        Args:
            message: Human-readable error message
            raw_input: Original user input that failed parsing
        """
        super().__init__(message, "PARSE_ERROR")
        self.raw_input = raw_input


class ExecutionError(AgentError):
    """Script execution failures."""

    def __init__(self, message: str, script: str = "", stderr: str = "") -> None:
        """
        Initialize execution error.

        Args:
            message: Human-readable error message
            script: Script that failed to execute
            stderr: Standard error output from failed execution
        """
        super().__init__(message, "EXECUTION_ERROR")
        self.script = script
        self.stderr = stderr


class DataError(ValidationError):
    """Data-specific validation failures."""

    def __init__(
        self,
        message: str,
        data_shape: Tuple[int, ...] = (),
        missing_columns: Optional[List[str]] = None
    ) -> None:
        """
        Initialize data error.

        Args:
            message: Human-readable error message
            data_shape: Shape of the problematic dataset
            missing_columns: List of missing required columns
        """
        super().__init__(message, "data", "")
        self.data_shape = data_shape
        self.missing_columns = missing_columns or []


class BotError(AgentError):
    """Telegram bot-specific errors."""

    def __init__(self, message: str, user_id: int = 0, chat_id: int = 0) -> None:
        """
        Initialize bot error.

        Args:
            message: Human-readable error message
            user_id: Telegram user ID associated with error
            chat_id: Telegram chat ID associated with error
        """
        super().__init__(message, "BOT_ERROR")
        self.user_id = user_id
        self.chat_id = chat_id


class ConfigurationError(AgentError):
    """Configuration and environment setup errors."""

    def __init__(self, message: str, config_key: str = "") -> None:
        """
        Initialize configuration error.

        Args:
            message: Human-readable error message
            config_key: Configuration key that caused the error
        """
        super().__init__(message, "CONFIG_ERROR")
        self.config_key = config_key


class ScriptGenerationError(AgentError):
    """Script generation failures."""

    def __init__(self, message: str, operation: str = "", template: str = "") -> None:
        """
        Initialize script generation error.

        Args:
            message: Human-readable error message
            operation: Operation that failed
            template: Template that was being used
        """
        super().__init__(message, "SCRIPT_GENERATION_ERROR")
        self.operation = operation
        self.template = template


class SecurityViolationError(ExecutionError):
    """Security constraint violations."""

    def __init__(self, message: str, violations: Optional[List[Any]] = None) -> None:
        """
        Initialize security violation error.

        Args:
            message: Human-readable error message
            violations: List of security violations
        """
        super().__init__(message, "", "")
        self.violations = violations or []


class ResourceLimitError(ExecutionError):
    """Resource limit exceeded."""

    def __init__(self, message: str, resource_type: str = "", limit_value: str = "") -> None:
        """
        Initialize resource limit error.

        Args:
            message: Human-readable error message
            resource_type: Type of resource (memory, cpu, time)
            limit_value: The limit that was exceeded
        """
        super().__init__(message, "", "")
        self.resource_type = resource_type
        self.limit_value = limit_value


class TemplateError(ScriptGenerationError):
    """Template-related errors."""

    def __init__(self, message: str, template_path: str = "", line_number: int = 0) -> None:
        """
        Initialize template error.

        Args:
            message: Human-readable error message
            template_path: Path to template that caused error
            line_number: Line number in template if available
        """
        super().__init__(message, "", template_path)
        self.template_path = template_path
        self.line_number = line_number


# ============================================================================
# ML-Specific Exceptions
# ============================================================================


class MLError(AgentError):
    """Base exception for ML operations."""

    def __init__(self, message: str, operation: str = "") -> None:
        """
        Initialize ML error.

        Args:
            message: Human-readable error message
            operation: ML operation that failed (train, predict, etc.)
        """
        super().__init__(message, "ML_ERROR")
        self.operation = operation


class DataValidationError(MLError):
    """Input data validation failures for ML operations."""

    def __init__(
        self,
        message: str,
        data_shape: Tuple[int, ...] = (),
        missing_columns: Optional[List[str]] = None,
        invalid_columns: Optional[List[str]] = None
    ) -> None:
        """
        Initialize data validation error.

        Args:
            message: Human-readable error message
            data_shape: Shape of the problematic dataset
            missing_columns: List of missing required columns
            invalid_columns: List of columns with invalid data
        """
        super().__init__(message, "data_validation")
        self.data_shape = data_shape
        self.missing_columns = missing_columns or []
        self.invalid_columns = invalid_columns or []


class ModelNotFoundError(MLError):
    """Model doesn't exist or user doesn't own it."""

    def __init__(self, message: str, model_id: str = "", user_id: int = 0) -> None:
        """
        Initialize model not found error.

        Args:
            message: Human-readable error message
            model_id: ID of the model that wasn't found
            user_id: User ID attempting to access the model
        """
        super().__init__(message, "model_access")
        self.model_id = model_id
        self.user_id = user_id


class TrainingError(MLError):
    """Model training failures."""

    def __init__(
        self,
        message: str,
        model_type: str = "",
        training_time: float = 0.0,
        error_details: str = ""
    ) -> None:
        """
        Initialize training error.

        Args:
            message: Human-readable error message
            model_type: Type of model being trained
            training_time: Time elapsed before failure
            error_details: Detailed error information
        """
        super().__init__(message, "train_model")
        self.model_type = model_type
        self.training_time = training_time
        self.error_details = error_details


class PredictionError(MLError):
    """Prediction execution failures."""

    def __init__(
        self,
        message: str,
        model_id: str = "",
        num_samples: int = 0,
        error_details: str = ""
    ) -> None:
        """
        Initialize prediction error.

        Args:
            message: Human-readable error message
            model_id: ID of the model used for prediction
            num_samples: Number of samples attempted to predict
            error_details: Detailed error information
        """
        super().__init__(message, "predict")
        self.model_id = model_id
        self.num_samples = num_samples
        self.error_details = error_details


class FeatureMismatchError(PredictionError):
    """Prediction data doesn't match model schema."""

    def __init__(
        self,
        message: str,
        expected_features: Optional[List[str]] = None,
        provided_features: Optional[List[str]] = None,
        missing_features: Optional[List[str]] = None,
        extra_features: Optional[List[str]] = None
    ) -> None:
        """
        Initialize feature mismatch error.

        Args:
            message: Human-readable error message
            expected_features: Features expected by the model
            provided_features: Features provided in prediction data
            missing_features: Features required but not provided
            extra_features: Features provided but not required
        """
        super().__init__(message)
        self.expected_features = expected_features or []
        self.provided_features = provided_features or []
        self.missing_features = missing_features or []
        self.extra_features = extra_features or []


class ConvergenceError(TrainingError):
    """Model failed to converge during training."""

    def __init__(
        self,
        message: str,
        model_type: str = "",
        iterations: int = 0,
        tolerance: float = 0.0
    ) -> None:
        """
        Initialize convergence error.

        Args:
            message: Human-readable error message
            model_type: Type of model that failed to converge
            iterations: Number of iterations attempted
            tolerance: Convergence tolerance threshold
        """
        super().__init__(message, model_type)
        self.iterations = iterations
        self.tolerance = tolerance


class HyperparameterError(MLError):
    """Invalid hyperparameter values."""

    def __init__(
        self,
        message: str,
        parameter_name: str = "",
        parameter_value: Any = None,
        allowed_range: Optional[Tuple[Any, Any]] = None
    ) -> None:
        """
        Initialize hyperparameter error.

        Args:
            message: Human-readable error message
            parameter_name: Name of invalid hyperparameter
            parameter_value: Invalid value provided
            allowed_range: Allowed range for the parameter
        """
        super().__init__(message, "hyperparameter_validation")
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.allowed_range = allowed_range


class ModelSerializationError(MLError):
    """Model serialization/deserialization failures."""

    def __init__(
        self,
        message: str,
        model_id: str = "",
        operation: str = "",
        file_path: str = ""
    ) -> None:
        """
        Initialize model serialization error.

        Args:
            message: Human-readable error message
            model_id: ID of the model
            operation: Operation that failed (save/load)
            file_path: Path to the model file
        """
        super().__init__(message, operation)
        self.model_id = model_id
        self.file_path = file_path


# ============================================================================
# State Management Exceptions
# ============================================================================


class StateError(AgentError):
    """Base exception for state management errors."""

    def __init__(self, message: str, session_key: str = "") -> None:
        """
        Initialize state error.

        Args:
            message: Human-readable error message
            session_key: Session key associated with error
        """
        super().__init__(message, "STATE_ERROR")
        self.session_key = session_key


class SessionNotFoundError(StateError):
    """Session doesn't exist."""

    def __init__(self, message: str, user_id: int = 0, conversation_id: str = "") -> None:
        """
        Initialize session not found error.

        Args:
            message: Human-readable error message
            user_id: User ID of missing session
            conversation_id: Conversation ID of missing session
        """
        session_key = f"{user_id}_{conversation_id}" if user_id and conversation_id else ""
        super().__init__(message, session_key)
        self.user_id = user_id
        self.conversation_id = conversation_id


class SessionExpiredError(StateError):
    """Session has expired due to timeout."""

    def __init__(
        self,
        message: str,
        session_key: str = "",
        expired_at: str = "",
        timeout_minutes: int = 0
    ) -> None:
        """
        Initialize session expired error.

        Args:
            message: Human-readable error message
            session_key: Key of expired session
            expired_at: Timestamp when session expired
            timeout_minutes: Timeout duration in minutes
        """
        super().__init__(message, session_key)
        self.expired_at = expired_at
        self.timeout_minutes = timeout_minutes


class InvalidStateTransitionError(StateError):
    """State transition is not valid for current workflow."""

    def __init__(
        self,
        message: str,
        current_state: str = "",
        requested_state: str = "",
        workflow_type: str = ""
    ) -> None:
        """
        Initialize invalid state transition error.

        Args:
            message: Human-readable error message
            current_state: Current workflow state
            requested_state: Requested new state
            workflow_type: Type of workflow
        """
        super().__init__(message)
        self.current_state = current_state
        self.requested_state = requested_state
        self.workflow_type = workflow_type


class PrerequisiteNotMetError(StateError):
    """State transition prerequisite not satisfied."""

    def __init__(
        self,
        message: str,
        state: str = "",
        missing_prerequisites: Optional[List[str]] = None
    ) -> None:
        """
        Initialize prerequisite not met error.

        Args:
            message: Human-readable error message
            state: State that requires prerequisites
            missing_prerequisites: List of missing prerequisites
        """
        super().__init__(message)
        self.state = state
        self.missing_prerequisites = missing_prerequisites or []


class DataSizeLimitError(StateError):
    """Uploaded data exceeds size limit."""

    def __init__(
        self,
        message: str,
        actual_size_mb: float = 0.0,
        limit_mb: int = 0
    ) -> None:
        """
        Initialize data size limit error.

        Args:
            message: Human-readable error message
            actual_size_mb: Actual data size in MB
            limit_mb: Maximum allowed size in MB
        """
        super().__init__(message)
        self.actual_size_mb = actual_size_mb
        self.limit_mb = limit_mb


class SessionLimitError(StateError):
    """Maximum number of concurrent sessions reached."""

    def __init__(
        self,
        message: str,
        current_count: int = 0,
        limit: int = 0
    ) -> None:
        """
        Initialize session limit error.

        Args:
            message: Human-readable error message
            current_count: Current number of sessions
            limit: Maximum allowed sessions
        """
        super().__init__(message)
        self.current_count = current_count
        self.limit = limit


# ============================================================================
# Local File Path Exceptions (Phase 1: Security Foundation)
# ============================================================================


class PathValidationError(ValidationError):
    """Local file path validation failures."""

    def __init__(
        self,
        message: str,
        path: str = "",
        reason: str = ""
    ) -> None:
        """
        Initialize path validation error.

        Args:
            message: Human-readable error message
            path: File path that failed validation
            reason: Specific reason for validation failure
        """
        super().__init__(message, "file_path", path)
        self.path = path
        self.reason = reason


# ============================================================================
# Cloud Infrastructure Exceptions
# ============================================================================

# Import cloud-specific exceptions for centralized access
from src.cloud.exceptions import (  # noqa: E402
    CloudError,
    AWSError,
    S3Error,
    EC2Error,
    LambdaError,
    CostTrackingError,
    CloudConfigurationError,
)

__all__ = [
    # Base exceptions
    "AgentError",
    "ValidationError",
    "ParseError",
    "ExecutionError",
    "DataError",
    "BotError",
    "ConfigurationError",
    "ScriptGenerationError",
    "SecurityViolationError",
    "ResourceLimitError",
    "TemplateError",
    # ML exceptions
    "MLError",
    "DataValidationError",
    "ModelNotFoundError",
    "TrainingError",
    "PredictionError",
    "FeatureMismatchError",
    "ConvergenceError",
    "HyperparameterError",
    "ModelSerializationError",
    # State management
    "StateError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "InvalidStateTransitionError",
    "PrerequisiteNotMetError",
    "DataSizeLimitError",
    "SessionLimitError",
    # Path validation
    "PathValidationError",
    # Cloud exceptions
    "CloudError",
    "AWSError",
    "S3Error",
    "EC2Error",
    "LambdaError",
    "CostTrackingError",
    "CloudConfigurationError",
]