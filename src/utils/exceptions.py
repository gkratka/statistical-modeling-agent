"""
Custom exception hierarchy for the Statistical Modeling Agent.

This module defines the exception hierarchy as specified in CLAUDE.md
to enable proper error handling and propagation throughout the system.
"""


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
        data_shape: tuple[int, ...] = (),
        missing_columns: list[str] = None
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