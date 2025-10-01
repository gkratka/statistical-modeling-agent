"""
Comprehensive error handling system for script generation and execution.

This module provides structured error handling, recovery mechanisms,
and detailed error reporting for all components of the system.
"""

import traceback
import sys
import asyncio
from typing import Dict, Any, Optional, List, Type, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.logger import get_logger
from src.utils.exceptions import (
    AgentError, ValidationError, ParseError, ExecutionError,
    SecurityViolationError, ResourceLimitError, TemplateError,
    ScriptGenerationError
)

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    SECURITY = "security"
    RESOURCE = "resource"
    EXECUTION = "execution"
    TEMPLATE = "template"
    NETWORK = "network"
    DATA = "data"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error reporting."""
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    context: ErrorContext
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    user_message: Optional[str] = None
    technical_details: Dict[str, Any] = field(default_factory=dict)


class ErrorHandler:
    """Comprehensive error handling and reporting system."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.error_mappings = self._init_error_mappings()
        self.recovery_strategies = self._init_recovery_strategies()
        self.error_count = 0

    def _init_error_mappings(self) -> Dict[Type[Exception], Dict[str, Any]]:
        """Initialize error type mappings."""
        return {
            ValidationError: {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "The provided data or parameters are invalid. Please check your input and try again."
            },
            SecurityViolationError: {
                "category": ErrorCategory.SECURITY,
                "severity": ErrorSeverity.HIGH,
                "user_message": "A security violation was detected. The operation has been blocked for safety."
            },
            ResourceLimitError: {
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.HIGH,
                "user_message": "Resource limits exceeded. Please try with smaller data or simpler operations."
            },
            ExecutionError: {
                "category": ErrorCategory.EXECUTION,
                "severity": ErrorSeverity.HIGH,
                "user_message": "Script execution failed. Please check your data and try again."
            },
            TemplateError: {
                "category": ErrorCategory.TEMPLATE,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Template processing failed. Please contact support if this persists."
            },
            ScriptGenerationError: {
                "category": ErrorCategory.TEMPLATE,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Script generation failed. Please check your parameters and try again."
            },
            ParseError: {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Unable to parse your request. Please rephrase and try again."
            },
            FileNotFoundError: {
                "category": ErrorCategory.SYSTEM,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Required file not found. Please check your data upload."
            },
            PermissionError: {
                "category": ErrorCategory.SYSTEM,
                "severity": ErrorSeverity.HIGH,
                "user_message": "Permission denied. Please contact support."
            },
            MemoryError: {
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.CRITICAL,
                "user_message": "Insufficient memory. Please try with smaller data."
            },
            TimeoutError: {
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.HIGH,
                "user_message": "Operation timed out. Please try again or use smaller data."
            },
            asyncio.TimeoutError: {
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.HIGH,
                "user_message": "Operation timed out. Please try again or use smaller data."
            },
            ConnectionError: {
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Network connection failed. Please try again later."
            },
            ValueError: {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Invalid value provided. Please check your input."
            },
            TypeError: {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Invalid data type. Please check your input format."
            },
            KeyError: {
                "category": ErrorCategory.DATA,
                "severity": ErrorSeverity.MEDIUM,
                "user_message": "Required data field missing. Please check your data format."
            },
            ImportError: {
                "category": ErrorCategory.SYSTEM,
                "severity": ErrorSeverity.HIGH,
                "user_message": "System dependency missing. Please contact support."
            },
            OSError: {
                "category": ErrorCategory.SYSTEM,
                "severity": ErrorSeverity.HIGH,
                "user_message": "System error occurred. Please try again or contact support."
            }
        }

    def _init_recovery_strategies(self) -> Dict[ErrorCategory, List[str]]:
        """Initialize recovery strategies for each error category."""
        return {
            ErrorCategory.VALIDATION: [
                "Validate and sanitize input parameters",
                "Provide clear validation error messages",
                "Suggest correct input format",
                "Use default values for optional parameters"
            ],
            ErrorCategory.SECURITY: [
                "Block the operation immediately",
                "Log security violation details",
                "Alert system administrators",
                "Review and strengthen security measures"
            ],
            ErrorCategory.RESOURCE: [
                "Reduce resource usage (smaller data, simpler operations)",
                "Implement resource pooling",
                "Use streaming for large data",
                "Optimize algorithm complexity"
            ],
            ErrorCategory.EXECUTION: [
                "Retry with exponential backoff",
                "Fallback to simpler algorithms",
                "Check data quality and format",
                "Provide alternative approaches"
            ],
            ErrorCategory.TEMPLATE: [
                "Use fallback templates",
                "Regenerate with simplified parameters",
                "Check template syntax and variables",
                "Update template registry"
            ],
            ErrorCategory.NETWORK: [
                "Retry with exponential backoff",
                "Use cached data if available",
                "Implement circuit breaker pattern",
                "Provide offline mode"
            ],
            ErrorCategory.DATA: [
                "Validate data format and structure",
                "Clean and preprocess data",
                "Use data imputation for missing values",
                "Provide data format examples"
            ],
            ErrorCategory.SYSTEM: [
                "Check system dependencies",
                "Restart affected services",
                "Scale system resources",
                "Contact system administrators"
            ]
        }

    async def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        include_traceback: bool = True,
        attempt_recovery: bool = True
    ) -> ErrorReport:
        """
        Handle an error and generate comprehensive error report.

        Args:
            error: The exception that occurred
            context: Context information about the error
            include_traceback: Whether to include stack trace
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            ErrorReport with detailed error information
        """
        self.error_count += 1
        error_id = f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.error_count:04d}"

        # Get error mapping
        error_mapping = self._get_error_mapping(type(error))

        # Create error report
        report = ErrorReport(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=error_mapping["severity"],
            category=error_mapping["category"],
            error_type=type(error).__name__,
            message=str(error),
            context=context,
            user_message=error_mapping.get("user_message"),
            stack_trace=traceback.format_exc() if include_traceback else None,
            recovery_suggestions=self.recovery_strategies.get(
                error_mapping["category"], []
            )
        )

        # Add technical details
        report.technical_details = {
            "python_version": sys.version,
            "error_class": error.__class__.__module__ + "." + error.__class__.__name__,
            "error_args": error.args,
            "context_operation": context.operation,
            "context_parameters": context.parameters
        }

        # Log the error
        await self._log_error(report)

        # Attempt recovery if requested
        if attempt_recovery:
            await self._attempt_recovery(report)

        return report

    def _get_error_mapping(self, error_type: Type[Exception]) -> Dict[str, Any]:
        """Get error mapping for exception type."""
        # Check for exact match first
        if error_type in self.error_mappings:
            return self.error_mappings[error_type]

        # Check for parent class matches
        for mapped_type, mapping in self.error_mappings.items():
            if issubclass(error_type, mapped_type):
                return mapping

        # Default mapping for unknown errors
        return {
            "category": ErrorCategory.UNKNOWN,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": "An unexpected error occurred. Please try again or contact support."
        }

    async def _log_error(self, report: ErrorReport) -> None:
        """Log error report with appropriate level."""
        log_message = (
            f"Error {report.error_id}: {report.error_type} in {report.context.operation} - "
            f"{report.message}"
        )

        if report.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={"error_report": report})
        elif report.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={"error_report": report})
        elif report.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={"error_report": report})
        else:
            logger.info(log_message, extra={"error_report": report})

    async def _attempt_recovery(self, report: ErrorReport) -> None:
        """Attempt automatic recovery based on error category."""
        try:
            if report.category == ErrorCategory.RESOURCE:
                await self._recover_resource_error(report)
            elif report.category == ErrorCategory.EXECUTION:
                await self._recover_execution_error(report)
            elif report.category == ErrorCategory.TEMPLATE:
                await self._recover_template_error(report)
            # Add more recovery strategies as needed

        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed for {report.error_id}: {recovery_error}")

    async def _recover_resource_error(self, report: ErrorReport) -> None:
        """Attempt recovery for resource-related errors."""
        # Implementation depends on specific resource constraints
        logger.info(f"Attempting resource error recovery for {report.error_id}")

    async def _recover_execution_error(self, report: ErrorReport) -> None:
        """Attempt recovery for execution-related errors."""
        # Implementation depends on execution context
        logger.info(f"Attempting execution error recovery for {report.error_id}")

    async def _recover_template_error(self, report: ErrorReport) -> None:
        """Attempt recovery for template-related errors."""
        # Implementation depends on template system
        logger.info(f"Attempting template error recovery for {report.error_id}")

    def create_user_response(self, report: ErrorReport) -> Dict[str, Any]:
        """
        Create user-friendly error response.

        Args:
            report: Error report to convert

        Returns:
            Dictionary with user-friendly error information
        """
        return {
            "success": False,
            "error": {
                "id": report.error_id,
                "message": report.user_message or "An error occurred during processing.",
                "type": "user_error" if report.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] else "system_error",
                "suggestions": self._get_user_suggestions(report),
                "timestamp": report.timestamp.isoformat()
            }
        }

    def _get_user_suggestions(self, report: ErrorReport) -> List[str]:
        """Get user-friendly suggestions based on error category."""
        suggestions = {
            ErrorCategory.VALIDATION: [
                "Check your input data format",
                "Ensure all required fields are provided",
                "Verify data types match expected format"
            ],
            ErrorCategory.RESOURCE: [
                "Try with smaller dataset",
                "Use simpler analysis methods",
                "Wait a moment and try again"
            ],
            ErrorCategory.EXECUTION: [
                "Check your data for any formatting issues",
                "Try a different analysis approach",
                "Ensure your data has enough samples"
            ],
            ErrorCategory.SECURITY: [
                "Review your input for any restricted content",
                "Contact support if you believe this is an error"
            ]
        }

        return suggestions.get(report.category, [
            "Try again in a few moments",
            "Contact support if the problem persists"
        ])

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_errors": self.error_count,
            "supported_error_types": len(self.error_mappings),
            "recovery_strategies": len(self.recovery_strategies)
        }


# Global error handler instance
_global_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler


async def handle_error(
    error: Exception,
    operation: str,
    user_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> ErrorReport:
    """
    Convenience function to handle errors with minimal context.

    Args:
        error: The exception that occurred
        operation: Name of the operation that failed
        user_id: User ID if available
        parameters: Operation parameters if available

    Returns:
        ErrorReport with error details
    """
    context = ErrorContext(
        operation=operation,
        user_id=user_id,
        parameters=parameters or {}
    )

    handler = get_error_handler()
    return await handler.handle_error(error, context)


def safe_execute(func: Callable, *args, **kwargs) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or None if error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return None


async def safe_execute_async(func: Callable, *args, **kwargs) -> Any:
    """
    Safely execute an async function with error handling.

    Args:
        func: Async function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or None if error occurred
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe async execution failed for {func.__name__}: {e}")
        return None