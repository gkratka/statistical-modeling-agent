"""
Input sanitization utilities for secure parameter processing.

This module provides comprehensive sanitization functions to ensure
all user inputs are safe before being used in script generation.
"""

import re
import html
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class InputSanitizer:
    """Sanitizes various types of user inputs for security."""

    def __init__(self) -> None:
        """Initialize the sanitizer with security patterns."""
        self.dangerous_chars = r'[<>&"\'\`\$\{\}\[\]\(\);|]'
        self.sql_injection_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(?i)(script|javascript|vbscript|onload|onerror)",
            r"(?i)(\-\-|\#|\/\*|\*\/)",
        ]
        self.path_traversal_patterns = [
            r"\.\.",
            r"\/\.\.",
            r"\\\.\.",
            r"~\/",
            r"~\\",
        ]

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize a string input for safe usage.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input is dangerous or invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")

        # Check length
        if len(value) > max_length:
            raise ValidationError(f"String too long: {len(value)} > {max_length}")

        # Check for dangerous characters
        if re.search(self.dangerous_chars, value):
            logger.warning(f"Dangerous characters found in string: {value}")
            raise ValidationError("String contains dangerous characters")

        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value):
                logger.warning(f"Potential SQL injection detected: {value}")
                raise ValidationError("String contains SQL injection patterns")

        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, value):
                logger.warning(f"Path traversal detected: {value}")
                raise ValidationError("String contains path traversal patterns")

        # HTML escape for safety
        sanitized = html.escape(value)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Strip whitespace
        sanitized = sanitized.strip()

        logger.debug(f"String sanitized: {value} -> {sanitized}")
        return sanitized

    def sanitize_column_name(self, name: str) -> str:
        """
        Sanitize a column name for safe usage in scripts.

        Args:
            name: Column name to sanitize

        Returns:
            Sanitized column name

        Raises:
            ValidationError: If column name is invalid
        """
        if not isinstance(name, str):
            raise ValidationError(f"Column name must be string, got {type(name)}")

        if not name.strip():
            raise ValidationError("Column name cannot be empty")

        # Remove dangerous characters, keep only alphanumeric and underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())

        # Ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f'_{sanitized}'

        # Ensure it's not empty after sanitization
        if not sanitized:
            raise ValidationError("Column name becomes empty after sanitization")

        # Check against Python keywords
        python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'with', 'yield', 'True', 'False',
            'None'
        }

        if sanitized.lower() in python_keywords:
            sanitized = f'{sanitized}_col'

        logger.debug(f"Column name sanitized: {name} -> {sanitized}")
        return sanitized

    def sanitize_number(self, value: Union[int, float, str],
                       min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> Union[int, float]:
        """
        Sanitize a numeric input.

        Args:
            value: Numeric value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Sanitized numeric value

        Raises:
            ValidationError: If value is invalid or out of range
        """
        # Convert to number if string
        if isinstance(value, str):
            try:
                # Try int first, then float
                if '.' in value or 'e' in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                raise ValidationError(f"Cannot convert '{value}' to number")

        if not isinstance(value, (int, float)):
            raise ValidationError(f"Expected number, got {type(value)}")

        # Check for special float values
        if isinstance(value, float):
            if not (value == value):  # NaN check
                raise ValidationError("NaN values not allowed")
            if value == float('inf') or value == float('-inf'):
                raise ValidationError("Infinite values not allowed")

        # Check range
        if min_val is not None and value < min_val:
            raise ValidationError(f"Value {value} below minimum {min_val}")

        if max_val is not None and value > max_val:
            raise ValidationError(f"Value {value} above maximum {max_val}")

        logger.debug(f"Number sanitized: {value}")
        return value

    def sanitize_list(self, value: List[Any],
                     max_length: int = 1000,
                     item_sanitizer: Optional[callable] = None) -> List[Any]:
        """
        Sanitize a list input.

        Args:
            value: List to sanitize
            max_length: Maximum allowed list length
            item_sanitizer: Function to sanitize individual items

        Returns:
            Sanitized list

        Raises:
            ValidationError: If list is invalid
        """
        if not isinstance(value, list):
            raise ValidationError(f"Expected list, got {type(value)}")

        if len(value) > max_length:
            raise ValidationError(f"List too long: {len(value)} > {max_length}")

        if not value:
            raise ValidationError("List cannot be empty")

        sanitized = []
        for i, item in enumerate(value):
            try:
                if item_sanitizer:
                    sanitized_item = item_sanitizer(item)
                else:
                    sanitized_item = item
                sanitized.append(sanitized_item)
            except Exception as e:
                raise ValidationError(f"List item {i} sanitization failed: {str(e)}")

        logger.debug(f"List sanitized: {len(value)} items")
        return sanitized

    def sanitize_dict(self, value: Dict[str, Any],
                     max_keys: int = 100) -> Dict[str, Any]:
        """
        Sanitize a dictionary input.

        Args:
            value: Dictionary to sanitize
            max_keys: Maximum allowed number of keys

        Returns:
            Sanitized dictionary

        Raises:
            ValidationError: If dictionary is invalid
        """
        if not isinstance(value, dict):
            raise ValidationError(f"Expected dict, got {type(value)}")

        if len(value) > max_keys:
            raise ValidationError(f"Dict too large: {len(value)} > {max_keys}")

        sanitized = {}
        for key, val in value.items():
            # Sanitize key
            if not isinstance(key, str):
                raise ValidationError(f"Dict key must be string, got {type(key)}")

            sanitized_key = self.sanitize_string(key, max_length=100)

            # Basic value sanitization (can be enhanced based on needs)
            if isinstance(val, str):
                sanitized_val = self.sanitize_string(val)
            elif isinstance(val, (int, float)):
                sanitized_val = self.sanitize_number(val)
            elif isinstance(val, list):
                sanitized_val = self.sanitize_list(val, item_sanitizer=self.sanitize_string if all(isinstance(x, str) for x in val) else None)
            else:
                sanitized_val = val  # Allow other types as-is for now

            sanitized[sanitized_key] = sanitized_val

        logger.debug(f"Dict sanitized: {len(value)} keys")
        return sanitized

    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a complete parameters dictionary.

        Args:
            params: Parameters dictionary to sanitize

        Returns:
            Sanitized parameters dictionary

        Raises:
            ValidationError: If parameters are invalid
        """
        if not isinstance(params, dict):
            raise ValidationError(f"Parameters must be dict, got {type(params)}")

        sanitized = {}

        for key, value in params.items():
            # Sanitize parameter name
            sanitized_key = self.sanitize_string(key, max_length=100)

            # Sanitize based on parameter type and name
            try:
                if key in ['columns', 'features'] and isinstance(value, list):
                    # Column/feature names
                    sanitized_value = self.sanitize_list(
                        value,
                        max_length=1000,
                        item_sanitizer=self.sanitize_column_name
                    )
                elif key in ['target_column', 'target', 'column'] and isinstance(value, str):
                    # Single column name
                    sanitized_value = self.sanitize_column_name(value)
                elif key in ['statistics', 'operations'] and isinstance(value, list):
                    # Statistical operations
                    sanitized_value = self.sanitize_list(
                        value,
                        max_length=50,
                        item_sanitizer=lambda x: self.sanitize_string(x, max_length=50)
                    )
                elif key in ['method', 'model_type', 'operation'] and isinstance(value, str):
                    # Method/type parameters
                    sanitized_value = self.sanitize_string(value, max_length=100)
                elif key in ['test_size', 'alpha', 'timeout', 'random_state'] and isinstance(value, (int, float, str)):
                    # Numeric parameters
                    if key == 'test_size':
                        sanitized_value = self.sanitize_number(value, min_val=0.01, max_val=0.99)
                    elif key == 'alpha':
                        sanitized_value = self.sanitize_number(value, min_val=0.001, max_val=0.5)
                    elif key == 'timeout':
                        sanitized_value = self.sanitize_number(value, min_val=1, max_val=300)
                    elif key == 'random_state':
                        sanitized_value = self.sanitize_number(value, min_val=0, max_val=2**31-1)
                    else:
                        sanitized_value = self.sanitize_number(value)
                elif isinstance(value, str):
                    # Generic string
                    sanitized_value = self.sanitize_string(value)
                elif isinstance(value, (int, float)):
                    # Generic number
                    sanitized_value = self.sanitize_number(value)
                elif isinstance(value, list):
                    # Generic list
                    sanitized_value = self.sanitize_list(value)
                elif isinstance(value, dict):
                    # Nested dictionary
                    sanitized_value = self.sanitize_dict(value)
                else:
                    # Allow other types (bool, None) as-is
                    sanitized_value = value

                sanitized[sanitized_key] = sanitized_value

            except Exception as e:
                raise ValidationError(f"Parameter '{key}' sanitization failed: {str(e)}")

        logger.info(f"Parameters sanitized: {len(params)} parameters")
        return sanitized

    def validate_file_path(self, path: Union[str, Path]) -> Path:
        """
        Validate and sanitize a file path.

        Args:
            path: File path to validate

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid or dangerous
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise ValidationError(f"Path must be string or Path, got {type(path)}")

        path_str = str(path)

        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path_str):
                raise ValidationError(f"Path traversal detected in: {path_str}")

        # Check for dangerous characters
        if re.search(r'[<>&"\'\`\$\{\}]', path_str):
            raise ValidationError(f"Dangerous characters in path: {path_str}")

        # Resolve to absolute path to prevent traversal
        try:
            resolved_path = path.resolve()
        except Exception as e:
            raise ValidationError(f"Cannot resolve path {path_str}: {str(e)}")

        logger.debug(f"Path validated: {path_str} -> {resolved_path}")
        return resolved_path


# Convenience functions for common sanitization tasks
def sanitize_column_names(names: List[str]) -> List[str]:
    """Sanitize a list of column names."""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_list(names, item_sanitizer=sanitizer.sanitize_column_name)


def sanitize_operation_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize operation parameters dictionary."""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_parameters(params)


def validate_numeric_range(value: Union[int, float], name: str,
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None) -> Union[int, float]:
    """Validate a numeric value is within specified range."""
    sanitizer = InputSanitizer()
    try:
        return sanitizer.sanitize_number(value, min_val, max_val)
    except ValidationError as e:
        raise ValidationError(f"Parameter '{name}': {str(e)}")