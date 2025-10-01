"""
Script generator for creating secure Python scripts from task definitions.

This module uses Jinja2 templates to generate Python scripts for statistical
and machine learning operations with comprehensive security validation.
"""

import datetime
import re
from pathlib import Path
from typing import Dict, Any

from src.core.parser import TaskDefinition
from src.generators.template_registry import TemplateRegistry
from src.generators.validator import ScriptValidator
from src.utils.exceptions import ScriptGenerationError, SecurityViolationError
from src.utils.logger import get_logger
from src.utils.sanitization import InputSanitizer

logger = get_logger(__name__)


class ScriptGenerator:
    """Generates secure Python scripts from task definitions."""

    def __init__(self, template_dir: Path) -> None:
        """
        Initialize script generator.

        Args:
            template_dir: Directory containing Jinja2 templates

        Raises:
            FileNotFoundError: If template directory doesn't exist
        """
        self.template_registry = TemplateRegistry(template_dir)
        self.validator = ScriptValidator()
        self.sanitizer = InputSanitizer()

        # Operation mapping for template selection (fallback logic handles most cases)
        self.operation_templates = {
            # Core operations (explicit mapping for primary commands)
            "descriptive": ("stats", "descriptive"),
            "correlation": ("stats", "correlation"),
            "predict": ("ml", "predict"),
            "linear_regression": ("ml", "train_regressor"),
            "describe_data": ("utils", "data_info"),
            # Note: Other operations handled by intelligent fallback in _get_template_mapping
        }

        logger.info(f"Script generator initialized with {len(self.operation_templates)} operation mappings")

    def enable_test_mode(self) -> None:
        """Enable test mode for flexible template mapping."""
        self._test_mode = True

    def generate(self, task: TaskDefinition, context_data: Dict[str, Any] = None) -> str:
        """
        Generate Python script from task definition.

        Args:
            task: Task definition with operation details
            context_data: Additional context data for template rendering

        Returns:
            Generated Python script as string

        Raises:
            ScriptGenerationError: If script generation fails
            SecurityViolationError: If generated script fails security validation
        """
        logger.info(f"Generating script for operation: {task.operation}")

        try:
            # Get template mapping
            category, template_name = self._get_template_mapping(task.operation)

            # Sanitize parameters
            sanitized_params = self.sanitizer.sanitize_parameters(task.parameters)

            # Prepare template context
            template_context = self._prepare_template_context(
                task, sanitized_params, context_data or {}
            )

            # Get and render template
            template = self.template_registry.get_template(category, template_name)
            script = template.render(**template_context)

            # Validate generated script
            self._validate_generated_script(script, task.operation)

            logger.info(f"Script generated successfully for {task.operation}")
            return script

        except Exception as e:
            error_msg = f"Failed to generate script for {task.operation}: {str(e)}"
            logger.error(error_msg)
            raise ScriptGenerationError(error_msg, task.operation)

    def _get_template_mapping(self, operation: str) -> tuple[str, str]:
        """
        Get template category and name for operation.

        Args:
            operation: Operation name

        Returns:
            Tuple of (category, template_name)

        Raises:
            ScriptGenerationError: If operation not supported
        """
        if operation in self.operation_templates:
            return self.operation_templates[operation]

        # Try to infer from operation name
        if "train" in operation.lower():
            return "ml", "train_classifier"
        elif "predict" in operation.lower():
            return "ml", "predict"
        elif "correlation" in operation.lower():
            return "stats", "correlation"
        elif any(stat in operation.lower() for stat in ["mean", "std", "descriptive"]):
            return "stats", "descriptive"
        else:
            # For test cases, try to determine category from task_type or assume stats
            # This allows flexibility for testing custom operations
            if hasattr(self, '_test_mode') and self._test_mode:
                return "stats", operation  # Use operation name directly for tests
            raise ScriptGenerationError(f"No template mapping found for operation: {operation}")

    def _prepare_template_context(
        self,
        task: TaskDefinition,
        sanitized_params: Dict[str, Any],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare context data for template rendering.

        Args:
            task: Task definition
            sanitized_params: Sanitized parameters
            context_data: Additional context data

        Returns:
            Template context dictionary
        """
        # Base context
        template_context = {
            "operation": task.operation,
            "task_type": task.task_type,
            "user_id": task.user_id,
            "conversation_id": task.conversation_id,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Add sanitized parameters
        template_context.update(sanitized_params)

        # Add context data
        template_context.update(context_data)

        # Ensure required fields have defaults
        if "columns" not in template_context:
            template_context["columns"] = []

        if "statistics" not in template_context:
            template_context["statistics"] = ["mean", "std", "count"]

        return template_context

    def _validate_generated_script(self, script: str, operation: str) -> None:
        """
        Validate generated script for security and syntax.

        Args:
            script: Generated Python script
            operation: Operation name for error reporting

        Raises:
            SecurityViolationError: If script fails security validation
        """
        is_valid, violations = self.validator.validate_script(script)

        if not is_valid:
            error_msg = f"Generated script for {operation} failed security validation"
            logger.error(f"{error_msg}: {violations}")
            raise SecurityViolationError(error_msg, violations)

    def get_supported_operations(self) -> Dict[str, tuple]:
        """
        Get list of supported operations and their template mappings.

        Returns:
            Dictionary mapping operations to (category, template) tuples
        """
        return self.operation_templates.copy()

    def add_operation_mapping(self, operation: str, category: str, template: str) -> None:
        """
        Add new operation to template mapping.

        Args:
            operation: Operation name
            category: Template category
            template: Template name
        """
        self.operation_templates[operation] = (category, template)
        logger.info(f"Added operation mapping: {operation} -> {category}/{template}")

    def validate_operation(self, operation: str) -> bool:
        """
        Check if operation is supported.

        Args:
            operation: Operation name to check

        Returns:
            True if operation is supported
        """
        try:
            self._get_template_mapping(operation)
            return True
        except ScriptGenerationError:
            return False

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about script generation.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "supported_operations": len(self.operation_templates),
            "template_stats": self.template_registry.get_template_stats(),
            "validator_patterns": len(self.validator.forbidden_patterns),
            "allowed_imports": len(self.validator.allowed_imports),
        }