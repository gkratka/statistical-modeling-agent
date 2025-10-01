"""
Template registry system for script generation.

This module manages Jinja2 templates for generating Python scripts
with caching and metadata extraction capabilities.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

import jinja2

from src.utils.exceptions import ScriptGenerationError, TemplateError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemplateRegistry:
    """Registry for managing script generation templates."""

    def __init__(self, template_dir: Path) -> None:
        """
        Initialize template registry.

        Args:
            template_dir: Directory containing template files

        Raises:
            FileNotFoundError: If template directory doesn't exist
        """
        if not template_dir.exists() or not template_dir.is_dir():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        self.template_dir = template_dir
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            undefined=jinja2.StrictUndefined,
            autoescape=False,  # We're generating Python code, not HTML
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters for template processing
        self.env.filters['sanitize_column'] = self._sanitize_column_name
        self.env.filters['python_list'] = self._format_python_list

        logger.info(f"Template registry initialized with directory: {template_dir}")

    @staticmethod
    def _sanitize_column_name(name: str) -> str:
        """Sanitize column names for safe Python usage."""
        # Replace non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

        # Ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f'_{sanitized}'

        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        return sanitized or 'column'

    @staticmethod
    def _format_python_list(items: list) -> str:
        """Format list as Python list literal."""
        if not items:
            return "[]"

        # Sanitize each item and format as string list
        sanitized_items = [f"'{TemplateRegistry._sanitize_column_name(item)}'" for item in items]
        return f"[{', '.join(sanitized_items)}]"

    @lru_cache(maxsize=128)
    def get_template(self, category: str, operation: str) -> jinja2.Template:
        """
        Get template with caching.

        Args:
            category: Template category (stats, ml, utils)
            operation: Operation name (descriptive, correlation, etc.)

        Returns:
            Compiled Jinja2 template

        Raises:
            ScriptGenerationError: If template not found or invalid
        """
        template_path = f"{category}/{operation}.j2"

        try:
            template = self.env.get_template(template_path)
            logger.debug(f"Template loaded successfully: {template_path}")
            return template

        except jinja2.TemplateNotFound:
            error_msg = f"Template not found: {template_path}"
            logger.error(error_msg)
            raise ScriptGenerationError(error_msg, operation, template_path)

        except jinja2.TemplateSyntaxError as e:
            error_msg = f"Template syntax error in {template_path}: {e.message}"
            logger.error(error_msg)
            raise TemplateError(error_msg, template_path, e.lineno)

        except Exception as e:
            error_msg = f"Failed to load template {template_path}: {str(e)}"
            logger.error(error_msg)
            raise ScriptGenerationError(error_msg, operation, template_path)

    def list_available_templates(self) -> Dict[str, list]:
        """
        List all available templates by category.

        Returns:
            Dictionary mapping categories to template names
        """
        templates = {}

        for category_dir in self.template_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                templates[category] = []

                for template_file in category_dir.glob("*.j2"):
                    template_name = template_file.stem  # Remove .j2 extension
                    templates[category].append(template_name)

                templates[category].sort()

        logger.debug(f"Available templates: {templates}")
        return templates

    def get_template_metadata(self, category: str, operation: str) -> Dict[str, Any]:
        """
        Extract metadata from template comments.

        Args:
            category: Template category
            operation: Operation name

        Returns:
            Dictionary containing template metadata
        """
        template_path = self.template_dir / category / f"{operation}.j2"

        if not template_path.exists():
            raise ScriptGenerationError(f"Template not found: {template_path}", operation)

        metadata = {
            "template": operation,
            "category": category,
            "path": str(template_path),
            "size": template_path.stat().st_size,
            "modified": template_path.stat().st_mtime,
        }

        try:
            content = template_path.read_text()

            # Extract metadata from comment block at the top
            comment_pattern = r'{#\s*(.*?)\s*#}'
            match = re.search(comment_pattern, content, re.DOTALL)

            if match:
                comment_text = match.group(1)

                # Parse key-value pairs from comments
                for line in comment_text.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        metadata[key] = value

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {template_path}: {e}")

        return metadata

    def reload(self) -> None:
        """Reload all templates and clear cache."""
        # Clear the LRU cache
        self.get_template.cache_clear()

        # Recreate the Jinja2 environment to pick up file changes
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Re-add custom filters
        self.env.filters['sanitize_column'] = self._sanitize_column_name
        self.env.filters['python_list'] = self._format_python_list

        logger.info("Template registry reloaded")

    def validate_template(self, category: str, operation: str) -> tuple[bool, list]:
        """
        Validate a template for syntax and basic structure.

        Args:
            category: Template category
            operation: Operation name

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            template = self.get_template(category, operation)

            # Try to render with minimal context to check for missing variables
            try:
                template.render(
                    columns=[],
                    statistics=[],
                    timestamp="2023-01-01T00:00:00",
                    operation=operation
                )
            except jinja2.UndefinedError as e:
                # This is expected for templates that require specific variables
                logger.debug(f"Template {category}/{operation} has undefined variables: {e}")

        except Exception as e:
            errors.append(str(e))

        return len(errors) == 0, errors

    def get_template_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the template registry.

        Returns:
            Dictionary with template statistics
        """
        templates = self.list_available_templates()

        total_templates = sum(len(templates[cat]) for cat in templates)

        stats = {
            "total_templates": total_templates,
            "categories": list(templates.keys()),
            "templates_by_category": {cat: len(templates[cat]) for cat in templates},
            "cache_info": self.get_template.cache_info()._asdict(),
            "template_dir": str(self.template_dir),
        }

        return stats