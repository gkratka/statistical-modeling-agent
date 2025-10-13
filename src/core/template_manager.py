"""
ML Training Template Manager.

This module provides CRUD operations for ML training templates,
allowing users to save, load, list, and delete training configurations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.training_template import TemplateConfig, TrainingTemplate

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manage ML training templates with CRUD operations."""

    def __init__(self, config: TemplateConfig):
        """
        Initialize TemplateManager.

        Args:
            config: TemplateConfig with system settings
        """
        self.templates_dir = Path(config.templates_dir)
        self.max_templates_per_user = config.max_templates_per_user
        self.name_pattern = config.allowed_name_pattern
        self.name_max_length = config.name_max_length
        self.config = config

        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TemplateManager initialized with directory: {self.templates_dir}")

    def save_template(
        self,
        user_id: int,
        template_name: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Save a new template or update existing one.

        Args:
            user_id: User ID
            template_name: User-provided template name
            config: Template configuration dict with keys:
                   - file_path: str
                   - target_column: str
                   - feature_columns: List[str]
                   - model_category: str
                   - model_type: str
                   - hyperparameters: Dict[str, Any]

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Validate template name
            is_valid, error_msg = self.validate_template_name(template_name)
            if not is_valid:
                return False, error_msg

            # Check template count limit
            user_dir = self._get_user_directory(user_id)
            existing_templates = self.list_templates(user_id)

            # If template exists, we're updating; otherwise check limit
            template_exists = self.template_exists(user_id, template_name)
            if not template_exists and len(existing_templates) >= self.max_templates_per_user:
                return False, f"Maximum {self.max_templates_per_user} templates per user exceeded"

            # Create template object
            template_id = TrainingTemplate.generate_template_id(user_id, template_name)

            # If updating existing template, preserve created_at
            existing_template = self.load_template(user_id, template_name)
            created_at = existing_template.created_at if existing_template else None

            template = TrainingTemplate(
                template_id=template_id,
                template_name=template_name,
                user_id=user_id,
                file_path=config["file_path"],
                target_column=config["target_column"],
                feature_columns=config["feature_columns"],
                model_category=config["model_category"],
                model_type=config["model_type"],
                hyperparameters=config.get("hyperparameters", {}),
                created_at=created_at or template_id.split("_")[-2] + "_" + template_id.split("_")[-1],
                last_used=config.get("last_used"),
                description=config.get("description")
            )

            # Save to file
            template_file = user_dir / f"{template_name}.json"
            with open(template_file, 'w') as f:
                json.dump(template.to_dict(), f, indent=2)

            action = "updated" if template_exists else "saved"
            logger.info(f"Template '{template_name}' {action} for user {user_id}")
            return True, f"Template '{template_name}' {action} successfully"

        except KeyError as e:
            error_msg = f"Missing required configuration field: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to save template: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def load_template(
        self,
        user_id: int,
        template_name: str
    ) -> Optional[TrainingTemplate]:
        """
        Load template by name.

        Args:
            user_id: User ID
            template_name: Template name

        Returns:
            TrainingTemplate object or None if not found
        """
        try:
            user_dir = self._get_user_directory(user_id)
            template_file = user_dir / f"{template_name}.json"

            if not template_file.exists():
                logger.warning(f"Template '{template_name}' not found for user {user_id}")
                return None

            with open(template_file, 'r') as f:
                data = json.load(f)

            template = TrainingTemplate.from_dict(data)
            logger.info(f"Template '{template_name}' loaded for user {user_id}")
            return template

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load template: {e}", exc_info=True)
            return None

    def list_templates(
        self,
        user_id: int
    ) -> List[TrainingTemplate]:
        """
        List all templates for user, sorted by last_used (most recent first).

        Args:
            user_id: User ID

        Returns:
            List of TrainingTemplate objects
        """
        try:
            user_dir = self._get_user_directory(user_id)

            if not user_dir.exists():
                return []

            templates = []
            for template_file in user_dir.glob("*.json"):
                try:
                    with open(template_file, 'r') as f:
                        data = json.load(f)
                    template = TrainingTemplate.from_dict(data)
                    templates.append(template)
                except Exception as e:
                    logger.error(f"Failed to load template {template_file}: {e}")
                    continue

            # Sort by last_used (most recent first), then by created_at
            def sort_key(t: TrainingTemplate) -> str:
                return t.last_used or t.created_at or ""

            templates.sort(key=sort_key, reverse=True)

            logger.info(f"Listed {len(templates)} templates for user {user_id}")
            return templates

        except Exception as e:
            logger.error(f"Failed to list templates: {e}", exc_info=True)
            return []

    def delete_template(
        self,
        user_id: int,
        template_name: str
    ) -> bool:
        """
        Delete template by name.

        Args:
            user_id: User ID
            template_name: Template name

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            user_dir = self._get_user_directory(user_id)
            template_file = user_dir / f"{template_name}.json"

            if not template_file.exists():
                logger.warning(f"Template '{template_name}' not found for user {user_id}")
                return False

            template_file.unlink()
            logger.info(f"Template '{template_name}' deleted for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete template: {e}", exc_info=True)
            return False

    def rename_template(
        self,
        user_id: int,
        old_name: str,
        new_name: str
    ) -> Tuple[bool, str]:
        """
        Rename existing template.

        Args:
            user_id: User ID
            old_name: Current template name
            new_name: New template name

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Validate new name
            is_valid, error_msg = self.validate_template_name(new_name)
            if not is_valid:
                return False, error_msg

            # Check if old template exists
            if not self.template_exists(user_id, old_name):
                return False, f"Template '{old_name}' not found"

            # Check if new name already exists
            if self.template_exists(user_id, new_name):
                return False, f"Template '{new_name}' already exists"

            # Load old template
            template = self.load_template(user_id, old_name)
            if not template:
                return False, f"Failed to load template '{old_name}'"

            # Update template name and save with new name
            template.template_name = new_name
            config = {
                "file_path": template.file_path,
                "target_column": template.target_column,
                "feature_columns": template.feature_columns,
                "model_category": template.model_category,
                "model_type": template.model_type,
                "hyperparameters": template.hyperparameters,
                "last_used": template.last_used,
                "description": template.description
            }

            success, message = self.save_template(user_id, new_name, config)
            if not success:
                return False, f"Failed to save renamed template: {message}"

            # Delete old template
            if not self.delete_template(user_id, old_name):
                # Rollback: delete the new template
                self.delete_template(user_id, new_name)
                return False, f"Failed to delete old template '{old_name}'"

            logger.info(f"Template renamed from '{old_name}' to '{new_name}' for user {user_id}")
            return True, f"Template renamed from '{old_name}' to '{new_name}'"

        except Exception as e:
            error_msg = f"Failed to rename template: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def validate_template_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate template name.

        Rules:
        - Only alphanumeric and underscore
        - Max length from config
        - Not empty
        - No leading/trailing whitespace

        Args:
            name: Template name to validate

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if not name:
            return False, "Template name cannot be empty"

        if name != name.strip():
            return False, "Template name cannot have leading or trailing whitespace"

        if len(name) > self.name_max_length:
            return False, f"Template name exceeds maximum length of {self.name_max_length}"

        if not re.match(self.name_pattern, name):
            return False, "Template name can only contain letters, numbers, and underscores"

        # Check for reserved names
        reserved_names = ['', '.', '..', 'CON', 'PRN', 'AUX', 'NUL',
                         'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                         'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        if name.upper() in reserved_names:
            return False, f"'{name}' is a reserved name"

        return True, ""

    def template_exists(self, user_id: int, name: str) -> bool:
        """
        Check if template exists.

        Args:
            user_id: User ID
            name: Template name

        Returns:
            True if template exists, False otherwise
        """
        user_dir = self._get_user_directory(user_id)
        template_file = user_dir / f"{name}.json"
        return template_file.exists()

    def get_template_count(self, user_id: int) -> int:
        """
        Get number of templates for user.

        Args:
            user_id: User ID

        Returns:
            Number of templates
        """
        return len(self.list_templates(user_id))

    def _get_user_directory(self, user_id: int) -> Path:
        """
        Get user-specific template directory.

        Args:
            user_id: User ID

        Returns:
            Path to user directory
        """
        user_dir = self.templates_dir / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
