"""
ML Prediction Template Manager.

This module provides CRUD operations for ML prediction templates,
allowing users to save, load, list, and delete prediction configurations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.prediction_template import PredictionTemplate, PredictionTemplateConfig

logger = logging.getLogger(__name__)


class PredictionTemplateManager:
    """Manage ML prediction templates with CRUD operations."""

    def __init__(self, config: PredictionTemplateConfig):
        """
        Initialize PredictionTemplateManager.

        Args:
            config: PredictionTemplateConfig with system settings
        """
        self.templates_dir = Path(config.templates_dir)
        self.max_templates_per_user = config.max_templates_per_user
        self.name_pattern = config.allowed_name_pattern
        self.name_max_length = config.name_max_length
        self.config = config

        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PredictionTemplateManager initialized with directory: {self.templates_dir}")

    def _validate_save_operation(
        self,
        user_id: int,
        template_name: str,
        user_dir: Path
    ) -> Tuple[bool, str, bool]:
        """
        Validate save operation and check if updating existing template.

        Args:
            user_id: User ID
            template_name: Template name
            user_dir: User directory path

        Returns:
            Tuple of (is_valid: bool, error_msg: str, is_update: bool)
        """
        # Validate template name
        is_valid, error_msg = self.validate_template_name(template_name)
        if not is_valid:
            return False, error_msg, False

        # Check template count limit
        template_exists = (user_dir / f"{template_name}.json").exists()
        if not template_exists:
            existing_count = len(self.list_templates(user_id))
            if existing_count >= self.max_templates_per_user:
                return False, f"Maximum {self.max_templates_per_user} templates per user exceeded", False

        return True, "", template_exists

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
                   - model_id: str
                   - feature_columns: List[str]
                   - output_column_name: str
                   - save_path: Optional[str]
                   - description: Optional[str]

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            user_dir = self._get_user_directory(user_id)
            is_valid, error_msg, template_exists = self._validate_save_operation(
                user_id, template_name, user_dir
            )
            if not is_valid:
                return False, error_msg

            # Create template object
            template_id = PredictionTemplate.generate_template_id(user_id, template_name)

            # If updating existing template, preserve created_at
            existing_template = self.load_template(user_id, template_name)
            created_at = existing_template.created_at if existing_template else None

            template = PredictionTemplate(
                template_id=template_id,
                template_name=template_name,
                user_id=user_id,
                file_path=config["file_path"],
                model_id=config["model_id"],
                feature_columns=config["feature_columns"],
                output_column_name=config["output_column_name"],
                save_path=config.get("save_path"),
                defer_loading=config.get("defer_loading", False),
                description=config.get("description"),
                created_at=created_at or template_id.split("_")[-2] + "_" + template_id.split("_")[-1],
                last_used=config.get("last_used")
            )

            # Save to file
            template_file = user_dir / f"{template_name}.json"
            if not self._write_template_json(template_file, template.to_dict()):
                return False, "Failed to write template file"

            action = "updated" if template_exists else "saved"
            logger.info(f"Prediction template '{template_name}' {action} for user {user_id}")
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
    ) -> Optional[PredictionTemplate]:
        """
        Load template by name.

        Args:
            user_id: User ID
            template_name: Template name

        Returns:
            PredictionTemplate object or None if not found
        """
        user_dir = self._get_user_directory(user_id)
        template_file = user_dir / f"{template_name}.json"

        if not template_file.exists():
            logger.warning(f"Prediction template '{template_name}' not found for user {user_id}")
            return None

        data = self._read_template_json(template_file)
        if not data:
            return None

        try:
            template = PredictionTemplate.from_dict(data)
            logger.info(f"Prediction template '{template_name}' loaded for user {user_id}")
            return template
        except Exception as e:
            logger.error(f"Failed to parse template: {e}", exc_info=True)
            return None

    def list_templates(
        self,
        user_id: int
    ) -> List[PredictionTemplate]:
        """
        List all templates for user, sorted by last_used (most recent first).

        Args:
            user_id: User ID

        Returns:
            List of PredictionTemplate objects
        """
        user_dir = self._get_user_directory(user_id)
        if not user_dir.exists():
            return []

        templates = []
        for template_file in user_dir.glob("*.json"):
            if data := self._read_template_json(template_file):
                try:
                    templates.append(PredictionTemplate.from_dict(data))
                except Exception as e:
                    logger.error(f"Failed to parse template {template_file}: {e}")

        # Sort by last_used (most recent first), then by created_at
        templates.sort(key=lambda t: t.last_used or t.created_at or "", reverse=True)

        logger.info(f"Listed {len(templates)} prediction templates for user {user_id}")
        return templates

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
                logger.warning(f"Prediction template '{template_name}' not found for user {user_id}")
                return False

            template_file.unlink()
            logger.info(f"Prediction template '{template_name}' deleted for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete template: {e}", exc_info=True)
            return False

    def template_exists(self, user_id: int, name: str) -> bool:
        """Check if template exists."""
        return (self._get_user_directory(user_id) / f"{name}.json").exists()

    def get_template_count(self, user_id: int) -> int:
        """Get number of templates for user."""
        return len(self.list_templates(user_id))

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
        reserved_names = {'', '.', '..', 'CON', 'PRN', 'AUX', 'NUL'} | {
            f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in '123456789'
        }
        if name.upper() in reserved_names:
            return False, f"'{name}' is a reserved name"

        return True, ""


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

    def _read_template_json(self, template_file: Path) -> Optional[Dict[str, Any]]:
        """
        Read and parse template JSON file.

        Args:
            template_file: Path to template JSON file

        Returns:
            Parsed JSON dict or None if error
        """
        try:
            with open(template_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file {template_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read template file {template_file}: {e}")
            return None

    def _write_template_json(self, template_file: Path, data: Dict[str, Any]) -> bool:
        """
        Write template data to JSON file.

        Args:
            template_file: Path to template JSON file
            data: Template data dict

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(template_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write template file {template_file}: {e}")
            return False
