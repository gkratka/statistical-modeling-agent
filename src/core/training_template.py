"""
ML Training Template Data Structures.

This module defines dataclasses for ML training template management,
allowing users to save and reuse complete training configurations.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingTemplate:
    """Complete ML training configuration template."""

    template_id: str  # "tmpl_{user_id}_{sanitized_name}_{timestamp}"
    template_name: str  # User-provided name
    user_id: int

    # Data configuration
    file_path: str  # Absolute path to data file

    # Schema configuration
    target_column: str
    feature_columns: List[str]

    # Model configuration
    model_category: str  # "regression", "classification", "neural_network"
    model_type: str  # "random_forest", "linear", "keras_binary_classification", etc.

    # Training configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for JSON serialization."""
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "user_id": self.user_id,
            "file_path": self.file_path,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "model_category": self.model_category,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingTemplate":
        """Create template from dictionary."""
        return cls(
            template_id=data["template_id"],
            template_name=data["template_name"],
            user_id=data["user_id"],
            file_path=data["file_path"],
            target_column=data["target_column"],
            feature_columns=data["feature_columns"],
            model_category=data["model_category"],
            model_type=data["model_type"],
            hyperparameters=data.get("hyperparameters", {}),
            created_at=data["created_at"],
            last_used=data.get("last_used"),
            description=data.get("description")
        )

    @staticmethod
    def generate_template_id(user_id: int, template_name: str) -> str:
        """Generate unique template ID from user ID and template name."""
        # Sanitize template name for use in ID
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', template_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"tmpl_{user_id}_{sanitized_name}_{timestamp}"


@dataclass
class TemplateConfig:
    """Configuration for template system."""

    enabled: bool = True
    templates_dir: str = "./templates"
    max_templates_per_user: int = 50
    allowed_name_pattern: str = r"^[a-zA-Z0-9_]{1,32}$"
    name_max_length: int = 32

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TemplateConfig":
        """Create from config dictionary."""
        template_config = config.get("templates", {})
        return cls(
            enabled=template_config.get("enabled", True),
            templates_dir=template_config.get("templates_dir", "./templates"),
            max_templates_per_user=template_config.get("max_templates_per_user", 50),
            allowed_name_pattern=template_config.get("allowed_name_pattern", r"^[a-zA-Z0-9_]{1,32}$"),
            name_max_length=template_config.get("name_max_length", 32)
        )

    def get_templates_dir(self) -> Path:
        """Get templates directory as Path object."""
        return Path(self.templates_dir)

    def validate_name_pattern(self, name: str) -> bool:
        """Check if template name matches allowed pattern."""
        return bool(re.match(self.allowed_name_pattern, name))
