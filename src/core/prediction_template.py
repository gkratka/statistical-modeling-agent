"""
ML Prediction Template Data Structures.

This module defines dataclasses for ML prediction template management,
allowing users to save and reuse complete prediction configurations.
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PredictionTemplate:
    """Complete ML prediction configuration template."""

    template_id: str  # "pred_tmpl_{user_id}_{sanitized_name}_{timestamp}"
    template_name: str  # User-provided name
    user_id: int

    # Prediction configuration
    file_path: str  # Path to data file for predictions
    model_id: str  # Trained model to use for predictions
    feature_columns: List[str]  # Features (must match model's expected features)
    output_column_name: str  # Name for prediction column

    # Optional fields
    save_path: Optional[str] = None  # Default save location for predictions
    description: Optional[str] = None  # User notes

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionTemplate":
        """Create template from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def generate_template_id(user_id: int, template_name: str) -> str:
        """Generate unique template ID from user ID and template name."""
        # Sanitize template name for use in ID
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', template_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"pred_tmpl_{user_id}_{sanitized_name}_{timestamp}"


@dataclass
class PredictionTemplateConfig:
    """Configuration for prediction template system."""

    enabled: bool = True
    templates_dir: str = "./templates/predictions"
    max_templates_per_user: int = 50
    allowed_name_pattern: str = r"^[a-zA-Z0-9_]+$"
    name_max_length: int = 255

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PredictionTemplateConfig":
        """Create from config dictionary."""
        pred_template_config = config.get("prediction_templates", {})
        return cls(
            enabled=pred_template_config.get("enabled", True),
            templates_dir=pred_template_config.get("templates_dir", "./templates/predictions"),
            max_templates_per_user=pred_template_config.get("max_templates_per_user", 50),
            allowed_name_pattern=pred_template_config.get("allowed_name_pattern", r"^[a-zA-Z0-9_]{1,32}$"),
            name_max_length=pred_template_config.get("name_max_length", 32)
        )

    def validate_name_pattern(self, name: str) -> bool:
        """Check if template name matches allowed pattern."""
        return bool(re.match(self.allowed_name_pattern, name))
