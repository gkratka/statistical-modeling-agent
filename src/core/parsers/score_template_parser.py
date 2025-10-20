"""
Score Template Parser - Parse and validate comprehensive train-predict templates.

This module handles parsing of single-prompt templates for the /score workflow,
which combines model training and prediction into one comprehensive submission.

Template Format:
    TRAIN_DATA: /path/to/training_data.csv
    TARGET: target_column_name
    FEATURES: feature1, feature2, feature3
    MODEL: random_forest
    PREDICT_DATA: /path/to/prediction_data.csv

Optional Fields:
    OUTPUT_COLUMN: custom_prediction_column_name (default: "prediction")
    HYPERPARAMETERS: {"n_estimators": 100, "max_depth": 10}
    TASK_TYPE: regression | classification (auto-detected if not provided)
"""

import re
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from src.utils.exceptions import ValidationError


# Supported model types mapped to task types
SUPPORTED_MODELS = {
    # Regression models
    "linear": "regression",
    "ridge": "regression",
    "lasso": "regression",
    "elasticnet": "regression",
    "polynomial": "regression",
    "random_forest_regression": "regression",
    "gradient_boosting_regression": "regression",
    "mlp_regression": "regression",

    # Classification models
    "logistic": "classification",
    "decision_tree": "classification",
    "random_forest": "classification",
    "random_forest_classification": "classification",
    "gradient_boosting": "classification",
    "gradient_boosting_classification": "classification",
    "svm": "classification",
    "naive_bayes": "classification",
    "mlp_classification": "classification",

    # Keras models (binary classification)
    "keras_binary_classification": "classification",
}


@dataclass
class ScoreConfig:
    """Configuration for score workflow combining training and prediction.

    Attributes:
        train_data_path: Absolute path to training dataset
        target_column: Name of the target/dependent variable column
        feature_columns: List of feature/independent variable column names
        model_type: Type of ML model to train (must be in SUPPORTED_MODELS)
        predict_data_path: Absolute path to prediction dataset
        output_column: Column name for predictions (default: "prediction")
        hyperparameters: Optional dict of model hyperparameters
        task_type: regression | classification (auto-detected from model_type)
        train_data_shape: Shape of training data (rows, cols) - populated during validation
        predict_data_shape: Shape of prediction data (rows, cols) - populated during validation
        model_id: Generated model ID after training - populated during execution
    """
    train_data_path: str
    target_column: str
    feature_columns: List[str]
    model_type: str
    predict_data_path: str
    output_column: str = "prediction"
    hyperparameters: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = None

    # Metadata fields populated during validation/execution
    train_data_shape: Optional[Tuple[int, int]] = None
    predict_data_shape: Optional[Tuple[int, int]] = None
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and initialize configuration."""
        # Validate all required fields in one pass
        required = {
            'TRAIN_DATA': (self.train_data_path, str),
            'TARGET': (self.target_column, str),
            'FEATURES': (self.feature_columns, list),
            'MODEL': (self.model_type, str),
            'PREDICT_DATA': (self.predict_data_path, str)
        }

        empty_fields = [
            name for name, (value, expected_type) in required.items()
            if not value or (isinstance(value, str) and not value.strip()) or
               (isinstance(value, list) and len(value) == 0)
        ]

        if empty_fields:
            raise ValidationError(f"Empty required fields: {', '.join(empty_fields)}")

        # Validate model type
        normalized_model = self.model_type.lower().strip()
        if normalized_model not in SUPPORTED_MODELS:
            raise ValidationError(
                f"Unsupported model type '{self.model_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_MODELS.keys()))}"
            )

        # Validate feature uniqueness and target exclusion
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValidationError("FEATURES list contains duplicate columns")

        if self.target_column in self.feature_columns:
            raise ValidationError(f"TARGET '{self.target_column}' cannot be in FEATURES")

        # Infer task type if not provided
        if self.task_type is None:
            self.task_type = SUPPORTED_MODELS.get(normalized_model)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreConfig":
        """Create ScoreConfig from dictionary."""
        # Get valid field names for this dataclass
        field_names = {f.name for f in fields(cls)}

        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}

        # Convert tuple fields from list if needed
        if 'train_data_shape' in filtered_data and filtered_data['train_data_shape']:
            filtered_data['train_data_shape'] = tuple(filtered_data['train_data_shape'])
        if 'predict_data_shape' in filtered_data and filtered_data['predict_data_shape']:
            filtered_data['predict_data_shape'] = tuple(filtered_data['predict_data_shape'])

        return cls(**filtered_data)


def parse_score_template(template_text: str) -> ScoreConfig:
    """Parse score template text into ScoreConfig object.

    Template format (case-insensitive, whitespace-tolerant):
        TRAIN_DATA: /path/to/training.csv
        TARGET: price
        FEATURES: sqft, bedrooms, bathrooms
        MODEL: random_forest
        PREDICT_DATA: /path/to/prediction.csv

    Optional fields:
        OUTPUT_COLUMN: predicted_price
        HYPERPARAMETERS: {"n_estimators": 100}
        TASK_TYPE: regression

    Args:
        template_text: Raw template text from user

    Returns:
        ScoreConfig object with parsed and validated configuration

    Raises:
        ValidationError: If template is malformed or missing required fields
    """
    if not template_text or not template_text.strip():
        raise ValidationError("Template text cannot be empty")

    # Parse key-value pairs
    parsed_data = _parse_key_value_pairs(template_text)

    # Validate required fields are present
    required_fields = ["TRAIN_DATA", "TARGET", "FEATURES", "MODEL", "PREDICT_DATA"]
    missing_fields = [field for field in required_fields if field not in parsed_data]

    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}. "
            f"Required: {', '.join(required_fields)}"
        )

    # Parse FEATURES as comma-separated list
    features_raw = parsed_data["FEATURES"]
    feature_columns = _parse_feature_list(features_raw)

    # Parse optional HYPERPARAMETERS as JSON
    hyperparameters = None
    if "HYPERPARAMETERS" in parsed_data:
        hyperparameters = _parse_hyperparameters(parsed_data["HYPERPARAMETERS"])

    # Create ScoreConfig (validation happens in __post_init__)
    try:
        config = ScoreConfig(
            train_data_path=parsed_data["TRAIN_DATA"].strip(),
            target_column=parsed_data["TARGET"].strip(),
            feature_columns=feature_columns,
            model_type=parsed_data["MODEL"].strip(),
            predict_data_path=parsed_data["PREDICT_DATA"].strip(),
            output_column=parsed_data.get("OUTPUT_COLUMN", "prediction").strip(),
            hyperparameters=hyperparameters,
            task_type=parsed_data.get("TASK_TYPE", "").strip() or None,
        )
    except (TypeError, KeyError) as e:
        raise ValidationError(f"Error creating configuration: {str(e)}")

    return config


def _parse_key_value_pairs(text: str) -> Dict[str, str]:
    """Parse key-value pairs from template text.

    Format: KEY: value (case-insensitive, whitespace-tolerant)
    Multi-line values are supported (continue until next KEY:)

    Args:
        text: Raw template text

    Returns:
        Dictionary mapping normalized keys to values

    Raises:
        ValidationError: If text contains malformed key-value pairs
    """
    parsed = {}
    lines = text.strip().split("\n")

    current_key = None
    current_value = []

    # Pattern to match KEY: value (case-insensitive)
    key_pattern = re.compile(r'^([A-Z_]+)\s*:\s*(.*)$', re.IGNORECASE)

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if line starts with a key
        match = key_pattern.match(line)

        if match:
            # Save previous key-value pair
            if current_key is not None:
                parsed[current_key] = "\n".join(current_value).strip()

            # Start new key-value pair
            current_key = match.group(1).upper()
            current_value = [match.group(2)]
        else:
            # Continuation of previous value (multi-line)
            if current_key is None:
                raise ValidationError(
                    f"Line {line_num} does not start with a key: '{line}'"
                )
            current_value.append(line)

    # Save last key-value pair
    if current_key is not None:
        parsed[current_key] = "\n".join(current_value).strip()

    if not parsed:
        raise ValidationError("No valid key-value pairs found in template")

    return parsed


def _parse_value(value: str, parse_type: Literal['list', 'json', 'string']) -> Any:
    """Generic value parser with type-specific logic.

    Args:
        value: Raw value string to parse
        parse_type: Type of parsing to apply ('list', 'json', 'string')

    Returns:
        Parsed value (list, dict, or str depending on parse_type)

    Raises:
        ValidationError: If parsing fails or value is invalid
    """
    if parse_type == 'list':
        if not value or not value.strip():
            raise ValidationError("FEATURES list cannot be empty")
        features = [f.strip() for f in value.split(",") if f.strip()]
        if not features:
            raise ValidationError("FEATURES list contains no valid column names")
        return features

    elif parse_type == 'json':
        if not value or not value.strip():
            return {}
        try:
            parsed = json.loads(value.strip())
            if not isinstance(parsed, dict):
                raise ValidationError("HYPERPARAMETERS must be a JSON object (dict)")
            return parsed
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid HYPERPARAMETERS JSON: {str(e)}. "
                f"Example: {{\"n_estimators\": 100}}"
            )

    else:  # 'string'
        return value.strip()


def _parse_feature_list(features_text: str) -> List[str]:
    """Parse comma-separated feature list."""
    return _parse_value(features_text, 'list')


def _parse_hyperparameters(hyperparams_text: str) -> Dict[str, Any]:
    """Parse hyperparameters from JSON string."""
    return _parse_value(hyperparams_text, 'json')


def validate_score_config(config: ScoreConfig) -> List[str]:
    """Validate score configuration and return list of warnings.

    Args:
        config: ScoreConfig to validate

    Returns:
        List of warning messages (empty if no warnings)
    """
    warnings = []

    # Validation rules with corresponding warning templates
    validations = [
        (
            not config.train_data_path.startswith(("/", "./")),
            f"TRAIN_DATA '{config.train_data_path}' is not absolute. Use absolute paths for clarity."
        ),
        (
            not config.predict_data_path.startswith(("/", "./")),
            f"PREDICT_DATA '{config.predict_data_path}' is not absolute. Use absolute paths for clarity."
        ),
        (
            len(config.feature_columns) < 2,
            f"Only {len(config.feature_columns)} feature(s). Models perform better with multiple features."
        ),
        (
            len(config.feature_columns) > 50,
            f"{len(config.feature_columns)} features. Consider feature selection for dimensionality."
        ),
    ]

    # Apply validation rules
    for condition, message in validations:
        if condition:
            warnings.append(message)

    return warnings


def format_config_summary(config: ScoreConfig) -> str:
    """Format configuration summary for user confirmation.

    Args:
        config: ScoreConfig to summarize

    Returns:
        Formatted multi-line string with configuration details
    """
    # Base template
    summary = f"""📋 Score Configuration Summary:

🎯 Model Type: {config.model_type} ({config.task_type})
📂 Training Data: {config.train_data_path}
🎯 Target Column: {config.target_column}
📊 Features: {', '.join(config.feature_columns)} ({len(config.feature_columns)} total)
📂 Prediction Data: {config.predict_data_path}
📝 Output Column: {config.output_column}"""

    # Add optional fields
    optional_fields = []

    if config.hyperparameters:
        optional_fields.append(f"⚙️ Hyperparameters: {json.dumps(config.hyperparameters, indent=2)}")

    if config.train_data_shape:
        rows, cols = config.train_data_shape
        optional_fields.append(f"📏 Training Data Shape: {rows:,} rows × {cols} columns")

    if config.predict_data_shape:
        rows, cols = config.predict_data_shape
        optional_fields.append(f"📏 Prediction Data Shape: {rows:,} rows × {cols} columns")

    if optional_fields:
        summary += "\n" + "\n".join(optional_fields)

    return summary
