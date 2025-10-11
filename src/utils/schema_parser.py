"""Schema parser for manual schema input supporting multiple formats."""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.utils.exceptions import ValidationError


@dataclass
class ParsedSchema:
    """Parsed schema information from user input."""
    target: str
    features: List[str]
    raw_input: str
    format_detected: str  # "key_value", "json", "simple_list"


class SchemaParser:
    """Parse user-provided schema in multiple formats."""

    @staticmethod
    def parse(user_input: str) -> ParsedSchema:
        """
        Parse schema from user input. Supports 3 formats:

        Format 1 - Key-Value (most user-friendly):
            target: price
            features: sqft, bedrooms, bathrooms

        Format 2 - JSON (most explicit):
            {"target": "price", "features": ["sqft", "bedrooms", "bathrooms"]}

        Format 3 - Simple List (compact):
            price, sqft, bedrooms, bathrooms
            (first column = target, rest = features)

        Args:
            user_input: Raw user input string

        Returns:
            ParsedSchema object with parsed information

        Raises:
            ValidationError: If input format is invalid or missing required fields
        """
        user_input = user_input.strip()

        # Clean input: Remove explanatory text that may be copied from bot prompt
        # Remove lines starting with "(" which are typically format explanations
        lines = user_input.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('(')]
        user_input = '\n'.join(cleaned_lines).strip()

        if not user_input:
            raise ValidationError("Schema input cannot be empty")

        # Try parsing in order of explicitness (most explicit first)
        errors = []

        # Format 2: JSON (detect by braces)
        if user_input.strip().startswith('{'):
            try:
                return SchemaParser._parse_json(user_input)
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(("JSON", str(e)))

        # Format 1: Key-Value (detect by "target:" or "features:")
        if 'target:' in user_input.lower() or 'features:' in user_input.lower():
            try:
                return SchemaParser._parse_key_value(user_input)
            except ValidationError as e:
                errors.append(("Key-Value", str(e)))

        # Format 3: Simple List (fallback if no format markers detected)
        if not errors:  # No format-specific errors, so try simple list
            try:
                return SchemaParser._parse_simple_list(user_input)
            except ValidationError as e:
                errors.append(("Simple List", str(e)))

        # If all formats fail, raise detailed error with specific issues
        if errors:
            error_details = "\n".join(f"• {fmt}: {err}" for fmt, err in errors)
            raise ValidationError(
                f"Could not parse schema. Please use one of these formats:\n\n"
                f"**Format 1 - Key-Value:**\n"
                f"`target: price`\n"
                f"`features: sqft, bedrooms`\n\n"
                f"**Format 2 - JSON:**\n"
                f'`{{"target": "price", "features": ["sqft", "bedrooms"]}}`\n\n'
                f"**Format 3 - Simple List:**\n"
                f"`price, sqft, bedrooms` (first = target)\n\n"
                f"**Errors detected:**\n{error_details}"
            )

        # Shouldn't reach here, but just in case
        raise ValidationError("Unable to parse schema in any supported format")

    @staticmethod
    def _parse_json(user_input: str) -> ParsedSchema:
        """Parse JSON format."""
        try:
            data = json.loads(user_input)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {str(e)}")

        if not isinstance(data, dict):
            raise ValidationError("JSON must be an object/dict")

        # Extract target
        if "target" not in data:
            raise ValidationError("JSON missing 'target' field")

        target = data.get("target")
        if not isinstance(target, str):
            raise ValidationError("'target' must be a string")

        target = target.strip()
        if not target:
            raise ValidationError("'target' cannot be empty")

        # Extract features
        features = data.get("features")
        if not features:
            raise ValidationError("JSON missing 'features' field")

        # Features must be a list (not string)
        if not isinstance(features, list):
            raise ValidationError("'features' must be a list, not a string")

        feature_list = [str(f).strip() for f in features if str(f).strip()]

        if not feature_list:
            raise ValidationError("'features' cannot be empty")

        # Validate column names
        SchemaParser._validate_column_names(target, feature_list)

        return ParsedSchema(
            target=target,
            features=feature_list,
            raw_input=user_input,
            format_detected="json"
        )

    @staticmethod
    def _parse_key_value(user_input: str) -> ParsedSchema:
        """Parse key-value format (target: X, features: Y, Z)."""
        # Split by lines or semicolons
        lines = [line.strip() for line in re.split(r'[\n;]', user_input) if line.strip()]

        target = None
        features = None

        for line in lines:
            # Match "key: value" pattern
            match = re.match(r'^(target|features)\s*:\s*(.+)$', line, re.IGNORECASE)
            if not match:
                continue

            key = match.group(1).lower()
            value = match.group(2).strip()

            if key == "target":
                target = value
            elif key == "features":
                features = value

        if not target:
            raise ValidationError("Key-value format missing 'target: <column>' line")
        if not features:
            raise ValidationError("Key-value format missing 'features: <columns>' line")

        # Parse features (comma-separated)
        feature_list = [f.strip() for f in features.split(",") if f.strip()]
        if not feature_list:
            raise ValidationError("'features' cannot be empty")

        # Validate column names
        SchemaParser._validate_column_names(target, feature_list)

        return ParsedSchema(
            target=target,
            features=feature_list,
            raw_input=user_input,
            format_detected="key_value"
        )

    @staticmethod
    def _parse_simple_list(user_input: str) -> ParsedSchema:
        """Parse simple comma-separated list (first = target, rest = features)."""
        # Remove extra whitespace and split by comma
        columns = [col.strip() for col in user_input.split(",") if col.strip()]

        if len(columns) < 2:
            raise ValidationError(
                "Simple list format requires at least 2 columns (target + features)"
            )

        target = columns[0]
        feature_list = columns[1:]

        # Validate column names
        SchemaParser._validate_column_names(target, feature_list)

        return ParsedSchema(
            target=target,
            features=feature_list,
            raw_input=user_input,
            format_detected="simple_list"
        )

    @staticmethod
    def _validate_column_names(target: str, features: List[str]) -> None:
        """Validate column names are non-empty and unique."""
        # Check target not empty
        if not target or not target.strip():
            raise ValidationError("Target column name cannot be empty")

        # Check features not empty
        for i, feature in enumerate(features):
            if not feature or not feature.strip():
                raise ValidationError(f"Feature column at position {i+1} is empty")

        # Check for duplicates (case-insensitive)
        all_columns = [target] + features
        lowercase_columns = [col.lower() for col in all_columns]

        if len(lowercase_columns) != len(set(lowercase_columns)):
            # Find duplicates
            seen = set()
            duplicates = []
            for col in lowercase_columns:
                if col in seen and col not in duplicates:
                    duplicates.append(col)
                seen.add(col)

            raise ValidationError(
                f"Duplicate column names detected: {', '.join(duplicates)}"
            )

        # Check target not in features
        if target.lower() in [f.lower() for f in features]:
            raise ValidationError(
                f"Target '{target}' cannot also be a feature column"
            )

    @staticmethod
    def format_schema_for_display(schema: ParsedSchema) -> str:
        """Format parsed schema for user confirmation display."""
        features_str = ", ".join(f"`{f}`" for f in schema.features)
        if len(schema.features) > 5:
            features_str = ", ".join(f"`{f}`" for f in schema.features[:5])
            features_str += f" ... (+{len(schema.features) - 5} more)"

        return (
            f"**Target:** `{schema.target}`\n"
            f"**Features:** {features_str}\n"
            f"**Format:** {schema.format_detected}"
        )
