"""
Natural language parser for the Statistical Modeling Agent.

This module converts user messages into structured TaskDefinition objects
that can be processed by the orchestrator and engines.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional, Literal
from src.utils.exceptions import ParseError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataSource:
    """Information about data source for analysis."""

    file_id: Optional[str] = None  # Telegram file ID
    file_name: Optional[str] = None
    file_type: str = "unknown"  # csv, xlsx, json, etc.
    columns: Optional[list[str]] = None
    shape: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        """Validate data source after initialization."""
        if self.file_type not in ["csv", "xlsx", "json", "parquet", "unknown"]:
            raise ValueError(f"Unsupported file type: {self.file_type}")


@dataclass
class TaskDefinition:
    """Structured task definition for orchestrator processing."""

    task_type: Literal["stats", "ml_train", "ml_score", "data_info"]
    operation: str  # "descriptive_stats", "correlation", "train_model", etc.
    parameters: dict[str, Any]
    data_source: Optional[DataSource]
    user_id: int
    conversation_id: str
    confidence_score: float = 0.0  # 0-1 confidence in parsing

    def __post_init__(self) -> None:
        """Validate task definition after initialization."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

        if not self.operation:
            raise ValueError("Operation cannot be empty")

        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary")


class RequestParser:
    """Main parser for converting natural language requests to structured tasks."""

    def __init__(self) -> None:
        """Initialize the parser with pattern dictionaries."""
        self.stats_patterns = self._init_stats_patterns()
        self.ml_patterns = self._init_ml_patterns()
        self.column_patterns = self._init_column_patterns()

    def _init_stats_patterns(self) -> dict[str, str]:
        """Initialize statistical operation patterns."""
        return {
            'mean': r'(calculate|compute|find|show|get)?\s*(the)?\s*mean',
            'median': r'(calculate|compute|find|show|get)?\s*(the)?\s*median',
            'mode': r'(calculate|compute|find|show|get)?\s*(the)?\s*mode',
            'std': r'(standard deviation|std|stdev)',
            'variance': r'(variance|var)',
            'correlation': r'(correlation|correlate|relationship|corr)',
            'distribution': r'(distribution|histogram|frequency)',
            'summary': r'(summary|describe|overview|descriptive\s*stats)',
            'min': r'(minimum|min)\s*(value)?',
            'max': r'(maximum|max)\s*(value)?',
            'count': r'(count|number\s*of)',
            'quartile': r'(quartile|percentile|quantile)',
        }

    def _init_ml_patterns(self) -> dict[str, str]:
        """Initialize machine learning operation patterns."""
        return {
            'train': r'(train|build|create|make)\s*(a|the)?\s*(model|classifier|predictor)',
            'predict': r'(predict|forecast|estimate|score)',
            'regression': r'(regression|linear\s*model|predict.*continuous)',
            'classification': r'(classify|classification|categorize)',
            'neural_network': r'(neural\s*network|nn|deep\s*learning)',
            'random_forest': r'(random\s*forest|rf)',
            'linear_regression': r'(linear\s*regression|lr)',
            'logistic_regression': r'(logistic\s*regression)',
        }

    def _init_column_patterns(self) -> dict[str, str]:
        """Initialize column extraction patterns."""
        return {
            'column_reference': r'(column|field|variable)\s*["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?',
            'for_column': r'for\s*(the)?\s*(column|field)?\s*["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?',
            'target_variable': r'(to\s*predict|target|dependent\s*variable)\s*["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?',
            'features': r'(based\s*on|using|features?|independent\s*variables?)\s*["\']?([a-zA-Z_][a-zA-Z0-9_,\s]*)["\']?',
        }

    def parse_request(
        self,
        text: str,
        user_id: int,
        conversation_id: str,
        data_source: Optional[DataSource] = None
    ) -> TaskDefinition:
        """
        Main entry point for parsing any user request.

        Args:
            text: User's natural language request
            user_id: Telegram user ID
            conversation_id: Unique conversation identifier
            data_source: Optional data source information

        Returns:
            TaskDefinition object with parsed task information

        Raises:
            ParseError: If the request cannot be parsed
        """
        if not text or not text.strip():
            raise ParseError("Empty request text", raw_input=text)

        text_clean = text.lower().strip()
        logger.info(f"Parsing request: {text_clean[:100]}...")

        # Determine task type based on patterns
        task_type, operation, parameters, confidence = self._classify_request(text_clean)

        if confidence < 0.3:
            raise ParseError(
                f"Could not understand request. Please be more specific about what analysis you want.",
                raw_input=text
            )

        return TaskDefinition(
            task_type=task_type,
            operation=operation,
            parameters=parameters,
            data_source=data_source,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=confidence
        )

    def _classify_request(self, text: str) -> tuple[str, str, dict[str, Any], float]:
        """
        Classify the request into task type and extract parameters.

        Returns:
            Tuple of (task_type, operation, parameters, confidence_score)
        """
        # Check for ML patterns first (more specific)
        ml_result = self._check_ml_patterns(text)
        if ml_result[3] > 0.5:  # confidence threshold
            return ml_result

        # Check for statistical patterns
        stats_result = self._check_stats_patterns(text)
        if stats_result[3] > 0.3:  # lower threshold for stats
            return stats_result

        # Check for data info patterns
        data_info_result = self._check_data_info_patterns(text)
        if data_info_result[3] > 0.3:
            return data_info_result

        # Default to low confidence stats request
        return ("stats", "unknown", {}, 0.1)

    def _check_stats_patterns(self, text: str) -> tuple[str, str, dict[str, Any], float]:
        """Check for statistical operation patterns."""
        operations = []
        columns = self.extract_column_names(text)
        total_confidence = 0.0

        for operation, pattern in self.stats_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                operations.append(operation)
                total_confidence += 0.8

        if not operations:
            return ("stats", "unknown", {}, 0.0)

        # Determine specific operation
        if len(operations) == 1:
            operation = f"{operations[0]}_analysis"
        elif 'summary' in operations or len(operations) > 3:
            operation = "descriptive_stats"
        elif 'correlation' in operations:
            operation = "correlation_analysis"
        else:
            operation = "descriptive_stats"

        parameters = {
            "statistics": operations,
            "columns": columns if columns else ["all"]
        }

        confidence = min(total_confidence / len(operations), 1.0) if operations else 0.0
        return ("stats", operation, parameters, confidence)

    def _check_ml_patterns(self, text: str) -> tuple[str, str, dict[str, Any], float]:
        """Check for machine learning operation patterns."""
        ml_operations = []
        model_types = []
        confidence = 0.0

        for operation, pattern in self.ml_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if operation in ['train', 'predict']:
                    ml_operations.append(operation)
                    confidence += 0.9
                else:
                    model_types.append(operation)
                    confidence += 0.3

        if not ml_operations:
            return ("ml_train", "unknown", {}, 0.0)

        # Extract target and features
        target = self._extract_target_variable(text)
        features = self._extract_features(text)

        # Determine operation
        if 'predict' in ml_operations:
            task_type = "ml_score"
            operation = "predict"
        else:
            task_type = "ml_train"
            operation = "train_model"

        # Determine model type
        model_type = "auto"  # default
        if model_types:
            model_type = model_types[0]
        elif 'regression' in text.lower():
            model_type = "regression"
        elif 'classification' in text.lower() or 'classify' in text.lower():
            model_type = "classification"

        parameters = {
            "model_type": model_type,
            "target": target,
            "features": features
        }

        # Adjust confidence based on completeness
        if target:
            confidence += 0.2
        if features:
            confidence += 0.2

        return (task_type, operation, parameters, min(confidence, 1.0))

    def _check_data_info_patterns(self, text: str) -> tuple[str, str, dict[str, Any], float]:
        """Check for data information requests."""
        info_patterns = [
            r'(what|show|display|tell me about).*data',
            r'(columns|fields|variables)',
            r'(shape|size|dimensions)',
            r'(info|information|details)',
            r'(head|first.*rows)',
            r'(describe|overview)',
        ]

        confidence = 0.0
        for pattern in info_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence += 0.6

        if confidence > 0.3:
            return ("data_info", "describe_data", {}, min(confidence, 1.0))

        return ("data_info", "unknown", {}, 0.0)

    def extract_column_names(self, text: str) -> list[str]:
        """Extract column names referenced in the text."""
        columns = []

        for pattern in self.column_patterns.values():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the last group (column name)
                column = match.groups()[-1]
                if column and column not in columns:
                    columns.append(column.strip())

        # Also look for quoted column names
        quoted_pattern = r'["\']([a-zA-Z_][a-zA-Z0-9_\s]*)["\']'
        quoted_matches = re.finditer(quoted_pattern, text)
        for match in quoted_matches:
            column = match.group(1).strip()
            if column and column not in columns:
                columns.append(column)

        return columns

    def _extract_target_variable(self, text: str) -> Optional[str]:
        """Extract target variable from ML request."""
        pattern = self.column_patterns['target_variable']
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.groups()[-1].strip()
        return None

    def _extract_features(self, text: str) -> list[str]:
        """Extract feature variables from ML request."""
        pattern = self.column_patterns['features']
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            features_text = match.groups()[-1]
            # Split by comma and clean up
            features = [f.strip() for f in features_text.split(',')]
            return [f for f in features if f and f.lower() not in ['and', 'or']]
        return []


# Convenience functions for backward compatibility
def parse_stats_request(text: str) -> dict[str, Any]:
    """Parse statistical operation request and return parameters."""
    parser = RequestParser()
    result = parser._check_stats_patterns(text.lower())
    return result[2]  # parameters


def parse_ml_request(text: str) -> dict[str, Any]:
    """Parse ML operation request and return parameters."""
    parser = RequestParser()
    result = parser._check_ml_patterns(text.lower())
    return result[2]  # parameters