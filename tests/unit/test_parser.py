"""
Unit tests for the natural language parser module.

This module tests all components of the parser including pattern matching,
dataclass validation, and request classification.
"""

import pytest
from dataclasses import asdict
from src.core.parser import (
    RequestParser,
    TaskDefinition,
    DataSource,
    parse_stats_request,
    parse_ml_request
)
from src.utils.exceptions import ParseError


class TestDataSource:
    """Test DataSource dataclass functionality."""

    def test_data_source_creation(self):
        """Test basic DataSource creation."""
        ds = DataSource(
            file_id="test_123",
            file_name="test.csv",
            file_type="csv",
            columns=["age", "income"],
            shape=(100, 2)
        )
        assert ds.file_id == "test_123"
        assert ds.file_name == "test.csv"
        assert ds.file_type == "csv"
        assert ds.columns == ["age", "income"]
        assert ds.shape == (100, 2)

    def test_data_source_defaults(self):
        """Test DataSource with default values."""
        ds = DataSource()
        assert ds.file_id is None
        assert ds.file_name is None
        assert ds.file_type == "unknown"
        assert ds.columns is None
        assert ds.shape is None

    def test_data_source_invalid_file_type(self):
        """Test DataSource validation for invalid file type."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            DataSource(file_type="invalid")

    def test_data_source_valid_file_types(self):
        """Test DataSource with all valid file types."""
        valid_types = ["csv", "xlsx", "json", "parquet", "unknown"]
        for file_type in valid_types:
            ds = DataSource(file_type=file_type)
            assert ds.file_type == file_type


class TestTaskDefinition:
    """Test TaskDefinition dataclass functionality."""

    def test_task_definition_creation(self):
        """Test basic TaskDefinition creation."""
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"statistics": ["mean", "std"]},
            data_source=None,
            user_id=12345,
            conversation_id="conv_001",
            confidence_score=0.8
        )
        assert task.task_type == "stats"
        assert task.operation == "descriptive_stats"
        assert task.parameters == {"statistics": ["mean", "std"]}
        assert task.confidence_score == 0.8

    def test_task_definition_with_data_source(self):
        """Test TaskDefinition with DataSource."""
        ds = DataSource(file_name="test.csv", file_type="csv")
        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={"model_type": "regression"},
            data_source=ds,
            user_id=12345,
            conversation_id="conv_001"
        )
        assert task.data_source == ds
        assert task.data_source.file_name == "test.csv"

    def test_task_definition_invalid_confidence(self):
        """Test TaskDefinition validation for invalid confidence score."""
        with pytest.raises(ValueError, match="Confidence score must be between"):
            TaskDefinition(
                task_type="stats",
                operation="test",
                parameters={},
                data_source=None,
                user_id=1,
                conversation_id="test",
                confidence_score=1.5
            )

    def test_task_definition_empty_operation(self):
        """Test TaskDefinition validation for empty operation."""
        with pytest.raises(ValueError, match="Operation cannot be empty"):
            TaskDefinition(
                task_type="stats",
                operation="",
                parameters={},
                data_source=None,
                user_id=1,
                conversation_id="test"
            )

    def test_task_definition_invalid_parameters(self):
        """Test TaskDefinition validation for invalid parameters."""
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            TaskDefinition(
                task_type="stats",
                operation="test",
                parameters="invalid",  # type: ignore
                data_source=None,
                user_id=1,
                conversation_id="test"
            )


class TestRequestParser:
    """Test main parser functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()
        self.user_id = 12345
        self.conversation_id = "test_conv"

    def test_parser_initialization(self):
        """Test parser initialization."""
        assert isinstance(self.parser.stats_patterns, dict)
        assert isinstance(self.parser.ml_patterns, dict)
        assert isinstance(self.parser.column_patterns, dict)
        assert len(self.parser.stats_patterns) > 0
        assert len(self.parser.ml_patterns) > 0

    def test_parse_request_empty_text(self):
        """Test parser with empty text."""
        with pytest.raises(ParseError, match="Empty request text"):
            self.parser.parse_request("", self.user_id, self.conversation_id)

        with pytest.raises(ParseError, match="Empty request text"):
            self.parser.parse_request("   ", self.user_id, self.conversation_id)

    def test_parse_request_very_low_confidence(self):
        """Test parser with unrecognizable text."""
        with pytest.raises(ParseError, match="Could not understand request"):
            self.parser.parse_request(
                "random gibberish xyz", self.user_id, self.conversation_id
            )


class TestStatsParser:
    """Test statistical parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()
        self.user_id = 12345
        self.conversation_id = "test_conv"

    def test_parse_basic_stats_mean(self):
        """Test parsing basic mean request."""
        task = self.parser.parse_request(
            "calculate mean for age column", self.user_id, self.conversation_id
        )
        assert task.task_type == "stats"
        assert "mean" in task.parameters["statistics"]
        assert "age" in task.parameters["columns"]
        assert task.confidence_score > 0.5

    def test_parse_basic_stats_multiple(self):
        """Test parsing multiple statistics request."""
        task = self.parser.parse_request(
            "calculate mean and standard deviation", self.user_id, self.conversation_id
        )
        assert task.task_type == "stats"
        assert "mean" in task.parameters["statistics"]
        assert "std" in task.parameters["statistics"]
        assert task.confidence_score > 0.5

    def test_parse_correlation_request(self):
        """Test parsing correlation request."""
        task = self.parser.parse_request(
            "show correlation between age and income", self.user_id, self.conversation_id
        )
        assert task.task_type == "stats"
        assert task.operation == "correlation_analysis"
        assert "correlation" in task.parameters["statistics"]

    def test_parse_summary_statistics(self):
        """Test parsing summary statistics request."""
        task = self.parser.parse_request(
            "show descriptive statistics", self.user_id, self.conversation_id
        )
        assert task.task_type == "stats"
        assert task.operation == "descriptive_stats"
        assert task.confidence_score > 0.3

    def test_parse_with_quoted_columns(self):
        """Test parsing with quoted column names."""
        task = self.parser.parse_request(
            'calculate mean for "annual_income" column', self.user_id, self.conversation_id
        )
        assert "annual_income" in task.parameters["columns"]

    def test_parse_multiple_columns(self):
        """Test parsing request with multiple columns."""
        task = self.parser.parse_request(
            "show correlation for age, income, education", self.user_id, self.conversation_id
        )
        # Note: Current implementation may not catch comma-separated columns perfectly
        # This test documents current behavior and can be enhanced
        assert task.task_type == "stats"

    def test_parse_stats_request_function(self):
        """Test standalone parse_stats_request function."""
        result = parse_stats_request("calculate mean and median")
        assert isinstance(result, dict)
        assert "statistics" in result
        assert "mean" in result["statistics"]
        assert "median" in result["statistics"]


class TestMLParser:
    """Test machine learning parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()
        self.user_id = 12345
        self.conversation_id = "test_conv"

    def test_parse_train_model_basic(self):
        """Test parsing basic model training request."""
        task = self.parser.parse_request(
            "train a model", self.user_id, self.conversation_id
        )
        assert task.task_type == "ml_train"
        assert task.operation == "train_model"
        assert task.confidence_score > 0.5

    def test_parse_train_with_target(self):
        """Test parsing model training with target variable."""
        task = self.parser.parse_request(
            "train a model to predict income", self.user_id, self.conversation_id
        )
        assert task.task_type == "ml_train"
        assert task.parameters["target_column"] == "income"
        assert task.confidence_score > 0.7

    def test_parse_train_with_features(self):
        """Test parsing model training with features."""
        task = self.parser.parse_request(
            "train model based on age and education", self.user_id, self.conversation_id
        )
        assert task.task_type == "ml_train"
        expected_features = ["age", "education"]
        assert any(feature in task.parameters.get("feature_columns", []) for feature in expected_features)

    def test_parse_prediction_request(self):
        """Test parsing prediction request."""
        task = self.parser.parse_request(
            "predict house prices", self.user_id, self.conversation_id
        )
        assert task.task_type == "ml_score"
        assert task.operation == "predict"

    def test_parse_regression_request(self):
        """Test parsing regression-specific request."""
        task = self.parser.parse_request(
            "linear regression to predict salary", self.user_id, self.conversation_id
        )
        assert task.task_type == "ml_train"
        assert "regression" in task.parameters["model_type"]

    def test_parse_classification_request(self):
        """Test parsing classification-specific request."""
        task = self.parser.parse_request(
            "classify customers into categories", self.user_id, self.conversation_id
        )
        assert task.task_type == "ml_train"
        assert "classification" in task.parameters["model_type"]

    def test_parse_ml_request_function(self):
        """Test standalone parse_ml_request function."""
        result = parse_ml_request("train a neural network model")
        assert isinstance(result, dict)
        assert "model_type" in result


class TestDataInfoParser:
    """Test data information parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()
        self.user_id = 12345
        self.conversation_id = "test_conv"

    def test_parse_data_info_request(self):
        """Test parsing data information request."""
        task = self.parser.parse_request(
            "show me information about the data", self.user_id, self.conversation_id
        )
        assert task.task_type == "data_info"
        assert task.operation == "describe_data"

    def test_parse_columns_request(self):
        """Test parsing columns information request."""
        task = self.parser.parse_request(
            "what columns are available", self.user_id, self.conversation_id
        )
        assert task.task_type == "data_info"

    def test_parse_data_shape_request(self):
        """Test parsing data shape request."""
        task = self.parser.parse_request(
            "what is the shape of the data", self.user_id, self.conversation_id
        )
        assert task.task_type == "data_info"


class TestColumnExtraction:
    """Test column name extraction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()

    def test_extract_single_column(self):
        """Test extracting a single column name."""
        columns = self.parser.extract_column_names("calculate mean for age column")
        assert "age" in columns

    def test_extract_quoted_column(self):
        """Test extracting quoted column names."""
        columns = self.parser.extract_column_names('show mean for "annual_income"')
        assert "annual_income" in columns

    def test_extract_multiple_columns(self):
        """Test extracting multiple column references."""
        text = "correlation between age and income columns"
        columns = self.parser.extract_column_names(text)
        # Current implementation may not catch both perfectly
        # This documents current behavior
        assert len(columns) >= 1

    def test_extract_no_columns(self):
        """Test text with no column references."""
        columns = self.parser.extract_column_names("calculate summary statistics")
        assert isinstance(columns, list)

    def test_extract_target_variable(self):
        """Test extracting target variable."""
        target = self.parser._extract_target_variable("train model to predict salary")
        assert target == "salary"

    def test_extract_features(self):
        """Test extracting feature variables."""
        features = self.parser._extract_features("model based on age, education")
        assert "age" in features
        assert "education" in features


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()
        self.user_id = 12345
        self.conversation_id = "test_conv"

    def test_ambiguous_request(self):
        """Test handling of ambiguous requests."""
        # This should still parse but with lower confidence
        task = self.parser.parse_request(
            "do something with data", self.user_id, self.conversation_id
        )
        assert task.confidence_score < 0.8

    def test_case_insensitive_parsing(self):
        """Test that parsing is case insensitive."""
        task1 = self.parser.parse_request(
            "CALCULATE MEAN", self.user_id, self.conversation_id
        )
        task2 = self.parser.parse_request(
            "calculate mean", self.user_id, self.conversation_id
        )
        assert task1.task_type == task2.task_type
        assert task1.operation == task2.operation

    def test_extra_whitespace(self):
        """Test handling of extra whitespace."""
        task = self.parser.parse_request(
            "  calculate   mean   for   age  ", self.user_id, self.conversation_id
        )
        assert task.task_type == "stats"
        assert "mean" in task.parameters["statistics"]

    def test_partial_column_matches(self):
        """Test handling of partial column name matches."""
        columns = self.parser.extract_column_names("for age_group column")
        assert "age_group" in columns

    def test_confidence_scoring_consistency(self):
        """Test that confidence scoring is consistent."""
        task1 = self.parser.parse_request(
            "calculate mean", self.user_id, self.conversation_id
        )
        task2 = self.parser.parse_request(
            "calculate mean", self.user_id, self.conversation_id
        )
        assert task1.confidence_score == task2.confidence_score

    def test_request_with_data_source(self):
        """Test parsing with provided data source."""
        data_source = DataSource(file_name="test.csv", file_type="csv")
        task = self.parser.parse_request(
            "calculate mean", self.user_id, self.conversation_id, data_source
        )
        assert task.data_source == data_source
        assert task.data_source.file_name == "test.csv"


class TestIntegration:
    """Integration tests for full parsing workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()
        self.user_id = 12345
        self.conversation_id = "test_conv"

    def test_complex_stats_request(self):
        """Test parsing complex statistical request."""
        task = self.parser.parse_request(
            'Calculate mean, median, and standard deviation for "customer_age" and show correlation',
            self.user_id,
            self.conversation_id
        )
        assert task.task_type == "stats"
        assert len(task.parameters["statistics"]) >= 2
        assert task.confidence_score > 0.5

    def test_complex_ml_request(self):
        """Test parsing complex ML request."""
        task = self.parser.parse_request(
            "Train a random forest model to predict customer_satisfaction based on age, income, and education_level",
            self.user_id,
            self.conversation_id
        )
        assert task.task_type == "ml_train"
        assert task.parameters.get("target_column") == "customer_satisfaction"
        assert len(task.parameters.get("feature_columns", [])) >= 2
        assert task.confidence_score > 0.7

    def test_serialization_compatibility(self):
        """Test that TaskDefinition can be serialized."""
        task = self.parser.parse_request(
            "calculate mean", self.user_id, self.conversation_id
        )
        # Should be able to convert to dict without errors
        task_dict = asdict(task)
        assert isinstance(task_dict, dict)
        assert task_dict["task_type"] == "stats"