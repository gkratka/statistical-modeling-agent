"""
Test suite for script generation integration in parser.
"""

import pytest
from src.core.parser import RequestParser, TaskDefinition
from src.utils.exceptions import ParseError


class TestScriptParsingIntegration:
    """Test script generation request parsing."""

    def setup_method(self):
        """Set up test environment."""
        self.parser = RequestParser()

    def test_parse_script_command_basic(self):
        """Test basic /script command parsing."""
        task = self.parser.parse_request(
            text="/script descriptive",
            user_id=123,
            conversation_id="test_conv"
        )

        assert task.task_type == "script"
        assert task.operation == "descriptive"
        assert task.confidence_score > 0.8

    def test_parse_script_command_with_parameters(self):
        """Test /script command with parameters."""
        task = self.parser.parse_request(
            text="/script correlation for sales and profit",
            user_id=123,
            conversation_id="test_conv"
        )

        assert task.task_type == "script"
        assert task.operation == "correlation"
        assert "columns" in task.parameters
        assert "sales" in task.parameters["columns"]
        assert "profit" in task.parameters["columns"]

    def test_parse_natural_language_script_generation(self):
        """Test natural language script generation requests."""
        test_cases = [
            ("Generate a script for descriptive statistics", "descriptive"),
            ("Create a Python script for correlation analysis", "correlation"),
            ("Generate script to train a classifier", "train_classifier"),
            ("Make a Python script for data analysis", "descriptive"),
        ]

        for text, expected_operation in test_cases:
            task = self.parser.parse_request(
                text=text,
                user_id=123,
                conversation_id="test_conv"
            )

            assert task.task_type == "script"
            assert task.operation == expected_operation
            assert task.confidence_score > 0.7

    def test_parse_script_with_column_extraction(self):
        """Test script parsing with column extraction."""
        task = self.parser.parse_request(
            text="Generate a script to analyze sales, profit, and quantity columns",
            user_id=123,
            conversation_id="test_conv"
        )

        assert task.task_type == "script"
        assert "columns" in task.parameters
        expected_columns = {"sales", "profit", "quantity"}
        parsed_columns = set(task.parameters["columns"])
        assert expected_columns.issubset(parsed_columns)

    def test_parse_script_with_statistics_specification(self):
        """Test script parsing with specific statistics."""
        task = self.parser.parse_request(
            text="/script descriptive with mean, std, and correlation",
            user_id=123,
            conversation_id="test_conv"
        )

        assert task.task_type == "script"
        assert task.operation == "descriptive"
        assert "statistics" in task.parameters
        stats = task.parameters["statistics"]
        assert "mean" in stats
        assert "std" in stats

    def test_parse_invalid_script_request(self):
        """Test parsing of invalid script requests."""
        invalid_requests = [
            "/script unknown_operation",
            "script without proper format",
            "/script",  # No operation specified
        ]

        for invalid_text in invalid_requests:
            with pytest.raises(ParseError):
                self.parser.parse_request(
                    text=invalid_text,
                    user_id=123,
                    conversation_id="test_conv"
                )

    def test_parse_script_ml_operations(self):
        """Test parsing ML script generation requests."""
        ml_test_cases = [
            ("Generate script to train a classifier", "train_classifier"),
            ("Create Python code for model training", "train_classifier"),
            ("Generate prediction script", "predict"),
            ("Make a script for machine learning", "train_classifier"),
        ]

        for text, expected_operation in ml_test_cases:
            task = self.parser.parse_request(
                text=text,
                user_id=123,
                conversation_id="test_conv"
            )

            assert task.task_type == "script"
            assert task.operation == expected_operation

    def test_script_vs_direct_stats_disambiguation(self):
        """Test that script requests are distinguished from direct stats requests."""
        # Direct stats request
        direct_task = self.parser.parse_request(
            text="Calculate statistics for sales",
            user_id=123,
            conversation_id="test_conv"
        )
        assert direct_task.task_type == "stats"

        # Script generation request
        script_task = self.parser.parse_request(
            text="Generate a script to calculate statistics for sales",
            user_id=123,
            conversation_id="test_conv"
        )
        assert script_task.task_type == "script"

    def test_parse_script_with_method_specification(self):
        """Test script parsing with method specification."""
        task = self.parser.parse_request(
            text="/script correlation using pearson method",
            user_id=123,
            conversation_id="test_conv"
        )

        assert task.task_type == "script"
        assert task.operation == "correlation"
        assert "method" in task.parameters
        assert task.parameters["method"] == "pearson"

    def test_confidence_scoring_for_script_requests(self):
        """Test confidence scoring for different script request types."""
        # High confidence: explicit script command
        high_conf_task = self.parser.parse_request(
            text="/script descriptive",
            user_id=123,
            conversation_id="test_conv"
        )
        assert high_conf_task.confidence_score > 0.9

        # Medium confidence: natural language with clear intent
        med_conf_task = self.parser.parse_request(
            text="Generate a Python script for data analysis",
            user_id=123,
            conversation_id="test_conv"
        )
        assert 0.7 <= med_conf_task.confidence_score <= 0.9

        # Lower confidence: ambiguous request
        try:
            low_conf_task = self.parser.parse_request(
                text="Create something for my data",
                user_id=123,
                conversation_id="test_conv"
            )
            # If it doesn't raise an exception, confidence should be low
            assert low_conf_task.confidence_score < 0.7
        except ParseError:
            # ParseError is also acceptable for ambiguous requests
            pass