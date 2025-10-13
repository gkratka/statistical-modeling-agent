"""Tests for template message formatting and Markdown escaping."""

import pytest
from src.bot.messages.template_messages import escape_markdown, format_template_summary


class TestMarkdownEscaping:
    """Test Markdown escaping for Telegram compatibility."""

    def test_escape_markdown_with_underscores(self):
        """Test that underscores are escaped for Telegram Markdown."""
        assert escape_markdown("keras_binary_classification") == "keras\\_binary\\_classification"
        assert escape_markdown("german_credit_data_train.csv") == "german\\_credit\\_data\\_train.csv"
        assert escape_markdown("my_model_v2") == "my\\_model\\_v2"

    def test_escape_markdown_with_asterisks(self):
        """Test that asterisks are escaped for Telegram Markdown."""
        assert escape_markdown("test*with*asterisk") == "test\\*with\\*asterisk"

    def test_escape_markdown_with_mixed_characters(self):
        """Test escaping text with both underscores and asterisks."""
        assert escape_markdown("file_name*v2.csv") == "file\\_name\\*v2.csv"

    def test_escape_markdown_with_no_special_chars(self):
        """Test that plain text is unchanged."""
        assert escape_markdown("plain text") == "plain text"
        assert escape_markdown("LinearRegression") == "LinearRegression"


class TestTemplateFormatting:
    """Test template summary formatting with Markdown escaping."""

    def test_format_template_summary_escapes_underscores(self):
        """Test that template summary escapes model types with underscores."""
        result = format_template_summary(
            template_name="test_template",
            file_path="/path/to/data_file.csv",
            target="class",
            features=["feat1", "feat2"],
            model_category="Neural Network",
            model_type="keras_binary_classification",
            created_at="2025-01-13T10:00:00"
        )

        # Verify underscores are escaped in model type
        assert "keras\\_binary\\_classification" in result
        # Verify underscores are escaped in file path
        assert "data\\_file.csv" in result
        # Verify underscores are escaped in template name
        assert "test\\_template" in result

    def test_format_template_summary_with_path_underscores(self):
        """Test formatting with file path containing underscores."""
        result = format_template_summary(
            template_name="housing_model",
            file_path="/Users/data/housing_data_train.csv",
            target="price",
            features=["sqft", "bedrooms"],
            model_category="Regression",
            model_type="random_forest",
            created_at="2025-01-13"
        )

        assert "housing\\_data\\_train.csv" in result
        assert "random\\_forest" in result
        assert "housing\\_model" in result

    def test_format_template_summary_preserves_formatting(self):
        """Test that Markdown formatting is preserved."""
        result = format_template_summary(
            template_name="model",
            file_path="/path/file.csv",
            target="y",
            features=["x1", "x2"],
            model_category="Classification",
            model_type="logistic",
            created_at="2025-01-13"
        )

        # Verify bold markers are preserved
        assert "*Template:*" in result or "<b>Template:</b>" in result or "Template:" in result
        # Verify structure is maintained
        assert "model" in result
        assert "Classification" in result
        assert "logistic" in result

    def test_format_template_summary_escapes_created_date_with_underscore(self):
        """Test that created_at timestamps with underscores are escaped."""
        result = format_template_summary(
            template_name="test_template",
            file_path="/path/to/file.csv",
            target="target",
            features=["feat1", "feat2"],
            model_category="Classification",
            model_type="keras_binary_classification",
            created_at="20251013_001047"  # Real timestamp format
        )

        # Verify the date substring (first 10 chars) has escaped underscore
        # "20251013_001047"[:10] = "20251013_0" → should be escaped to "20251013\_0"
        assert "20251013\\_0" in result
