"""
TDD Tests for Prediction Workflow Markdown Escaping.

These tests verify that all user-provided dynamic content (column names, file paths,
model types, etc.) is properly escaped before being inserted into Telegram markdown
messages.

Background:
Bug discovered in /predict workflow where column names with special characters
caused Telegram API to fail with "Can't parse entities" error. This was because
dynamic content was inserted into markdown messages without proper escaping.

Bug Fix: All dynamic user content must be escaped with escape_markdown_v1()
before being inserted into markdown-formatted messages.
"""

import pytest
from src.bot.messages.prediction_messages import PredictionMessages


class TestSchemaConfirmationEscaping:
    """Test markdown escaping in schema confirmation messages."""

    def test_schema_confirmation_with_underscore_columns(self):
        """
        Test that column names with underscores are properly escaped.

        Underscores are special in Markdown (italic formatting), so they must
        be escaped as \\_
        """
        summary = "Test dataset"
        columns_with_underscores = [
            "user_id",
            "first_name",
            "last_name",
            "created_at"
        ]

        message = PredictionMessages.schema_confirmation_prompt(
            summary, columns_with_underscores
        )

        # Verify escaped underscores in output
        assert "user\\_id" in message
        assert "first\\_name" in message
        assert "last\\_name" in message
        assert "created\\_at" in message

        # Verify unescaped underscores NOT in output
        assert "user_id`" not in message or "`user\\_id`" in message

    def test_schema_confirmation_with_asterisk_columns(self):
        """
        Test that column names with asterisks are properly escaped.

        Asterisks are special in Markdown (bold formatting).
        """
        summary = "Test dataset"
        columns_with_asterisks = ["feature*1", "feature*2", "**important**"]

        message = PredictionMessages.schema_confirmation_prompt(
            summary, columns_with_asterisks
        )

        # Verify escaped asterisks
        assert "\\*" in message
        assert "feature\\*1" in message

    def test_schema_confirmation_with_backtick_columns(self):
        """
        Test that column names with backticks are properly escaped.

        Backticks are special in Markdown (code formatting).
        """
        summary = "Test dataset"
        columns_with_backticks = ["col`1", "col`2"]

        message = PredictionMessages.schema_confirmation_prompt(
            summary, columns_with_backticks
        )

        # Verify escaped backticks
        assert "\\`" in message

    def test_schema_confirmation_with_bracket_columns(self):
        """
        Test that column names with brackets are properly escaped.

        Brackets are special in Markdown (link formatting).
        """
        summary = "Test dataset"
        columns_with_brackets = ["col[0]", "data[key]"]

        message = PredictionMessages.schema_confirmation_prompt(
            summary, columns_with_brackets
        )

        # Verify escaped brackets
        assert "\\[" in message

    def test_schema_confirmation_with_many_columns(self):
        """
        Test handling of datasets with many columns (German credit dataset had 20).

        This reproduces the exact bug scenario from user's report.
        """
        summary = "📊 **Dataset Loaded**\n\n**Shape:** 800 rows × 20 columns"
        columns = [f"Attribute{i}" for i in range(1, 21)]

        message = PredictionMessages.schema_confirmation_prompt(summary, columns)

        # Should not raise Telegram API error
        assert len(message) > 0
        assert "Attribute1" in message
        assert "Available Columns (20)" in message


class TestFeatureSelectionEscaping:
    """Test markdown escaping in feature selection messages."""

    def test_feature_selection_prompt_escaping(self):
        """Test that feature names in selection prompt are escaped."""
        columns = ["user_id", "total_amount", "status*", "data[0]"]
        dataset_shape = (1000, 4)

        message = PredictionMessages.feature_selection_prompt(columns, dataset_shape)

        # Verify escaped special characters
        assert "user\\_id" in message
        assert "\\*" in message
        assert "\\[" in message

    def test_features_selected_message_escaping(self):
        """Test that selected features in confirmation are escaped."""
        selected_features = ["user_id", "amount*important", "data[key]"]

        message = PredictionMessages.features_selected_message(selected_features)

        # Verify escaped special characters
        assert "user\\_id" in message
        assert "\\*" in message
        assert "\\[" in message

    def test_feature_validation_error_escaping(self):
        """Test that invalid feature names in errors are escaped."""
        details = {
            'invalid': ["bad_feature*", "wrong[column]", "test_col"]
        }

        message = PredictionMessages.feature_validation_error(
            "Some features are invalid",
            details
        )

        # Verify escaped special characters
        assert "\\*" in message
        assert "\\[" in message
        assert "\\_" in message


class TestModelSelectionEscaping:
    """Test markdown escaping in model selection messages."""

    def test_model_selection_prompt_escaping(self):
        """Test that model types and target columns are escaped."""
        models = [
            {
                'model_id': 'model_1',
                'model_type': 'random_forest',
                'task_type': 'classification',
                'target_column': 'is_fraud',
                'metrics': {'accuracy': 0.95}
            },
            {
                'model_id': 'model_2',
                'model_type': 'gradient_boosting',
                'task_type': 'regression',
                'target_column': 'total_amount',
                'metrics': {'r2': 0.88}
            }
        ]
        selected_features = ["user_id", "transaction_amount"]

        message = PredictionMessages.model_selection_prompt(models, selected_features)

        # Verify model type escaping
        assert "random\\_forest" in message or "Random\\_Forest" in message or "Random Forest" in message

        # Verify target column escaping
        assert "is\\_fraud" in message
        assert "total\\_amount" in message

    def test_model_selected_message_escaping(self):
        """Test that model info in selection confirmation is escaped."""
        message = PredictionMessages.model_selected_message(
            model_id="model_12345_random_forest",
            model_type="keras_binary_classification",
            target_column="is_default"
        )

        # Verify escaping
        assert "\\`model\\_12345\\_random\\_forest\\`" in message or "model\\_12345\\_random\\_forest" in message
        assert "is\\_default" in message

    def test_model_selection_escapes_task_type_with_underscores(self):
        """
        REGRESSION TEST: Verify task_type with underscores is properly escaped.

        Bug Report:
        When user selected 20 features for prediction, bot showed "Loading compatible models..."
        but never displayed the model list. Log showed: "BadRequest - Can't parse entities:
        can't find end of the entity starting at byte offset 921"

        Root Cause:
        Line 216 in prediction_messages.py did NOT escape task_type before inserting into
        markdown message. When task_type='binary_classification', the underscore triggered
        Telegram's italic formatting parser, which failed when it couldn't find a closing
        underscore.

        Fix:
        task_type must be escaped with escape_markdown_v1() before insertion.

        Test Strategy:
        This test uses task_type values that contain underscores (the bug trigger):
        - 'binary_classification'
        - 'multiclass_classification'

        Expected: Underscores must be escaped as \\_

        Note: Display format changed to "| task_type" instead of "(task_type)"
        but escaping requirement remains critical.
        """
        models = [
            {
                'model_id': 'model_1',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',  # ← Bug trigger!
                'target_column': 'Churn',
                'metrics': {'accuracy': 0.925}
            },
            {
                'model_id': 'model_2',
                'model_type': 'random_forest',
                'task_type': 'multiclass_classification',  # ← Another underscore case
                'target_column': 'Category',
                'metrics': {'accuracy': 0.88}
            }
        ]
        selected_features = ['feature1', 'feature2']

        message = PredictionMessages.model_selection_prompt(models, selected_features)

        # Critical Assertion 1: task_type underscores MUST be escaped
        # Telegram will reject message if unescaped underscores exist
        assert "| binary\\_classification" in message, \
            "BUG: task_type='binary_classification' underscore not escaped!"

        assert "| multiclass\\_classification" in message, \
            "BUG: task_type='multiclass_classification' underscore not escaped!"

        # Critical Assertion 2: Verify NO unescaped underscores in task_type
        # If these assertions fail, the bug still exists
        # Check that unescaped versions don't appear after pipe separator
        assert "| binary_classification\n" not in message, \
            "BUG: Unescaped task_type found - will cause Telegram API error!"

        assert "| multiclass_classification\n" not in message, \
            "BUG: Unescaped task_type found - will cause Telegram API error!"

        # Additional verification: Message should be safe to send
        assert len(message) > 0, "Message should not be empty"


class TestPredictionColumnEscaping:
    """Test markdown escaping in prediction column messages."""

    def test_prediction_column_prompt_escaping(self):
        """Test that target column is escaped."""
        target_column = "total_revenue"
        existing_columns = ["user_id", "signup_date", "plan_type"]

        message = PredictionMessages.prediction_column_prompt(
            target_column, existing_columns
        )

        # Verify target column escaping
        assert "total\\_revenue" in message
        # Verify predicted column name also escaped
        assert "total\\_revenue\\_predicted" in message

    def test_column_name_conflict_error_escaping(self):
        """Test that column names in conflict errors are escaped."""
        column_name = "predicted_value"
        existing_columns = ["user_id", "amount*", "status[0]"]

        message = PredictionMessages.column_name_conflict_error(
            column_name, existing_columns
        )

        # Verify escaping
        assert "predicted\\_value" in message
        assert "user\\_id" in message
        assert "\\*" in message
        assert "\\[" in message


class TestErrorMessageEscaping:
    """Test markdown escaping in error messages."""

    def test_file_loading_error_escaping(self):
        """
        Test that file paths and error details are escaped in error messages.

        This is critical - error messages themselves must not cause markdown errors!
        """
        file_path = "/Users/test/data_files/dataset_v2.csv"
        error_details = "Path not in allowed directories"

        message = PredictionMessages.file_loading_error(file_path, error_details)

        # Verify file path is escaped
        assert "\\_" in message  # Escaped underscores

        # Message should not crash when sent to Telegram
        assert len(message) > 0

    def test_file_loading_error_with_special_chars(self):
        """Test file paths with various special characters."""
        file_path = "/tmp/test*file[1]_data.csv"
        error_details = "Invalid format: missing column `user_id`"

        message = PredictionMessages.file_loading_error(file_path, error_details)

        # Verify escaping
        assert "\\*" in message
        assert "\\[" in message
        assert "\\_" in message
        assert "\\`" in message

    def test_unexpected_error_escaping(self):
        """Test that unexpected error messages are safe."""
        error_msg = "Database error: table `users_data` not found"

        message = PredictionMessages.unexpected_error(error_msg)

        # Should escape backticks and underscores
        assert "\\`" in message
        assert "\\_" in message


class TestButtonCreationSafety:
    """Test that button creation doesn't break with special characters."""

    def test_model_selection_buttons_with_special_chars(self):
        """Test button creation with model types containing special characters."""
        from src.bot.messages.prediction_messages import create_model_selection_buttons

        models = [
            {
                'model_id': 'model_1_test',
                'model_type': 'random_forest',
                'target_column': 'is_fraud'
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Should not crash
        assert len(buttons) > 0

        # Callback data should use index-based format (pred_model_0)
        # Model buttons are all except the last (back button)
        model_buttons = buttons[:-1]
        assert any('pred_model_0' == btn[0].callback_data for btn in model_buttons if len(btn) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
