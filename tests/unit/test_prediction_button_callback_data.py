"""
TDD Tests for Prediction Workflow Button Callback Data.

These tests verify that inline keyboard button callback_data respects Telegram's
64-byte limit to prevent "Button_data_invalid" API errors.

Background:
Bug discovered when user selected 20 features for prediction. Bot tried to display
model selection buttons with callback_data containing full model IDs like:
"pred_model_model_7715560927_keras_binary_classification_20251009_211219"

This exceeded Telegram's 64-byte limit (actual: 71 bytes), causing API to reject
with "BadRequest - Button_data_invalid" error.

Bug Fix: Use button indices (0-9) instead of full model IDs in callback_data,
storing the model list in session state for lookup.
"""

import pytest
from src.bot.messages.prediction_messages import create_model_selection_buttons


class TestCallbackDataLength:
    """Test callback data respects Telegram's 64-byte limit."""

    def test_telegram_callback_data_max_length(self):
        """
        Document Telegram's callback_data constraint.

        Telegram Bot API documentation:
        - callback_data: String, 1-64 bytes
        - Data to be sent in a callback query to the bot when button is pressed
        """
        MAX_CALLBACK_DATA_BYTES = 64
        assert MAX_CALLBACK_DATA_BYTES == 64, "Telegram API constraint"

    def test_short_model_id_callback_data_length(self):
        """
        Test callback data length for short model IDs (should pass).

        Short model types like 'linear', 'svm', 'logistic' produce
        callback data well under 64 bytes.
        """
        models = [
            {
                'model_id': 'model_12345_linear_20251009_120000',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'metrics': {'r2': 0.85}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Extract callback data
        callback_data = buttons[0][0].callback_data
        byte_length = len(callback_data.encode('utf-8'))

        # Should be under 64 bytes
        assert byte_length <= 64, f"Callback data {byte_length} bytes exceeds 64-byte limit: {callback_data}"

    def test_keras_model_id_callback_data_length(self):
        """
        REGRESSION TEST: Verify Keras model callback data fits within limit.

        Bug Report:
        User selected 20 features for prediction. Bot found compatible Keras models
        but failed to display them. Log showed: "BadRequest - Button_data_invalid"

        Root Cause:
        Callback data for Keras binary classification models exceeded 64 bytes:
        "pred_model_model_7715560927_keras_binary_classification_20251009_211219"
        = 71 bytes (7 bytes over limit)

        Fix:
        Use button indices instead of full model IDs in callback_data.

        Test Strategy:
        This test uses the EXACT model ID format that caused the bug.
        """
        # Real model that caused the bug
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Extract callback data
        callback_data = buttons[0][0].callback_data
        byte_length = len(callback_data.encode('utf-8'))

        # Critical assertion: Must be under 64 bytes
        assert byte_length <= 64, \
            f"BUG: Keras model callback data {byte_length} bytes exceeds 64-byte limit!\n" \
            f"Callback data: {callback_data}\n" \
            f"This will cause Telegram API 'Button_data_invalid' error."

    def test_multiple_keras_models_all_under_limit(self):
        """
        Test that ALL buttons for multiple Keras models fit within limit.

        Scenario: User has 5 Keras models trained, bot shows all 5 as buttons.
        ALL buttons must have valid callback data.
        """
        models = [
            {
                'model_id': f'model_7715560927_keras_binary_classification_2025100{i}_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'metrics': {'accuracy': 0.90 + i * 0.01}
            }
            for i in range(5)
        ]

        buttons = create_model_selection_buttons(models)

        # Verify ALL buttons have valid callback data
        for i, button_row in enumerate(buttons):
            callback_data = button_row[0].callback_data
            byte_length = len(callback_data.encode('utf-8'))

            assert byte_length <= 64, \
                f"Button {i} callback data {byte_length} bytes exceeds limit: {callback_data}"


class TestIndexBasedButtonCreation:
    """Test button creation using indices instead of model IDs."""

    def test_index_based_callback_data_format(self):
        """
        Test that callback data uses index format after fix.

        Expected format after fix: "pred_model_0", "pred_model_1", etc.
        This guarantees callback data stays under 64 bytes.
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'metrics': {'accuracy': 0.925}
            },
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251010_101010',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Status',
                'metrics': {'accuracy': 0.88}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # After fix, callback data should be "pred_model_0", "pred_model_1"
        assert buttons[0][0].callback_data == "pred_model_0", \
            "First button should use index 0"

        assert buttons[1][0].callback_data == "pred_model_1", \
            "Second button should use index 1"

    def test_index_based_callback_data_length(self):
        """
        Test that index-based callback data is always short.

        "pred_model_0" = 12 bytes (well under 64-byte limit)
        "pred_model_9" = 12 bytes (for 10th model)
        "pred_model_99" = 13 bytes (future-proof for >10 models)
        """
        models = [
            {
                'model_id': f'model_{i}_very_long_model_type_name_that_would_exceed_limit',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': f'target_{i}',
                'metrics': {'accuracy': 0.9}
            }
            for i in range(10)
        ]

        buttons = create_model_selection_buttons(models)

        # ALL model buttons (excluding back button) should have short callback data
        model_buttons = buttons[:-1]  # Exclude last button (back button)
        for i, button_row in enumerate(model_buttons):
            callback_data = button_row[0].callback_data
            byte_length = len(callback_data.encode('utf-8'))

            # Should be ~12 bytes for single digit indices
            assert byte_length <= 15, \
                f"Index-based callback data should be short, got {byte_length} bytes"

            # Verify format
            assert callback_data == f"pred_model_{i}", \
                f"Expected 'pred_model_{i}', got '{callback_data}'"


class TestButtonDisplayText:
    """Test that button display text is correct (independent of callback data)."""

    def test_button_text_shows_model_number_and_type(self):
        """
        Test button display text shows user-friendly model information.

        Display text: "1. Keras Binary Classification"
        Callback data: "pred_model_0" (internal index)
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Display text should be user-friendly (not affected by callback data fix)
        button_text = buttons[0][0].text
        assert "1." in button_text, "Button should show number 1"
        assert "Keras" in button_text or "keras" in button_text.lower(), \
            "Button should show model type"

    def test_button_text_numbering_starts_at_one(self):
        """
        Test that button display text starts numbering at 1 for users.

        Internal indices: 0, 1, 2
        User-facing display: 1, 2, 3
        """
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': f'type_{i}',
                'task_type': 'regression',
                'target_column': 'target',
                'metrics': {'r2': 0.8}
            }
            for i in range(3)
        ]

        buttons = create_model_selection_buttons(models)

        # Display text should start at 1 for users
        assert buttons[0][0].text.startswith("1."), "First button should show '1.'"
        assert buttons[1][0].text.startswith("2."), "Second button should show '2.'"
        assert buttons[2][0].text.startswith("3."), "Third button should show '3.'"

        # But callback data should use 0-based indices internally
        assert buttons[0][0].callback_data == "pred_model_0"
        assert buttons[1][0].callback_data == "pred_model_1"
        assert buttons[2][0].callback_data == "pred_model_2"


class TestEdgeCases:
    """Test edge cases for button creation."""

    def test_empty_model_list(self):
        """Test button creation with empty model list."""
        models = []
        buttons = create_model_selection_buttons(models)
        # Should only have back button
        assert len(buttons) == 1, "Empty model list should have only back button"
        assert buttons[0][0].callback_data == "workflow_back"

    def test_single_model(self):
        """Test button creation with single model."""
        models = [
            {
                'model_id': 'model_12345_linear',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'metrics': {'r2': 0.85}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Should have 1 model button + 1 back button = 2 total
        assert len(buttons) == 2, "Should create 1 model button + 1 back button"
        assert buttons[0][0].callback_data == "pred_model_0"
        assert buttons[1][0].callback_data == "workflow_back"

    def test_max_ten_models_displayed(self):
        """
        Test that only first 10 models are shown as buttons.

        Telegram best practice: Don't overwhelm users with too many buttons.
        """
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'target',
                'metrics': {'r2': 0.8}
            }
            for i in range(15)  # 15 models
        ]

        buttons = create_model_selection_buttons(models)

        # Should have 10 model buttons + 1 back button = 11 total
        assert len(buttons) == 11, "Should limit to 10 models + 1 back button"

        # Check last model button (index 9, 10th button)
        assert buttons[9][0].callback_data == "pred_model_9"

        # Check back button is last
        assert buttons[10][0].callback_data == "workflow_back"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
