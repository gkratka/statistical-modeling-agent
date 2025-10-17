"""
TDD Tests for Prediction Button Text Enhancement.

These tests verify that model selection buttons display:
1. Feature counts (e.g., "20 features")
2. Custom model names when present
3. Model type when no custom name

User Issue:
When 21 Keras models are displayed, they all show identical text
"Keras_Binary_Classification" making them indistinguishable.

Solution:
Enhance button text to show: "{number}. {name} ({count} feature{s})"
Examples:
- "1. My Credit Model (20 features)"
- "1. Keras Binary Classification (20 features)"
- "1. Linear Regression (1 feature)"
"""

import pytest
from src.bot.messages.prediction_messages import create_model_selection_buttons


class TestButtonTextShowsFeatureCounts:
    """Test that button text displays feature counts."""

    def test_button_shows_feature_count(self):
        """
        Button text should include feature count.

        Expected: "1. Keras Binary Classification (20 features)"
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': ['Age', 'Income', 'Balance', 'NumProducts', 'HasCrCard',
                                   'IsActiveMember', 'EstimatedSalary', 'Geography_France',
                                   'Geography_Germany', 'Geography_Spain', 'Gender_Female',
                                   'Gender_Male', 'Tenure', 'CreditScore', 'Complain',
                                   'Satisfaction Score', 'Card Type', 'Point Earned',
                                   'NumOfServices', 'HasMultipleProducts'],  # 20 features
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should show feature count
        assert "(20 features)" in button_text, \
            f"Button text should show '(20 features)', got: {button_text}"

        # Should start with number
        assert button_text.startswith("1."), \
            f"Button text should start with '1.', got: {button_text}"

    def test_button_shows_singular_feature_text(self):
        """
        Button text should use correct grammar for single feature.

        Expected: "1. Linear Regression (1 feature)" not "1 features"
        """
        models = [
            {
                'model_id': 'model_12345_linear_20251009_120000',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft'],  # 1 feature
                'metrics': {'r2': 0.85}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should use singular "feature" not "features"
        assert "(1 feature)" in button_text, \
            f"Button text should show '(1 feature)', got: {button_text}"

        # Should NOT have "(1 features)"
        assert "(1 features)" not in button_text, \
            f"Button text should not show '(1 features)', got: {button_text}"

    def test_multiple_models_all_show_counts(self):
        """
        All buttons in list should show their respective feature counts.

        When multiple models with different feature counts are displayed,
        each button should show its own count.
        """
        models = [
            {
                'model_id': 'model_1',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],  # 3 features
                'metrics': {'r2': 0.85}
            },
            {
                'model_id': 'model_2',
                'model_type': 'random_forest',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms', 'age', 'location'],  # 5 features
                'metrics': {'r2': 0.92}
            },
            {
                'model_id': 'model_3',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(20)),  # 20 features (using list for brevity)
                'metrics': {'accuracy': 0.90}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Model buttons (exclude back button which is last)
        model_buttons = buttons[:-1]

        # First model: 3 features
        assert "(3 features)" in model_buttons[0][0].text, \
            f"First button should show '(3 features)', got: {model_buttons[0][0].text}"

        # Second model: 5 features
        assert "(5 features)" in model_buttons[1][0].text, \
            f"Second button should show '(5 features)', got: {model_buttons[1][0].text}"

        # Third model: 20 features
        assert "(20 features)" in model_buttons[2][0].text, \
            f"Third button should show '(20 features)', got: {model_buttons[2][0].text}"


class TestButtonTextShowsCustomNames:
    """Test that button text shows custom model names when present."""

    def test_button_shows_custom_name_when_present(self):
        """
        Custom name should take precedence over model type.

        Expected: "1. My Credit Model (20 features)"
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(20)),  # 20 features
                'model_name': 'My Credit Model',  # Custom name
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should show custom name
        assert "My Credit Model" in button_text, \
            f"Button text should show custom name 'My Credit Model', got: {button_text}"

        # Should NOT show model type when custom name is present
        assert "Keras" not in button_text, \
            f"Button text should not show model type when custom name exists, got: {button_text}"

        # Should show feature count
        assert "(20 features)" in button_text, \
            f"Button text should show '(20 features)', got: {button_text}"

    def test_button_shows_model_type_when_no_custom_name(self):
        """
        Model type should be shown when no custom name is provided.

        Expected: "1. Random Forest (5 features)"
        """
        models = [
            {
                'model_id': 'model_12345_random_forest_20251009_120000',
                'model_type': 'random_forest',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms', 'age', 'location'],  # 5 features
                # No 'model_name' key
                'metrics': {'r2': 0.92}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should show model type (title case)
        assert "Random Forest" in button_text or "Random_Forest" in button_text, \
            f"Button text should show model type, got: {button_text}"

        # Should show feature count
        assert "(5 features)" in button_text, \
            f"Button text should show '(5 features)', got: {button_text}"

    def test_button_handles_empty_custom_name(self):
        """
        Empty string custom name should fallback to model type.

        When model_name is empty string or None, should use model_type.
        """
        models = [
            {
                'model_id': 'model_12345_linear_20251009_120000',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms'],  # 2 features
                'model_name': '',  # Empty string
                'metrics': {'r2': 0.85}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should fallback to model type
        assert "Linear" in button_text or "linear" in button_text.lower(), \
            f"Button text should show model type when custom name is empty, got: {button_text}"

        # Should show feature count
        assert "(2 features)" in button_text, \
            f"Button text should show '(2 features)', got: {button_text}"


class TestButtonTextEdgeCases:
    """Test edge cases for button text generation."""

    def test_button_handles_missing_feature_columns(self):
        """
        Graceful fallback when feature_columns is missing.

        Should show model name/type without feature count.
        Expected: "1. Keras Binary Classification"
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                # Missing 'feature_columns' key
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should show model type
        assert "Keras" in button_text or "keras" in button_text.lower(), \
            f"Button text should show model type, got: {button_text}"

        # Should start with number
        assert button_text.startswith("1."), \
            f"Button text should start with '1.', got: {button_text}"

        # Should NOT crash or show "(0 features)"
        # Allow either no feature count or graceful message

    def test_button_handles_empty_feature_columns_list(self):
        """
        Handle empty feature_columns list gracefully.

        Should show model name/type without feature count or with "(0 features)".
        """
        models = [
            {
                'model_id': 'model_12345_linear_20251009_120000',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': [],  # Empty list
                'metrics': {'r2': 0.85}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Should show model type
        assert "Linear" in button_text or "linear" in button_text.lower(), \
            f"Button text should show model type, got: {button_text}"

        # Should start with number
        assert button_text.startswith("1."), \
            f"Button text should start with '1.', got: {button_text}"

        # Should NOT crash

    def test_button_text_reasonable_length(self):
        """
        Button display text should be reasonable length.

        While Telegram has no byte limit for button text (only callback_data),
        very long text is poor UX.
        """
        models = [
            {
                'model_id': 'model_12345',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(100)),  # 100 features
                'model_name': 'This Is A Very Long Custom Model Name That A User Might Create',
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        button_text = buttons[0][0].text

        # Button text should be reasonable (< 100 chars is good UX)
        # This is a soft guideline, not a hard requirement
        assert len(button_text) < 150, \
            f"Button text should be reasonably short, got {len(button_text)} chars: {button_text}"


class TestButtonCallbackDataUnchanged:
    """Test that callback_data format remains unchanged (regression prevention)."""

    def test_callback_data_format_unchanged(self):
        """
        CRITICAL: Callback data format must remain "pred_model_{index}".

        Enhancement should ONLY modify display text, NOT callback_data.
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(20)),
                'model_name': 'My Credit Model',
                'metrics': {'accuracy': 0.925}
            },
            {
                'model_id': 'model_12345_random_forest_20251009_120000',
                'model_type': 'random_forest',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms', 'age', 'location'],
                'metrics': {'r2': 0.92}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Model buttons (exclude back button)
        model_buttons = buttons[:-1]

        # Callback data must use index format
        assert model_buttons[0][0].callback_data == "pred_model_0", \
            f"First button callback_data must be 'pred_model_0', got: {model_buttons[0][0].callback_data}"

        assert model_buttons[1][0].callback_data == "pred_model_1", \
            f"Second button callback_data must be 'pred_model_1', got: {model_buttons[1][0].callback_data}"

    def test_callback_data_stays_under_64_bytes(self):
        """
        Callback data must remain under Telegram's 64-byte limit.

        Even with enhancement, callback_data format should not change.
        """
        models = [
            {
                'model_id': 'model_7715560927_keras_binary_classification_20251009_211219',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(100)),  # Large feature count
                'model_name': 'Very Long Custom Model Name With Many Words',
                'metrics': {'accuracy': 0.925}
            }
        ]

        buttons = create_model_selection_buttons(models)

        callback_data = buttons[0][0].callback_data
        byte_length = len(callback_data.encode('utf-8'))

        # Must stay under 64 bytes (Telegram API constraint)
        assert byte_length <= 64, \
            f"Callback data {byte_length} bytes exceeds 64-byte limit: {callback_data}"


class TestButtonTextConsistencyWithPrompt:
    """Test that button text aligns with model_selection_prompt text."""

    def test_button_numbering_matches_prompt_numbering(self):
        """
        Button numbers should match the numbering in model_selection_prompt.

        Both should use 1-based numbering for user clarity.
        """
        models = [
            {
                'model_id': 'model_1',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms'],
                'metrics': {'r2': 0.85}
            },
            {
                'model_id': 'model_2',
                'model_type': 'random_forest',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
                'metrics': {'r2': 0.92}
            }
        ]

        buttons = create_model_selection_buttons(models)

        # Model buttons (exclude back button)
        model_buttons = buttons[:-1]

        # All buttons should start with "1.", "2.", etc.
        assert model_buttons[0][0].text.startswith("1."), \
            f"First button should start with '1.', got: {model_buttons[0][0].text}"

        assert model_buttons[1][0].text.startswith("2."), \
            f"Second button should start with '2.', got: {model_buttons[1][0].text}"


class TestModelSelectionPromptEnhancement:
    """Test that model_selection_prompt also shows feature counts for consistency."""

    def test_prompt_shows_feature_counts(self):
        """
        Model selection prompt should show feature counts like buttons do.

        This ensures consistency between button text and prompt text.
        """
        from src.bot.messages.prediction_messages import PredictionMessages

        models = [
            {
                'model_id': 'model_1',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms'],  # 2 features
                'metrics': {'r2': 0.85}
            },
            {
                'model_id': 'model_2',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(20)),  # 20 features
                'metrics': {'accuracy': 0.925}
            }
        ]

        prompt = PredictionMessages.model_selection_prompt(models, selected_features=['test'])

        # Should show feature counts
        assert "(2 features)" in prompt, \
            f"Prompt should show '(2 features)' for first model"

        assert "(20 features)" in prompt, \
            f"Prompt should show '(20 features)' for second model"

    def test_prompt_shows_custom_names(self):
        """
        Model selection prompt should show custom names when present.
        """
        from src.bot.messages.prediction_messages import PredictionMessages

        models = [
            {
                'model_id': 'model_1',
                'model_type': 'keras_binary_classification',
                'task_type': 'binary_classification',
                'target_column': 'Churn',
                'feature_columns': list(range(15)),
                'model_name': 'Credit Risk Model',  # Custom name
                'metrics': {'accuracy': 0.925}
            }
        ]

        prompt = PredictionMessages.model_selection_prompt(models, selected_features=['test'])

        # Should show custom name
        assert "Credit Risk Model" in prompt, \
            f"Prompt should show custom model name"

        # Should still show feature count
        assert "(15 features)" in prompt, \
            f"Prompt should show feature count even with custom name"

    def test_prompt_singular_feature_grammar(self):
        """
        Prompt should use correct grammar for single feature.
        """
        from src.bot.messages.prediction_messages import PredictionMessages

        models = [
            {
                'model_id': 'model_1',
                'model_type': 'linear',
                'task_type': 'regression',
                'target_column': 'price',
                'feature_columns': ['sqft'],  # 1 feature
                'metrics': {'r2': 0.85}
            }
        ]

        prompt = PredictionMessages.model_selection_prompt(models, selected_features=['test'])

        # Should use singular "feature"
        assert "(1 feature)" in prompt, \
            f"Prompt should show '(1 feature)' not '(1 features)'"

        # Should NOT have "(1 features)"
        assert "(1 features)" not in prompt, \
            f"Prompt should not show '(1 features)'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
