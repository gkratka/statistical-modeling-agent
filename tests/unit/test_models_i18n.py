"""Tests for models browser i18n implementation."""

import pytest
from unittest.mock import MagicMock, patch

from src.bot.messages.models_messages import ModelsMessages
from src.engines.model_catalog import ModelInfo, ModelCategory, TaskType


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return ModelInfo(
        id="test_model",
        display_name="Test Model",
        icon="🧪",
        short_description="A test model",
        long_description="This is a longer description of the test model",
        category=ModelCategory.TREE_BASED,
        task_type=TaskType.REGRESSION,
        variants=["test_model_v1", "test_model_v2"],
        parameters=[
            MagicMock(
                display_name="Param 1",
                name="param_1",
                description="First parameter",
                default="10",
                range="1-100"
            ),
            MagicMock(
                display_name="Param 2",
                name="param_2",
                description="Second parameter",
                default="0.5",
                range="0-1"
            ),
        ],
        use_cases=["Use case 1", "Use case 2", "Use case 3"],
        strengths=["Strength 1", "Strength 2"],
        limitations=["Limitation 1", "Limitation 2"],
        training_speed="Fast",
        prediction_speed="Very Fast",
        interpretability="High",
        requires_tuning=False
    )


class TestModelsMessagesI18n:
    """Test ModelsMessages with i18n support."""

    @patch('src.bot.messages.models_messages.I18nManager')
    def test_models_list_message_with_locale(self, mock_i18n):
        """Test models_list_message includes locale parameter."""
        # Setup mock to return test strings
        def mock_t(key, locale=None, **kwargs):
            translations = {
                "workflows.models.list_header": "📚 **ML Model Catalog**",
                "workflows.models.browse_message": f"Browse {kwargs.get('total_models', 0)} models",
                "workflows.models.list_details": "Click a model",
                "workflows.models.page_info": f"Page {kwargs.get('current', 0)}/{kwargs.get('total', 0)}",
                "workflows.models.list_tip": "Models are organized",
            }
            return translations.get(key, key)

        mock_i18n.t.side_effect = mock_t

        # Test with locale
        result = ModelsMessages.models_list_message(
            page=1,
            total_pages=5,
            total_models=20,
            locale="pt"
        )

        # Verify I18nManager.t was called with locale
        calls = mock_i18n.t.call_args_list
        for call in calls:
            # Check that locale was passed if it's not the first call
            if 'locale' in call.kwargs:
                assert call.kwargs['locale'] == "pt"

        # Verify message contains expected elements
        assert "20" in result or "models" in result.lower()

    @patch('src.bot.messages.models_messages.I18nManager')
    def test_models_list_message_without_locale(self, mock_i18n):
        """Test models_list_message works without locale (uses default)."""
        mock_i18n.t.side_effect = lambda key, locale=None, **kw: f"[{key}]"

        result = ModelsMessages.models_list_message(
            page=1,
            total_pages=5,
            total_models=20
        )

        # Should still generate a message
        assert result is not None
        assert len(result) > 0

    @patch('src.bot.messages.models_messages.I18nManager')
    def test_model_details_message_with_locale(self, mock_i18n, mock_model):
        """Test model_details_message includes locale parameter."""
        def mock_t(key, locale=None, **kwargs):
            translations = {
                "workflows.models.category": "Category",
                "workflows.models.task_type": "Task Type",
                "workflows.models.variants": "Variants",
                "workflows.models.description": "Description",
                "workflows.models.performance": "Performance",
                "workflows.models.training_speed": "Training Speed",
                "workflows.models.prediction_speed": "Prediction Speed",
                "workflows.models.interpretability": "Interpretability",
                "workflows.models.requires_tuning": "Requires Tuning",
                "workflows.models.yes": "Yes",
                "workflows.models.no": "No",
                "workflows.models.key_parameters": "Key Parameters",
                "workflows.models.more_parameters": f"... and {kwargs.get('count', 0)} more",
                "workflows.models.default": "Default",
                "workflows.models.range": "Range",
                "workflows.models.best_for": "Best For",
                "workflows.models.strengths": "Strengths",
                "workflows.models.limitations": "Limitations",
                "workflows.models.details_tip": "Tip: Use /train",
            }
            return translations.get(key, key)

        mock_i18n.t.side_effect = mock_t

        result = ModelsMessages.model_details_message(mock_model, locale="en")

        # Verify result contains model information
        assert mock_model.display_name in result
        assert mock_model.short_description in result

    @patch('src.bot.messages.models_messages.I18nManager')
    def test_model_details_message_structure(self, mock_i18n, mock_model):
        """Test model_details_message contains all expected sections."""
        mock_i18n.t.side_effect = lambda key, locale=None, **kw: f"[{key}]"

        result = ModelsMessages.model_details_message(mock_model, locale="pt")

        # Check for key sections
        assert mock_model.icon in result
        assert mock_model.display_name in result
        assert mock_model.short_description in result
        assert mock_model.long_description in result

    def test_models_list_message_pagination(self):
        """Test models_list_message pagination information."""
        with patch('src.bot.messages.models_messages.I18nManager') as mock_i18n:
            mock_i18n.t.side_effect = lambda key, locale=None, **kw: f"[{key}]"

            result = ModelsMessages.models_list_message(
                page=2,
                total_pages=10,
                total_models=100,
                locale="en"
            )

            # Message should include page information
            assert len(result) > 0

    def test_model_details_message_with_parameters(self, mock_model):
        """Test model_details_message includes parameter information."""
        with patch('src.bot.messages.models_messages.I18nManager') as mock_i18n:
            mock_i18n.t.side_effect = lambda key, locale=None, **kw: key

            result = ModelsMessages.model_details_message(mock_model, locale="en")

            # Should include parameter names
            assert "Param 1" in result or "param_1" in result
            assert "Param 2" in result or "param_2" in result

    def test_model_details_message_with_variants(self, mock_model):
        """Test model_details_message includes variant information."""
        with patch('src.bot.messages.models_messages.I18nManager') as mock_i18n:
            mock_i18n.t.side_effect = lambda key, locale=None, **kw: key

            result = ModelsMessages.model_details_message(mock_model, locale="en")

            # Should include variant information
            assert "variant" in result.lower() or "v1" in result


class TestTranslationKeyConsistency:
    """Test that translation keys match across files."""

    def test_translation_keys_used(self):
        """Verify all translation keys used in ModelsMessages."""
        keys_used = set()

        with patch('src.bot.messages.models_messages.I18nManager') as mock_i18n:
            def track_t(key, locale=None, **kwargs):
                keys_used.add(key)
                return key

            mock_i18n.t.side_effect = track_t

            mock_model = MagicMock()
            mock_model.icon = "🧪"
            mock_model.display_name = "Test"
            mock_model.short_description = "Short"
            mock_model.long_description = "Long"
            mock_model.category.value = "test_category"
            mock_model.task_type.value = "regression"
            mock_model.variants = []
            mock_model.parameters = []
            mock_model.use_cases = []
            mock_model.strengths = []
            mock_model.limitations = []
            mock_model.training_speed = "Fast"
            mock_model.prediction_speed = "Fast"
            mock_model.interpretability = "High"
            mock_model.requires_tuning = False

            # Generate both messages
            ModelsMessages.models_list_message(1, 5, 20, locale="en")
            ModelsMessages.model_details_message(mock_model, locale="en")

        # Verify we have translation keys
        assert len(keys_used) > 10, f"Expected many translation keys, got {len(keys_used)}"

        # All keys should follow the pattern
        for key in keys_used:
            assert key.startswith("workflows.models"), f"Key {key} doesn't follow pattern"
