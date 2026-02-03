"""Tests for model selection pagination functionality."""

import pytest
from src.bot.messages.prediction_messages import (
    create_model_selection_buttons,
    PredictionMessages,
    MODELS_PER_PAGE
)
from src.utils.i18n_manager import I18nManager


@pytest.fixture(scope="module", autouse=True)
def initialize_i18n():
    """Initialize i18n for all tests in this module."""
    I18nManager.initialize("./locales", "en")


class TestPaginationButtonCreation:
    """Test button creation with pagination."""

    def test_no_pagination_with_exactly_10_models(self):
        """With exactly MODELS_PER_PAGE models, no pagination buttons should appear."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(10)
        ]

        buttons = create_model_selection_buttons(models, page=0)

        # Should have: 10 models + back + delete = 12 rows (no pagination)
        assert len(buttons) == 12, f"Expected 12 rows, got {len(buttons)}"

        # Verify no pagination buttons
        for row in buttons:
            for btn in row:
                assert not btn.callback_data.startswith('pred_page_'), \
                    f"Should have no pagination buttons with exactly 10 models"

    def test_pagination_with_11_models(self):
        """With 11 models, pagination buttons should appear."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(11)
        ]

        # Page 0 (first page)
        buttons = create_model_selection_buttons(models, page=0)

        # Should have: 10 models + nav row (Next) + back + delete = 13 rows
        assert len(buttons) == 13, f"Expected 13 rows on page 0, got {len(buttons)}"

        # Check for Next button (no Prev on first page)
        assert buttons[10][0].callback_data == "pred_page_1", \
            f"Expected Next button on page 0, got {buttons[10][0].callback_data}"

    def test_pagination_middle_page(self):
        """Middle page should have both Prev and Next buttons."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(25)  # 3 pages
        ]

        # Page 1 (middle page)
        buttons = create_model_selection_buttons(models, page=1)

        # Should have: 10 models + nav row (Prev + Next) + back + delete = 13 rows
        assert len(buttons) == 13, f"Expected 13 rows on page 1, got {len(buttons)}"

        # Check for both Prev and Next buttons
        nav_row = buttons[10]
        assert len(nav_row) == 2, "Middle page should have Prev and Next buttons"
        assert nav_row[0].callback_data == "pred_page_0", "Should have Prev button"
        assert nav_row[1].callback_data == "pred_page_2", "Should have Next button"

    def test_pagination_last_page(self):
        """Last page should only have Prev button."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(15)  # 2 pages
        ]

        # Page 1 (last page, 5 models)
        buttons = create_model_selection_buttons(models, page=1, total_models=15)

        # Should have: 5 models + nav row (Prev only) + back + delete = 8 rows
        assert len(buttons) == 8, f"Expected 8 rows on last page, got {len(buttons)}"

        # Check for Prev button only (no Next on last page)
        nav_row = buttons[5]
        assert len(nav_row) == 1, "Last page should have only Prev button"
        assert nav_row[0].callback_data == "pred_page_0", \
            f"Should have Prev button on last page"

    def test_model_numbering_across_pages(self):
        """Model numbering should be continuous across pages."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(25)
        ]

        # Page 0: should show models 1-10
        buttons_p0 = create_model_selection_buttons(models, page=0)
        assert buttons_p0[0][0].text.startswith("1."), "First model on page 0 should be #1"
        assert buttons_p0[9][0].text.startswith("10."), "Last model on page 0 should be #10"

        # Page 1: should show models 11-20
        buttons_p1 = create_model_selection_buttons(models, page=1)
        assert buttons_p1[0][0].text.startswith("11."), "First model on page 1 should be #11"
        assert buttons_p1[9][0].text.startswith("20."), "Last model on page 1 should be #20"

        # Page 2: should show models 21-25
        buttons_p2 = create_model_selection_buttons(models, page=2, total_models=25)
        assert buttons_p2[0][0].text.startswith("21."), "First model on page 2 should be #21"
        assert buttons_p2[4][0].text.startswith("25."), "Last model on page 2 should be #25"


class TestPaginationPromptText:
    """Test prompt text includes pagination information."""

    def test_prompt_shows_page_info(self):
        """Prompt should show page number when multiple pages exist."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(25)
        ]

        # Page 0
        prompt_p0 = PredictionMessages.model_selection_prompt(
            models, ['a', 'b'], page=0, total_models=25
        )
        assert "Page 1 of 3" in prompt_p0, "Should show page 1 of 3"

        # Page 1
        prompt_p1 = PredictionMessages.model_selection_prompt(
            models, ['a', 'b'], page=1, total_models=25
        )
        assert "Page 2 of 3" in prompt_p1, "Should show page 2 of 3"

    def test_prompt_no_page_info_single_page(self):
        """Prompt should not show page info with single page."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(5)
        ]

        prompt = PredictionMessages.model_selection_prompt(
            models, ['a', 'b'], page=0, total_models=5
        )
        assert "Page" not in prompt, "Should not show page info with single page"


class TestPaginationCallbackData:
    """Test pagination callback data format."""

    def test_pagination_callback_format(self):
        """Pagination buttons should use pred_page_{number} format."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(15)
        ]

        buttons = create_model_selection_buttons(models, page=0)

        # Find Next button
        next_button = buttons[10][0]
        assert next_button.callback_data == "pred_page_1", \
            f"Next button should have callback 'pred_page_1', got '{next_button.callback_data}'"

    def test_model_callback_remains_page_relative(self):
        """Model button callbacks should be page-relative indices."""
        models = [
            {
                'model_id': f'model_{i}',
                'model_type': 'linear',
                'feature_columns': ['a', 'b'],
                'task_type': 'regression',
                'target_column': 'y',
                'metrics': {'r2': 0.8}
            }
            for i in range(25)
        ]

        # Page 0: callbacks should be 0-9
        buttons_p0 = create_model_selection_buttons(models, page=0)
        assert buttons_p0[0][0].callback_data == "pred_model_0"
        assert buttons_p0[9][0].callback_data == "pred_model_9"

        # Page 1: callbacks should ALSO be 0-9 (page-relative)
        buttons_p1 = create_model_selection_buttons(models, page=1)
        assert buttons_p1[0][0].callback_data == "pred_model_0", \
            "Page 1 first model should have callback pred_model_0 (page-relative)"
        assert buttons_p1[9][0].callback_data == "pred_model_9", \
            "Page 1 last model should have callback pred_model_9 (page-relative)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
