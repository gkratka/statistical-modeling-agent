"""
TDD Tests for Prediction Workflow Format Detection.

These tests verify that the prediction workflow correctly handles user format
confusion between training workflow schema format and prediction feature format.

Background:
Users confuse the training workflow's schema format (`features: col1, col2`) with
the prediction workflow's simple format (`col1, col2`). This causes validation
errors and confusion.

Fix: Auto-detect and correct schema format prefixes, provide clear feedback.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.ml_handlers.prediction_handlers import PredictionHandler
from src.core.state_manager import StateManager, MLPredictionState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
import pandas as pd


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = AsyncMock(spec=StateManager)

    # Mock session object with uploaded data
    mock_session = MagicMock()
    mock_session.user_id = 12345
    mock_session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value
    mock_session.workflow_type = WorkflowType.ML_PREDICTION
    mock_session.uploaded_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    mock_session.selections = {}

    manager.get_session.return_value = mock_session
    manager.transition_state = AsyncMock()

    return manager


@pytest.fixture
def mock_data_loader():
    """Create mock data loader."""
    loader = MagicMock(spec=DataLoader)
    loader.allowed_directories = ["/Users/test/data"]
    loader.local_max_size_mb = 1000
    loader.local_extensions = [".csv"]
    return loader


@pytest.fixture
def mock_path_validator():
    """Create mock path validator."""
    return MagicMock(spec=PathValidator)


@pytest.fixture
def prediction_handler(mock_state_manager, mock_data_loader, mock_path_validator):
    """Create prediction handler with mocked dependencies."""
    handler = PredictionHandler(
        state_manager=mock_state_manager,
        data_loader=mock_data_loader,
        path_validator=mock_path_validator
    )
    # Mock ML engine to avoid initialization issues
    handler.ml_engine = MagicMock()
    return handler


@pytest.fixture
def mock_update():
    """Create mock update with message."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.message = AsyncMock(spec=Message)
    update.message.reply_text = AsyncMock()
    update.effective_message = update.message
    return update


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


class TestFeaturesPrefixDetection:
    """Test auto-correction of 'features:' prefix."""

    @pytest.mark.asyncio
    async def test_features_prefix_auto_corrected(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that 'features: col1, col2' is auto-corrected to 'col1, col2'.

        This is the core bug fix - users type training format by mistake.
        """
        # User types with 'features:' prefix
        mock_update.message.text = "features: col1, col2"

        # Handle feature selection (expect ApplicationHandlerStop)
        try:
            await prediction_handler.handle_feature_selection_input(
                mock_update,
                mock_context
            )
        except Exception:
            pass  # ApplicationHandlerStop is expected

        # Verify session selections were updated correctly
        session = await mock_state_manager.get_session(12345, "chat_67890")
        assert 'selected_features' in session.selections
        assert session.selections['selected_features'] == ['col1', 'col2']

        # Verify auto-correction message was sent
        assert mock_update.message.reply_text.call_count >= 2
        calls = [str(call) for call in mock_update.message.reply_text.call_args_list]
        assert any('Format Auto-Corrected' in str(call) or 'auto' in str(call).lower()
                   for call in calls)

    @pytest.mark.asyncio
    async def test_features_prefix_case_insensitive(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """Test that 'Features:' and 'FEATURES:' are also handled."""
        test_cases = [
            "Features: col1, col2",
            "FEATURES: col1, col2",
            "FeAtUrEs: col1, col2"
        ]

        for features_input in test_cases:
            mock_update.message.text = features_input

            try:
                await prediction_handler.handle_feature_selection_input(
                    mock_update,
                    mock_context
                )
            except Exception:
                pass  # ApplicationHandlerStop is expected

            # Verify features were extracted correctly
            session = await mock_state_manager.get_session(12345, "chat_67890")
            assert session.selections['selected_features'] == ['col1', 'col2']

    @pytest.mark.asyncio
    async def test_features_prefix_with_extra_spaces(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """Test that extra spaces around prefix are handled."""
        mock_update.message.text = "features:  col1,  col2  "

        try:
            await prediction_handler.handle_feature_selection_input(
                mock_update,
                mock_context
            )
        except Exception:
            pass  # ApplicationHandlerStop is expected

        session = await mock_state_manager.get_session(12345, "chat_67890")
        assert session.selections['selected_features'] == ['col1', 'col2']


class TestTargetPrefixRejection:
    """Test rejection of 'target:' prefix in prediction workflow."""

    @pytest.mark.asyncio
    async def test_target_prefix_shows_clear_error(
        self,
        prediction_handler,
        mock_update,
        mock_context
    ):
        """
        Test that 'target: price' shows clear error.

        Prediction data should NOT have target column - that's what we're predicting!
        """
        mock_update.message.text = "target: price"

        await prediction_handler.handle_feature_selection_input(
            mock_update,
            mock_context
        )

        # Verify error message was sent
        assert mock_update.message.reply_text.called
        error_message = str(mock_update.message.reply_text.call_args)

        # Should mention that target is not needed in predictions
        assert 'target' in error_message.lower() or 'predict' in error_message.lower()


class TestNormalFormat:
    """Test that normal comma-separated format still works."""

    @pytest.mark.asyncio
    async def test_plain_comma_separated_works(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """Test that normal format 'col1, col2' works without changes."""
        mock_update.message.text = "col1, col2"

        try:
            await prediction_handler.handle_feature_selection_input(
                mock_update,
                mock_context
            )
        except Exception:
            pass  # ApplicationHandlerStop is expected

        session = await mock_state_manager.get_session(12345, "chat_67890")
        assert session.selections['selected_features'] == ['col1', 'col2']

    @pytest.mark.asyncio
    async def test_single_feature_works(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """Test single feature selection."""
        mock_update.message.text = "col1"

        try:
            await prediction_handler.handle_feature_selection_input(
                mock_update,
                mock_context
            )
        except Exception:
            pass  # ApplicationHandlerStop is expected

        session = await mock_state_manager.get_session(12345, "chat_67890")
        assert session.selections['selected_features'] == ['col1']

    @pytest.mark.asyncio
    async def test_many_features_works(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """Test many features (like the user's 20 attributes)."""
        features = ', '.join([f'col{i}' for i in range(1, 21)])
        mock_update.message.text = features

        # Add columns to mock dataframe
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.uploaded_data = pd.DataFrame({f'col{i}': [1] for i in range(1, 21)})

        try:
            await prediction_handler.handle_feature_selection_input(
                mock_update,
                mock_context
            )
        except Exception:
            pass  # ApplicationHandlerStop is expected

        assert len(session.selections['selected_features']) == 20


class TestInvalidFeatureErrors:
    """Test error messages for invalid features."""

    @pytest.mark.asyncio
    async def test_invalid_feature_with_prefix_shows_tip(
        self,
        prediction_handler,
        mock_update,
        mock_context
    ):
        """
        Test that error for 'features: invalid_col' shows helpful tip.

        If user types 'features: col1' but 'col1' doesn't exist, the error
        should detect the prefix and suggest removing it.
        """
        mock_update.message.text = "features: invalid_col"

        await prediction_handler.handle_feature_selection_input(
            mock_update,
            mock_context
        )

        # Verify error was sent
        assert mock_update.message.reply_text.called
        error_calls = [str(call) for call in mock_update.message.reply_text.call_args_list]

        # Should show tip about not using prefix
        has_tip = any('tip' in str(call).lower() or 'prefix' in str(call).lower()
                      for call in error_calls)
        assert has_tip, "Error message should include tip about format"

    @pytest.mark.asyncio
    async def test_invalid_feature_without_prefix_normal_error(
        self,
        prediction_handler,
        mock_update,
        mock_context
    ):
        """Test that normal invalid features get normal error (no tip)."""
        mock_update.message.text = "invalid_col"

        await prediction_handler.handle_feature_selection_input(
            mock_update,
            mock_context
        )

        # Verify error was sent
        assert mock_update.message.reply_text.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
