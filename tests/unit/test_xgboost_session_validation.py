"""
Test XGBoost session validation in parameter configuration handlers.

Tests that all 5 XGBoost parameter handlers properly handle session expiration
with user-friendly error messages instead of crashing with AttributeError.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import StateManager


class TestXGBoostSessionValidation:
    """Test session expiration handling during XGBoost parameter configuration."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update object."""
        update = AsyncMock()
        update.callback_query = AsyncMock()
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        update.callback_query.data = "test:0.5"
        update.effective_user.id = 12345
        update.effective_chat.id = 67890
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock context object."""
        return AsyncMock()

    @pytest.fixture
    def handler(self):
        """Create LocalPathMLTrainingHandler instance."""
        state_manager = MagicMock(spec=StateManager)
        data_loader = MagicMock()
        return LocalPathMLTrainingHandler(state_manager, data_loader)

    @pytest.mark.asyncio
    async def test_n_estimators_session_expired(self, handler, mock_update, mock_context):
        """Test handle_xgboost_n_estimators with expired session."""
        # Setup: session returns None (expired)
        handler.state_manager.get_session = AsyncMock(return_value=None)

        # Execute
        await handler.handle_xgboost_n_estimators(mock_update, mock_context)

        # Verify: Shows session expired message
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        assert "Session Expired" in call_args[0][0]
        assert "/train" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_max_depth_session_expired(self, handler, mock_update, mock_context):
        """Test handle_xgboost_max_depth with expired session."""
        handler.state_manager.get_session = AsyncMock(return_value=None)

        await handler.handle_xgboost_max_depth(mock_update, mock_context)

        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        assert "Session Expired" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_learning_rate_session_expired(self, handler, mock_update, mock_context):
        """Test handle_xgboost_learning_rate with expired session."""
        handler.state_manager.get_session = AsyncMock(return_value=None)

        await handler.handle_xgboost_learning_rate(mock_update, mock_context)

        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        assert "Session Expired" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_subsample_session_expired(self, handler, mock_update, mock_context):
        """Test handle_xgboost_subsample with expired session (original failing handler)."""
        handler.state_manager.get_session = AsyncMock(return_value=None)

        await handler.handle_xgboost_subsample(mock_update, mock_context)

        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        assert "Session Expired" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_colsample_session_expired(self, handler, mock_update, mock_context):
        """Test handle_xgboost_colsample with expired session."""
        handler.state_manager.get_session = AsyncMock(return_value=None)

        await handler.handle_xgboost_colsample(mock_update, mock_context)

        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        assert "Session Expired" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_no_attribute_error_on_none_session(self, handler, mock_update, mock_context):
        """Test that None session doesn't cause AttributeError (regression test)."""
        handler.state_manager.get_session = AsyncMock(return_value=None)

        # Should NOT raise AttributeError: 'NoneType' object has no attribute 'selections'
        try:
            await handler.handle_xgboost_subsample(mock_update, mock_context)
            # If we get here, no exception was raised - test passes
        except AttributeError as e:
            if "'NoneType' object has no attribute 'selections'" in str(e):
                pytest.fail(f"Session validation failed: {e}")
            raise
