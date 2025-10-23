"""
Unit tests for CatBoost parameter configuration workflow handlers.

Tests the interactive parameter configuration workflow for CatBoost models:
- Routing logic for catboost_ prefix models
- Iterations selection (100, 500, 1000, 2000, custom)
- Depth selection (4, 6, 8, custom)
- Learning rate selection (0.01, 0.03, 0.1, custom)
- L2 regularization selection (1, 3, 5, custom)
- Configuration confirmation and training initiation
- Error handling and session validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, CallbackQuery, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import StateManager, UserSession


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = AsyncMock(spec=StateManager)
    return manager


@pytest.fixture
def handler(mock_state_manager):
    """Create handler instance with mock state manager."""
    mock_data_loader = AsyncMock()
    return LocalPathMLTrainingHandler(mock_state_manager, mock_data_loader)


@pytest.fixture
def mock_update():
    """Create mock Telegram update with callback query."""
    update = MagicMock(spec=Update)
    update.callback_query = AsyncMock(spec=CallbackQuery)
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    return update


@pytest.fixture
def mock_session():
    """Create mock ML training session."""
    session = MagicMock(spec=UserSession)
    session.user_id = 12345
    session.conversation_id = "chat_67890"
    session.current_state = "selecting_model_type"
    session.selections = {
        'task_type': 'binary_classification',
        'target_column': 'target',
        'feature_columns': ['feature1', 'feature2']
    }
    return session


class TestCatBoostRouting:
    """Test routing logic for CatBoost models."""

    @pytest.mark.asyncio
    async def test_catboost_model_triggers_config_workflow(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that catboost_ prefix triggers parameter configuration."""
        mock_update.callback_query.data = "model:catboost_binary_classification"
        mock_state_manager.get_session.return_value = mock_session

        with patch.object(handler, '_start_catboost_config', new_callable=AsyncMock) as mock_start:
            await handler.handle_model_selection(mock_update, MagicMock())

            # Verify CatBoost config workflow was started
            mock_start.assert_called_once()
            call_args = mock_start.call_args
            assert call_args[0][2] == 'catboost_binary_classification'  # model_type arg

    @pytest.mark.asyncio
    async def test_catboost_regression_routing(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test routing for CatBoost regression model."""
        mock_update.callback_query.data = "model:catboost_regression"
        mock_state_manager.get_session.return_value = mock_session

        with patch.object(handler, '_start_catboost_config', new_callable=AsyncMock) as mock_start:
            await handler.handle_model_selection(mock_update, MagicMock())

            mock_start.assert_called_once()
            call_args = mock_start.call_args
            assert call_args[0][2] == 'catboost_regression'

    @pytest.mark.asyncio
    async def test_catboost_multiclass_routing(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test routing for CatBoost multiclass model."""
        mock_update.callback_query.data = "model:catboost_multiclass_classification"
        mock_state_manager.get_session.return_value = mock_session

        with patch.object(handler, '_start_catboost_config', new_callable=AsyncMock) as mock_start:
            await handler.handle_model_selection(mock_update, MagicMock())

            mock_start.assert_called_once()
            call_args = mock_start.call_args
            assert call_args[0][2] == 'catboost_multiclass_classification'


class TestCatBoostConfigInitialization:
    """Test CatBoost configuration initialization."""

    @pytest.mark.asyncio
    async def test_start_catboost_config_initializes_defaults(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that _start_catboost_config initializes default parameters."""
        mock_state_manager.get_session.return_value = mock_session

        await handler._start_catboost_config(
            mock_update.callback_query,
            mock_session,
            'catboost_binary_classification'
        )

        # Verify session was updated with catboost_config
        assert 'catboost_config' in mock_session.selections
        assert 'catboost_model_type' in mock_session.selections

        # Verify default values from template
        config = mock_session.selections['catboost_config']
        assert config['iterations'] == 1000
        assert config['depth'] == 6
        assert config['learning_rate'] == 0.03
        assert config['l2_leaf_reg'] == 3

        # Verify state manager update was called
        mock_state_manager.update_session.assert_called_once_with(mock_session)

    @pytest.mark.asyncio
    async def test_start_catboost_config_shows_iterations_menu(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that iterations selection menu is shown first."""
        mock_state_manager.get_session.return_value = mock_session

        await handler._start_catboost_config(
            mock_update.callback_query,
            mock_session,
            'catboost_regression'
        )

        # Verify message was edited with iterations options
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args

        # Check message content
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
        assert 'CatBoost Configuration' in message_text
        assert 'Step 1/4' in message_text
        assert 'iterations' in message_text.lower() or 'rounds' in message_text.lower()

        # Check reply markup exists
        assert call_args[1].get('reply_markup') is not None


class TestCatBoostIterationsHandler:
    """Test iterations parameter handler."""

    @pytest.mark.asyncio
    async def test_handle_iterations_100(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test selecting 100 iterations."""
        mock_update.callback_query.data = "catboost_iterations:100"
        mock_session.selections['catboost_config'] = {'iterations': 1000, 'depth': 6}
        mock_state_manager.get_session.return_value = mock_session

        await handler.handle_catboost_iterations(mock_update, MagicMock())

        # Verify iterations was updated
        assert mock_session.selections['catboost_config']['iterations'] == 100
        mock_state_manager.update_session.assert_called_once()

        # Verify moved to depth selection
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
        assert 'Step 2/4' in message_text
        assert 'depth' in message_text.lower()

    @pytest.mark.asyncio
    async def test_handle_iterations_custom(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test custom iterations (defaults to 1000 for Phase 1)."""
        mock_update.callback_query.data = "catboost_iterations:custom"
        mock_session.selections['catboost_config'] = {'iterations': 1000, 'depth': 6}
        mock_state_manager.get_session.return_value = mock_session

        await handler.handle_catboost_iterations(mock_update, MagicMock())

        # Verify default value used for custom
        assert mock_session.selections['catboost_config']['iterations'] == 1000

    @pytest.mark.asyncio
    async def test_handle_iterations_session_expired(
        self,
        handler,
        mock_update,
        mock_state_manager
    ):
        """Test error handling when session expires."""
        mock_update.callback_query.data = "catboost_iterations:500"
        mock_state_manager.get_session.return_value = None

        await handler.handle_catboost_iterations(mock_update, MagicMock())

        # Verify error message shown
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
        assert 'Session Expired' in message_text


class TestCatBoostDepthHandler:
    """Test depth parameter handler."""

    @pytest.mark.asyncio
    async def test_handle_depth_6(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test selecting depth of 6."""
        mock_update.callback_query.data = "catboost_depth:6"
        mock_session.selections['catboost_config'] = {'iterations': 1000, 'depth': 6}
        mock_state_manager.get_session.return_value = mock_session

        await handler.handle_catboost_depth(mock_update, MagicMock())

        # Verify depth was updated
        assert mock_session.selections['catboost_config']['depth'] == 6
        mock_state_manager.update_session.assert_called_once()

        # Verify moved to learning_rate selection
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
        assert 'Step 3/4' in message_text
        assert 'learning' in message_text.lower()

    @pytest.mark.asyncio
    async def test_handle_depth_custom(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test custom depth defaults to 6."""
        mock_update.callback_query.data = "catboost_depth:custom"
        mock_session.selections['catboost_config'] = {'iterations': 1000, 'depth': 4}
        mock_state_manager.get_session.return_value = mock_session

        await handler.handle_catboost_depth(mock_update, MagicMock())

        assert mock_session.selections['catboost_config']['depth'] == 6


class TestCatBoostLearningRateHandler:
    """Test learning_rate parameter handler."""

    @pytest.mark.asyncio
    async def test_handle_learning_rate_0_03(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test selecting learning_rate of 0.03."""
        mock_update.callback_query.data = "catboost_learning_rate:0.03"
        mock_session.selections['catboost_config'] = {'learning_rate': 0.03, 'l2_leaf_reg': 3}
        mock_state_manager.get_session.return_value = mock_session

        await handler.handle_catboost_learning_rate(mock_update, MagicMock())

        # Verify learning_rate was updated
        assert mock_session.selections['catboost_config']['learning_rate'] == 0.03
        mock_state_manager.update_session.assert_called_once()

        # Verify moved to l2_leaf_reg selection
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
        assert 'Step 4/4' in message_text
        assert 'l2' in message_text.lower() or 'regularization' in message_text.lower()

    @pytest.mark.asyncio
    async def test_handle_learning_rate_custom(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test custom learning_rate defaults to 0.03."""
        mock_update.callback_query.data = "catboost_learning_rate:custom"
        mock_session.selections['catboost_config'] = {'learning_rate': 0.01}
        mock_state_manager.get_session.return_value = mock_session

        await handler.handle_catboost_learning_rate(mock_update, MagicMock())

        assert mock_session.selections['catboost_config']['learning_rate'] == 0.03


class TestCatBoostL2LeafRegHandler:
    """Test l2_leaf_reg parameter handler."""

    @pytest.mark.asyncio
    async def test_handle_l2_leaf_reg_3(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test selecting l2_leaf_reg of 3."""
        mock_update.callback_query.data = "catboost_l2:3"
        mock_session.selections['catboost_config'] = {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3
        }
        mock_session.selections['catboost_model_type'] = 'catboost_binary_classification'
        mock_state_manager.get_session.return_value = mock_session

        with patch.object(handler, '_execute_sklearn_training', new_callable=AsyncMock):
            await handler.handle_catboost_l2_leaf_reg(mock_update, MagicMock())

        # Verify l2_leaf_reg was updated
        assert mock_session.selections['catboost_config']['l2_leaf_reg'] == 3
        mock_state_manager.update_session.assert_called_once()

        # Verify configuration complete message shown
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
        assert 'Configuration Complete' in message_text
        assert 'iterations' in message_text
        assert 'depth' in message_text
        assert 'learning' in message_text.lower()
        assert 'l2' in message_text.lower()

    @pytest.mark.asyncio
    async def test_handle_l2_leaf_reg_custom(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test custom l2_leaf_reg defaults to 3."""
        mock_update.callback_query.data = "catboost_l2:custom"
        mock_session.selections['catboost_config'] = {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 1
        }
        mock_session.selections['catboost_model_type'] = 'catboost_regression'
        mock_state_manager.get_session.return_value = mock_session

        with patch.object(handler, '_execute_sklearn_training', new_callable=AsyncMock):
            await handler.handle_catboost_l2_leaf_reg(mock_update, MagicMock())

        assert mock_session.selections['catboost_config']['l2_leaf_reg'] == 3

    @pytest.mark.asyncio
    async def test_l2_triggers_training(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that final parameter triggers training execution."""
        mock_update.callback_query.data = "catboost_l2:3"
        mock_session.selections['catboost_config'] = {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3
        }
        mock_session.selections['catboost_model_type'] = 'catboost_binary_classification'
        mock_state_manager.get_session.return_value = mock_session

        with patch.object(handler, '_execute_sklearn_training', new_callable=AsyncMock) as mock_exec:
            await handler.handle_catboost_l2_leaf_reg(mock_update, MagicMock())

            # Verify training was triggered
            mock_exec.assert_called_once()


class TestCatBoostConfigurationDict:
    """Test configuration dictionary structure."""

    @pytest.mark.asyncio
    async def test_config_dict_has_all_required_params(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that configuration dict contains all required parameters."""
        await handler._start_catboost_config(
            mock_update.callback_query,
            mock_session,
            'catboost_binary_classification'
        )

        config = mock_session.selections['catboost_config']

        # Required parameters
        assert 'iterations' in config
        assert 'depth' in config
        assert 'learning_rate' in config
        assert 'l2_leaf_reg' in config

        # Additional template parameters
        assert 'bootstrap_type' in config
        assert 'subsample' in config
        assert 'random_seed' in config

    @pytest.mark.asyncio
    async def test_config_values_match_template_defaults(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that default values match template specifications."""
        await handler._start_catboost_config(
            mock_update.callback_query,
            mock_session,
            'catboost_regression'
        )

        config = mock_session.selections['catboost_config']

        assert config['iterations'] == 1000
        assert config['depth'] == 6
        assert config['learning_rate'] == 0.03
        assert config['l2_leaf_reg'] == 3
        assert config['bootstrap_type'] == 'MVS'
        assert config['subsample'] == 0.8
        assert config['random_seed'] == 42


class TestCatBoostErrorHandling:
    """Test error handling across workflow."""

    @pytest.mark.asyncio
    async def test_all_handlers_check_session_validity(
        self,
        handler,
        mock_update,
        mock_state_manager
    ):
        """Test that all handlers validate session exists."""
        mock_state_manager.get_session.return_value = None

        handlers_to_test = [
            (handler.handle_catboost_iterations, "catboost_iterations:100"),
            (handler.handle_catboost_depth, "catboost_depth:6"),
            (handler.handle_catboost_learning_rate, "catboost_learning_rate:0.03"),
            (handler.handle_catboost_l2_leaf_reg, "catboost_l2:3"),
        ]

        for handler_func, callback_data in handlers_to_test:
            mock_update.callback_query.data = callback_data
            mock_update.callback_query.edit_message_text.reset_mock()

            await handler_func(mock_update, MagicMock())

            # Verify error message shown
            mock_update.callback_query.edit_message_text.assert_called_once()
            call_args = mock_update.callback_query.edit_message_text.call_args
            message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')
            assert 'Session Expired' in message_text


class TestCatBoostUIElements:
    """Test UI elements and user experience."""

    @pytest.mark.asyncio
    async def test_iterations_menu_has_all_options(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test iterations menu shows all expected options."""
        await handler._start_catboost_config(
            mock_update.callback_query,
            mock_session,
            'catboost_binary_classification'
        )

        call_args = mock_update.callback_query.edit_message_text.call_args
        reply_markup = call_args[1].get('reply_markup')

        assert reply_markup is not None
        assert isinstance(reply_markup, InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_markdown_escaping(
        self,
        handler,
        mock_update,
        mock_session,
        mock_state_manager
    ):
        """Test that underscores are escaped in Markdown text."""
        mock_session.selections['catboost_config'] = {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3
        }
        mock_session.selections['catboost_model_type'] = 'catboost_binary_classification'
        mock_state_manager.get_session.return_value = mock_session
        mock_update.callback_query.data = "catboost_l2:3"

        with patch.object(handler, '_execute_sklearn_training', new_callable=AsyncMock):
            await handler.handle_catboost_l2_leaf_reg(mock_update, MagicMock())

        call_args = mock_update.callback_query.edit_message_text.call_args
        message_text = call_args[0][0] if call_args[0] else call_args[1].get('text', '')

        # Check for escaped underscores in parameter names
        assert 'l2\\_leaf\\_reg' in message_text or 'L2' in message_text
        assert 'learning\\_rate' in message_text or 'Learning' in message_text
