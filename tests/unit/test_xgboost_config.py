"""Unit tests for XGBoost parameter configuration workflow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, CallbackQuery, User, Chat, Message
from telegram.ext import ContextTypes

from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import UserSession
from src.engines.trainers.xgboost_templates import get_template


@pytest.fixture
def mock_update():
    """Create mock Telegram update object."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.callback_query = AsyncMock(spec=CallbackQuery)
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = ""
    return update


@pytest.fixture
def mock_context():
    """Create mock context object."""
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    return context


@pytest.fixture
def mock_session():
    """Create mock session object."""
    session = MagicMock(spec=UserSession)
    session.selections = {}
    session.file_path = "/test/data.csv"
    return session


@pytest.fixture
def handler():
    """Create handler instance with mocked dependencies."""
    # Create mock dependencies
    mock_state_manager = MagicMock()
    mock_ml_engine = MagicMock()
    mock_data_loader = MagicMock()
    mock_template_handler = MagicMock()

    handler = LocalPathMLTrainingHandler(
        state_manager=mock_state_manager,
        ml_engine=mock_ml_engine,
        data_loader=mock_data_loader
    )
    handler.template_handlers = mock_template_handler
    return handler


class TestXGBoostConfigInitialization:
    """Test XGBoost configuration initialization."""

    def test_xgboost_binary_classification_template(self):
        """Test XGBoost binary classification template has correct defaults."""
        config = get_template('xgboost_binary_classification')

        assert config['n_estimators'] == 100
        assert config['max_depth'] == 6
        assert config['learning_rate'] == 0.1
        assert config['subsample'] == 0.8
        assert config['colsample_bytree'] == 0.8
        assert config['objective'] == 'binary:logistic'
        assert config['eval_metric'] == 'logloss'

    def test_xgboost_regression_template(self):
        """Test XGBoost regression template has correct defaults."""
        config = get_template('xgboost_regression')

        assert config['n_estimators'] == 100
        assert config['max_depth'] == 6
        assert config['learning_rate'] == 0.1
        assert config['subsample'] == 0.8
        assert config['colsample_bytree'] == 0.8
        assert config['objective'] == 'reg:squarederror'
        assert 'eval_metric' in config


class TestCustomParameterOverride:
    """Test custom parameters override defaults."""

    def test_custom_n_estimators(self):
        """Test n_estimators can be customized."""
        custom_config = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        assert custom_config['n_estimators'] == 200

    def test_custom_learning_rate(self):
        """Test learning_rate can be customized."""
        custom_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        assert custom_config['learning_rate'] == 0.01

    def test_all_parameters_customizable(self):
        """Test all 5 parameters can be customized."""
        custom_config = {
            'n_estimators': 200,
            'max_depth': 9,
            'learning_rate': 0.01,
            'subsample': 0.6,
            'colsample_bytree': 1.0
        }

        assert custom_config['n_estimators'] == 200
        assert custom_config['max_depth'] == 9
        assert custom_config['learning_rate'] == 0.01
        assert custom_config['subsample'] == 0.6
        assert custom_config['colsample_bytree'] == 1.0


class TestSessionStateManagement:
    """Test session state management for XGBoost config."""

    @pytest.mark.asyncio
    async def test_xgboost_config_stored_in_session(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test XGBoost config is stored in session selections."""
        mock_update.callback_query.data = "xgboost_n_estimators:100"
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_n_estimators(mock_update, mock_context)

        # Verify session.selections was updated
        assert 'xgboost_config' in mock_session.selections
        assert mock_session.selections['xgboost_config']['n_estimators'] == 100
        handler.state_manager.update_session.assert_called_once_with(mock_session)

    @pytest.mark.asyncio
    async def test_xgboost_model_type_stored(
        self, handler, mock_update, mock_context
    ):
        """Test XGBoost model_type is stored in session."""
        query = mock_update.callback_query
        session = MagicMock(spec=Session)
        session.selections = {}

        handler.state_manager.get_session = AsyncMock(return_value=session)
        handler.state_manager.update_session = AsyncMock()

        # Call _start_xgboost_config
        await handler._start_xgboost_config(
            query, session, 'xgboost_binary_classification'
        )

        assert session.selections['xgboost_model_type'] == 'xgboost_binary_classification'
        assert 'xgboost_config' in session.selections


class TestParameterHandlers:
    """Test individual parameter handlers."""

    @pytest.mark.asyncio
    async def test_handle_n_estimators_50(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test n_estimators selection with 50 trees."""
        mock_update.callback_query.data = "xgboost_n_estimators:50"
        mock_session.selections['xgboost_config'] = {}
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_n_estimators(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['n_estimators'] == 50
        # Verify next step message
        mock_update.callback_query.edit_message_text.assert_called_once()
        call_args = mock_update.callback_query.edit_message_text.call_args[0][0]
        assert "Step 2/5: Maximum Tree Depth" in call_args

    @pytest.mark.asyncio
    async def test_handle_max_depth_9(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test max_depth selection with 9 levels."""
        mock_update.callback_query.data = "xgboost_max_depth:9"
        mock_session.selections['xgboost_config'] = {}
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_max_depth(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['max_depth'] == 9
        # Verify next step message
        call_args = mock_update.callback_query.edit_message_text.call_args[0][0]
        assert "Step 3/5: Learning Rate" in call_args

    @pytest.mark.asyncio
    async def test_handle_learning_rate_001(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test learning_rate selection with 0.01 (conservative)."""
        mock_update.callback_query.data = "xgboost_learning_rate:0.01"
        mock_session.selections['xgboost_config'] = {}
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_learning_rate(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['learning_rate'] == 0.01
        # Verify next step message
        call_args = mock_update.callback_query.edit_message_text.call_args[0][0]
        assert "Step 4/5: Subsample Ratio" in call_args

    @pytest.mark.asyncio
    async def test_handle_subsample_06(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test subsample selection with 0.6 (60%)."""
        mock_update.callback_query.data = "xgboost_subsample:0.6"
        mock_session.selections['xgboost_config'] = {}
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_subsample(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['subsample'] == 0.6
        # Verify next step message
        call_args = mock_update.callback_query.edit_message_text.call_args[0][0]
        assert "Step 5/5: Column Subsample Ratio" in call_args

    @pytest.mark.asyncio
    async def test_handle_colsample_10(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test colsample_bytree selection with 1.0 (100%)."""
        mock_update.callback_query.data = "xgboost_colsample:1.0"
        mock_session.selections['xgboost_config'] = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
        mock_session.selections['xgboost_model_type'] = 'xgboost_binary_classification'
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()
        handler._execute_sklearn_training = AsyncMock()

        await handler.handle_xgboost_colsample(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['colsample_bytree'] == 1.0
        # Verify configuration complete message
        call_args = mock_update.callback_query.edit_message_text.call_args[0][0]
        assert "XGBoost Configuration Complete" in call_args
        # Verify training started
        handler._execute_sklearn_training.assert_called_once()


class TestCustomValueDefaults:
    """Test custom value defaults when 'custom' is selected."""

    @pytest.mark.asyncio
    async def test_custom_n_estimators_defaults_to_100(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test custom n_estimators defaults to 100."""
        mock_update.callback_query.data = "xgboost_n_estimators:custom"
        mock_session.selections['xgboost_config'] = {}
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_n_estimators(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['n_estimators'] == 100

    @pytest.mark.asyncio
    async def test_custom_max_depth_defaults_to_6(
        self, handler, mock_update, mock_context, mock_session
    ):
        """Test custom max_depth defaults to 6."""
        mock_update.callback_query.data = "xgboost_max_depth:custom"
        mock_session.selections['xgboost_config'] = {}
        handler.state_manager.get_session = AsyncMock(return_value=mock_session)
        handler.state_manager.update_session = AsyncMock()

        await handler.handle_xgboost_max_depth(mock_update, mock_context)

        assert mock_session.selections['xgboost_config']['max_depth'] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
