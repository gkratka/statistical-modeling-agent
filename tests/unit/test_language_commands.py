"""Tests for language switch commands (/pt and /en)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, User, Chat, Message
from telegram.ext import ContextTypes

from src.bot.handlers import pt_handler, en_handler
from src.core.state_manager import StateManager, UserSession
from src.utils.i18n_manager import I18nManager


# Initialize I18nManager for tests
I18nManager.initialize(locales_dir="./locales", default_locale="en")


class TestLanguageCommands:
    """Test /pt and /en language switching commands."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update object."""
        update = AsyncMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_user.username = "testuser"
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.message = AsyncMock(spec=Message)
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock context with StateManager."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        state_manager = StateManager()
        context.bot_data = {"state_manager": state_manager}
        return context

    @pytest.mark.asyncio
    async def test_pt_command_sets_portuguese(self, mock_update, mock_context):
        """Test /pt command sets session language to Portuguese."""
        await pt_handler(mock_update, mock_context)

        # Verify session language set to Portuguese
        state_manager = mock_context.bot_data["state_manager"]
        session = await state_manager.get_session(
            user_id=12345,
            conversation_id="67890"
        )
        assert session is not None
        assert session.language == "pt"

    @pytest.mark.asyncio
    async def test_pt_command_shows_portuguese_welcome(self, mock_update, mock_context):
        """Test /pt command shows Portuguese welcome message."""
        await pt_handler(mock_update, mock_context)

        # Verify Portuguese welcome was sent
        mock_update.message.reply_text.assert_called_once()
        response = mock_update.message.reply_text.call_args[0][0]
        assert "Bem-vindo" in response
        assert "Agente de Modelagem Estatística" in response

    @pytest.mark.asyncio
    async def test_en_command_sets_english(self, mock_update, mock_context):
        """Test /en command sets session language to English."""
        # First set to Portuguese
        state_manager = mock_context.bot_data["state_manager"]
        await state_manager.get_or_create_session(12345, "67890")
        session = await state_manager.get_session(12345, "67890")
        session.language = "pt"
        await state_manager.update_session(session)

        # Now switch to English
        await en_handler(mock_update, mock_context)

        # Verify session language changed to English
        session = await state_manager.get_session(12345, "67890")
        assert session is not None
        assert session.language == "en"

    @pytest.mark.asyncio
    async def test_en_command_shows_english_welcome(self, mock_update, mock_context):
        """Test /en command shows English welcome message."""
        await en_handler(mock_update, mock_context)

        # Verify English welcome was sent
        mock_update.message.reply_text.assert_called_once()
        response = mock_update.message.reply_text.call_args[0][0]
        assert "Welcome" in response
        assert "Statistical Modeling Agent" in response

    @pytest.mark.asyncio
    async def test_language_persists_across_session(self, mock_update, mock_context):
        """Test language setting persists in session."""
        # Set to Portuguese
        await pt_handler(mock_update, mock_context)

        state_manager = mock_context.bot_data["state_manager"]
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "pt"

        # Create new context (simulating new message)
        session_check = await state_manager.get_session(12345, "67890")
        assert session_check.language == "pt"

    @pytest.mark.asyncio
    async def test_pt_command_without_state_manager(self, mock_update):
        """Test /pt command gracefully handles missing StateManager."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {}  # No state_manager

        # Should not crash
        await pt_handler(mock_update, context)

        # Should still send response
        mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_en_command_without_state_manager(self, mock_update):
        """Test /en command gracefully handles missing StateManager."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {}  # No state_manager

        # Should not crash
        await en_handler(mock_update, context)

        # Should still send response
        mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_switching_between_languages(self, mock_update, mock_context):
        """Test switching back and forth between languages."""
        # Start with Portuguese
        await pt_handler(mock_update, mock_context)
        state_manager = mock_context.bot_data["state_manager"]
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "pt"

        # Switch to English
        mock_update.message.reply_text.reset_mock()
        await en_handler(mock_update, mock_context)
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "en"

        # Switch back to Portuguese
        mock_update.message.reply_text.reset_mock()
        await pt_handler(mock_update, mock_context)
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "pt"
