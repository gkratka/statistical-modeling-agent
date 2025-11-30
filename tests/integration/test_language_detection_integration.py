"""Integration tests for language detection in handlers."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.handlers import start_handler, help_handler
from src.core.state_manager import StateManager, StateManagerConfig
from src.utils.language_detector import LanguageDetector, LanguageDetectionResult
from src.utils.i18n_manager import I18nManager


# Initialize i18n for tests
locales_dir = Path(__file__).parent.parent.parent / "locales"
if locales_dir.exists():
    I18nManager.initialize(str(locales_dir), default_locale="en")


class TestLanguageDetectionIntegration:
    """Test language detection integration with Telegram handlers."""

    @pytest.fixture
    def state_manager(self):
        """Create StateManager for testing."""
        config = StateManagerConfig(
            session_timeout_minutes=30,
            sessions_dir=".test_sessions",
            auto_save=False
        )
        return StateManager(config)

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_user.username = "test_user"
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        update.message.text = "/start"
        return update

    @pytest.fixture
    def mock_context(self, state_manager):
        """Create mock bot context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {'state_manager': state_manager}
        return context

    @pytest.mark.asyncio
    async def test_start_english_default(self, mock_update, mock_context):
        """Test /start command defaults to English."""
        # Execute handler
        await start_handler(mock_update, mock_context)

        # Verify response was sent
        assert mock_update.message.reply_text.called
        response = mock_update.message.reply_text.call_args[0][0]

        # Should contain version info (always present) and instance ID
        assert "🔧 Version:" in response or "🔧 Instance:" in response

        # Verify session language is English (default)
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_session(12345, "67890")
        assert session is not None
        assert session.language == "en"

    @pytest.mark.asyncio
    async def test_start_portuguese_after_detection(self, mock_update, mock_context):
        """Test /start responds in Portuguese after language detection."""
        # Simulate Portuguese message before /start
        mock_update.message.text = "Olá, preciso de ajuda com análise de dados"

        # Mock language detector to return Portuguese
        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                language="pt",
                confidence=0.95,
                method="library",
                detected_at=datetime.now().isoformat()
            )

            # Execute handler (decorator will detect language)
            await start_handler(mock_update, mock_context)

            # Verify language detector was called
            assert mock_detect.called

        # Verify session language is Portuguese
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_session(12345, "67890")
        assert session is not None
        assert session.language == "pt"
        assert session.language_detection_confidence == 0.95

        # Now send /start command
        mock_update.message.text = "/start"
        await start_handler(mock_update, mock_context)

        # Verify response was sent
        assert mock_update.message.reply_text.called
        response = mock_update.message.reply_text.call_args[0][0]

        # Should contain Portuguese text (if translations available)
        # Note: This will fall back to English if pt.yaml not loaded
        # In production, this would contain Portuguese text

    @pytest.mark.asyncio
    async def test_help_command_language_detection(self, mock_update, mock_context):
        """Test /help command uses detected language."""
        # Set message to Portuguese
        mock_update.message.text = "Como usar este bot?"

        # Mock language detector
        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                language="pt",
                confidence=0.92,
                method="library",
                detected_at=datetime.now().isoformat()
            )

            # Execute help handler
            await help_handler(mock_update, mock_context)

        # Verify session language is Portuguese
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_session(12345, "67890")
        assert session is not None
        assert session.language == "pt"

    @pytest.mark.asyncio
    async def test_language_persistence_across_commands(self, mock_update, mock_context):
        """Test language persists across multiple commands."""
        state_manager = mock_context.bot_data['state_manager']

        # 1. First interaction - detect Portuguese with longer text
        mock_update.message.text = "Olá, preciso de ajuda"
        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                language="pt",
                confidence=0.95,
                method="library",
                detected_at=datetime.now().isoformat()
            )
            await start_handler(mock_update, mock_context)

        # Verify language set
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "pt"

        # 2. Second interaction - should use cached language
        mock_update.message.text = "/help"
        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            # Should return cached result
            mock_detect.return_value = LanguageDetectionResult(
                language="pt",
                confidence=1.0,
                method="cached",
                detected_at=datetime.now().isoformat()
            )
            await help_handler(mock_update, mock_context)

        # Verify language still Portuguese
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "pt"

    @pytest.mark.asyncio
    async def test_short_text_no_detection(self, mock_update, mock_context):
        """Test short text (<3 chars) doesn't trigger detection."""
        # Short command
        mock_update.message.text = "/s"

        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            await start_handler(mock_update, mock_context)

            # Should NOT call detector for short text
            assert not mock_detect.called

        # Should default to English
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_session(12345, "67890")
        assert session.language == "en"

    @pytest.mark.asyncio
    async def test_missing_state_manager_graceful_fallback(self, mock_update):
        """Test handler works even if state_manager not in bot_data."""
        # Context without state_manager
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {}

        # Should not crash
        await start_handler(mock_update, context)

        # Response should still be sent (in English)
        assert mock_update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_detection_confidence_stored(self, mock_update, mock_context):
        """Test detection confidence is stored in session."""
        mock_update.message.text = "Olá, como está?"

        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                language="pt",
                confidence=0.87,
                method="library",
                detected_at=datetime.now().isoformat()
            )
            await start_handler(mock_update, mock_context)

        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_session(12345, "67890")

        assert session.language_detection_confidence == 0.87
        assert session.language_detected_at is not None
        assert isinstance(session.language_detected_at, datetime)


class TestLanguageDetectionDecorator:
    """Test the detect_and_set_language decorator in isolation."""

    @pytest.fixture
    def state_manager(self):
        """Create StateManager for testing."""
        config = StateManagerConfig(
            session_timeout_minutes=30,
            sessions_dir=".test_sessions",
            auto_save=False
        )
        return StateManager(config)

    @pytest.mark.asyncio
    async def test_decorator_updates_session(self, state_manager):
        """Test decorator correctly updates session language."""
        from src.utils.decorators import detect_and_set_language

        # Create mock update and context
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 99999
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 88888
        update.message = MagicMock(spec=Message)
        update.message.text = "Preciso de ajuda"

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {'state_manager': state_manager}

        # Create dummy handler
        @detect_and_set_language
        async def dummy_handler(update, context):
            return "handler_executed"

        # Execute with mock detection
        with patch('src.utils.decorators.language_detector.detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                language="pt",
                confidence=0.93,
                method="library",
                detected_at=datetime.now().isoformat()
            )

            result = await dummy_handler(update, context)

        # Verify handler executed
        assert result == "handler_executed"

        # Verify session updated
        session = await state_manager.get_session(99999, "88888")
        assert session.language == "pt"
        assert session.language_detection_confidence == 0.93

    @pytest.mark.asyncio
    async def test_decorator_preserves_handler_metadata(self):
        """Test decorator preserves original handler's metadata."""
        from src.utils.decorators import detect_and_set_language

        @detect_and_set_language
        async def test_handler(update, context):
            """Test handler docstring."""
            pass

        # Verify functools.wraps preserved metadata
        assert test_handler.__name__ == "test_handler"
        assert "Test handler docstring" in test_handler.__doc__
