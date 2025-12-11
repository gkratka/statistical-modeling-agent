"""
Common decorators and utilities for the Statistical Modeling Agent.

This module provides reusable decorators and utility functions
to reduce code duplication across handlers and processors.
"""

import functools
import logging
from typing import Callable, Any, Optional, List

from telegram import Update
from telegram.ext import ContextTypes

from src.utils.exceptions import BotError
from src.utils.logger import get_logger
from src.utils.language_detector import LanguageDetector
from src.utils.i18n_manager import I18nManager

logger = get_logger(__name__)

# Initialize language detector (singleton pattern)
language_detector = LanguageDetector()


def validate_telegram_update(func: Callable) -> Callable:
    """
    Decorator to validate Telegram update and user before handler execution.

    Args:
        func: Handler function to wrap

    Returns:
        Wrapped function with validation
    """
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        if not update.effective_user:
            logger.error(f"{func.__name__}: Missing user")
            return
        # Allow both message and callback_query updates (for inline button handlers)
        if not update.message and not update.callback_query:
            logger.error(f"{func.__name__}: Missing message or callback_query")
            return
        return await func(update, context)
    return wrapper


def handle_telegram_errors(func: Callable) -> Callable:
    """
    Decorator to handle common Telegram bot errors with consistent logging.

    Args:
        func: Handler function to wrap

    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        try:
            return await func(update, context)
        except Exception as e:
            # Enhanced diagnostic logging
            user_id = getattr(update.effective_user, 'id', 'unknown') if update.effective_user else 'unknown'
            message_text = update.message.text[:100] if update.message and update.message.text else 'N/A'

            # Try to get session state for context
            session_state = 'N/A'
            try:
                if context.bot_data and 'state_manager' in context.bot_data:
                    state_manager = context.bot_data['state_manager']
                    session = await state_manager.get_session(
                        user_id,
                        f"chat_{update.effective_chat.id}"
                    )
                    if session:
                        session_state = session.current_state
            except Exception:
                pass  # Ignore session lookup errors during error handling

            # Log comprehensive error context
            logger.error(
                f"💥 ERROR in handler '{func.__name__}' | "
                f"User: {user_id} | "
                f"Message: '{message_text}' | "
                f"Session State: {session_state} | "
                f"Error: {e}"
            )

            # Log full stack trace for debugging
            import traceback
            logger.error(f"💥 Stack trace:\n{traceback.format_exc()}")

            # Try to send error message to user
            if update.message:
                try:
                    await update.message.reply_text(
                        "⚠️ **Error Occurred**\n\n"
                        "I encountered an error processing your request. "
                        "Please try again or contact support if the problem persists."
                    )
                except Exception as reply_error:
                    logger.error(f"Failed to send error message: {reply_error}")

            raise BotError(f"Failed in {func.__name__}: {e}", user_id=user_id) from e
    return wrapper


def telegram_handler(func: Callable) -> Callable:
    """
    Combined decorator for Telegram handlers with validation and error handling.

    Args:
        func: Handler function to wrap

    Returns:
        Wrapped function with validation and error handling
    """
    return handle_telegram_errors(validate_telegram_update(func))


def log_user_action(action: str) -> Callable:
    """
    Decorator to log user actions for monitoring and debugging.

    Args:
        action: Description of the action being performed

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
            user_id = getattr(update.effective_user, 'id', 'unknown') if update.effective_user else 'unknown'
            username = getattr(update.effective_user, 'username', 'Unknown') if update.effective_user else 'Unknown'

            logger.info(f"{action}: {username} (ID: {user_id})")

            result = await func(update, context)

            logger.info(f"{action} completed for user {user_id}")
            return result
        return wrapper
    return decorator


def detect_and_set_language(func: Callable) -> Callable:
    """
    Decorator to auto-detect and set user language before handler execution.

    This decorator integrates language detection with the StateManager session.
    It detects language from user message text and updates the session language
    preference for use in I18nManager.t() calls.

    Detection strategy:
    1. First checks if language is already set in session
    2. If message text available (>3 chars), detects language
    3. Updates session with detected language, confidence, and timestamp

    Args:
        func: Handler function to wrap

    Returns:
        Wrapped function with language detection

    Example:
        @telegram_handler
        @detect_and_set_language
        @log_user_action("Bot start")
        async def start_handler(update, context):
            session = await state_manager.get_session(...)
            locale = session.language  # Use detected language
            await update.message.reply_text(I18nManager.t('welcome', locale=locale))
    """
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        from datetime import datetime

        user_id = update.effective_user.id
        state_manager = context.bot_data.get('state_manager')

        if state_manager:
            session = await state_manager.get_or_create_session(
                user_id=user_id,
                conversation_id=str(update.effective_chat.id)
            )

            # Detect language if message text available
            text = update.message.text if update.message else ""
            if text and len(text) > 3:
                # Use cached language if already detected
                cached_language = session.language if session.language != "en" else None

                detection = await language_detector.detect_language(
                    text=text,
                    user_id=user_id,
                    cached_language=cached_language
                )

                # Update session with detection results
                session.language = detection.language
                session.language_detected_at = datetime.now()
                session.language_detection_confidence = detection.confidence

                await state_manager.update_session(session)

                # Debug print (temporary)
                print(f"🌍 LANG DETECTED: {detection.language} (confidence: {detection.confidence:.2f}, method: {detection.method})")

                logger.info(
                    f"Language detected: user={user_id}, "
                    f"lang={detection.language}, "
                    f"confidence={detection.confidence:.2f}, "
                    f"method={detection.method}"
                )

        return await func(update, context)

    return wrapper


# Message constants with i18n support
class Messages:
    """Common message templates used across handlers with i18n support."""

    @staticmethod
    def upload_data_first(locale: Optional[str] = None) -> str:
        """Get the upload data first message in the appropriate language."""
        return I18nManager.t('file_handling.upload_data_first', locale=locale)

    @staticmethod
    def processing_file(locale: Optional[str] = None) -> str:
        """Get the processing file message in the appropriate language."""
        return I18nManager.t('file_handling.processing_file', locale=locale)

    @staticmethod
    def error_occurred(locale: Optional[str] = None) -> str:
        """Get the error occurred message in the appropriate language."""
        return I18nManager.t('file_handling.error_occurred', locale=locale)

    @staticmethod
    def workflow_active(workflow_type: str, current_state: str, locale: Optional[str] = None) -> str:
        """Get the workflow active message in the appropriate language."""
        return I18nManager.t(
            'file_handling.workflow_active',
            locale=locale,
            workflow_type=workflow_type,
            current_state=current_state
        )

    @staticmethod
    def file_already_processed(filename: str, locale: Optional[str] = None) -> str:
        """Get the file already processed message in the appropriate language."""
        return I18nManager.t('file_handling.file_already_processed', locale=locale, filename=filename)

    @staticmethod
    def ready_to_train(locale: Optional[str] = None) -> str:
        """Get the ready to train message in the appropriate language."""
        return I18nManager.t('file_handling.ready_to_train', locale=locale)

    # Keep class attributes for backward compatibility (deprecated)
    UPLOAD_DATA_FIRST = (
        "📁 **Upload Data First**\n\n"
        "To get started, please upload a CSV or Excel file with your data.\n\n"
        "Once you upload a file, I can help you with:\n"
        "📊 Statistical analysis (mean, correlation, etc.)\n"
        "🧠 Machine learning model training\n"
        "📈 Data predictions and insights\n\n"
        "📤 **Upload your CSV/Excel file now!**"
    )

    PROCESSING_FILE = (
        "📁 **Processing your file...**\n\n"
        "Validating and loading data...\n\n"
        "🔧 *DataLoader v2.0 active*"
    )

    ERROR_OCCURRED = (
        "⚠️ **Error Occurred**\n\n"
        "I encountered an error processing your request. "
        "Please try again or contact support if the problem persists.\n\n"
        "🔧 The error has been logged for investigation."
    )


def get_user_data_key(user_id: int) -> str:
    """
    Generate consistent user data key for context storage.

    Args:
        user_id: Telegram user ID

    Returns:
        Formatted user data key
    """
    return f'data_{user_id}'


def safe_get_user_data(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Optional[dict]:
    """
    Safely retrieve user data from context.

    Args:
        context: Telegram bot context
        user_id: User ID to look up

    Returns:
        User data dictionary or None if not found
    """
    if not hasattr(context, 'user_data'):
        return None

    return context.user_data.get(get_user_data_key(user_id))


def format_data_columns(columns: List[str], max_display: int = 10) -> str:
    """
    Format column list for display with truncation.

    Args:
        columns: List of column names
        max_display: Maximum columns to display before truncating

    Returns:
        Formatted column string
    """
    if not columns:
        return "No columns"

    if len(columns) <= max_display:
        return "\n".join(f"• {col}" for col in columns)

    displayed = columns[:max_display]
    remaining = len(columns) - max_display

    return "\n".join(f"• {col}" for col in displayed) + f"\n... and {remaining} more"