"""
Common decorators and utilities for the Statistical Modeling Agent.

This module provides reusable decorators and utility functions
to reduce code duplication across handlers and processors.
"""

import functools
import logging
from typing import Callable, Any, Optional

from telegram import Update
from telegram.ext import ContextTypes

from src.utils.exceptions import BotError
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        if not update.effective_user or not update.message:
            logger.error(f"{func.__name__}: Missing user or message")
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
            user_id = getattr(update.effective_user, 'id', 'unknown') if update.effective_user else 'unknown'
            logger.error(f"Error in {func.__name__} for user {user_id}: {e}")

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


# Message constants to reduce duplication
class Messages:
    """Common message templates used across handlers."""

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


def format_data_columns(columns: list[str], max_display: int = 10) -> str:
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