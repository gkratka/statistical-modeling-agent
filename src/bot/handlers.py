"""
Telegram bot message handlers for the Statistical Modeling Agent.

This module implements message routing and basic command handling
as specified in the CLAUDE.md guidelines. All business logic is kept
separate from bot interaction logic.
"""

import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.exceptions import BotError, AgentError
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def start_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /start command - welcome new users.

    Args:
        update: Telegram update object
        context: Bot context
    """
    if not update.effective_user or not update.message:
        logger.error("Received start command with missing user or message")
        return

    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"

    logger.info(f"New user started bot: {username} (ID: {user_id})")

    welcome_message = (
        "🤖 Welcome to the Statistical Modeling Agent!\n\n"
        "I can help you with:\n"
        "📊 Statistical analysis of your data\n"
        "🧠 Machine learning model training\n"
        "📈 Data predictions and insights\n\n"
        "To get started:\n"
        "1. Upload a CSV file with your data\n"
        "2. Tell me what analysis you'd like\n"
        "3. I'll process it and send you results!\n\n"
        "Type /help for more information."
    )

    try:
        await update.message.reply_text(welcome_message)
    except Exception as e:
        logger.error(f"Failed to send welcome message to user {user_id}: {e}")
        raise BotError(f"Failed to send message: {e}", user_id=user_id)


async def help_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /help command - show usage instructions.

    Args:
        update: Telegram update object
        context: Bot context
    """
    if not update.effective_user or not update.message:
        logger.error("Received help command with missing user or message")
        return

    user_id = update.effective_user.id
    logger.info(f"User requested help: {user_id}")

    help_message = (
        "🆘 Statistical Modeling Agent Help\n\n"
        "Commands:\n"
        "/start - Start using the bot\n"
        "/help - Show this help message\n\n"
        "How to use:\n"
        "1. Upload Data: Send a CSV file\n"
        "2. Request Analysis: Tell me what you want:\n"
        "   • Calculate mean and std for age column\n"
        "   • Show correlation matrix\n"
        "   • Train a model to predict income\n"
        "3. Get Results: I'll analyze and respond\n\n"
        "Supported Operations:\n"
        "📊 Descriptive statistics\n"
        "📈 Correlation analysis\n"
        "🧠 Machine learning training\n"
        "🔮 Model predictions\n\n"
        "Example:\n"
        "1. Upload: housing_data.csv\n"
        "2. Message: Train a model to predict house prices\n"
        "3. Get: Model training results and performance metrics\n\n"
        "Need more help? Just ask me anything!"
    )

    try:
        await update.message.reply_text(help_message)
    except Exception as e:
        logger.error(f"Failed to send help message to user {user_id}: {e}")
        raise BotError(f"Failed to send help message: {e}", user_id=user_id)


async def message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle regular text messages from users.

    For now, this provides a confirmation response. In the full implementation,
    this will route to the parser and orchestrator for processing.

    Args:
        update: Telegram update object
        context: Bot context
    """
    if not update.effective_user or not update.message:
        logger.error("Received message with missing user or message")
        return

    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    message_text = update.message.text or ""

    logger.info(
        f"Received message from {username} (ID: {user_id}): "
        f"{message_text[:100]}..." if len(message_text) > 100 else message_text
    )

    # For now, send a confirmation response
    # TODO: In full implementation, this will:
    # 1. Parse the message using parser.py
    # 2. Route to orchestrator.py for task processing
    # 3. Return formatted results
    confirmation_message = (
        f"✅ **Message Received**\n\n"
        f"I got your message: \"{message_text}\"\n\n"
        f"🚧 **Development Mode**\n"
        f"I'm currently under development. Soon I'll be able to:\n"
        f"• Parse your request\n"
        f"• Analyze your data\n"
        f"• Train ML models\n"
        f"• Provide insights\n\n"
        f"For now, I'm confirming that I can receive and respond to messages!\n"
        f"User ID: {user_id}"
    )

    try:
        await update.message.reply_text(confirmation_message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to send confirmation to user {user_id}: {e}")
        raise BotError(f"Failed to send confirmation: {e}", user_id=user_id)


async def document_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle document uploads (CSV files, etc.).

    Args:
        update: Telegram update object
        context: Bot context
    """
    if not update.effective_user or not update.message or not update.message.document:
        logger.error("Received document with missing user, message, or document")
        return

    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    document = update.message.document
    file_name = document.file_name or "unknown"
    file_size = document.file_size or 0

    logger.info(
        f"Received document from {username} (ID: {user_id}): "
        f"{file_name} ({file_size} bytes)"
    )

    # TODO: In full implementation, this will:
    # 1. Validate file type and size
    # 2. Download and process the file
    # 3. Store data for analysis
    # 4. Confirm successful upload

    confirmation_message = (
        f"📁 **File Received**\n\n"
        f"**File:** {file_name}\n"
        f"**Size:** {file_size:,} bytes\n"
        f"**Type:** {document.mime_type or 'unknown'}\n\n"
        f"🚧 **Development Mode**\n"
        f"File upload handling is under development. Soon I'll:\n"
        f"• Validate file format\n"
        f"• Process your data\n"
        f"• Prepare it for analysis\n\n"
        f"For now, I'm confirming I can receive files!"
    )

    try:
        await update.message.reply_text(confirmation_message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to send file confirmation to user {user_id}: {e}")
        raise BotError(f"Failed to send file confirmation: {e}", user_id=user_id)


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle errors in bot operation.

    Args:
        update: Telegram update object (might be None)
        context: Bot context containing error information
    """
    error = context.error
    logger.error(f"Bot error occurred: {error}", exc_info=error)

    # If we have an update with a message, try to inform the user
    if isinstance(update, Update) and update.effective_message:
        try:
            error_message = (
                "⚠️ **Error Occurred**\n\n"
                "I encountered an error processing your request. "
                "Please try again or contact support if the problem persists.\n\n"
                "🔧 The error has been logged for investigation."
            )
            await update.effective_message.reply_text(
                error_message,
                parse_mode="Markdown"
            )
        except Exception as reply_error:
            logger.error(f"Failed to send error message to user: {reply_error}")

    # Re-raise if it's a critical AgentError
    if isinstance(error, AgentError):
        logger.critical(f"Critical agent error: {error.error_code} - {error.message}")
    else:
        logger.error(f"Unexpected error: {error}")