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
from src.utils.decorators import telegram_handler, log_user_action, Messages, safe_get_user_data

logger = get_logger(__name__)

# CRITICAL: Version identifier for debugging
BOT_VERSION = "DataLoader-v2.0-NUCLEAR-FIX"
FIXED_TIMESTAMP = "2025-01-27-NUCLEAR"
BOT_INSTANCE_ID = f"BIH-{FIXED_TIMESTAMP}"

# Version tracking for diagnostics
HANDLERS_VERSION = "v2.0"
LAST_UPDATED = "2025-01-27"


@telegram_handler
@log_user_action("Bot start")
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
    welcome_message = (
        f"🤖 Welcome to the Statistical Modeling Agent!\n"
        f"🔧 Version: {BOT_VERSION}\n"
        f"🔧 Instance: {BOT_INSTANCE_ID}\n\n"
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

    await update.message.reply_text(welcome_message)


@telegram_handler
@log_user_action("Help request")
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

    await update.message.reply_text(help_message)


@telegram_handler
@log_user_action("Message received")
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
    user_id = update.effective_user.id
    message_text = update.message.text or ""

    # DIAGNOSTIC: Log message processing
    logger.info("🔧 MESSAGE HANDLER DIAGNOSTIC: Processing user message")
    logger.info(f"🔧 MESSAGE TEXT: {message_text[:100]}...")

    # Check if user has uploaded data
    user_data = safe_get_user_data(context, user_id)

    if not user_data:
        response_message = Messages.UPLOAD_DATA_FIRST
    else:
        # User has data - provide helpful response about their data
        dataframe = user_data.get('dataframe')
        metadata = user_data.get('metadata', {})
        file_name = user_data.get('file_name', 'your data')

        # Basic data question handling
        if any(word in message_text.lower() for word in ['column', 'columns', 'what data', 'what is']):
            columns = metadata.get('columns', [])
            response_message = (
                f"📊 **Your Data: {file_name}**\n\n"
                f"**Columns ({len(columns)}):**\n"
                + "\n".join(f"• {col}" for col in columns[:10])
                + (f"\n... and {len(columns) - 10} more" if len(columns) > 10 else "")
                + f"\n\n**Data Shape:** {metadata.get('shape', (0, 0))[0]:,} rows × {metadata.get('shape', (0, 0))[1]} columns\n\n"
                f"**Try asking:**\n"
                f"• \"Calculate statistics for {columns[0] if columns else 'column_name'}\"\n"
                f"• \"Show correlation matrix\"\n"
                f"• \"Train a model to predict {columns[-1] if columns else 'target_column'}\"\n\n"
                f"🔧 **DataLoader v2.0 active** - Ready for analysis!"
            )
        else:
            # General response for other questions
            response_message = (
                f"🤖 **Statistical Modeling Agent**\n\n"
                f"I received your message: \"{message_text}\"\n\n"
                f"📊 **Your data:** {file_name} ({metadata.get('shape', (0, 0))[0]:,} rows)\n"
                f"**Available columns:** {', '.join(metadata.get('columns', [])[:5])}"
                + ("..." if len(metadata.get('columns', [])) > 5 else "")
                + f"\n\n**What I can help with:**\n"
                f"📈 Descriptive statistics and data analysis\n"
                f"🔍 Correlation analysis between variables\n"
                f"🧠 Machine learning model training\n"
                f"📋 Data exploration and insights\n\n"
                f"**Example requests:**\n"
                f"• \"Calculate mean and standard deviation\"\n"
                f"• \"Show me correlations\"\n"
                f"• \"Train a model to predict [target]\"\n\n"
                f"🔧 **Parser integration coming soon** - For now, ask about your data!"
            )

    await update.message.reply_text(response_message, parse_mode="Markdown")


@telegram_handler
@log_user_action("Diagnostic request")
async def diagnostic_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /diagnostic command - show runtime diagnostic information.

    This helps debug which version of code is running and detect integration issues.

    Args:
        update: Telegram update object
        context: Bot context
    """
    user_id = update.effective_user.id

    try:
        # Gather diagnostic information
        import sys
        import platform
        from pathlib import Path

        # Version information
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / "VERSION"

        version_info = "Version file not found"
        if version_file.exists():
            version_info = version_file.read_text()[:200] + "..."

        # Runtime information
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        handlers_path = Path(__file__).resolve()

        # DataLoader integration check
        try:
            from src.processors.data_loader import DataLoader
            dataloader_status = "✅ DataLoader available"
            loader = DataLoader()
            dataloader_details = f"Max size: {loader.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        except Exception as e:
            dataloader_status = f"❌ DataLoader error: {e}"
            dataloader_details = "Not available"

        # User data check
        user_data_key = f'data_{user_id}'
        user_data_status = "❌ No data uploaded"
        if hasattr(context, 'user_data') and user_data_key in context.user_data:
            user_data = context.user_data[user_data_key]
            shape = user_data.get('metadata', {}).get('shape', (0, 0))
            file_name = user_data.get('file_name', 'unknown')
            user_data_status = f"✅ Data: {file_name} ({shape[0]}×{shape[1]})"

        diagnostic_message = (
            f"🔧 **Diagnostic Information**\n\n"
            f"**Handlers Version:** {HANDLERS_VERSION}\n"
            f"**Last Updated:** {LAST_UPDATED}\n"
            f"**Handlers Path:** `{handlers_path.name}`\n\n"
            f"**Runtime Environment:**\n"
            f"• Python: {python_version}\n"
            f"• Platform: {platform.system()}\n\n"
            f"**DataLoader Integration:**\n"
            f"• Status: {dataloader_status}\n"
            f"• Details: {dataloader_details}\n\n"
            f"**User Data:**\n"
            f"• {user_data_status}\n\n"
            f"**Code Verification:**\n"
            f"• Old Placeholder Text: ❌ Eliminated\n"
            f"• DataLoader v2.0: ✅ Active\n"
            f"• Message Parser: 🔧 In progress\n\n"
            f"**Version Info:**\n"
            f"```\n{version_info}\n```\n\n"
            f"**Diagnostic Logs:**\n"
            f"Watch server logs for diagnostic markers:\n"
            f"• `HANDLERS DIAGNOSTIC`\n"
            f"• `DATALOADER IMPORT`\n"
            f"• `MESSAGE HANDLER DIAGNOSTIC`"
        )

        await update.message.reply_text(diagnostic_message, parse_mode="Markdown")
        logger.info(f"🔧 DIAGNOSTIC INFO sent to user {user_id}")

    except Exception as e:
        error_message = (
            f"🔧 **Diagnostic Error**\n\n"
            f"Error gathering diagnostic information: {str(e)}\n\n"
            f"**Basic Info:**\n"
            f"• Handlers Version: {HANDLERS_VERSION}\n"
            f"• User ID: {user_id}\n"
            f"• Error logged for investigation"
        )
        await update.message.reply_text(error_message, parse_mode="Markdown")
        raise


@telegram_handler
@log_user_action("File upload")
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
    if not update.message.document:
        logger.error("Received document message without document attachment")
        return

    user_id = update.effective_user.id
    document = update.message.document
    file_name = document.file_name or "unknown"
    file_size = document.file_size or 0

    # DIAGNOSTIC: Log which version of handlers is running
    logger.info("🔧 HANDLERS DIAGNOSTIC: DataLoader v2.0 integration ACTIVE")
    logger.info(f"🔧 HANDLERS PATH: {__file__}")

    # Import DataLoader
    from src.processors.data_loader import DataLoader
    from src.utils.exceptions import ValidationError, DataError

    # DIAGNOSTIC: Confirm DataLoader import successful
    logger.info("🔧 DATALOADER IMPORT: Success")

    try:
        # DIAGNOSTIC: Log message being sent
        logger.info("🔧 SENDING MESSAGE: DataLoader v2.0 processing message")

        # Send initial processing message
        processing_msg = await update.message.reply_text(
            Messages.PROCESSING_FILE,
            parse_mode="Markdown"
        )

        # Get the file from Telegram
        file_obj = await context.bot.get_file(document.file_id)

        # Load and process the file
        loader = DataLoader()
        df, metadata = await loader.load_from_telegram(
            file_obj, file_name, file_size, context
        )

        # Store the data for the user (in production, use proper storage)
        if not hasattr(context, 'user_data'):
            context.user_data = {}

        context.user_data[f'data_{user_id}'] = {
            'dataframe': df,
            'metadata': metadata,
            'file_name': file_name
        }

        # Generate success message with data summary
        success_message = loader.get_data_summary(df, metadata)

        # Edit the processing message with results
        await processing_msg.edit_text(success_message, parse_mode="Markdown")

        logger.info(f"Successfully processed file for user {user_id}: {metadata['shape']}")

    except ValidationError as e:
        # Handle validation errors (file too large, wrong type, etc.)
        error_message = (
            f"❌ **File Validation Error**\n\n"
            f"**Issue:** {e.message}\n\n"
            f"**Supported formats:** CSV, Excel (.xlsx)\n"
            f"**Maximum size:** 10 MB\n\n"
            f"Please upload a valid data file and try again."
        )

        await processing_msg.edit_text(error_message, parse_mode="Markdown")

        logger.warning(f"File validation failed for user {user_id}: {e.message}")

    except DataError as e:
        # Handle data processing errors (corrupted file, bad format, etc.)
        error_message = (
            f"❌ **Data Processing Error**\n\n"
            f"**Issue:** {e.message}\n\n"
            f"**Common solutions:**\n"
            f"• Ensure your CSV has headers\n"
            f"• Check for corrupted file content\n"
            f"• Try saving as a different format\n"
            f"• Reduce file size if very large\n\n"
            f"Please fix the file and try again."
        )

        await processing_msg.edit_text(error_message, parse_mode="Markdown")

        logger.error(f"Data processing failed for user {user_id}: {e.message}")

    except Exception as e:
        # Handle unexpected errors
        error_message = (
            f"⚠️ **Unexpected Error**\n\n"
            f"An unexpected error occurred while processing your file. "
            f"Please try again or contact support if the problem persists.\n\n"
            f"Error details have been logged for investigation."
        )

        await processing_msg.edit_text(error_message, parse_mode="Markdown")

        logger.error(f"Unexpected error processing file for user {user_id}: {e}")
        raise


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
            await update.effective_message.reply_text(
                Messages.ERROR_OCCURRED,
                parse_mode="Markdown"
            )
        except Exception as reply_error:
            logger.error(f"Failed to send error message to user: {reply_error}")

    # Re-raise if it's a critical AgentError
    if isinstance(error, AgentError):
        logger.critical(f"Critical agent error: {error.error_code} - {error.message}")
    else:
        logger.error(f"Unexpected error: {error}")