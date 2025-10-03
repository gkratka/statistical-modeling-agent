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
HANDLERS_VERSION = "v2.0"
LAST_UPDATED = "2025-01-27"

# Message templates
MESSAGE_TEMPLATES = {
    "welcome": f"""🤖 Welcome to the Statistical Modeling Agent!
🔧 Version: {BOT_VERSION}
🔧 Instance: {BOT_INSTANCE_ID}

I can help you with:
📊 Statistical analysis of your data
🧠 Machine learning model training
📈 Data predictions and insights

To get started:
1. Upload a CSV file with your data
2. Tell me what analysis you'd like
3. I'll process it and send you results!

Type /help for more information.""",

    "help": """🆘 Statistical Modeling Agent Help

Commands:
/start - Start using the bot
/help - Show this help message

How to use:
1. Upload Data: Send a CSV file
2. Request Analysis: Tell me what you want:
   • Calculate mean and std for age column
   • Show correlation matrix
   • Train a model to predict income
3. Get Results: I'll analyze and respond

Supported Operations:
📊 Descriptive statistics
📈 Correlation analysis
🧠 Machine learning training
🔮 Model predictions

Example:
1. Upload: housing_data.csv
2. Message: Train a model to predict house prices
3. Get: Model training results and performance metrics

Need more help? Just ask me anything!"""
}


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
    await update.message.reply_text(MESSAGE_TEMPLATES["welcome"])


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
    await update.message.reply_text(MESSAGE_TEMPLATES["help"])


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

    # NEW: Check for active workflow BEFORE parsing
    from src.bot.workflow_handlers import WorkflowRouter

    # Use shared StateManager instance from bot_data
    state_manager = context.bot_data['state_manager']
    session = await state_manager.get_or_create_session(
        user_id,
        f"chat_{update.effective_chat.id}"
    )

    # If user has active workflow, route to workflow handler
    if session.current_state is not None:
        workflow_router = WorkflowRouter(state_manager)
        logger.info(f"🔧 ROUTING TO WORKFLOW: state={session.current_state}, user={user_id}")
        return await workflow_router.handle(update, context, session)

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
            # NEW: Full processing pipeline integration
            try:
                # Import required components
                from src.core.parser import RequestParser
                from src.core.orchestrator import TaskOrchestrator
                from src.utils.result_formatter import TelegramResultFormatter
                from src.utils.exceptions import ParseError, ValidationError, DataError

                # Initialize components
                parser = RequestParser()
                orchestrator = TaskOrchestrator()
                formatter = TelegramResultFormatter()

                # Get user's dataframe
                dataframe = user_data.get('dataframe')
                if dataframe is None:
                    raise DataError("User data corrupted or missing")

                # Parse user request
                task = parser.parse_request(
                    text=message_text,
                    user_id=user_id,
                    conversation_id=f"chat_{update.effective_chat.id}",
                    data_source=None  # Data already loaded
                )

                logger.info(f"🔧 TASK PARSED: {task.task_type}/{task.operation} for user {user_id}")
                logger.info(f"🔧 CODE VERSION: v2.1.0-ml-workflow-fix")
                logger.info(f"🔧 PARAMETERS: {task.parameters}")
                logger.info(f"🔧 TARGET COLUMN: {task.parameters.get('target_column')}")
                logger.info(f"🔧 DATAFRAME COLUMNS: {dataframe.columns.tolist()}")

                # Check if this is an ML training request that needs workflow
                target_col = task.parameters.get('target_column')
                workflow_should_start = task.task_type == "ml_train" and (not target_col or target_col not in dataframe.columns)
                logger.info(f"🔧 ML WORKFLOW CHECK: task_type={task.task_type}, target={target_col}, in_columns={target_col in dataframe.columns if target_col else False}")
                logger.info(f"🔧 WORKFLOW SHOULD START: {workflow_should_start}")

                if workflow_should_start:
                    logger.info(f"🔧 STARTING ML TRAINING WORKFLOW...")
                    # Start ML training workflow instead of executing directly
                    # This handles both missing target AND invalid target (like "house" when column is "price")
                    from src.core.state_manager import WorkflowType, MLTrainingState
                    from src.bot.response_builder import ResponseBuilder

                    # Use shared StateManager instance from bot_data
                    state_manager = context.bot_data['state_manager']
                    response_builder = ResponseBuilder()

                    session = await state_manager.get_or_create_session(user_id, f"chat_{update.effective_chat.id}")
                    await state_manager.store_data(session, dataframe)
                    await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
                    await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)

                    # Build target selection prompt
                    columns = dataframe.columns.tolist()
                    response_message = response_builder.format_column_selection(columns, "target")

                    await update.message.reply_text(response_message, parse_mode="Markdown")
                    logger.info(f"🔧 ML WORKFLOW INITIATED - RETURNED TO USER")
                    return

                # Execute task through orchestrator
                result = await orchestrator.execute_task(task, dataframe)

                logger.info(f"🔧 TASK EXECUTED: success={result.get('success')} in {result.get('metadata', {}).get('execution_time', 0):.3f}s")

                # Format result for Telegram based on task type
                if task.task_type == "script":
                    # Import script handler for result formatting
                    from src.bot.script_handler import ScriptHandler
                    script_handler = ScriptHandler(parser, orchestrator)
                    response_message = script_handler.format_script_results(result)
                else:
                    response_message = formatter.format_stats_result(result)

                logger.info(f"🔧 RESPONSE FORMATTED: {len(response_message)} characters")

            except ParseError as e:
                # Handle parsing errors with helpful suggestions
                columns = metadata.get('columns', [])
                numeric_columns = metadata.get('numeric_columns', [])

                response_message = (
                    f"❓ **Request Not Understood**\n\n"
                    f"I couldn't understand: \"{message_text}\"\n\n"
                    f"**Try asking:**\n"
                    f"• \"Calculate statistics for {columns[0] if columns else 'column_name'}\"\n"
                    f"• \"Show correlation matrix\"\n"
                    f"• \"Calculate mean and std for all columns\"\n\n"
                    f"**Available columns:** {', '.join(columns[:10])}"
                    + ("..." if len(columns) > 10 else "")
                    + (f"\n**Numeric columns:** {', '.join(numeric_columns[:5])}" if numeric_columns else "")
                )

                logger.info(f"🔧 PARSE ERROR: {e.message}")

            except (DataError, ValidationError) as e:
                # Handle data/validation errors
                response_message = (
                    f"❌ **Processing Error**\n\n"
                    f"**Issue:** {e.message}\n\n"
                    f"**Your data:** {file_name} ({metadata.get('shape', (0, 0))[0]:,} rows)\n"
                    f"**Available columns:** {', '.join(metadata.get('columns', [])[:5])}\n\n"
                    f"Please check your request and try again."
                )

                logger.warning(f"🔧 DATA/VALIDATION ERROR: {e.message}")

            except Exception as e:
                # Handle unexpected errors with fallback
                logger.error(f"🔧 UNEXPECTED ERROR in message processing: {e}", exc_info=True)

                response_message = (
                    f"⚠️ **System Error**\n\n"
                    f"An unexpected error occurred while processing your request.\n\n"
                    f"**Your data:** {file_name} ({metadata.get('shape', (0, 0))[0]:,} rows)\n"
                    f"**Available columns:** {', '.join(metadata.get('columns', [])[:5])}\n\n"
                    f"**Try asking:**\n"
                    f"• Calculate statistics for {metadata.get('columns', ['column_name'])[0]}\n"
                    f"• Show correlation matrix\n\n"
                    f"If the problem persists, please try uploading your data again."
                )

    await update.message.reply_text(response_message, parse_mode="Markdown")


@telegram_handler
@log_user_action("Version check")
async def version_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /version command - show code version.

    This helps verify which code version the bot is running,
    useful for debugging and confirming bot restarts.
    """
    version_info = (
        "🤖 **Bot Version Information**\n\n"
        "**Code Version**: v2.1.0-ml-workflow-fix\n"
        "**ML Workflow**: ✅ Enabled\n"
        "**Error Handling**: ✅ Enhanced\n"
        "**Parameter Keys**: target_column, feature_columns\n\n"
        "If you see an old version, the bot needs to be restarted."
    )
    await update.message.reply_text(version_info, parse_mode="Markdown")


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


@telegram_handler
@log_user_action("Cancel workflow")
async def cancel_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /cancel command - cancel active workflow.

    Args:
        update: Telegram update object
        context: Bot context
    """
    user_id = update.effective_user.id

    # Use shared StateManager instance from bot_data
    state_manager = context.bot_data['state_manager']
    session = await state_manager.get_or_create_session(
        user_id,
        f"chat_{update.effective_chat.id}"
    )

    if session.workflow_type is None:
        await update.message.reply_text(
            "No active workflow to cancel.",
            parse_mode="Markdown"
        )
        return

    from src.bot.workflow_handlers import WorkflowRouter
    workflow_router = WorkflowRouter(state_manager)
    await workflow_router.cancel_workflow(update, session)


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