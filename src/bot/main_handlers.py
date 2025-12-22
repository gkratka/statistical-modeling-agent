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
from src.utils.decorators import telegram_handler, log_user_action, detect_and_set_language, Messages, safe_get_user_data
from src.utils.i18n_manager import I18nManager

# Import worker connection handlers
from src.bot.handlers.connect_handler import (
    handle_connect_command,
    handle_worker_connect_button,
    handle_worker_autostart_command,
    get_start_message_with_worker_status,
    notify_worker_connected,
    notify_worker_disconnected,
)

logger = get_logger(__name__)

# CRITICAL: Version identifier for debugging
BOT_VERSION = "DataLoader-v2.0-NUCLEAR-FIX"
FIXED_TIMESTAMP = "2025-01-27-NUCLEAR"
BOT_INSTANCE_ID = f"BIH-{FIXED_TIMESTAMP}"
HANDLERS_VERSION = "v2.0"
LAST_UPDATED = "2025-01-27"

# Helper function to build localized messages
def get_welcome_message(locale: str = None) -> str:
    """Build welcome message with version info."""
    return (
        f"{I18nManager.t('commands.start.welcome', locale=locale)}\n"
        f"{I18nManager.t('commands.start.version', locale=locale, version=BOT_VERSION)}\n"
        f"🔧 Instance: {BOT_INSTANCE_ID}\n\n"
        f"{I18nManager.t('commands.start.features', locale=locale)}\n\n"
        f"{I18nManager.t('commands.start.instructions', locale=locale)}"
    )


def get_help_message(locale: str = None) -> str:
    """Build help message with all commands."""
    return (
        f"{I18nManager.t('commands.help.title', locale=locale)}\n\n"
        f"{I18nManager.t('commands.help.description', locale=locale)}\n\n"
        f"{I18nManager.t('commands.help.commands_section', locale=locale)}\n"
        f"{I18nManager.t('commands.help.start_cmd', locale=locale)}\n"
        f"{I18nManager.t('commands.help.help_cmd', locale=locale)}\n"
        f"{I18nManager.t('commands.help.train_cmd', locale=locale)}\n"
        f"{I18nManager.t('commands.help.models_cmd', locale=locale)}\n"
        f"{I18nManager.t('commands.help.predict_cmd', locale=locale)}\n"
        f"{I18nManager.t('commands.help.pt_cmd', locale=locale)}\n"
        f"{I18nManager.t('commands.help.en_cmd', locale=locale)}\n\n"
        f"{I18nManager.t('commands.help.features_section', locale=locale)}\n\n"
        f"{I18nManager.t('commands.help.support', locale=locale)}"
    )


@telegram_handler
@detect_and_set_language
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
    # Get user's language from session (set by detect_and_set_language decorator)
    user_id = update.effective_user.id
    state_manager = context.bot_data.get('state_manager')

    locale = 'en'  # Default to English (explicit string, not None)
    if state_manager:
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )
        locale = session.language if session.language else 'en'

    # Debug print (temporary)
    print(f"🌍 START HANDLER: Using locale='{locale}' (session.language={session.language if state_manager else 'N/A'})")

    # Build base welcome message
    message = get_welcome_message(locale)

    # Add worker connection status (if feature enabled)
    worker_status, worker_keyboard = get_start_message_with_worker_status(user_id, context)
    message += worker_status

    # Send message with optional keyboard
    await update.message.reply_text(
        message,
        parse_mode="Markdown",
        reply_markup=worker_keyboard
    )


@telegram_handler
@detect_and_set_language
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
    # Get user's language from session (set by detect_and_set_language decorator)
    user_id = update.effective_user.id
    state_manager = context.bot_data.get('state_manager')

    locale = 'en'  # Default to English (explicit string, not None)
    if state_manager:
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )
        locale = session.language if session.language else 'en'

    await update.message.reply_text(get_help_message(locale))


@telegram_handler
@log_user_action("Language switch to Portuguese")
async def pt_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /pt command - switch to Portuguese language.

    Args:
        update: Telegram update object
        context: Bot context
    """
    user_id = update.effective_user.id
    state_manager = context.bot_data.get('state_manager')

    # Set language to Portuguese
    if state_manager:
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )
        session.language = "pt"
        await state_manager.update_session(session)
        print(f"🌐 PT HANDLER: Set session.language='pt' for user {user_id}")

    # Send welcome message in Portuguese
    await update.message.reply_text(get_welcome_message(locale="pt"))


@telegram_handler
@log_user_action("Language switch to English")
async def en_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /en command - switch to English language.

    Args:
        update: Telegram update object
        context: Bot context
    """
    user_id = update.effective_user.id
    state_manager = context.bot_data.get('state_manager')

    # Set language to English
    if state_manager:
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )
        session.language = "en"
        await state_manager.update_session(session)

    # Send welcome message in English
    await update.message.reply_text(get_welcome_message(locale="en"))


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
    from src.core.state_manager import MLTrainingState, MLPredictionState, ScoreWorkflowState

    # Use shared StateManager instance from bot_data
    state_manager = context.bot_data['state_manager']
    session = await state_manager.get_or_create_session(
        user_id,
        f"chat_{update.effective_chat.id}"
    )

    # Check for score template submission (before state routing)
    # Template markers: TRAIN_DATA:, MODEL:, TARGET:, PREDICT_DATA:
    score_template_markers = ['TRAIN_DATA:', 'PREDICT_DATA:', 'MODEL:', 'TARGET:']
    if any(marker in message_text for marker in score_template_markers):
        logger.info(f"📋 Score template detected for user {user_id}")

        # Cancel non-score workflow if active
        from src.core.state_manager import WorkflowType, ScoreWorkflowState
        if session.workflow_type is not None and session.workflow_type != WorkflowType.SCORE_WORKFLOW:
            await state_manager.cancel_workflow(session)
            logger.info(f"🔄 Cancelled workflow ({session.workflow_type.value}) for score template submission")

        # Start score workflow if not active
        if session.workflow_type != WorkflowType.SCORE_WORKFLOW:
            await state_manager.start_workflow(session, WorkflowType.SCORE_WORKFLOW)
            session.current_state = ScoreWorkflowState.AWAITING_TEMPLATE.value
            await state_manager.update_session(session)
            logger.info(f"🎯 Started score workflow for user {user_id}")

        # Route to score handler
        score_handler = context.bot_data.get('score_handler')
        if score_handler:
            logger.info(f"📤 Routing template to score handler for user {user_id}")
            return await score_handler.handle_template_submission(update, context, session)
        else:
            logger.error(f"❌ Score handler not found in bot_data for user {user_id}")
            await update.message.reply_text(
                "⚠️ **Score Handler Error**\n\nScore workflow is not properly configured. Please contact support.",
                parse_mode="Markdown"
            )
            return

    # Check for /score command - cancel workflow and let CommandHandler process it
    # This prevents workflow state from blocking command execution
    if message_text.strip().startswith('/score'):
        if session.current_state is not None:
            await state_manager.cancel_workflow(session)
            logger.info(f"🔄 Cancelled workflow ({session.current_state}) to allow /score command")
        # Let CommandHandler process the command
        return

    # Enhanced workflow state checking with specific ML training state detection
    # This prevents handler collision with local path text handler
    if session.current_state is not None:
        # Check for specific ML training states that require specialized handling
        ml_training_states = [
            MLTrainingState.CHOOSING_DATA_SOURCE.value,
            MLTrainingState.AWAITING_FILE_PATH.value,
            MLTrainingState.CHOOSING_LOAD_OPTION.value,
            MLTrainingState.CONFIRMING_SCHEMA.value,
            MLTrainingState.AWAITING_SCHEMA_INPUT.value,
            MLTrainingState.SELECTING_TARGET.value,
            MLTrainingState.SELECTING_FEATURES.value,
            MLTrainingState.CONFIRMING_MODEL.value,
            # Model naming workflow states (NEW - Bug Fix)
            MLTrainingState.TRAINING_COMPLETE.value,  # After training, showing naming options
            MLTrainingState.NAMING_MODEL.value,       # User entering custom model name (THE FIX)
        ]

        if session.current_state in ml_training_states:
            logger.info(f"🛑 EARLY EXIT: ML training state detected ({session.current_state}), deferring to specialized handler")
            # Return early - specialized handler should process this
            # Don't even route to workflow handler, just exit
            return

        # Check for ML prediction states that require specialized handling
        ml_prediction_states = [
            MLPredictionState.STARTED.value,
            MLPredictionState.CHOOSING_DATA_SOURCE.value,
            MLPredictionState.AWAITING_FILE_UPLOAD.value,
            MLPredictionState.AWAITING_FILE_PATH.value,
            MLPredictionState.CHOOSING_LOAD_OPTION.value,
            MLPredictionState.CONFIRMING_SCHEMA.value,
            MLPredictionState.AWAITING_FEATURE_SELECTION.value,
            MLPredictionState.SELECTING_MODEL.value,
            MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value,
            MLPredictionState.READY_TO_RUN.value,
            MLPredictionState.RUNNING_PREDICTION.value,
            MLPredictionState.COMPLETE.value,
            MLPredictionState.AWAITING_SAVE_PATH.value,
            MLPredictionState.CONFIRMING_SAVE_FILENAME.value,
        ]

        if session.current_state in ml_prediction_states:
            logger.info(f"🛑 EARLY EXIT: ML prediction state detected ({session.current_state}), deferring to specialized handler")
            # Return early - specialized handler should process this
            return

        # Check for score workflow states (NEW)
        score_workflow_states = [
            ScoreWorkflowState.AWAITING_TEMPLATE.value,
            ScoreWorkflowState.VALIDATING_INPUTS.value,
            ScoreWorkflowState.CONFIRMING_EXECUTION.value,
            ScoreWorkflowState.TRAINING_MODEL.value,
            ScoreWorkflowState.RUNNING_PREDICTION.value,
            ScoreWorkflowState.COMPLETE.value,
        ]

        if session.current_state in score_workflow_states:
            logger.info(f"🛑 EARLY EXIT: Score workflow state detected ({session.current_state}), deferring to score handler")
            # Route to score handler
            score_handler = context.bot_data.get('score_handler')
            if score_handler and session.current_state == ScoreWorkflowState.AWAITING_TEMPLATE.value:
                return await score_handler.handle_template_submission(update, context, session)
            return

        # For other workflow states, route to workflow handler
        workflow_router = WorkflowRouter(state_manager)
        logger.info(f"🔧 ROUTING TO WORKFLOW: state={session.current_state}, user={user_id}")
        return await workflow_router.handle(update, context, session)

    # Check if user has uploaded data
    user_data = safe_get_user_data(context, user_id)

    # Get locale from session for i18n
    locale = session.language if session.language else None

    if not user_data:
        response_message = Messages.upload_data_first(locale=locale)
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
        # Check if user has active workflow - prevent data upload during workflow
        state_manager = context.bot_data.get('state_manager')
        if state_manager:
            session = await state_manager.get_or_create_session(
                user_id,
                f"chat_{update.effective_chat.id}"
            )

            # Get locale from session for i18n
            locale = session.language if session.language else None

            if session.current_state is not None:
                await update.message.reply_text(
                    Messages.workflow_active(
                        workflow_type=session.workflow_type.value if session.workflow_type else 'unknown',
                        current_state=session.current_state,
                        locale=locale
                    ),
                    parse_mode="Markdown"
                )
                return

            # Idempotency check: Has this exact file already been processed?
            processed_file_id = session.selections.get('last_processed_file_id')
            current_file_id = document.file_id

            if processed_file_id == current_file_id:
                logger.info(
                    f"📄 Idempotency: file_id {current_file_id} already processed for user {user_id}"
                )
                await update.message.reply_text(
                    Messages.file_already_processed(filename=file_name, locale=locale),
                    parse_mode="Markdown"
                )
                return

            logger.info(
                f"📄 New file detected: file_id={current_file_id}, "
                f"previous_file_id={processed_file_id or 'none'}"
            )

        # DIAGNOSTIC: Log message being sent
        logger.info("🔧 SENDING MESSAGE: DataLoader v2.0 processing message")

        # Get locale if session was retrieved
        if 'locale' not in dir():
            locale = None  # Fallback if no session

        # Send initial processing message
        processing_msg = await update.message.reply_text(
            Messages.processing_file(locale=locale),
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

        # ALSO store in StateManager for ML workflow integration
        state_manager = context.bot_data['state_manager']
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )
        session.uploaded_data = df

        # Store file_id for idempotency (prevent duplicate processing)
        session.selections['last_processed_file_id'] = document.file_id
        session.selections['last_processed_file_name'] = file_name

        await state_manager.update_session(session)
        logger.info(
            f"💾 Stored file_id={document.file_id} in session for idempotency tracking"
        )

        # Generate success message with data summary
        success_message = loader.get_data_summary(df, metadata)
        success_message += Messages.ready_to_train(locale=locale)

        # Edit the processing message with results
        await processing_msg.edit_text(success_message, parse_mode="Markdown")

        logger.info(f"✅ Successfully processed file for user {user_id}: {metadata['shape']}")

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


@telegram_handler
@log_user_action("Start ML training")
async def train_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle /train command - start ML training workflow.

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

    # Check if workflow already active
    if session.workflow_type is not None:
        await update.message.reply_text(
            f"⚠️ Workflow already active: {session.workflow_type.value}\n\n"
            f"Use /cancel to cancel current workflow first.",
            parse_mode="Markdown"
        )
        return

    # Check if user has uploaded data
    if session.uploaded_data is None:
        await update.message.reply_text(
            "📂 **No data uploaded**\n\n"
            "Please upload a CSV file first, then use /train to start training.\n\n"
            "Example:\n"
            "1. Upload: `german_credit_data_train.csv`\n"
            "2. Command: `/train`\n"
            "3. Follow the prompts to configure your model",
            parse_mode="Markdown"
        )
        return

    # Start ML training workflow
    from src.core.state_manager import WorkflowType, MLTrainingState
    await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)

    # Transition to target selection state (workflow starts in awaiting_data but we already have data)
    session.current_state = MLTrainingState.SELECTING_TARGET.value
    await state_manager.update_session(session)

    # Get column information
    columns = session.uploaded_data.columns.tolist()

    await update.message.reply_text(
        f"🎯 **ML Training Workflow Started**\n\n"
        f"**Step 1/4: Select Target Column**\n\n"
        f"Your data has {len(columns)} columns:\n"
        + "\n".join(f"{i+1}. {col}" for i, col in enumerate(columns[:20]))
        + (f"\n... and {len(columns) - 20} more" if len(columns) > 20 else "")
        + f"\n\n**Reply with:**\n"
        f"• Column number (e.g., `21` for column 21)\n"
        f"• Column name (e.g., `class`)\n\n"
        f"_Use /cancel to stop at any time_",
        parse_mode="Markdown"
    )


@log_user_action("Workflow back button")
async def handle_workflow_back(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Universal back button handler for workflow navigation.

    Handles the 'workflow_back' callback from inline keyboards across
    all workflows. Implements debouncing (500ms) to prevent race conditions
    from rapid clicks.

    Behavior:
    1. Retrieve user session from StateManager
    2. Check if back navigation is possible (state history not empty)
    3. Apply debouncing (reject if last_back_action < 500ms ago)
    4. Restore previous state from history
    5. Re-render the previous step's UI

    Args:
        update: Telegram update object (callback query)
        context: Bot context

    Related: dev/implemented/workflow-back-button.md (Phase 2, Phase 3: Error Handling)
    """
    query = update.callback_query
    await query.answer()  # Acknowledge the button press

    user_id = update.effective_user.id

    try:
        # Get session from StateManager
        state_manager = context.bot_data['state_manager']
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )

        # Enhanced logging: Current state before back navigation (Phase 3 fix)
        logger.info(
            f"🔙 Back button pressed - user_id={user_id}, "
            f"current_state={session.current_state}, "
            f"history_depth={session.state_history.get_depth()}"
        )

        # SIMPLIFIED: For early /train and /predict states (before model selection), just show welcome message
        from src.core.state_manager import MLTrainingState, MLPredictionState
        early_training_states = [
            MLTrainingState.CHOOSING_DATA_SOURCE.value,
            MLTrainingState.AWAITING_FILE_PATH.value,
            MLTrainingState.AWAITING_PASSWORD.value,
            MLTrainingState.CHOOSING_LOAD_OPTION.value,
            MLTrainingState.CONFIRMING_SCHEMA.value,
            MLTrainingState.AWAITING_SCHEMA_INPUT.value,
        ]
        early_prediction_states = [
            MLPredictionState.STARTED.value,
            MLPredictionState.CHOOSING_DATA_SOURCE.value,
            MLPredictionState.AWAITING_FILE_UPLOAD.value,
            MLPredictionState.AWAITING_FILE_PATH.value,
            MLPredictionState.AWAITING_PASSWORD.value,
            MLPredictionState.CHOOSING_LOAD_OPTION.value,
            MLPredictionState.CONFIRMING_SCHEMA.value,
        ]

        if session.current_state in early_training_states or session.current_state in early_prediction_states:
            logger.info(f"🔙 Early training state - cancelling workflow and showing welcome for user {user_id}")
            await state_manager.cancel_workflow(session)

            # Get locale for welcome message
            locale = session.language if session.language else 'en'

            await query.edit_message_text(
                get_welcome_message(locale),
                parse_mode="Markdown"
            )
            return

        # Check if back navigation is possible
        if not session.can_go_back():
            logger.warning(f"⚠️ Cannot go back - history empty for user {user_id}")
            await query.edit_message_text(
                "⚠️ **Cannot Go Back**\n\n"
                "You're at the beginning of the workflow.\n\n"
                "Use /cancel to exit.",
                parse_mode="Markdown"
            )
            return

        # Debouncing: prevent rapid clicks (500ms cooldown)
        import time
        current_time = time.time()

        if session.last_back_action is not None:
            time_since_last = current_time - session.last_back_action
            if time_since_last < 0.5:  # 500ms debounce
                logger.info(
                    f"🔙 Back button debounced for user {user_id}: "
                    f"{time_since_last * 1000:.0f}ms since last action"
                )
                await query.answer(
                    "⏳ Please wait a moment...",
                    show_alert=False
                )
                return

        # Update debounce timestamp
        session.last_back_action = current_time

        # Store previous state for logging (Phase 3 fix)
        previous_state = session.current_state

        # Restore previous state
        logger.debug(f"📦 Attempting to restore previous state for user {user_id}")
        success = session.restore_previous_state()

        if not success:
            logger.error(
                f"❌ restore_previous_state() failed for user {user_id} "
                f"at state {previous_state}"
            )
            await query.edit_message_text(
                "❌ **Navigation Error**\n\n"
                "Failed to restore previous state.\n\n"
                "Use /cancel to exit workflow.",
                parse_mode="Markdown"
            )
            return

        # Log successful restoration (Phase 3 fix)
        restored_state = session.current_state
        logger.info(
            f"✅ State restored successfully - user_id={user_id}, "
            f"{previous_state} → {restored_state}, "
            f"remaining_depth={session.state_history.get_depth()}"
        )

        # Save updated session
        await state_manager.update_session(session)
        logger.debug(f"💾 Session updated after state restoration for user {user_id}")

        # Re-render the UI for the restored state (Phase 3 fix: added logging)
        logger.info(
            f"🎨 Attempting to render state '{restored_state}' for user {user_id}"
        )

        from src.bot.workflow_handlers import WorkflowRouter
        workflow_router = WorkflowRouter(state_manager)

        await workflow_router.render_current_state(update, context, session)

        logger.info(
            f"✅ Successfully rendered state '{restored_state}' for user {user_id}"
        )

    except Exception as e:
        # Comprehensive error handling and logging (Phase 3 fix)
        logger.exception(
            f"💥 CRITICAL ERROR in handle_workflow_back() for user {user_id}"
        )
        logger.error(f"💥 Error type: {type(e).__name__}")
        logger.error(f"💥 Error message: {str(e)}")

        try:
            # Try to show user-friendly error
            await query.edit_message_text(
                f"❌ **Critical Error**\n\n"
                f"An error occurred during back navigation:\n\n"
                f"`{str(e)[:200]}`\n\n"
                f"Please use /cancel to restart the workflow.",
                parse_mode="Markdown"
            )
        except Exception as msg_error:
            logger.error(
                f"💥 Failed to send error message to user {user_id}: {msg_error}"
            )


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

    # Import telegram error types for filtering
    from telegram import error as telegram_error

    # Filter Telegram API errors - log but don't show to users
    if isinstance(error, (telegram_error.Conflict, telegram_error.NetworkError, telegram_error.BadRequest)):
        logger.warning(
            f"⚠️ Telegram API error (not user-facing): {type(error).__name__} - {error}"
        )
        # Don't send error message to user for API-level errors
        return

    # Log all other errors
    logger.error(f"Bot error occurred: {error}", exc_info=error)

    # If we have an update with a message, try to inform the user
    # (Only for application logic errors, not API errors)
    if isinstance(update, Update) and update.effective_message:
        try:
            # Use default locale since we can't reliably get session in error handler
            await update.effective_message.reply_text(
                Messages.error_occurred(),
                parse_mode="Markdown"
            )
        except Exception as reply_error:
            logger.error(f"Failed to send error message to user: {reply_error}")

    # Re-raise if it's a critical AgentError
    if isinstance(error, AgentError):
        logger.critical(f"Critical agent error: {error.error_code} - {error.message}")
    else:
        logger.error(f"Unexpected error: {error}")