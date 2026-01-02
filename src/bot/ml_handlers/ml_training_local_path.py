"""Telegram bot handlers for ML training with local file path workflow."""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ApplicationHandlerStop

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
from src.core.template_manager import TemplateManager
from src.core.training_template import TemplateConfig
from src.processors.data_loader import DataLoader
from src.utils.exceptions import PathValidationError, DataError, ValidationError, TrainingError
from src.utils.schema_detector import DatasetSchema
from src.utils.path_validator import PathValidator
from src.utils.password_validator import PasswordValidator
from src.bot.messages import LocalPathMessages
from src.bot.messages.local_path_messages import add_back_button
from src.utils.i18n_manager import I18nManager
from src.bot.utils.markdown_escape import escape_markdown_v1
from src.bot.ml_handlers.template_handlers import TemplateHandlers
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.keras_templates import get_template
from src.utils.task_type_detector import detect_task_type, get_recommended_models

logger = logging.getLogger(__name__)


class LocalPathMLTrainingHandler:
    """Handler for ML training workflow with local file path support."""

    def __init__(
        self,
        state_manager: StateManager,
        data_loader: DataLoader,
        template_manager: TemplateManager = None,
        path_validator: PathValidator = None
    ):
        """Initialize handler with state manager and data loader."""
        self.state_manager = state_manager
        self.data_loader = data_loader
        self.logger = logger

        # Initialize ML Engine for training
        ml_config = MLEngineConfig.get_default()
        self.ml_engine = MLEngine(ml_config)

        # Initialize template handler (Phase 6: Templates)
        if template_manager is None:
            # Create default template manager from config
            import yaml
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
            with open(config_path) as f:
                config = yaml.safe_load(f)
            template_config = TemplateConfig(
                enabled=config['templates']['enabled'],
                templates_dir=config['templates']['templates_dir'],
                max_templates_per_user=config['templates']['max_templates_per_user'],
                allowed_name_pattern=config['templates']['allowed_name_pattern'],
                name_max_length=config['templates']['name_max_length']
            )
            template_manager = TemplateManager(template_config)

        if path_validator is None:
            path_validator = PathValidator(
                allowed_directories=self.data_loader.allowed_directories,
                max_size_mb=self.data_loader.local_max_size_mb,
                allowed_extensions=self.data_loader.local_extensions
            )

        self.template_handlers = TemplateHandlers(
            state_manager=state_manager,
            template_manager=template_manager,
            data_loader=data_loader,
            path_validator=path_validator,
            training_executor=self.handle_training_execution
        )

        # NEW: Initialize password validator (Phase 3: Password Protection)
        self.password_validator = PasswordValidator()

    async def handle_start_training(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /train command - shows data source selection or file upload."""
        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_start_training: {e}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.malformed_update'),
                    parse_mode="Markdown"
                )
            return

        # Get or create session
        session = await self.state_manager.get_or_create_session(
            user_id=user_id,
            conversation_id=f"chat_{chat_id}"
        )

        # Check if local path feature is enabled
        if self.data_loader.local_enabled:
            # NEW: Show data source selection
            await self._show_data_source_selection(update, context, session)
        else:
            # Legacy: Go directly to file upload
            await self._start_telegram_upload_workflow(update, context, session)

    async def _start_telegram_upload_workflow(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Start legacy Telegram upload workflow (when local paths disabled)."""
        # Manually initialize workflow at AWAITING_DATA state
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_DATA.value
        await self.state_manager.update_session(session)

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        await update.message.reply_text(
            LocalPathMessages.telegram_upload_prompt(locale=locale),
            parse_mode="Markdown"
        )

    async def _show_data_source_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show data source selection (Telegram upload vs local path)."""
        # Manually initialize workflow at CHOOSING_DATA_SOURCE state
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        await self.state_manager.update_session(session)

        # Extract locale from session for i18n
        locale = session.language if session.language else None
        print(f"🌐 ML TRAINING: session.language={session.language}, extracted locale={locale}")

        # Create inline keyboard with i18n button labels
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflows.ml_training.upload_button', locale=locale),
                callback_data="data_source:telegram"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflows.ml_training.local_path_button', locale=locale),
                callback_data="data_source:local_path"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflows.ml_training.template_button', locale=locale),
                callback_data="data_source:template"
            )]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        message = LocalPathMessages.data_source_selection_prompt(locale=locale)
        print(f"🌐 ML TRAINING: Generated message (first 100 chars): {message[:100]}")

        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_data_source_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle data source selection callback."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split(":")[-1]  # "telegram" or "local_path"
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_data_source_selection: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        if choice == "telegram":
            # User chose Telegram upload
            print("🔀 DEBUG: User chose telegram data source")
            session.data_source = "telegram"

            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to AWAITING_DATA")

            old_state = session.current_state
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.AWAITING_DATA.value
            )

            if not success:
                print(f"❌ DEBUG: State transition FAILED! Error: {error_msg}")
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.state_transition_failed',
                                 error=error_msg, locale=locale),
                    parse_mode="Markdown"
                )
                return

            print(f"🔀 DEBUG: State transition SUCCESS: {old_state} → {session.current_state}")

            await query.edit_message_text(
                LocalPathMessages.telegram_upload_prompt(locale=locale),
                parse_mode="Markdown"
            )

        elif choice == "local_path":
            # User chose local path
            print("🔀 DEBUG: User chose local_path data source")
            session.data_source = "local_path"

            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to AWAITING_FILE_PATH")

            # Store old state before transition
            old_state = session.current_state
            target_state = MLTrainingState.AWAITING_FILE_PATH.value

            # Perform transition (returns: success, error_msg, missing_prerequisites)
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                target_state
            )

            if not success:
                print(f"❌ DEBUG: State transition FAILED! Error: {error_msg}, Missing: {missing}")
                await query.edit_message_text(
                    f"❌ State transition failed: {error_msg}",
                    parse_mode="Markdown"
                )
                return

            print(f"🔀 DEBUG: State transition SUCCESS: {old_state} → {session.current_state}")

            # Show allowed directories with polished prompt
            allowed_dirs = self.data_loader.allowed_directories

            # Wrap message editing with defensive error handling
            try:
                await query.edit_message_text(
                    LocalPathMessages.file_path_input_prompt(allowed_dirs, locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.BadRequest as e:
                logger.warning(f"Failed to edit message in handle_data_source_selection: {e}")
                # Fallback: send new message instead
                await update.effective_message.reply_text(
                    LocalPathMessages.file_path_input_prompt(allowed_dirs, locale=locale),
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.warning(f"Telegram API error in handle_data_source_selection: {e}")
                # Continue even if message editing fails
            print("🔀 DEBUG: File path prompt sent to user")

        elif choice == "template":
            # User chose to use a saved template
            print("🔀 DEBUG: User chose template data source")
            session.data_source = "template"

            # Route to template handler
            await self.template_handlers.handle_template_source_selection(update, context)

    async def handle_text_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Unified text input handler that routes based on current state.
        Handles both file path input and schema input.
        """
        try:
            # Defensive extraction - protect against None values in update object
            try:
                user_id = update.effective_user.id
                chat_id = update.effective_chat.id
                text_input = update.message.text.strip()
            except AttributeError as e:
                logger.error(f"Malformed update object in handle_text_input: {e}")
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request'),
                        parse_mode="Markdown"
                    )
                return

            print(f"📥 DEBUG: Unified text handler called with: {text_input[:50]}...")
            print(f"🔍 PATH_DEBUG: Raw text_input from Telegram = '{text_input}'")

            # Get or create session - prevents false "Session Expired" errors
            session = await self.state_manager.get_or_create_session(user_id, f"chat_{chat_id}")

            # WORKFLOW ISOLATION: Only process if in TRAINING workflow
            # Prevents collision with prediction workflow handler (group=2)
            if session.workflow_type != WorkflowType.ML_TRAINING:
                print(f"⏭️  DEBUG: Session is in {session.workflow_type} workflow, not ML_TRAINING - ignoring")
                return

            current_state = session.current_state
            print(f"📊 DEBUG: Current session state = {current_state}")

            # Check for score template submission (highest priority - before state routing)
            score_template_markers = ['TRAIN_DATA:', 'PREDICT_DATA:', 'MODEL:', 'TARGET:']
            if any(marker in text_input for marker in score_template_markers):
                print(f"📋 Score template detected in ML handler - cancelling ML workflow")
                # Cancel ML training workflow to allow score workflow
                if session.workflow_type == WorkflowType.ML_TRAINING:
                    await self.state_manager.cancel_workflow(session)
                    print(f"🔄 Cancelled ML_TRAINING workflow for score template")
                # Don't stop propagation - return to allow message_handler to process template
                return

            # Route based on current state
            if current_state == MLTrainingState.AWAITING_FILE_PATH.value:
                print("🔀 DEBUG: Routing to file path logic")
                await self._process_file_path_input(update, context, session, text_input)
            elif current_state == MLTrainingState.AWAITING_PASSWORD.value:  # NEW
                print("🔀 DEBUG: Routing to password input logic")
                await self.handle_password_input(update, context)
            elif current_state == MLTrainingState.AWAITING_SCHEMA_INPUT.value:
                print("🔀 DEBUG: Routing to schema input logic")
                await self._process_schema_input(update, context, session, text_input)
            elif current_state == MLTrainingState.SAVING_TEMPLATE.value:
                print("🔀 DEBUG: Routing to template name input logic")
                await self.template_handlers.handle_template_name_input(update, context)
            elif current_state == MLTrainingState.NAMING_MODEL.value:
                print("🔀 DEBUG: Routing to model name input logic")
                await self.handle_model_name_input(update, context)
            else:
                print(f"⏭️  DEBUG: State {current_state} doesn't require text input, ignoring")
                return

        except ApplicationHandlerStop:
            # Re-raise immediately - this is control flow, not an error
            raise

        except Exception as e:
            # Enhanced error logging to capture actual exception
            logger.error(f"💥 CRITICAL ERROR in handle_text_input: {e}")
            logger.error(f"💥 Error type: {type(e).__name__}")
            logger.error(f"💥 User ID: {getattr(update.effective_user, 'id', 'unknown') if update.effective_user else 'unknown'}")

            import traceback
            logger.error(f"💥 Full traceback:\n{traceback.format_exc()}")
            print(f"💥 EXCEPTION CAUGHT in handle_text_input: {e}")
            print(f"💥 Full traceback:\n{traceback.format_exc()}")

            # Re-raise to let global error handler send message to user
            raise

    async def _process_file_path_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        file_path: str
    ) -> None:
        """Process file path input from user. Validates path and shows load options."""
        print(f"🔍 DEBUG: Processing file path: {file_path}")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Show validating message
        validating_msg = await update.message.reply_text(
            I18nManager.t('workflows.ml_training_local_path.path.validating', locale=locale)
        )
        print("✅ DEBUG: Validating message sent")

        try:
            # Validate path only (don't load data yet)
            from src.utils.path_validator import validate_local_path, get_file_size_mb
            from pathlib import Path

            is_valid, error_msg, resolved_path = validate_local_path(
                path=file_path,
                allowed_dirs=self.data_loader.allowed_directories,
                max_size_mb=self.data_loader.local_max_size_mb,
                allowed_extensions=self.data_loader.local_extensions,
                forbidden_dirs=self.data_loader.forbidden_directories
            )

            if not is_valid:
                # NEW: Check if error is specifically whitelist failure
                if "not in allowed directories" in error_msg.lower():
                    # Whitelist check failed - prompt for password
                    await validating_msg.delete()
                    await self._prompt_for_password(
                        update, context, session, file_path, resolved_path
                    )
                    raise ApplicationHandlerStop
                else:
                    # Other validation error (path traversal, size, etc.)
                    await validating_msg.delete()
                    error_display = LocalPathMessages.format_path_error(
                        error_type="security_validation",
                        path=file_path,
                        error_details=error_msg,
                        locale=locale
                    )
                    await update.message.reply_text(error_display)
                    return

            # Path valid - store it and get file size
            session.file_path = str(resolved_path)
            size_mb = get_file_size_mb(resolved_path)
            print(f"📏 DEBUG: File size calculated: {size_mb}MB")

            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to CHOOSING_LOAD_OPTION")
            print(f"📸 DEBUG: Snapshot saved, transitioning to CHOOSING_LOAD_OPTION")

            # Transition to choosing load option
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CHOOSING_LOAD_OPTION.value
            )
            print(f"🔀 DEBUG: State transition complete: {session.current_state}")

            # Delete validating message
            await validating_msg.delete()
            print("🗑️ DEBUG: Validating message deleted")

            # Show load option selection
            keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('workflows.ml_training.load_now_button', locale=locale),
                    callback_data="load_option:immediate"
                )],
                [InlineKeyboardButton(
                    I18nManager.t('workflows.ml_training.defer_loading_button', locale=locale),
                    callback_data="load_option:defer"
                )]
            ]
            add_back_button(keyboard)  # Phase 2: Workflow Back Button
            reply_markup = InlineKeyboardMarkup(keyboard)
            print("⌨️ DEBUG: Keyboard created with Load Now/Defer Loading buttons")

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(str(resolved_path), size_mb, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
            print("📤 DEBUG: Load option message sent to user")

            # Stop handler propagation to prevent generic message handler from processing
            print("🛑 DEBUG: Stopping handler propagation after successful path validation")
            raise ApplicationHandlerStop

        except ApplicationHandlerStop:
            # Re-raise immediately - this is control flow, not an error
            raise

        except Exception as e:
            print(f"❌ DEBUG: Unexpected error validating path: {e}")
            import traceback
            traceback.print_exc()
            try:
                await validating_msg.delete()
            except Exception as delete_err:
                print(f"⚠️  DEBUG: Failed to delete validating message: {delete_err}")

            await update.message.reply_text(
                LocalPathMessages.format_path_error(
                    error_type="unexpected",
                    path=file_path,
                    error_details=str(e),
                    locale=locale
                )
            )

    # =========================================================================
    # Password-Protected Path Access Methods (Phase 3: Password Implementation)
    # =========================================================================

    async def _prompt_for_password(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        original_path: str,
        resolved_path
    ) -> None:
        """Prompt user for password to access non-whitelisted path.

        Args:
            update: Telegram update
            context: Telegram context
            session: User session
            original_path: Original path string from user
            resolved_path: Resolved absolute path (Path object)
        """
        from pathlib import Path

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Store pending path for later validation
        session.pending_auth_path = str(resolved_path)

        # Transition to password state
        session.save_state_snapshot()
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLTrainingState.AWAITING_PASSWORD.value
        )

        if not success:
            await update.message.reply_text(
                I18nManager.t('workflows.ml_training_local_path.errors.state_transition_failed',
                             error=error_msg, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get parent directory for display
        parent_dir = str(Path(resolved_path).parent)

        # Show password prompt
        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="password:cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            LocalPathMessages.password_prompt(original_path, parent_dir, locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_password_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle password input for path access."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            password_input = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_password_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Validate we're in password state
        if session.current_state != MLTrainingState.AWAITING_PASSWORD.value:
            return

        # Get pending path
        pending_path = session.pending_auth_path
        if not pending_path:
            await update.message.reply_text(
                I18nManager.t('workflows.ml_training_local_path.password.session_error', locale=locale).replace('/predict', '/train'),
                parse_mode="Markdown"
            )
            return

        # Validate password
        is_valid, error_msg = self.password_validator.validate_password(
            user_id=user_id,
            password_input=password_input,
            path=pending_path
        )

        if is_valid:
            # FIX: Double-check pending_path is still valid (defense in depth)
            if not pending_path:
                await update.message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
                return

            # Password correct - add directory to dynamic whitelist
            from pathlib import Path
            from src.utils.path_validator import get_file_size_mb

            parent_dir = str(Path(pending_path).parent)
            self.state_manager.add_dynamic_directory(session, parent_dir)

            # Clear pending auth
            session.pending_auth_path = None
            session.password_attempts = 0

            # Store file path and get size
            session.file_path = pending_path
            print(f"🔍 PATH_DEBUG: session.file_path AFTER password = '{session.file_path}'")

            # Check if worker is connected (for prod where file is on user's machine)
            websocket_server = context.bot_data.get('websocket_server')
            worker_connected = (
                websocket_server and
                websocket_server.worker_manager.is_user_connected(user_id)
            )

            if worker_connected:
                # Use worker (for prod where file is on user's machine)
                file_info = await self._get_file_info_from_worker(user_id, pending_path, context)
                if file_info and file_info.get('exists'):
                    size_mb = file_info.get('size_mb', 0)
                else:
                    await update.message.reply_text(
                        "❌ **File Not Accessible**\n\n"
                        "The worker couldn't access the file. Please verify:\n"
                        "• The file path is correct\n"
                        "• The worker is still connected\n"
                        "• The file exists on your machine",
                        parse_mode="Markdown"
                    )
                    return
            else:
                # Use local (for dev where file is on same machine as bot)
                size_mb = get_file_size_mb(Path(pending_path))

            # Transition to load options
            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CHOOSING_LOAD_OPTION.value
            )

            await update.message.reply_text(
                LocalPathMessages.password_success(parent_dir, locale=locale),
                parse_mode="Markdown"
            )

            # Show load options
            keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('workflows.ml_training.load_now_button', locale=locale),
                    callback_data="load_option:immediate"
                )],
                [InlineKeyboardButton(
                    I18nManager.t('workflows.ml_training.defer_loading_button', locale=locale),
                    callback_data="load_option:defer"
                )]
            ]
            add_back_button(keyboard)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(pending_path, size_mb, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            raise ApplicationHandlerStop

        else:
            # Password incorrect or rate limited
            session.password_attempts += 1

            if "locked" in error_msg.lower() or "maximum attempts" in error_msg.lower():
                # Rate limit exceeded - reset to file path input
                session.pending_auth_path = None
                session.password_attempts = 0

                # Save state snapshot BEFORE transition (enables back button)
                session.save_state_snapshot()

                await self.state_manager.transition_state(
                    session,
                    MLTrainingState.AWAITING_FILE_PATH.value
                )

                await update.message.reply_text(
                    f"❌ {error_msg}\n\nPlease try again or choose a different path.",
                    parse_mode="Markdown"
                )
            else:
                # Failed attempt, allow retry
                await update.message.reply_text(
                    LocalPathMessages.password_failure(error_msg, locale=locale),
                    parse_mode="Markdown"
                )

    async def handle_password_cancel(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle password cancel callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_password_cancel: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Clear pending auth
        session.pending_auth_path = None
        session.password_attempts = 0

        # Reset password validator attempts
        self.password_validator.reset_attempts(user_id)

        # Go back to file path input
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLTrainingState.AWAITING_FILE_PATH.value
        )

        allowed_dirs = self.data_loader.allowed_directories
        await query.edit_message_text(
            LocalPathMessages.file_path_input_prompt(allowed_dirs, locale=locale),
            parse_mode="Markdown"
        )

    async def handle_load_option_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle load option selection (immediate or defer)."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split(":")[-1]  # "immediate" or "defer"
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_load_option_selection: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        if choice == "immediate":
            # Load data immediately
            self.logger.info(f"[LOAD_NOW] User {user_id} selected: {choice}")
            self.logger.info(f"[LOAD_NOW] Session state: {session.current_state}")
            self.logger.info(f"[LOAD_NOW] File path: {session.file_path}")
            print(f"🔍 PATH_DEBUG: session.file_path at LOAD_NOW = '{session.file_path}'")

            loading_msg = await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.path.loading', locale=locale)
            )

            try:
                self.logger.info(f"[LOAD_NOW] Starting data load from: {session.file_path}")

                # Load data from local path with schema detection
                df, metadata, schema = await self.data_loader.load_from_local_path(
                    file_path=session.file_path,
                    detect_schema_flag=True
                )

                self.logger.info(f"[LOAD_NOW] ✅ Data loaded successfully: shape={df.shape}")
                self.logger.info(f"[LOAD_NOW] Metadata: {metadata}")

                # Store data in session
                session.uploaded_data = df
                session.load_deferred = False

                # Store schema information
                if schema:
                    session.detected_schema = {
                        'task_type': schema.suggested_task_type,
                        'target': schema.suggested_target,
                        'features': schema.suggested_features,
                        'quality_score': schema.overall_quality_score,
                        'n_rows': schema.n_rows,
                        'n_columns': schema.n_columns
                    }
                    self.logger.info(f"[LOAD_NOW] Schema detected: target={schema.suggested_target}, "
                                   f"features={len(schema.suggested_features)}, task={schema.suggested_task_type}")

                # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
                session.save_state_snapshot()
                self.logger.debug("📸 State snapshot saved before transition to CONFIRMING_SCHEMA")

                # Transition to schema confirmation
                await self.state_manager.transition_state(
                    session,
                    MLTrainingState.CONFIRMING_SCHEMA.value
                )

                self.logger.info(f"[LOAD_NOW] State transition complete: {session.current_state}")

                # Show schema confirmation
                await self._show_schema_confirmation_after_load(
                    query, context, session, metadata, schema
                )

            except (FileNotFoundError, PathValidationError) as e:
                self.logger.error(f"[LOAD_NOW] ❌ File not found or path validation failed: {e}")
                self.logger.error(f"[LOAD_NOW] Path attempted: {session.file_path}")

                # Revert state to allow retry (direct assignment since this is error recovery)
                session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
                await self.state_manager.update_session(session)

                await query.edit_message_text(
                    "❌ **File Not Found**\n\n"
                    f"The file `{session.file_path}` does not exist.\n\n"
                    f"**Possible causes:**\n"
                    f"• File was moved or deleted\n"
                    f"• Path was typed incorrectly\n"
                    f"• File is in a restricted directory\n\n"
                    f"Please send a new file path to try again, or use /train to restart.",
                    parse_mode="Markdown"
                )

            except PermissionError as e:
                self.logger.error(f"[LOAD_NOW] ❌ Permission denied: {e}")
                self.logger.error(f"[LOAD_NOW] Path attempted: {session.file_path}")

                # Revert state to allow retry
                await self.state_manager.transition_state(
                    session,
                    MLTrainingState.AWAITING_FILE_PATH.value
                )

                await query.edit_message_text(
                    "❌ **Permission Denied**\n\n"
                    f"Cannot read file `{session.file_path}`\n\n"
                    f"**Possible causes:**\n"
                    f"• Insufficient read permissions\n"
                    f"• File is locked by another process\n"
                    f"• Operating system restrictions\n\n"
                    f"Please check file permissions and try again.",
                    parse_mode="Markdown"
                )

            except pd.errors.EmptyDataError as e:
                self.logger.error(f"[LOAD_NOW] ❌ Empty file detected: {e}")
                self.logger.error(f"[LOAD_NOW] File: {session.file_path}")

                # Revert state to allow retry (direct assignment since this is error recovery)
                session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
                await self.state_manager.update_session(session)

                await query.edit_message_text(
                    "❌ **Empty File Error**\n\n"
                    f"The file `{session.file_path}` is empty (0 bytes).\n\n"
                    f"**Required:**\n"
                    f"• CSV file must contain data\n"
                    f"• At least a header row and one data row\n\n"
                    f"Please provide a file with actual data, or use /train to restart.",
                    parse_mode="Markdown"
                )

            except Exception as e:
                # Check if it's a pandas ParserError
                error_type = type(e).__name__

                if "ParserError" in error_type or "CSV" in str(e):
                    self.logger.error(f"[LOAD_NOW] ❌ CSV parsing failed: {e}")
                    self.logger.error(f"[LOAD_NOW] File: {session.file_path}")

                    # Revert state to allow retry
                    await self.state_manager.transition_state(
                        session,
                        MLTrainingState.AWAITING_FILE_PATH.value
                    )

                    await query.edit_message_text(
                        "❌ **CSV Parsing Error**\n\n"
                        f"Failed to parse `{session.file_path}`\n\n"
                        f"**Error details:**\n"
                        f"```\n{str(e)[:200]}\n```\n\n"
                        f"**Possible causes:**\n"
                        f"• Corrupted CSV file\n"
                        f"• Inconsistent column count\n"
                        f"• Invalid encoding\n"
                        f"• File is not actually CSV format\n\n"
                        f"Please fix the file or try a different one.",
                        parse_mode="Markdown"
                    )
                else:
                    # Unknown error - log full details
                    self.logger.exception(f"[LOAD_NOW] ❌ Unexpected error loading data:")
                    self.logger.error(f"[LOAD_NOW] Error type: {error_type}")
                    self.logger.error(f"[LOAD_NOW] File: {session.file_path}")
                    self.logger.error(f"[LOAD_NOW] User: {user_id}")

                    # Revert state to allow retry
                    await self.state_manager.transition_state(
                        session,
                        MLTrainingState.AWAITING_FILE_PATH.value
                    )

                    await query.edit_message_text(
                        "❌ **Unexpected Error**\n\n"
                        f"An unexpected error occurred while loading data.\n\n"
                        f"**Error type:** `{error_type}`\n"
                        f"**Details:** {str(e)[:150]}\n\n"
                        f"Please check the logs for more information or try a different file.",
                        parse_mode="Markdown"
                    )

        elif choice == "defer":
            # Defer loading - ask for manual schema
            session.load_deferred = True

            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to AWAITING_SCHEMA_INPUT")

            await self.state_manager.transition_state(
                session,
                MLTrainingState.AWAITING_SCHEMA_INPUT.value
            )

            await query.edit_message_text(
                LocalPathMessages.schema_input_prompt(locale=locale),
                parse_mode="Markdown"
            )

    async def _process_schema_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        schema_text: str
    ) -> None:
        """Process manual schema input from user (deferred loading)."""
        print(f"📝 DEBUG: Processing schema input: {schema_text[:50]}...")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        try:
            # Parse schema using schema parser
            from src.utils.schema_parser import SchemaParser

            parsed_schema = SchemaParser.parse(schema_text)

            print(f"📝 DEBUG: Parsed schema - Target: {parsed_schema.target}")
            print(f"📝 DEBUG: Parsed schema - Features: {parsed_schema.features} (count={len(parsed_schema.features)})")
            print(f"📝 DEBUG: Parsed schema - Format: {parsed_schema.format_detected}")

            # Store manual schema in session
            session.manual_schema = {
                'target': parsed_schema.target,
                'features': parsed_schema.features,
                'format_detected': parsed_schema.format_detected,
                'raw_input': parsed_schema.raw_input
            }

            # Also store in selections for compatibility with training workflow
            session.selections['target_column'] = parsed_schema.target
            session.selections['feature_columns'] = parsed_schema.features

            # Detect task type by loading a small sample of the file
            # This enables smart model filtering even with deferred loading
            detected_task = None
            if session.file_path:
                try:
                    import pandas as pd
                    # Load only first 100 rows for task type detection
                    sample_df = pd.read_csv(session.file_path, nrows=100)

                    if parsed_schema.target in sample_df.columns:
                        detected_task = detect_task_type(sample_df[parsed_schema.target])
                        session.selections['detected_task_type'] = detected_task
                        print(f"🔍 DEBUG: Detected task type = {detected_task} for target '{parsed_schema.target}' (from sample)")
                    else:
                        print(f"⚠️ DEBUG: Target column '{parsed_schema.target}' not found in data, skipping detection")
                except Exception as e:
                    logger.warning(f"Task type detection failed: {e}")
                    print(f"⚠️ DEBUG: Task type detection error: {e}")

            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to CONFIRMING_MODEL")

            # Transition to CONFIRMING_MODEL (skip target/feature selection since schema has both)
            old_state = session.current_state
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.CONFIRMING_MODEL.value
            )

            if not success:
                print(f"❌ DEBUG: State transition to CONFIRMING_MODEL failed: {error_msg}")
                # Fallback: at least confirm the schema was accepted (use plain text to avoid markdown errors)
                await update.message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.state_transition_failed',
                                 error=error_msg, locale=locale)
                )
                return

            print(f"🔀 DEBUG: State transition SUCCESS: {old_state} → {session.current_state}")

            # Show confirmation
            await update.message.reply_text(
                LocalPathMessages.schema_accepted_deferred(
                    parsed_schema.target,
                    len(parsed_schema.features),
                    locale=locale
                ),
                parse_mode="Markdown"
            )

            # Immediately show model selection options
            await self._show_model_selection(update, context, session)

            # Stop handler propagation to prevent generic message handler from processing
            print("🛑 DEBUG: Stopping handler propagation after successful schema input")
            raise ApplicationHandlerStop

        except ApplicationHandlerStop:
            # Re-raise immediately - this is control flow, not an error
            raise

        except ValidationError as e:
            # Schema parsing failed - show error and ask to try again
            await update.message.reply_text(
                LocalPathMessages.schema_parse_error(str(e), locale=locale),
                parse_mode="Markdown"
            )

    async def _show_schema_confirmation_after_load(
        self,
        query,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        metadata: dict,
        schema: Optional[DatasetSchema]
    ) -> None:
        """Show detected schema confirmation (after immediate load)."""
        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Generate summary with schema info
        summary = self.data_loader.get_local_path_summary(
            session.uploaded_data,
            metadata,
            schema
        )

        # Create confirmation keyboard
        keyboard = [
            [InlineKeyboardButton("✅ Accept Schema", callback_data="schema:accept")],
            [InlineKeyboardButton("❌ Try Different File", callback_data="schema:reject")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Extract schema details for polished message
        suggested_target = schema.suggested_target if schema else None
        suggested_features = schema.suggested_features if schema else []
        task_type = schema.suggested_task_type if schema else None

        await query.edit_message_text(
            LocalPathMessages.schema_confirmation_prompt(
                summary, suggested_target, suggested_features, task_type, locale=locale
            ),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _show_schema_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        metadata: dict,
        schema: Optional[DatasetSchema]
    ) -> None:
        """Show detected schema and ask for confirmation."""
        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Generate summary with schema info
        summary = self.data_loader.get_local_path_summary(
            session.uploaded_data,
            metadata,
            schema
        )

        # Create confirmation keyboard
        keyboard = [
            [InlineKeyboardButton("✅ Accept Schema", callback_data="schema:accept")],
            [InlineKeyboardButton("❌ Try Different File", callback_data="schema:reject")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Extract schema details for polished message
        suggested_target = schema.suggested_target if schema else None
        suggested_features = schema.suggested_features if schema else []
        task_type = schema.suggested_task_type if schema else None

        await update.message.reply_text(
            LocalPathMessages.schema_confirmation_prompt(
                summary, suggested_target, suggested_features, task_type, locale=locale
            ),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _show_model_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show model selection menu with categorized options."""
        print("📊 DEBUG: Showing model selection menu")

        # Extract locale from session
        locale = session.language if session.language else None
        print(f"🌐 MODEL SELECTION: session.language={session.language}, extracted locale={locale}")

        # Create inline keyboard with i18n button labels
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflows.model_selection.regression_label', locale=locale),
                callback_data="model_category:regression"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflows.model_selection.classification_label', locale=locale),
                callback_data="model_category:classification"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflows.model_selection.neural_label', locale=locale),
                callback_data="model_category:neural"
            )]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Build message using i18n
        title = I18nManager.t('workflows.model_selection.title', locale=locale)
        description = I18nManager.t('workflows.model_selection.description', locale=locale)
        regression_desc = I18nManager.t('workflows.model_selection.regression_desc', locale=locale)
        classification_desc = I18nManager.t('workflows.model_selection.classification_desc', locale=locale)
        neural_desc = I18nManager.t('workflows.model_selection.neural_desc', locale=locale)
        which_category = I18nManager.t('workflows.model_selection.which_category', locale=locale)

        await update.message.reply_text(
            f"{title}\n\n"
            f"{description}\n\n"
            f"📈 **Regression**: {regression_desc}\n"
            f"🎯 **Classification**: {classification_desc}\n"
            f"🧠 **Neural Networks**: {neural_desc}\n\n"
            f"{which_category}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_model_category_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle model category selection and show specific models."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            category = query.data.split(":")[-1]
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_model_category_selection: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"📊 DEBUG: User selected model category: {category}")

        # Get session to access stored detection from schema processing
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None
        print(f"🌐 CATEGORY SELECTION: session.language={session.language}, extracted locale={locale}")

        # Save state snapshot BEFORE showing specific model selection (Phase 2: Back Button Fix)
        session.save_state_snapshot()
        self.logger.debug("📸 State snapshot saved before showing specific model selection")

        # Use previously detected task type from schema processing (deferred loading compatible)
        detected_task = session.selections.get('detected_task_type') if session else None

        if detected_task:
            print(f"🔍 DEBUG: Task type detected = {detected_task}")
        else:
            print(f"⚠️ DEBUG: No task type detected")

        # Model options with i18n keys
        model_option_keys = {
            "regression": [
                ('workflows.model_types.linear', 'linear'),
                ('workflows.model_types.ridge', 'ridge'),
                ('workflows.model_types.lasso', 'lasso'),
                ('workflows.model_types.elasticnet', 'elasticnet'),
                ('workflows.model_types.polynomial', 'polynomial'),
                ('workflows.model_types.xgboost_regression', 'xgboost_regression'),
                ('workflows.model_types.lightgbm_regression', 'lightgbm_regression'),
                ('workflows.model_types.catboost_regression', 'catboost_regression')
            ],
            "classification": [
                ('workflows.model_types.logistic', 'logistic'),
                ('workflows.model_types.decision_tree', 'decision_tree'),
                ('workflows.model_types.random_forest', 'random_forest'),
                ('workflows.model_types.gradient_boosting', 'gradient_boosting'),
                ('workflows.model_types.xgboost_binary_classification', 'xgboost_binary_classification'),
                ('workflows.model_types.lightgbm_binary_classification', 'lightgbm_binary_classification'),
                ('workflows.model_types.catboost_binary_classification', 'catboost_binary_classification'),
                ('workflows.model_types.svm', 'svm'),
                ('workflows.model_types.naive_bayes', 'naive_bayes')
            ],
            "neural": [
                ('workflows.model_types.mlp_regression', 'mlp_regression'),
                ('workflows.model_types.mlp_classification', 'mlp_classification'),
                ('workflows.model_types.keras_binary_classification', 'keras_binary_classification'),
                ('workflows.model_types.keras_multiclass_classification', 'keras_multiclass_classification'),
                ('workflows.model_types.keras_regression', 'keras_regression')
            ]
        }

        # Translate model names using i18n
        model_keys = model_option_keys.get(category, [])
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t(name_key, locale=locale),
                callback_data=f"model_select:{model_type}"
            )]
            for name_key, model_type in model_keys
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button (replaces manual back button)
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Translate category header
        category_header = I18nManager.t(
            f'workflows.category_headers.{category}',
            locale=locale
        )

        # Build message with detection info if available
        message = f"{category_header}\n\n"

        if detected_task:
            task_emoji = "📈" if detected_task == "regression" else "🎯"
            detected_msg = I18nManager.t(
                'workflows.category_headers.detected_task',
                locale=locale,
                task_type=detected_task.title()
            )
            all_models_note = I18nManager.t(
                'workflows.category_headers.all_models_note',
                locale=locale
            )
            message += f"{task_emoji} {detected_msg}\n"
            message += f"{all_models_note}\n\n"

        select_msg = I18nManager.t(
            'workflows.category_headers.select_specific',
            locale=locale
        )
        message += select_msg

        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_model_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle final model selection and start training."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            model_type = query.data.split(":")[-1]
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_model_selection: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"📊 DEBUG: User selected model: {model_type}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Store model selection
        session.selections['model_type'] = model_type

        # Determine task type based on model
        if model_type in ['linear', 'ridge', 'lasso', 'elasticnet', 'polynomial', 'mlp_regression', 'keras_regression', 'xgboost_regression', 'lightgbm_regression', 'catboost_regression']:
            session.selections['task_type'] = 'regression'
        elif model_type in ['logistic', 'decision_tree', 'random_forest', 'gradient_boosting', 'svm', 'naive_bayes', 'mlp_classification', 'keras_binary_classification', 'keras_multiclass_classification', 'xgboost_binary_classification', 'xgboost_multiclass_classification', 'lightgbm_binary_classification', 'lightgbm_multiclass_classification', 'catboost_binary_classification', 'catboost_multiclass_classification']:
            session.selections['task_type'] = 'classification'
        else:
            session.selections['task_type'] = 'neural_network'

        await self.state_manager.update_session(session)

        # Check if this is a Keras model - needs parameter configuration
        if model_type.startswith('keras_'):
            print(f"🧠 DEBUG: Keras model selected, starting parameter configuration")
            await self._start_keras_config(query, session)
        # Check if this is an XGBoost model - needs parameter configuration
        elif model_type.startswith('xgboost_'):
            print(f"🚀 DEBUG: XGBoost model selected, starting parameter configuration")
            await self._start_xgboost_config(query, session, model_type)
        # Check if this is a LightGBM model - needs parameter configuration
        elif model_type.startswith('lightgbm_'):
            print(f"⚡ DEBUG: LightGBM model selected, starting parameter configuration")
            await self._start_lightgbm_config(query, session, model_type)
        # Check if this is a CatBoost model - needs parameter configuration
        elif model_type.startswith('catboost_'):
            print(f"🐈 DEBUG: CatBoost model selected, starting parameter configuration")
            await self._start_catboost_config(query, session, model_type)
        else:
            # sklearn models - start training immediately
            # Escape underscores in model_type for Markdown
            model_display = model_type.replace('_', '\\_')
            await query.edit_message_text(
                f"✅ **Model Selected**: {model_display}\n\n"
                f"🚀 Starting training...\n\n"
                f"{I18nManager.t('workflow_state.training.patience', locale=locale)}",
                parse_mode="Markdown"
            )

            # Execute sklearn training
            await self._execute_sklearn_training(update, context, session, model_type)

    async def _start_keras_config(
        self,
        query,
        session
    ) -> None:
        """Start Keras parameter configuration workflow."""
        # Initialize Keras config dict
        session.selections['keras_config'] = {}
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Start with epochs configuration
        keyboard = [
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.epochs.btn_50', locale=locale), callback_data="keras_epochs:50")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.epochs.btn_100', locale=locale), callback_data="keras_epochs:100")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.epochs.btn_200', locale=locale), callback_data="keras_epochs:200")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.epochs.btn_custom', locale=locale), callback_data="keras_epochs:custom")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            f"{I18nManager.t('workflow_state.model_config.keras.header', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.epochs.title', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.epochs.question', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.model_config.keras.epochs.help', locale=locale)}"
        )

        await query.edit_message_text(
            message_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _start_xgboost_config(
        self,
        query,
        session,
        model_type: str
    ) -> None:
        """Start XGBoost parameter configuration workflow."""
        # Initialize XGBoost config dict with defaults from template
        from src.engines.trainers.xgboost_templates import get_template
        default_config = get_template(model_type)

        session.selections['xgboost_config'] = default_config
        session.selections['xgboost_model_type'] = model_type  # Store for later
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Start with n_estimators configuration
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.n_estimators.btn_50', locale=locale),
                callback_data="xgboost_n_estimators:50"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.n_estimators.btn_100', locale=locale),
                callback_data="xgboost_n_estimators:100"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.n_estimators.btn_200', locale=locale),
                callback_data="xgboost_n_estimators:200"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.n_estimators.btn_custom', locale=locale),
                callback_data="xgboost_n_estimators:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.xgboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.xgboost.n_estimators.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.xgboost.n_estimators.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.xgboost.n_estimators.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _start_lightgbm_config(
        self,
        query,
        session,
        model_type: str
    ) -> None:
        """Start LightGBM parameter configuration workflow."""
        # Initialize LightGBM config dict with defaults from template
        from src.engines.trainers.lightgbm_templates import get_template
        default_config = get_template(model_type)

        session.selections['lightgbm_config'] = default_config
        session.selections['lightgbm_model_type'] = model_type  # Store for later
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Start with num_leaves configuration (LightGBM uses leaves not depth)
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.btn_15', locale=locale),
                callback_data="lightgbm_num_leaves:15"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.btn_31', locale=locale),
                callback_data="lightgbm_num_leaves:31"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.btn_63', locale=locale),
                callback_data="lightgbm_num_leaves:63"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.btn_defaults', locale=locale),
                callback_data="lightgbm_use_defaults"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.lightgbm.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.title', locale=locale)
        description = I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.description', locale=locale)
        question = I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.question', locale=locale)
        tip = I18nManager.t('workflow_state.model_config.lightgbm.num_leaves.tip', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{title}\n\n"
            f"{description}\n"
            f"{question}\n\n"
            f"{tip}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _start_catboost_config(
        self,
        query,
        session,
        model_type: str
    ) -> None:
        """Start CatBoost parameter configuration workflow."""
        # Initialize CatBoost config dict with defaults from template
        from src.engines.trainers.catboost_templates import get_template
        default_config = get_template(model_type)

        session.selections['catboost_config'] = default_config
        session.selections['catboost_model_type'] = model_type  # Store for later
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Start with iterations configuration
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.iterations.btn_100', locale=locale),
                callback_data="catboost_iterations:100"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.iterations.btn_500', locale=locale),
                callback_data="catboost_iterations:500"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.iterations.btn_1000', locale=locale),
                callback_data="catboost_iterations:1000"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.iterations.btn_2000', locale=locale),
                callback_data="catboost_iterations:2000"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.iterations.btn_custom', locale=locale),
                callback_data="catboost_iterations:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.catboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.catboost.iterations.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.catboost.iterations.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.catboost.iterations.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _execute_sklearn_training(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        model_type: str
    ) -> None:
        """Execute training for sklearn and XGBoost models."""
        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in _execute_sklearn_training: {e}")
            return

        print(f"🚀 DEBUG: Starting sklearn/XGBoost training for model={model_type}")

        # Transition to TRAINING state
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.TRAINING.value
        )

        if not success:
            logger.error(f"Failed to transition to TRAINING: {error_msg}")
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    f"❌ **State Transition Error**\n\n{error_msg}\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            return

        print(f"✅ DEBUG: State transition to TRAINING successful, state={session.current_state}")

        # Trigger actual sklearn/XGBoost training
        try:
            # Extract training parameters from session
            target = session.selections.get('target_column')
            features = session.selections.get('feature_columns')
            file_path = session.file_path
            task_type = session.selections.get('task_type', 'classification')

            print(f"🚀 DEBUG: Training params - target={target}, features={features}, model={model_type}")
            print(f"🚀 DEBUG: File path={file_path}, task_type={task_type}")

            # Get hyperparameters based on model type
            hyperparameters = None
            if model_type.startswith('xgboost_'):
                # Check if user configured parameters or use defaults
                xgboost_config = session.selections.get('xgboost_config')
                if xgboost_config:
                    # User selected custom parameters
                    hyperparameters = xgboost_config
                    print(f"🚀 DEBUG: Using user-configured XGBoost parameters: {hyperparameters}")
                else:
                    # Use XGBoost template defaults
                    from src.engines.trainers.xgboost_templates import get_template
                    hyperparameters = get_template(model_type)
                    print(f"🚀 DEBUG: Using XGBoost template hyperparameters: {hyperparameters}")
            elif model_type.startswith('lightgbm_'):
                # Check if user configured parameters or use defaults
                lightgbm_config = session.selections.get('lightgbm_config')
                if lightgbm_config:
                    # User selected custom parameters
                    hyperparameters = lightgbm_config
                    print(f"⚡ DEBUG: Using user-configured LightGBM parameters: {hyperparameters}")
                else:
                    # Use LightGBM template defaults
                    from src.engines.trainers.lightgbm_templates import get_template
                    hyperparameters = get_template(model_type)
                    print(f"⚡ DEBUG: Using LightGBM template hyperparameters: {hyperparameters}")

            # Check data source to determine execution path
            # local_path files exist on user's machine → must route through JobQueue to worker
            # uploaded files are available on server → can execute directly
            if session.data_source == "local_path":
                print(f"🔍 DEBUG: Routing sklearn training through JobQueue (local_path)")
                # Route through JobQueue to worker (file is on user's machine)
                result = await self._execute_training_on_worker(
                    user_id=user_id,
                    file_path=file_path,
                    task_type=task_type,
                    model_type=model_type,
                    target_column=target,
                    feature_columns=features,
                    hyperparameters=hyperparameters,
                    test_size=0.2,
                    context=context
                )
            else:
                # Direct execution for uploaded files (existing code path)
                print(f"🔍 DEBUG: Executing sklearn training directly (uploaded file)")
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    lambda: self.ml_engine.train_model(
                        file_path=file_path,  # Lazy loading from deferred file
                        task_type=task_type,
                        model_type=model_type,
                        target_column=target,
                        feature_columns=features,
                        user_id=user_id,
                        hyperparameters=hyperparameters,
                        test_size=0.2  # Default test split
                    )
                )

            print(f"🚀 DEBUG: Training result = {result}")

            # Send success message with metrics and transition to naming workflow
            if result.get('success'):
                model_id = result.get('model_id', 'N/A')
                metrics = result.get('metrics', {})
                dataset_stats = result.get('dataset_stats', {})
                model_info = result.get('model_info', {})  # For local worker training

                # Transition to TRAINING_COMPLETE state
                print(f"🔍 DEBUG: Transitioning to TRAINING_COMPLETE state")
                success, error_msg, _ = await self.state_manager.transition_state(
                    session,
                    MLTrainingState.TRAINING_COMPLETE.value
                )

                if not success:
                    logger.error(f"Failed to transition to TRAINING_COMPLETE: {error_msg}")
                    await update.effective_message.reply_text(
                        f"❌ **State Transition Error**\n\n{error_msg}",
                        parse_mode="Markdown"
                    )
                    return

                print(f"✅ DEBUG: State transition to TRAINING_COMPLETE successful, state={session.current_state}")

                # Store model_id and model_info in session for naming workflow
                session.selections['pending_model_id'] = model_id
                session.selections['pending_model_info'] = model_info  # For local worker: use instead of filesystem lookup
                await self.state_manager.update_session(session)

                # Get locale from session
                locale = session.language if session.language else None

                # Format metrics based on task type (pass dataset_stats for display)
                metrics_text = self._format_sklearn_metrics(metrics, task_type, locale, dataset_stats)

                # Show naming options with inline keyboard
                keyboard = [
                    [InlineKeyboardButton(
                        I18nManager.t('workflow_state.training.completion.buttons.name_model', locale=locale),
                        callback_data="name_model"
                    )],
                    [InlineKeyboardButton(
                        I18nManager.t('workflow_state.training.completion.buttons.skip_naming', locale=locale),
                        callback_data="skip_naming"
                    )],
                    [InlineKeyboardButton(
                        I18nManager.t('templates.save.button', locale=locale, default="💾 Save as Template"),
                        callback_data="save_as_template"
                    )]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Escape underscores for Markdown display
                model_display = model_type.replace('_', '\\_')

                await update.effective_message.reply_text(
                    f"{I18nManager.t('workflow_state.training.completion.header', locale=locale)}\n\n"
                    f"{I18nManager.t('workflow_state.training.completion.model_label', locale=locale)}: {model_display}\n"
                    f"{I18nManager.t('workflow_state.training.completion.model_id_label', locale=locale)}: `{model_id}`\n\n"
                    f"{metrics_text}\n\n"
                    f"{I18nManager.t('workflow_state.training.completion.naming_prompt', locale=locale)}",
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            else:
                # Training failed
                error_msg = result.get('error', 'Unknown error during training')
                logger.error(f"Training failed: {error_msg}")
                await update.effective_message.reply_text(
                    f"❌ **Training Failed**\n\n"
                    f"Error: {escape_markdown_v1(str(error_msg))}\n\n"
                    f"Please try again with /train",
                    parse_mode="Markdown"
                )

        except TrainingError as e:
            # Build complete error message including hidden details
            error_msg = str(e)
            if hasattr(e, 'error_details') and e.error_details:
                error_msg += f"\n\nDetails: {e.error_details}"

            logger.error(f"Training error: {e}", exc_info=True)
            await update.effective_message.reply_text(
                f"❌ **Training Error**\n\n{escape_markdown_v1(error_msg)}\n\nPlease try again with /train",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error during sklearn/XGBoost training: {e}", exc_info=True)
            await update.effective_message.reply_text(
                f"❌ **Training Error**\n\n"
                f"An unexpected error occurred: {escape_markdown_v1(str(e))}\n\n"
                f"Please try again with /train",
                parse_mode="Markdown"
            )

    def _format_sklearn_metrics(self, metrics: dict, task_type: str, locale: Optional[str] = None, dataset_stats: dict = None) -> str:
        """Format sklearn/XGBoost metrics for user display."""
        header = I18nManager.t('workflow_state.training.metrics.performance_header', locale=locale)
        output_lines = []

        # Dataset stats section (if available)
        if dataset_stats:
            output_lines.append("📊 Dataset:")
            output_lines.append(f"• Rows: {dataset_stats.get('n_rows', 'N/A')}")

            if 'class_distribution' in dataset_stats:
                dist = dataset_stats['class_distribution']
                dist_str = ", ".join(f"Class {k}: {v['count']} ({v['pct']}%)" for k, v in sorted(dist.items()))
                output_lines.append(f"• Classes: {dist_str}")
            elif 'quartiles' in dataset_stats:
                q = dataset_stats['quartiles']
                output_lines.append(f"• Target: Q1={q['q1']:.2f}, Median={q['median']:.2f}, Q3={q['q3']:.2f}")

            output_lines.append("")  # Blank line before metrics

        if task_type == 'regression':
            # Regression metrics
            mse = metrics.get('mse', 'N/A')
            rmse = metrics.get('rmse', 'N/A')
            mae = metrics.get('mae', 'N/A')
            r2 = metrics.get('r2', 'N/A')

            # Format values before f-string
            r2_str = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else str(rmse)
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
            mse_str = f"{mse:.4f}" if isinstance(mse, float) else str(mse)

            output_lines.extend([
                header,
                f"• R² Score: {r2_str}",
                f"• RMSE: {rmse_str}",
                f"• MAE: {mae_str}",
                f"• MSE: {mse_str}"
            ])
            return "\n".join(output_lines)
        else:
            # Classification metrics (priority order: probability-based first)
            roc_auc = metrics.get('roc_auc')
            auc_pr = metrics.get('auc_pr')
            brier_score = metrics.get('brier_score')
            log_loss_val = metrics.get('log_loss')
            f1 = metrics.get('f1', 'N/A')
            accuracy = metrics.get('accuracy', 'N/A')
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')

            output_lines.append(header)

            # Probability-based metrics first (if available)
            if roc_auc is not None:
                output_lines.append(f"• AUC-ROC: {roc_auc:.4f}")
            if auc_pr is not None:
                output_lines.append(f"• AUC-PR: {auc_pr:.4f}")
            if brier_score is not None:
                output_lines.append(f"• Brier Score: {brier_score:.4f}")
            if log_loss_val is not None:
                output_lines.append(f"• Log Loss: {log_loss_val:.4f}")

            # Standard metrics
            output_lines.append(f"• F1 Score: {f1:.4f}" if isinstance(f1, float) else f"• F1 Score: {f1}")
            output_lines.append(f"• Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"• Accuracy: {accuracy}")
            output_lines.append(f"• Precision: {precision:.4f}" if isinstance(precision, float) else f"• Precision: {precision}")
            output_lines.append(f"• Recall: {recall:.4f}" if isinstance(recall, float) else f"• Recall: {recall}")

            return "\n".join(output_lines)

    async def handle_keras_epochs(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Keras epochs selection."""
        query = update.callback_query

        # Wrap Telegram API call with error handling
        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"Failed to answer callback in handle_keras_epochs: {e}")
            # Continue - not critical

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            epochs_value = query.data.split(":")[-1]
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_keras_epochs: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_epochs - user={user_id}, epochs_value={epochs_value}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Defensive check: keras_config exists (initialize if missing)
        if 'keras_config' not in session.selections:
            print(f"⚠️  WARNING: keras_config missing, initializing for user {user_id}")
            session.selections['keras_config'] = {}

        # Store epochs value
        if epochs_value == "custom":
            # TODO: Handle custom epochs input
            session.selections['keras_config']['epochs'] = 100  # Default for now
        else:
            session.selections['keras_config']['epochs'] = int(epochs_value)

        print(f"📦 DEBUG: keras_config after epochs = {session.selections['keras_config']}")

        # Get locale from session
        locale = session.language if session.language else None

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed_short', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Get locale from session
        locale = session.language if session.language else None

        # Move to batch size configuration
        keyboard = [
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.batch_size.btn_16', locale=locale), callback_data="keras_batch:16")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.batch_size.btn_32', locale=locale), callback_data="keras_batch:32")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.batch_size.btn_64', locale=locale), callback_data="keras_batch:64")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.batch_size.btn_128', locale=locale), callback_data="keras_batch:128")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            f"{I18nManager.t('workflow_state.model_config.keras.header', locale=locale)}\n\n"
            f"✅ Epochs: {session.selections['keras_config']['epochs']}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.batch_size.title', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.batch_size.question', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.model_config.keras.batch_size.help', locale=locale)}"
        )

        # Wrap message editing with error handling
        try:
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            logger.error(f"Failed to edit message in handle_keras_epochs: {e}")
            # Fallback: send new message
            await update.effective_message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Telegram API error in handle_keras_epochs: {e}")
            await update.effective_message.reply_text(
                I18nManager.t('workflow_state.training.errors.message_update_error', locale=locale),
                parse_mode="Markdown"
            )

    async def handle_keras_batch(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Keras batch size selection."""
        query = update.callback_query

        # Wrap Telegram API call with error handling
        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"Failed to answer callback in handle_keras_batch: {e}")
            # Continue - not critical

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            batch_size = int(query.data.split(":")[-1])
        except (AttributeError, ValueError) as e:
            logger.error(f"Malformed update object in handle_keras_batch: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_batch - user={user_id}, batch_size={batch_size}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Get locale from session
        locale = session.language if session.language else None

        # Defensive check: keras_config exists
        if 'keras_config' not in session.selections:
            print(f"❌ ERROR: keras_config missing in handle_keras_batch for user {user_id}")
            try:
                await query.edit_message_text(
                    I18nManager.t('workflow_state.training.errors.config_data_lost', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflow_state.training.errors.config_data_lost_short', locale=locale),
                    parse_mode="Markdown"
                )
            return

        session.selections['keras_config']['batch_size'] = batch_size
        print(f"📦 DEBUG: keras_config after batch = {session.selections['keras_config']}")

        # Get locale from session
        locale = session.language if session.language else None

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed_short', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Get epochs with default fallback
        epochs = session.selections['keras_config'].get('epochs', 100)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to kernel initializer configuration
        keyboard = [
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.weight_init.btn_glorot', locale=locale), callback_data="keras_init:glorot_uniform")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.weight_init.btn_random_normal', locale=locale), callback_data="keras_init:random_normal")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.weight_init.btn_random_uniform', locale=locale), callback_data="keras_init:random_uniform")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.weight_init.btn_he_normal', locale=locale), callback_data="keras_init:he_normal")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.weight_init.btn_he_uniform', locale=locale), callback_data="keras_init:he_uniform")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            f"{I18nManager.t('workflow_state.model_config.keras.header', locale=locale)}\n\n"
            f"✅ Epochs: {epochs}\n"
            f"✅ Batch Size: {batch_size}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.weight_init.title', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.weight_init.question', locale=locale)}"
        )

        # Wrap message editing with error handling
        try:
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            logger.error(f"Failed to edit message in handle_keras_batch: {e}")
            # Fallback: send new message
            await update.effective_message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Telegram API error in handle_keras_batch: {e}")
            await update.effective_message.reply_text(
                "❌ **Error Updating Message**\n\nPlease use /train to restart.",
                parse_mode="Markdown"
            )

    async def handle_keras_initializer(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Keras kernel initializer selection."""
        query = update.callback_query

        # Wrap Telegram API call with error handling
        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"Failed to answer callback in handle_keras_initializer: {e}")
            # Continue - not critical

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            initializer = query.data.split(":")[-1]
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_keras_initializer: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_initializer - user={user_id}, initializer={initializer}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Defensive check: keras_config exists
        if 'keras_config' not in session.selections:
            print(f"❌ ERROR: keras_config missing in handle_keras_initializer for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Error**\n\n"
                    "Configuration data lost. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Error**\n\nPlease start over with /train",
                    parse_mode="Markdown"
                )
            return

        session.selections['keras_config']['kernel_initializer'] = initializer
        print(f"📦 DEBUG: keras_config after initializer = {session.selections['keras_config']}")

        # Get locale from session
        locale = session.language if session.language else None

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed_short', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Get previous values with default fallbacks
        epochs = session.selections['keras_config'].get('epochs', 100)
        batch_size = session.selections['keras_config'].get('batch_size', 32)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to verbose configuration
        keyboard = [
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.verbosity.btn_0', locale=locale), callback_data="keras_verbose:0")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.verbosity.btn_1', locale=locale), callback_data="keras_verbose:1")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.verbosity.btn_2', locale=locale), callback_data="keras_verbose:2")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            f"{I18nManager.t('workflow_state.model_config.keras.header', locale=locale)}\n\n"
            f"✅ Epochs: {epochs}\n"
            f"✅ Batch Size: {batch_size}\n"
            f"✅ Initializer: {escape_markdown_v1(initializer)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.verbosity.title', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.verbosity.question', locale=locale)}"
        )

        # Wrap message editing with error handling (CRITICAL FIX FOR BUTTON ERROR!)
        try:
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            logger.error(f"Failed to edit message in handle_keras_initializer: {e}")
            # Fallback: send new message
            await update.effective_message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Telegram API error in handle_keras_initializer: {e}")
            await update.effective_message.reply_text(
                "❌ **Error Updating Message**\n\nPlease use /train to restart.",
                parse_mode="Markdown"
            )

    async def handle_keras_verbose(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Keras verbose selection."""
        query = update.callback_query

        # Wrap Telegram API call with error handling
        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"Failed to answer callback in handle_keras_verbose: {e}")
            # Continue - not critical

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            verbose = int(query.data.split(":")[-1])
        except (AttributeError, ValueError) as e:
            logger.error(f"Malformed update object in handle_keras_verbose: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_verbose - user={user_id}, verbose={verbose}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Defensive check: keras_config exists
        if 'keras_config' not in session.selections:
            print(f"❌ ERROR: keras_config missing in handle_keras_verbose for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Error**\n\n"
                    "Configuration data lost. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Error**\n\nPlease start over with /train",
                    parse_mode="Markdown"
                )
            return

        session.selections['keras_config']['verbose'] = verbose
        print(f"📦 DEBUG: keras_config after verbose = {session.selections['keras_config']}")

        # Get locale from session
        locale = session.language if session.language else None

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed_short', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Get previous values with default fallbacks
        epochs = session.selections['keras_config'].get('epochs', 100)
        batch_size = session.selections['keras_config'].get('batch_size', 32)
        initializer = session.selections['keras_config'].get('kernel_initializer', 'glorot_uniform')

        # Get locale from session
        locale = session.language if session.language else None

        # Move to validation split configuration
        keyboard = [
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.validation.btn_0', locale=locale), callback_data="keras_val:0.0")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.validation.btn_10', locale=locale), callback_data="keras_val:0.1")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.validation.btn_20', locale=locale), callback_data="keras_val:0.2")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.model_config.keras.validation.btn_30', locale=locale), callback_data="keras_val:0.3")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            f"{I18nManager.t('workflow_state.model_config.keras.header', locale=locale)}\n\n"
            f"✅ Epochs: {epochs}\n"
            f"✅ Batch Size: {batch_size}\n"
            f"✅ Initializer: {escape_markdown_v1(initializer)}\n"
            f"✅ Verbosity: {verbose}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.validation.title', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.model_config.keras.validation.question', locale=locale)}"
        )

        # Wrap message editing with error handling
        try:
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            logger.error(f"Failed to edit message in handle_keras_verbose: {e}")
            # Fallback: send new message
            await update.effective_message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Telegram API error in handle_keras_verbose: {e}")
            await update.effective_message.reply_text(
                "❌ **Error Updating Message**\n\nPlease use /train to restart.",
                parse_mode="Markdown"
            )

    async def handle_keras_validation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Keras validation split selection and start training."""
        query = update.callback_query

        # Wrap Telegram API call with error handling
        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"Failed to answer callback in handle_keras_validation: {e}")
            # Continue - not critical

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            validation_split = float(query.data.split(":")[-1])
        except (AttributeError, ValueError) as e:
            logger.error(f"Malformed update object in handle_keras_validation: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_validation - user={user_id}, validation_split={validation_split}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Defensive check: keras_config exists
        if 'keras_config' not in session.selections:
            print(f"❌ ERROR: keras_config missing in handle_keras_validation for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Error**\n\n"
                    "Configuration data lost. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Error**\n\nPlease start over with /train",
                    parse_mode="Markdown"
                )
            return

        session.selections['keras_config']['validation_split'] = validation_split
        print(f"📦 DEBUG: keras_config final = {session.selections['keras_config']}")

        # Get locale from session
        locale = session.language if session.language else None

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed', locale=locale),
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    I18nManager.t('workflow_state.training.errors.config_save_failed_short', locale=locale),
                    parse_mode="Markdown"
                )
            return

        # Get all configuration with defaults
        config = session.selections['keras_config']
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 32)
        initializer = config.get('kernel_initializer', 'glorot_uniform')
        verbose = config.get('verbose', 1)

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        # Get i18n strings
        title = I18nManager.t('workflow_state.model_config.keras.configuration_complete.title', locale=locale)
        settings_header = I18nManager.t('workflow_state.model_config.keras.configuration_complete.settings_header', locale=locale)
        what_next = I18nManager.t('workflow_state.model_config.keras.configuration_complete.what_next', locale=locale)

        btn_train = I18nManager.t('workflow_state.model_config.keras.configuration_complete.buttons.start_training', locale=locale)
        btn_save = I18nManager.t('workflow_state.model_config.keras.configuration_complete.buttons.save_template', locale=locale)

        # Offer to save as template or start training
        keyboard = [
            [InlineKeyboardButton(btn_train, callback_data="start_training")],
            [InlineKeyboardButton(btn_save, callback_data="template_save")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            f"{title}\n\n"
            f"{settings_header}\n"
            f"• Epochs: {epochs}\n"
            f"• Batch Size: {batch_size}\n"
            f"• Initializer: {escape_markdown_v1(initializer)}\n"
            f"• Verbosity: {verbose}\n"
            f"• Validation Split: {validation_split * 100:.0f}%\n\n"
            f"{what_next}"
        )

        # Wrap message editing with error handling
        try:
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            logger.error(f"Failed to edit message in handle_keras_validation: {e}")
            # Fallback: send new message
            await update.effective_message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Telegram API error in handle_keras_validation: {e}")
            await update.effective_message.reply_text(
                "❌ **Error Updating Message**\n\nPlease use /train to restart.",
                parse_mode="Markdown"
            )

    async def handle_xgboost_n_estimators(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle XGBoost n_estimators selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        n_estimators_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Handle custom input (default to 100 for now - Phase 2 enhancement)
        if n_estimators_value == "custom":
            n_estimators = 100
        else:
            n_estimators = int(n_estimators_value)

        session.selections['xgboost_config']['n_estimators'] = n_estimators
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to max_depth selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.max_depth.btn_3', locale=locale),
                callback_data="xgboost_max_depth:3"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.max_depth.btn_6', locale=locale),
                callback_data="xgboost_max_depth:6"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.max_depth.btn_9', locale=locale),
                callback_data="xgboost_max_depth:9"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.max_depth.btn_custom', locale=locale),
                callback_data="xgboost_max_depth:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.xgboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.xgboost.max_depth.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.xgboost.max_depth.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.xgboost.max_depth.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n{title}\n\n{question}\n{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_xgboost_max_depth(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle XGBoost max_depth selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        max_depth_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        if max_depth_value == "custom":
            max_depth = 6
        else:
            max_depth = int(max_depth_value)

        session.selections['xgboost_config']['max_depth'] = max_depth
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to learning_rate selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.learning_rate.btn_001', locale=locale),
                callback_data="xgboost_learning_rate:0.01"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.learning_rate.btn_01', locale=locale),
                callback_data="xgboost_learning_rate:0.1"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.learning_rate.btn_03', locale=locale),
                callback_data="xgboost_learning_rate:0.3"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.learning_rate.btn_custom', locale=locale),
                callback_data="xgboost_learning_rate:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.xgboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.xgboost.learning_rate.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.xgboost.learning_rate.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.xgboost.learning_rate.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n{title}\n\n{question}\n{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_xgboost_learning_rate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle XGBoost learning_rate selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        learning_rate_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        if learning_rate_value == "custom":
            learning_rate = 0.1
        else:
            learning_rate = float(learning_rate_value)

        session.selections['xgboost_config']['learning_rate'] = learning_rate
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to subsample selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.subsample.btn_06', locale=locale),
                callback_data="xgboost_subsample:0.6"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.subsample.btn_08', locale=locale),
                callback_data="xgboost_subsample:0.8"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.subsample.btn_10', locale=locale),
                callback_data="xgboost_subsample:1.0"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.subsample.btn_custom', locale=locale),
                callback_data="xgboost_subsample:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.xgboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.xgboost.subsample.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.xgboost.subsample.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.xgboost.subsample.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n{title}\n\n{question}\n{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_xgboost_subsample(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle XGBoost subsample selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        subsample_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        if subsample_value == "custom":
            subsample = 0.8
        else:
            subsample = float(subsample_value)

        session.selections['xgboost_config']['subsample'] = subsample
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to colsample_bytree selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.colsample.btn_06', locale=locale),
                callback_data="xgboost_colsample:0.6"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.colsample.btn_08', locale=locale),
                callback_data="xgboost_colsample:0.8"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.colsample.btn_10', locale=locale),
                callback_data="xgboost_colsample:1.0"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.xgboost.colsample.btn_custom', locale=locale),
                callback_data="xgboost_colsample:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.xgboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.xgboost.colsample.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.xgboost.colsample.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.xgboost.colsample.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n{title}\n\n{question}\n{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_xgboost_colsample(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle XGBoost colsample_bytree selection and start training."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        colsample_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        if colsample_value == "custom":
            colsample = 0.8
        else:
            colsample = float(colsample_value)

        session.selections['xgboost_config']['colsample_bytree'] = colsample
        await self.state_manager.update_session(session)

        # Get locale from session
        locale = session.language if session.language else None

        # Configuration complete - start training
        model_type = session.selections.get('xgboost_model_type')
        xgboost_config = session.selections.get('xgboost_config')

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        title = I18nManager.t('workflow_state.model_config.xgboost.configuration_complete.title', locale=locale)
        model_label = I18nManager.t('workflow_state.model_config.xgboost.configuration_complete.model_label', locale=locale)
        parameters_label = I18nManager.t('workflow_state.model_config.xgboost.configuration_complete.parameters_label', locale=locale)
        param_n_estimators = I18nManager.t(
            'workflow_state.model_config.xgboost.configuration_complete.param_n_estimators',
            locale=locale,
            value=xgboost_config['n_estimators']
        )
        param_max_depth = I18nManager.t(
            'workflow_state.model_config.xgboost.configuration_complete.param_max_depth',
            locale=locale,
            value=xgboost_config['max_depth']
        )
        param_learning_rate = I18nManager.t(
            'workflow_state.model_config.xgboost.configuration_complete.param_learning_rate',
            locale=locale,
            value=xgboost_config['learning_rate']
        )
        param_subsample = I18nManager.t(
            'workflow_state.model_config.xgboost.configuration_complete.param_subsample',
            locale=locale,
            value=xgboost_config['subsample']
        )
        param_colsample = I18nManager.t(
            'workflow_state.model_config.xgboost.configuration_complete.param_colsample_bytree',
            locale=locale,
            value=xgboost_config['colsample_bytree']
        )
        starting_training = I18nManager.t('workflow_state.model_config.xgboost.configuration_complete.starting_training', locale=locale)
        patience = I18nManager.t('workflow_state.training.patience', locale=locale)

        await query.edit_message_text(
            f"{title}\n\n"
            f"{model_label} {model_display}\n"
            f"{parameters_label}\n"
            f"{param_n_estimators}\n"
            f"{param_max_depth}\n"
            f"{param_learning_rate}\n"
            f"{param_subsample}\n"
            f"{param_colsample}\n\n"
            f"{starting_training}\n\n"
            f"{patience}",
            parse_mode="Markdown"
        )

        # Execute training with XGBoost config
        await self._execute_sklearn_training(update, context, session, model_type)

    async def handle_lightgbm_num_leaves(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle LightGBM num_leaves selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        num_leaves_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        num_leaves = int(num_leaves_value)
        session.selections['lightgbm_config']['num_leaves'] = num_leaves
        await self.state_manager.update_session(session)

        print(f"⚡ DEBUG: LightGBM num_leaves set to {num_leaves}")

        # Get locale from session
        locale = session.language if session.language else None

        # Move to n_estimators selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.btn_50', locale=locale),
                callback_data="lightgbm_n_estimators:50"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.btn_100', locale=locale),
                callback_data="lightgbm_n_estimators:100"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.btn_200', locale=locale),
                callback_data="lightgbm_n_estimators:200"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.btn_defaults', locale=locale),
                callback_data="lightgbm_use_defaults"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.lightgbm.header', locale=locale)
        status_leaves = I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.status_num_leaves', locale=locale, value=num_leaves)
        title = I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.lightgbm.n_estimators.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{status_leaves}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_lightgbm_n_estimators(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle LightGBM n_estimators selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        n_estimators_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        n_estimators = int(n_estimators_value)
        session.selections['lightgbm_config']['n_estimators'] = n_estimators
        await self.state_manager.update_session(session)

        print(f"⚡ DEBUG: LightGBM n_estimators set to {n_estimators}")

        # Get previous values for display
        num_leaves = session.selections['lightgbm_config'].get('num_leaves', 31)

        # Get locale from session
        locale = session.language if session.language else None

        # Move to learning_rate selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.btn_001', locale=locale),
                callback_data="lightgbm_learning_rate:0.01"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.btn_005', locale=locale),
                callback_data="lightgbm_learning_rate:0.05"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.btn_01', locale=locale),
                callback_data="lightgbm_learning_rate:0.1"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.btn_02', locale=locale),
                callback_data="lightgbm_learning_rate:0.2"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.btn_defaults', locale=locale),
                callback_data="lightgbm_use_defaults"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.lightgbm.header', locale=locale)
        status_leaves = I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.status_num_leaves', locale=locale, value=num_leaves)
        status_estimators = I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.status_n_estimators', locale=locale, value=n_estimators)
        title = I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.lightgbm.learning_rate.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{status_leaves}\n"
            f"{status_estimators}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_lightgbm_learning_rate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle LightGBM learning_rate selection and start training."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        learning_rate_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        learning_rate = float(learning_rate_value)
        session.selections['lightgbm_config']['learning_rate'] = learning_rate
        await self.state_manager.update_session(session)

        print(f"⚡ DEBUG: LightGBM learning_rate set to {learning_rate}")

        # Get locale from session
        locale = session.language if session.language else None

        # Configuration complete - start training
        model_type = session.selections.get('lightgbm_model_type')
        lightgbm_config = session.selections.get('lightgbm_config')

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        title = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.title', locale=locale)
        model_label = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.model_label', locale=locale)
        params_label = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.parameters_label', locale=locale)
        param_num_leaves = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_num_leaves', locale=locale, value=lightgbm_config['num_leaves'])
        param_n_estimators = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_n_estimators', locale=locale, value=lightgbm_config['n_estimators'])
        param_learning_rate = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_learning_rate', locale=locale, value=lightgbm_config['learning_rate'])
        param_feature_fraction = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_feature_fraction', locale=locale, value=lightgbm_config.get('feature_fraction', 0.8))
        param_bagging_fraction = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_bagging_fraction', locale=locale, value=lightgbm_config.get('bagging_fraction', 0.8))
        starting_training = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.starting_training', locale=locale)
        patience = I18nManager.t('workflow_state.training.patience', locale=locale)

        await query.edit_message_text(
            f"{title}\n\n"
            f"{model_label} {model_display}\n"
            f"{params_label}\n"
            f"{param_num_leaves}\n"
            f"{param_n_estimators}\n"
            f"{param_learning_rate}\n"
            f"{param_feature_fraction}\n"
            f"{param_bagging_fraction}\n\n"
            f"{starting_training}\n\n"
            f"{patience}",
            parse_mode="Markdown"
        )

        # Execute training with LightGBM config
        await self._execute_sklearn_training(update, context, session, model_type)

    async def handle_lightgbm_use_defaults(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle 'Use Defaults' button - skip configuration and start training."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        # Use the default config that was already loaded
        model_type = session.selections.get('lightgbm_model_type')
        lightgbm_config = session.selections.get('lightgbm_config')

        print(f"⚡ DEBUG: Using LightGBM defaults: {lightgbm_config}")

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        title = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.title', locale=locale)
        model_label = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.model_label', locale=locale)
        params_label = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.parameters_label', locale=locale)
        param_num_leaves = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_num_leaves', locale=locale, value=lightgbm_config['num_leaves'])
        param_n_estimators = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_n_estimators', locale=locale, value=lightgbm_config['n_estimators'])
        param_learning_rate = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_learning_rate', locale=locale, value=lightgbm_config['learning_rate'])
        param_feature_fraction = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_feature_fraction', locale=locale, value=lightgbm_config.get('feature_fraction', 0.8))
        param_bagging_fraction = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.param_bagging_fraction', locale=locale, value=lightgbm_config.get('bagging_fraction', 0.8))
        starting_training = I18nManager.t('workflow_state.model_config.lightgbm.configuration_complete.starting_training', locale=locale)
        patience = I18nManager.t('workflow_state.training.patience', locale=locale)

        await query.edit_message_text(
            f"{title}\n\n"
            f"{model_label} {model_display}\n"
            f"{params_label}\n"
            f"{param_num_leaves}\n"
            f"{param_n_estimators}\n"
            f"{param_learning_rate}\n"
            f"{param_feature_fraction}\n"
            f"{param_bagging_fraction}\n\n"
            f"{starting_training}\n\n"
            f"{patience}",
            parse_mode="Markdown"
        )

        # Execute training with default LightGBM config
        await self._execute_sklearn_training(update, context, session, model_type)

    async def handle_catboost_iterations(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle CatBoost iterations selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        iterations_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        # Handle custom input (default to 1000 for now - Phase 2 enhancement)
        if iterations_value == "custom":
            iterations = 1000
        else:
            iterations = int(iterations_value)

        session.selections['catboost_config']['iterations'] = iterations
        await self.state_manager.update_session(session)

        # Move to depth selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.depth.btn_4', locale=locale),
                callback_data="catboost_depth:4"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.depth.btn_6', locale=locale),
                callback_data="catboost_depth:6"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.depth.btn_8', locale=locale),
                callback_data="catboost_depth:8"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.depth.btn_custom', locale=locale),
                callback_data="catboost_depth:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.catboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.catboost.depth.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.catboost.depth.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.catboost.depth.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_catboost_depth(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle CatBoost depth selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        depth_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        if depth_value == "custom":
            depth = 6
        else:
            depth = int(depth_value)

        session.selections['catboost_config']['depth'] = depth
        await self.state_manager.update_session(session)

        # Move to learning_rate selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.learning_rate.btn_001', locale=locale),
                callback_data="catboost_learning_rate:0.01"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.learning_rate.btn_003', locale=locale),
                callback_data="catboost_learning_rate:0.03"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.learning_rate.btn_01', locale=locale),
                callback_data="catboost_learning_rate:0.1"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.learning_rate.btn_custom', locale=locale),
                callback_data="catboost_learning_rate:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.catboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.catboost.learning_rate.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.catboost.learning_rate.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.catboost.learning_rate.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_catboost_learning_rate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle CatBoost learning_rate selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        learning_rate_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        if learning_rate_value == "custom":
            learning_rate = 0.03
        else:
            learning_rate = float(learning_rate_value)

        session.selections['catboost_config']['learning_rate'] = learning_rate
        await self.state_manager.update_session(session)

        # Move to l2_leaf_reg selection
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.btn_1', locale=locale),
                callback_data="catboost_l2:1"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.btn_3', locale=locale),
                callback_data="catboost_l2:3"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.btn_5', locale=locale),
                callback_data="catboost_l2:5"
            )],
            [InlineKeyboardButton(
                I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.btn_custom', locale=locale),
                callback_data="catboost_l2:custom"
            )]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        header = I18nManager.t('workflow_state.model_config.catboost.header', locale=locale)
        title = I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.title', locale=locale)
        question = I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.question', locale=locale)
        help_text = I18nManager.t('workflow_state.model_config.catboost.l2_leaf_reg.help', locale=locale)

        await query.edit_message_text(
            f"{header}\n\n"
            f"{title}\n\n"
            f"{question}\n"
            f"{help_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_catboost_l2_leaf_reg(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle CatBoost l2_leaf_reg selection and start training."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        l2_value = query.data.split(":")[-1]

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        if l2_value == "custom":
            l2_leaf_reg = 3
        else:
            l2_leaf_reg = int(l2_value)

        session.selections['catboost_config']['l2_leaf_reg'] = l2_leaf_reg
        await self.state_manager.update_session(session)

        # Configuration complete - start training
        model_type = session.selections.get('catboost_model_type')
        catboost_config = session.selections.get('catboost_config')

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        # Build i18n message
        title = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.title', locale=locale)
        model_label = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.model_label', locale=locale)
        parameters_label = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.parameters_label', locale=locale)
        param_iterations = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.param_iterations', locale=locale, value=catboost_config['iterations'])
        param_depth = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.param_depth', locale=locale, value=catboost_config['depth'])
        param_learning_rate = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.param_learning_rate', locale=locale, value=catboost_config['learning_rate'])
        param_l2_leaf_reg = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.param_l2_leaf_reg', locale=locale, value=catboost_config['l2_leaf_reg'])
        starting_training = I18nManager.t('workflow_state.model_config.catboost.configuration_complete.starting_training', locale=locale)
        patience = I18nManager.t('workflow_state.training.patience', locale=locale)

        await query.edit_message_text(
            f"{title}\n\n"
            f"{model_label} {model_display}\n"
            f"{parameters_label}\n"
            f"{param_iterations}\n"
            f"{param_depth}\n"
            f"{param_learning_rate}\n"
            f"{param_l2_leaf_reg}\n\n"
            f"{starting_training}\n\n"
            f"{patience}",
            parse_mode="Markdown"
        )

        # Execute training with CatBoost config
        await self._execute_sklearn_training(update, context, session, model_type)

    async def handle_training_execution(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Execute training after 'Start Training' button click."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please restart with /train")
            return

        # Wrap message editing with defensive error handling
        try:
            locale = session.language if session.language else None
            await query.edit_message_text(f"{I18nManager.t('workflow_state.training.starting', locale=locale)}\n\n{I18nManager.t('workflow_state.training.patience', locale=locale)}", parse_mode="Markdown")
        except telegram.error.BadRequest as e:
            logger.warning(f"Failed to edit message in handle_training_execution: {e}")
            # Fallback: send new message instead
            await update.effective_message.reply_text(
                f"{I18nManager.t('workflow_state.training.starting', locale=locale)}\n\n{I18nManager.t('workflow_state.training.patience', locale=locale)}",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning(f"Telegram API error in handle_training_execution message edit: {e}")
            # Continue with training even if message editing fails

        # Transition to TRAINING state before starting training
        # BUG FIX: Must transition to TRAINING state first, otherwise transition to
        # TRAINING_COMPLETE will fail (state machine only allows training → training_complete)
        print(f"🔍 DEBUG: Transitioning to TRAINING state")
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.TRAINING.value
        )

        if not success:
            logger.error(f"Failed to transition to TRAINING: {error_msg}")
            await query.edit_message_text(
                f"❌ **State Transition Error**\n\n{error_msg}\n\nPlease restart with /train",
                parse_mode="Markdown"
            )
            return

        print(f"✅ DEBUG: State transition to TRAINING successful, state={session.current_state}")

        # Trigger actual Keras training
        print(f"🚀 DEBUG: Starting ML Engine training")
        try:
            # Extract training parameters from session
            target = session.selections.get('target_column')
            features = session.selections.get('feature_columns')
            file_path = session.file_path
            model_type = session.selections.get('model_type')
            config = session.selections.get('keras_config', {})

            print(f"🚀 DEBUG: Training params - target={target}, features={features}, model={model_type}")
            print(f"🔍 PATH_DEBUG: session.file_path BEFORE training = '{session.file_path}'")
            print(f"🔍 PATH_DEBUG: file_path variable = '{file_path}'")
            print(f"🚀 DEBUG: config={config}")

            # Calculate number of features
            n_features = len(features)

            # Generate default architecture using template system
            architecture = get_template(
                model_type=model_type,
                n_features=n_features,
                kernel_initializer=config.get('kernel_initializer', 'glorot_uniform')
            )

            print(f"🚀 DEBUG: Generated architecture - layers={len(architecture['layers'])}, "
                  f"compile={architecture['compile']}")

            # Build complete hyperparameters dict with architecture
            hyperparameters = {
                **config,  # epochs, batch_size, kernel_initializer, verbose, validation_split
                'architecture': architecture,
                'n_features': n_features
            }

            # WORKER ROUTING: Check if user has connected worker
            websocket_server = context.bot_data.get('websocket_server')
            worker_manager = websocket_server.worker_manager if websocket_server else None

            if worker_manager and worker_manager.is_user_connected(user_id):
                # Route to local worker
                print(f"🖥️ DEBUG: User has connected worker, routing to local worker")
                result = await self._execute_training_on_worker(
                    user_id=user_id,
                    file_path=file_path,
                    task_type='neural_network',
                    model_type=model_type,
                    target_column=target,
                    feature_columns=features,
                    hyperparameters=hyperparameters,
                    test_size=1.0 - config.get('validation_split', 0.2),
                    context=context
                )
            else:
                # Execute on bot (original code path)
                print(f"🤖 DEBUG: No worker connected, executing on bot")
                # Call ML Engine to train (wrapped in executor to prevent blocking event loop)
                # FIX: ml_engine.train_model() is synchronous and would block the async event loop
                # during training (several minutes for 100 epochs). Using run_in_executor() runs
                # the blocking call in a separate thread, keeping the event loop responsive.
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    lambda: self.ml_engine.train_model(
                        file_path=file_path,  # Lazy loading from deferred file
                        task_type='neural_network',
                        model_type=model_type,  # 'keras_binary_classification'
                        target_column=target,
                        feature_columns=features,
                        user_id=user_id,
                        hyperparameters=hyperparameters,  # Complete hyperparameters with architecture
                        test_size=1.0 - config.get('validation_split', 0.2)
                    )
                )

            print(f"🚀 DEBUG: Training result = {result}")

            # Send success message with metrics and transition to naming workflow
            if result.get('success'):
                model_id = result.get('model_id', 'N/A')
                model_info = result.get('model_info', {})  # For local worker training
                metrics_text = self._format_keras_metrics(result.get('metrics', {}))

                # Transition to TRAINING_COMPLETE state
                print(f"🔍 DEBUG: Transitioning to TRAINING_COMPLETE state")
                success, error_msg, _ = await self.state_manager.transition_state(
                    session,
                    MLTrainingState.TRAINING_COMPLETE.value
                )

                if not success:
                    logger.error(f"Failed to transition to TRAINING_COMPLETE: {error_msg}")
                    await update.effective_message.reply_text(
                        f"❌ **State Transition Error**\n\n{error_msg}",
                        parse_mode="Markdown"
                    )
                    return

                print(f"✅ DEBUG: State transition to TRAINING_COMPLETE successful, state={session.current_state}")

                # Store model_id and model_info in session for naming workflow
                session.selections['pending_model_id'] = model_id
                session.selections['pending_model_info'] = model_info  # For local worker: use instead of filesystem lookup
                await self.state_manager.update_session(session)

                # Get locale from session
                locale = session.language if session.language else None

                # Show naming options with inline keyboard
                # NOTE: callback_data is short (no model_id) to stay within Telegram's 64-byte limit
                # model_id is already stored in session.selections['pending_model_id'] at line 1831
                keyboard = [
                    [InlineKeyboardButton(
                        I18nManager.t('workflow_state.training.completion.buttons.name_model', locale=locale),
                        callback_data="name_model"
                    )],
                    [InlineKeyboardButton(
                        I18nManager.t('workflow_state.training.completion.buttons.skip_naming', locale=locale),
                        callback_data="skip_naming"
                    )]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.effective_message.reply_text(
                    f"{I18nManager.t('workflow_state.training.completion.header', locale=locale)}\n\n"
                    f"{I18nManager.t('workflow_state.training.completion.metrics_label', locale=locale)}\n{metrics_text}\n\n"
                    f"{I18nManager.t('workflow_state.training.completion.model_id_label', locale=locale)}: `{model_id}`\n\n"
                    f"{I18nManager.t('workflow_state.training.completion.naming_prompt', locale=locale)}",
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                await update.effective_message.reply_text(
                    f"❌ **Training Failed**\n\n{escape_markdown_v1(str(error_msg))}",
                    parse_mode="Markdown"
                )

        except TrainingError as e:
            # Build complete error message including hidden details
            error_msg = str(e)
            if hasattr(e, 'error_details') and e.error_details:
                error_msg += f"\n\nDetails: {e.error_details}"

            logger.error(f"Training error: {e}", exc_info=True)
            await update.effective_message.reply_text(
                f"❌ **Training Error**\n\n{escape_markdown_v1(error_msg)}\n\nPlease check your data and configuration.",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Unexpected training error: {e}", exc_info=True)
            await update.effective_message.reply_text(
                f"❌ **Unexpected Error**\n\nAn error occurred during training. Please try again or check logs.",
                parse_mode="Markdown"
            )

    async def handle_name_model_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle 'Name Model' button click - prompt for custom name."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_name_model_callback: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        # Retrieve model_id from session (stored at training completion)
        model_id = session.selections.get('pending_model_id')
        if not model_id:
            logger.error("No pending_model_id found in session")
            await query.edit_message_text(
                I18nManager.t('workflow_state.training.errors.model_id_not_found', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Transition to NAMING_MODEL state
        await self.state_manager.transition_state(
            session,
            MLTrainingState.NAMING_MODEL.value
        )

        # model_id already in session, no need to store again
        await self.state_manager.update_session(session)

        # Send prompt for custom name
        await query.edit_message_text(
            f"{I18nManager.t('workflow_state.training.naming.header', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.training.naming.description', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.training.naming.rules_header', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.training.naming.rules_length', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.training.naming.rules_chars', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.training.naming.examples_header', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.training.naming.example1', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.training.naming.example2', locale=locale)}\n"
            f"{I18nManager.t('workflow_state.training.naming.example3', locale=locale)}\n\n"
            f"{I18nManager.t('workflow_state.training.naming.input_prompt', locale=locale)}",
            parse_mode="Markdown"
        )

    async def handle_model_name_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle custom model name text input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            custom_name = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_model_name_input: {e}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request'),
                    parse_mode="Markdown"
                )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            await update.message.reply_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired'),
                parse_mode="Markdown"
            )
            return

        # Only process if in NAMING_MODEL state
        if session.current_state != MLTrainingState.NAMING_MODEL.value:
            # Not in naming state, ignore this text input
            return

        # Get pending model_id from session
        model_id = session.selections.get('pending_model_id')

        # Get locale from session for i18n
        locale = session.language if session.language else None

        if not model_id:
            await update.message.reply_text(
                "❌ **Error**\n\nModel ID not found. Please restart with /train",
                parse_mode="Markdown"
            )
            return

        # Set custom name using ML Engine
        try:
            # For local worker training, use session metadata and send name to worker
            session_model_info = session.selections.get('pending_model_info', {})

            if session_model_info:
                # Local worker training: send custom name to worker
                # Model is on user's machine, so we need to update it via worker
                model_info = session_model_info

                # Send set_model_name job to worker
                success = await self._set_model_name_through_worker(
                    user_id=user_id,
                    model_id=model_id,
                    custom_name=custom_name,
                    context=context
                )
                if not success:
                    logger.warning(f"Failed to set model name on worker for {model_id}")
                    # Continue anyway - the name is stored locally in session_model_info
            else:
                # Normal training: model is on server
                self.ml_engine.set_model_name(user_id, model_id, custom_name)
                model_info = self.ml_engine.get_model_info(user_id, model_id)

            # Transition to MODEL_NAMED state
            await self.state_manager.transition_state(
                session,
                MLTrainingState.MODEL_NAMED.value
            )

            # Send success confirmation with template button
            template_keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('templates.save.button', locale=locale, default="💾 Save as Template"),
                    callback_data="save_as_template"
                )]
            ]
            template_reply_markup = InlineKeyboardMarkup(template_keyboard)

            await update.message.reply_text(
                f"✅ **Model Named Successfully!**\n\n"
                f"📝 **Name:** {escape_markdown_v1(custom_name)}\n"
                f"{I18nManager.t('workflow_state.training.completion.model_id_label', locale=locale)}: `{model_id}`\n"
                f"{I18nManager.t('workflow_state.training.model_type_display', locale=locale)} {model_info.get('model_type', 'N/A')}\n\n"
                f"{I18nManager.t('workflow_state.training.ready_for_predictions', locale=locale)}",
                parse_mode="Markdown",
                reply_markup=template_reply_markup
            )

            # Stay in MODEL_NAMED state to allow "Save as Template" button click
            # Workflow completes when user ignores button or after template is saved

            # Stop handler propagation
            raise ApplicationHandlerStop

        except ApplicationHandlerStop:
            # Re-raise immediately - this is control flow, not an error
            raise

        except ValidationError as e:
            # Invalid name format - show error and stay in NAMING_MODEL state
            await update.message.reply_text(
                f"❌ **Invalid Name**\n\n"
                f"{escape_markdown_v1(str(e))}\n\n"
                f"Please try again with a valid name.",
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"Error setting model name: {e}")
            await update.message.reply_text(
                f"❌ **Error**\n\nFailed to set model name. Please try again.",
                parse_mode="Markdown"
            )

    async def handle_skip_naming_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle 'Skip - Use Default' button click."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_skip_naming_callback: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            # Get locale for i18n (best effort)
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflows.ml_training_local_path.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Get locale from session
        locale = session.language if session.language else None

        # Retrieve model_id from session (stored at training completion)
        model_id = session.selections.get('pending_model_id')
        if not model_id:
            logger.error("No pending_model_id found in session")
            await query.edit_message_text(
                I18nManager.t('workflow_state.training.errors.model_id_not_found', locale=locale),
                parse_mode="Markdown"
            )
            return

        try:
            # For local worker training, use metadata from session instead of filesystem lookup
            model_info = session.selections.get('pending_model_info', {})

            if model_info:
                # Local worker training: use session metadata
                default_name = self.ml_engine._generate_default_name(
                    model_type=model_info.get('model_type', 'model'),
                    task_type=model_info.get('task_type', 'unknown'),
                    created_at=model_info.get('created_at', '')
                )
                # Skip set_model_name() - model is on user's machine, not server
                # The name is for display purposes only
            else:
                # Normal training: model is on server, use filesystem lookup
                model_info = self.ml_engine.get_model_info(user_id, model_id)
                default_name = self.ml_engine._generate_default_name(
                    model_type=model_info.get('model_type', 'model'),
                    task_type=model_info.get('task_type', 'unknown'),
                    created_at=model_info.get('created_at', '')
                )
                self.ml_engine.set_model_name(user_id, model_id, default_name)

            # Transition to MODEL_NAMED state
            await self.state_manager.transition_state(
                session,
                MLTrainingState.MODEL_NAMED.value
            )

            # Send confirmation with template button
            template_keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('templates.save.button', locale=locale, default="💾 Save as Template"),
                    callback_data="save_as_template"
                )]
            ]
            template_reply_markup = InlineKeyboardMarkup(template_keyboard)

            await query.edit_message_text(
                f"{I18nManager.t('workflow_state.training.model_ready', locale=locale)}\n\n"
                f"{I18nManager.t('workflow_state.training.default_name_display', locale=locale)} {escape_markdown_v1(default_name)}\n"
                f"{I18nManager.t('workflow_state.training.completion.model_id_label', locale=locale)} `{model_id}`\n"
                f"{I18nManager.t('workflow_state.training.model_type_display', locale=locale)} {model_info.get('model_type', 'N/A')}\n\n"
                f"{I18nManager.t('workflow_state.training.ready_for_predictions', locale=locale)}",
                parse_mode="Markdown",
                reply_markup=template_reply_markup
            )

            # Stay in MODEL_NAMED state to allow "Save as Template" button click
            # Workflow completes when user ignores button or after template is saved

        except Exception as e:
            logger.error(f"Error setting default model name: {e}")
            await query.edit_message_text(
                f"❌ **Error**\n\nFailed to set default name. Model ID: `{model_id}`",
                parse_mode="Markdown"
            )
            # Cancel workflow after naming error
            await self.state_manager.cancel_workflow(session)

    async def _execute_training_on_worker(
        self,
        user_id: int,
        file_path: str,
        task_type: str,
        model_type: str,
        target_column: str,
        feature_columns: list,
        hyperparameters: dict,
        test_size: float,
        context: ContextTypes.DEFAULT_TYPE,
        custom_name: str = None
    ) -> dict:
        """Execute training on connected local worker.

        Creates a job and dispatches it to the worker, then waits for result.

        Args:
            user_id: User ID
            file_path: Path to training data on worker machine
            task_type: ML task type (regression, classification, neural_network)
            model_type: Model type
            target_column: Target column name
            feature_columns: List of feature column names
            hyperparameters: Model hyperparameters
            test_size: Test set size
            context: Bot context
            custom_name: User-provided custom model name

        Returns:
            Training result dict with success, model_id, metrics
        """
        from src.worker.job_queue import JobType
        import asyncio

        # Get job queue from context
        websocket_server = context.bot_data.get('websocket_server')
        job_queue = websocket_server.job_queue if websocket_server else None

        if not job_queue:
            logger.error("Job queue not available")
            return {
                'success': False,
                'error': 'Worker system not available'
            }

        # Create job parameters
        job_params = {
            'file_path': file_path,
            'task_type': task_type,
            'model_type': model_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'hyperparameters': hyperparameters,
            'test_size': test_size,
            'custom_name': custom_name
        }
        print(f"🔍 PATH_DEBUG: file_path in job_params = '{job_params['file_path']}'")

        # Create job
        job_id = await job_queue.create_job(
            user_id=user_id,
            job_type=JobType.TRAIN,
            params=job_params,
            timeout=600.0  # 10 minutes timeout for training
        )

        logger.info(f"Created training job {job_id} for user {user_id}")

        # Wait for job to complete (poll job status)
        max_wait = 600  # 10 minutes
        poll_interval = 2  # 2 seconds
        elapsed = 0

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            job = job_queue.get_job(job_id)
            if not job:
                return {
                    'success': False,
                    'error': 'Job not found'
                }

            # Check if completed
            from src.worker.job_queue import JobStatus
            if job.status == JobStatus.COMPLETED:
                logger.info(f"Training job {job_id} completed successfully")
                return {
                    'success': True,
                    'model_id': job.result.get('model_id'),
                    'metrics': job.result.get('metrics', {}),
                    'dataset_stats': job.result.get('dataset_stats', {}),
                    'training_time': job.result.get('training_time', 0),
                    'model_info': job.result.get('model_info', {})  # For local worker naming
                }
            elif job.status == JobStatus.FAILED:
                logger.error(f"Training job {job_id} failed: {job.error}")
                return {
                    'success': False,
                    'error': job.error
                }
            elif job.status == JobStatus.TIMEOUT:
                logger.error(f"Training job {job_id} timed out")
                return {
                    'success': False,
                    'error': 'Training timed out'
                }

        # Timeout reached
        logger.error(f"Training job {job_id} exceeded wait time")
        return {
            'success': False,
            'error': 'Training exceeded maximum wait time'
        }

    async def _set_model_name_through_worker(
        self,
        user_id: int,
        model_id: str,
        custom_name: str,
        context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Set custom model name through worker.

        Sends SET_MODEL_NAME job to worker to update model metadata.

        Args:
            user_id: User ID
            model_id: Model ID on worker machine
            custom_name: User-provided custom name
            context: Bot context

        Returns:
            True if successful, False otherwise
        """
        from src.worker.job_queue import JobType, JobStatus
        import asyncio

        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return False

        job_queue = getattr(websocket_server, 'job_queue', None)
        worker_manager = websocket_server.worker_manager

        if not job_queue or not worker_manager.is_user_connected(user_id):
            return False

        try:
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.SET_MODEL_NAME,
                params={'model_id': model_id, 'custom_name': custom_name},
                timeout=30.0
            )

            # Poll for result
            max_wait, poll_interval, elapsed = 30, 0.5, 0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    return False
                if job.status == JobStatus.COMPLETED:
                    logger.info(f"Set model name through worker: {model_id} → {custom_name}")
                    return True
                elif job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    logger.error(f"Failed to set model name: {job.error}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Error setting model name through worker: {e}")
            return False

    async def _get_file_info_from_worker(
        self,
        user_id: int,
        file_path: str,
        context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[Dict[str, Any]]:
        """Get file info from worker. Returns None if no worker or error.

        Used for backwards-compatible file size checking:
        - If worker is connected, uses worker to check file on user's machine
        - If no worker, caller should fall back to local file access

        Args:
            user_id: Telegram user ID
            file_path: Path to file on user's machine
            context: Bot context

        Returns:
            Dict with file info if successful, None otherwise
        """
        from src.worker.job_queue import JobType, JobStatus
        import asyncio

        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return None

        job_queue = getattr(websocket_server, 'job_queue', None)
        worker_manager = websocket_server.worker_manager

        if not job_queue or not worker_manager.is_user_connected(user_id):
            return None

        try:
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.FILE_INFO,
                params={'file_path': file_path},
                timeout=30.0
            )

            # Poll for result
            max_wait, poll_interval, elapsed = 30, 0.5, 0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    return None
                if job.status == JobStatus.COMPLETED:
                    return job.result
                elif job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    return None

            return None
        except Exception as e:
            logger.error(f"Error getting file info from worker: {e}")
            return None

    def _format_keras_metrics(self, metrics: dict) -> str:
        """
        Format Keras training metrics for display.

        Args:
            metrics: Dictionary containing training metrics

        Returns:
            Formatted string with metrics
        """
        formatted = []

        # Common metrics
        if 'loss' in metrics:
            formatted.append(f"• Loss: {metrics['loss']:.4f}")
        if 'accuracy' in metrics:
            formatted.append(f"• Accuracy: {metrics['accuracy']:.2%}")
        if 'val_loss' in metrics:
            formatted.append(f"• Val Loss: {metrics['val_loss']:.4f}")
        if 'val_accuracy' in metrics:
            formatted.append(f"• Val Accuracy: {metrics['val_accuracy']:.2%}")

        # Regression metrics
        if 'mse' in metrics:
            formatted.append(f"• MSE: {metrics['mse']:.4f}")
        if 'rmse' in metrics:
            formatted.append(f"• RMSE: {metrics['rmse']:.4f}")
        if 'mae' in metrics:
            formatted.append(f"• MAE: {metrics['mae']:.4f}")
        if 'r2' in metrics:
            formatted.append(f"• R²: {metrics['r2']:.4f}")

        return "\n".join(formatted) if formatted else "Training completed"

    async def handle_schema_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle schema confirmation/rejection."""
        query = update.callback_query
        await query.answer()

        # Defensive extraction - protect against None values in update object
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split(":")[-1]  # "accept" or "reject"
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_schema_confirmation: {e}")
            # Get locale for i18n (best effort)
            locale = None
            try:
                await query.edit_message_text(
                    I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        I18nManager.t('workflows.ml_training_local_path.errors.invalid_request', locale=locale),
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session for i18n
        locale = session.language if session.language else None

        if choice == "accept":
            # Transfer detected schema to selections (required for training)
            if session.detected_schema:
                session.selections['target_column'] = session.detected_schema.get('target')
                session.selections['feature_columns'] = session.detected_schema.get('features', [])
                session.selections['detected_task_type'] = session.detected_schema.get('task_type')

            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to CONFIRMING_MODEL (schema accept)")

            # Skip target/feature selection since schema is confirmed
            # Go directly to model selection (matching deferred loading behavior)
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CONFIRMING_MODEL.value
            )

            suggested_target = session.detected_schema.get('target') if session.detected_schema else None

            await query.edit_message_text(
                LocalPathMessages.schema_accepted_message(suggested_target, locale=locale),
                parse_mode="Markdown"
            )

            # Immediately show model selection (matching deferred loading)
            await self._show_model_selection(update, context, session)

        elif choice == "reject":
            # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button Fix)
            session.save_state_snapshot()
            self.logger.debug("📸 State snapshot saved before transition to AWAITING_FILE_PATH (schema reject)")

            # User rejects schema - go back to file path input
            await self.state_manager.transition_state(
                session,
                MLTrainingState.AWAITING_FILE_PATH.value
            )

            # Clear stored data
            session.uploaded_data = None
            session.file_path = None
            session.detected_schema = None

            await query.edit_message_text(
                LocalPathMessages.schema_rejected_message(locale=locale),
                parse_mode="Markdown"
            )


def register_local_path_handlers(
    application,
    state_manager: StateManager,
    data_loader: DataLoader
) -> None:
    """Register local path workflow handlers with Telegram application."""
    from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters

    print("🔧 Registering local path ML training handlers...")

    handler = LocalPathMLTrainingHandler(state_manager, data_loader)

    # Command handlers
    application.add_handler(
        CommandHandler("train", handler.handle_start_training)
    )

    # Callback query handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_data_source_selection,
            pattern=r"^data_source:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_load_option_selection,
            pattern=r"^load_option:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_schema_confirmation,
            pattern=r"^schema:"
        )
    )

    # Model selection callback handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_model_category_selection,
            pattern=r"^model_category:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_model_selection,
            pattern=r"^model_select:"
        )
    )

    # Keras configuration callback handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_keras_epochs,
            pattern=r"^keras_epochs:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_keras_batch,
            pattern=r"^keras_batch:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_keras_initializer,
            pattern=r"^keras_init:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_keras_verbose,
            pattern=r"^keras_verbose:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_keras_validation,
            pattern=r"^keras_val:"
        )
    )

    # XGBoost configuration callback handlers
    print("  ✓ Registering XGBoost parameter handlers")
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_xgboost_n_estimators,
            pattern=r"^xgboost_n_estimators:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_xgboost_max_depth,
            pattern=r"^xgboost_max_depth:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_xgboost_learning_rate,
            pattern=r"^xgboost_learning_rate:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_xgboost_subsample,
            pattern=r"^xgboost_subsample:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_xgboost_colsample,
            pattern=r"^xgboost_colsample:"
        )
    )

    # LightGBM configuration callback handlers
    print("  ✓ Registering LightGBM parameter handlers")
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_lightgbm_num_leaves,
            pattern=r"^lightgbm_num_leaves:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_lightgbm_n_estimators,
            pattern=r"^lightgbm_n_estimators:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_lightgbm_learning_rate,
            pattern=r"^lightgbm_learning_rate:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_lightgbm_use_defaults,
            pattern=r"^lightgbm_use_defaults$"
        )
    )

    # CatBoost configuration callback handlers
    print("  ✓ Registering CatBoost parameter handlers")
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_catboost_iterations,
            pattern=r"^catboost_iterations:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_catboost_depth,
            pattern=r"^catboost_depth:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_catboost_learning_rate,
            pattern=r"^catboost_learning_rate:"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_catboost_l2_leaf_reg,
            pattern=r"^catboost_l2:"
        )
    )

    # Unified text message handler for file paths and schema input
    # Accepts all text including Unix paths starting with "/"
    # Internal state-based routing handles different inputs
    # GROUP 1: Ensures this handler is checked before general message_handler (group 0)
    print("  ✓ Registering unified text input handler (group=1)")
    application.add_handler(
        MessageHandler(
            filters.TEXT,
            handler.handle_text_input  # Single handler routes based on state
        ),
        group=1  # Priority group to prevent collision with general handler
    )

    # Universal back button handler (Phase 2: Workflow Back Button)
    from src.bot.main_handlers import handle_workflow_back
    application.add_handler(
        CallbackQueryHandler(
            handle_workflow_back,
            pattern=r"^workflow_back$"
        )
    )
    print("  ✓ Registered universal back button handler")

    # Template workflow handlers (Phase 6: Templates)
    print("  ✓ Registering template workflow handlers")

    # Start training button (after template save or configuration)
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_training_execution,
            pattern=r"^start_training$"
        )
    )

    # Save template button
    application.add_handler(
        CallbackQueryHandler(
            handler.template_handlers.handle_template_save_request,
            pattern=r"^template_save$"
        )
    )

    # Template selection
    application.add_handler(
        CallbackQueryHandler(
            handler.template_handlers.handle_template_selection,
            pattern=r"^load_template:"
        )
    )

    # Template load options
    application.add_handler(
        CallbackQueryHandler(
            handler.template_handlers.handle_template_load_option,
            pattern=r"^template_load_now$|^template_defer$"
        )
    )

    # Template load & train button (for deferred templates)
    application.add_handler(
        CallbackQueryHandler(
            handler.template_handlers.handle_template_load_and_train,
            pattern=r"^template_load_and_train$"
        )
    )

    # Cancel template
    application.add_handler(
        CallbackQueryHandler(
            handler.template_handlers.handle_cancel_template,
            pattern=r"^cancel_template$"
        )
    )

    # Back button in template selection
    application.add_handler(
        CallbackQueryHandler(
            handler.template_handlers.handle_template_source_selection,
            pattern=r"^back$"
        )
    )

    print("  ✓ Template handlers registered")

    # Model naming workflow handlers (Phase 4)
    print("  ✓ Registering model naming workflow handlers")

    # Name model button
    # Pattern matches simple callback_data (no colon, no model_id)
    # model_id is retrieved from session.selections['pending_model_id']
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_name_model_callback,
            pattern=r"^name_model$"  # Exact match (no colon separator)
        )
    )

    # Skip naming button
    # Pattern matches simple callback_data (no colon, no model_id)
    # model_id is retrieved from session.selections['pending_model_id']
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_skip_naming_callback,
            pattern=r"^skip_naming$"  # Exact match (no colon separator)
        )
    )

    print("  ✓ Model naming handlers registered")

    print("✅ Local path ML training handlers registered successfully")
