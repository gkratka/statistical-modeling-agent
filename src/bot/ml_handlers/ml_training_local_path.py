"""Telegram bot handlers for ML training with local file path workflow."""

import logging
from typing import Optional

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
            path_validator=path_validator
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
                    "❌ **Invalid Request**\n\nPlease try /train again.",
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

        await update.message.reply_text(
            LocalPathMessages.telegram_upload_prompt(),
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

        # Create inline keyboard (no back button - this is the entry point)
        keyboard = [
            [InlineKeyboardButton("📤 Upload File", callback_data="data_source:telegram")],
            [InlineKeyboardButton("📂 Use Local Path", callback_data="data_source:local_path")],
            [InlineKeyboardButton("📋 Use Template", callback_data="data_source:template")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            LocalPathMessages.data_source_selection_prompt(),
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

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
                    f"❌ State transition failed: {error_msg}",
                    parse_mode="Markdown"
                )
                return

            print(f"🔀 DEBUG: State transition SUCCESS: {old_state} → {session.current_state}")

            await query.edit_message_text(
                LocalPathMessages.telegram_upload_prompt(),
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
                    LocalPathMessages.file_path_input_prompt(allowed_dirs),
                    parse_mode="Markdown"
                )
            except telegram.error.BadRequest as e:
                logger.warning(f"Failed to edit message in handle_data_source_selection: {e}")
                # Fallback: send new message instead
                await update.effective_message.reply_text(
                    LocalPathMessages.file_path_input_prompt(allowed_dirs),
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
                        "❌ **Invalid Request**\n\nPlease restart with /train",
                        parse_mode="Markdown"
                    )
                return

            print(f"📥 DEBUG: Unified text handler called with: {text_input[:50]}...")

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

        # Show validating message
        validating_msg = await update.message.reply_text(
            "🔍 **Validating path...**"
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
                allowed_extensions=self.data_loader.local_extensions
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
                        error_details=error_msg
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
                [InlineKeyboardButton("🔄 Load Now", callback_data="load_option:immediate")],
                [InlineKeyboardButton("⏳ Defer Loading", callback_data="load_option:defer")]
            ]
            add_back_button(keyboard)  # Phase 2: Workflow Back Button
            reply_markup = InlineKeyboardMarkup(keyboard)
            print("⌨️ DEBUG: Keyboard created with Load Now/Defer Loading buttons")

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(str(resolved_path), size_mb),
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
                    error_details=str(e)
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
                f"❌ **State Transition Failed**\n\n{error_msg}",
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
            LocalPathMessages.password_prompt(original_path, parent_dir),
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

        # Validate we're in password state
        if session.current_state != MLTrainingState.AWAITING_PASSWORD.value:
            return

        # Get pending path
        pending_path = session.pending_auth_path
        if not pending_path:
            await update.message.reply_text(
                "❌ **Session Error**\n\nNo pending path found. Please try /train again.",
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
                    "❌ **Session Expired**\n\nPlease restart with /train.",
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
            size_mb = get_file_size_mb(Path(pending_path))

            # Transition to load options
            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CHOOSING_LOAD_OPTION.value
            )

            await update.message.reply_text(
                LocalPathMessages.password_success(parent_dir),
                parse_mode="Markdown"
            )

            # Show load options
            keyboard = [
                [InlineKeyboardButton("🔄 Load Now", callback_data="load_option:immediate")],
                [InlineKeyboardButton("⏳ Defer Loading", callback_data="load_option:defer")]
            ]
            add_back_button(keyboard)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(pending_path, size_mb),
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
                    LocalPathMessages.password_failure(error_msg),
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
            LocalPathMessages.file_path_input_prompt(allowed_dirs),
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if choice == "immediate":
            # Load data immediately
            self.logger.info(f"[LOAD_NOW] User {user_id} selected: {choice}")
            self.logger.info(f"[LOAD_NOW] Session state: {session.current_state}")
            self.logger.info(f"[LOAD_NOW] File path: {session.file_path}")

            loading_msg = await query.edit_message_text(
                LocalPathMessages.loading_data_message()
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
                LocalPathMessages.schema_input_prompt(),
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
                    f"⚠️ Schema accepted but workflow transition failed: {error_msg}"
                )
                return

            print(f"🔀 DEBUG: State transition SUCCESS: {old_state} → {session.current_state}")

            # Show confirmation
            await update.message.reply_text(
                LocalPathMessages.schema_accepted_deferred(
                    parsed_schema.target,
                    len(parsed_schema.features)
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
                LocalPathMessages.schema_parse_error(str(e)),
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
                summary, suggested_target, suggested_features, task_type
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
                summary, suggested_target, suggested_features, task_type
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

        # Create inline keyboard with model categories
        keyboard = [
            [InlineKeyboardButton("📈 Regression Models", callback_data="model_category:regression")],
            [InlineKeyboardButton("🎯 Classification Models", callback_data="model_category:classification")],
            [InlineKeyboardButton("🧠 Neural Networks", callback_data="model_category:neural")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🤖 **Choose Model Type**\n\n"
            "Select the type of model for your training:\n\n"
            "📈 **Regression**: Predict continuous values (prices, temperatures, etc.)\n"
            "🎯 **Classification**: Categorize data (spam/not spam, approve/reject, etc.)\n"
            "🧠 **Neural Networks**: Advanced deep learning models\n\n"
            "Which category fits your task?",
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
                        parse_mode="Markdown"
                    )
            return

        print(f"📊 DEBUG: User selected model category: {category}")

        # Get session to access stored detection from schema processing
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Save state snapshot BEFORE showing specific model selection (Phase 2: Back Button Fix)
        session.save_state_snapshot()
        self.logger.debug("📸 State snapshot saved before showing specific model selection")

        # Use previously detected task type from schema processing (deferred loading compatible)
        detected_task = session.selections.get('detected_task_type') if session else None

        if detected_task:
            print(f"🔍 DEBUG: Task type detected = {detected_task}")
        else:
            print(f"⚠️ DEBUG: No task type detected")

        # Show ALL models in category (user chooses based on their needs)
        model_options = {
            "regression": [
                ("Linear Regression", "linear"),
                ("Ridge Regression (L2)", "ridge"),
                ("Lasso Regression (L1)", "lasso"),
                ("ElasticNet (L1+L2)", "elasticnet"),
                ("Polynomial Regression", "polynomial"),
                ("XGBoost Regression", "xgboost_regression"),
                ("LightGBM Regression", "lightgbm_regression"),
                ("CatBoost Regression", "catboost_regression")
            ],
            "classification": [
                ("Logistic Regression", "logistic"),
                ("Decision Tree", "decision_tree"),
                ("Random Forest", "random_forest"),
                ("Gradient Boosting (sklearn)", "gradient_boosting"),
                ("XGBoost Classification", "xgboost_binary_classification"),
                ("LightGBM Classification", "lightgbm_binary_classification"),
                ("CatBoost Classification", "catboost_binary_classification"),
                ("Support Vector Machine", "svm"),
                ("Naive Bayes", "naive_bayes")
            ],
            "neural": [
                ("MLP Regression", "mlp_regression"),
                ("MLP Classification", "mlp_classification"),
                ("Keras Binary Classification", "keras_binary_classification"),
                ("Keras Multiclass Classification", "keras_multiclass_classification"),
                ("Keras Regression", "keras_regression")
            ]
        }
        models = model_options.get(category, [])
        keyboard = [
            [InlineKeyboardButton(name, callback_data=f"model_select:{model_type}")]
            for name, model_type in models
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button (replaces manual back button)
        reply_markup = InlineKeyboardMarkup(keyboard)

        category_names = {
            "regression": "📈 Regression Models",
            "classification": "🎯 Classification Models",
            "neural": "🧠 Neural Networks"
        }

        # Build message with detection info if available
        message = f"{category_names.get(category, 'Models')}\n\n"

        if detected_task:
            task_emoji = "📈" if detected_task == "regression" else "🎯"
            message += f"{task_emoji} **Detected Task**: {detected_task.title()}\n"
            message += f"_(All models shown - choose what works best for you)_\n\n"

        message += "Select a specific model:"

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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
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
                f"This may take a few moments.",
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

        # Start with epochs configuration
        keyboard = [
            [InlineKeyboardButton("50 epochs", callback_data="keras_epochs:50")],
            [InlineKeyboardButton("100 epochs (recommended)", callback_data="keras_epochs:100")],
            [InlineKeyboardButton("200 epochs", callback_data="keras_epochs:200")],
            [InlineKeyboardButton("Custom", callback_data="keras_epochs:custom")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **Keras Neural Network Configuration**\n\n"
            "**Step 1/5: Training Epochs**\n\n"
            "How many training epochs?\n"
            "(More epochs = longer training, potentially better accuracy)",
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

        # Start with n_estimators configuration
        keyboard = [
            [InlineKeyboardButton("50 trees", callback_data="xgboost_n_estimators:50")],
            [InlineKeyboardButton("100 trees (recommended)", callback_data="xgboost_n_estimators:100")],
            [InlineKeyboardButton("200 trees", callback_data="xgboost_n_estimators:200")],
            [InlineKeyboardButton("Custom", callback_data="xgboost_n_estimators:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **XGBoost Configuration**\n\n"
            "**Step 1/5: Number of Boosting Rounds**\n\n"
            "How many trees to build?\n"
            "(More trees = better fit, but slower training)",
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

        # Start with num_leaves configuration (LightGBM uses leaves not depth)
        keyboard = [
            [InlineKeyboardButton("15 leaves (fast)", callback_data="lightgbm_num_leaves:15")],
            [InlineKeyboardButton("31 leaves (recommended)", callback_data="lightgbm_num_leaves:31")],
            [InlineKeyboardButton("63 leaves (complex)", callback_data="lightgbm_num_leaves:63")],
            [InlineKeyboardButton("Use Defaults", callback_data="lightgbm_use_defaults")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "⚡ **LightGBM Configuration**\n\n"
            "**Step 1: Number of Leaves**\n\n"
            "LightGBM uses leaf-wise growth (faster than XGBoost).\n"
            "How many leaves per tree?\n\n"
            "💡 **Tip**: 15 = fast, 31 = balanced, 63 = complex",
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

        # Start with iterations configuration
        keyboard = [
            [InlineKeyboardButton("100 iterations (fast)", callback_data="catboost_iterations:100")],
            [InlineKeyboardButton("500 iterations", callback_data="catboost_iterations:500")],
            [InlineKeyboardButton("1000 iterations (recommended)", callback_data="catboost_iterations:1000")],
            [InlineKeyboardButton("2000 iterations (thorough)", callback_data="catboost_iterations:2000")],
            [InlineKeyboardButton("Custom", callback_data="catboost_iterations:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🐈 **CatBoost Configuration**\n\n"
            "**Step 1/4: Number of Boosting Rounds**\n\n"
            "How many iterations (trees) to build?\n"
            "(More iterations = better fit, but slower training)",
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

            # Call ML Engine to train (wrapped in executor to prevent blocking event loop)
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

                # Store model_id in session for naming workflow
                session.selections['pending_model_id'] = model_id
                await self.state_manager.update_session(session)

                # Format metrics based on task type
                metrics_text = self._format_sklearn_metrics(metrics, task_type)

                # Show naming options with inline keyboard
                keyboard = [
                    [InlineKeyboardButton("📝 Name Model", callback_data="name_model")],
                    [InlineKeyboardButton("⏭️ Skip - Use Default", callback_data="skip_naming")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Escape underscores for Markdown display
                model_display = model_type.replace('_', '\\_')

                await update.effective_message.reply_text(
                    f"✅ **Training Complete!**\n\n"
                    f"🎯 **Model**: {model_display}\n"
                    f"🆔 **Model ID**: `{model_id}`\n\n"
                    f"{metrics_text}\n\n"
                    f"Would you like to give this model a custom name?",
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            else:
                # Training failed
                error_msg = result.get('error', 'Unknown error during training')
                logger.error(f"Training failed: {error_msg}")
                await update.effective_message.reply_text(
                    f"❌ **Training Failed**\n\n"
                    f"Error: {error_msg}\n\n"
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
                f"An unexpected error occurred: {str(e)}\n\n"
                f"Please try again with /train",
                parse_mode="Markdown"
            )

    def _format_sklearn_metrics(self, metrics: dict, task_type: str) -> str:
        """Format sklearn/XGBoost metrics for user display."""
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

            return (
                f"📊 **Performance Metrics**:\n"
                f"• R² Score: {r2_str}\n"
                f"• RMSE: {rmse_str}\n"
                f"• MAE: {mae_str}\n"
                f"• MSE: {mse_str}"
            )
        else:
            # Classification metrics
            accuracy = metrics.get('accuracy', 'N/A')
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1 = metrics.get('f1', 'N/A')

            # Format values before f-string
            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy)
            precision_str = f"{precision:.4f}" if isinstance(precision, float) else str(precision)
            recall_str = f"{recall:.4f}" if isinstance(recall, float) else str(recall)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

            return (
                f"📊 **Performance Metrics**:\n"
                f"• Accuracy: {accuracy_str}\n"
                f"• Precision: {precision_str}\n"
                f"• Recall: {recall_str}\n"
                f"• F1 Score: {f1_str}"
            )

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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\n"
                    "This action is no longer valid. Please restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ **Invalid Request**\n\nPlease restart with /train",
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_epochs - user={user_id}, epochs_value={epochs_value}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Session Expired**\n\n"
                    "Your session has expired. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Session Expired**\n\nPlease start over with /train",
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

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Save Failed**\n\n"
                    "Unable to save your selection. Please try again.",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Save Failed**\n\nPlease try again.",
                    parse_mode="Markdown"
                )
            return

        # Move to batch size configuration
        keyboard = [
            [InlineKeyboardButton("16 (small)", callback_data="keras_batch:16")],
            [InlineKeyboardButton("32 (recommended)", callback_data="keras_batch:32")],
            [InlineKeyboardButton("64 (medium)", callback_data="keras_batch:64")],
            [InlineKeyboardButton("128 (large)", callback_data="keras_batch:128")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            "🧠 **Keras Neural Network Configuration**\n\n"
            f"✅ Epochs: {session.selections['keras_config']['epochs']}\n\n"
            "**Step 2/5: Batch Size**\n\n"
            "Select batch size:\n"
            "(Smaller = slower but more accurate, Larger = faster but less precise)"
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
                "❌ **Error Updating Message**\n\nPlease use /train to restart.",
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\n"
                    "This action is no longer valid. Please restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ **Invalid Request**\n\nPlease restart with /train",
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_batch - user={user_id}, batch_size={batch_size}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Session Expired**\n\n"
                    "Your session has expired. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Session Expired**\n\nPlease start over with /train",
                    parse_mode="Markdown"
                )
            return

        # Defensive check: keras_config exists
        if 'keras_config' not in session.selections:
            print(f"❌ ERROR: keras_config missing in handle_keras_batch for user {user_id}")
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

        session.selections['keras_config']['batch_size'] = batch_size
        print(f"📦 DEBUG: keras_config after batch = {session.selections['keras_config']}")

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Save Failed**\n\n"
                    "Unable to save your selection. Please try again.",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Save Failed**\n\nPlease try again.",
                    parse_mode="Markdown"
                )
            return

        # Get epochs with default fallback
        epochs = session.selections['keras_config'].get('epochs', 100)

        # Move to kernel initializer configuration
        keyboard = [
            [InlineKeyboardButton("glorot_uniform (recommended)", callback_data="keras_init:glorot_uniform")],
            [InlineKeyboardButton("random_normal", callback_data="keras_init:random_normal")],
            [InlineKeyboardButton("random_uniform", callback_data="keras_init:random_uniform")],
            [InlineKeyboardButton("he_normal", callback_data="keras_init:he_normal")],
            [InlineKeyboardButton("he_uniform", callback_data="keras_init:he_uniform")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            "🧠 **Keras Neural Network Configuration**\n\n"
            f"✅ Epochs: {epochs}\n"
            f"✅ Batch Size: {batch_size}\n\n"
            "**Step 3/5: Weight Initialization**\n\n"
            "Select kernel initializer:"
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\n"
                    "This action is no longer valid. Please restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ **Invalid Request**\n\nPlease restart with /train",
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_initializer - user={user_id}, initializer={initializer}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Session Expired**\n\n"
                    "Your session has expired. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Session Expired**\n\nPlease start over with /train",
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

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Save Failed**\n\n"
                    "Unable to save your selection. Please try again.",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Save Failed**\n\nPlease try again.",
                    parse_mode="Markdown"
                )
            return

        # Get previous values with default fallbacks
        epochs = session.selections['keras_config'].get('epochs', 100)
        batch_size = session.selections['keras_config'].get('batch_size', 32)

        # Move to verbose configuration
        keyboard = [
            [InlineKeyboardButton("0 - Silent", callback_data="keras_verbose:0")],
            [InlineKeyboardButton("1 - Progress bar (recommended)", callback_data="keras_verbose:1")],
            [InlineKeyboardButton("2 - One line per epoch", callback_data="keras_verbose:2")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            "🧠 **Keras Neural Network Configuration**\n\n"
            f"✅ Epochs: {epochs}\n"
            f"✅ Batch Size: {batch_size}\n"
            f"✅ Initializer: {escape_markdown_v1(initializer)}\n\n"
            "**Step 4/5: Training Output**\n\n"
            "Select verbosity level:"
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\n"
                    "This action is no longer valid. Please restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ **Invalid Request**\n\nPlease restart with /train",
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_verbose - user={user_id}, verbose={verbose}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Session Expired**\n\n"
                    "Your session has expired. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Session Expired**\n\nPlease start over with /train",
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

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Save Failed**\n\n"
                    "Unable to save your selection. Please try again.",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Save Failed**\n\nPlease try again.",
                    parse_mode="Markdown"
                )
            return

        # Get previous values with default fallbacks
        epochs = session.selections['keras_config'].get('epochs', 100)
        batch_size = session.selections['keras_config'].get('batch_size', 32)
        initializer = session.selections['keras_config'].get('kernel_initializer', 'glorot_uniform')

        # Move to validation split configuration
        keyboard = [
            [InlineKeyboardButton("0% - No validation", callback_data="keras_val:0.0")],
            [InlineKeyboardButton("10% validation", callback_data="keras_val:0.1")],
            [InlineKeyboardButton("20% validation (recommended)", callback_data="keras_val:0.2")],
            [InlineKeyboardButton("30% validation", callback_data="keras_val:0.3")]
        ]
        add_back_button(keyboard)  # Phase 2: Workflow Back Button
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            "🧠 **Keras Neural Network Configuration**\n\n"
            f"✅ Epochs: {epochs}\n"
            f"✅ Batch Size: {batch_size}\n"
            f"✅ Initializer: {escape_markdown_v1(initializer)}\n"
            f"✅ Verbosity: {verbose}\n\n"
            "**Step 5/5: Validation Split**\n\n"
            "Percentage of data to use for validation:"
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\n"
                    "This action is no longer valid. Please restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                # Fallback to new message if edit fails
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ **Invalid Request**\n\nPlease restart with /train",
                        parse_mode="Markdown"
                    )
            return

        print(f"🧠 DEBUG: handle_keras_validation - user={user_id}, validation_split={validation_split}")

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Defensive check: session exists
        if session is None:
            print(f"❌ ERROR: Session not found for user {user_id}")
            try:
                await query.edit_message_text(
                    "❌ **Session Expired**\n\n"
                    "Your session has expired. Please start over with /train",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Session Expired**\n\nPlease start over with /train",
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

        # Save session with error handling
        try:
            await self.state_manager.update_session(session)
        except Exception as e:
            print(f"❌ ERROR: Failed to update session: {e}")
            try:
                await query.edit_message_text(
                    "❌ **Configuration Save Failed**\n\n"
                    "Unable to save your selection. Please try again.",
                    parse_mode="Markdown"
                )
            except telegram.error.TelegramError:
                await update.effective_message.reply_text(
                    "❌ **Configuration Save Failed**\n\nPlease try again.",
                    parse_mode="Markdown"
                )
            return

        # Get all configuration with defaults
        config = session.selections['keras_config']
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 32)
        initializer = config.get('kernel_initializer', 'glorot_uniform')
        verbose = config.get('verbose', 1)

        # Offer to save as template or start training
        keyboard = [
            [InlineKeyboardButton("🚀 Start Training", callback_data="start_training")],
            [InlineKeyboardButton("💾 Save as Template", callback_data="template_save")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = (
            "✅ **Keras Configuration Complete**\n\n"
            f"📊 **Settings:**\n"
            f"• Epochs: {epochs}\n"
            f"• Batch Size: {batch_size}\n"
            f"• Initializer: {escape_markdown_v1(initializer)}\n"
            f"• Verbosity: {verbose}\n"
            f"• Validation Split: {validation_split * 100:.0f}%\n\n"
            f"What would you like to do next?"
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
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

        # Move to max_depth selection
        keyboard = [
            [InlineKeyboardButton("3 levels", callback_data="xgboost_max_depth:3")],
            [InlineKeyboardButton("6 levels (recommended)", callback_data="xgboost_max_depth:6")],
            [InlineKeyboardButton("9 levels", callback_data="xgboost_max_depth:9")],
            [InlineKeyboardButton("Custom", callback_data="xgboost_max_depth:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **XGBoost Configuration**\n\n"
            "**Step 2/5: Maximum Tree Depth**\n\n"
            "How deep should each tree be?\n"
            "(Deeper = more complex patterns, risk of overfitting)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        if max_depth_value == "custom":
            max_depth = 6
        else:
            max_depth = int(max_depth_value)

        session.selections['xgboost_config']['max_depth'] = max_depth
        await self.state_manager.update_session(session)

        # Move to learning_rate selection
        keyboard = [
            [InlineKeyboardButton("0.01 (conservative)", callback_data="xgboost_learning_rate:0.01")],
            [InlineKeyboardButton("0.1 (recommended)", callback_data="xgboost_learning_rate:0.1")],
            [InlineKeyboardButton("0.3 (aggressive)", callback_data="xgboost_learning_rate:0.3")],
            [InlineKeyboardButton("Custom", callback_data="xgboost_learning_rate:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **XGBoost Configuration**\n\n"
            "**Step 3/5: Learning Rate**\n\n"
            "How fast should the model learn?\n"
            "(Lower = more stable, but needs more trees)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        if learning_rate_value == "custom":
            learning_rate = 0.1
        else:
            learning_rate = float(learning_rate_value)

        session.selections['xgboost_config']['learning_rate'] = learning_rate
        await self.state_manager.update_session(session)

        # Move to subsample selection
        keyboard = [
            [InlineKeyboardButton("0.6 (60%)", callback_data="xgboost_subsample:0.6")],
            [InlineKeyboardButton("0.8 (80% - recommended)", callback_data="xgboost_subsample:0.8")],
            [InlineKeyboardButton("1.0 (100% - all data)", callback_data="xgboost_subsample:1.0")],
            [InlineKeyboardButton("Custom", callback_data="xgboost_subsample:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **XGBoost Configuration**\n\n"
            "**Step 4/5: Subsample Ratio**\n\n"
            "What fraction of data to use per tree?\n"
            "(Lower = more diverse trees, prevents overfitting)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        if subsample_value == "custom":
            subsample = 0.8
        else:
            subsample = float(subsample_value)

        session.selections['xgboost_config']['subsample'] = subsample
        await self.state_manager.update_session(session)

        # Move to colsample_bytree selection
        keyboard = [
            [InlineKeyboardButton("0.6 (60%)", callback_data="xgboost_colsample:0.6")],
            [InlineKeyboardButton("0.8 (80% - recommended)", callback_data="xgboost_colsample:0.8")],
            [InlineKeyboardButton("1.0 (100% - all features)", callback_data="xgboost_colsample:1.0")],
            [InlineKeyboardButton("Custom", callback_data="xgboost_colsample:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **XGBoost Configuration**\n\n"
            "**Step 5/5: Column Subsample Ratio**\n\n"
            "What fraction of features to use per tree?\n"
            "(Lower = more diverse trees, prevents overfitting)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        if colsample_value == "custom":
            colsample = 0.8
        else:
            colsample = float(colsample_value)

        session.selections['xgboost_config']['colsample_bytree'] = colsample
        await self.state_manager.update_session(session)

        # Configuration complete - start training
        model_type = session.selections.get('xgboost_model_type')
        xgboost_config = session.selections.get('xgboost_config')

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        await query.edit_message_text(
            f"✅ **XGBoost Configuration Complete**\n\n"
            f"🎯 **Model**: {model_display}\n"
            f"⚙️ **Parameters**:\n"
            f"• n\\_estimators: {xgboost_config['n_estimators']}\n"
            f"• max\\_depth: {xgboost_config['max_depth']}\n"
            f"• learning\\_rate: {xgboost_config['learning_rate']}\n"
            f"• subsample: {xgboost_config['subsample']}\n"
            f"• colsample\\_bytree: {xgboost_config['colsample_bytree']}\n\n"
            f"🚀 Starting training...\n\n"
            f"This may take a few moments.",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        num_leaves = int(num_leaves_value)
        session.selections['lightgbm_config']['num_leaves'] = num_leaves
        await self.state_manager.update_session(session)

        print(f"⚡ DEBUG: LightGBM num_leaves set to {num_leaves}")

        # Move to n_estimators selection
        keyboard = [
            [InlineKeyboardButton("50 trees (fast)", callback_data="lightgbm_n_estimators:50")],
            [InlineKeyboardButton("100 trees (recommended)", callback_data="lightgbm_n_estimators:100")],
            [InlineKeyboardButton("200 trees (accurate)", callback_data="lightgbm_n_estimators:200")],
            [InlineKeyboardButton("Use Defaults", callback_data="lightgbm_use_defaults")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "⚡ **LightGBM Configuration**\n\n"
            f"✅ Num Leaves: {num_leaves}\n\n"
            "**Step 2/3: Number of Trees**\n\n"
            "How many boosting rounds?\n"
            "(More trees = better fit, but slower training)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        n_estimators = int(n_estimators_value)
        session.selections['lightgbm_config']['n_estimators'] = n_estimators
        await self.state_manager.update_session(session)

        print(f"⚡ DEBUG: LightGBM n_estimators set to {n_estimators}")

        # Get previous values for display
        num_leaves = session.selections['lightgbm_config'].get('num_leaves', 31)

        # Move to learning_rate selection
        keyboard = [
            [InlineKeyboardButton("0.01 (conservative)", callback_data="lightgbm_learning_rate:0.01")],
            [InlineKeyboardButton("0.05 (balanced)", callback_data="lightgbm_learning_rate:0.05")],
            [InlineKeyboardButton("0.1 (recommended)", callback_data="lightgbm_learning_rate:0.1")],
            [InlineKeyboardButton("0.2 (aggressive)", callback_data="lightgbm_learning_rate:0.2")],
            [InlineKeyboardButton("Use Defaults", callback_data="lightgbm_use_defaults")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "⚡ **LightGBM Configuration**\n\n"
            f"✅ Num Leaves: {num_leaves}\n"
            f"✅ N Estimators: {n_estimators}\n\n"
            "**Step 3/3: Learning Rate**\n\n"
            "How fast should the model learn?\n"
            "(Lower = more stable, Higher = faster but risky)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        learning_rate = float(learning_rate_value)
        session.selections['lightgbm_config']['learning_rate'] = learning_rate
        await self.state_manager.update_session(session)

        print(f"⚡ DEBUG: LightGBM learning_rate set to {learning_rate}")

        # Configuration complete - start training
        model_type = session.selections.get('lightgbm_model_type')
        lightgbm_config = session.selections.get('lightgbm_config')

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        await query.edit_message_text(
            f"✅ **LightGBM Configuration Complete**\n\n"
            f"🎯 **Model**: {model_display}\n"
            f"⚙️ **Parameters**:\n"
            f"• num\\_leaves: {lightgbm_config['num_leaves']}\n"
            f"• n\\_estimators: {lightgbm_config['n_estimators']}\n"
            f"• learning\\_rate: {lightgbm_config['learning_rate']}\n"
            f"• feature\\_fraction: {lightgbm_config.get('feature_fraction', 0.8)}\n"
            f"• bagging\\_fraction: {lightgbm_config.get('bagging_fraction', 0.8)}\n\n"
            f"🚀 Starting training...\n\n"
            f"This may take a few moments.",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        # Use the default config that was already loaded
        model_type = session.selections.get('lightgbm_model_type')
        lightgbm_config = session.selections.get('lightgbm_config')

        print(f"⚡ DEBUG: Using LightGBM defaults: {lightgbm_config}")

        # Escape underscores for Markdown
        model_display = model_type.replace('_', '\\_')

        await query.edit_message_text(
            f"✅ **Using Default LightGBM Configuration**\n\n"
            f"🎯 **Model**: {model_display}\n"
            f"⚙️ **Parameters** (defaults):\n"
            f"• num\\_leaves: {lightgbm_config['num_leaves']}\n"
            f"• n\\_estimators: {lightgbm_config['n_estimators']}\n"
            f"• learning\\_rate: {lightgbm_config['learning_rate']}\n"
            f"• feature\\_fraction: {lightgbm_config.get('feature_fraction', 0.8)}\n"
            f"• bagging\\_fraction: {lightgbm_config.get('bagging_fraction', 0.8)}\n\n"
            f"🚀 Starting training...\n\n"
            f"This may take a few moments.",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        # Handle custom input (default to 1000 for now - Phase 2 enhancement)
        if iterations_value == "custom":
            iterations = 1000
        else:
            iterations = int(iterations_value)

        session.selections['catboost_config']['iterations'] = iterations
        await self.state_manager.update_session(session)

        # Move to depth selection
        keyboard = [
            [InlineKeyboardButton("4 levels (fast)", callback_data="catboost_depth:4")],
            [InlineKeyboardButton("6 levels (recommended)", callback_data="catboost_depth:6")],
            [InlineKeyboardButton("8 levels (complex)", callback_data="catboost_depth:8")],
            [InlineKeyboardButton("Custom", callback_data="catboost_depth:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🐈 **CatBoost Configuration**\n\n"
            "**Step 2/4: Maximum Tree Depth**\n\n"
            "How deep should each tree be?\n"
            "(Deeper = more complex patterns, risk of overfitting)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        if depth_value == "custom":
            depth = 6
        else:
            depth = int(depth_value)

        session.selections['catboost_config']['depth'] = depth
        await self.state_manager.update_session(session)

        # Move to learning_rate selection
        keyboard = [
            [InlineKeyboardButton("0.01 (conservative)", callback_data="catboost_learning_rate:0.01")],
            [InlineKeyboardButton("0.03 (recommended)", callback_data="catboost_learning_rate:0.03")],
            [InlineKeyboardButton("0.1 (aggressive)", callback_data="catboost_learning_rate:0.1")],
            [InlineKeyboardButton("Custom", callback_data="catboost_learning_rate:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🐈 **CatBoost Configuration**\n\n"
            "**Step 3/4: Learning Rate**\n\n"
            "How fast should the model learn?\n"
            "(Lower = more stable, but needs more iterations)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        if learning_rate_value == "custom":
            learning_rate = 0.03
        else:
            learning_rate = float(learning_rate_value)

        session.selections['catboost_config']['learning_rate'] = learning_rate
        await self.state_manager.update_session(session)

        # Move to l2_leaf_reg selection
        keyboard = [
            [InlineKeyboardButton("1 (low regularization)", callback_data="catboost_l2:1")],
            [InlineKeyboardButton("3 (recommended)", callback_data="catboost_l2:3")],
            [InlineKeyboardButton("5 (high regularization)", callback_data="catboost_l2:5")],
            [InlineKeyboardButton("Custom", callback_data="catboost_l2:custom")]
        ]
        add_back_button(keyboard)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🐈 **CatBoost Configuration**\n\n"
            "**Step 4/4: L2 Regularization**\n\n"
            "How much regularization to prevent overfitting?\n"
            "(Higher = simpler model, lower risk of overfitting)",
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
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

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

        await query.edit_message_text(
            f"✅ **CatBoost Configuration Complete**\n\n"
            f"🎯 **Model**: {model_display}\n"
            f"⚙️ **Parameters**:\n"
            f"• iterations: {catboost_config['iterations']}\n"
            f"• depth: {catboost_config['depth']}\n"
            f"• learning\\_rate: {catboost_config['learning_rate']}\n"
            f"• l2\\_leaf\\_reg: {catboost_config['l2_leaf_reg']}\n\n"
            f"🚀 Starting training...\n\n"
            f"This may take a few moments.",
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
            await query.edit_message_text("🚀 Starting training...\n\nThis may take a few moments.", parse_mode="Markdown")
        except telegram.error.BadRequest as e:
            logger.warning(f"Failed to edit message in handle_training_execution: {e}")
            # Fallback: send new message instead
            await update.effective_message.reply_text(
                "🚀 Starting training...\n\nThis may take a few moments.",
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
            print(f"🚀 DEBUG: File path={file_path}, config={config}")

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

                # Store model_id in session for naming workflow
                session.selections['pending_model_id'] = model_id
                await self.state_manager.update_session(session)

                # Show naming options with inline keyboard
                # NOTE: callback_data is short (no model_id) to stay within Telegram's 64-byte limit
                # model_id is already stored in session.selections['pending_model_id'] at line 1831
                keyboard = [
                    [InlineKeyboardButton("📝 Name Model", callback_data="name_model")],
                    [InlineKeyboardButton("⏭️ Skip - Use Default", callback_data="skip_naming")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.effective_message.reply_text(
                    f"✅ **Training Complete!**\n\n"
                    f"📊 **Metrics:**\n{metrics_text}\n\n"
                    f"🆔 **Model ID:** `{model_id}`\n\n"
                    f"Would you like to give this model a custom name?",
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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        # Retrieve model_id from session (stored at training completion)
        model_id = session.selections.get('pending_model_id')
        if not model_id:
            logger.error("No pending_model_id found in session")
            await query.edit_message_text(
                "❌ **Session Error**\n\nModel ID not found. Please restart with /train",
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
            "📝 **Name Your Model**\n\n"
            "Choose a memorable name for your model.\n\n"
            "**Rules:**\n"
            "• 3-100 characters\n"
            "• Letters, numbers, spaces, hyphens, underscores only\n\n"
            "**Examples:**\n"
            "• `Housing Price Predictor`\n"
            "• `Customer Churn Model v2`\n"
            "• `Sales Forecast 2025`\n\n"
            "💬 **Type your custom name:**",
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
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            await update.message.reply_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        # Only process if in NAMING_MODEL state
        if session.current_state != MLTrainingState.NAMING_MODEL.value:
            # Not in naming state, ignore this text input
            return

        # Get pending model_id from session
        model_id = session.selections.get('pending_model_id')

        if not model_id:
            await update.message.reply_text(
                "❌ **Error**\n\nModel ID not found. Please restart with /train",
                parse_mode="Markdown"
            )
            return

        # Set custom name using ML Engine
        try:
            self.ml_engine.set_model_name(user_id, model_id, custom_name)

            # Transition to MODEL_NAMED state
            await self.state_manager.transition_state(
                session,
                MLTrainingState.MODEL_NAMED.value
            )

            # Get model info for confirmation
            model_info = self.ml_engine.get_model_info(user_id, model_id)

            # Send success confirmation
            await update.message.reply_text(
                f"✅ **Model Named Successfully!**\n\n"
                f"📝 **Name:** {escape_markdown_v1(custom_name)}\n"
                f"🆔 **Model ID:** `{model_id}`\n"
                f"🎯 **Type:** {model_info.get('model_type', 'N/A')}\n\n"
                f"💾 Your model is ready for predictions!",
                parse_mode="Markdown"
            )

            # Complete workflow after successful naming
            await self.state_manager.transition_state(session, MLTrainingState.COMPLETE.value)
            await self.state_manager.complete_workflow(user_id)

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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease start over with /train",
                parse_mode="Markdown"
            )
            return

        # Retrieve model_id from session (stored at training completion)
        model_id = session.selections.get('pending_model_id')
        if not model_id:
            logger.error("No pending_model_id found in session")
            await query.edit_message_text(
                "❌ **Session Error**\n\nModel ID not found. Please restart with /train",
                parse_mode="Markdown"
            )
            return

        try:
            # Get model info to generate default name
            model_info = self.ml_engine.get_model_info(user_id, model_id)

            # Generate default name
            default_name = self.ml_engine._generate_default_name(
                model_type=model_info.get('model_type', 'model'),
                task_type=model_info.get('task_type', 'unknown'),
                created_at=model_info.get('created_at', '')
            )

            # Set default name
            self.ml_engine.set_model_name(user_id, model_id, default_name)

            # Transition to MODEL_NAMED state
            await self.state_manager.transition_state(
                session,
                MLTrainingState.MODEL_NAMED.value
            )

            # Send confirmation
            await query.edit_message_text(
                f"✅ **Model Ready!**\n\n"
                f"📝 **Default Name:** {escape_markdown_v1(default_name)}\n"
                f"🆔 **Model ID:** `{model_id}`\n"
                f"🎯 **Type:** {model_info.get('model_type', 'N/A')}\n\n"
                f"💾 Your model is ready for predictions!",
                parse_mode="Markdown"
            )

            # Complete workflow after successful skip naming
            await self.state_manager.transition_state(session, MLTrainingState.COMPLETE.value)
            await self.state_manager.complete_workflow(user_id)

        except Exception as e:
            logger.error(f"Error setting default model name: {e}")
            await query.edit_message_text(
                f"❌ **Error**\n\nFailed to set default name. Model ID: `{model_id}`",
                parse_mode="Markdown"
            )
            # Cancel workflow after naming error
            await self.state_manager.cancel_workflow(session)

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
            try:
                await query.edit_message_text(
                    "❌ **Invalid Request**\n\nPlease restart with /train",
                    parse_mode="Markdown"
                )
            except Exception:
                if update and update.effective_message:
                    await update.effective_message.reply_text(
                        "❌ Please restart with /train",
                        parse_mode="Markdown"
                    )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

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
                LocalPathMessages.schema_accepted_message(suggested_target),
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
                LocalPathMessages.schema_rejected_message(),
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
    from src.bot.handlers import handle_workflow_back
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
