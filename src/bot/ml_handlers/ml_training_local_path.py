"""Telegram bot handlers for ML training with local file path workflow."""

import logging
from typing import Optional

import pandas as pd
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ApplicationHandlerStop

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.exceptions import PathValidationError, DataError, ValidationError, TrainingError
from src.utils.schema_detector import DatasetSchema
from src.bot.messages import LocalPathMessages
from src.bot.utils.markdown_escape import escape_markdown_v1
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
        data_loader: DataLoader
    ):
        """Initialize handler with state manager and data loader."""
        self.state_manager = state_manager
        self.data_loader = data_loader
        self.logger = logger

        # Initialize ML Engine for training
        ml_config = MLEngineConfig.get_default()
        self.ml_engine = MLEngine(ml_config)

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
            conversation_id=str(chat_id)
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

        # Create inline keyboard
        keyboard = [
            [InlineKeyboardButton("📤 Upload File", callback_data="data_source:telegram")],
            [InlineKeyboardButton("📂 Use Local Path", callback_data="data_source:local_path")]
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

        session = await self.state_manager.get_session(user_id, str(chat_id))

        if choice == "telegram":
            # User chose Telegram upload
            print("🔀 DEBUG: User chose telegram data source")
            session.data_source = "telegram"

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

            await query.edit_message_text(
                LocalPathMessages.file_path_input_prompt(allowed_dirs),
                parse_mode="Markdown"
            )
            print("🔀 DEBUG: File path prompt sent to user")

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

            session = await self.state_manager.get_session(user_id, str(chat_id))

            if session is None:
                print(f"❌ DEBUG: Session not found for user {user_id}")
                await update.message.reply_text(
                    "⚠️ **Session Expired**\n\nYour session has expired. Please start again with /train",
                    parse_mode="Markdown"
                )
                return

            current_state = session.current_state
            print(f"📊 DEBUG: Current session state = {current_state}")

            # Route based on current state
            if current_state == MLTrainingState.AWAITING_FILE_PATH.value:
                print("🔀 DEBUG: Routing to file path logic")
                await self._process_file_path_input(update, context, session, text_input)
            elif current_state == MLTrainingState.AWAITING_SCHEMA_INPUT.value:
                print("🔀 DEBUG: Routing to schema input logic")
                await self._process_schema_input(update, context, session, text_input)
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
                await validating_msg.delete()
                # Show validation error
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

            # Transition to choosing load option
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CHOOSING_LOAD_OPTION.value
            )

            # Delete validating message
            await validating_msg.delete()

            # Show load option selection
            keyboard = [
                [InlineKeyboardButton("🔄 Load Now", callback_data="load_option:immediate")],
                [InlineKeyboardButton("⏳ Defer Loading", callback_data="load_option:defer")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(str(resolved_path), size_mb),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

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

        session = await self.state_manager.get_session(user_id, str(chat_id))

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
        session = await self.state_manager.get_session(user_id, str(chat_id))

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
                ("Polynomial Regression", "polynomial")
            ],
            "classification": [
                ("Logistic Regression", "logistic"),
                ("Decision Tree", "decision_tree"),
                ("Random Forest", "random_forest"),
                ("Gradient Boosting", "gradient_boosting"),
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
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="model_category:back")])
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

        session = await self.state_manager.get_session(user_id, str(chat_id))

        # Store model selection
        session.selections['model_type'] = model_type

        # Determine task type based on model
        if model_type in ['linear', 'ridge', 'lasso', 'elasticnet', 'polynomial', 'mlp_regression', 'keras_regression']:
            session.selections['task_type'] = 'regression'
        elif model_type in ['logistic', 'decision_tree', 'random_forest', 'gradient_boosting', 'svm', 'naive_bayes', 'mlp_classification', 'keras_binary_classification', 'keras_multiclass_classification']:
            session.selections['task_type'] = 'classification'
        else:
            session.selections['task_type'] = 'neural_network'

        await self.state_manager.update_session(session)

        # Check if this is a Keras model - needs parameter configuration
        if model_type.startswith('keras_'):
            print(f"🧠 DEBUG: Keras model selected, starting parameter configuration")
            await self._start_keras_config(query, session)
        else:
            # Non-Keras models - start training immediately
            await query.edit_message_text(
                f"✅ **Model Selected**: {model_type}\n\n"
                f"🚀 Starting training...\n\n"
                f"This may take a few moments.",
                parse_mode="Markdown"
            )

            # TODO: Trigger actual training here
            print(f"🚀 DEBUG: Ready to start training with model={model_type}")
            print(f"🚀 DEBUG: Target={session.selections.get('target_column')}")
            print(f"🚀 DEBUG: Features={session.selections.get('feature_columns')}")
        print(f"🚀 DEBUG: File path={session.file_path}")
        print(f"🚀 DEBUG: Deferred loading={session.load_deferred}")

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
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🧠 **Keras Neural Network Configuration**\n\n"
            "**Step 1/5: Training Epochs**\n\n"
            "How many training epochs?\n"
            "(More epochs = longer training, potentially better accuracy)",
            reply_markup=reply_markup,
            parse_mode="Markdown"
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

        session = await self.state_manager.get_session(user_id, str(chat_id))

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

        session = await self.state_manager.get_session(user_id, str(chat_id))

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

        session = await self.state_manager.get_session(user_id, str(chat_id))

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

        session = await self.state_manager.get_session(user_id, str(chat_id))

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

        session = await self.state_manager.get_session(user_id, str(chat_id))

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

        message_text = (
            "✅ **Keras Configuration Complete**\n\n"
            f"📊 **Settings:**\n"
            f"• Epochs: {epochs}\n"
            f"• Batch Size: {batch_size}\n"
            f"• Initializer: {escape_markdown_v1(initializer)}\n"
            f"• Verbosity: {verbose}\n"
            f"• Validation Split: {validation_split * 100:.0f}%\n\n"
            f"🚀 Starting training..."
        )

        # Wrap message editing with error handling
        try:
            await query.edit_message_text(
                message_text,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            logger.error(f"Failed to edit message in handle_keras_validation: {e}")
            # Fallback: send new message
            await update.effective_message.reply_text(
                message_text,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Telegram API error in handle_keras_validation: {e}")
            await update.effective_message.reply_text(
                "❌ **Error Updating Message**\n\nPlease use /train to restart.",
                parse_mode="Markdown"
            )

        # Trigger actual Keras training
        print(f"🚀 DEBUG: Starting ML Engine training")
        try:
            # Extract training parameters from session
            target = session.selections.get('target_column')
            features = session.selections.get('feature_columns')
            file_path = session.file_path
            model_type = session.selections.get('model_type')

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

            # Call ML Engine to train
            result = self.ml_engine.train_model(
                file_path=file_path,  # Lazy loading from deferred file
                task_type='neural_network',
                model_type=model_type,  # 'keras_binary_classification'
                target_column=target,
                feature_columns=features,
                user_id=user_id,
                hyperparameters=hyperparameters,  # Complete hyperparameters with architecture
                test_size=1.0 - config.get('validation_split', 0.2)
            )

            print(f"🚀 DEBUG: Training result = {result}")

            # Send success message with metrics
            if result.get('success'):
                metrics_text = self._format_keras_metrics(result.get('metrics', {}))
                await update.effective_message.reply_text(
                    f"✅ **Training Complete!**\n\n"
                    f"📊 **Metrics:**\n{metrics_text}\n\n"
                    f"🆔 **Model ID:** `{result.get('model_id', 'N/A')}`\n\n"
                    f"You can now use this model for predictions.",
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

        session = await self.state_manager.get_session(user_id, str(chat_id))

        if choice == "accept":
            # Transfer detected schema to selections (required for training)
            if session.detected_schema:
                session.selections['target_column'] = session.detected_schema.get('target')
                session.selections['feature_columns'] = session.detected_schema.get('features', [])
                session.selections['detected_task_type'] = session.detected_schema.get('task_type')

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

    # Unified text message handler for file paths and schema input
    # Accepts all text including Unix paths starting with "/"
    # Internal state-based routing handles different inputs
    print("  ✓ Registering unified text input handler")
    application.add_handler(
        MessageHandler(
            filters.TEXT,
            handler.handle_text_input  # Single handler routes based on state
        )
    )

    print("✅ Local path ML training handlers registered successfully")
