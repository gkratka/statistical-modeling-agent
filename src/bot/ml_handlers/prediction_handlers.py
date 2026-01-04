"""Telegram bot handlers for ML prediction workflow."""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ApplicationHandlerStop
from telegram import error as telegram_error

from src.core.state_manager import StateManager, MLPredictionState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.exceptions import PathValidationError, DataError, ValidationError
from src.utils.path_validator import PathValidator
from src.utils.password_validator import PasswordValidator
from src.bot.messages import prediction_messages
from src.bot.messages.prediction_messages import (
    PredictionMessages,
    create_data_source_buttons,
    create_load_option_buttons,
    create_schema_confirmation_buttons,
    create_column_confirmation_buttons,
    create_ready_to_run_buttons,
    create_model_selection_buttons,
    create_path_error_recovery_buttons
)
from src.bot.messages.local_path_messages import add_back_button, LocalPathMessages
from src.bot.utils.markdown_escape import escape_markdown_v1
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.utils.i18n_manager import I18nManager
from src.utils.stats_utils import compute_dataset_stats

logger = logging.getLogger(__name__)


async def safe_delete_message(message) -> None:
    """
    Safely delete a Telegram message, ignoring errors if already deleted.

    This prevents spurious "Message to delete not found" errors when messages
    fail to send due to markdown parsing or other API errors.

    Args:
        message: Telegram Message object to delete
    """
    try:
        await message.delete()
    except Exception as e:
        # Log at debug level - message deletion errors are usually non-critical
        logger.debug(f"Message deletion failed (non-critical): {e}")


class PredictionHandler:
    """Handler for ML prediction workflow."""

    def __init__(
        self,
        state_manager: StateManager,
        data_loader: DataLoader,
        path_validator: PathValidator = None
    ):
        """Initialize handler with state manager and data loader."""
        self.state_manager = state_manager
        self.data_loader = data_loader
        self.logger = logger

        # Initialize ML Engine for predictions
        ml_config = MLEngineConfig.get_default()
        self.ml_engine = MLEngine(ml_config)

        # Initialize path validator if needed
        if path_validator is None:
            path_validator = PathValidator(
                allowed_directories=self.data_loader.allowed_directories,
                max_size_mb=self.data_loader.local_max_size_mb,
                allowed_extensions=self.data_loader.local_extensions
            )
        self.path_validator = path_validator

        # Initialize password validator
        self.password_validator = PasswordValidator()

    # =========================================================================
    # Step 1: Workflow Start
    # =========================================================================

    async def handle_start_prediction(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /predict command - start prediction workflow."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_start_prediction: {e}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    I18nManager.t('prediction.errors.malformed_update', command='/predict'),
                    parse_mode="Markdown"
                )
            return

        # Get or create session
        session = await self.state_manager.get_or_create_session(
            user_id=user_id,
            conversation_id=f"chat_{chat_id}"
        )

        # Initialize prediction workflow
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.STARTED.value
        await self.state_manager.update_session(session)

        # Extract locale from session
        locale = session.language if session.language else None

        # Show start message
        await update.message.reply_text(
            PredictionMessages.prediction_start_message(locale=locale),
            parse_mode="Markdown"
        )

        # Immediately transition to data source selection
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.CHOOSING_DATA_SOURCE.value
        )

        # Show data source selection
        keyboard = create_data_source_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            PredictionMessages.data_source_selection_prompt(locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    # =========================================================================
    # Steps 2-3: Data Loading
    # =========================================================================

    async def handle_data_source_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle data source selection callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split("_")[-1]  # "upload" or "path"
        except AttributeError as e:
            logger.error(f"Malformed update in handle_data_source_selection: {e}")
            await query.edit_message_text(
                I18nManager.t('prediction.errors.malformed_update'),
                parse_mode="Markdown"
            )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if choice == "upload":
            # User chose Telegram upload
            session.data_source = "telegram"
            session.save_state_snapshot()

            await self.state_manager.transition_state(
                session,
                MLPredictionState.AWAITING_FILE_UPLOAD.value
            )

            await query.edit_message_text(
                PredictionMessages.telegram_upload_prompt(locale=locale),
                parse_mode="Markdown"
            )

        elif choice == "path":
            # User chose local path
            session.data_source = "local_path"
            session.save_state_snapshot()

            await self.state_manager.transition_state(
                session,
                MLPredictionState.AWAITING_FILE_PATH.value
            )

            allowed_dirs = self.data_loader.allowed_directories
            await query.edit_message_text(
                PredictionMessages.file_path_input_prompt(allowed_dirs, locale=locale),
                parse_mode="Markdown"
            )

    async def handle_file_upload(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle Telegram file upload."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_file_upload: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if session is None or session.current_state != MLPredictionState.AWAITING_FILE_UPLOAD.value:
            return

        loading_msg = await update.message.reply_text(
            PredictionMessages.loading_data_message(locale=locale)
        )

        try:
            # Download file
            file = await context.bot.get_file(update.message.document.file_id)
            file_name = update.message.document.file_name

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name

            # Load data
            df = pd.read_csv(tmp_path)
            os.unlink(tmp_path)

            # Store data
            session.uploaded_data = df

            # Save snapshot and transition
            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLPredictionState.CONFIRMING_SCHEMA.value
            )

            await safe_delete_message(loading_msg)

            # Show schema confirmation
            await self._show_schema_confirmation(update, context, session, df)

        except Exception as e:
            logger.error(f"Error loading file: {e}")
            await loading_msg.edit_text(
                PredictionMessages.file_loading_error(
                    update.message.document.file_name,
                    str(e),
                    locale=locale
                ),
                parse_mode="Markdown"
            )

    async def handle_file_path_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle local file path input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            file_path = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_file_path_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if session is None or session.current_state != MLPredictionState.AWAITING_FILE_PATH.value:
            return

        validating_msg = await update.message.reply_text(
            I18nManager.t('prediction.save.validating_path', locale=locale)
        )

        try:
            from pathlib import Path

            # Check if worker is connected (for prod where file is on user's machine)
            websocket_server = context.bot_data.get('websocket_server')
            worker_connected = (
                websocket_server and
                websocket_server.worker_manager.is_user_connected(user_id)
            )

            if worker_connected:
                # Use worker to validate file (prod scenario - file is on user's machine)
                file_info = await self._get_file_info_from_worker(user_id, file_path, context)

                if file_info and file_info.get('exists'):
                    resolved_path = Path(file_info.get('file_path', file_path))
                    size_mb = file_info.get('size_mb', 0)
                else:
                    await safe_delete_message(validating_msg)
                    error_msg = file_info.get('error', f'File not found: {file_path}') if file_info else f'File not found: {file_path}'
                    await update.message.reply_text(f"❌ {error_msg}")
                    return
            else:
                # No worker - use local validation (dev scenario)
                result = self.path_validator.validate_path(file_path)

                if not result['is_valid']:
                    await safe_delete_message(validating_msg)

                    # Check if error is specifically whitelist failure
                    if "not in allowed directories" in result['error'].lower():
                        # Whitelist check failed - prompt for password
                        await self._prompt_for_password(
                            update, context, session, file_path, result.get('resolved_path')
                        )
                        raise ApplicationHandlerStop
                    else:
                        # Other validation error (path traversal, size, etc.)
                        keyboard = create_path_error_recovery_buttons(locale=locale)
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        await update.message.reply_text(
                            PredictionMessages.file_loading_error(file_path, result['error'], locale=locale),
                            reply_markup=reply_markup,
                            parse_mode="Markdown"
                        )
                        # Stay in AWAITING_FILE_PATH state to allow retry
                        return

                resolved_path = result['resolved_path']
                from src.utils.path_validator import get_file_size_mb
                size_mb = get_file_size_mb(resolved_path)

            # Store path and file size
            session.file_path = str(resolved_path)

            # Save snapshot and transition to load option selection
            session.save_state_snapshot()
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLPredictionState.CHOOSING_LOAD_OPTION.value
            )

            # BUG FIX: Validate transition succeeded before continuing
            if not success:
                await safe_delete_message(validating_msg)
                missing_str = ', '.join(missing) if missing else 'unknown'
                await update.message.reply_text(
                    I18nManager.t('prediction.errors.transition_failed', locale=locale, error=error_msg, missing=missing_str),
                    parse_mode="Markdown"
                )
                logger.error(
                    f"Transition to CHOOSING_LOAD_OPTION failed: {error_msg} | Missing: {missing}"
                )
                return

            await safe_delete_message(validating_msg)

            # Show load option selection
            from src.bot.messages.prediction_messages import create_load_option_buttons
            from src.bot.messages.local_path_messages import LocalPathMessages

            keyboard = create_load_option_buttons(locale=locale)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(str(resolved_path), size_mb, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            raise ApplicationHandlerStop

        except ApplicationHandlerStop:
            raise

        except Exception as e:
            logger.error(f"Error loading path: {e}")
            await safe_delete_message(validating_msg)
            await update.message.reply_text(
                PredictionMessages.file_loading_error(file_path, str(e), locale=locale),
                parse_mode="Markdown"
            )

    async def handle_load_option_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle load option selection (immediate or defer) for predictions."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split("_")[-1]  # "immediate" or "defer"
        except AttributeError as e:
            logger.error(f"Malformed update in handle_load_option_selection: {e}")
            await query.edit_message_text(
                I18nManager.t('prediction.errors.malformed_update'),
                parse_mode="Markdown"
            )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if choice == "immediate":
            # LOAD NOW PATH
            session.load_deferred = False

            loading_msg = await query.edit_message_text(
                PredictionMessages.loading_data_message(locale=locale)
            )

            try:
                # Load data from local path
                df = await self.data_loader.load_from_local_path(
                    file_path=session.file_path,
                    detect_schema_flag=False
                )
                session.uploaded_data = df[0] if isinstance(df, tuple) else df

                # Transition to schema confirmation
                session.save_state_snapshot()
                success, error_msg, missing = await self.state_manager.transition_state(
                    session,
                    MLPredictionState.CONFIRMING_SCHEMA.value
                )

                if not success:
                    missing_str = ', '.join(missing) if missing else 'unknown'
                    await loading_msg.edit_text(
                        I18nManager.t('prediction.errors.transition_failed', locale=locale, error=error_msg, missing=missing_str),
                        parse_mode="Markdown"
                    )
                    logger.error(
                        f"Transition to CONFIRMING_SCHEMA failed: {error_msg} | Missing: {missing}"
                    )
                    return

                await safe_delete_message(loading_msg)

                # Show schema confirmation
                await self._show_schema_confirmation(
                    update, context, session, session.uploaded_data
                )

            except Exception as e:
                logger.error(f"Error loading data: {e}")
                await loading_msg.edit_text(
                    PredictionMessages.file_loading_error(session.file_path, str(e), locale=locale),
                    parse_mode="Markdown"
                )

        elif choice == "defer":
            # DEFER PATH
            session.load_deferred = True

            # Skip schema confirmation, go directly to feature selection
            session.save_state_snapshot()
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLPredictionState.AWAITING_FEATURE_SELECTION.value
            )

            if not success:
                missing_str = ', '.join(missing) if missing else 'unknown'
                await query.edit_message_text(
                    I18nManager.t('prediction.errors.transition_failed', locale=locale, error=error_msg, missing=missing_str),
                    parse_mode="Markdown"
                )
                logger.error(
                    f"Transition to AWAITING_FEATURE_SELECTION failed: {error_msg} | Missing: {missing}"
                )
                return

            await query.edit_message_text(
                PredictionMessages.deferred_loading_confirmed_message(locale=locale),
                parse_mode="Markdown"
            )

            # Show feature selection prompt (no preview since data not loaded)
            await update.effective_message.reply_text(
                PredictionMessages.feature_selection_prompt_no_preview(locale=locale),
                parse_mode="Markdown"
            )

    async def _show_schema_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        df: pd.DataFrame
    ) -> None:
        """Show schema confirmation with dataset summary."""
        # Extract locale from session
        locale = session.language if session.language else None

        # Generate summary
        summary = (
            f"📊 **Dataset Loaded**\n\n"
            f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            f"**Memory:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n"
        )

        available_columns = df.columns.tolist()

        keyboard = create_schema_confirmation_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        if update.callback_query:
            await update.callback_query.edit_message_text(
                PredictionMessages.schema_confirmation_prompt(summary, available_columns, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                PredictionMessages.schema_confirmation_prompt(summary, available_columns, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    async def handle_schema_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle schema confirmation callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split("_")[-1]  # "accept" or "reject"
        except AttributeError as e:
            logger.error(f"Malformed update in handle_schema_confirmation: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # BUG FIX: Validate we're in the correct state before processing
        if session.current_state != MLPredictionState.CONFIRMING_SCHEMA.value:
            await query.edit_message_text(
                I18nManager.t('prediction.errors.invalid_state', locale=locale,
                             expected='CONFIRMING_SCHEMA', current=session.current_state),
                parse_mode="Markdown"
            )
            logger.error(
                f"Schema confirmation attempted in wrong state: {session.current_state}"
            )
            return

        if choice == "accept":
            # Schema accepted - move to feature selection
            session.save_state_snapshot()
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLPredictionState.AWAITING_FEATURE_SELECTION.value
            )

            # BUG FIX: Validate transition succeeded
            if not success:
                missing_str = ', '.join(missing) if missing else 'unknown'
                await query.edit_message_text(
                    I18nManager.t('prediction.errors.transition_failed', locale=locale, error=error_msg, missing=missing_str),
                    parse_mode="Markdown"
                )
                logger.error(
                    f"Transition to AWAITING_FEATURE_SELECTION failed: {error_msg} | Missing: {missing}"
                )
                return

            await query.edit_message_text(
                PredictionMessages.schema_accepted_message(locale=locale),
                parse_mode="Markdown"
            )

            # Show feature selection prompt
            await self._show_feature_selection(update, context, session)

        elif choice == "reject":
            # Schema rejected - go back to data source selection
            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLPredictionState.CHOOSING_DATA_SOURCE.value
            )

            session.uploaded_data = None
            session.file_path = None

            await query.edit_message_text(
                PredictionMessages.schema_rejected_message(locale=locale),
                parse_mode="Markdown"
            )

    # =========================================================================
    # Steps 4-5: Feature Selection
    # =========================================================================

    async def _show_feature_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show feature selection prompt."""
        # Extract locale from session
        locale = session.language if session.language else None

        df = session.uploaded_data
        available_columns = df.columns.tolist()
        dataset_shape = df.shape

        await update.effective_message.reply_text(
            PredictionMessages.feature_selection_prompt(available_columns, dataset_shape, locale=locale),
            parse_mode="Markdown"
        )

    async def handle_feature_selection_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle feature selection text input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            features_input = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_feature_selection_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if session is None or session.current_state != MLPredictionState.AWAITING_FEATURE_SELECTION.value:
            return

        # =====================================================================
        # Smart Format Detection & Auto-Correction
        # =====================================================================

        # Check for "target:" prefix (not valid in prediction workflow)
        if features_input.lower().startswith('target:'):
            await update.message.reply_text(
                I18nManager.t('prediction.features.invalid_format', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Auto-correct "features:" prefix (common confusion with training format)
        features_input_clean = features_input
        format_auto_corrected = False

        if features_input.lower().startswith('features:'):
            # Strip the prefix
            features_input_clean = features_input.split(':', 1)[1].strip()
            format_auto_corrected = True

            # Show helpful auto-correction message with error handling
            try:
                await update.message.reply_text(
                    I18nManager.t('prediction.features.auto_corrected', locale=locale, features=features_input_clean),
                    parse_mode="Markdown"
                )
            except telegram_error.BadRequest as e:
                logger.error(f"Telegram markdown parse error in auto-correction message: {e}")
                # Fallback: send without markdown to prevent workflow hang
                try:
                    await update.message.reply_text(
                        I18nManager.t('prediction.features.auto_corrected', locale=locale, features=features_input_clean),
                        parse_mode=None
                    )
                except Exception as fallback_error:
                    logger.error(f"Failed to send auto-correction message even without markdown: {fallback_error}")
                    # Continue workflow anyway - auto-correction message is informational only

        # Parse features (comma-separated)
        selected_features = [f.strip() for f in features_input_clean.split(',')]

        # Validate features exist in dataset (skip if data loading deferred)
        if not getattr(session, 'load_deferred', False):
            df = session.uploaded_data
            invalid_features = [f for f in selected_features if f not in df.columns]

            if invalid_features:
                await update.message.reply_text(
                    PredictionMessages.feature_validation_error(
                        "Some features are not in the dataset.",
                        {'invalid': invalid_features},
                        locale=locale
                    ),
                    parse_mode="Markdown"
                )
                return
        # Otherwise, accept user's features without validation - will validate during execution

        # Store selected features
        session.selections['selected_features'] = selected_features

        # Save snapshot and transition with validation
        session.save_state_snapshot()
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLPredictionState.SELECTING_MODEL.value
        )

        if not success:
            missing_str = ', '.join(missing) if missing else 'unknown'
            await update.message.reply_text(
                I18nManager.t('prediction.errors.transition_failed',
                             locale=locale, error=error_msg, missing=missing_str),
                parse_mode="Markdown"
            )
            logger.error(f"Transition to SELECTING_MODEL failed: {error_msg} | Missing: {missing}")
            return

        # Send feature selection confirmation with error handling
        try:
            await update.message.reply_text(
                PredictionMessages.features_selected_message(selected_features, locale=locale),
                parse_mode="Markdown"
            )
        except telegram_error.BadRequest as e:
            logger.error(f"Telegram markdown parse error: {e}")
            # Fallback: send without markdown to prevent workflow hang
            await update.message.reply_text(
                PredictionMessages.features_selected_message(selected_features, locale=locale),
                parse_mode=None
            )

        # Show model selection WITH ERROR HANDLING
        try:
            await self._show_model_selection(update, context, session)
        except Exception as e:
            logger.error(f"Error loading compatible models: {e}", exc_info=True)

            # Rollback state transition
            session.restore_previous_state()
            await self.state_manager.update_session(session)

            # Show user-friendly error
            await update.message.reply_text(
                I18nManager.t('prediction.errors.model_loading_error', locale=locale, error=str(e)),
                parse_mode="Markdown"
            )
            return  # Stop execution

        raise ApplicationHandlerStop

    # =========================================================================
    # Steps 6-7: Model Selection
    # =========================================================================

    async def _show_model_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show compatible models for selection."""
        # Extract locale from session
        locale = session.language if session.language else None

        user_id = session.user_id
        selected_features = session.selections.get('selected_features', [])

        # DEFENSIVE: Log model loading attempt
        logger.info(f"Loading models for user {user_id} with {len(selected_features)} features")

        # Check if worker is connected for model listing
        websocket_server = context.bot_data.get('websocket_server')
        worker_connected = (
            websocket_server and
            websocket_server.worker_manager.is_user_connected(user_id)
        )

        # List all user models (route through worker if connected)
        if worker_connected:
            # Use worker (for prod where models are on user's machine)
            all_models = await self._list_models_from_worker(user_id, context)
            logger.info(f"Listed {len(all_models)} models from worker for user {user_id}")
        else:
            # Use local ml_engine (for dev mode)
            all_models = self.ml_engine.list_models(user_id=user_id)
            logger.debug(f"Found {len(all_models)} total models for user {user_id}")

        if not all_models:
            await update.effective_message.reply_text(
                PredictionMessages.no_models_available_error(locale=locale),
                parse_mode="Markdown"
            )
            return

        # Filter models matching feature count and names
        compatible_models = []
        for model in all_models:
            model_features = model.get('feature_columns', [])
            if set(selected_features) == set(model_features):
                compatible_models.append(model)

        # Store models in session.selections for persistence (enables back button)
        session.selections['compatible_models'] = compatible_models

        # Show model selection
        keyboard = create_model_selection_buttons(compatible_models, locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await update.effective_message.reply_text(
                PredictionMessages.model_selection_prompt(compatible_models, selected_features, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            # Defensive: Catch markdown parsing errors (e.g., unescaped special chars)
            self.logger.error(f"Telegram markdown parse error in model selection: {e}")

            # Fallback: Send user-friendly error without markdown
            await update.effective_message.reply_text(
                I18nManager.t('prediction.errors.model_display_error', locale=locale, count=len(compatible_models)),
                reply_markup=reply_markup  # Still show selection buttons if possible
            )

    async def handle_model_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle model selection callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            # Parse index from callback_data (format: pred_model_{index})
            index = int(query.data.split("pred_model_")[-1])
        except (AttributeError, ValueError) as e:
            logger.error(f"Malformed update in handle_model_selection: {e}")
            await query.edit_message_text(
                I18nManager.t('prediction.errors.invalid_selection'),
                parse_mode="Markdown"
            )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Validate session has compatible_models
        compatible_models = session.selections.get('compatible_models', [])
        if not compatible_models:
            await query.edit_message_text(
                I18nManager.t('prediction.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Validate index is in range
        if index >= len(compatible_models):
            await query.edit_message_text(
                I18nManager.t('prediction.errors.invalid_selection', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Lookup model from session by index
        selected_model = compatible_models[index]
        model_id = selected_model['model_id']

        # Use model info from session (already populated from worker or local list_models)
        model_info = selected_model

        if not model_info:
            await query.edit_message_text(
                I18nManager.t('prediction.errors.model_not_found', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Validate feature match
        selected_features = session.selections.get('selected_features', [])
        model_features = model_info.get('feature_columns', [])

        if set(selected_features) != set(model_features):
            await query.edit_message_text(
                PredictionMessages.model_feature_mismatch_error(model_features, selected_features, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Store model selection
        session.selections['selected_model_id'] = model_id
        session.selections['model_target_column'] = model_info.get('target_column')
        session.selections['model_type'] = model_info.get('model_type')
        session.selections['task_type'] = model_info.get('task_type', 'classification')

        # Save snapshot and transition
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value
        )

        await query.edit_message_text(
            PredictionMessages.model_selected_message(
                model_id,
                model_info.get('model_type', 'Unknown'),
                model_info.get('target_column', 'Unknown'),
                locale=locale
            ),
            parse_mode="Markdown"
        )

        # Show prediction column confirmation
        await self._show_column_confirmation(update, context, session)

    # =========================================================================
    # Delete Models Workflow
    # =========================================================================

    async def handle_delete_models_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle entering delete models mode - show checkbox selection."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_delete_models_start: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        locale = session.language if session.language else None

        # Get compatible models from session
        compatible_models = session.selections.get('compatible_models', [])
        if not compatible_models:
            await query.edit_message_text(
                I18nManager.t('prediction.errors.session_expired', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Initialize empty selection set
        session.delete_selected_indices = set()
        await self.state_manager.update_session(session)

        # Show checkbox selection UI
        from src.bot.messages.prediction_messages import create_delete_models_checkbox_buttons
        keyboard = create_delete_models_checkbox_buttons(
            compatible_models,
            session.delete_selected_indices,
            locale=locale
        )
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            I18nManager.t('prediction.delete_models.title', locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_delete_model_toggle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle toggling model selection for deletion."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            # Parse index from callback_data (format: pred_delete_toggle_{index})
            index = int(query.data.split("pred_delete_toggle_")[-1])
        except (AttributeError, ValueError) as e:
            logger.error(f"Malformed update in handle_delete_model_toggle: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        locale = session.language if session.language else None

        # Toggle selection
        if not hasattr(session, 'delete_selected_indices'):
            session.delete_selected_indices = set()

        if index in session.delete_selected_indices:
            session.delete_selected_indices.remove(index)
        else:
            session.delete_selected_indices.add(index)

        await self.state_manager.update_session(session)

        # Refresh checkbox UI
        compatible_models = session.selections.get('compatible_models', [])
        from src.bot.messages.prediction_messages import create_delete_models_checkbox_buttons
        keyboard = create_delete_models_checkbox_buttons(
            compatible_models,
            session.delete_selected_indices,
            locale=locale
        )
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_reply_markup(reply_markup=reply_markup)

    async def handle_delete_models_confirm(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle confirming model deletion."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_delete_models_confirm: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        locale = session.language if session.language else None

        # Check if any models selected
        if not hasattr(session, 'delete_selected_indices') or not session.delete_selected_indices:
            await query.edit_message_text(
                I18nManager.t('prediction.delete_models.none_selected', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Check worker connection status for routing
        websocket_server = context.bot_data.get('websocket_server')
        worker_connected = (
            websocket_server and
            websocket_server.worker_manager.is_user_connected(user_id)
        )

        # Delete selected models
        compatible_models = session.selections.get('compatible_models', [])
        deleted_count = 0
        for index in session.delete_selected_indices:
            if index < len(compatible_models):
                model = compatible_models[index]
                model_id = model.get('model_id')
                try:
                    if worker_connected:
                        # Route deletion through worker (models on user's machine)
                        success = await self._delete_model_through_worker(
                            user_id, model_id, context
                        )
                        if success:
                            deleted_count += 1
                            logger.info(f"Deleted model {model_id} for user {user_id} (via worker)")
                        else:
                            logger.error(f"Worker failed to delete model {model_id}")
                    else:
                        # Local deletion (dev mode)
                        self.ml_engine.delete_model(user_id, model_id)
                        deleted_count += 1
                        logger.info(f"Deleted model {model_id} for user {user_id} (local)")
                except Exception as e:
                    logger.error(f"Failed to delete model {model_id}: {e}")

        # Clear selection state
        session.delete_selected_indices = set()
        await self.state_manager.update_session(session)

        # Show success message
        await query.edit_message_text(
            I18nManager.t('prediction.delete_models.success', locale=locale, count=deleted_count),
            parse_mode="Markdown"
        )

        # Refresh model selection (reload compatible models)
        await self._show_model_selection(update, context, session)

    async def handle_delete_models_cancel(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle canceling model deletion - return to model selection."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_delete_models_cancel: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Clear selection state
        if hasattr(session, 'delete_selected_indices'):
            session.delete_selected_indices = set()
            await self.state_manager.update_session(session)

        # Return to model selection
        await self._show_model_selection(update, context, session)

    # =========================================================================
    # Steps 8-9: Prediction Column Confirmation
    # =========================================================================

    async def _show_column_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show prediction column name confirmation."""
        # Extract locale from session
        locale = session.language if session.language else None

        target_column = session.selections.get('model_target_column', 'target')
        # PHASE 1: Skip existing columns check if data not loaded (deferred loading)
        existing_columns = session.uploaded_data.columns.tolist() if session.uploaded_data is not None else []

        keyboard = create_column_confirmation_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.prediction_column_prompt(target_column, existing_columns, locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_column_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle prediction column confirmation callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_column_confirmation: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Use default column name (target_predicted)
        target_column = session.selections.get('model_target_column', 'target')
        prediction_column = f"{target_column}_predicted"

        # PHASE 3: Check for conflicts ONLY if data is loaded (skip if deferred)
        if session.uploaded_data is not None:
            existing_columns = session.uploaded_data.columns.tolist()
            if prediction_column in existing_columns:
                await query.edit_message_text(
                    PredictionMessages.column_name_conflict_error(prediction_column, existing_columns, locale=locale),
                    parse_mode="Markdown"
                )
                return
        # If data not loaded (deferred), conflict check happens later in _execute_prediction()

        # Store column name
        session.selections['prediction_column_name'] = prediction_column

        # Save snapshot and transition
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.READY_TO_RUN.value
        )

        await query.edit_message_text(
            PredictionMessages.column_name_confirmed_message(prediction_column, locale=locale),
            parse_mode="Markdown"
        )

        # Show ready to run prompt
        await self._show_ready_to_run(update, context, session)

    async def handle_custom_column_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle custom prediction column name input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            column_name = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_custom_column_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if session is None or session.current_state != MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value:
            return

        # PHASE 3: Check for conflicts ONLY if data is loaded (skip if deferred)
        if session.uploaded_data is not None:
            existing_columns = session.uploaded_data.columns.tolist()
            if column_name in existing_columns:
                await update.message.reply_text(
                    PredictionMessages.column_name_conflict_error(column_name, existing_columns, locale=locale),
                    parse_mode="Markdown"
                )
                return
        # If data not loaded (deferred), conflict check happens later in _execute_prediction()

        # Store column name
        session.selections['prediction_column_name'] = column_name

        # Save snapshot and transition
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.READY_TO_RUN.value
        )

        await update.message.reply_text(
            PredictionMessages.column_name_confirmed_message(column_name, locale=locale),
            parse_mode="Markdown"
        )

        # Show ready to run prompt
        await self._show_ready_to_run(update, context, session)

        raise ApplicationHandlerStop

    # =========================================================================
    # Steps 10-11: Ready to Run
    # =========================================================================

    async def _show_ready_to_run(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show ready to run prompt."""
        # Extract locale from session
        locale = session.language if session.language else None

        model_type = session.selections.get('model_type', 'Unknown')
        target_column = session.selections.get('model_target_column', 'Unknown')
        prediction_column = session.selections.get('prediction_column_name', 'Unknown')
        n_rows = session.uploaded_data.shape[0] if session.uploaded_data is not None else "Deferred"
        n_features = len(session.selections.get('selected_features', []))

        keyboard = create_ready_to_run_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.ready_to_run_prompt(
                model_type, target_column, prediction_column, n_rows, n_features, locale=locale
            ),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_run_prediction(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle run prediction callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_run_prediction: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Transition to running state
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.RUNNING_PREDICTION.value
        )

        await query.edit_message_text(
            PredictionMessages.running_prediction_message(locale=locale)
        )

        # Execute prediction
        await self._execute_prediction(update, context, session)

    async def handle_go_back(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle go back callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_go_back: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Go back to model selection
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.SELECTING_MODEL.value
        )

        # Extract locale from session
        locale = session.language if session.language else None

        await query.edit_message_text(
            I18nManager.t('prediction.navigation.going_back', locale=locale),
            parse_mode="Markdown"
        )

        # Show model selection again
        await self._show_model_selection(update, context, session)

    async def handle_prediction_back(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle unified back button for prediction workflow.

        Navigates to previous step based on current state:
        - CHOOSING_DATA_SOURCE: Show "at beginning" message
        - CHOOSING_LOAD_OPTION: Go back to data source selection
        - CONFIRMING_SCHEMA: Go back to load option selection
        - AWAITING_FEATURE_SELECTION: Go back to schema confirmation
        - SELECTING_MODEL: Go back to feature selection
        - CONFIRMING_PREDICTION_COLUMN: Go back to model selection
        - READY_TO_RUN: Go back to model selection
        """
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_prediction_back: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        locale = session.language if session.language else None
        current_state = session.current_state

        logger.info(f"handle_prediction_back: current_state={current_state}")

        # Define back transitions
        back_transitions = {
            MLPredictionState.CHOOSING_LOAD_OPTION.value: MLPredictionState.CHOOSING_DATA_SOURCE.value,
            MLPredictionState.CONFIRMING_SCHEMA.value: MLPredictionState.CHOOSING_LOAD_OPTION.value,
            MLPredictionState.AWAITING_FEATURE_SELECTION.value: MLPredictionState.CONFIRMING_SCHEMA.value,
            MLPredictionState.SELECTING_MODEL.value: MLPredictionState.AWAITING_FEATURE_SELECTION.value,
            MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value: MLPredictionState.SELECTING_MODEL.value,
            MLPredictionState.READY_TO_RUN.value: MLPredictionState.SELECTING_MODEL.value,
        }

        # Handle first step: show "at beginning" message
        if current_state == MLPredictionState.CHOOSING_DATA_SOURCE.value:
            await query.answer(
                I18nManager.t('workflows.prediction.navigation.at_beginning', locale=locale),
                show_alert=True
            )
            return

        # Get target state
        target_state = back_transitions.get(current_state)
        if not target_state:
            logger.warning(f"No back transition defined for state: {current_state}")
            await query.answer(
                I18nManager.t('workflows.prediction.navigation.cannot_go_back', locale=locale),
                show_alert=True
            )
            return

        # Transition to target state
        session.save_state_snapshot()
        await self.state_manager.transition_state(session, target_state)

        # Show appropriate UI for target state
        if target_state == MLPredictionState.CHOOSING_DATA_SOURCE.value:
            # Clear previous data
            session.uploaded_data = None
            session.file_path = None
            keyboard = create_data_source_buttons(locale=locale)
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                PredictionMessages.data_source_selection_prompt(locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif target_state == MLPredictionState.CHOOSING_LOAD_OPTION.value:
            # Show load option selection
            file_path = session.file_path or "file"
            size_mb = 0
            if session.file_path:
                try:
                    size_mb = Path(session.file_path).stat().st_size / (1024 * 1024)
                except Exception:
                    pass
            keyboard = create_load_option_buttons(locale=locale)
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                LocalPathMessages.load_option_prompt(str(file_path), size_mb, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif target_state == MLPredictionState.CONFIRMING_SCHEMA.value:
            await self._show_schema_confirmation(update, context, session)

        elif target_state == MLPredictionState.AWAITING_FEATURE_SELECTION.value:
            await self._show_feature_selection(update, context, session)

        elif target_state == MLPredictionState.SELECTING_MODEL.value:
            await self._show_model_selection(update, context, session)

    # =========================================================================
    # Error Recovery Handlers
    # =========================================================================

    async def handle_retry_path(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle retry path callback - re-show path input prompt."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_retry_path: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Re-show the path input prompt (stay in AWAITING_FILE_PATH state)
        allowed_dirs = self.data_loader.allowed_directories
        await query.edit_message_text(
            PredictionMessages.file_path_input_prompt(allowed_dirs, locale=locale),
            parse_mode="Markdown"
        )

    async def handle_back_to_source(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle back to data source callback - return to data source selection."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_back_to_source: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Transition back to data source selection
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.CHOOSING_DATA_SOURCE.value
        )

        # Clear any previous data
        session.uploaded_data = None
        session.file_path = None

        # Show data source selection again
        keyboard = create_data_source_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            PredictionMessages.data_source_selection_prompt(locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_cancel_workflow(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cancel workflow callback - reset session and exit."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_cancel_workflow: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Reset session
        await self.state_manager.reset_session(session)

        await query.edit_message_text(
            I18nManager.t('prediction.navigation.workflow_cancelled', locale=locale),
            parse_mode="Markdown"
        )

    # =========================================================================
    # Steps 12-13: Execution and Results
    # =========================================================================

    async def _execute_prediction(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Execute prediction and return results."""
        start_time = time.time()

        # Extract locale from session at the beginning (used throughout function)
        locale = session.language if session.language else None

        # PHASE 2: Enhanced error logging at start of execution
        logger.info(f"🚀 Starting prediction execution for user {session.user_id}")
        logger.debug(f"Session state: {session.current_state}")
        logger.debug(f"Has uploaded_data: {session.uploaded_data is not None}")
        logger.debug(f"Selected model: {session.selections.get('selected_model_id')}")
        logger.debug(f"Load deferred: {getattr(session, 'load_deferred', False)}")

        try:
            # Extract parameters
            model_id = session.selections.get('selected_model_id')
            selected_features = session.selections.get('selected_features', [])
            prediction_column = session.selections.get('prediction_column_name')

            # Check if worker is connected for this user (local file path scenarios)
            websocket_server = context.bot_data.get('websocket_server')
            worker_connected = (
                websocket_server and
                websocket_server.worker_manager.is_user_connected(session.user_id)
            )

            # Determine if we should route to worker
            # Route to worker when: worker connected AND using local file path
            use_worker = worker_connected and session.file_path

            if use_worker:
                # Route entire prediction to worker (data loading + prediction)
                logger.info(f"Routing prediction to worker for user {session.user_id}")

                processing_msg = await update.effective_message.reply_text(
                    PredictionMessages.loading_deferred_data_message(session.file_path, locale=locale),
                    parse_mode="Markdown"
                )

                result = await self._execute_prediction_on_worker(
                    user_id=session.user_id,
                    model_id=model_id,
                    file_path=session.file_path,
                    selected_features=selected_features,
                    context=context
                )

                await safe_delete_message(processing_msg)

                if not result.get('success', False):
                    raise Exception(result.get('error', 'Worker prediction failed'))

                # Worker returns truncated predictions sample (to prevent OOM)
                predictions = result.get('predictions', [])
                total_predictions = result.get('count', len(predictions))
                df = result.get('dataframe')  # Worker returns truncated sample
                output_file = result.get('output_file')  # Full results saved here

                # Convert from JSON/dict if worker returned serialized dataframe sample
                if df is not None and not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)

                # If worker didn't return dataframe sample, create minimal one for display
                if df is None:
                    df = pd.DataFrame(columns=selected_features)
                    for i, pred in enumerate(predictions):
                        df.loc[i] = [None] * len(selected_features)

                # Store output file path for user reference
                if output_file:
                    session.selections['output_file'] = output_file
                    logger.info(f"Full predictions saved to: {output_file}")

                # Add predictions to dataframe
                df[prediction_column] = predictions

            else:
                # Local execution path (dev mode or Telegram uploads)
                # Check if data loading was deferred
                if getattr(session, 'load_deferred', False):
                    loading_msg = await update.effective_message.reply_text(
                        PredictionMessages.loading_deferred_data_message(session.file_path, locale=locale),
                        parse_mode="Markdown"
                    )

                    try:
                        # Load deferred data from file path
                        # Pass session to merge static + dynamic whitelists (password-authenticated paths)
                        loaded_df = await self.data_loader.load_from_local_path(
                            file_path=session.file_path,
                            detect_schema_flag=False,
                            session=session
                        )
                        session.uploaded_data = loaded_df[0] if isinstance(loaded_df, tuple) else loaded_df

                        await safe_delete_message(loading_msg)

                    except Exception as e:
                        logger.error(f"Error loading deferred data: {e}")
                        await loading_msg.edit_text(
                            PredictionMessages.file_loading_error(session.file_path, str(e), locale=locale),
                            parse_mode="Markdown"
                        )
                        return

                # PHASE 3: Explicit data validation before prediction execution
                if session.uploaded_data is None:
                    await update.effective_message.reply_text(
                        I18nManager.t('prediction.data_not_found', locale=locale),
                        parse_mode="Markdown"
                    )
                    logger.error(
                        f"Missing uploaded_data for user {session.user_id} during prediction execution. "
                        f"Model: {model_id}, Features: {selected_features}"
                    )
                    return

                # Prepare data (only selected features)
                # FIX: Use .copy() to prevent in-memory mutations from persisting across workflow runs
                df = session.uploaded_data.copy()

                # PHASE 2: Deferred conflict check (for workflows where data loaded late)
                if prediction_column in df.columns:
                    columns_preview = ', '.join(df.columns.tolist()[:10]) + ('...' if len(df.columns) > 10 else '')
                    await update.effective_message.reply_text(
                        I18nManager.t('prediction.column_conflict_deferred', locale=locale,
                                     column=prediction_column, columns_preview=columns_preview),
                        parse_mode="Markdown"
                    )
                    logger.warning(
                        f"Prediction column conflict for user {session.user_id}: "
                        f"'{prediction_column}' already in dataset columns"
                    )
                    return

                prediction_data = df[selected_features].copy()

                # Run prediction locally
                result = self.ml_engine.predict(
                    user_id=session.user_id,
                    model_id=model_id,
                    data=prediction_data
                )

                if not result.get('success', True):
                    raise Exception(result.get('error', 'Unknown error'))

                # Add predictions to original dataframe
                predictions = result['predictions']
                df[prediction_column] = predictions

            execution_time = time.time() - start_time

            # Store DataFrame with predictions in session for later save
            session.selections['predictions_result'] = df

            # Compute prediction stats including class distribution for classification
            task_type = session.selections.get('task_type', 'classification')
            prediction_stats = {
                'mean': float(pd.Series(predictions).mean()),
                'std': float(pd.Series(predictions).std()),
                'min': float(pd.Series(predictions).min()),
                'max': float(pd.Series(predictions).max()),
                'median': float(pd.Series(predictions).median()),
                'n_rows': len(predictions)
            }

            # Add class distribution for classification predictions
            if task_type == 'classification':
                try:
                    stats = compute_dataset_stats(df, prediction_column, task_type)
                    if 'class_distribution' in stats:
                        prediction_stats['class_distribution'] = stats['class_distribution']
                except Exception:
                    pass  # Skip if stats computation fails

            session.selections['prediction_stats'] = prediction_stats
            session.selections['execution_time'] = execution_time

            # Generate preview (first 10 rows with prediction column)
            preview_cols = selected_features[:3] + [prediction_column]
            preview_df = df[preview_cols].head(10)
            preview_data = preview_df.to_string(index=False)

            # Transition to complete
            await self.state_manager.transition_state(
                session,
                MLPredictionState.COMPLETE.value
            )

            # Send success message with statistics
            success_msg = PredictionMessages.prediction_success_message(
                session.selections.get('model_type', 'Model'),
                len(predictions),
                prediction_column,
                execution_time,
                preview_data,
                session.selections['prediction_stats'],
                locale=locale
            )

            # Add output file info if worker saved full results
            output_file = session.selections.get('output_file')
            if output_file:
                success_msg += f"\n\n📁 *Full results saved to:*\n`{output_file}`"

            await update.effective_message.reply_text(
                success_msg,
                parse_mode="Markdown"
            )

            # PHASE 1: Show template save prompt after prediction completes
            await self._show_template_save_prompt(update, context, session)

        except Exception as e:
            logger.error(f"Prediction execution error: {e}", exc_info=True)
            # Add back button so user can retry with different model
            from src.bot.messages.local_path_messages import create_back_button
            keyboard = [[create_back_button(locale=locale)]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            # Truncate long error messages to prevent Telegram API markdown errors
            error_msg = str(e)[:500] + "..." if len(str(e)) > 500 else str(e)
            await update.effective_message.reply_text(
                PredictionMessages.prediction_error_message(error_msg, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    # =========================================================================
    # NEW: Local File Save Workflow Handlers
    # =========================================================================

    async def _show_template_save_prompt(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show template save prompt after prediction completion."""
        # Extract locale from session
        locale = session.language if session.language else None

        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('prediction.template_save.button_save', locale=locale),
                callback_data="save_pred_template"
            )],
            [InlineKeyboardButton(
                I18nManager.t('prediction.template_save.button_skip', locale=locale),
                callback_data="skip_to_output"
            )]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            I18nManager.t('prediction.template_save.prompt', locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _show_output_options(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show output method selection after predictions complete."""
        from src.bot.messages.prediction_messages import create_output_option_buttons

        # Extract locale from session
        locale = session.language if session.language else None

        keyboard = create_output_option_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.output_options_prompt(locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_output_option_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle output method selection callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split("_")[-1]  # "local", "telegram", or "done"
        except AttributeError as e:
            logger.error(f"Malformed update in handle_output_option_selection: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if choice == "telegram":
            # Send file via Telegram (legacy behavior)
            df = session.selections.get('predictions_result')
            output_path = Path(tempfile.gettempdir()) / f"predictions_{session.user_id}_{int(time.time())}.csv"
            df.to_csv(output_path, index=False)

            await query.edit_message_text(
                I18nManager.t('prediction.output.preparing_download', locale=locale)
            )

            with open(output_path, 'rb') as f:
                await update.effective_message.reply_document(
                    document=f,
                    filename=f"predictions_{session.user_id}.csv",
                    caption=I18nManager.t('prediction.output.download_complete', locale=locale)
                )

            output_path.unlink()

            await update.effective_message.reply_text(
                PredictionMessages.workflow_complete_message(locale=locale),
                parse_mode="Markdown"
            )

        elif choice == "done":
            # Skip both options
            await query.edit_message_text(
                PredictionMessages.workflow_complete_message(locale=locale),
                parse_mode="Markdown"
            )

    async def handle_save_directory_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle output directory path input with smart path parsing."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            user_input = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_save_directory_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if session is None or session.current_state != MLPredictionState.AWAITING_SAVE_PATH.value:
            # DEBUG: Log state mismatch for troubleshooting
            if session:
                logger.debug(
                    f"handle_save_directory_input called but state mismatch: "
                    f"expected AWAITING_SAVE_PATH, got {session.current_state}"
                )
            return

        validating_msg = await update.message.reply_text(
            I18nManager.t('prediction.save.validating_path', locale=locale)
        )

        try:
            # Smart path parsing - detect if user provided full path or directory-only
            user_path = Path(user_input).expanduser()

            # Check if user provided full path (with filename) or directory-only
            if user_path.suffix:  # Has file extension - treat as full path
                directory_path = str(user_path.parent)
                user_filename = user_path.name
                use_user_filename = True
                logger.info(f"User {user_id} provided full path. Directory: {directory_path}, Filename: {user_filename}")
            else:  # No extension - treat as directory
                directory_path = user_input
                user_filename = None
                use_user_filename = False
                logger.info(f"User {user_id} provided directory: {directory_path}")

            # Generate default filename if needed
            model_id = session.selections.get('selected_model_id', 'model')
            timestamp = int(time.time())
            default_filename = self._generate_default_filename(model_id, timestamp)

            # Use user's filename if provided, otherwise use default
            filename = user_filename if use_user_filename else default_filename

            # Check if worker is connected - skip local validation, worker will validate
            worker_connected = self._is_worker_connected(user_id, context)

            if worker_connected:
                # Worker connected - skip local validation
                # Worker will validate path when saving
                full_path = Path(directory_path).expanduser() / filename
                logger.info(f"Worker connected for user {user_id}, skipping local path validation")
                await safe_delete_message(validating_msg)

                # Store directory and filename in session
                session.selections['save_directory'] = directory_path
                session.selections['save_filename'] = filename
                session.selections['save_full_path'] = str(full_path)
            else:
                # No worker - validate locally
                result = self.path_validator.validate_output_path(
                    directory_path=directory_path,
                    filename=filename
                )

                await safe_delete_message(validating_msg)

                if not result['is_valid']:
                    # Check if error is whitelist failure - prompt for password
                    if "not in allowed" in result['error'].lower():
                        await self._prompt_for_save_password(
                            update, context, session, directory_path, filename
                        )
                        raise ApplicationHandlerStop

                    # Other validation error - show error message
                    from src.bot.messages.prediction_messages import create_path_error_recovery_buttons
                    keyboard = create_path_error_recovery_buttons(locale=locale)
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    # Enhanced error logging
                    logger.error(f"Path validation failed for user {user_id}: {result['error']}")

                    await update.message.reply_text(
                        PredictionMessages.file_save_error_message("Path Validation", result['error'], locale=locale),
                        reply_markup=reply_markup,
                        parse_mode="Markdown"
                    )
                    return

                # Store directory and filename in session
                session.selections['save_directory'] = directory_path
                session.selections['save_filename'] = filename
                session.selections['save_full_path'] = str(result['resolved_path'])

            # Transition to filename confirmation
            await self.state_manager.transition_state(
                session,
                MLPredictionState.CONFIRMING_SAVE_FILENAME.value
            )

            # Show filename confirmation
            await self._show_filename_confirmation(update, context, session)

            raise ApplicationHandlerStop

        except ApplicationHandlerStop:
            raise
        except Exception as e:
            logger.error(f"Error processing save path for user {user_id}: {e}", exc_info=True)
            await safe_delete_message(validating_msg)
            await update.message.reply_text(
                PredictionMessages.file_save_error_message("Processing Error", str(e), locale=locale),
                parse_mode="Markdown"
            )

    async def _show_filename_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Show filename confirmation prompt."""
        from src.bot.messages.prediction_messages import create_filename_confirmation_buttons

        # Extract locale from session
        locale = session.language if session.language else None

        default_name = session.selections.get('save_filename')
        directory = session.selections.get('save_directory')

        keyboard = create_filename_confirmation_buttons(locale=locale)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.filename_confirmation_prompt(default_name, directory, locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_filename_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle filename confirmation callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data.split("_")[-1]  # "default" or "custom"
        except AttributeError as e:
            logger.error(f"Malformed update in handle_filename_confirmation: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if choice == "default":
            # Use default filename - execute save
            await query.edit_message_text(
                I18nManager.t('prediction.save.saving', locale=locale)
            )
            await self._execute_file_save(update, context, session)

        elif choice == "custom":
            # Prompt for custom filename
            await query.edit_message_text(
                I18nManager.t('prediction.save.custom_filename_prompt', locale=locale),
                parse_mode="Markdown"
            )

    async def handle_filename_custom_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle custom filename input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            custom_filename = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_filename_custom_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        if session is None or session.current_state != MLPredictionState.CONFIRMING_SAVE_FILENAME.value:
            return

        # Validate custom filename
        directory = session.selections.get('save_directory')
        result = self.path_validator.validate_output_path(
            directory_path=directory,
            filename=custom_filename
        )

        if not result['is_valid']:
            await update.message.reply_text(
                PredictionMessages.file_save_error_message("Invalid Filename", result['error'], locale=locale),
                parse_mode="Markdown"
            )
            return

        # Update filename in session
        session.selections['save_filename'] = custom_filename
        session.selections['save_full_path'] = str(result['resolved_path'])

        # Execute save
        saving_msg = await update.message.reply_text(
            I18nManager.t('prediction.save.saving', locale=locale)
        )
        await self._execute_file_save(update, context, session)
        await safe_delete_message(saving_msg)

        raise ApplicationHandlerStop

    async def _execute_file_save(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Execute file save operation."""
        # Extract locale from session
        locale = session.language if session.language else None

        try:
            # Get saved path and data
            full_path = session.selections.get('save_full_path')
            df = session.selections.get('predictions_result')
            user_id = session.user_id

            # Check if worker is connected - route save through worker
            if self._is_worker_connected(user_id, context):
                logger.info(f"Routing save to worker for user {user_id}")
                result = await self._save_file_via_worker(
                    user_id, full_path, df, context
                )

                if not result['success']:
                    raise Exception(result.get('error', 'Save via worker failed'))

                rows_saved = result.get('rows', len(df))
            else:
                # Local save (no worker connected)
                df.to_csv(full_path, index=False)
                rows_saved = len(df)

            # Transition back to COMPLETE
            await self.state_manager.transition_state(
                session,
                MLPredictionState.COMPLETE.value
            )

            # Send success message
            await update.effective_message.reply_text(
                PredictionMessages.file_save_success_message(
                    full_path,
                    rows_saved,
                    locale=locale
                ),
                parse_mode="Markdown"
            )

            # Show template save option
            keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('prediction.template_save.button_save', locale=locale),
                    callback_data="save_pred_template"
                )],
                [InlineKeyboardButton(
                    I18nManager.t('workflows.prediction.buttons.done', locale=locale),
                    callback_data="pred_output_done"
                )]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.effective_message.reply_text(
                I18nManager.t('prediction.template_save.prompt', locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"File save error: {e}", exc_info=True)
            await update.effective_message.reply_text(
                PredictionMessages.file_save_error_message("Save Failed", str(e), locale=locale),
                parse_mode="Markdown"
            )

    def _generate_default_filename(self, model_id: str, timestamp: int) -> str:
        """Generate descriptive default filename."""
        from datetime import datetime

        # Extract model type from model_id (format: model_{user_id}_{type}_{timestamp})
        parts = model_id.split('_')
        model_type = parts[2] if len(parts) > 2 else 'model'

        # Format timestamp
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime("%Y%m%d_%H%M%S")

        return f"predictions_{model_type}_{date_str}.csv"

    # =========================================================================
    # PHASE 2: Template Save Skip Handler
    # =========================================================================

    async def handle_skip_to_output(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle skip button - show output options directly."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_skip_to_output: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Extract locale from session
        locale = session.language if session.language else None

        # Edit message to show skip confirmation
        await query.edit_message_text(
            I18nManager.t('prediction.template_save.skipped', locale=locale),
            parse_mode="Markdown"
        )

        # Show output options
        await self._show_output_options(update, context, session)

    # =========================================================================
    # Password Protection Handlers
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
            resolved_path: Resolved absolute path (Path object or str)
        """
        from pathlib import Path

        # Store pending path for later validation
        session.pending_auth_path = str(resolved_path)

        # Transition to password state
        session.save_state_snapshot()
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_PASSWORD.value
        )

        # Extract locale from session
        locale = session.language if session.language else None

        if not success:
            missing_str = ', '.join(missing) if missing else 'unknown'
            await update.message.reply_text(
                I18nManager.t('prediction.errors.transition_failed', locale=locale, error=error_msg, missing=missing_str),
                parse_mode="Markdown"
            )
            return

        # Get parent directory for display
        parent_dir = str(Path(resolved_path).parent)

        # Show password prompt
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('common.buttons.cancel', locale=locale),
                callback_data="pred_password_cancel"
            )]
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

        # Validate we're in password state
        if session.current_state != MLPredictionState.AWAITING_PASSWORD.value:
            return

        # Extract locale from session
        locale = session.language if session.language else None

        # Get pending path
        pending_path = session.pending_auth_path
        if not pending_path:
            await update.message.reply_text(
                I18nManager.t('prediction.errors.session_error_no_pending_path', locale=locale),
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
                    I18nManager.t('prediction.errors.session_expired', locale=locale),
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
            if hasattr(session, 'password_attempts'):
                session.password_attempts = 0

            # Store file path and get size
            session.file_path = pending_path

            # Check if worker is connected for file size lookup
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

            # Save snapshot and transition to load options
            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLPredictionState.CHOOSING_LOAD_OPTION.value
            )

            await update.message.reply_text(
                LocalPathMessages.password_success(parent_dir, locale=locale),
                parse_mode="Markdown"
            )

            # Show load options
            from src.bot.messages.prediction_messages import create_load_option_buttons
            keyboard = create_load_option_buttons(locale=locale)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(pending_path, size_mb, locale=locale),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            raise ApplicationHandlerStop

        else:
            # Password incorrect or rate limited
            if not hasattr(session, 'password_attempts'):
                session.password_attempts = 0
            session.password_attempts += 1

            if "locked" in error_msg.lower() or "maximum attempts" in error_msg.lower():
                # Rate limit exceeded - reset to file path input
                session.pending_auth_path = None
                session.password_attempts = 0

                await self.state_manager.transition_state(
                    session,
                    MLPredictionState.AWAITING_FILE_PATH.value
                )

                await update.message.reply_text(
                    LocalPathMessages.password_lockout(60, locale=locale),
                    parse_mode="Markdown"
                )
            else:
                # Show error and allow retry
                await update.message.reply_text(
                    LocalPathMessages.password_failure(error_msg, locale=locale),
                    parse_mode="Markdown"
                )

            raise ApplicationHandlerStop

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

        # Extract locale from session
        locale = session.language if session.language else None

        # Clear pending auth
        session.pending_auth_path = None
        if hasattr(session, 'password_attempts'):
            session.password_attempts = 0

        # Reset password validator attempts
        self.password_validator.reset_attempts(user_id)

        # Go back to file path input
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_FILE_PATH.value
        )

        allowed_dirs = self.data_loader.allowed_directories
        await query.edit_message_text(
            PredictionMessages.file_path_input_prompt(allowed_dirs, locale=locale),
            parse_mode="Markdown"
        )

    # =========================================================================
    # Worker File Operations
    # =========================================================================

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

    async def _list_models_from_worker(
        self,
        user_id: int,
        context: ContextTypes.DEFAULT_TYPE
    ) -> List[Dict[str, Any]]:
        """List user models from worker. Returns empty list if no worker or error.

        Used for listing models when worker is connected - models are stored
        on user's local machine, not on the bot server.

        Args:
            user_id: Telegram user ID
            context: Bot context

        Returns:
            List of model dicts if successful, empty list otherwise
        """
        from src.worker.job_queue import JobType, JobStatus
        import asyncio

        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return []

        job_queue = getattr(websocket_server, 'job_queue', None)
        worker_manager = websocket_server.worker_manager

        if not job_queue or not worker_manager.is_user_connected(user_id):
            return []

        try:
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.LIST_MODELS,
                params={},
                timeout=30.0
            )

            # Poll for result
            max_wait, poll_interval, elapsed = 30, 0.5, 0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    return []
                if job.status == JobStatus.COMPLETED:
                    return job.result.get('models', [])
                elif job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    return []

            return []
        except Exception as e:
            logger.error(f"Error listing models from worker: {e}")
            return []

    async def _delete_model_through_worker(
        self,
        user_id: int,
        model_id: str,
        context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Delete model through worker.

        Used for deleting models when worker is connected - models are stored
        on user's local machine, not on the bot server.

        Args:
            user_id: Telegram user ID
            model_id: Model ID to delete
            context: Bot context

        Returns:
            True if deletion successful, False otherwise
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
                job_type=JobType.DELETE_MODEL,
                params={"model_id": model_id},
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
                    return True
                elif job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    logger.error(f"Delete model job failed: {job.error}")
                    return False

            return False
        except Exception as e:
            logger.error(f"Error deleting model through worker: {e}")
            return False

    async def _execute_prediction_on_worker(
        self,
        user_id: int,
        model_id: str,
        file_path: str,
        selected_features: List[str],
        context: ContextTypes.DEFAULT_TYPE
    ) -> Dict[str, Any]:
        """Execute prediction via local worker.

        Sends prediction job to the worker which:
        1. Loads data from the file path
        2. Loads the model from its local storage
        3. Runs prediction
        4. Returns results

        Args:
            user_id: Telegram user ID
            model_id: Model ID to use for prediction
            file_path: Path to data file on user's machine
            selected_features: List of feature columns to use
            context: Bot context

        Returns:
            Dict with prediction results including 'success', 'predictions', etc.
        """
        from src.worker.job_queue import JobType

        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return {'success': False, 'error': 'WebSocket server not available'}

        job_queue = getattr(websocket_server, 'job_queue', None)
        if not job_queue:
            return {'success': False, 'error': 'Job queue not available'}

        try:
            # Create prediction job
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.PREDICT,
                params={
                    'model_id': model_id,
                    'file_path': file_path,
                    'feature_columns': selected_features
                },
                timeout=300.0
            )

            logger.info(f"Created prediction job {job_id} for user {user_id}")

            # Wait for result
            return await self._wait_for_prediction_job(job_id, context)

        except Exception as e:
            logger.error(f"Error executing prediction on worker: {e}")
            return {'success': False, 'error': str(e)}

    async def _wait_for_prediction_job(
        self,
        job_id: str,
        context: ContextTypes.DEFAULT_TYPE,
        max_wait: float = 300.0
    ) -> Dict[str, Any]:
        """Poll for prediction job completion.

        Args:
            job_id: Job ID to wait for
            context: Bot context
            max_wait: Maximum time to wait in seconds

        Returns:
            Dict with job results or error
        """
        from src.worker.job_queue import JobStatus
        import asyncio

        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return {'success': False, 'error': 'WebSocket server not available'}

        job_queue = getattr(websocket_server, 'job_queue', None)
        if not job_queue:
            return {'success': False, 'error': 'Job queue not available'}

        poll_interval = 0.5
        elapsed = 0

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            job = job_queue.get_job(job_id)
            if not job:
                return {'success': False, 'error': 'Job not found'}

            if job.status == JobStatus.COMPLETED:
                logger.info(f"Prediction job {job_id} completed successfully")
                # Support both new truncated format and old full format (backwards compatible)
                return {
                    'success': True,
                    'predictions': job.result.get('predictions_sample') or job.result.get('predictions', []),
                    'count': job.result.get('predictions_count') or job.result.get('count', 0),
                    'dataframe': job.result.get('dataframe_sample') or job.result.get('dataframe'),
                    'dataframe_rows': job.result.get('dataframe_rows'),
                    'dataframe_columns': job.result.get('dataframe_columns'),
                    'output_file': job.result.get('output_file'),
                }
            elif job.status == JobStatus.FAILED:
                logger.warning(f"Prediction job {job_id} failed: {job.error}")
                return {'success': False, 'error': job.error}
            elif job.status == JobStatus.TIMEOUT:
                logger.warning(f"Prediction job {job_id} timed out")
                return {'success': False, 'error': 'Prediction timed out'}

        return {'success': False, 'error': 'Exceeded maximum wait time'}

    async def _save_file_via_worker(
        self,
        user_id: int,
        file_path: str,
        dataframe,
        context: ContextTypes.DEFAULT_TYPE
    ) -> Dict[str, Any]:
        """Save file via local worker.

        Routes the save operation to the worker which validates
        the path and saves the file on the user's local machine.

        Args:
            user_id: Telegram user ID
            file_path: Full path to save file on user's machine
            dataframe: Pandas DataFrame to save
            context: Bot context

        Returns:
            Dict with 'success', 'file_path', 'rows' or 'error'
        """
        from src.worker.job_queue import JobType, JobStatus
        import asyncio

        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return {'success': False, 'error': 'WebSocket server not available'}

        job_queue = getattr(websocket_server, 'job_queue', None)
        if not job_queue:
            return {'success': False, 'error': 'Job queue not available'}

        try:
            # Convert DataFrame to dict for JSON serialization
            dataframe_dict = dataframe.to_dict('records')

            # Create save file job
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.SAVE_FILE,
                params={
                    'file_path': file_path,
                    'dataframe': dataframe_dict
                },
                timeout=60.0
            )

            logger.info(f"Created save file job {job_id} for user {user_id}")

            # Poll for result
            max_wait, poll_interval, elapsed = 60, 0.5, 0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    return {'success': False, 'error': 'Job not found'}

                if job.status == JobStatus.COMPLETED:
                    logger.info(f"Save file job {job_id} completed successfully")
                    return {
                        'success': True,
                        'file_path': job.result.get('file_path', file_path),
                        'rows': job.result.get('rows', 0)
                    }
                elif job.status == JobStatus.FAILED:
                    logger.warning(f"Save file job {job_id} failed: {job.error}")
                    return {'success': False, 'error': job.error}
                elif job.status == JobStatus.TIMEOUT:
                    logger.warning(f"Save file job {job_id} timed out")
                    return {'success': False, 'error': 'Save operation timed out'}

            return {'success': False, 'error': 'Exceeded maximum wait time'}

        except Exception as e:
            logger.error(f"Error saving file via worker: {e}")
            return {'success': False, 'error': str(e)}

    def _is_worker_connected(self, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Check if worker is connected for user.

        Args:
            user_id: Telegram user ID
            context: Bot context

        Returns:
            True if worker is connected, False otherwise
        """
        websocket_server = context.bot_data.get('websocket_server')
        if not websocket_server:
            return False

        worker_manager = getattr(websocket_server, 'worker_manager', None)
        if not worker_manager:
            return False

        return worker_manager.is_user_connected(user_id)

    # =========================================================================
    # Save Path Password Bypass
    # =========================================================================

    async def _prompt_for_save_password(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        directory_path: str,
        filename: str
    ) -> None:
        """Prompt user for password to save to non-whitelisted directory.

        Args:
            update: Telegram update
            context: Telegram context
            session: User session
            directory_path: Directory path for saving
            filename: Filename to save as
        """
        from pathlib import Path

        # Store pending save info
        session.pending_save_directory = directory_path
        session.pending_save_filename = filename

        # Transition to save password state
        session.save_state_snapshot()
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_SAVE_PASSWORD.value
        )

        locale = session.language if session.language else None

        if not success:
            missing_str = ', '.join(missing) if missing else 'unknown'
            await update.message.reply_text(
                I18nManager.t('prediction.errors.transition_failed', locale=locale, error=error_msg, missing=missing_str),
                parse_mode="Markdown"
            )
            return

        # Show password prompt
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('common.buttons.cancel', locale=locale),
                callback_data="pred_save_password_cancel"
            )]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            LocalPathMessages.password_prompt(f"{directory_path}/{filename}", directory_path, locale=locale),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_save_password_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle password input for save path access."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            password_input = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_save_password_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Validate we're in save password state
        if session.current_state != MLPredictionState.AWAITING_SAVE_PASSWORD.value:
            return

        locale = session.language if session.language else None

        # Get pending save info
        directory_path = getattr(session, 'pending_save_directory', None)
        filename = getattr(session, 'pending_save_filename', None)

        if not directory_path or not filename:
            await update.message.reply_text(
                I18nManager.t('prediction.errors.session_error_no_pending_path', locale=locale),
                parse_mode="Markdown"
            )
            return

        # Validate password
        is_valid, error_msg = self.password_validator.validate_password(
            user_id=user_id,
            password_input=password_input,
            path=directory_path
        )

        if is_valid:
            # Password correct - add directory to dynamic whitelist
            self.state_manager.add_dynamic_directory(session, directory_path)

            # Reset password attempts (but DON'T clear pending data yet - we need it)
            if hasattr(session, 'password_attempts'):
                session.password_attempts = 0

            await update.message.reply_text(
                LocalPathMessages.password_success(directory_path, locale=locale),
                parse_mode="Markdown"
            )

            # After password success, build path directly (skip whitelist re-check)
            # This matches the load handler pattern - password already verified access
            from pathlib import Path

            # Auto-fix missing leading slash (same as validate_output_path)
            if not directory_path.startswith('/') and not directory_path.startswith('./'):
                absolute_patterns = ['Users/', 'home/', 'var/', 'tmp/', 'opt/']
                for pattern in absolute_patterns:
                    if directory_path.startswith(pattern):
                        directory_path = '/' + directory_path
                        break

            # Sanitize filename
            sanitized_filename = self.path_validator.sanitize_filename(filename)
            if not sanitized_filename.lower().endswith('.csv'):
                sanitized_filename += '.csv'

            # Build full path
            dir_path = Path(directory_path).resolve()
            full_path = dir_path / sanitized_filename

            # Store path and proceed
            session.selections['save_directory'] = directory_path
            session.selections['save_filename'] = sanitized_filename
            session.selections['save_full_path'] = str(full_path)

            # Transition to filename confirmation
            await self.state_manager.transition_state(
                session,
                MLPredictionState.CONFIRMING_SAVE_FILENAME.value
            )

            # Clear pending auth
            session.pending_save_directory = None
            session.pending_save_filename = None

            # Show filename confirmation
            await self._show_filename_confirmation(update, context, session)

            raise ApplicationHandlerStop

        else:
            # Password incorrect
            if not hasattr(session, 'password_attempts'):
                session.password_attempts = 0
            session.password_attempts += 1

            if "locked" in error_msg.lower() or "maximum attempts" in error_msg.lower():
                # Rate limit - reset to save path input
                session.pending_save_directory = None
                session.pending_save_filename = None
                session.password_attempts = 0

                await self.state_manager.transition_state(
                    session,
                    MLPredictionState.AWAITING_SAVE_PATH.value
                )

                await update.message.reply_text(
                    LocalPathMessages.password_lockout(60, locale=locale),
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    LocalPathMessages.password_failure(error_msg, locale=locale),
                    parse_mode="Markdown"
                )

            raise ApplicationHandlerStop

    async def handle_save_password_cancel(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle save password cancel callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_save_password_cancel: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        locale = session.language if session.language else None

        # Clear pending save
        session.pending_save_directory = None
        session.pending_save_filename = None
        if hasattr(session, 'password_attempts'):
            session.password_attempts = 0

        # Reset password validator attempts
        self.password_validator.reset_attempts(user_id)

        # Go back to save path input
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_SAVE_PATH.value
        )

        allowed_dirs = self.data_loader.allowed_directories
        await query.edit_message_text(
            PredictionMessages.save_path_input_prompt(allowed_dirs, locale=locale),
            parse_mode="Markdown"
        )

    # =========================================================================
    # Unified Text Handler
    # =========================================================================

    async def handle_text_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Unified text input handler that routes based on current state."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            text_input = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_text_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if session is None:
            return

        # WORKFLOW ISOLATION: Only process if in PREDICTION workflow
        # Prevents collision with training workflow handler (group=1)
        if session.workflow_type != WorkflowType.ML_PREDICTION:
            return

        current_state = session.current_state

        # Route based on current state
        if current_state == MLPredictionState.AWAITING_FILE_PATH.value:
            await self.handle_file_path_input(update, context)
        elif current_state == MLPredictionState.AWAITING_PASSWORD.value:
            await self.handle_password_input(update, context)
        elif current_state == MLPredictionState.AWAITING_SAVE_PASSWORD.value:
            await self.handle_save_password_input(update, context)
        elif current_state == MLPredictionState.AWAITING_FEATURE_SELECTION.value:
            await self.handle_feature_selection_input(update, context)
        elif current_state == MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value:
            await self.handle_custom_column_input(update, context)
        elif current_state == MLPredictionState.AWAITING_SAVE_PATH.value:
            await self.handle_save_directory_input(update, context)
        elif current_state == MLPredictionState.CONFIRMING_SAVE_FILENAME.value:
            await self.handle_filename_custom_input(update, context)
        else:
            # State doesn't require text input
            return


def register_prediction_handlers(
    application,
    state_manager: StateManager,
    data_loader: DataLoader
) -> None:
    """Register prediction workflow handlers with Telegram application."""
    from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters

    logger.info("🔧 Registering ML prediction handlers...")

    handler = PredictionHandler(state_manager, data_loader)

    # Command handler
    application.add_handler(
        CommandHandler("predict", handler.handle_start_prediction)
    )

    # Callback query handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_data_source_selection,
            pattern=r"^pred_(upload|local_path)$"
        )
    )

    # NEW: Defer loading workflow
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_load_option_selection,
            pattern=r"^pred_load_(immediate|defer)$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_schema_confirmation,
            pattern=r"^pred_schema_(accept|reject)$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_model_selection,
            pattern=r"^pred_model_"
        )
    )

    # Delete models workflow handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_delete_models_start,
            pattern=r"^pred_delete_start$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_delete_model_toggle,
            pattern=r"^pred_delete_toggle_"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_delete_models_confirm,
            pattern=r"^pred_delete_confirm$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_delete_models_cancel,
            pattern=r"^pred_delete_cancel$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_column_confirmation,
            pattern=r"^pred_column_default$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_run_prediction,
            pattern=r"^pred_run$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_go_back,
            pattern=r"^pred_go_back$"
        )
    )

    # Unified back button handler for all prediction workflow steps
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_prediction_back,
            pattern=r"^pred_back$"
        )
    )

    # Password protection handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_password_cancel,
            pattern=r"^pred_password_cancel$"
        )
    )

    # Save password cancel handler
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_save_password_cancel,
            pattern=r"^pred_save_password_cancel$"
        )
    )

    # Error recovery handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_retry_path,
            pattern=r"^pred_retry_path$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_back_to_source,
            pattern=r"^pred_back_to_source$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_cancel_workflow,
            pattern=r"^pred_cancel$"
        )
    )

    # PHASE 3: Template save skip handler
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_skip_to_output,
            pattern=r"^skip_to_output$"
        )
    )

    # NEW: Local file save workflow handlers
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_output_option_selection,
            pattern=r"^pred_output_(local|telegram|done)$"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            handler.handle_filename_confirmation,
            pattern=r"^pred_filename_(default|custom)$"
        )
    )

    # File upload handler
    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handler.handle_file_upload
        )
    )

    # Text input handler (for paths, features, column names)
    # GROUP 2: Different from ML training (group 1) to prevent collision
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handler.handle_text_input
        ),
        group=2  # Separate group for prediction workflow
    )

    logger.info("✅ ML prediction handlers registered successfully")
