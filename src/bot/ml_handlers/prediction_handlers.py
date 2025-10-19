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

from src.core.state_manager import StateManager, MLPredictionState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.exceptions import PathValidationError, DataError, ValidationError
from src.utils.path_validator import PathValidator
from src.bot.messages import prediction_messages
from src.bot.messages.prediction_messages import (
    PredictionMessages,
    create_data_source_buttons,
    create_schema_confirmation_buttons,
    create_column_confirmation_buttons,
    create_ready_to_run_buttons,
    create_model_selection_buttons,
    create_path_error_recovery_buttons
)
from src.bot.messages.local_path_messages import add_back_button
from src.bot.utils.markdown_escape import escape_markdown_v1
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig

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
                    "❌ **Invalid Request**\n\nPlease try /predict again.",
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

        # Show start message
        await update.message.reply_text(
            PredictionMessages.prediction_start_message(),
            parse_mode="Markdown"
        )

        # Immediately transition to data source selection
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.CHOOSING_DATA_SOURCE.value
        )

        # Show data source selection
        keyboard = create_data_source_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            PredictionMessages.data_source_selection_prompt(),
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
                "❌ **Invalid Request**\n\nPlease restart with /predict",
                parse_mode="Markdown"
            )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if choice == "upload":
            # User chose Telegram upload
            session.data_source = "telegram"
            session.save_state_snapshot()

            await self.state_manager.transition_state(
                session,
                MLPredictionState.AWAITING_FILE_UPLOAD.value
            )

            await query.edit_message_text(
                PredictionMessages.telegram_upload_prompt(),
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
                PredictionMessages.file_path_input_prompt(allowed_dirs),
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

        if session is None or session.current_state != MLPredictionState.AWAITING_FILE_UPLOAD.value:
            return

        loading_msg = await update.message.reply_text(
            PredictionMessages.loading_data_message()
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
                    str(e)
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

        if session is None or session.current_state != MLPredictionState.AWAITING_FILE_PATH.value:
            return

        validating_msg = await update.message.reply_text("🔍 **Validating path...**")

        try:
            # Validate path
            result = self.path_validator.validate_path(file_path)

            if not result['is_valid']:
                await safe_delete_message(validating_msg)
                # Add recovery buttons to help user recover from validation error
                keyboard = create_path_error_recovery_buttons()
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    PredictionMessages.file_loading_error(file_path, result['error']),
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
                # Stay in AWAITING_FILE_PATH state to allow retry
                return

            # Store path and file size
            resolved_path = result['resolved_path']
            session.file_path = str(resolved_path)

            # Get file size for load option prompt
            from src.utils.path_validator import get_file_size_mb
            size_mb = get_file_size_mb(resolved_path)

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
                    f"❌ **State Transition Failed**\n\n"
                    f"Error: {error_msg}\n"
                    f"Missing prerequisites: {missing_str}\n\n"
                    f"Please try again or use /predict to restart.",
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

            keyboard = create_load_option_buttons()
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(str(resolved_path), size_mb),
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
                PredictionMessages.file_loading_error(file_path, str(e)),
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
                "❌ **Invalid Request**\n\nPlease restart with /predict",
                parse_mode="Markdown"
            )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if choice == "immediate":
            # LOAD NOW PATH
            session.load_deferred = False

            loading_msg = await query.edit_message_text(
                PredictionMessages.loading_data_message()
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
                        f"❌ **Transition Failed**\n\n"
                        f"Error: {error_msg}\n"
                        f"Missing: {missing_str}\n\n"
                        f"Please try again or use /predict to restart.",
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
                    PredictionMessages.file_loading_error(session.file_path, str(e)),
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
                    f"❌ **Transition Failed**\n\n"
                    f"Error: {error_msg}\n"
                    f"Missing: {missing_str}\n\n"
                    f"Please try again or use /predict to restart.",
                    parse_mode="Markdown"
                )
                logger.error(
                    f"Transition to AWAITING_FEATURE_SELECTION failed: {error_msg} | Missing: {missing}"
                )
                return

            await query.edit_message_text(
                PredictionMessages.deferred_loading_confirmed_message(),
                parse_mode="Markdown"
            )

            # Show feature selection prompt (no preview since data not loaded)
            await update.effective_message.reply_text(
                PredictionMessages.feature_selection_prompt_no_preview(),
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
        # Generate summary
        summary = (
            f"📊 **Dataset Loaded**\n\n"
            f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            f"**Memory:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n"
        )

        available_columns = df.columns.tolist()

        keyboard = create_schema_confirmation_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        if update.callback_query:
            await update.callback_query.edit_message_text(
                PredictionMessages.schema_confirmation_prompt(summary, available_columns),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                PredictionMessages.schema_confirmation_prompt(summary, available_columns),
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

        # BUG FIX: Validate we're in the correct state before processing
        if session.current_state != MLPredictionState.CONFIRMING_SCHEMA.value:
            await query.edit_message_text(
                f"❌ **Invalid State**\n\n"
                f"Expected state: `CONFIRMING_SCHEMA`\n"
                f"Current state: `{session.current_state}`\n\n"
                f"Please use /predict to restart the workflow.",
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
                    f"❌ **Transition Failed**\n\n"
                    f"Error: {error_msg}\n"
                    f"Missing: {missing_str}\n\n"
                    f"Please try again or use /predict to restart.",
                    parse_mode="Markdown"
                )
                logger.error(
                    f"Transition to AWAITING_FEATURE_SELECTION failed: {error_msg} | Missing: {missing}"
                )
                return

            await query.edit_message_text(
                PredictionMessages.schema_accepted_message(),
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
                PredictionMessages.schema_rejected_message(),
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
        df = session.uploaded_data
        available_columns = df.columns.tolist()
        dataset_shape = df.shape

        await update.effective_message.reply_text(
            PredictionMessages.feature_selection_prompt(available_columns, dataset_shape),
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

        if session is None or session.current_state != MLPredictionState.AWAITING_FEATURE_SELECTION.value:
            return

        # =====================================================================
        # Smart Format Detection & Auto-Correction
        # =====================================================================

        # Check for "target:" prefix (not valid in prediction workflow)
        if features_input.lower().startswith('target:'):
            await update.message.reply_text(
                "❌ **Invalid Format for Predictions**\n\n"
                "Prediction data should NOT include the target column "
                "- that's what you're trying to predict!\n\n"
                "**Just list the feature columns:**\n"
                "`Attribute1, Attribute2, Attribute3, ...`\n\n"
                "Do NOT use `target:` or `features:` prefixes.",
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

            # Show helpful auto-correction message
            await update.message.reply_text(
                "ℹ️ **Format Auto-Corrected**\n\n"
                "Detected training workflow format. For predictions, "
                "just list feature names without the `features:` prefix.\n\n"
                f"**Processing:** `{features_input_clean}`",
                parse_mode="Markdown"
            )

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
                        {'invalid': invalid_features}
                    ),
                    parse_mode="Markdown"
                )
                return
        # Otherwise, accept user's features without validation - will validate during execution

        # Store selected features
        session.selections['selected_features'] = selected_features

        # Save snapshot and transition
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.SELECTING_MODEL.value
        )

        await update.message.reply_text(
            PredictionMessages.features_selected_message(selected_features),
            parse_mode="Markdown"
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
                "❌ **Model Loading Error**\n\n"
                f"Failed to load compatible models: {str(e)}\n\n"
                "**Possible causes:**\n"
                "• No trained models available\n"
                "• Model metadata corrupted\n"
                "• File permission issues\n\n"
                "**Solution:** Use /train to create a model first, then try /predict again.",
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
        user_id = session.user_id
        selected_features = session.selections.get('selected_features', [])

        # DEFENSIVE: Log model loading attempt
        logger.info(f"Loading models for user {user_id} with {len(selected_features)} features")

        # List all user models
        all_models = self.ml_engine.list_models(user_id=user_id)
        logger.debug(f"Found {len(all_models)} total models for user {user_id}")

        if not all_models:
            await update.effective_message.reply_text(
                PredictionMessages.no_models_available_error(),
                parse_mode="Markdown"
            )
            return

        # Filter models matching feature count and names
        compatible_models = []
        for model in all_models:
            model_features = model.get('feature_columns', [])
            if set(selected_features) == set(model_features):
                compatible_models.append(model)

        # Store models in session for index-based button lookup
        session.compatible_models = compatible_models

        # Show model selection
        keyboard = create_model_selection_buttons(compatible_models)
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await update.effective_message.reply_text(
                PredictionMessages.model_selection_prompt(compatible_models, selected_features),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except telegram.error.BadRequest as e:
            # Defensive: Catch markdown parsing errors (e.g., unescaped special chars)
            self.logger.error(f"Telegram markdown parse error in model selection: {e}")

            # Fallback: Send user-friendly error without markdown
            await update.effective_message.reply_text(
                "❌ Error displaying model list due to formatting issue.\n\n"
                f"Found {len(compatible_models)} compatible model(s) but cannot display details.\n\n"
                "Please contact support or try /predict again.",
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
                "❌ **Invalid Selection**\n\nPlease try again or use /predict to restart.",
                parse_mode="Markdown"
            )
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Validate session has compatible_models
        if not hasattr(session, 'compatible_models') or not session.compatible_models:
            await query.edit_message_text(
                "❌ **Session Expired**\n\nPlease restart with /predict",
                parse_mode="Markdown"
            )
            return

        # Validate index is in range
        if index >= len(session.compatible_models):
            await query.edit_message_text(
                "❌ **Invalid Selection**\n\nPlease try again.",
                parse_mode="Markdown"
            )
            return

        # Lookup model from session by index
        selected_model = session.compatible_models[index]
        model_id = selected_model['model_id']

        # Get model info (using actual model_id)
        model_info = self.ml_engine.get_model_info(user_id, model_id)

        if not model_info:
            await query.edit_message_text(
                "❌ **Model Not Found**\n\nPlease try again.",
                parse_mode="Markdown"
            )
            return

        # Validate feature match
        selected_features = session.selections.get('selected_features', [])
        model_features = model_info.get('feature_columns', [])

        if set(selected_features) != set(model_features):
            await query.edit_message_text(
                PredictionMessages.model_feature_mismatch_error(model_features, selected_features),
                parse_mode="Markdown"
            )
            return

        # Store model selection
        session.selections['selected_model_id'] = model_id
        session.selections['model_target_column'] = model_info.get('target_column')
        session.selections['model_type'] = model_info.get('model_type')

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
                model_info.get('target_column', 'Unknown')
            ),
            parse_mode="Markdown"
        )

        # Show prediction column confirmation
        await self._show_column_confirmation(update, context, session)

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
        target_column = session.selections.get('model_target_column', 'target')
        # PHASE 1: Skip existing columns check if data not loaded (deferred loading)
        existing_columns = session.uploaded_data.columns.tolist() if session.uploaded_data is not None else []

        keyboard = create_column_confirmation_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.prediction_column_prompt(target_column, existing_columns),
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

        # Use default column name (target_predicted)
        target_column = session.selections.get('model_target_column', 'target')
        prediction_column = f"{target_column}_predicted"

        # PHASE 3: Check for conflicts ONLY if data is loaded (skip if deferred)
        if session.uploaded_data is not None:
            existing_columns = session.uploaded_data.columns.tolist()
            if prediction_column in existing_columns:
                await query.edit_message_text(
                    PredictionMessages.column_name_conflict_error(prediction_column, existing_columns),
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
            PredictionMessages.column_name_confirmed_message(prediction_column),
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

        if session is None or session.current_state != MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value:
            return

        # PHASE 3: Check for conflicts ONLY if data is loaded (skip if deferred)
        if session.uploaded_data is not None:
            existing_columns = session.uploaded_data.columns.tolist()
            if column_name in existing_columns:
                await update.message.reply_text(
                    PredictionMessages.column_name_conflict_error(column_name, existing_columns),
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
            PredictionMessages.column_name_confirmed_message(column_name),
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
        model_type = session.selections.get('model_type', 'Unknown')
        target_column = session.selections.get('model_target_column', 'Unknown')
        prediction_column = session.selections.get('prediction_column_name', 'Unknown')
        n_rows = session.uploaded_data.shape[0] if session.uploaded_data is not None else "Deferred"
        n_features = len(session.selections.get('selected_features', []))

        keyboard = create_ready_to_run_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.ready_to_run_prompt(
                model_type, target_column, prediction_column, n_rows, n_features
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

        # Transition to running state
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLPredictionState.RUNNING_PREDICTION.value
        )

        await query.edit_message_text(
            PredictionMessages.running_prediction_message()
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

        await query.edit_message_text(
            "⬅️ **Going Back**\n\nLet's select a different model.",
            parse_mode="Markdown"
        )

        # Show model selection again
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

        # Re-show the path input prompt (stay in AWAITING_FILE_PATH state)
        allowed_dirs = self.data_loader.allowed_directories
        await query.edit_message_text(
            PredictionMessages.file_path_input_prompt(allowed_dirs),
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
        keyboard = create_data_source_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            PredictionMessages.data_source_selection_prompt(),
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

        # Reset session
        await self.state_manager.reset_session(session)

        await query.edit_message_text(
            "❌ **Workflow Canceled**\n\n"
            "Your prediction workflow has been canceled.\n\n"
            "**Start Over:**\n"
            "• /predict - Start new prediction\n"
            "• /train - Train a new model",
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

        # PHASE 2: Enhanced error logging at start of execution
        logger.info(f"🚀 Starting prediction execution for user {session.user_id}")
        logger.debug(f"Session state: {session.current_state}")
        logger.debug(f"Has uploaded_data: {session.uploaded_data is not None}")
        logger.debug(f"Selected model: {session.selections.get('selected_model_id')}")
        logger.debug(f"Load deferred: {getattr(session, 'load_deferred', False)}")

        try:
            # Check if data loading was deferred
            if getattr(session, 'load_deferred', False):
                loading_msg = await update.effective_message.reply_text(
                    PredictionMessages.loading_deferred_data_message(session.file_path),
                    parse_mode="Markdown"
                )

                try:
                    # Load deferred data from file path
                    df = await self.data_loader.load_from_local_path(
                        file_path=session.file_path,
                        detect_schema_flag=False
                    )
                    session.uploaded_data = df[0] if isinstance(df, tuple) else df

                    await safe_delete_message(loading_msg)

                except Exception as e:
                    logger.error(f"Error loading deferred data: {e}")
                    await loading_msg.edit_text(
                        PredictionMessages.file_loading_error(session.file_path, str(e)),
                        parse_mode="Markdown"
                    )
                    return

            # Extract parameters
            model_id = session.selections.get('selected_model_id')
            selected_features = session.selections.get('selected_features', [])
            prediction_column = session.selections.get('prediction_column_name')

            # PHASE 3: Explicit data validation before prediction execution
            if session.uploaded_data is None:
                error_msg = (
                    "❌ **Data Not Found**\n\n"
                    "Prediction data is missing from session. This can happen if:\n"
                    "• Session expired\n"
                    "• Data loading failed silently\n"
                    "• Memory limit exceeded\n\n"
                    "**Solution:** Please reload the template or restart with /predict"
                )
                await update.effective_message.reply_text(error_msg, parse_mode="Markdown")
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
                error_msg = (
                    f"❌ **Column Name Conflict**\n\n"
                    f"Column `{prediction_column}` already exists in your dataset.\n\n"
                    f"**Existing columns:** {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}\n\n"
                    f"**Solution:** Please restart /predict and choose a different column name."
                )
                await update.effective_message.reply_text(error_msg, parse_mode="Markdown")
                logger.warning(
                    f"Prediction column conflict for user {session.user_id}: "
                    f"'{prediction_column}' already in dataset columns"
                )
                return

            prediction_data = df[selected_features].copy()

            # Run prediction
            result = self.ml_engine.predict(
                user_id=session.user_id,
                model_id=model_id,
                data=prediction_data
            )

            execution_time = time.time() - start_time

            if not result.get('success', True):
                raise Exception(result.get('error', 'Unknown error'))

            # Add predictions to original dataframe
            predictions = result['predictions']
            df[prediction_column] = predictions

            # Store DataFrame with predictions in session for later save
            session.selections['predictions_result'] = df
            session.selections['prediction_stats'] = {
                'mean': float(pd.Series(predictions).mean()),
                'std': float(pd.Series(predictions).std()),
                'min': float(pd.Series(predictions).min()),
                'max': float(pd.Series(predictions).max()),
                'median': float(pd.Series(predictions).median())
            }
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
            await update.effective_message.reply_text(
                PredictionMessages.prediction_success_message(
                    session.selections.get('model_type', 'Model'),
                    len(predictions),
                    prediction_column,
                    execution_time,
                    preview_data,
                    session.selections['prediction_stats']
                ),
                parse_mode="Markdown"
            )

            # PHASE 1: Show template save prompt after prediction completes
            await self._show_template_save_prompt(update, context, session)

        except Exception as e:
            logger.error(f"Prediction execution error: {e}", exc_info=True)
            await update.effective_message.reply_text(
                PredictionMessages.prediction_error_message(str(e)),
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
        keyboard = [
            [InlineKeyboardButton("💾 Save as Template", callback_data="save_pred_template")],
            [InlineKeyboardButton("⏭️ Skip to Output", callback_data="skip_to_output")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            "**What would you like to do next?**\n\n"
            "You can save this prediction configuration as a template for quick reuse, "
            "or skip directly to output options.",
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

        keyboard = create_output_option_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.output_options_prompt(),
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

        if choice == "local":
            # Transition to AWAITING_SAVE_PATH with validation
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLPredictionState.AWAITING_SAVE_PATH.value
            )

            # Validate transition succeeded before continuing
            if not success:
                missing_str = ', '.join(missing) if missing else 'unknown'
                await query.edit_message_text(
                    f"❌ **State Transition Failed**\n\n"
                    f"Error: {error_msg}\n"
                    f"Missing prerequisites: {missing_str}\n\n"
                    f"Please try /predict to restart.",
                    parse_mode="Markdown"
                )
                logger.error(
                    f"Transition to AWAITING_SAVE_PATH failed: {error_msg} | Missing: {missing}"
                )
                return

            allowed_dirs = self.data_loader.allowed_directories
            await query.edit_message_text(
                PredictionMessages.save_path_input_prompt(allowed_dirs),
                parse_mode="Markdown"
            )

        elif choice == "telegram":
            # Send file via Telegram (legacy behavior)
            df = session.selections.get('predictions_result')
            output_path = Path(tempfile.gettempdir()) / f"predictions_{session.user_id}_{int(time.time())}.csv"
            df.to_csv(output_path, index=False)

            await query.edit_message_text("📥 **Preparing download...**")

            with open(output_path, 'rb') as f:
                await update.effective_message.reply_document(
                    document=f,
                    filename=f"predictions_{session.user_id}.csv",
                    caption="📥 **Download Complete Results**"
                )

            output_path.unlink()

            await update.effective_message.reply_text(
                PredictionMessages.workflow_complete_message(),
                parse_mode="Markdown"
            )

        elif choice == "done":
            # Skip both options
            await query.edit_message_text(
                PredictionMessages.workflow_complete_message(),
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

        if session is None or session.current_state != MLPredictionState.AWAITING_SAVE_PATH.value:
            # DEBUG: Log state mismatch for troubleshooting
            if session:
                logger.debug(
                    f"handle_save_directory_input called but state mismatch: "
                    f"expected AWAITING_SAVE_PATH, got {session.current_state}"
                )
            return

        validating_msg = await update.message.reply_text("🔍 **Validating path...**")

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

            # Validate directory and filename
            result = self.path_validator.validate_output_path(
                directory_path=directory_path,
                filename=filename
            )

            await safe_delete_message(validating_msg)

            if not result['is_valid']:
                from src.bot.messages.prediction_messages import create_path_error_recovery_buttons
                keyboard = create_path_error_recovery_buttons()
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Enhanced error logging
                logger.error(f"Path validation failed for user {user_id}: {result['error']}")

                await update.message.reply_text(
                    PredictionMessages.file_save_error_message("Path Validation", result['error']),
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
                PredictionMessages.file_save_error_message("Processing Error", str(e)),
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

        default_name = session.selections.get('save_filename')
        directory = session.selections.get('save_directory')

        keyboard = create_filename_confirmation_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.filename_confirmation_prompt(default_name, directory),
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

        if choice == "default":
            # Use default filename - execute save
            await query.edit_message_text("💾 **Saving file...**")
            await self._execute_file_save(update, context, session)

        elif choice == "custom":
            # Prompt for custom filename
            await query.edit_message_text(
                "✏️ **Custom Filename**\n\n"
                "Enter your desired filename (include .csv extension):\n\n"
                "**Example:** `my_predictions.csv`",
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
                PredictionMessages.file_save_error_message("Invalid Filename", result['error']),
                parse_mode="Markdown"
            )
            return

        # Update filename in session
        session.selections['save_filename'] = custom_filename
        session.selections['save_full_path'] = str(result['resolved_path'])

        # Execute save
        saving_msg = await update.message.reply_text("💾 **Saving file...**")
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
        try:
            # Get saved path and data
            full_path = session.selections.get('save_full_path')
            df = session.selections.get('predictions_result')

            # Save to file
            df.to_csv(full_path, index=False)

            # Transition back to COMPLETE
            await self.state_manager.transition_state(
                session,
                MLPredictionState.COMPLETE.value
            )

            # Send success message
            await update.effective_message.reply_text(
                PredictionMessages.file_save_success_message(
                    full_path,
                    len(df)
                ),
                parse_mode="Markdown"
            )

            # Show template save option
            keyboard = [
                [InlineKeyboardButton("💾 Save as Template", callback_data="save_pred_template")],
                [InlineKeyboardButton("✅ Done", callback_data="pred_output_done")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.effective_message.reply_text(
                "**What would you like to do next?**\n\n"
                "You can save this prediction configuration as a template for quick reuse.",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"File save error: {e}", exc_info=True)
            await update.effective_message.reply_text(
                PredictionMessages.file_save_error_message("Save Failed", str(e)),
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

        # Edit message to show skip confirmation
        await query.edit_message_text(
            "⏭️ **Skipped Template Save**\n\nProceeding to output options...",
            parse_mode="Markdown"
        )

        # Show output options
        await self._show_output_options(update, context, session)

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
