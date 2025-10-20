"""
Prediction Template workflow handlers.

This module provides handlers for saving and loading ML prediction templates.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackContext

from src.core.state_manager import MLPredictionState, StateManager
from src.core.prediction_template_manager import PredictionTemplateManager
from src.bot.messages import prediction_template_messages as pt_messages
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
from src.engines.ml_engine import MLEngine

logger = logging.getLogger(__name__)


class PredictionTemplateHandlers:
    """Handlers for ML prediction template operations."""

    def __init__(
        self,
        state_manager: StateManager,
        template_manager: PredictionTemplateManager,
        data_loader: DataLoader,
        path_validator: PathValidator,
        ml_engine: MLEngine
    ):
        """
        Initialize prediction template handlers.

        Args:
            state_manager: StateManager instance
            template_manager: PredictionTemplateManager instance
            data_loader: DataLoader instance
            path_validator: PathValidator instance
            ml_engine: MLEngine instance for model validation
        """
        self.state_manager = state_manager
        self.template_manager = template_manager
        self.data_loader = data_loader
        self.path_validator = path_validator
        self.ml_engine = ml_engine

    async def _get_session_or_error(self, update: Update, query_or_message) -> Optional:
        """
        Get session or send error message.

        Args:
            update: Telegram update
            query_or_message: Query or message object to reply to

        Returns:
            Session object or None if not found
        """
        user_id = update.effective_user.id

        # Get chat_id based on object type
        if isinstance(query_or_message, CallbackQuery):
            chat_id = query_or_message.message.chat_id
        else:  # Message
            chat_id = query_or_message.chat_id

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            error_msg = "❌ Session not found. Please start a new prediction session with /predict"
            if isinstance(query_or_message, CallbackQuery):
                await query_or_message.edit_message_text(error_msg)
            else:
                await query_or_message.reply_text(error_msg)
            return None

        return session

    async def _transition_or_error(
        self,
        session,
        new_state: str,
        query_or_message
    ) -> bool:
        """
        Transition state or send error message.

        Args:
            session: Session object
            new_state: New state value
            query_or_message: Query or message object to reply to

        Returns:
            True if successful, False otherwise
        """
        success, error_msg, _ = await self.state_manager.transition_state(session, new_state)
        if not success:
            error_text = f"❌ {error_msg}"
            if isinstance(query_or_message, CallbackQuery):
                await query_or_message.edit_message_text(error_text)
            else:
                await query_or_message.reply_text(error_text)
            return False
        return True

    async def _validate_and_load_template_data(self, session, query):
        """
        Validate file path and load data from template.

        Args:
            session: Session object
            query: Callback query object

        Returns:
            DataFrame if successful, None otherwise
        """
        file_path = session.file_path
        validation_result = self.path_validator.validate_path(file_path)

        if not validation_result["is_valid"]:
            await query.edit_message_text(
                pt_messages.PRED_TEMPLATE_FILE_PATH_INVALID.format(
                    path=file_path,
                    error=validation_result['error']
                ),
                parse_mode="Markdown"
            )
            return None

        try:
            df, _, _ = await self.data_loader.load_from_local_path(file_path)
            await self.state_manager.store_data(session, df)
            return df
        except Exception as e:
            logger.error(f"Error loading prediction template data: {e}", exc_info=True)
            await query.edit_message_text(
                pt_messages.PRED_TEMPLATE_LOAD_FAILED.format(error=str(e)),
                parse_mode="Markdown"
            )
            return None

    async def _show_output_options(
        self,
        update: Update,
        context: CallbackContext,
        session
    ) -> None:
        """Show output method selection after template operations."""
        from src.bot.messages.prediction_messages import PredictionMessages, create_output_option_buttons

        keyboard = create_output_option_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.output_options_prompt(),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    # =========================================================================
    # Save Template Workflow
    # =========================================================================

    async def handle_template_save_request(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle 'Save as Template' button click."""
        query = update.callback_query
        await query.answer()

        session = await self._get_session_or_error(update, query)
        if not session:
            return

        session.save_state_snapshot()

        if not await self._transition_or_error(
            session, MLPredictionState.SAVING_PRED_TEMPLATE.value, query
        ):
            return

        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_pred_template")]]
        await query.edit_message_text(
            pt_messages.PRED_TEMPLATE_SAVE_PROMPT,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_template_name_input(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template name text input."""
        user_id = update.effective_user.id
        chat_id = update.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session or session.current_state != MLPredictionState.SAVING_PRED_TEMPLATE.value:
            return

        template_name = update.message.text.strip()

        # Validate name
        is_valid, error_msg = self.template_manager.validate_template_name(template_name)
        if not is_valid:
            await update.message.reply_text(
                pt_messages.PRED_TEMPLATE_INVALID_NAME.format(error=error_msg),
                parse_mode="Markdown"
            )
            return

        # Check if exists
        if self.template_manager.template_exists(user_id, template_name):
            await update.message.reply_text(
                pt_messages.PRED_TEMPLATE_EXISTS.format(name=template_name),
                parse_mode="Markdown"
            )
            return

        # Build template config from session
        model_id = session.selections.get("selected_model_id", "")
        feature_columns = session.selections.get("selected_features", [])
        output_column_name = session.selections.get("prediction_column_name", "prediction")

        template_config = {
            "file_path": session.file_path or "",
            "model_id": model_id,
            "feature_columns": feature_columns,
            "output_column_name": output_column_name,
            "save_path": session.selections.get("output_file_path"),
            "description": None  # Could be extended to ask user for description
        }

        # Validate required fields
        if not all([template_config["file_path"], template_config["model_id"],
                   template_config["feature_columns"], template_config["output_column_name"]]):
            await update.message.reply_text(
                "❌ Cannot save template: Missing required configuration. "
                "Please complete the prediction workflow first."
            )
            return

        # Save template
        success, message = self.template_manager.save_template(
            user_id=user_id,
            template_name=template_name,
            config=template_config
        )

        if success:
            await update.message.reply_text(
                pt_messages.PRED_TEMPLATE_SAVED_SUCCESS.format(name=template_name),
                parse_mode="Markdown"
            )

            # Transition back to COMPLETE
            await self.state_manager.transition_state(
                session,
                MLPredictionState.COMPLETE.value
            )

            logger.info(f"Prediction template '{template_name}' saved for user {user_id}")

            # PHASE 4: Show output options after template save
            await self._show_output_options(update, context, session)
        else:
            await update.message.reply_text(
                pt_messages.PRED_TEMPLATE_SAVE_FAILED.format(error=message),
                parse_mode="Markdown"
            )

    # =========================================================================
    # Load Template Workflow
    # =========================================================================

    async def handle_template_source_selection(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle 'Use Template' data source selection."""
        query = update.callback_query
        await query.answer()

        session = await self._get_session_or_error(update, query)
        if not session:
            return

        session.save_state_snapshot()

        if not await self._transition_or_error(
            session, MLPredictionState.LOADING_PRED_TEMPLATE.value, query
        ):
            return

        # Get user's templates
        user_id = update.effective_user.id
        templates = self.template_manager.list_templates(user_id)

        if not templates:
            await query.edit_message_text(
                pt_messages.PRED_TEMPLATE_NO_TEMPLATES,
                parse_mode="Markdown"
            )
            return

        # Display templates as buttons
        keyboard = [
            [InlineKeyboardButton(f"📄 {t.template_name}", callback_data=f"load_pred_template:{t.template_name}")]
            for t in templates
        ] + [[InlineKeyboardButton("🔙 Back", callback_data="back")]]

        await query.edit_message_text(
            pt_messages.PRED_TEMPLATE_LOAD_PROMPT.format(count=len(templates)),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_template_selection(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle specific template selection."""
        query = update.callback_query
        await query.answer()

        session = await self._get_session_or_error(update, query)
        if not session:
            return

        user_id = update.effective_user.id
        template_name = query.data.split(":", 1)[1]

        # Load template
        template = self.template_manager.load_template(user_id, template_name)
        if not template:
            await query.edit_message_text(
                pt_messages.PRED_TEMPLATE_NOT_FOUND.format(name=template_name),
                parse_mode="Markdown"
            )
            return

        # Validate model exists
        model_info = self.ml_engine.get_model_info(user_id, template.model_id)
        if not model_info:
            await query.edit_message_text(
                pt_messages.PRED_TEMPLATE_MODEL_INVALID.format(model_id=template.model_id),
                parse_mode="Markdown"
            )
            return

        # Update last_used timestamp
        template.last_used = datetime.now(timezone.utc).isoformat()
        config = {
            "file_path": template.file_path,
            "model_id": template.model_id,
            "feature_columns": template.feature_columns,
            "output_column_name": template.output_column_name,
            "save_path": template.save_path,
            "description": template.description,
            "last_used": template.last_used
        }
        self.template_manager.save_template(user_id, template_name, config)

        # Populate session with template data
        session.file_path = template.file_path
        session.selections["selected_model_id"] = template.model_id
        session.selections["selected_features"] = template.feature_columns
        session.selections["prediction_column_name"] = template.output_column_name
        if template.save_path:
            session.selections["output_file_path"] = template.save_path

        session.save_state_snapshot()

        if not await self._transition_or_error(
            session, MLPredictionState.CONFIRMING_PRED_TEMPLATE.value, query
        ):
            return

        # Display configuration summary
        summary = pt_messages.format_pred_template_summary(
            template_name=template.template_name,
            file_path=template.file_path,
            model_id=template.model_id,
            features=template.feature_columns,
            output_column=template.output_column_name,
            description=template.description,
            created_at=template.created_at
        )

        # Ask about loading data
        keyboard = [
            [InlineKeyboardButton("✅ Use This Template", callback_data="confirm_pred_template")],
            [InlineKeyboardButton("🔙 Back to Templates", callback_data="back_to_pred_templates")]
        ]

        await query.edit_message_text(
            summary,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_template_confirmation(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template confirmation and data loading."""
        query = update.callback_query
        await query.answer()

        session = await self._get_session_or_error(update, query)
        if not session:
            return

        # Validate and load data
        df = await self._validate_and_load_template_data(session, query)
        if df is None:
            return

        await query.edit_message_text(
            pt_messages.PRED_TEMPLATE_DATA_LOADED.format(
                rows=len(df),
                columns=len(df.columns)
            ),
            parse_mode="Markdown"
        )

        # PHASE 1: Verify data persistence BEFORE transitioning to READY_TO_RUN
        if session.uploaded_data is None or len(session.uploaded_data) == 0:
            await query.message.reply_text(
                "❌ **Data Persistence Error**\n\n"
                "Template data failed to persist in session. This indicates a system issue.\n\n"
                "**Troubleshooting:**\n"
                "• Try reloading the template\n"
                "• Use /predict to start a fresh session\n"
                "• Check if data file is accessible",
                parse_mode="Markdown"
            )
            logger.error(
                f"Template data not persisted for user {session.user_id}. "
                f"uploaded_data is None: {session.uploaded_data is None}"
            )
            return

        session.save_state_snapshot()
        if not await self._transition_or_error(
            session, MLPredictionState.READY_TO_RUN.value, query.message
        ):
            return

        keyboard = [[InlineKeyboardButton("🚀 Run Prediction", callback_data="pred_run")]]
        await query.message.reply_text(
            "Ready to run predictions!",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # =========================================================================
    # Cancel Template Workflow
    # =========================================================================

    async def handle_cancel_template(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template workflow cancellation."""
        query = update.callback_query
        await query.answer()

        session = await self._get_session_or_error(update, query)
        if not session:
            return

        if session.restore_previous_state():
            await query.edit_message_text("❌ Prediction template operation cancelled.")

            # PHASE 5: Show output options after template cancel
            await self._show_output_options(update, context, session)
        else:
            await query.edit_message_text("❌ Cannot cancel: No previous state available.")

    async def handle_back_to_templates(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle back button to return to template list."""
        query = update.callback_query
        await query.answer()

        session = await self._get_session_or_error(update, query)
        if not session:
            return

        session.restore_previous_state()
        if not await self._transition_or_error(
            session, MLPredictionState.LOADING_PRED_TEMPLATE.value, query
        ):
            return

        await self.handle_template_source_selection(update, context)
