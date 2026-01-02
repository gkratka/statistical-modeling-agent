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
from src.utils.i18n_manager import I18nManager
from src.utils.exceptions import ModelNotFoundError

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
            # Try to get locale from context, default to None
            locale = None
            error_msg = I18nManager.t(
                'workflow_state.session_not_found',
                locale=locale,
                command='/predict'
            )
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
            locale = session.language if session.language else None
            error_text = I18nManager.t(
                'prediction.errors.transition_failed',
                locale=locale,
                error=error_msg,
                missing='N/A'
            )
            if isinstance(query_or_message, CallbackQuery):
                await query_or_message.edit_message_text(error_text)
            else:
                await query_or_message.reply_text(error_text)
            return False
        return True

    async def _model_exists_on_worker(
        self,
        user_id: int,
        model_id: str,
        context: CallbackContext
    ) -> bool:
        """Check if model exists on worker.

        Args:
            user_id: Telegram user ID
            model_id: Model ID to check
            context: Bot context

        Returns:
            True if model exists on worker, False otherwise
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
                job_type=JobType.LIST_MODELS,
                params={},
                timeout=30.0
            )

            max_wait, poll_interval, elapsed = 30, 0.5, 0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    return False
                if job.status == JobStatus.COMPLETED:
                    models = job.result.get('models', [])
                    return any(m.get('model_id') == model_id for m in models)
                elif job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    return False

            return False
        except Exception as e:
            logger.error(f"Error checking model on worker: {e}")
            return False

    async def _validate_and_load_template_data(self, session, query, context):
        """
        Validate file path and load data from template.

        Args:
            session: Session object
            query: Callback query object
            context: Telegram context for worker access

        Returns:
            DataFrame if successful, None otherwise
        """
        file_path = session.file_path
        user_id = session.user_id

        # Extract locale from session
        locale = session.language if session.language else None

        # Check if worker is connected FIRST (for prod where file is on user's machine)
        websocket_server = context.bot_data.get('websocket_server')
        worker_manager = websocket_server.worker_manager if websocket_server else None
        worker_connected = worker_manager and worker_manager.is_user_connected(user_id)

        if not worker_connected:
            # No worker - validate path locally (dev scenario)
            validation_result = self.path_validator.validate_path(file_path)
            if not validation_result["is_valid"]:
                await query.edit_message_text(
                    pt_messages.pred_template_file_path_invalid(
                        path=file_path,
                        error=validation_result['error'],
                        locale=locale
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
                pt_messages.pred_template_load_failed(error=str(e), locale=locale),
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

        # Extract locale from session
        locale = session.language if session.language else None

        keyboard = create_output_option_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            PredictionMessages.output_options_prompt(locale=locale),
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

        # Extract locale from session
        locale = session.language if session.language else None

        keyboard = [[InlineKeyboardButton(
            I18nManager.t('templates.save.cancel_button', locale=locale, default="❌ Cancel"),
            callback_data="cancel_pred_template"
        )]]
        await query.edit_message_text(
            pt_messages.pred_template_save_prompt(locale=locale),
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

        # Extract locale from session
        locale = session.language if session.language else None

        template_name = update.message.text.strip()

        # Validate name
        is_valid, error_msg = self.template_manager.validate_template_name(template_name)
        if not is_valid:
            await update.message.reply_text(
                pt_messages.pred_template_invalid_name(error=error_msg, locale=locale),
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
                I18nManager.t(
                    'templates.save.missing_config',
                    locale=locale
                )
            )
            return

        # Check if exists - offer to overwrite
        if self.template_manager.template_exists(user_id, template_name):
            # Store pending template data in session for overwrite confirmation
            session.selections["pending_template_name"] = template_name
            session.selections["pending_template_config"] = template_config

            keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('templates.overwrite.yes', locale=locale, default="✅ Yes, Overwrite"),
                    callback_data="confirm_overwrite_pred_template"
                )],
                [InlineKeyboardButton(
                    I18nManager.t('templates.overwrite.no', locale=locale, default="❌ No, Cancel"),
                    callback_data="cancel_pred_template"
                )]
            ]
            await update.message.reply_text(
                I18nManager.t(
                    'templates.overwrite.prompt',
                    locale=locale,
                    name=template_name,
                    default=f"⚠️ Template '*{template_name}*' already exists.\n\nDo you want to overwrite it?"
                ),
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
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
                pt_messages.pred_template_saved_success(name=template_name, locale=locale),
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
                pt_messages.pred_template_save_failed(error=message, locale=locale),
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

        # Extract locale from session
        locale = session.language if session.language else None

        # Get user's templates
        user_id = update.effective_user.id
        templates = self.template_manager.list_templates(user_id)

        if not templates:
            await query.edit_message_text(
                pt_messages.pred_template_no_templates(locale=locale),
                parse_mode="Markdown"
            )
            return

        # Display templates as buttons
        keyboard = [
            [InlineKeyboardButton(f"📄 {t.template_name}", callback_data=f"load_pred_template:{t.template_name}")]
            for t in templates
        ] + [[InlineKeyboardButton(
            I18nManager.t('templates.load.back_button', locale=locale, default="🔙 Back"),
            callback_data="pred_back"
        )]]

        await query.edit_message_text(
            pt_messages.pred_template_load_prompt(count=len(templates), locale=locale),
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

        # Extract locale from session
        locale = session.language if session.language else None

        user_id = update.effective_user.id
        template_name = query.data.split(":", 1)[1]

        # Load template
        template = self.template_manager.load_template(user_id, template_name)
        if not template:
            await query.edit_message_text(
                pt_messages.pred_template_not_found(name=template_name, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Validate model exists on worker (not local storage)
        model_exists = await self._model_exists_on_worker(user_id, template.model_id, context)

        if not model_exists:
            # Show delete button so user can remove stale template
            delete_keyboard = [[InlineKeyboardButton(
                I18nManager.t('templates.delete.button', locale=locale, default="🗑️ Delete This Template"),
                callback_data=f"delete_pred_template:{template_name}"
            )]]
            await query.edit_message_text(
                pt_messages.pred_template_model_invalid(model_id=template.model_id, locale=locale),
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(delete_keyboard)
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
            created_at=template.created_at,
            locale=locale
        )

        # Ask about loading data
        keyboard = [
            [InlineKeyboardButton(
                I18nManager.t('templates.load.use_template_button', locale=locale, default="✅ Use This Template"),
                callback_data="confirm_pred_template"
            )],
            [InlineKeyboardButton(
                I18nManager.t('templates.load.back_to_list_button', locale=locale, default="🔙 Back to Templates"),
                callback_data="back_to_pred_templates"
            )]
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

        # Extract locale from session
        locale = session.language if session.language else None

        # Validate and load data
        df = await self._validate_and_load_template_data(session, query, context)
        if df is None:
            return

        await query.edit_message_text(
            pt_messages.pred_template_data_loaded(
                rows=len(df),
                columns=len(df.columns),
                locale=locale
            ),
            parse_mode="Markdown"
        )

        # PHASE 1: Verify data persistence BEFORE transitioning to READY_TO_RUN
        if session.uploaded_data is None or len(session.uploaded_data) == 0:
            await query.message.reply_text(
                I18nManager.t(
                    'templates.errors.data_persistence_error',
                    locale=locale,
                    default=(
                        "❌ **Data Persistence Error**\n\n"
                        "Template data failed to persist in session. This indicates a system issue.\n\n"
                        "**Troubleshooting:**\n"
                        "• Try reloading the template\n"
                        "• Use /predict to start a fresh session\n"
                        "• Check if data file is accessible"
                    )
                ),
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

        keyboard = [[InlineKeyboardButton(
            I18nManager.t('templates.messages.run_prediction_button', locale=locale, default="🚀 Run Prediction"),
            callback_data="pred_run"
        )]]
        await query.message.reply_text(
            I18nManager.t('templates.messages.ready_to_run', locale=locale, default="Ready to run predictions!"),
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

        # Extract locale from session
        locale = session.language if session.language else None

        if session.restore_previous_state():
            await query.edit_message_text(
                I18nManager.t(
                    'templates.cancel.cancelled',
                    locale=locale
                )
            )

            # PHASE 5: Show output options after template cancel
            await self._show_output_options(update, context, session)
        else:
            await query.edit_message_text(
                I18nManager.t(
                    'templates.cancel.no_previous_state',
                    locale=locale
                )
            )

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

    async def handle_template_delete(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template deletion request."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id

        # Extract locale from session if available
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        locale = session.language if session and session.language else None

        # Extract template name from callback data
        template_name = query.data.split(":", 1)[1]

        # Delete template
        success = self.template_manager.delete_template(user_id, template_name)

        if success:
            await query.edit_message_text(
                I18nManager.t(
                    'templates.delete.success',
                    locale=locale,
                    name=template_name,
                    default=f"✅ Template '{template_name}' has been deleted."
                ),
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(
                I18nManager.t(
                    'templates.delete.failed',
                    locale=locale,
                    name=template_name,
                    default=f"❌ Failed to delete template '{template_name}'."
                ),
                parse_mode="Markdown"
            )

    async def handle_overwrite_confirmation(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template overwrite confirmation."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session expired. Please start again with /predict")
            return

        locale = session.language if session.language else None

        # Get pending template data from session
        template_name = session.selections.get("pending_template_name")
        template_config = session.selections.get("pending_template_config")

        if not template_name or not template_config:
            await query.edit_message_text(
                I18nManager.t('templates.overwrite.no_pending', locale=locale, default="❌ No pending template to save."),
                parse_mode="Markdown"
            )
            return

        # Delete old template first
        self.template_manager.delete_template(user_id, template_name)

        # Save new template
        success, message = self.template_manager.save_template(
            user_id=user_id,
            template_name=template_name,
            config=template_config
        )

        # Clean up pending data
        session.selections.pop("pending_template_name", None)
        session.selections.pop("pending_template_config", None)

        if success:
            await query.edit_message_text(
                pt_messages.pred_template_saved_success(name=template_name, locale=locale),
                parse_mode="Markdown"
            )
            # Transition back to COMPLETE
            await self.state_manager.transition_state(session, MLPredictionState.COMPLETE.value)
        else:
            await query.edit_message_text(
                I18nManager.t('templates.save.failed', locale=locale, default="❌ Failed to save template."),
                parse_mode="Markdown"
            )
