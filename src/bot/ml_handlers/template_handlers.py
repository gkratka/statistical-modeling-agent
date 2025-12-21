"""
Template workflow handlers for ML training.

This module provides handlers for saving and loading ML training templates.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackContext

from src.core.state_manager import MLTrainingState, StateManager
from src.core.template_manager import TemplateManager
from src.core.training_template import TemplateConfig
from src.bot.messages import template_messages
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
from src.utils.i18n_manager import I18nManager

logger = logging.getLogger(__name__)


class TemplateHandlers:
    """Handlers for ML training template operations."""

    def __init__(
        self,
        state_manager: StateManager,
        template_manager: TemplateManager,
        data_loader: DataLoader,
        path_validator: PathValidator
    ):
        """
        Initialize template handlers.

        Args:
            state_manager: StateManager instance
            template_manager: TemplateManager instance
            data_loader: DataLoader instance
            path_validator: PathValidator instance
        """
        self.state_manager = state_manager
        self.template_manager = template_manager
        self.data_loader = data_loader
        self.path_validator = path_validator

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

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        # Save state snapshot before transition
        session.save_state_snapshot()

        # Transition to SAVING_TEMPLATE state
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.SAVING_TEMPLATE.value
        )

        if not success:
            await query.edit_message_text(f"❌ {error_msg}")
            return

        # Prompt for template name with i18n
        locale = session.language if session.language else None
        keyboard = [[InlineKeyboardButton(I18nManager.t('workflow_state.buttons.cancel', locale=locale), callback_data="cancel_template")]]
        await query.edit_message_text(
            template_messages.TEMPLATE_SAVE_PROMPT,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        logger.info(f"User {user_id} initiated template save")

    async def handle_template_name_input(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template name text input."""
        user_id = update.effective_user.id
        chat_id = update.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session or session.current_state != MLTrainingState.SAVING_TEMPLATE.value:
            return

        template_name = update.message.text.strip()

        # Validate name
        is_valid, error_msg = self.template_manager.validate_template_name(template_name)
        if not is_valid:
            await update.message.reply_text(
                template_messages.TEMPLATE_INVALID_NAME.format(error=error_msg),
                parse_mode="Markdown"
            )
            return

        # Check if exists
        if self.template_manager.template_exists(user_id, template_name):
            await update.message.reply_text(
                template_messages.TEMPLATE_EXISTS.format(name=template_name),
                parse_mode="Markdown"
            )
            return

        # Build template config from session
        template_config = {
            "file_path": session.file_path or "",
            "target_column": session.selections.get("target_column", ""),
            "feature_columns": session.selections.get("feature_columns", []),
            "model_category": session.selections.get("model_category", ""),
            "model_type": session.selections.get("model_type", ""),
            "hyperparameters": session.selections.get("hyperparameters", {})
        }

        # Validate required fields
        if not all([template_config["file_path"], template_config["target_column"],
                   template_config["feature_columns"], template_config["model_type"]]):
            await update.message.reply_text(
                "❌ Cannot save template: Missing required configuration. "
                "Please complete the training setup first."
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
                template_messages.TEMPLATE_SAVED_SUCCESS.format(name=template_name),
                parse_mode="Markdown"
            )

            # Offer to continue training or finish with i18n
            locale = session.language if session.language else None
            keyboard = [
                [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.start_training', locale=locale), callback_data="start_training")],
                [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.done_exit', locale=locale), callback_data="complete")]
            ]
            await update.message.reply_text(
                template_messages.TEMPLATE_CONTINUE_TRAINING,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

            logger.info(f"Template '{template_name}' saved for user {user_id}")
        else:
            await update.message.reply_text(
                template_messages.TEMPLATE_SAVE_FAILED.format(error=message),
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

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        # Save state snapshot before transition
        session.save_state_snapshot()

        # Transition to LOADING_TEMPLATE state
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.LOADING_TEMPLATE.value
        )

        if not success:
            await query.edit_message_text(f"❌ {error_msg}")
            return

        # Get user's templates
        templates = self.template_manager.list_templates(user_id)

        if not templates:
            await query.edit_message_text(
                template_messages.TEMPLATE_NO_TEMPLATES,
                parse_mode="Markdown"
            )
            return

        # Display templates as buttons with i18n
        locale = session.language if session.language else None
        keyboard = []
        for template in templates:
            button_text = f"📄 {template.template_name}"
            callback_data = f"load_template:{template.template_name}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

        keyboard.append([InlineKeyboardButton(I18nManager.t('workflow_state.buttons.back', locale=locale), callback_data="workflow_back")])

        await query.edit_message_text(
            template_messages.TEMPLATE_LOAD_PROMPT.format(count=len(templates)),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        logger.info(f"User {user_id} browsing {len(templates)} templates")

    async def handle_template_selection(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle specific template selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        # Extract template name from callback_data
        template_name = query.data.split(":", 1)[1]

        # Load template
        template = self.template_manager.load_template(user_id, template_name)
        if not template:
            await query.edit_message_text(
                template_messages.TEMPLATE_NOT_FOUND.format(name=template_name),
                parse_mode="Markdown"
            )
            return

        # Update last_used timestamp
        template.last_used = datetime.now(timezone.utc).isoformat()
        config = {
            "file_path": template.file_path,
            "target_column": template.target_column,
            "feature_columns": template.feature_columns,
            "model_category": template.model_category,
            "model_type": template.model_type,
            "hyperparameters": template.hyperparameters,
            "last_used": template.last_used,
            "description": template.description
        }
        self.template_manager.save_template(user_id, template_name, config)

        # Populate session with template data
        session.file_path = template.file_path
        session.selections["target_column"] = template.target_column
        session.selections["feature_columns"] = template.feature_columns
        session.selections["model_category"] = template.model_category
        session.selections["model_type"] = template.model_type
        session.selections["hyperparameters"] = template.hyperparameters

        # Save state snapshot before transition
        session.save_state_snapshot()

        # Transition to CONFIRMING_TEMPLATE
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.CONFIRMING_TEMPLATE.value
        )

        if not success:
            await query.edit_message_text(f"❌ {error_msg}")
            return

        # Display configuration summary
        summary = template_messages.format_template_summary(
            template_name=template.template_name,
            file_path=template.file_path,
            target=template.target_column,
            features=template.feature_columns,
            model_category=template.model_category,
            model_type=template.model_type,
            created_at=template.created_at
        )

        # Ask about loading data with i18n
        locale = session.language if session.language else None
        keyboard = [
            [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.load_now', locale=locale), callback_data="template_load_now")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.defer_loading', locale=locale), callback_data="template_defer")],
            [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.back_to_templates', locale=locale), callback_data="workflow_back")]
        ]

        await query.edit_message_text(
            summary,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        logger.info(f"User {user_id} selected template '{template_name}'")

    async def handle_template_load_option(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template data loading option (now vs defer)."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        if query.data == "template_load_now":
            session.load_deferred = False

            # Validate and load file
            file_path = session.file_path
            validation_result = self.path_validator.validate_path(file_path)

            if not validation_result["is_valid"]:
                await query.edit_message_text(
                    template_messages.TEMPLATE_FILE_PATH_INVALID.format(
                        path=file_path,
                        error=validation_result['error']
                    ),
                    parse_mode="Markdown"
                )
                return

            try:
                # Load data (returns tuple: df, metadata, schema)
                df, _, _ = await self.data_loader.load_from_local_path(file_path)

                # Store in bot_data
                data_key = f"user_{user_id}_conv_chat_{chat_id}_data"
                context.bot_data[data_key] = df

                await query.edit_message_text(
                    template_messages.TEMPLATE_DATA_LOADED.format(
                        rows=len(df),
                        columns=len(df.columns)
                    ),
                    parse_mode="Markdown"
                )

                # Offer training action (Bug #10 fix) with i18n
                locale = session.language if session.language else None
                keyboard = [
                    [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.start_training', locale=locale), callback_data="start_training")]
                ]
                await query.message.reply_text(
                    I18nManager.t('workflows.ml_training.what_next', locale=locale),
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )

                # Save state snapshot before transition
                session.save_state_snapshot()

                # Transition to TRAINING
                success, error_msg, _ = await self.state_manager.transition_state(
                    session,
                    MLTrainingState.TRAINING.value
                )

                if not success:
                    await query.message.reply_text(f"❌ {error_msg}")
                    return

                # Start training (will be handled by main handler)
                logger.info(f"User {user_id} loading template data immediately")

            except Exception as e:
                logger.error(f"Error loading template data: {e}", exc_info=True)
                await query.edit_message_text(
                    template_messages.TEMPLATE_LOAD_FAILED.format(error=str(e)),
                    parse_mode="Markdown"
                )

        elif query.data == "template_defer":
            session.load_deferred = True

            await query.edit_message_text(
                template_messages.TEMPLATE_DATA_DEFERRED,
                parse_mode="Markdown"
            )

            # Save state snapshot before transition
            session.save_state_snapshot()

            # Transition to COMPLETE
            success, error_msg, _ = await self.state_manager.transition_state(
                session,
                MLTrainingState.COMPLETE.value
            )

            if not success:
                await query.message.reply_text(f"❌ {error_msg}")

            logger.info(f"User {user_id} deferred template data loading")

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

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found.")
            return

        # Restore previous state using back navigation
        if session.restore_previous_state():
            await query.edit_message_text("❌ Template operation cancelled.")
            logger.info(f"User {user_id} cancelled template operation")
        else:
            await query.edit_message_text("❌ Cannot cancel: No previous state available.")
