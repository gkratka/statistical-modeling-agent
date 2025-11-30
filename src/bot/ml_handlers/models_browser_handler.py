"""Telegram bot handlers for /models command - interactive model browser."""

import logging
from typing import Optional, List
from math import ceil

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, ModelsBrowserState, WorkflowType
from src.engines.model_catalog import (
    MODEL_CATALOG, ModelInfo, get_all_models, get_model_by_id
)
from src.bot.messages.models_messages import ModelsMessages
from src.utils.i18n_manager import I18nManager

logger = logging.getLogger(__name__)

# Pagination settings
MODELS_PER_PAGE = 4  # Telegram best practice: 4-5 buttons per screen


class ModelsBrowserHandler:
    """Handler for /models command - browse ML model catalog."""

    def __init__(self, state_manager: StateManager):
        """Initialize handler with state manager."""
        self.state_manager = state_manager
        self.logger = logger

    async def handle_models_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /models command - show paginated model list.

        Entry point for models browser workflow.
        """
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_models_command: {e}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    I18nManager.t('models_browser.malformed_update', locale=None, command='/models'),
                    parse_mode="Markdown"
                )
            return

        # Get or create session
        session = await self.state_manager.get_or_create_session(
            user_id=user_id,
            conversation_id=str(chat_id)
        )

        # Initialize workflow at VIEWING_MODEL_LIST state
        session.workflow_type = WorkflowType.MODELS_BROWSER
        session.current_state = ModelsBrowserState.VIEWING_MODEL_LIST.value
        await self.state_manager.update_session(session)

        # Show first page of models
        await self._show_model_list(update, context, page=0, locale=session.language)

    async def _show_model_list(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        page: int = 0,
        locale: Optional[str] = None
    ) -> None:
        """
        Show paginated list of models.

        Args:
            update: Telegram update
            context: Bot context
            page: Page number (0-indexed)
            locale: Language code for translations
        """
        # Get all models
        all_models = get_all_models()
        total_models = len(all_models)
        total_pages = ceil(total_models / MODELS_PER_PAGE)

        # Validate page number
        page = max(0, min(page, total_pages - 1))

        # Get models for current page
        start_idx = page * MODELS_PER_PAGE
        end_idx = start_idx + MODELS_PER_PAGE
        page_models = all_models[start_idx:end_idx]

        # Build inline keyboard
        keyboard = []

        # Model buttons (one per row)
        for model in page_models:
            # Display with icon and name
            # Callback format: "model:id:page" (for back navigation)
            button_text = f"{model.icon} {model.display_name}"
            callback_data = f"model:{model.id}:{page}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

        # Pagination + Cancel row
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton(
                I18nManager.t('models_browser.navigation.prev_button', locale=locale),
                callback_data=f"page:{page-1}"
            ))
        nav_row.append(InlineKeyboardButton(
            I18nManager.t('models_browser.navigation.cancel_button', locale=locale),
            callback_data="cancel_models"
        ))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton(
                I18nManager.t('models_browser.navigation.next_button', locale=locale),
                callback_data=f"page:{page+1}"
            ))
        keyboard.append(nav_row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        # Build message
        message_text = ModelsMessages.models_list_message(
            page=page + 1,
            total_pages=total_pages,
            total_models=total_models,
            locale=locale
        )

        # Send or edit message
        if update.message:
            # New command - send message
            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            # Callback query - edit existing message
            query = update.callback_query
            await query.answer()
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    async def handle_model_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle model selection - show model details.

        Callback format: "model:id:page"
        """
        query = update.callback_query
        await query.answer()

        # Parse callback data
        try:
            _, model_id, page_str = query.data.split(":")
            page = int(page_str)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid callback data in handle_model_selection: {e}")
            await query.edit_message_text(
                I18nManager.t('models_browser.invalid_selection', locale=None),
                parse_mode="Markdown"
            )
            return

        # Get user session
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object: {e}")
            return

        session = await self.state_manager.get_session(user_id, str(chat_id))

        # Save state snapshot for back navigation
        session.save_state_snapshot()

        # Transition to VIEWING_MODEL_DETAILS
        session.current_state = ModelsBrowserState.VIEWING_MODEL_DETAILS.value
        session.selections['current_page'] = page  # Store page for back navigation
        await self.state_manager.update_session(session)

        # Get model info
        model = get_model_by_id(model_id)
        if not model:
            locale = session.language if session.language else None
            await query.edit_message_text(
                I18nManager.t('models_browser.model_not_found', locale=locale, model_id=model_id),
                parse_mode="Markdown"
            )
            return

        # Build message with locale
        locale = session.language if session.language else None
        message_text = ModelsMessages.model_details_message(model, locale=locale)

        # Build keyboard with Back button
        keyboard = [[InlineKeyboardButton(
            I18nManager.t('models_browser.navigation.back_button', locale=locale),
            callback_data=f"back_to_list:{page}"
        )]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Edit message with details
        await query.edit_message_text(
            message_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_pagination(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle pagination callbacks.

        Callback format: "page:number"
        """
        query = update.callback_query
        await query.answer()

        # Parse page number
        try:
            _, page_str = query.data.split(":")
            page = int(page_str)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid pagination callback: {e}")
            return

        # Get user session for locale
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            session = await self.state_manager.get_session(user_id, str(chat_id))
            locale = session.language
        except (AttributeError, Exception) as e:
            logger.error(f"Error getting session in pagination: {e}")
            locale = None

        # Show requested page
        await self._show_model_list(update, context, page=page, locale=locale)

    async def handle_back_to_list(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle back button - return to model list.

        Callback format: "back_to_list:page"
        """
        query = update.callback_query
        await query.answer()

        # Parse page number
        try:
            _, page_str = query.data.split(":")
            page = int(page_str)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid back callback: {e}")
            page = 0

        # Get user session
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object: {e}")
            return

        session = await self.state_manager.get_session(user_id, str(chat_id))

        # Restore previous state
        session.restore_previous_state()
        await self.state_manager.update_session(session)

        # Show model list
        await self._show_model_list(update, context, page=page, locale=session.language)

    async def handle_cancel(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle cancel button - exit models browser.

        Callback format: "cancel_models"
        """
        query = update.callback_query
        await query.answer()

        # Get user session
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object: {e}")
            return

        session = await self.state_manager.get_session(user_id, str(chat_id))

        # Cancel workflow
        await self.state_manager.cancel_workflow(session)

        # Edit message
        locale = session.language if session.language else None
        await query.edit_message_text(
            I18nManager.t('models_browser.closed', locale=locale),
            parse_mode="Markdown"
        )
