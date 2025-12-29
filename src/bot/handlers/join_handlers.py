"""Telegram bot handlers for join workflow (/join command)."""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, JoinWorkflowState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
from src.bot.messages.join_messages import (
    JoinMessages,
    create_operation_buttons,
    create_dataframe_count_buttons,
    create_dataframe_source_buttons,
    create_key_column_buttons,
    create_output_path_buttons,
    create_filter_buttons,
)

logger = logging.getLogger(__name__)


class JoinHandler:
    """Handler for join workflow (/join command)."""

    def __init__(
        self,
        state_manager: StateManager,
        data_loader: DataLoader,
        path_validator: PathValidator = None,
        websocket_server=None
    ):
        """Initialize handler with state manager and data loader."""
        self.state_manager = state_manager
        self.data_loader = data_loader
        self.websocket_server = websocket_server
        self.logger = logger

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

    async def handle_start_join(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /join command - start join workflow."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_start_join: {e}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "An error occurred. Please try again.",
                    parse_mode="Markdown"
                )
            return

        # Get or create session
        session = await self.state_manager.get_or_create_session(
            user_id=user_id,
            conversation_id=f"chat_{chat_id}"
        )

        # Initialize join workflow
        session.workflow_type = WorkflowType.JOIN_WORKFLOW
        session.current_state = JoinWorkflowState.CHOOSING_OPERATION.value
        session.selections = {}  # Clear any previous selections
        await self.state_manager.update_session(session)

        # Show intro message with operation buttons
        keyboard = create_operation_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            JoinMessages.join_intro_message(),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    # =========================================================================
    # Step 2: Operation Selection
    # =========================================================================

    async def handle_operation_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle operation type selection callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            # Extract operation from callback_data like "join_op_left_join"
            operation = query.data.replace("join_op_", "")
        except AttributeError as e:
            logger.error(f"Malformed update in handle_operation_selection: {e}")
            await query.edit_message_text("An error occurred. Please try again.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await query.edit_message_text("Session expired. Please start again with /join")
            return

        # Store selected operation
        session.selections["join_operation"] = operation
        session.current_state = JoinWorkflowState.CHOOSING_DATAFRAME_COUNT.value
        await self.state_manager.update_session(session)

        # Show dataframe count selection
        keyboard = create_dataframe_count_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            JoinMessages.dataframe_count_prompt(operation),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    # =========================================================================
    # Step 3: Dataframe Count Selection
    # =========================================================================

    async def handle_dataframe_count(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle dataframe count selection callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            # Extract count from callback_data like "join_count_2"
            count = int(query.data.replace("join_count_", ""))
        except (AttributeError, ValueError) as e:
            logger.error(f"Malformed update in handle_dataframe_count: {e}")
            await query.edit_message_text("An error occurred. Please try again.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await query.edit_message_text("Session expired. Please start again with /join")
            return

        # Store selected count and initialize dataframes list
        session.selections["dataframe_count"] = count
        session.selections["dataframes"] = []
        session.selections["current_df_index"] = 0
        session.current_state = JoinWorkflowState.AWAITING_DF1_SOURCE.value
        await self.state_manager.update_session(session)

        # Show source selection for first dataframe
        keyboard = create_dataframe_source_buttons(1)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            JoinMessages.dataframe_source_prompt(1, count),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    # =========================================================================
    # Steps 4-N: Dataframe Collection
    # =========================================================================

    async def handle_dataframe_source(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle dataframe source selection (upload vs local path)."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            # Extract source and df_num from callback_data like "join_df_source_upload_1"
            parts = query.data.split("_")
            source = parts[3]  # "upload" or "local"
            df_num = int(parts[4])
        except (AttributeError, ValueError, IndexError) as e:
            logger.error(f"Malformed update in handle_dataframe_source: {e}")
            await query.edit_message_text("An error occurred. Please try again.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await query.edit_message_text("Session expired. Please start again with /join")
            return

        if source == "local":
            # Transition to path input state
            state_name = f"AWAITING_DF{df_num}_PATH"
            session.current_state = getattr(JoinWorkflowState, state_name).value
            session.selections["current_source"] = "local_path"
            await self.state_manager.update_session(session)

            await query.edit_message_text(
                JoinMessages.file_path_input_prompt(
                    df_num,
                    self.data_loader.allowed_directories
                ),
                parse_mode="Markdown"
            )
        else:  # upload
            # Transition to upload waiting state
            state_name = f"AWAITING_DF{df_num}_UPLOAD"
            session.current_state = getattr(JoinWorkflowState, state_name).value
            session.selections["current_source"] = "telegram"
            await self.state_manager.update_session(session)

            await query.edit_message_text(
                JoinMessages.upload_file_prompt(df_num),
                parse_mode="Markdown"
            )

    async def handle_file_path_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle local file path text input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            file_path = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_file_path_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await update.message.reply_text("Session expired. Please start again with /join")
            return

        # Determine current dataframe number from state
        current_state = session.current_state
        df_num = self._get_df_num_from_state(current_state)
        if df_num is None:
            await update.message.reply_text("Unexpected state. Please start again with /join")
            return

        # Basic path validation (no directory whitelist - allow any path)
        resolved_path = Path(file_path).resolve()
        if not resolved_path.exists():
            await update.message.reply_text(
                JoinMessages.path_validation_error("not_found", file_path),
                parse_mode="Markdown"
            )
            return

        allowed_extensions = [".csv", ".xlsx", ".xls", ".parquet"]
        if resolved_path.suffix.lower() not in allowed_extensions:
            await update.message.reply_text(
                JoinMessages.path_validation_error("invalid_extension", file_path),
                parse_mode="Markdown"
            )
            return

        # Load dataframe schema (columns only, not full data)
        try:
            if resolved_path.suffix.lower() == ".csv":
                # Read only header row
                df_sample = pd.read_csv(resolved_path, nrows=0)
            elif resolved_path.suffix.lower() in [".xlsx", ".xls"]:
                df_sample = pd.read_excel(resolved_path, nrows=0)
            elif resolved_path.suffix.lower() == ".parquet":
                df_sample = pd.read_parquet(resolved_path).head(0)
            else:
                await update.message.reply_text(
                    JoinMessages.path_validation_error("invalid_extension", file_path),
                    parse_mode="Markdown"
                )
                return

            columns = list(df_sample.columns)
            # Get row count (approximate for large files)
            suffix = resolved_path.suffix.lower()
            if suffix == ".csv":
                with open(resolved_path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Minus header
            elif suffix in [".xlsx", ".xls"]:
                row_count = len(pd.read_excel(resolved_path))
            else:
                row_count = len(pd.read_parquet(resolved_path))

        except Exception as e:
            await update.message.reply_text(
                JoinMessages.error_message(f"Could not read file: {str(e)}"),
                parse_mode="Markdown"
            )
            return

        # Store dataframe info
        df_info = {
            "source": "local_path",
            "path": str(resolved_path),
            "columns": columns,
            "row_count": row_count
        }
        session.selections["dataframes"].append(df_info)

        # Send confirmation
        await update.message.reply_text(
            JoinMessages.file_loaded_message(df_num, file_path, row_count, columns),
            parse_mode="Markdown"
        )

        # Move to next step
        await self._advance_to_next_step(update, session, df_num)

    async def handle_file_upload(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle file upload via Telegram."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            document = update.message.document
        except AttributeError as e:
            logger.error(f"Malformed update in handle_file_upload: {e}")
            return

        if not document:
            await update.message.reply_text("Please send a file.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await update.message.reply_text("Session expired. Please start again with /join")
            return

        # Determine current dataframe number
        current_state = session.current_state
        df_num = self._get_df_num_from_state(current_state)
        if df_num is None:
            await update.message.reply_text("Unexpected state. Please start again with /join")
            return

        # Download file to temp location
        try:
            file = await context.bot.get_file(document.file_id)
            file_name = document.file_name or f"dataframe_{df_num}.csv"
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"join_df_{user_id}_{df_num}_{file_name}")

            await file.download_to_drive(temp_path)

            # Read file to get schema
            resolved_path = Path(temp_path)
            if resolved_path.suffix == ".csv":
                df_sample = pd.read_csv(resolved_path, nrows=5)
            elif resolved_path.suffix in [".xlsx", ".xls"]:
                df_sample = pd.read_excel(resolved_path, nrows=5)
            elif resolved_path.suffix == ".parquet":
                df_sample = pd.read_parquet(resolved_path).head(5)
            else:
                await update.message.reply_text(
                    JoinMessages.path_validation_error("invalid_extension", file_name),
                    parse_mode="Markdown"
                )
                return

            columns = list(df_sample.columns)
            row_count = len(pd.read_csv(temp_path) if resolved_path.suffix == ".csv"
                           else pd.read_excel(temp_path) if resolved_path.suffix in [".xlsx", ".xls"]
                           else pd.read_parquet(temp_path))

        except Exception as e:
            await update.message.reply_text(
                JoinMessages.error_message(f"Could not read file: {str(e)}"),
                parse_mode="Markdown"
            )
            return

        # Store dataframe info
        df_info = {
            "source": "telegram",
            "path": temp_path,
            "columns": columns,
            "row_count": row_count
        }
        session.selections["dataframes"].append(df_info)

        # Send confirmation
        await update.message.reply_text(
            JoinMessages.file_loaded_message(df_num, file_name, row_count, columns),
            parse_mode="Markdown"
        )

        # Move to next step
        await self._advance_to_next_step(update, session, df_num)

    async def _advance_to_next_step(
        self,
        update: Update,
        session,
        current_df_num: int
    ) -> None:
        """Advance to next dataframe or key column selection."""
        total_dfs = session.selections["dataframe_count"]
        operation = session.selections["join_operation"]

        if current_df_num < total_dfs:
            # More dataframes to collect
            next_df_num = current_df_num + 1
            state_name = f"AWAITING_DF{next_df_num}_SOURCE"
            session.current_state = getattr(JoinWorkflowState, state_name).value
            await self.state_manager.update_session(session)

            keyboard = create_dataframe_source_buttons(next_df_num)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                JoinMessages.dataframe_source_prompt(next_df_num, total_dfs),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            # All dataframes collected, move to key selection or filter step
            if operation in JoinMessages.JOIN_OPERATIONS:
                # Need key column selection first
                await self._show_key_column_selection(update, session)
            else:
                # Stack operations skip key columns, go to filter step
                await self._show_filter_step(update, session)

    async def _show_key_column_selection(self, update: Update, session) -> None:
        """Show key column selection for join operations."""
        dataframes = session.selections["dataframes"]

        # Find common columns across all dataframes
        common_columns = set(dataframes[0]["columns"])
        for df_info in dataframes[1:]:
            common_columns &= set(df_info["columns"])

        common_columns = sorted(list(common_columns))

        if not common_columns:
            # No common columns - cannot perform join
            await update.message.reply_text(
                JoinMessages.no_common_columns_error(
                    dataframes[0]["columns"],
                    dataframes[1]["columns"]
                ),
                parse_mode="Markdown"
            )
            return

        session.selections["common_columns"] = common_columns
        session.selections["selected_keys"] = []
        session.current_state = JoinWorkflowState.CHOOSING_KEY_COLUMNS.value
        await self.state_manager.update_session(session)

        keyboard = create_key_column_buttons(common_columns, [])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            JoinMessages.key_column_selection_prompt(common_columns),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _show_filter_step(self, update: Update, session, edit_message: bool = False) -> None:
        """Show optional filter input step.

        Args:
            update: Telegram update object
            session: Current user session
            edit_message: If True, edit existing message; if False, send new message
        """
        # Initialize filters list if not present
        if "filters" not in session.selections:
            session.selections["filters"] = []

        existing_filters = session.selections["filters"]
        session.current_state = JoinWorkflowState.CHOOSING_FILTER.value
        await self.state_manager.update_session(session)

        keyboard = create_filter_buttons(has_filters=len(existing_filters) > 0)
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = JoinMessages.filter_prompt(existing_filters)

        if edit_message and update.callback_query:
            await update.callback_query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    async def _show_output_path_selection(self, update: Update, session, edit_message: bool = False) -> None:
        """Show output path selection.

        Args:
            update: Telegram update object
            session: Current user session
            edit_message: If True, edit existing message; if False, send new message
        """
        # Generate default output path (same directory as first input)
        first_df_path = session.selections["dataframes"][0]["path"]
        first_dir = os.path.dirname(first_df_path)
        operation = session.selections["join_operation"]
        default_filename = f"joined_{operation}.csv"
        default_path = os.path.join(first_dir, default_filename)

        session.selections["default_output_path"] = default_path
        session.current_state = JoinWorkflowState.CHOOSING_OUTPUT_PATH.value
        await self.state_manager.update_session(session)

        keyboard = create_output_path_buttons()
        reply_markup = InlineKeyboardMarkup(keyboard)

        message_text = JoinMessages.output_path_prompt(default_path)

        if edit_message and update.callback_query:
            await update.callback_query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    # =========================================================================
    # Key Column Selection
    # =========================================================================

    async def handle_key_column_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle key column toggle or confirmation."""
        query = update.callback_query
        # Don't answer yet - we may need show_alert for validation errors

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            callback_data = query.data
        except AttributeError as e:
            logger.error(f"Malformed update in handle_key_column_selection: {e}")
            await query.edit_message_text("An error occurred. Please try again.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await query.answer()
            await query.edit_message_text("Session expired. Please start again with /join")
            return

        if callback_data == "join_key_confirm":
            # Confirm key selection
            selected_keys = session.selections.get("selected_keys", [])
            if not selected_keys:
                await query.answer("Please select at least one key column", show_alert=True)
                return

            await query.answer()  # Acknowledge before editing
            session.selections["key_columns"] = selected_keys

            # Move to filter step (instead of output path directly)
            await self._show_filter_step(update, session, edit_message=True)
        else:
            # Toggle a key column
            await query.answer()  # Acknowledge toggle
            column = callback_data.replace("join_key_", "")
            selected_keys = session.selections.get("selected_keys", [])

            if column in selected_keys:
                selected_keys.remove(column)
            else:
                selected_keys.append(column)

            session.selections["selected_keys"] = selected_keys
            await self.state_manager.update_session(session)

            # Update button display
            common_columns = session.selections["common_columns"]
            keyboard = create_key_column_buttons(common_columns, selected_keys)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_reply_markup(reply_markup=reply_markup)

    # =========================================================================
    # Filter Step Handlers (NEW)
    # =========================================================================

    async def handle_filter_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle filter step button callbacks (skip or done)."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            callback_data = query.data
        except AttributeError as e:
            logger.error(f"Malformed update in handle_filter_selection: {e}")
            await query.edit_message_text("An error occurred. Please try again.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await query.edit_message_text("Session expired. Please start again with /join")
            return

        if callback_data in ("join_filter_skip", "join_filter_done"):
            # User chose to skip filters or is done adding filters
            # Move to output path selection
            await self._show_output_path_selection(update, session, edit_message=True)

    async def handle_filter_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle filter text input from user."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            filter_text = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_filter_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await update.message.reply_text("Session expired. Please start again with /join")
            return

        # Validate filter format and get corrected filter with proper column case
        validation_error, corrected_filter = self._validate_filter_expression(filter_text, session)
        if validation_error:
            await update.message.reply_text(
                JoinMessages.filter_error_message(validation_error),
                parse_mode="Markdown"
            )
            return

        # Add corrected filter (with proper column case) to list
        if "filters" not in session.selections:
            session.selections["filters"] = []
        session.selections["filters"].append(corrected_filter)
        await self.state_manager.update_session(session)

        # Show confirmation with updated buttons
        keyboard = create_filter_buttons(has_filters=True)
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            JoinMessages.filter_added_message(corrected_filter, session.selections["filters"]),
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    def _validate_filter_expression(self, filter_text: str, session) -> tuple:
        """Validate filter expression format and column existence.

        Args:
            filter_text: Filter expression like "month = 1" or "status = 'active'"
            session: Current session with dataframe info

        Returns:
            Tuple of (error_message, corrected_filter)
            - If invalid: (error_string, None)
            - If valid: (None, corrected_filter_with_proper_case)
        """
        import re

        # Supported operators
        operators = [">=", "<=", "!=", "==", ">", "<", "="]

        # Check format: column operator value
        found_operator = None
        for op in operators:
            if op in filter_text:
                found_operator = op
                break

        if not found_operator:
            return ("Missing operator. Use =, !=, >, <, >=, or <=", None)

        parts = filter_text.split(found_operator, 1)
        if len(parts) != 2:
            return (f"Invalid format. Expected: column {found_operator} value", None)

        column_name = parts[0].strip()
        value = parts[1].strip()

        if not column_name:
            return ("Column name is empty", None)

        if not value:
            return ("Value is empty", None)

        # Check if column exists in any of the dataframes (case-insensitive)
        all_columns = set()
        for df_info in session.selections.get("dataframes", []):
            all_columns.update(df_info.get("columns", []))

        # Build case-insensitive lookup: lowercase -> actual case
        column_lookup = {col.lower(): col for col in all_columns}

        if column_name.lower() not in column_lookup:
            return (f"Column '{column_name}' not found in any dataframe", None)

        # Get the actual column name with correct case
        actual_column_name = column_lookup[column_name.lower()]

        # Build corrected filter with proper column case
        corrected_filter = f"{actual_column_name} {found_operator} {value}"

        return (None, corrected_filter)

    # =========================================================================
    # Output Path Selection
    # =========================================================================

    async def handle_output_path(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle output path selection callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            choice = query.data
        except AttributeError as e:
            logger.error(f"Malformed update in handle_output_path: {e}")
            await query.edit_message_text("An error occurred. Please try again.")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await query.edit_message_text("Session expired. Please start again with /join")
            return

        if choice == "join_output_default":
            # Use default path
            output_path = session.selections["default_output_path"]
            session.selections["output_path"] = output_path
            await self._execute_join(update, context, session)
        else:
            # Custom path requested
            session.current_state = JoinWorkflowState.AWAITING_CUSTOM_OUTPUT_PATH.value
            await self.state_manager.update_session(session)

            await query.edit_message_text(
                JoinMessages.custom_output_path_prompt(self.data_loader.allowed_directories),
                parse_mode="Markdown"
            )

    async def handle_custom_output_path(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle custom output path text input."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            output_path = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_custom_output_path: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")
        if not session or session.workflow_type != WorkflowType.JOIN_WORKFLOW:
            await update.message.reply_text("Session expired. Please start again with /join")
            return

        # Validate output directory exists (removed whitelist requirement)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            await update.message.reply_text(
                JoinMessages.path_validation_error("not_found", output_path, "Parent directory does not exist."),
                parse_mode="Markdown"
            )
            return

        session.selections["output_path"] = output_path
        await self._execute_join(update, context, session)

    # =========================================================================
    # Execution
    # =========================================================================

    async def _execute_join(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session
    ) -> None:
        """Execute the join operation via worker."""
        operation = session.selections["join_operation"]
        dataframes = session.selections["dataframes"]
        output_path = session.selections["output_path"]
        key_columns = session.selections.get("key_columns", [])

        session.current_state = JoinWorkflowState.EXECUTING_JOIN.value
        await self.state_manager.update_session(session)

        # Send executing message
        msg = update.callback_query.message if update.callback_query else update.message
        executing_msg = await msg.reply_text(
            JoinMessages.executing_message(operation, len(dataframes)),
            parse_mode="Markdown"
        )

        # Check if worker is connected (get from bot_data at runtime)
        user_id = session.user_id
        job_queue = context.bot_data.get('job_queue')
        websocket_server = context.bot_data.get('websocket_server')

        if not job_queue or not websocket_server or not websocket_server.is_connected(user_id):
            await executing_msg.edit_text(
                "*Error*\n\nNo worker connected. Please run /connect to connect a local worker first.",
                parse_mode="Markdown"
            )
            return

        # Prepare job parameters
        file_paths = [df["path"] for df in dataframes]
        filters = session.selections.get("filters", [])  # NEW: Include filters
        job_params = {
            "operation": operation,
            "file_paths": file_paths,
            "key_columns": key_columns,
            "output_path": output_path,
            "filters": filters,  # NEW: Pass filters to worker
        }

        try:
            # Import JobType and JobStatus for job queue
            from src.worker.job_queue import JobType, JobStatus

            # Create job via job queue (this registers it for result handling)
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.JOIN,
                params=job_params,
                timeout=300.0  # 5 minutes for large joins
            )

            logger.info(f"Join job {job_id} created for user {user_id}")

            # Poll for result
            max_wait, poll_interval, elapsed = 300, 1.0, 0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    await executing_msg.edit_text(
                        JoinMessages.error_message("Job was lost"),
                        parse_mode="Markdown"
                    )
                    return

                if job.status == JobStatus.COMPLETED:
                    # Success - show result
                    result = job.result or {}
                    await executing_msg.edit_text(
                        JoinMessages.complete_message(
                            output_path=result.get("output_file", output_path),
                            row_count=result.get("row_count", 0),
                            column_count=result.get("column_count", 0),
                            columns=result.get("columns", []),
                            operation=operation
                        ),
                        parse_mode="Markdown"
                    )

                    # Mark workflow complete
                    session.current_state = JoinWorkflowState.COMPLETE.value
                    await self.state_manager.update_session(session)
                    return

                elif job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    error_msg = job.error or "Unknown error"
                    await executing_msg.edit_text(
                        JoinMessages.error_message(error_msg),
                        parse_mode="Markdown"
                    )
                    return

            # Timeout
            await executing_msg.edit_text(
                JoinMessages.error_message("Operation timed out after 5 minutes"),
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"Failed to execute join job: {e}")
            await executing_msg.edit_text(
                JoinMessages.error_message(str(e)),
                parse_mode="Markdown"
            )

    async def handle_job_result(
        self,
        user_id: int,
        message: Dict[str, Any]
    ) -> None:
        """Handle job result from worker."""
        job_id = message.get("job_id")
        success = message.get("success", False)
        data = message.get("data", {})
        error = message.get("error")

        # Get session
        # Note: We need chat_id here, which we don't have directly
        # This would need to be stored in the job or session
        logger.info(f"Join job {job_id} result: success={success}")

        if success:
            output_file = data.get("output_file")
            row_count = data.get("row_count", 0)
            column_count = data.get("column_count", 0)
            columns = data.get("columns", [])
            operation = data.get("operation", "join")

            logger.info(f"Join complete: {output_file}, {row_count} rows")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_df_num_from_state(self, state: str) -> Optional[int]:
        """Extract dataframe number from state name."""
        for i in range(1, 5):
            if f"DF{i}" in state.upper():
                return i
        return None
