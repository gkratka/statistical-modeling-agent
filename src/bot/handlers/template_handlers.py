"""
Template workflow handlers for ML training and prediction.

This module provides unified handlers for saving and loading ML templates
for both training and prediction workflows.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackContext

from src.core.state_manager import MLTrainingState, MLPredictionState, StateManager
from src.bot.messages.unified_template_messages import TemplateMessages
from src.utils.i18n_manager import I18nManager

logger = logging.getLogger(__name__)


class TemplateHandlers:
    """Handlers for unified ML template operations (train + predict)."""

    # Template naming pattern: TRAIN_XXXXXXXX_XXXXXXXX or PREDICT_XXXXXXXX_XXXXXXXX
    TEMPLATE_NAME_PATTERN = r'^(TRAIN|PREDICT)_[A-Z0-9]{1,8}_[A-Z0-9]{1,8}$'
    MAX_TEMPLATES_PER_USER = 50
    TEMPLATES_DIR = Path("templates")

    def __init__(self, state_manager: StateManager):
        """
        Initialize template handlers.

        Args:
            state_manager: StateManager instance
        """
        self.state_manager = state_manager
        self.templates_dir = self.TEMPLATES_DIR
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TemplateHandlers initialized with directory: {self.templates_dir}")

    # =========================================================================
    # Template Command Handlers
    # =========================================================================

    async def handle_template_command(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """
        Handle /template command with various usages:
        - /template → Show help and list templates
        - /template <name> → Execute template
        - /template delete <name> → Delete template
        - /template list → List all templates with details
        """
        user_id = update.effective_user.id
        locale = await self._get_user_locale(user_id)

        # Parse command arguments
        args = context.args or []

        if len(args) == 0:
            # Show help + template list
            await self._show_template_help(update, user_id, locale)
        elif len(args) == 1 and args[0].lower() == "list":
            # List templates with details (check BEFORE generic single-arg)
            await self.handle_template_list(update, user_id, locale)
        elif len(args) == 2 and args[0].lower() == "delete":
            # Delete template
            template_name = args[1].upper()
            await self.handle_template_delete(update, user_id, template_name, locale)
        elif len(args) == 1:
            # Execute template (generic single-arg, after "list" check)
            template_name = args[0].upper()
            await self.handle_template_execution(update, context, user_id, template_name, locale)
        else:
            # Invalid usage
            await update.message.reply_text(
                TemplateMessages.template_invalid_usage(locale=locale),
                parse_mode="Markdown"
            )

    async def handle_template_execution(
        self,
        update: Update,
        context: CallbackContext,
        user_id: int,
        template_name: str,
        locale: Optional[str]
    ) -> None:
        """Execute a template workflow."""
        # Validate template exists
        template = self._load_template(user_id, template_name)
        if not template:
            await update.message.reply_text(
                TemplateMessages.template_not_found(name=template_name, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Execute based on type
        template_type = template.get("type")
        config = template.get("config", {})

        if template_type == "train":
            await self._execute_train_template(update, context, user_id, template_name, config, locale)
        elif template_type == "predict":
            await self._execute_predict_template(update, context, user_id, template_name, config, locale)
        else:
            await update.message.reply_text(
                TemplateMessages.template_invalid_type(template_type=template_type, locale=locale),
                parse_mode="Markdown"
            )

    async def handle_template_delete(
        self,
        update: Update,
        user_id: int,
        template_name: str,
        locale: Optional[str]
    ) -> None:
        """Delete a template."""
        success = self._delete_template(user_id, template_name)
        if success:
            await update.message.reply_text(
                TemplateMessages.template_deleted(name=template_name, locale=locale),
                parse_mode="Markdown"
            )
            logger.info(f"Template '{template_name}' deleted for user {user_id}")
        else:
            await update.message.reply_text(
                TemplateMessages.template_not_found(name=template_name, locale=locale),
                parse_mode="Markdown"
            )

    async def handle_template_list(
        self,
        update: Update,
        user_id: int,
        locale: Optional[str]
    ) -> None:
        """List all templates with detailed information."""
        templates = self._list_templates(user_id)

        if not templates:
            await update.message.reply_text(
                TemplateMessages.template_no_templates(locale=locale),
                parse_mode="Markdown"
            )
            return

        # Group by type (templates is list of (name, data) tuples)
        train_templates = [(name, data) for name, data in templates if data.get("type") == "train"]
        predict_templates = [(name, data) for name, data in templates if data.get("type") == "predict"]

        message = TemplateMessages.template_list_header(
            total=len(templates),
            train_count=len(train_templates),
            predict_count=len(predict_templates),
            locale=locale
        )

        # List train templates
        if train_templates:
            message += "\n\n*TRAIN Templates:*\n"
            for template_name, template_data in train_templates:
                created = template_data.get("created_at", "Unknown")[:10]
                config = template_data.get("config", {})
                target = config.get("target_column", "N/A")
                model = config.get("model_type", "N/A")
                message += f"• `{template_name}` - Target: {target}, Model: {model} (Created: {created})\n"

        # List predict templates
        if predict_templates:
            message += "\n\n*PREDICT Templates:*\n"
            for template_name, template_data in predict_templates:
                created = template_data.get("created_at", "Unknown")[:10]
                config = template_data.get("config", {})
                model_id = config.get("model_id", "N/A")
                message += f"• `{template_name}` - Model: {model_id[:30]}... (Created: {created})\n"

        await update.message.reply_text(message, parse_mode="Markdown")

    # =========================================================================
    # Template Save Handlers (Button-triggered)
    # =========================================================================

    async def handle_save_template_button(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle 'Save as Template' button click (train or predict workflow)."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            locale = None
            await query.edit_message_text(
                I18nManager.t('workflow_state.session_not_found', locale=locale, command='/train or /predict')
            )
            return

        locale = session.language if session.language else None

        # Determine workflow type
        if session.workflow_type and session.workflow_type.value == "ml_training":
            state_value = MLTrainingState.SAVING_TEMPLATE.value
        elif session.workflow_type and session.workflow_type.value == "ml_prediction":
            state_value = MLPredictionState.SAVING_PRED_TEMPLATE.value
        else:
            await query.edit_message_text(
                TemplateMessages.template_invalid_workflow(locale=locale),
                parse_mode="Markdown"
            )
            return

        # Save state snapshot and transition
        session.save_state_snapshot()
        success, error_msg, _ = await self.state_manager.transition_state(session, state_value)
        if not success:
            await query.edit_message_text(
                TemplateMessages.template_transition_error(error=error_msg, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Show template name prompt
        keyboard = [[InlineKeyboardButton(
            I18nManager.t('templates.save.cancel_button', locale=locale, default="❌ Cancel"),
            callback_data="cancel_template"
        )]]
        await query.edit_message_text(
            TemplateMessages.template_name_prompt(locale=locale),
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

        if not session:
            return

        # Check if in template saving state
        current_state = session.current_state
        is_train_template = current_state == MLTrainingState.SAVING_TEMPLATE.value
        is_predict_template = current_state == MLPredictionState.SAVING_PRED_TEMPLATE.value

        if not (is_train_template or is_predict_template):
            return

        locale = session.language if session.language else None
        template_name = update.message.text.strip().upper()

        # Validate name
        is_valid, error_msg = self._validate_template_name(template_name)
        if not is_valid:
            await update.message.reply_text(
                TemplateMessages.template_invalid_name(error=error_msg, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Check if exists
        if self._template_exists(user_id, template_name):
            await update.message.reply_text(
                TemplateMessages.template_already_exists(name=template_name, locale=locale),
                parse_mode="Markdown"
            )
            return

        # Build template config based on workflow type
        if is_train_template:
            template_config = self._build_train_template_config(session)
            template_type = "train"
        else:
            template_config = self._build_predict_template_config(session)
            template_type = "predict"

        # Validate required fields
        if not self._validate_template_config(template_config, template_type):
            await update.message.reply_text(
                TemplateMessages.template_missing_config(locale=locale),
                parse_mode="Markdown"
            )
            return

        # Save template
        template_data = {
            "type": template_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": template_config
        }

        success = self._save_template(user_id, template_name, template_data)
        if success:
            # Save local backup (non-blocking)
            file_path = template_config.get("file_path", "")
            self._save_local_backup(
                template_name=template_name,
                template_config=template_config,
                template_type=template_type,
                file_path=file_path
            )

            await update.message.reply_text(
                TemplateMessages.template_saved(name=template_name, locale=locale),
                parse_mode="Markdown"
            )
            logger.info(f"Template '{template_name}' saved for user {user_id}")

            # Sync to ML Training storage (TemplateManager) for train templates
            # This ensures both storage systems have consistent defer_loading values
            if is_train_template:
                try:
                    from src.core.template_manager import TemplateManager
                    from src.core.training_template import TemplateConfig
                    tm = TemplateManager(TemplateConfig())
                    ml_config = {
                        "file_path": template_config.get("file_path", ""),
                        "defer_loading": template_config.get("defer_loading", False),
                        "target_column": template_config.get("target_column", ""),
                        "feature_columns": template_config.get("feature_columns", []),
                        "model_category": template_config.get("model_category", ""),
                        "model_type": template_config.get("model_type", ""),
                        "hyperparameters": template_config.get("model_parameters", {})
                    }
                    tm.save_template(user_id, template_name, ml_config)
                    logger.info(f"Template '{template_name}' synced to ML Training storage")
                except Exception as e:
                    logger.warning(f"Failed to sync template to ML storage: {e}")

            # Restore previous state
            session.restore_previous_state()
            await self.state_manager.update_session(session)
        else:
            await update.message.reply_text(
                TemplateMessages.template_save_failed(locale=locale),
                parse_mode="Markdown"
            )

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    async def _show_template_help(
        self,
        update: Update,
        user_id: int,
        locale: Optional[str]
    ) -> None:
        """Show template help message with user's template list."""
        templates = self._list_templates(user_id)

        # Group by type
        train_templates = [name for name, data in templates if data.get("type") == "train"]
        predict_templates = [name for name, data in templates if data.get("type") == "predict"]

        message = TemplateMessages.template_help(
            train_templates=train_templates,
            predict_templates=predict_templates,
            locale=locale
        )

        await update.message.reply_text(message, parse_mode="Markdown")

    async def _execute_train_template(
        self,
        update: Update,
        context: CallbackContext,
        user_id: int,
        template_name: str,
        config: Dict[str, Any],
        locale: Optional[str]
    ) -> None:
        """
        Execute a training template.

        Note: For MVP, templates just display configuration.
        Full automation (auto-submit jobs) can be added in future iterations.
        """
        await update.message.reply_text(
            TemplateMessages.template_executing(name=template_name, locale=locale),
            parse_mode="Markdown"
        )

        try:
            # Extract training parameters from config
            file_path = config.get("file_path")
            target_column = config.get("target_column")
            feature_columns = config.get("feature_columns", [])
            model_type = config.get("model_type")

            # Validate file exists
            if not Path(file_path).exists():
                await update.message.reply_text(
                    TemplateMessages.template_file_not_found(path=file_path, locale=locale),
                    parse_mode="Markdown"
                )
                return

            # Display template configuration
            features_str = ", ".join(f"`{f}`" for f in feature_columns[:5])
            if len(feature_columns) > 5:
                features_str += f" (+{len(feature_columns) - 5} more)"

            config_msg = (
                f"✅ *Template Configuration Loaded*\n\n"
                f"📁 *Data:* `{file_path}`\n"
                f"🎯 *Target:* `{target_column}`\n"
                f"📊 *Features:* {features_str}\n"
                f"🤖 *Model:* `{model_type}`\n\n"
                f"*Next Steps:*\n"
                f"1. Use `/train` to start training workflow\n"
                f"2. Select 'Use Template' → `{template_name}`\n"
                f"Or manually train with these settings."
            )

            await update.message.reply_text(config_msg, parse_mode="Markdown")
            logger.info(f"Train template '{template_name}' loaded for user {user_id}")

        except Exception as e:
            logger.error(f"Error loading train template: {e}", exc_info=True)
            await update.message.reply_text(
                TemplateMessages.template_execution_failed(error=str(e), locale=locale),
                parse_mode="Markdown"
            )

    async def _execute_predict_template(
        self,
        update: Update,
        context: CallbackContext,
        user_id: int,
        template_name: str,
        config: Dict[str, Any],
        locale: Optional[str]
    ) -> None:
        """
        Execute a prediction template.

        Note: For MVP, templates just display configuration.
        Full automation (auto-submit jobs) can be added in future iterations.
        """
        await update.message.reply_text(
            TemplateMessages.template_executing(name=template_name, locale=locale),
            parse_mode="Markdown"
        )

        try:
            # Extract prediction parameters from config
            file_path = config.get("file_path")
            model_id = config.get("model_id")
            feature_columns = config.get("feature_columns", [])
            output_column_name = config.get("output_column_name", "prediction")

            # Validate file exists
            if not Path(file_path).exists():
                await update.message.reply_text(
                    TemplateMessages.template_file_not_found(path=file_path, locale=locale),
                    parse_mode="Markdown"
                )
                return

            # Display template configuration
            features_str = ", ".join(f"`{f}`" for f in feature_columns[:5])
            if len(feature_columns) > 5:
                features_str += f" (+{len(feature_columns) - 5} more)"

            config_msg = (
                f"✅ *Template Configuration Loaded*\n\n"
                f"📁 *Data:* `{file_path}`\n"
                f"🤖 *Model:* `{model_id[:50]}...`\n"
                f"📊 *Features:* {features_str}\n"
                f"📝 *Output Column:* `{output_column_name}`\n\n"
                f"*Next Steps:*\n"
                f"1. Use `/predict` to start prediction workflow\n"
                f"2. Select 'Use Template' → `{template_name}`\n"
                f"Or manually predict with these settings."
            )

            await update.message.reply_text(config_msg, parse_mode="Markdown")
            logger.info(f"Predict template '{template_name}' loaded for user {user_id}")

        except Exception as e:
            logger.error(f"Error loading predict template: {e}", exc_info=True)
            await update.message.reply_text(
                TemplateMessages.template_execution_failed(error=str(e), locale=locale),
                parse_mode="Markdown"
            )

    def _build_train_template_config(self, session) -> Dict[str, Any]:
        """Build training template config from session."""
        logger.info(f"Building train template config, session.file_path='{session.file_path}'")
        return {
            "file_path": session.file_path or "",
            "defer_loading": getattr(session, "load_deferred", False),
            "target_column": session.selections.get("target_column") or session.selections.get("target", ""),
            "feature_columns": session.selections.get("feature_columns") or session.selections.get("features", []),
            "model_type": session.selections.get("model_type", ""),
            "hyperparameters": session.selections.get("hyperparameters", {}),
            "model_name": session.selections.get("model_name")
        }

    def _build_predict_template_config(self, session) -> Dict[str, Any]:
        """Build prediction template config from session."""
        return {
            "file_path": session.file_path or "",
            "model_id": session.selections.get("selected_model_id", ""),
            "feature_columns": session.selections.get("selected_features", []),
            "output_column_name": session.selections.get("prediction_column_name", "prediction"),
            "output_path": session.selections.get("output_file_path")
        }

    def _validate_template_config(self, config: Dict[str, Any], template_type: str) -> bool:
        """Validate template configuration has required fields."""
        if template_type == "train":
            required = ["file_path", "target_column", "feature_columns", "model_type"]
        else:  # predict
            required = ["file_path", "model_id", "feature_columns", "output_column_name"]

        return all(config.get(field) for field in required)

    def _validate_template_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate template name format.

        Format: TRAIN_XXXXXXXX_XXXXXXXX or PREDICT_XXXXXXXX_XXXXXXXX
        - TRAIN or PREDICT prefix
        - Two segments of 1-8 uppercase alphanumeric characters
        - Underscore separators

        Args:
            name: Template name to validate

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if not name:
            return False, "Template name cannot be empty"

        if not re.match(self.TEMPLATE_NAME_PATTERN, name):
            return False, (
                "Invalid format. Use: TRAIN_XXXXXXXX_XXXXXXXX or PREDICT_XXXXXXXX_XXXXXXXX\n"
                "• TRAIN or PREDICT prefix\n"
                "• Two segments: 1-8 uppercase letters/numbers\n"
                "• Example: TRAIN_CATBST_CLASS2"
            )

        return True, ""

    async def _get_user_locale(self, user_id: int) -> Optional[str]:
        """Get user's language preference from session."""
        # Try to find any active session for this user
        from src.core.state_manager import WorkflowType
        for workflow_type in WorkflowType:
            session = await self.state_manager.get_session(user_id, f"chat_{user_id}")
            if session:
                return session.language if session.language else None
        return None

    def _get_user_file_path(self, user_id: int) -> Path:
        """Get path to user's template file."""
        return self.templates_dir / f"{user_id}.json"

    def _load_all_templates(self, user_id: int) -> Dict[str, Any]:
        """Load all templates for a user."""
        file_path = self._get_user_file_path(user_id)
        if not file_path.exists():
            return {}

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading templates for user {user_id}: {e}")
            return {}

    def _save_all_templates(self, user_id: int, templates: Dict[str, Any]) -> bool:
        """Save all templates for a user."""
        file_path = self._get_user_file_path(user_id)
        try:
            with open(file_path, 'w') as f:
                json.dump(templates, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving templates for user {user_id}: {e}")
            return False

    def _save_template(self, user_id: int, template_name: str, template_data: Dict[str, Any]) -> bool:
        """Save a single template."""
        templates = self._load_all_templates(user_id)

        # Check template limit
        if template_name not in templates and len(templates) >= self.MAX_TEMPLATES_PER_USER:
            logger.warning(f"User {user_id} exceeded template limit ({self.MAX_TEMPLATES_PER_USER})")
            return False

        templates[template_name] = template_data
        return self._save_all_templates(user_id, templates)

    def _load_template(self, user_id: int, template_name: str) -> Optional[Dict[str, Any]]:
        """Load a single template."""
        templates = self._load_all_templates(user_id)
        return templates.get(template_name)

    def _delete_template(self, user_id: int, template_name: str) -> bool:
        """Delete a single template."""
        templates = self._load_all_templates(user_id)
        if template_name not in templates:
            return False

        del templates[template_name]
        return self._save_all_templates(user_id, templates)

    def _template_exists(self, user_id: int, template_name: str) -> bool:
        """Check if template exists."""
        templates = self._load_all_templates(user_id)
        return template_name in templates

    def _list_templates(self, user_id: int) -> List[Tuple[str, Dict[str, Any]]]:
        """
        List all templates for user.

        Returns:
            List of (template_name, template_data) tuples, sorted by creation date (newest first)
        """
        templates = self._load_all_templates(user_id)
        template_list = list(templates.items())

        # Sort by created_at (newest first)
        template_list.sort(
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )

        return template_list

    def _save_local_backup(
        self,
        template_name: str,
        template_config: dict,
        template_type: str,
        file_path: str
    ) -> None:
        """Save template backup to local filesystem.

        Args:
            template_name: Name of the template
            template_config: Template configuration dict
            template_type: Type of template ('train' or 'predict')
            file_path: Path to data file (backup saved in same directory)
        """
        try:
            logger.info(f"_save_local_backup called with file_path='{file_path}'")
            if not file_path:
                logger.warning("No file_path provided, skipping local backup")
                return

            backup_dir = Path(file_path).parent
            backup_path = backup_dir / f"template_{template_name}.json"

            backup_data = {
                "name": template_name,
                "type": template_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **template_config
            }

            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)

            logger.info(f"Local template backup saved: {backup_path}")

        except Exception as e:
            logger.warning(f"Failed to save local template backup: {e}")
