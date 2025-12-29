"""
Unified Template UI messages for Telegram bot.

This module provides user-facing messages for the unified ML template system
(training + prediction templates).
"""

from typing import List, Optional
from src.utils.i18n_manager import I18nManager


class TemplateMessages:
    """User-facing messages for template operations."""

    @staticmethod
    def template_help(
        train_templates: List[str],
        predict_templates: List[str],
        locale: Optional[str] = None
    ) -> str:
        """
        Help message for /template command.

        Args:
            train_templates: List of training template names
            predict_templates: List of prediction template names
            locale: User's language preference

        Returns:
            Formatted help message
        """
        base_msg = I18nManager.t(
            'templates.help.header',
            locale=locale,
            default=(
                "📋 *ML Template System*\n\n"
                "*What are templates?*\n"
                "Templates let you save and replay ML workflows with a single command.\n\n"
                "*Naming Format:*\n"
                "• Training: `TRAIN_<MODEL>_<TARGET>`\n"
                "• Prediction: `PREDICT_<MODEL>_<COLUMN>`\n"
                "• Model/Target: 1-8 uppercase letters/numbers\n"
                "• Example: `TRAIN_CATBST_CLASS2`\n\n"
                "*Usage:*\n"
                "• `/template` - Show this help\n"
                "• `/template <name>` - Run template\n"
                "• `/template list` - List all templates\n"
                "• `/template delete <name>` - Delete template\n\n"
                "*Creating Templates:*\n"
                "1. Complete a /train or /predict workflow\n"
                "2. Click 'Save as Template' button\n"
                "3. Enter name in format above\n\n"
            )
        )

        # Add user's templates section
        total_count = len(train_templates) + len(predict_templates)
        if total_count > 0:
            templates_section = I18nManager.t(
                'templates.help.your_templates',
                locale=locale,
                default=f"*Your Templates ({total_count}):*\n"
            )

            if train_templates:
                templates_section += "*TRAIN:* " + ", ".join(f"`{t}`" for t in train_templates[:5])
                if len(train_templates) > 5:
                    templates_section += f" (+{len(train_templates) - 5} more)"
                templates_section += "\n"

            if predict_templates:
                templates_section += "*PREDICT:* " + ", ".join(f"`{t}`" for t in predict_templates[:5])
                if len(predict_templates) > 5:
                    templates_section += f" (+{len(predict_templates) - 5} more)"
                templates_section += "\n"

            base_msg += templates_section
        else:
            base_msg += I18nManager.t(
                'templates.help.no_templates',
                locale=locale,
                default="*You have no saved templates yet.*\n"
            )

        return base_msg

    @staticmethod
    def template_name_prompt(locale: Optional[str] = None) -> str:
        """Prompt for template name entry."""
        return I18nManager.t(
            'templates.save.name_prompt',
            locale=locale,
            default=(
                "📝 *Enter a template name:*\n\n"
                "*Format:*\n"
                "• Training: `TRAIN_<MODEL>_<TARGET>`\n"
                "• Prediction: `PREDICT_<MODEL>_<COLUMN>`\n"
                "• Model/Target/Column: 1-8 uppercase alphanumeric\n\n"
                "*Examples:*\n"
                "• `TRAIN_CATBST_CLASS2`\n"
                "• `PREDICT_XGBST_PRICE`\n"
                "• `TRAIN_KERAS_CHURN`\n\n"
                "Type your template name now:"
            )
        )

    @staticmethod
    def template_saved(name: str, locale: Optional[str] = None) -> str:
        """Success message after saving template."""
        return I18nManager.t(
            'templates.save.success',
            locale=locale,
            name=name,
            default=(
                f"✅ *Template '{name}' saved successfully!*\n\n"
                f"Use `/template {name}` to run this workflow anytime."
            )
        )

    @staticmethod
    def template_invalid_name(error: str, locale: Optional[str] = None) -> str:
        """Error message for invalid template name."""
        return I18nManager.t(
            'templates.save.invalid_name',
            locale=locale,
            error=error,
            default=f"❌ *Invalid template name*\n\n{error}"
        )

    @staticmethod
    def template_already_exists(name: str, locale: Optional[str] = None) -> str:
        """Error message when template name already exists."""
        return I18nManager.t(
            'templates.save.already_exists',
            locale=locale,
            name=name,
            default=(
                f"⚠️ *Template '{name}' already exists*\n\n"
                f"Choose a different name or delete the existing template first:\n"
                f"`/template delete {name}`"
            )
        )

    @staticmethod
    def template_missing_config(locale: Optional[str] = None) -> str:
        """Error message when template config is incomplete."""
        return I18nManager.t(
            'templates.save.missing_config',
            locale=locale,
            default=(
                "❌ *Incomplete configuration*\n\n"
                "Cannot save template - missing required workflow parameters.\n"
                "Please complete the workflow before saving as template."
            )
        )

    @staticmethod
    def template_save_failed(locale: Optional[str] = None) -> str:
        """Error message when template save fails."""
        return I18nManager.t(
            'templates.save.failed',
            locale=locale,
            default=(
                "❌ *Failed to save template*\n\n"
                "An error occurred while saving. Please try again."
            )
        )

    @staticmethod
    def template_not_found(name: str, locale: Optional[str] = None) -> str:
        """Error message when template doesn't exist."""
        return I18nManager.t(
            'templates.errors.not_found',
            locale=locale,
            name=name,
            default=(
                f"❌ *Template '{name}' not found*\n\n"
                f"Use `/template` to see available templates."
            )
        )

    @staticmethod
    def template_deleted(name: str, locale: Optional[str] = None) -> str:
        """Success message after deleting template."""
        return I18nManager.t(
            'templates.delete.success',
            locale=locale,
            name=name,
            default=f"✅ *Template '{name}' deleted successfully.*"
        )

    @staticmethod
    def template_list_header(
        total: int,
        train_count: int,
        predict_count: int,
        locale: Optional[str] = None
    ) -> str:
        """Header for template list."""
        return I18nManager.t(
            'templates.list.header',
            locale=locale,
            total=total,
            train_count=train_count,
            predict_count=predict_count,
            default=(
                f"📋 *Your Templates ({total} total)*\n"
                f"• TRAIN: {train_count}\n"
                f"• PREDICT: {predict_count}"
            )
        )

    @staticmethod
    def template_no_templates(locale: Optional[str] = None) -> str:
        """Message when user has no templates."""
        return I18nManager.t(
            'templates.list.no_templates',
            locale=locale,
            default=(
                "📋 *No templates found*\n\n"
                "Create templates by completing /train or /predict workflows "
                "and clicking 'Save as Template'."
            )
        )

    @staticmethod
    def template_executing(name: str, locale: Optional[str] = None) -> str:
        """Message when template execution starts."""
        return I18nManager.t(
            'templates.execute.starting',
            locale=locale,
            name=name,
            default=f"🚀 *Executing template '{name}'...*\n\nPlease wait while the workflow runs."
        )

    @staticmethod
    def template_execution_failed(error: str, locale: Optional[str] = None) -> str:
        """Error message when template execution fails."""
        return I18nManager.t(
            'templates.execute.failed',
            locale=locale,
            error=error,
            default=f"❌ *Template execution failed*\n\nError: {error}"
        )

    @staticmethod
    def template_file_not_found(path: str, locale: Optional[str] = None) -> str:
        """Error message when template's data file doesn't exist."""
        return I18nManager.t(
            'templates.execute.file_not_found',
            locale=locale,
            path=path,
            default=(
                f"❌ *Data file not found*\n\n"
                f"The file path in this template no longer exists:\n"
                f"`{path}`\n\n"
                f"Please verify the file location or use a different template."
            )
        )

    @staticmethod
    def template_invalid_type(template_type: str, locale: Optional[str] = None) -> str:
        """Error message for invalid template type."""
        return I18nManager.t(
            'templates.errors.invalid_type',
            locale=locale,
            template_type=template_type,
            default=f"❌ *Invalid template type: {template_type}*\n\nTemplate is corrupted."
        )

    @staticmethod
    def template_invalid_workflow(locale: Optional[str] = None) -> str:
        """Error message when template save triggered from wrong workflow."""
        return I18nManager.t(
            'templates.errors.invalid_workflow',
            locale=locale,
            default=(
                "❌ *Invalid workflow*\n\n"
                "Templates can only be saved from /train or /predict workflows."
            )
        )

    @staticmethod
    def template_transition_error(error: str, locale: Optional[str] = None) -> str:
        """Error message when state transition fails."""
        return I18nManager.t(
            'templates.errors.transition_failed',
            locale=locale,
            error=error,
            default=f"❌ *State transition failed*\n\nError: {error}"
        )

    @staticmethod
    def template_invalid_usage(locale: Optional[str] = None) -> str:
        """Error message for invalid command usage."""
        return I18nManager.t(
            'templates.errors.invalid_usage',
            locale=locale,
            default=(
                "❌ *Invalid usage*\n\n"
                "*Valid commands:*\n"
                "• `/template` - Show help\n"
                "• `/template <name>` - Run template\n"
                "• `/template list` - List templates\n"
                "• `/template delete <name>` - Delete template"
            )
        )
