"""
Prediction Template UI messages for Telegram bot.

This module provides user-facing messages for the ML prediction templates feature.
"""

from typing import Optional
from src.utils.i18n_manager import I18nManager

# Template save prompts
PRED_TEMPLATE_SAVE_PROMPT = (
    "📝 *Enter a template name:*\n\n"
    "*Format:*\n"
    "• Training: `TRAIN_<MODEL>_<TARGET>`\n"
    "• Prediction: `PREDICT_<MODEL>_<PREDICTED_COLUMN>`\n\n"
    "*Examples:*\n"
    "• `TRAIN_CATBST_CLASS2`\n"
    "• `PREDICT_XGBST_PRICE`\n"
    "• `TRAIN_KERAS_CHURN`\n\n"
    "Type your template name now:"
)

PRED_TEMPLATE_SAVED_SUCCESS = (
    "✅ *Template '{name}' saved successfully!*\n"
    "You can now use this template for future predictions."
)

PRED_TEMPLATE_UPDATED_SUCCESS = (
    "✅ *Template '{name}' updated successfully!*\n"
    "Your changes have been saved."
)

# Template load messages
PRED_TEMPLATE_LOAD_PROMPT = (
    "📋 *Select a prediction template:*\n"
    "You have {count} saved template(s)."
)

PRED_TEMPLATE_NO_TEMPLATES = (
    "📋 *No prediction templates found*\n"
    "Create a template by completing a prediction workflow "
    "and clicking '💾 Save as Template' after saving predictions."
)

PRED_TEMPLATE_LOAD_SUMMARY = (
    "📋 *Template:* `{name}`\n"
    "📁 *Data File:* `{file_path}`\n"
    "🤖 *Model:* `{model_id}`\n"
    "📊 *Features:* {features}\n"
    "📝 *Output Column:* `{output_column}`\n"
    "{description}"
    "*Created:* {created}"
)

# Error messages
PRED_TEMPLATE_INVALID_NAME = (
    "❌ *Invalid template name:* {error}\n"
    "Please try again:"
)

PRED_TEMPLATE_EXISTS = (
    "⚠️ *Template '{name}' already exists*\n"
    "Choose a different name or delete the existing template first."
)

PRED_TEMPLATE_NOT_FOUND = (
    "❌ *Template '{name}' not found*\n"
    "It may have been deleted."
)

PRED_TEMPLATE_SAVE_FAILED = (
    "❌ *Failed to save template:* {error}\n"
    "Please try again or contact support if the issue persists."
)

PRED_TEMPLATE_LOAD_FAILED = (
    "❌ *Failed to load template:* {error}\n"
    "Please try a different template or start a new prediction session."
)

PRED_TEMPLATE_MODEL_INVALID = (
    "❌ *Model not found*\n"
    "The model `{model_id}` from this template is no longer available.\n"
    "Please try a different template or train a new model."
)

PRED_TEMPLATE_FILE_PATH_INVALID = (
    "❌ *Invalid file path in template*\n"
    "The file path {path} is no longer valid:\n"
    "{error}\n"
    "Please try a different template or update the file path."
)

# Confirmation prompts
PRED_TEMPLATE_CONTINUE_PREDICTION = "What would you like to do next?"

# Success messages
PRED_TEMPLATE_DATA_LOADED = (
    "✅ *Data loaded from template!*\n"
    "📊 Rows: {rows}\n"
    "📊 Columns: {columns}\n"
    "Ready to run predictions."
)


# ============================================================================
# I18n-aware message methods
# ============================================================================

def pred_template_save_prompt(locale: Optional[str] = None) -> str:
    """Prompt for saving prediction template."""
    return I18nManager.t(
        'workflows.prediction.template_save_prompt',
        locale=locale,
        default=PRED_TEMPLATE_SAVE_PROMPT
    )


def pred_template_saved_success(name: str, locale: Optional[str] = None) -> str:
    """Message for successful template save."""
    return I18nManager.t(
        'workflows.prediction.template_saved_success',
        locale=locale,
        name=name,
        default=PRED_TEMPLATE_SAVED_SUCCESS.format(name=name)
    )


def pred_template_updated_success(name: str, locale: Optional[str] = None) -> str:
    """Message for successful template update."""
    return I18nManager.t(
        'workflows.prediction.template_updated_success',
        locale=locale,
        name=name,
        default=PRED_TEMPLATE_UPDATED_SUCCESS.format(name=name)
    )


def pred_template_load_prompt(count: int, locale: Optional[str] = None) -> str:
    """Prompt for loading prediction template."""
    return I18nManager.t(
        'workflows.prediction.template_load_prompt',
        locale=locale,
        count=count,
        default=PRED_TEMPLATE_LOAD_PROMPT.format(count=count)
    )


def pred_template_no_templates(locale: Optional[str] = None) -> str:
    """Message when no templates found."""
    return I18nManager.t(
        'workflows.prediction.template_no_templates',
        locale=locale,
        default=PRED_TEMPLATE_NO_TEMPLATES
    )


def pred_template_invalid_name(error: str, locale: Optional[str] = None) -> str:
    """Message for invalid template name."""
    return I18nManager.t(
        'workflows.prediction.template_invalid_name',
        locale=locale,
        error=error,
        default=PRED_TEMPLATE_INVALID_NAME.format(error=error)
    )


def pred_template_exists(name: str, locale: Optional[str] = None) -> str:
    """Message when template already exists."""
    return I18nManager.t(
        'workflows.prediction.template_exists',
        locale=locale,
        name=name,
        default=PRED_TEMPLATE_EXISTS.format(name=name)
    )


def pred_template_not_found(name: str, locale: Optional[str] = None) -> str:
    """Message when template not found."""
    return I18nManager.t(
        'workflows.prediction.template_not_found',
        locale=locale,
        name=name,
        default=PRED_TEMPLATE_NOT_FOUND.format(name=name)
    )


def pred_template_save_failed(error: str, locale: Optional[str] = None) -> str:
    """Message for failed template save."""
    return I18nManager.t(
        'workflows.prediction.template_save_failed',
        locale=locale,
        error=error,
        default=PRED_TEMPLATE_SAVE_FAILED.format(error=error)
    )


def pred_template_load_failed(error: str, locale: Optional[str] = None) -> str:
    """Message for failed template load."""
    return I18nManager.t(
        'workflows.prediction.template_load_failed',
        locale=locale,
        error=error,
        default=PRED_TEMPLATE_LOAD_FAILED.format(error=error)
    )


def pred_template_model_invalid(model_id: str, locale: Optional[str] = None) -> str:
    """Message when model from template is invalid."""
    return I18nManager.t(
        'workflows.prediction.template_model_invalid',
        locale=locale,
        model_id=model_id,
        default=PRED_TEMPLATE_MODEL_INVALID.format(model_id=model_id)
    )


def pred_template_file_path_invalid(
    path: str,
    error: str,
    locale: Optional[str] = None
) -> str:
    """Message when template file path is invalid."""
    return I18nManager.t(
        'workflows.prediction.template_file_path_invalid',
        locale=locale,
        path=path,
        error=error,
        default=PRED_TEMPLATE_FILE_PATH_INVALID.format(path=path, error=error)
    )


def pred_template_data_loaded(
    rows: int,
    columns: int,
    locale: Optional[str] = None
) -> str:
    """Message for successful data loading from template."""
    return I18nManager.t(
        'workflows.prediction.template_data_loaded',
        locale=locale,
        rows=rows,
        columns=columns,
        default=PRED_TEMPLATE_DATA_LOADED.format(rows=rows, columns=columns)
    )


# Helper functions
def escape_markdown(text: str) -> str:
    """Escape special Telegram Markdown characters (_ and *)."""
    return text.replace('_', '\\_').replace('*', '\\*')


def format_feature_list(features: list, max_display: int = 3) -> str:
    """Format feature list, truncating if too long."""
    if len(features) <= max_display:
        return ", ".join(f"`{f}`" for f in features)
    displayed = ", ".join(f"`{f}`" for f in features[:max_display])
    return f"{displayed} ... (+{len(features) - max_display} more)"


def format_pred_template_summary(
    template_name: str,
    file_path: str,
    model_id: str,
    features: list,
    output_column: str,
    description: Optional[str],
    created_at: str,
    locale: Optional[str] = None
) -> str:
    """Format prediction template summary with i18n support."""
    features_str = format_feature_list(features)
    created_date = escape_markdown(created_at[:10] if len(created_at) >= 10 else created_at)
    desc_str = f"*Description:* {escape_markdown(description)}\n\n" if description else ""

    return I18nManager.t(
        'workflows.prediction.template_load_summary',
        locale=locale,
        name=escape_markdown(template_name),
        file_path=escape_markdown(file_path),
        model_id=escape_markdown(model_id),
        features=features_str,
        output_column=escape_markdown(output_column),
        description=desc_str,
        created=created_date,
        default=PRED_TEMPLATE_LOAD_SUMMARY.format(
            name=escape_markdown(template_name),
            file_path=escape_markdown(file_path),
            model_id=escape_markdown(model_id),
            features=features_str,
            output_column=escape_markdown(output_column),
            description=desc_str,
            created=created_date
        )
    )
