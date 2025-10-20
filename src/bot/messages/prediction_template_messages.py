"""
Prediction Template UI messages for Telegram bot.

This module provides user-facing messages for the ML prediction templates feature.
"""

from typing import Optional

# Template save prompts
PRED_TEMPLATE_SAVE_PROMPT = (
    "📝 *Enter a name for this prediction template:*\n"
    "*Rules:*\n"
    "• Only letters, numbers, and underscores\n"
    "• Maximum 32 characters\n"
    "• Must be unique\n"
    "*Example:* sales\\_forecast\\_template"
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


def format_pred_template_summary(template_name: str, file_path: str, model_id: str,
                                  features: list, output_column: str,
                                  description: Optional[str], created_at: str) -> str:
    """Format prediction template summary."""
    features_str = format_feature_list(features)
    created_date = escape_markdown(created_at[:10] if len(created_at) >= 10 else created_at)
    desc_str = f"*Description:* {escape_markdown(description)}\n\n" if description else ""

    return PRED_TEMPLATE_LOAD_SUMMARY.format(
        name=escape_markdown(template_name),
        file_path=escape_markdown(file_path),
        model_id=escape_markdown(model_id),
        features=features_str,
        output_column=escape_markdown(output_column),
        description=desc_str,
        created=created_date
    )
