"""
Template-related UI messages for Telegram bot.

This module provides user-facing messages for the ML training templates feature.
"""

# Template save prompts
TEMPLATE_SAVE_PROMPT = (
    "📝 *Enter a name for this template:*\n\n"
    "*Rules:*\n"
    "• Only letters, numbers, and underscores\n"
    "• Maximum 32 characters\n"
    "• Must be unique\n\n"
    "*Example:* housing\\_rf\\_model"
)

TEMPLATE_SAVED_SUCCESS = (
    "✅ *Template '{name}' saved successfully!*\n\n"
    "You can now use this template for future training sessions."
)

TEMPLATE_UPDATED_SUCCESS = (
    "✅ *Template '{name}' updated successfully!*\n\n"
    "Your changes have been saved."
)

# Template load messages
TEMPLATE_LOAD_PROMPT = (
    "📋 *Select a template:*\n\n"
    "You have {count} saved template(s)."
)

TEMPLATE_NO_TEMPLATES = (
    "📋 *No templates found*\n\n"
    "Create a template by completing a training workflow "
    "and clicking '💾 Save as Template' before training starts."
)

TEMPLATE_LOAD_SUMMARY = (
    "📋 *Template:* `{name}`\n\n"
    "📁 *Data:* `{file_path}`\n"
    "🎯 *Target:* `{target}`\n"
    "📊 *Features:* {features}\n"
    "🤖 *Model:* {category} / {type}\n\n"
    "*Created:* {created}"
)

# Error messages
TEMPLATE_INVALID_NAME = (
    "❌ *Invalid template name:* {error}\n\n"
    "Please try again:"
)

TEMPLATE_EXISTS = (
    "⚠️ *Template '{name}' already exists*\n\n"
    "Choose a different name or delete the existing template first."
)

TEMPLATE_NOT_FOUND = (
    "❌ *Template '{name}' not found*\n\n"
    "It may have been deleted."
)

TEMPLATE_SAVE_FAILED = (
    "❌ *Failed to save template:* {error}\n\n"
    "Please try again or contact support if the issue persists."
)

TEMPLATE_LOAD_FAILED = (
    "❌ *Failed to load template:* {error}\n\n"
    "Please try a different template or start a new training session."
)

TEMPLATE_FILE_PATH_INVALID = (
    "❌ *Invalid file path in template*\n\n"
    "The file path `{path}` is no longer valid:\n"
    "{error}\n\n"
    "Please try a different template or update the file path."
)

# Confirmation prompts
TEMPLATE_CONTINUE_TRAINING = (
    "What would you like to do next?"
)

TEMPLATE_LOAD_DATA_PROMPT = (
    "📊 *Data loading options:*\n\n"
    "• *Load Data Now:* Load the data file and start training immediately\n"
    "• *Defer Loading:* Save configuration and load data later with /continue"
)

# Success messages
TEMPLATE_DATA_LOADED = (
    "✅ *Data loaded successfully!*\n\n"
    "📊 Rows: {rows}\n"
    "📊 Columns: {columns}\n\n"
    "Ready to train model."
)

TEMPLATE_DATA_DEFERRED = (
    "⏳ *Data loading deferred*\n\n"
    "Configuration saved. Use `/continue` when ready to load data and train."
)

# Helper functions
def escape_markdown(text: str) -> str:
    """
    Escape special Telegram Markdown characters.

    Telegram uses _ for italic and * for bold. Escape these in user-provided
    content to prevent parsing errors.

    Args:
        text: Text to escape

    Returns:
        Text with Markdown special characters escaped
    """
    return text.replace('_', '\\_').replace('*', '\\*')


def format_feature_list(features: list, max_display: int = 3) -> str:
    """Format feature list for display, truncating if too long."""
    if len(features) <= max_display:
        return ", ".join(f"`{f}`" for f in features)
    else:
        displayed = ", ".join(f"`{f}`" for f in features[:max_display])
        remaining = len(features) - max_display
        return f"{displayed} ... (+{remaining} more)"


def format_template_summary(template_name: str, file_path: str, target: str,
                            features: list, model_category: str, model_type: str,
                            created_at: str) -> str:
    """Format complete template summary for display."""
    features_str = format_feature_list(features)
    created_date = created_at[:10] if len(created_at) >= 10 else created_at
    # Escape created_date to handle underscores in timestamp format (e.g., "20251013_0")
    created_date = escape_markdown(created_date)

    return TEMPLATE_LOAD_SUMMARY.format(
        name=escape_markdown(template_name),
        file_path=escape_markdown(file_path),
        target=escape_markdown(target),
        features=features_str,
        category=escape_markdown(model_category),
        type=escape_markdown(model_type),
        created=created_date
    )
