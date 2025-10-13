"""User-facing messages for local path workflow."""

from typing import List, Optional
from telegram import InlineKeyboardButton


class LocalPathMessages:
    """Consolidated message templates for local path workflow."""

    @staticmethod
    def data_source_selection_prompt() -> str:
        return (
            "🤖 **ML Training Workflow**\n\n"
            "How would you like to provide your training data?\n\n"
            "**📤 Upload File**: Upload CSV/Excel through Telegram\n"
            "• Max size: 10MB | Best for: Quick analysis\n\n"
            "**📂 Use Local Path**: Provide filesystem path\n"
            "• No size limits | Best for: Large datasets\n\n"
            "Choose your data source:"
        )

    @staticmethod
    def file_path_input_prompt(allowed_dirs: List[str]) -> str:
        dirs = "\n".join(f"• `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n• ... and {len(allowed_dirs) - 5} more"

        return (
            "✅ **Local File Path**\n\n"
            f"**Allowed directories:**\n{dirs}\n\n"
            "**Formats:** CSV, Excel (.xlsx, .xls), Parquet\n\n"
            "**Examples:**\n"
            "• `/Users/username/datasets/housing.csv`\n"
            "• `~/Documents/data/sales.xlsx`\n\n"
            "Type or paste your file path:"
        )

    @staticmethod
    def loading_data_message() -> str:
        return "🔄 **Loading data...**\n⏳ Validating, loading, and analyzing schema..."

    @staticmethod
    def schema_confirmation_prompt(
        summary: str,
        suggested_target: Optional[str],
        suggested_features: List[str],
        task_type: Optional[str]
    ) -> str:
        msg = f"{summary}\n\n"

        if task_type and suggested_target:
            msg += (
                f"**🎯 Auto-Detected:** {task_type.title()}\n"
                f"**Target:** `{suggested_target}`\n"
            )
            if suggested_features:
                feats = ", ".join(f"`{f}`" for f in suggested_features[:5])
                if len(suggested_features) > 5:
                    feats += f" ... (+{len(suggested_features) - 5})"
                msg += f"**Features:** {feats}\n\n"

        msg += "**Proceed?**\n• ✅ Accept | ❌ Try Different File"
        return msg

    @staticmethod
    def schema_accepted_message(suggested_target: Optional[str]) -> str:
        if suggested_target:
            return (
                f"✅ **Schema Accepted!**\n\n"
                f"🎯 Using: `{suggested_target}`\n\n"
                "Proceeding to target selection..."
            )
        return "✅ **Schema Accepted!**\n\nProceeding to target selection..."

    @staticmethod
    def schema_rejected_message() -> str:
        return "❌ **Schema Rejected**\n\nPlease provide a different file path."

    @staticmethod
    def telegram_upload_prompt() -> str:
        return (
            "✅ **Telegram Upload**\n\n"
            "📤 Please upload your file.\n\n"
            "**Formats:** CSV, Excel, Parquet | **Max:** 10MB"
        )

    @staticmethod
    def load_option_prompt(file_path: str, size_mb: float) -> str:
        """Prompt for load strategy selection (immediate or deferred)."""
        return (
            f"✅ **Path Validated:** `{file_path}`\n"
            f"📊 **Size:** {size_mb:.2f} MB\n\n"
            "**Choose Loading Strategy:**\n\n"
            "**🔄 Load Now** - Load & analyze data immediately\n"
            "• Auto-detect schema | Preview statistics\n"
            "• Best for: Small to medium datasets (<100MB)\n\n"
            "**⏳ Defer Loading** - Provide schema, load later\n"
            "• Skip preview | Load at training time only\n"
            "• Best for: Large datasets (>100MB)\n\n"
            "**Select your strategy:**"
        )

    @staticmethod
    def schema_input_prompt() -> str:
        """Prompt for manual schema input (deferred loading)."""
        return (
            "📝 **Manual Schema Input**\n\n"
            "Provide target and feature columns. **3 formats supported:**\n\n"
            "**Format 1 - Key-Value (recommended):**\n"
            "`target: price`\n"
            "`features: sqft, bedrooms, bathrooms`\n\n"
            "**Format 2 - JSON:**\n"
            '`{"target": "price", "features": ["sqft", "bedrooms", "bathrooms"]}`\n\n'
            "**Format 3 - Simple List:**\n"
            "`price, sqft, bedrooms, bathrooms`\n"
            "(first = target, rest = features)\n\n"
            "**Type your schema:**"
        )

    @staticmethod
    def schema_accepted_deferred(target: str, n_features: int) -> str:
        """Message when manual schema is accepted (deferred loading)."""
        features_word = "feature" if n_features == 1 else "features"
        return (
            f"✅ **Schema Accepted**\n\n"
            f"🎯 Target: `{target}`\n"
            f"📊 Features: {n_features} {features_word}\n\n"
            "⏳ Data will load at training time.\n\n"
            "Proceeding to model selection..."
        )

    @staticmethod
    def schema_parse_error(error_msg: str) -> str:
        """Error message when schema parsing fails."""
        return (
            f"❌ **Schema Parse Error**\n\n"
            f"{error_msg}\n\n"
            "Please try again with correct format."
        )

    @staticmethod
    def format_path_error(
        error_type: str,
        path: str,
        allowed_dirs: List[str] = None,
        size_mb: float = None,
        max_size_mb: int = None,
        allowed_extensions: List[str] = None,
        error_details: str = None
    ) -> str:
        """Format error message based on type."""
        templates = {
            "not_found": (
                f"❌ **File Not Found**\n\n`{path}`\n\n"
                "Check: Path correct? File exists? Readable?"
            ),
            "not_in_whitelist": (
                f"🚫 **Access Denied**\n\n`{path}`\n\n**Allowed:**\n"
                + "\n".join(f"• `{d}`" for d in (allowed_dirs or [])[:5])
                + (f"\n• ... and {len(allowed_dirs) - 5} more" if allowed_dirs and len(allowed_dirs) > 5 else "")
            ),
            "path_traversal": f"🚫 **Security Error**\n\nPath contains suspicious patterns: `{path}`",
            "too_large": (
                f"⚠️ **File Too Large**\n\n"
                f"Size: {size_mb if size_mb is not None else 0:.1f}MB | "
                f"Limit: {max_size_mb if max_size_mb is not None else 0}MB\n\n"
                "Try: Smaller file or sample data"
            ),
            "invalid_extension": (
                f"❌ **Unsupported Format**\n\n`{path}`\n\n"
                f"**Supported:** {', '.join(allowed_extensions or [])}"
            ),
            "empty": f"❌ **Empty File**\n\n`{path}` (0 bytes)",
            "loading_error": f"❌ **Loading Error**\n\n`{path}`\n\n{error_details}",
            "security_validation": error_details or f"❌ **Validation Error**\n\n`{path}`",
            "feature_disabled": "⚠️ **Feature Disabled**\n\nUse Telegram upload instead.",
            "unexpected": (
                f"❌ **Unexpected Error**\n\n"
                f"`{path}`\n\n"
                + (f"{error_details}\n\n" if error_details else "")
                + "Try again or use /train to restart."
            )
        }

        return templates.get(error_type, templates["unexpected"])


# Back Button Utilities (Phase 2: Workflow Back Button)
def create_back_button() -> InlineKeyboardButton:
    """
    Create standardized back button for workflow navigation.

    Returns:
        InlineKeyboardButton with callback_data='workflow_back'
    """
    return InlineKeyboardButton("⬅️ Back", callback_data="workflow_back")


def add_back_button(keyboard: List[List[InlineKeyboardButton]]) -> None:
    """
    Add back button to keyboard layout as last row.

    Args:
        keyboard: Existing keyboard layout (modified in-place)

    Example:
        >>> keyboard = [[button1, button2]]
        >>> add_back_button(keyboard)
        >>> # keyboard is now: [[button1, button2], [back_button]]
    """
    keyboard.append([create_back_button()])
