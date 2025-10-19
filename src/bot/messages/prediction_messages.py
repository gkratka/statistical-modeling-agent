"""User-facing messages for ML prediction workflow."""

from typing import List, Dict, Any, Optional, Union
from telegram import InlineKeyboardButton
from src.bot.utils.markdown_escape import escape_markdown_v1


class PredictionMessages:
    """Consolidated message templates for ML prediction workflow."""

    # =============================================================================
    # Step 1: Workflow Start
    # =============================================================================

    @staticmethod
    def prediction_start_message() -> str:
        """Initial message when /predict command is invoked."""
        return (
            "🔮 **ML Prediction Workflow**\n\n"
            "Use trained models to predict target values on new data.\n\n"
            "**Requirements:**\n"
            "• Trained model (from /train)\n"
            "• New dataset with same features\n"
            "• Dataset should NOT include target column\n\n"
            "Let's get started! 🚀"
        )

    # =============================================================================
    # Steps 2-3: Data Loading
    # =============================================================================

    @staticmethod
    def data_source_selection_prompt() -> str:
        """Prompt for data source selection."""
        return (
            "📊 **Load Prediction Data**\n\n"
            "How would you like to provide your data?\n\n"
            "**📤 Upload File**: Upload CSV/Excel through Telegram\n"
            "• Max size: 10MB | Best for: Quick predictions\n\n"
            "**📂 Use Local Path**: Provide filesystem path\n"
            "• No size limits | Best for: Large datasets\n\n"
            "Choose your data source:"
        )

    @staticmethod
    def file_path_input_prompt(allowed_dirs: List[str]) -> str:
        """Prompt for local file path input."""
        dirs = "\n".join(f"• `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n• ... and {len(allowed_dirs) - 5} more"

        return (
            "✅ **Local File Path**\n\n"
            f"**Allowed directories:**\n{dirs}\n\n"
            "**Formats:** CSV, Excel (.xlsx, .xls), Parquet\n\n"
            "**Examples:**\n"
            "• `/Users/username/datasets/new_data.csv`\n"
            "• `~/Documents/predictions/test.xlsx`\n\n"
            "Type or paste your file path:"
        )

    @staticmethod
    def telegram_upload_prompt() -> str:
        """Prompt for Telegram file upload."""
        return (
            "✅ **Telegram Upload**\n\n"
            "📤 Please upload your prediction data.\n\n"
            "**Formats:** CSV, Excel, Parquet | **Max:** 10MB\n\n"
            "**Note:** Data should have same features as training, without target column."
        )

    @staticmethod
    def loading_data_message() -> str:
        """Loading indicator message."""
        return "🔄 **Loading data...**\n⏳ Validating and analyzing schema..."

    @staticmethod
    def loading_deferred_data_message(file_path: str) -> str:
        """Message shown when loading deferred data before prediction execution."""
        escaped_path = escape_markdown_v1(file_path)
        return (
            f"🔄 **Loading Deferred Data**\n\n"
            f"Loading: `{escaped_path}`\n\n"
            f"⏳ Please wait..."
        )

    @staticmethod
    def schema_confirmation_prompt(
        summary: str,
        available_columns: List[str]
    ) -> str:
        """Prompt for schema confirmation showing dataset summary."""
        # Escape column names before wrapping in backticks
        escaped_cols = [escape_markdown_v1(c) for c in available_columns[:10]]
        cols = ", ".join(f"`{c}`" for c in escaped_cols)
        if len(available_columns) > 10:
            cols += f" ... (+{len(available_columns) - 10})"

        msg = (
            f"{summary}\n\n"
            f"**Available Columns ({len(available_columns)}):**\n{cols}\n\n"
            "**Does this look correct?**\n"
            "• ✅ Continue | ❌ Try Different File"
        )
        return msg

    @staticmethod
    def schema_accepted_message() -> str:
        """Message when schema is accepted."""
        return "✅ **Data Loaded!**\n\nProceeding to feature selection..."

    @staticmethod
    def schema_rejected_message() -> str:
        """Message when schema is rejected."""
        return "❌ **Rejected**\n\nLet's try a different file."

    @staticmethod
    def deferred_loading_confirmed_message() -> str:
        """Message shown when user chooses to defer loading."""
        return (
            "⏳ **Data Loading Deferred**\n\n"
            "Your dataset will be loaded just before running predictions.\n\n"
            "**Next Step:** Select feature columns to use for predictions."
        )

    # =============================================================================
    # Steps 4-5: Feature Selection
    # =============================================================================

    @staticmethod
    def feature_selection_prompt(
        available_columns: List[str],
        dataset_shape: tuple
    ) -> str:
        """Prompt for feature selection."""
        # Escape column names before wrapping in backticks
        escaped_cols = [escape_markdown_v1(c) for c in available_columns]
        cols = "\n".join(f"• `{c}`" for c in escaped_cols)
        rows, cols_count = dataset_shape

        return (
            f"📊 **Feature Selection**\n\n"
            f"**Dataset:** {rows:,} rows × {cols_count} columns\n\n"
            f"**Available Columns:**\n{cols}\n\n"
            "**Select features for prediction:**\n"
            "• Enter column names (comma-separated)\n"
            "• ⚠️ Do NOT use `features:` or `target:` prefix\n"
            "• Must match your model's training features\n\n"
            "**Example:** `sqft, bedrooms, bathrooms`"
        )

    @staticmethod
    def feature_selection_prompt_no_preview() -> str:
        """Feature selection prompt when data not yet loaded (defer mode)."""
        return (
            "📝 **Feature Selection** (Deferred Mode)\n\n"
            "Enter the feature column names to use for predictions.\n\n"
            "**Format:** Comma-separated list\n"
            "**Example:** `Attribute1, Attribute2, Attribute3`\n\n"
            "⚠️ **Note:** Column names must exactly match your dataset.\n"
            "⚠️ **Do NOT** use `features:` or `target:` prefix\n\n"
            "**Type your feature columns:**"
        )

    @staticmethod
    def features_selected_message(selected_features: List[str]) -> str:
        """Message when features are selected."""
        # Escape feature names before wrapping in backticks
        escaped_feats = [escape_markdown_v1(f) for f in selected_features[:10]]
        feats = ", ".join(f"`{f}`" for f in escaped_feats)
        if len(selected_features) > 10:
            feats += f" ... (+{len(selected_features) - 10})"

        return (
            f"✅ **Features Selected ({len(selected_features)})**\n\n"
            f"{feats}\n\n"
            "Loading compatible models..."
        )

    @staticmethod
    def feature_validation_error(
        reason: str,
        details: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """Error message when feature validation fails."""
        msg = f"❌ **Feature Error**\n\n{reason}\n\n"

        # Detect if invalid features contain format prefixes
        format_prefix_detected = False
        if details and 'invalid' in details and details['invalid']:
            format_prefixes = ['features:', 'target:']
            for feat in details['invalid']:
                if any(prefix in feat.lower() for prefix in format_prefixes):
                    format_prefix_detected = True
                    break

        if details:
            if 'missing' in details and details['missing']:
                # Escape missing feature names
                escaped_missing = [escape_markdown_v1(f) for f in details['missing']]
                msg += f"**Missing:** {', '.join(f'`{f}`' for f in escaped_missing)}\n"
            if 'invalid' in details and details['invalid']:
                # Escape invalid feature names
                escaped_invalid = [escape_markdown_v1(f) for f in details['invalid']]
                msg += f"**Invalid:** {', '.join(f'`{f}`' for f in escaped_invalid)}\n"

        msg += "\nPlease try again with valid column names."

        # Add helpful tip if format prefix was detected
        if format_prefix_detected:
            msg += (
                "\n\n💡 **Tip:** Don't use `features:` or `target:` prefix "
                "- just list column names separated by commas!"
            )

        return msg

    # =============================================================================
    # Steps 6-7: Model Selection
    # =============================================================================

    @staticmethod
    def model_selection_prompt(
        models: List[Dict[str, Any]],
        selected_features: List[str]
    ) -> str:
        """Prompt for model selection with compatible models."""
        if not models:
            return (
                "⚠️ **No Compatible Models**\n\n"
                f"No trained models match the {len(selected_features)} features you selected.\n\n"
                "**Options:**\n"
                "• Use ⬅️ Back to select different features\n"
                "• Train a new model with /train"
            )

        models_text = ""
        for i, model in enumerate(models[:10], 1):
            # Get model name (custom name takes precedence)
            model_name = model.get('model_name', '').strip()
            model_type = model.get('model_type', 'Unknown')
            task_type = model.get('task_type', 'Unknown')
            target = model.get('target_column', 'Unknown')
            accuracy = model.get('metrics', {}).get('accuracy') or model.get('metrics', {}).get('r2')

            # Get feature count
            feature_columns = model.get('feature_columns', [])
            feature_count = len(feature_columns) if feature_columns else None

            # Use custom name if provided, otherwise use model type
            display_name = model_name if model_name else model_type

            # Escape for markdown
            escaped_display_name = escape_markdown_v1(display_name)
            escaped_target = escape_markdown_v1(target)
            escaped_task_type = escape_markdown_v1(task_type)

            # Build model line with feature count
            models_text += f"{i}. **{escaped_display_name.title()}**"
            if feature_count is not None and feature_count > 0:
                feature_word = "feature" if feature_count == 1 else "features"
                models_text += f" ({feature_count} {feature_word})"
            models_text += f" | {escaped_task_type}\n"

            # Add details line
            models_text += f"   Target: {escaped_target}"
            if accuracy:
                models_text += f" | Accuracy: {accuracy:.2%}"
            models_text += "\n\n"

        if len(models) > 10:
            models_text += f"... and {len(models) - 10} more models\n\n"

        return (
            f"🤖 **Select Model ({len(models)} compatible)**\n\n"
            f"{models_text}"
            "**Select a model to use for predictions:**"
        )

    @staticmethod
    def model_selected_message(
        model_id: str,
        model_type: str,
        target_column: str
    ) -> str:
        """Message when model is selected."""
        # Escape all dynamic content
        escaped_model_type = escape_markdown_v1(model_type)
        escaped_target = escape_markdown_v1(target_column)
        escaped_id = escape_markdown_v1(model_id)

        return (
            f"✅ **Model Selected**\n\n"
            f"**Type:** {escaped_model_type.title()}\n"
            f"**Target:** `{escaped_target}`\n"
            f"**ID:** `{escaped_id}`\n\n"
            "Preparing prediction configuration..."
        )

    @staticmethod
    def no_models_available_error() -> str:
        """Error when no models are available for user."""
        return (
            "⚠️ **No Trained Models**\n\n"
            "You don't have any trained models yet.\n\n"
            "**Get Started:**\n"
            "Use /train to create your first model!"
        )

    @staticmethod
    def model_feature_mismatch_error(
        model_features: List[str],
        selected_features: List[str]
    ) -> str:
        """Error when selected features don't match model requirements."""
        model_set = set(model_features)
        selected_set = set(selected_features)

        missing = sorted(model_set - selected_set)
        extra = sorted(selected_set - model_set)

        msg = (
            "❌ **Feature Mismatch**\n\n"
            f"Model requires {len(model_features)} features, you selected {len(selected_features)}.\n\n"
        )

        if missing:
            msg += f"**Missing:** {', '.join(f'`{f}`' for f in missing)}\n"
        if extra:
            msg += f"**Extra:** {', '.join(f'`{f}`' for f in extra)}\n"

        msg += "\n**Required Features:**\n"
        msg += ", ".join(f"`{f}`" for f in model_features)

        return msg

    # =============================================================================
    # Steps 8-9: Prediction Column Confirmation
    # =============================================================================

    @staticmethod
    def prediction_column_prompt(
        target_column: str,
        existing_columns: List[str]
    ) -> str:
        """Prompt for prediction column name confirmation."""
        # Escape target column name
        escaped_target = escape_markdown_v1(target_column)
        escaped_predicted = escape_markdown_v1(f"{target_column}_predicted")

        return (
            f"📝 **Prediction Column Name**\n\n"
            f"Your model predicts: **{escaped_target}**\n\n"
            f"Default name: `{escaped_predicted}`\n\n"
            "**Options:**\n"
            "• Press ✅ to use default\n"
            "• Type custom name (e.g., `price_forecast`)\n\n"
            "**Note:** Name must not conflict with existing columns."
        )

    @staticmethod
    def column_name_confirmed_message(column_name: str) -> str:
        """Message when column name is confirmed."""
        return (
            f"✅ **Column Name Set**\n\n"
            f"Predictions will be saved as: `{column_name}`\n\n"
            "Ready to run prediction!"
        )

    @staticmethod
    def column_name_conflict_error(
        column_name: str,
        existing_columns: List[str]
    ) -> str:
        """Error when column name conflicts with existing columns."""
        # Escape column names
        escaped_name = escape_markdown_v1(column_name)
        escaped_existing = [escape_markdown_v1(c) for c in existing_columns[:10]]

        return (
            f"❌ **Name Conflict**\n\n"
            f"`{escaped_name}` already exists in your dataset.\n\n"
            "**Existing columns:**\n"
            + ", ".join(f"`{c}`" for c in escaped_existing)
            + (f" ... (+{len(existing_columns) - 10})" if len(existing_columns) > 10 else "")
            + "\n\nPlease provide a different name."
        )

    # =============================================================================
    # Steps 10-11: Ready to Run
    # =============================================================================

    @staticmethod
    def ready_to_run_prompt(
        model_type: str,
        target_column: str,
        prediction_column: str,
        n_rows: Union[int, str],
        n_features: int
    ) -> str:
        """Prompt showing ready state before execution."""
        # Conditional formatting for n_rows (handle defer mode where n_rows="Deferred")
        dataset_info = f"{n_rows:,} rows" if isinstance(n_rows, int) else n_rows

        return (
            f"🚀 **Ready to Predict**\n\n"
            f"**Model:** {model_type.title()}\n"
            f"**Target:** `{target_column}`\n"
            f"**Output Column:** `{prediction_column}`\n\n"
            f"**Dataset:** {dataset_info} × {n_features} features\n\n"
            "**Choose Action:**\n"
            "• ▶️ Run Model - Execute predictions\n"
            "• ⬅️ Go Back - Change model\n\n"
            "Ready when you are!"
        )

    # =============================================================================
    # Steps 12-13: Execution and Results
    # =============================================================================

    @staticmethod
    def running_prediction_message() -> str:
        """Message during prediction execution."""
        return (
            "🔮 **Running Prediction...**\n\n"
            "⏳ Loading model, processing data, generating predictions..."
        )

    @staticmethod
    def prediction_success_message(
        model_type: str,
        n_predictions: int,
        prediction_column: str,
        execution_time: float,
        preview_data: str,
        statistics: Optional[Dict[str, float]] = None
    ) -> str:
        """Success message with prediction results and statistics."""
        msg = (
            f"✅ **Prediction Complete!**\n\n"
            f"**Model:** {model_type.title()}\n"
            f"**Predictions:** {n_predictions:,} rows\n"
            f"**Time:** {execution_time:.2f}s\n\n"
        )

        if statistics:
            msg += f"**📊 Prediction Statistics ({prediction_column}):**\n"
            if 'mean' in statistics:
                msg += f"• Mean: {statistics['mean']:.4f}\n"
            if 'std' in statistics:
                msg += f"• Std Dev: {statistics['std']:.4f}\n"
            if 'min' in statistics:
                msg += f"• Min: {statistics['min']:.4f}\n"
            if 'max' in statistics:
                msg += f"• Max: {statistics['max']:.4f}\n"
            if 'median' in statistics:
                msg += f"• Median: {statistics['median']:.4f}\n"
            msg += "\n"

        msg += (
            f"**Preview (first 10 rows):**\n{preview_data}\n\n"
            "📥 **Download complete results below** ⬇️"
        )

        return msg

    @staticmethod
    def prediction_error_message(error_details: str) -> str:
        """Error message when prediction fails."""
        return (
            "❌ **Prediction Failed**\n\n"
            f"{error_details}\n\n"
            "**Possible Issues:**\n"
            "• Feature mismatch\n"
            "• Invalid data types\n"
            "• Missing values\n\n"
            "Use ⬅️ Back to check your data and features."
        )

    # =============================================================================
    # Workflow Completion
    # =============================================================================

    @staticmethod
    def workflow_complete_message() -> str:
        """Message when workflow is complete."""
        return (
            "🎉 **Workflow Complete!**\n\n"
            "**Next Steps:**\n"
            "• Make more predictions: /predict\n"
            "• Train new models: /train\n"
            "• View models: /models"
        )

    # =============================================================================
    # Error Messages
    # =============================================================================

    @staticmethod
    def unexpected_error(error_msg: str) -> str:
        """Generic unexpected error message."""
        # Escape error message to prevent markdown issues
        escaped_msg = escape_markdown_v1(error_msg)

        return (
            f"❌ **Unexpected Error**\n\n"
            f"{escaped_msg}\n\n"
            "Try /predict to start over."
        )

    @staticmethod
    def file_loading_error(file_path: str, error_details: str) -> str:
        """Error when file loading fails."""
        # Escape file path and error details
        escaped_path = escape_markdown_v1(file_path)
        escaped_details = escape_markdown_v1(error_details)

        return (
            f"❌ **File Loading Error**\n\n"
            f"`{escaped_path}`\n\n"
            f"{escaped_details}\n\n"
            "Check: File format? Readable? Valid data?"
        )

    @staticmethod
    def validation_error(field: str, reason: str) -> str:
        """Generic validation error."""
        return (
            f"❌ **Validation Error**\n\n"
            f"**Field:** {field}\n"
            f"**Reason:** {reason}\n\n"
            "Please try again."
        )

    # =============================================================================
    # NEW: Local File Save Workflow Messages
    # =============================================================================

    @staticmethod
    def output_options_prompt() -> str:
        """Present output method choices after predictions complete."""
        return (
            "💾 **Save Prediction Results**\n\n"
            "Choose how to save your predictions:\n\n"
            "**📂 Save to Local Path**\n"
            "• No file size limits\n"
            "• Direct access to results\n"
            "• Choose output location\n\n"
            "**📥 Download via Telegram**\n"
            "• Max 10MB file size\n"
            "• Download to device\n\n"
            "**✅ Done**\n"
            "• Skip saving and finish\n\n"
            "Choose your option:"
        )

    @staticmethod
    def save_path_input_prompt(allowed_dirs: List[str]) -> str:
        """Prompt user for output directory path."""
        dirs = "\n".join(f"• `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n• ... and {len(allowed_dirs) - 5} more"

        return (
            "📂 **Choose Output Directory**\n\n"
            f"**Allowed directories:**\n{dirs}\n\n"
            "**Examples:**\n"
            "• `/Users/username/results`\n"
            "• `~/Documents/predictions`\n\n"
            "Type or paste the directory path:"
        )

    @staticmethod
    def filename_confirmation_prompt(
        default_name: str,
        directory: str
    ) -> str:
        """Show default filename with rename option."""
        escaped_default = escape_markdown_v1(default_name)
        escaped_dir = escape_markdown_v1(directory)

        return (
            f"📝 **Confirm Filename**\n\n"
            f"**Directory:** `{escaped_dir}`\n"
            f"**Default filename:** `{escaped_default}`\n\n"
            "**Options:**\n"
            "• ✅ Accept - Use default name\n"
            "• ✏️ Custom - Provide your own name\n\n"
            "Choose your option:"
        )

    @staticmethod
    def file_save_success_message(
        full_path: str,
        n_rows: int
    ) -> str:
        """Confirm successful file save."""
        escaped_path = escape_markdown_v1(full_path)

        return (
            f"✅ **File Saved Successfully!**\n\n"
            f"**Location:** `{escaped_path}`\n"
            f"**Rows:** {n_rows:,}\n\n"
            "Your predictions are ready to use!"
        )

    @staticmethod
    def file_save_error_message(
        error_type: str,
        details: str
    ) -> str:
        """Error messages for various save failures."""
        escaped_details = escape_markdown_v1(details)

        return (
            f"❌ **Save Failed: {error_type}**\n\n"
            f"{escaped_details}\n\n"
            "**Options:**\n"
            "• 🔄 Try Again\n"
            "• 📥 Download via Telegram\n"
            "• ❌ Cancel"
        )


# =============================================================================
# Button Creation Utilities
# =============================================================================

def create_data_source_buttons() -> List[List[InlineKeyboardButton]]:
    """Create data source selection buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [
            InlineKeyboardButton("📤 Upload File", callback_data="pred_upload"),
            InlineKeyboardButton("📂 Local Path", callback_data="pred_local_path")
        ],
        [InlineKeyboardButton("📋 Use Template", callback_data="use_pred_template")],
        [create_back_button()]
    ]


def create_load_option_buttons() -> List[List[InlineKeyboardButton]]:
    """Create load option selection buttons for defer loading workflow."""
    from src.bot.messages.local_path_messages import add_back_button
    keyboard = [
        [InlineKeyboardButton("🔄 Load Now", callback_data="pred_load_immediate")],
        [InlineKeyboardButton("⏳ Defer Loading", callback_data="pred_load_defer")]
    ]
    add_back_button(keyboard)  # Add back button support
    return keyboard


def create_schema_confirmation_buttons() -> List[List[InlineKeyboardButton]]:
    """Create schema confirmation buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [
            InlineKeyboardButton("✅ Continue", callback_data="pred_schema_accept"),
            InlineKeyboardButton("❌ Different File", callback_data="pred_schema_reject")
        ],
        [create_back_button()]
    ]


def create_column_confirmation_buttons() -> List[List[InlineKeyboardButton]]:
    """Create prediction column confirmation buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [InlineKeyboardButton("✅ Use Default", callback_data="pred_column_default")],
        [create_back_button()]
    ]


def create_ready_to_run_buttons() -> List[List[InlineKeyboardButton]]:
    """Create ready to run buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [InlineKeyboardButton("▶️ Run Model", callback_data="pred_run")],
        [InlineKeyboardButton("⬅️ Go Back", callback_data="pred_go_back")]
    ]


def create_model_selection_buttons(
    models: List[Dict[str, Any]]
) -> List[List[InlineKeyboardButton]]:
    """Create model selection buttons using indices (up to 10 models)."""
    from src.bot.messages.local_path_messages import create_back_button

    buttons = []
    for i, model in enumerate(models[:10], 0):  # Start at 0 for index-based lookup
        # Get model name (custom name takes precedence over model type)
        model_name = model.get('model_name', '').strip()
        model_type = model.get('model_type', 'Unknown')

        # Use custom name if provided, otherwise use model type
        display_name = model_name if model_name else model_type.title()

        # Get feature count
        feature_columns = model.get('feature_columns', [])
        feature_count = len(feature_columns) if feature_columns else None

        # Build button text
        button_text = f"{i+1}. {display_name}"

        # Add feature count if available
        if feature_count is not None and feature_count > 0:
            # Use singular "feature" for count of 1, plural "features" otherwise
            feature_word = "feature" if feature_count == 1 else "features"
            button_text += f" ({feature_count} {feature_word})"

        button = InlineKeyboardButton(
            button_text,  # Enhanced display text
            callback_data=f"pred_model_{i}"  # Callback uses 0-based index
        )
        buttons.append([button])

    buttons.append([create_back_button()])
    return buttons


def create_path_error_recovery_buttons() -> List[List[InlineKeyboardButton]]:
    """
    Create recovery buttons for path validation errors.

    These buttons help users recover from path validation failures without
    losing their workflow progress.
    """
    return [
        [InlineKeyboardButton("🔄 Try Again", callback_data="pred_retry_path")],
        [InlineKeyboardButton("⬅️ Different Data Source", callback_data="pred_back_to_source")],
        [InlineKeyboardButton("❌ Cancel Workflow", callback_data="pred_cancel")]
    ]


# NEW: Button helpers for local file save workflow
def create_output_option_buttons() -> List[List[InlineKeyboardButton]]:
    """Create output method selection buttons."""
    return [
        [InlineKeyboardButton("📂 Save to Local Path", callback_data="pred_output_local")],
        [InlineKeyboardButton("📥 Download via Telegram", callback_data="pred_output_telegram")],
        [InlineKeyboardButton("✅ Done", callback_data="pred_output_done")]
    ]


def create_filename_confirmation_buttons() -> List[List[InlineKeyboardButton]]:
    """Create filename confirmation buttons."""
    return [
        [InlineKeyboardButton("✅ Accept Default", callback_data="pred_filename_default")],
        [InlineKeyboardButton("✏️ Custom Name", callback_data="pred_filename_custom")]
    ]
