"""User-facing messages for score workflow (combined train + predict)."""

from typing import List, Optional
from telegram import InlineKeyboardButton

from src.bot.messages.base_messages import BaseMessages


class ScoreMessages(BaseMessages):
    """Consolidated message templates for /score workflow."""

    @staticmethod
    def template_prompt() -> str:
        """Initial prompt for score template submission."""
        return (
            "🎯 **Score Workflow** - Train & Predict in One Step\n\n"
            "Provide a template with all training and prediction details.\n\n"
            "**📋 Template Format:**\n"
            "```\n"
            "TRAIN_DATA: /path/to/training_data.csv\n"
            "TARGET: target_column_name\n"
            "FEATURES: feature1, feature2, feature3\n"
            "MODEL: random_forest\n"
            "PREDICT_DATA: /path/to/prediction_data.csv\n"
            "```\n\n"
            "**Optional Fields:**\n"
            "```\n"
            "OUTPUT_COLUMN: predicted_price\n"
            'HYPERPARAMETERS: {"n_estimators": 100, "max_depth": 10}\n'
            "TASK_TYPE: regression\n"
            "```\n\n"
            "**📌 Example:**\n"
            "```\n"
            "TRAIN_DATA: /Users/me/data/housing_train.csv\n"
            "TARGET: price\n"
            "FEATURES: sqft, bedrooms, bathrooms, age\n"
            "MODEL: gradient_boosting\n"
            "PREDICT_DATA: /Users/me/data/housing_test.csv\n"
            'HYPERPARAMETERS: {"n_estimators": 200}\n'
            "```\n\n"
            "💡 **Tip:** Use absolute paths for local files\n\n"
            "**Paste your template:**"
        )

    @staticmethod
    def supported_models_info() -> str:
        """Information about supported model types."""
        return (
            "**📊 Supported Models:**\n\n"
            "**Regression:**\n"
            "• `linear`, `ridge`, `lasso`, `elasticnet`\n"
            "• `polynomial`\n"
            "• `random_forest_regression`\n"
            "• `gradient_boosting_regression`\n"
            "• `mlp_regression`\n\n"
            "**Classification:**\n"
            "• `logistic`, `decision_tree`\n"
            "• `random_forest`, `gradient_boosting`\n"
            "• `svm`, `naive_bayes`\n"
            "• `mlp_classification`\n"
            "• `keras_binary_classification`"
        )

    @staticmethod
    def validating_template_message() -> str:
        """Message shown while validating template."""
        return (
            "🔄 **Validating Template...**\n\n"
            "⏳ Checking: Syntax | Paths | Schema | Compatibility"
        )

    @staticmethod
    def validation_complete_message() -> str:
        """Message when validation completes successfully."""
        return "✅ **Validation Complete!**"

    @staticmethod
    def confirmation_prompt(
        config_summary: str,
        warnings: Optional[List[str]] = None
    ) -> str:
        """Prompt user to confirm or reject configuration.

        Args:
            config_summary: Formatted configuration summary
            warnings: Optional list of warning messages

        Returns:
            Formatted confirmation prompt
        """
        msg = f"{config_summary}\n\n"

        if warnings:
            msg += "⚠️ **Warnings:**\n"
            for warning in warnings:
                msg += f"• {warning}\n"
            msg += "\n"

        msg += (
            "**Proceed with training and prediction?**\n"
            "• ✅ Confirm & Execute\n"
            "• ❌ Cancel & Restart"
        )

        return msg

    @staticmethod
    def execution_starting_message() -> str:
        """Message when execution starts."""
        return (
            "🚀 **Execution Starting...**\n\n"
            "**Phase 1:** Load Training Data\n"
            "**Phase 2:** Train Model\n"
            "**Phase 3:** Save Model\n"
            "**Phase 4:** Load Prediction Data\n"
            "**Phase 5:** Generate Predictions\n"
            "**Phase 6:** Format Results\n\n"
            "⏳ This may take a few minutes..."
        )

    # Phase configuration for consistent formatting
    PHASE_CONFIG = {
        1: ('📂', 'Loading Training Data'),
        2: ('🎓', 'Training Model'),
        3: ('💾', 'Saving Model'),
        4: ('📂', 'Loading Prediction Data'),
        5: ('🔮', 'Generating Predictions'),
        6: ('📊', 'Formatting Results')
    }

    @staticmethod
    def phase_update_message(phase: int, phase_name: str = None) -> str:
        """Update message for specific phase."""
        if phase_name is None:
            emoji, phase_name = ScoreMessages.PHASE_CONFIG.get(phase, ('⏳', 'Processing'))
        else:
            emoji, _ = ScoreMessages.PHASE_CONFIG.get(phase, ('⏳', ''))
        return f"{emoji} **Phase {phase}/6:** {phase_name}"

    @staticmethod
    def training_progress_message(elapsed_time: float, current_phase: str = "Training Model") -> str:
        """Progress message during training."""
        return f"🎓 **{current_phase}...**\n\n⏱️ Elapsed: {elapsed_time:.1f}s\n⏳ Please wait..."

    @staticmethod
    def prediction_progress_message(n_predictions: int, elapsed_time: float) -> str:
        """Progress message during prediction."""
        return f"🔮 **Generating Predictions...**\n\n📊 Samples: {n_predictions:,}\n⏱️ Elapsed: {elapsed_time:.1f}s"

    @staticmethod
    def success_message(
        model_id: str,
        training_metrics: dict,
        prediction_summary: dict,
        total_time: float
    ) -> str:
        """Success message with results."""
        summary = {"Model": f"`{model_id}`"}
        next_steps = [
            "Use model ID for new predictions: /predict",
            "Train different model: /train",
            "Run another score workflow: /score"
        ]

        return BaseMessages.format_success(
            title="Score Workflow Complete",
            summary=summary,
            metrics={
                **{f"Training {k}": v for k, v in training_metrics.items()},
                **{f"Prediction {k}": v for k, v in prediction_summary.items()},
                "Total Time": f"{total_time:.1f}s"
            },
            next_steps=next_steps
        )

    @staticmethod
    def partial_success_message(
        phase_completed: str,
        error_phase: str,
        error_message: str
    ) -> str:
        """Message when workflow partially succeeds.

        Args:
            phase_completed: Last successful phase
            error_phase: Phase where error occurred
            error_message: Error description

        Returns:
            Formatted partial success message
        """
        return (
            f"⚠️ **Partial Completion**\n\n"
            f"✅ Completed: {phase_completed}\n"
            f"❌ Failed: {error_phase}\n\n"
            f"**Error:** {error_message}\n\n"
            "Use /score to try again with corrected configuration."
        )

    @staticmethod
    def cancel_message() -> str:
        """Message when user cancels workflow."""
        return (
            "❌ **Score Workflow Cancelled**\n\n"
            "Use /score to start a new workflow."
        )

    @staticmethod
    def format_parse_error(error_message: str, line_number: Optional[int] = None) -> str:
        """Format template parsing error message."""
        details = f"**Line {line_number}:** {error_message}" if line_number else error_message
        suggestions = [
            "Check required fields (TRAIN_DATA, TARGET, FEATURES, MODEL, PREDICT_DATA)",
            "Verify key-value format (KEY: value)",
            "Validate HYPERPARAMETERS JSON syntax",
            "Ensure no empty values"
        ]
        return BaseMessages.format_error("Template Parse Error", details, suggestions)

    @staticmethod
    def format_path_error(
        field_name: str,
        path: str,
        error_type: str,
        allowed_dirs: Optional[List[str]] = None,
        allowed_extensions: Optional[List[str]] = None
    ) -> str:
        """Format path validation error message."""
        return BaseMessages.format_path_error(
            field_name, path, error_type,
            allowed_dirs=allowed_dirs,
            allowed_extensions=allowed_extensions
        )

    @staticmethod
    def format_schema_error(
        dataset_name: str,
        error_message: str,
        available_columns: Optional[List[str]] = None
    ) -> str:
        """Format schema validation error message."""
        suggestions = []
        if available_columns:
            suggestions.append(f"**Available Columns:** {BaseMessages.format_list(available_columns)}")
        suggestions.append("Update your template with correct column names")
        return BaseMessages.format_error(
            f"Schema Error ({dataset_name})",
            error_message,
            suggestions
        )

    @staticmethod
    def format_model_error(model_type: str, error_message: str) -> str:
        """Format model-related error message."""
        return BaseMessages.format_error(
            "Model Error",
            f"**Model:** `{model_type}`\n**Issue:** {error_message}",
            ["Check model name spelling and compatibility", "Use /score help to see supported models"]
        )

    @staticmethod
    def help_message() -> str:
        """Comprehensive help message for /score command."""
        return (
            "📖 **Score Workflow Help**\n\n"
            "The `/score` command combines training and prediction into a single step.\n\n"
            "**🎯 When to Use:**\n"
            "• You have both training and test datasets ready\n"
            "• You want to train and immediately get predictions\n"
            "• You prefer power-user workflow over step-by-step\n\n"
            "**📋 Template Format:**\n"
            "```\n"
            "TRAIN_DATA: /path/to/training.csv\n"
            "TARGET: target_column\n"
            "FEATURES: feature1, feature2, feature3\n"
            "MODEL: model_type\n"
            "PREDICT_DATA: /path/to/prediction.csv\n"
            "```\n\n"
            "**Optional:**\n"
            "• `OUTPUT_COLUMN`: Custom prediction column name\n"
            "• `HYPERPARAMETERS`: Model parameters as JSON\n"
            "• `TASK_TYPE`: regression | classification (auto-detected)\n\n"
            "**🔧 Workflow Steps:**\n"
            "1. Parse and validate template\n"
            "2. Load and validate both datasets\n"
            "3. Confirm configuration with you\n"
            "4. Train model on training data\n"
            "5. Generate predictions on test data\n"
            "6. Return results with metrics\n\n"
            "**💡 Tips:**\n"
            "• Use absolute paths for clarity\n"
            "• Ensure feature columns exist in both datasets\n"
            "• Check spelling of model names\n"
            "• Start with simple models before complex ones\n\n"
            "**🆚 Alternatives:**\n"
            "• `/train` - Step-by-step model training only\n"
            "• `/predict` - Use existing model for predictions\n\n"
            "Type `/score` to start!"
        )

    @staticmethod
    def create_confirmation_keyboard() -> List[List[InlineKeyboardButton]]:
        """Create keyboard for confirmation step.

        Returns:
            Keyboard layout with Confirm and Cancel buttons
        """
        return [
            [
                InlineKeyboardButton("✅ Confirm & Execute", callback_data="score_confirm"),
                InlineKeyboardButton("❌ Cancel", callback_data="score_cancel")
            ]
        ]

    @staticmethod
    def execution_cancelled_by_timeout_message(timeout_seconds: int) -> str:
        """Message when execution times out.

        Args:
            timeout_seconds: Timeout threshold

        Returns:
            Formatted timeout message
        """
        return (
            f"⏱️ **Execution Timeout**\n\n"
            f"Execution exceeded {timeout_seconds}s limit.\n\n"
            "**Possible Causes:**\n"
            "• Large dataset size\n"
            "• Complex model requiring extensive training\n"
            "• Insufficient system resources\n\n"
            "**Recommendations:**\n"
            "• Use smaller dataset sample\n"
            "• Try simpler model (e.g., linear, logistic)\n"
            "• Use `/train` for training only, then `/predict` separately"
        )

    @staticmethod
    def unexpected_error_message(error_details: str) -> str:
        """Generic error message for unexpected failures.

        Args:
            error_details: Technical error details

        Returns:
            Formatted error message
        """
        return (
            "❌ **Unexpected Error**\n\n"
            f"An unexpected error occurred:\n\n"
            f"```\n{error_details}\n```\n\n"
            "Please try again. If the issue persists, contact support."
        )
