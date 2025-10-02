"""Response builder for generating user-facing messages."""

from typing import Any, Dict, List, Literal, Optional

from src.core.state_manager import (
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)
from src.utils.exceptions import (
    AgentError,
    PrerequisiteNotMetError,
    InvalidStateTransitionError,
    DataSizeLimitError,
    SessionNotFoundError
)

MESSAGE_TEMPLATES = {
    "ml_training": {
        "awaiting_data": "📊 ML Training Workflow Started\n\nPlease upload your training data as a CSV file.\nThe file should contain:\n• Feature columns (predictors)\n• Target column (what you want to predict)\n\nMaximum file size: 100MB",
        "selecting_target": "🎯 Select Target Variable\n\nWhich column do you want to predict?\nType the exact column name.",
        "selecting_features": "📋 Select Features\n\nWhich columns should be used as features?\nType column names separated by commas.\n\nExample: age, income, education",
        "confirming_model": "🤖 Choose Model Type\n\nSelect the machine learning model:\n1. Neural Network - Deep learning model\n2. Random Forest - Ensemble tree method\n3. Gradient Boosting - Boosted trees\n\nType the number (1-3) or model name.",
        "training": "⚙️ Training Model...\n\nYour model is being trained. This may take a few moments.\nPlease wait...",
        "complete": "✅ Training Complete!"
    },
    "ml_prediction": {
        "awaiting_model": "🔮 ML Prediction Workflow\n\nSelect a trained model for predictions.\nAvailable models:\nUse /models to see your trained models.",
        "awaiting_data": "📊 Upload Prediction Data\n\nUpload a CSV file with the same features as your training data.",
        "predicting": "🔄 Generating predictions...",
        "complete": "✅ Predictions Complete!"
    }
}


class ResponseBuilder:
    """Build user-facing response messages for Telegram bot."""

    def build_workflow_prompt(self, workflow_state: str, session: UserSession) -> str:
        """Build prompt for current workflow state."""
        if session.workflow_type == WorkflowType.ML_TRAINING:
            return MESSAGE_TEMPLATES["ml_training"].get(workflow_state, f"State '{workflow_state}' not implemented.")
        elif session.workflow_type == WorkflowType.ML_PREDICTION:
            return MESSAGE_TEMPLATES["ml_prediction"].get(workflow_state, f"State '{workflow_state}' not implemented.")
        elif session.workflow_type == WorkflowType.STATS_ANALYSIS:
            return "📈 Stats analysis workflow not yet implemented."
        return "Workflow prompt not available for this workflow type."

    def format_column_selection(
        self,
        columns: List[str],
        prompt_type: Literal["target", "features"],
        numbered: bool = True
    ) -> str:
        """Format column selection prompt."""
        header = "🎯 Select Target Column:\n\n" if prompt_type == "target" else "📋 Select Feature Columns:\n\n"
        column_list = "\n".join(f"{i+1}. {col}" for i, col in enumerate(columns)) if numbered else "\n".join(f"• {col}" for col in columns)
        footer = "\n\nType column names separated by commas." if prompt_type == "features" else "\n\nType the column name or number."
        return header + column_list + footer

    def _build_progress_bar(self, current_idx: int, total: int, is_complete: bool) -> str:
        """Build progress bar visualization."""
        if is_complete:
            return f"✅ Complete ({total}/{total})"
        progress_bar = "▓" * current_idx + "░" * (total - current_idx - 1)
        return f"Progress: [{progress_bar}] Step {current_idx + 1}/{total}"

    def build_progress_indicator(self, current_step: str, workflow_type: WorkflowType) -> str:
        """Build progress indicator for workflow."""
        if workflow_type == WorkflowType.ML_TRAINING:
            steps = [s.value for s in MLTrainingState]
        elif workflow_type == WorkflowType.ML_PREDICTION:
            steps = [s.value for s in MLPredictionState]
        else:
            return "Progress tracking not available."

        try:
            current_idx = steps.index(current_step)
            total = len(steps)
            is_complete = current_step in ("complete", MLTrainingState.COMPLETE.value)
            return self._build_progress_bar(current_idx, total, is_complete)
        except ValueError:
            return f"Step: {current_step}"

    def format_prerequisite_error(self, missing_prerequisites: List[str], session: UserSession) -> str:
        """Format prerequisite error message."""
        prereq_messages = {
            "uploaded_data": "• Upload training data (CSV file)",
            "target_selection": "• Select target variable",
            "feature_selection": "• Select feature columns",
            "model_type_selection": "• Choose model type",
            "trained_model": "• Train or select a model"
        }
        missing_items = [prereq_messages.get(prereq, f"• {prereq}") for prereq in missing_prerequisites]
        return "⚠️ Prerequisites Not Met\n\nRequired:\n" + "\n".join(missing_items)

    def format_state_error(
        self,
        error: Exception,
        session: Optional[UserSession] = None
    ) -> str:
        """Format state-related error message."""
        if isinstance(error, SessionNotFoundError):
            return (
                "❌ Session Not Found\n\n"
                "Your session may have expired.\n"
                "Use /start to begin a new session."
            )

        elif isinstance(error, InvalidStateTransitionError):
            return (
                f"❌ Invalid Operation\n\n"
                f"Cannot transition from '{error.current_state}' "
                f"to '{error.requested_state}'.\n\n"
                f"Current workflow: {error.workflow_type}\n"
                f"Use /cancel to reset."
            )

        elif isinstance(error, DataSizeLimitError):
            return (
                f"❌ File Too Large\n\n"
                f"Size: {error.actual_size_mb:.1f}MB\n"
                f"Limit: {error.limit_mb}MB\n\n"
                f"Please upload a smaller file."
            )

        elif isinstance(error, PrerequisiteNotMetError):
            return self.format_prerequisite_error(
                error.missing_prerequisites,
                session
            )

        elif isinstance(error, AgentError):
            return f"❌ Error: {error.message}"

        return f"❌ An error occurred: {str(error)}"

    def format_success_message(self, operation: str, results: Dict[str, Any]) -> str:
        """Format success message for completed operation."""
        if operation == "training_complete":
            accuracy = results.get("accuracy", 0.0)
            model_id = results.get("model_id", "unknown")

            return (
                f"✅ Training Complete!\n\n"
                f"Model ID: {model_id}\n"
                f"Accuracy: {accuracy * 100:.1f}%\n\n"
                f"Use /predict to make predictions with this model."
            )

        elif operation == "prediction_complete":
            count = results.get("count", 0)
            predictions = results.get("predictions", [])

            return (
                f"✅ Predictions Complete!\n\n"
                f"Generated {count} predictions.\n"
                f"Results available for download."
            )

        elif operation == "stats_complete":
            return "✅ Statistical analysis complete!"

        return f"✅ Operation '{operation}' completed successfully."

    def format_model_options(self) -> str:
        """Format model type selection options."""
        return (
            "🤖 Available Models:\n\n"
            "1. Neural Network\n"
            "   • Deep learning model\n"
            "   • Best for: Complex patterns, large datasets\n\n"
            "2. Random Forest\n"
            "   • Ensemble tree method\n"
            "   • Best for: Robust predictions, feature importance\n\n"
            "3. Gradient Boosting\n"
            "   • Boosted decision trees\n"
            "   • Best for: High accuracy, structured data\n\n"
            "Type 1, 2, 3, or the model name."
        )

    def format_workflow_summary(self, session: UserSession) -> str:
        """Format summary of current workflow selections."""
        if not session.workflow_type:
            return "No active workflow."

        summary = f"📋 Workflow Summary\n\n"
        summary += f"Type: {session.workflow_type.value}\n"
        summary += f"State: {session.current_state}\n\n"

        if session.selections:
            summary += "Selections:\n"
            for key, value in session.selections.items():
                if key != "data_metadata":
                    summary += f"• {key}: {value}\n"

        if session.uploaded_data is not None:
            shape = session.uploaded_data.shape
            summary += f"\nData: {shape[0]} rows × {shape[1]} columns"

        return summary

    def format_help_message(self) -> str:
        """Format help message with available commands."""
        return (
            "📚 Available Commands:\n\n"
            "/start - Start new session\n"
            "/train - Train ML model\n"
            "/predict - Make predictions\n"
            "/stats - Statistical analysis\n"
            "/cancel - Cancel current workflow\n"
            "/help - Show this help message\n\n"
            "Upload CSV files to begin analysis."
        )

    def format_welcome_message(self) -> str:
        """Format welcome message for new users."""
        return (
            "👋 Welcome to Statistical Modeling Agent!\n\n"
            "I can help you with:\n"
            "• Machine learning model training\n"
            "• Statistical analysis\n"
            "• Data predictions\n\n"
            "Use /help to see available commands.\n"
            "Upload a CSV file to get started!"
        )

    def format_cancel_confirmation(self, session: UserSession) -> str:
        """Format workflow cancellation confirmation."""
        workflow = session.workflow_type.value if session.workflow_type else "unknown"
        return (
            f"🚫 Workflow Canceled\n\n"
            f"Canceled: {workflow}\n"
            f"All data and selections cleared.\n\n"
            f"Use /train, /predict, or /stats to start again."
        )

    def format_timeout_warning(self, session: UserSession, minutes_left: int) -> str:
        """Format session timeout warning."""
        return (
            f"⏰ Session Timeout Warning\n\n"
            f"Your session will expire in {minutes_left} minutes.\n"
            f"Send any message to keep it active."
        )

    def format_data_upload_confirmation(self, metadata: Dict[str, Any]) -> str:
        """Format data upload confirmation."""
        filename = metadata.get("filename", "unknown")
        rows = metadata.get("rows", 0)
        columns = metadata.get("columns", 0)
        return (
            f"✅ Data Uploaded Successfully\n\n"
            f"File: {filename}\n"
            f"Size: {rows} rows × {columns} columns\n\n"
            f"Ready for analysis!"
        )
