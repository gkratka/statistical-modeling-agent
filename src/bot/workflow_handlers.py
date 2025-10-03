"""Workflow state handlers for ML training and prediction workflows."""

from typing import List, Optional
from telegram import Update
from telegram.ext import ContextTypes

from src.core.state_manager import (
    StateManager,
    UserSession,
    MLTrainingState,
    WorkflowType
)
from src.core.parser import TaskDefinition
from src.utils.logger import get_logger


def parse_column_selection(user_input: str, columns: List[str]) -> str:
    """
    Parse column selection from user input.

    Supports:
    - Number: "5" → columns[4]
    - Name: "price" → "price"

    Args:
        user_input: User's input text
        columns: List of available column names

    Returns:
        Selected column name

    Raises:
        ValueError: If selection is invalid
    """
    user_input = user_input.strip()

    # Try parsing as number first
    if user_input.isdigit():
        index = int(user_input) - 1  # 1-based to 0-based

        if index < 0 or index >= len(columns):
            raise ValueError(
                f"Number out of range. Please select 1-{len(columns)}."
            )

        return columns[index]

    # Try matching by name (case-insensitive)
    normalized_input = user_input.lower()

    for col in columns:
        if col.lower() == normalized_input:
            return col

    # No match found
    raise ValueError(
        f"Column '{user_input}' not found. "
        f"Available columns: {', '.join(columns)}"
    )


def parse_feature_selection(
    user_input: str,
    available_features: List[str],
    target_column: str
) -> List[str]:
    """
    Parse feature column selection from user input.

    Supports:
    - Numbers: "1,2,3"
    - Names: "age,income,sqft"
    - Range: "1-5"
    - All: "all"

    Args:
        user_input: User's input text
        available_features: List of available feature column names (excluding target)
        target_column: Target column name (for reference in error messages)

    Returns:
        List of selected feature column names

    Raises:
        ValueError: If selection is invalid
    """
    user_input = user_input.strip().lower()

    # Handle "all" keyword
    if user_input == 'all':
        return available_features.copy()

    selected_features = []

    # Split by comma
    parts = [p.strip() for p in user_input.split(',')]

    for part in parts:
        # Check for range (e.g., "1-5")
        if '-' in part and part.replace('-', '').isdigit():
            range_parts = part.split('-')
            if len(range_parts) == 2:
                start = int(range_parts[0]) - 1  # 1-based to 0-based
                end = int(range_parts[1])  # end is inclusive in 1-based

                if start < 0 or end > len(available_features):
                    raise ValueError(
                        f"Range {part} out of bounds. "
                        f"Valid range: 1-{len(available_features)}"
                    )

                selected_features.extend(available_features[start:end])
                continue

        # Try parsing as number
        if part.isdigit():
            index = int(part) - 1

            if index < 0 or index >= len(available_features):
                raise ValueError(
                    f"Number {part} out of range. "
                    f"Please select 1-{len(available_features)}."
                )

            selected_features.append(available_features[index])
            continue

        # Try matching by name
        matched = False
        for feature in available_features:
            if feature.lower() == part:
                selected_features.append(feature)
                matched = True
                break

        if not matched:
            raise ValueError(
                f"Feature '{part}' not found or is the target column. "
                f"Available features: {', '.join(available_features)}"
            )

    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for feature in selected_features:
        if feature not in seen:
            seen.add(feature)
            unique_features.append(feature)

    return unique_features


class WorkflowRouter:
    """Route messages to appropriate workflow state handlers."""

    def __init__(self, state_manager: StateManager):
        """Initialize workflow router."""
        self.state_manager = state_manager
        self.logger = get_logger(__name__)

    async def handle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Route message to appropriate workflow state handler."""
        # Check for cancel command
        if update.message.text.lower() in ['/cancel', 'cancel']:
            return await self.cancel_workflow(update, session)

        # Route based on workflow state
        current_state = session.current_state

        if current_state == MLTrainingState.SELECTING_TARGET.value:
            return await self.handle_target_selection(update, context, session)
        elif current_state == MLTrainingState.SELECTING_FEATURES.value:
            return await self.handle_feature_selection(update, context, session)
        elif current_state == MLTrainingState.CONFIRMING_MODEL.value:
            return await self.handle_model_confirmation(update, context, session)
        elif current_state == MLTrainingState.TRAINING.value:
            # Training state is typically non-interactive
            await update.message.reply_text(
                "⏳ Training in progress... Please wait.",
                parse_mode="Markdown"
            )
            return
        else:
            # Unknown state - clear and restart
            await self.state_manager.cancel_workflow(session)
            await update.message.reply_text(
                "⚠️ Workflow state error. Please start again.",
                parse_mode="Markdown"
            )

    async def handle_target_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Handle target column selection."""
        user_input = update.message.text.strip()

        # Get dataframe from session
        dataframe = await self.state_manager.get_data(session)
        columns = dataframe.columns.tolist()

        # Parse column selection (number or name)
        try:
            selected_column = parse_column_selection(user_input, columns)
        except ValueError as e:
            # Invalid selection - show error and re-prompt
            error_message = (
                f"❌ Invalid selection: {str(e)}\n\n"
                f"Please select a column by number (1-{len(columns)}) or name.\n"
                f"Example: '5' or 'price'"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")
            return

        # Validate column exists (should always pass after parse, but double-check)
        if selected_column not in columns:
            await update.message.reply_text(
                f"❌ Column '{selected_column}' not found in dataset.",
                parse_mode="Markdown"
            )
            return

        # Store target column in session
        session.selections["target_column"] = selected_column

        # Transition to feature selection
        await self.state_manager.transition_state(
            session,
            MLTrainingState.SELECTING_FEATURES.value
        )

        # Prepare feature selection prompt (exclude target column)
        available_features = [col for col in columns if col != selected_column]

        from src.bot.response_builder import ResponseBuilder
        response_builder = ResponseBuilder()

        feature_prompt = response_builder.format_feature_selection(
            available_features,
            selected_target=selected_column
        )

        await update.message.reply_text(feature_prompt, parse_mode="Markdown")

        self.logger.info(
            f"User {session.user_id} selected target: {selected_column}, "
            f"transitioned to SELECTING_FEATURES"
        )

    async def handle_feature_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Handle feature column selection."""
        user_input = update.message.text.strip()

        # Get dataframe and target column
        dataframe = await self.state_manager.get_data(session)
        target_column = session.selections.get("target_column")

        # Available features (exclude target)
        available_features = [
            col for col in dataframe.columns.tolist()
            if col != target_column
        ]

        # Parse feature selection (supports multiple formats)
        try:
            selected_features = parse_feature_selection(
                user_input,
                available_features,
                target_column
            )
        except ValueError as e:
            error_message = (
                f"❌ Invalid selection: {str(e)}\n\n"
                f"Examples:\n"
                f"• '1,2,3' - Select columns 1, 2, and 3\n"
                f"• 'age,income,sqft' - Select by name\n"
                f"• '1-5' - Select range of columns\n"
                f"• 'all' - Select all available features"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")
            return

        # Validate at least one feature selected
        if not selected_features:
            await update.message.reply_text(
                "❌ Please select at least one feature column.",
                parse_mode="Markdown"
            )
            return

        # Store features in session
        session.selections["feature_columns"] = selected_features

        # Transition to model confirmation
        await self.state_manager.transition_state(
            session,
            MLTrainingState.CONFIRMING_MODEL.value
        )

        # Show model type selection prompt
        from src.bot.response_builder import ResponseBuilder
        response_builder = ResponseBuilder()

        model_prompt = response_builder.format_model_type_selection(
            target_column=target_column,
            feature_count=len(selected_features)
        )

        await update.message.reply_text(model_prompt, parse_mode="Markdown")

        self.logger.info(
            f"User {session.user_id} selected features: {selected_features}, "
            f"transitioned to CONFIRMING_MODEL"
        )

    async def handle_model_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Handle model type confirmation."""
        user_input = update.message.text.strip().lower()

        # Map user input to model types (must match trainer SUPPORTED_MODELS)
        model_type_map = {
            'linear': 'linear',
            'linear regression': 'linear',
            '1': 'linear',
            'random': 'random_forest',
            'random forest': 'random_forest',
            'forest': 'random_forest',
            '2': 'random_forest',
            'neural': 'neural_network',
            'neural network': 'neural_network',
            'nn': 'neural_network',
            '3': 'neural_network',
            'auto': 'auto',
            'automatic': 'auto',
            '4': 'auto'
        }

        model_type = model_type_map.get(user_input)

        if not model_type:
            error_message = (
                f"❌ Invalid model type: '{user_input}'\n\n"
                f"Please select:\n"
                f"1. Linear Regression\n"
                f"2. Random Forest\n"
                f"3. Neural Network\n"
                f"4. Auto (best model automatically selected)"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")
            return

        # Store model type in session
        session.selections["model_type"] = model_type

        # Transition to training state
        await self.state_manager.transition_state(
            session,
            MLTrainingState.TRAINING.value
        )

        # Show training started message
        await update.message.reply_text(
            "🚀 **Training Started**\n\nPlease wait while the model is being trained...",
            parse_mode="Markdown"
        )

        # Execute training
        await self.execute_training(update, context, session)

    async def execute_training(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Execute ML model training."""
        # Get all workflow data
        dataframe = await self.state_manager.get_data(session)
        target_column = session.selections.get("target_column")
        feature_columns = session.selections.get("feature_columns")
        model_type = session.selections.get("model_type")

        # Auto-detect task_type from target column data type
        target_data = dataframe[target_column]
        if target_data.dtype in ['int64', 'float64'] and target_data.nunique() > 10:
            task_type = "regression"  # Continuous numeric target
        else:
            task_type = "classification"  # Categorical or discrete target

        self.logger.info(
            f"Auto-detected task_type='{task_type}' for target '{target_column}' "
            f"(dtype={target_data.dtype}, unique_values={target_data.nunique()})"
        )

        # Create TaskDefinition with complete parameters
        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={
                "target_column": target_column,
                "feature_columns": feature_columns,
                "model_type": model_type,
                "task_type": task_type,  # Auto-detected regression/classification
                "user_id": session.user_id  # Pass user_id in parameters
            },
            data_source=None,  # Data already in session
            user_id=session.user_id,
            conversation_id=session.conversation_id
        )

        # Execute through orchestrator
        from src.core.orchestrator import TaskOrchestrator
        orchestrator = TaskOrchestrator()

        try:
            result = await orchestrator.execute_task(task, dataframe)

            # Format results using ResultProcessor
            from src.processors.result_processor import ResultProcessor
            from src.processors.dataclasses import ProcessorConfig

            result_processor = ResultProcessor(ProcessorConfig(
                use_emojis=True,
                enable_visualizations=False,  # Telegram bot doesn't support image uploads yet
                detail_level="detailed"
            ))

            processed = result_processor.process_result(result, "ml_training")
            formatted_result = processed.text

            # Send without Markdown to avoid parsing errors
            await update.message.reply_text(formatted_result)

            # Transition to COMPLETE
            await self.state_manager.transition_state(
                session,
                MLTrainingState.COMPLETE.value
            )

            # Clear workflow after successful completion
            await self.state_manager.cancel_workflow(session)

            self.logger.info(
                f"User {session.user_id} completed ML training workflow successfully"
            )

        except Exception as e:
            self.logger.error(f"Training execution failed: {str(e)}", exc_info=True)

            # Safe error message without Markdown to avoid parsing errors
            error_message = (
                "Training Failed\n\n"
                f"Error: {str(e)}\n\n"
                "The workflow has been cancelled. Please try /start to begin again."
            )

            await update.message.reply_text(error_message)  # NO parse_mode - plain text only

            # Clear workflow on error
            await self.state_manager.cancel_workflow(session)

    async def cancel_workflow(
        self,
        update: Update,
        session: UserSession
    ) -> None:
        """Cancel active workflow."""
        workflow_type = session.workflow_type
        current_state = session.current_state

        # Clear workflow
        await self.state_manager.cancel_workflow(session)

        # Send confirmation
        await update.message.reply_text(
            f"❌ **Workflow Cancelled**\n\n"
            f"Your {workflow_type.value if workflow_type else 'workflow'} has been cancelled.\n"
            f"Send a new request to start over.",
            parse_mode="Markdown"
        )

        self.logger.info(
            f"User {session.user_id} cancelled workflow at state {current_state}"
        )
