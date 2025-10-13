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


def is_keras_model(model_type: str) -> bool:
    """
    Check if model type is a Keras model.

    Args:
        model_type: Model type string

    Returns:
        True if Keras model, False otherwise
    """
    keras_models = [
        'keras_binary_classification',
        'keras_multiclass_classification',
        'keras_regression',
        'neural_network'  # Treat generic neural_network as Keras (will be auto-detected to specific variant)
    ]
    return model_type in keras_models


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
        user_msg = update.message.text[:50] if update.message.text else ""
        self.logger.info(
            f"📨 WorkflowRouter.handle() - user_id={session.user_id}, "
            f"current_state={session.current_state}, "
            f"workflow={session.workflow_type.value if session.workflow_type else None}, "
            f"message='{user_msg}...'"
        )

        # Check for cancel command
        if update.message.text.lower() in ['/cancel', 'cancel']:
            self.logger.info(f"🚫 User {session.user_id} requested cancel")
            return await self.cancel_workflow(update, session)

        # Route based on workflow state
        current_state = session.current_state
        self.logger.debug(f"🔀 Routing to handler for state: {current_state}")

        if current_state == MLTrainingState.SELECTING_TARGET.value:
            return await self.handle_target_selection(update, context, session)
        elif current_state == MLTrainingState.SELECTING_FEATURES.value:
            return await self.handle_feature_selection(update, context, session)
        elif current_state == MLTrainingState.CONFIRMING_MODEL.value:
            return await self.handle_model_confirmation(update, context, session)
        elif current_state == MLTrainingState.SPECIFYING_ARCHITECTURE.value:
            return await self.handle_architecture_specification(update, context, session)
        elif current_state == MLTrainingState.COLLECTING_HYPERPARAMETERS.value:
            return await self.handle_hyperparameter_collection(update, context, session)
        elif current_state == MLTrainingState.TRAINING.value:
            # Training state is typically non-interactive
            self.logger.warning(
                f"⚠️ User {session.user_id} sent message during TRAINING state (non-interactive)"
            )
            await update.message.reply_text(
                "⏳ Training in progress... Please wait.",
                parse_mode="Markdown"
            )
            return
        else:
            # Unknown state - clear and restart
            self.logger.error(
                f"❌ UNKNOWN STATE: user_id={session.user_id}, state='{current_state}' - clearing workflow"
            )
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
        self.logger.info(
            f"🎯 handle_target_selection() - user_id={session.user_id}, input='{user_input}'"
        )

        # Get dataframe from session
        dataframe = await self.state_manager.get_data(session)
        columns = dataframe.columns.tolist()
        self.logger.debug(f"📊 Available columns: {columns}")

        # Parse column selection (number or name)
        try:
            selected_column = parse_column_selection(user_input, columns)
            self.logger.info(f"✅ Parsed target column: '{selected_column}'")
        except ValueError as e:
            # Invalid selection - show error and re-prompt
            self.logger.warning(f"⚠️ Invalid target selection: {str(e)}")
            error_message = (
                f"❌ Invalid selection: {str(e)}\n\n"
                f"Please select a column by number (1-{len(columns)}) or name.\n"
                f"Example: '5' or 'price'"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")
            return

        # Validate column exists (should always pass after parse, but double-check)
        if selected_column not in columns:
            self.logger.error(f"❌ Column validation failed: '{selected_column}' not in {columns}")
            await update.message.reply_text(
                f"❌ Column '{selected_column}' not found in dataset.",
                parse_mode="Markdown"
            )
            return

        # Store target column in session
        session.selections["target_column"] = selected_column
        self.logger.info(f"💾 Stored target_column in session: '{selected_column}'")

        # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button)
        session.save_state_snapshot()
        self.logger.debug("📸 State snapshot saved before transition to SELECTING_FEATURES")

        # Transition to feature selection
        self.logger.info(f"🔄 Attempting transition: SELECTING_TARGET → SELECTING_FEATURES")
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLTrainingState.SELECTING_FEATURES.value
        )

        if not success:
            self.logger.error(
                f"❌ Transition FAILED: {error_msg}, missing prerequisites: {missing}"
            )
            await update.message.reply_text(
                f"⚠️ **Workflow Error**\n\n{error_msg}\n\nPlease use /cancel and start again.",
                parse_mode="Markdown"
            )
            return

        self.logger.info(f"✅ Transition successful: now in SELECTING_FEATURES state")

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
        self.logger.info(
            f"📋 handle_feature_selection() - user_id={session.user_id}, input='{user_input}'"
        )

        # Get dataframe and target column
        dataframe = await self.state_manager.get_data(session)
        target_column = session.selections.get("target_column")
        self.logger.debug(f"🎯 Target column: '{target_column}'")

        # Available features (exclude target)
        available_features = [
            col for col in dataframe.columns.tolist()
            if col != target_column
        ]
        self.logger.debug(f"📊 Available features ({len(available_features)}): {available_features}")

        # Parse feature selection (supports multiple formats)
        try:
            selected_features = parse_feature_selection(
                user_input,
                available_features,
                target_column
            )
            self.logger.info(f"✅ Parsed features ({len(selected_features)}): {selected_features}")
        except ValueError as e:
            self.logger.warning(f"⚠️ Invalid feature selection: {str(e)}")
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
            self.logger.error("❌ No features selected")
            await update.message.reply_text(
                "❌ Please select at least one feature column.",
                parse_mode="Markdown"
            )
            return

        # Store features in session
        session.selections["feature_columns"] = selected_features
        self.logger.info(f"💾 Stored feature_columns in session: {selected_features}")

        # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button)
        session.save_state_snapshot()
        self.logger.debug("📸 State snapshot saved before transition to CONFIRMING_MODEL")

        # Transition to model confirmation
        self.logger.info(f"🔄 Attempting transition: SELECTING_FEATURES → CONFIRMING_MODEL")
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLTrainingState.CONFIRMING_MODEL.value
        )

        if not success:
            self.logger.error(
                f"❌ Transition FAILED: {error_msg}, missing prerequisites: {missing}"
            )
            await update.message.reply_text(
                f"⚠️ **Workflow Error**\n\n{error_msg}\n\nPlease use /cancel and start again.",
                parse_mode="Markdown"
            )
            return

        self.logger.info(f"✅ Transition successful: now in CONFIRMING_MODEL state")

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
        self.logger.info(
            f"🤖 handle_model_confirmation() - user_id={session.user_id}, input='{user_input}'"
        )

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
            '4': 'auto',
            # Keras models
            'keras_binary': 'keras_binary_classification',
            'keras binary': 'keras_binary_classification',
            '5': 'keras_binary_classification',
            'keras_multi': 'keras_multiclass_classification',
            'keras multi': 'keras_multiclass_classification',
            '6': 'keras_multiclass_classification',
            'keras_reg': 'keras_regression',
            'keras regression': 'keras_regression',
            '7': 'keras_regression'
        }

        model_type = model_type_map.get(user_input)

        if not model_type:
            self.logger.warning(f"⚠️ Invalid model type input: '{user_input}'")
            error_message = (
                f"❌ Invalid model type: '{user_input}'\n\n"
                f"Please select:\n"
                f"1. Linear Regression\n"
                f"2. Random Forest\n"
                f"3. Neural Network (sklearn MLP)\n"
                f"4. Auto (best model automatically selected)\n"
                f"5. Keras Binary Classification (NN)\n"
                f"6. Keras Multiclass Classification (NN)\n"
                f"7. Keras Regression (NN)"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")
            return

        # Store model type in session
        session.selections["model_type"] = model_type
        self.logger.info(f"💾 Initial model_type stored: '{model_type}'")

        # Save state snapshot BEFORE transition (Phase 2: Workflow Back Button)
        session.save_state_snapshot()
        self.logger.debug("📸 State snapshot saved before model type transition")

        # Auto-detect Keras variant for generic "neural_network"
        if model_type == "neural_network":
            # Get dataframe from session
            dataframe = await self.state_manager.get_data(session)

            target_col = session.selections.get("target_column")
            target_data = dataframe[target_col]
            n_classes = target_data.nunique()

            self.logger.info(
                f"🔍 Auto-detecting Keras variant: target='{target_col}', "
                f"n_classes={n_classes}, dtype={target_data.dtype}"
            )

            # Auto-detect based on number of classes
            if n_classes == 2:
                model_type = "keras_binary_classification"
            elif n_classes > 10:
                model_type = "keras_regression"
            else:
                model_type = "keras_multiclass_classification"

            # Update session with detected Keras model type
            session.selections["model_type"] = model_type
            self.logger.info(f"✅ Auto-detected Keras model: {model_type} (n_classes={n_classes})")

        # Branch based on model type
        if is_keras_model(model_type):
            # Keras models: go to architecture specification
            self.logger.info(
                f"🔄 Keras model detected - transitioning: CONFIRMING_MODEL → SPECIFYING_ARCHITECTURE"
            )
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.SPECIFYING_ARCHITECTURE.value
            )

            if not success:
                self.logger.error(
                    f"❌ Transition FAILED: {error_msg}, missing prerequisites: {missing}"
                )
                await update.message.reply_text(
                    f"⚠️ **Workflow Error**\n\n{error_msg}\n\nPlease use /cancel and start again.",
                    parse_mode="Markdown"
                )
                return

            self.logger.info(f"✅ Transition successful: now in SPECIFYING_ARCHITECTURE state")

            await update.message.reply_text(
                f"✅ **Keras Model Selected**: {model_type}\n\n"
                f"**Architecture Configuration**\n"
                f"Choose architecture:\n"
                f"1. Default template (recommended for beginners)\n"
                f"2. Custom JSON (advanced)\n\n"
                f"Enter choice:",
                parse_mode="Markdown"
            )
        else:
            # sklearn models: go directly to training
            self.logger.info(
                f"🔄 sklearn model detected - transitioning: CONFIRMING_MODEL → TRAINING"
            )
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.TRAINING.value
            )

            if not success:
                self.logger.error(
                    f"❌ Transition FAILED: {error_msg}, missing prerequisites: {missing}"
                )
                await update.message.reply_text(
                    f"⚠️ **Workflow Error**\n\n{error_msg}\n\nPlease use /cancel and start again.",
                    parse_mode="Markdown"
                )
                return

            self.logger.info(f"✅ Transition successful: now in TRAINING state")

            # Show training started message
            await update.message.reply_text(
                f"🚀 **Training Started**: `{model_type}`\n\nPlease wait while the model is being trained...",
                parse_mode="Markdown"
            )

            # Execute training
            self.logger.info(f"🚀 Executing sklearn training for model_type='{model_type}'")
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

        # Build parameters dict
        parameters = {
            "target_column": target_column,
            "feature_columns": feature_columns,
            "model_type": model_type,
            "task_type": task_type,  # Auto-detected regression/classification
            "user_id": session.user_id  # Pass user_id in parameters
        }

        # Add Keras-specific parameters if present
        if is_keras_model(model_type):
            architecture = session.selections.get("architecture")
            hyperparameters = session.selections.get("hyperparameters", {})

            if architecture:
                # Combine architecture and hyperparameters for Keras
                parameters["hyperparameters"] = {
                    "architecture": architecture,
                    **hyperparameters
                }

        # Create TaskDefinition with complete parameters
        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters=parameters,
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

    async def handle_architecture_specification(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Handle architecture specification for Keras models.

        States:
        - Input: User choice (1=template, 2=custom JSON)
        - Output: Store architecture in session.selections
        - Next: COLLECTING_HYPERPARAMETERS
        """
        user_input = update.message.text.strip()
        self.logger.info(
            f"🏗️ handle_architecture_specification() - user_id={session.user_id}, input='{user_input}'"
        )

        try:
            choice = int(user_input)
            self.logger.info(f"📝 Architecture choice: {choice}")

            if choice == 1:
                # Default template
                model_type = session.selections['model_type']
                n_features = len(session.selections['feature_columns'])

                # Auto-detect n_classes for multiclass
                n_classes = 2
                if model_type == 'keras_multiclass_classification':
                    # Get unique values in target column from session data
                    target_col = session.selections['target_column']
                    data = session.uploaded_data
                    if data is not None:
                        n_classes = data[target_col].nunique()

                # Import template function
                from src.engines.trainers.keras_templates import get_template

                architecture = get_template(
                    model_type=model_type,
                    n_features=n_features,
                    n_classes=n_classes
                )

                session.selections['architecture'] = architecture
                self.logger.info(
                    f"✅ Default architecture template generated: "
                    f"n_features={n_features}, n_classes={n_classes}, layers={len(architecture['layers'])}"
                )

                # Show architecture summary
                await update.message.reply_text(
                    f"✅ <b>Default Architecture Selected</b>\n\n"
                    f"• Input: {n_features} features\n"
                    f"• Hidden: Dense({n_features}, relu)\n"
                    f"• Output: Dense({architecture['layers'][-1]['units']}, "
                    f"{architecture['layers'][-1]['activation']})\n"
                    f"• Loss: {architecture['compile']['loss']}\n"
                    f"• Optimizer: {architecture['compile']['optimizer']}\n\n"
                    f"<b>Training Parameters</b>\n"
                    f"How many epochs? (Recommended: 100-500 for this dataset size)",
                    parse_mode="HTML"
                )

                # Transition to hyperparameter collection
                self.logger.info(
                    f"🔄 Transitioning: SPECIFYING_ARCHITECTURE → COLLECTING_HYPERPARAMETERS"
                )
                session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS.value
                session.selections['hyperparam_step'] = 'epochs'
                await self.state_manager.update_session(session)
                self.logger.info(f"✅ Session updated: now in COLLECTING_HYPERPARAMETERS state")

            elif choice == 2:
                # Custom JSON architecture
                await update.message.reply_text(
                    "<b>Custom Architecture Mode</b>\n\n"
                    "Please send your architecture as JSON.\n\n"
                    "Example format:\n"
                    "<pre>"
                    "{\n"
                    '  "layers": [\n'
                    '    {"type": "Dense", "units": 14, "activation": "relu"},\n'
                    '    {"type": "Dense", "units": 1, "activation": "sigmoid"}\n'
                    '  ],\n'
                    '  "compile": {\n'
                    '    "loss": "binary_crossentropy",\n'
                    '    "optimizer": "adam",\n'
                    '    "metrics": ["accuracy"]\n'
                    '  }\n'
                    "}"
                    "</pre>",
                    parse_mode="HTML"
                )
                session.selections['expecting_json'] = True

            else:
                await update.message.reply_text(
                    "❌ Invalid choice. Please enter <b>1</b> for default template or <b>2</b> for custom JSON.",
                    parse_mode="HTML"
                )

        except ValueError:
            # Check if this is JSON input (for custom architecture)
            if session.selections.get('expecting_json', False):
                try:
                    import json
                    architecture = json.loads(user_input)

                    # Validate architecture structure
                    if 'layers' not in architecture or 'compile' not in architecture:
                        raise ValueError("Architecture must contain 'layers' and 'compile' keys")

                    session.selections['architecture'] = architecture
                    session.selections['expecting_json'] = False

                    await update.message.reply_text(
                        f"✅ <b>Custom Architecture Accepted</b>\n\n"
                        f"• Layers: {len(architecture['layers'])}\n"
                        f"• Loss: {architecture['compile'].get('loss', 'N/A')}\n\n"
                        f"<b>Training Parameters</b>\n"
                        f"How many epochs? (Recommended: 100-500)",
                        parse_mode="HTML"
                    )

                    session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS.value
                    session.selections['hyperparam_step'] = 'epochs'
                    await self.state_manager.update_session(session)

                except json.JSONDecodeError as e:
                    await update.message.reply_text(
                        f"❌ <b>Invalid JSON Format</b>\n\n"
                        f"Error: {e}\n\n"
                        f"Please send valid JSON or type <b>1</b> to use default template.",
                        parse_mode="HTML"
                    )
            else:
                await update.message.reply_text(
                    "Please enter a number (<b>1</b> or <b>2</b>).",
                    parse_mode="HTML"
                )

    async def render_current_state(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Render UI for current workflow state (for back button navigation).

        Called by handle_workflow_back() after state restoration to re-display
        the previous step's UI.

        Args:
            update: Telegram update object (callback query from back button)
            context: Bot context
            session: User session with restored state

        Related: dev/implemented/workflow-back-button.md (Phase 2)
        """
        query = update.callback_query
        current_state = session.current_state

        self.logger.info(
            f"🎨 render_current_state() - user_id={session.user_id}, state={current_state}"
        )

        # Render based on current state
        if current_state == MLTrainingState.SELECTING_TARGET.value:
            # Get dataframe for target selection
            dataframe = await self.state_manager.get_data(session)
            columns = dataframe.columns.tolist()


            # Target selection prompt
            message = (
                f"🎯 **Select Target Column**\n\n"
                f"Your data has {len(columns)} columns:\n"
                + "\n".join(f"{i+1}. {col}" for i, col in enumerate(columns[:20]))
                + (f"\n... and {len(columns) - 20} more" if len(columns) > 20 else "")
                + f"\n\n**Reply with:**\n"
                f"• Column number (e.g., `21`)\n"
                f"• Column name (e.g., `class`)"
            )
            await query.edit_message_text(message, parse_mode="Markdown")

        elif current_state == MLTrainingState.SELECTING_FEATURES.value:
            # Get dataframe for feature selection
            dataframe = await self.state_manager.get_data(session)
            columns = dataframe.columns.tolist()

            # Feature selection prompt
            target_column = session.selections.get("target_column")
            available_features = [col for col in columns if col != target_column]

            message = (
                f"📋 **Select Feature Columns**\n\n"
                f"Target: `{target_column}`\n\n"
                f"Available features ({len(available_features)}):\n"
                + "\n".join(f"{i+1}. {col}" for i, col in enumerate(available_features[:20]))
                + (f"\n... and {len(available_features) - 20} more" if len(available_features) > 20 else "")
                + f"\n\n**Reply with:**\n"
                f"• Numbers: `1,2,3`\n"
                f"• Range: `1-5`\n"
                f"• Names: `age,income`\n"
                f"• All: `all`"
            )
            await query.edit_message_text(message, parse_mode="Markdown")

        elif current_state == MLTrainingState.CONFIRMING_MODEL.value:
            # Detect which workflow (local path vs old) based on session data
            is_local_path_workflow = (
                session.data_source == "local_path" or
                session.load_deferred == True
            )

            if is_local_path_workflow:
                # LOCAL PATH WORKFLOW: Show button-based category UI
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                from src.bot.messages.local_path_messages import add_back_button

                keyboard = [
                    [InlineKeyboardButton("📈 Regression Models", callback_data="model_category:regression")],
                    [InlineKeyboardButton("🎯 Classification Models", callback_data="model_category:classification")],
                    [InlineKeyboardButton("🧠 Neural Networks", callback_data="model_category:neural")]
                ]
                add_back_button(keyboard)
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    "🤖 **Choose Model Type**\n\n"
                    "Select the type of model for your training:\n\n"
                    "📈 **Regression**: Predict continuous values (prices, temperatures, etc.)\n"
                    "🎯 **Classification**: Categorize data (spam/not spam, approve/reject, etc.)\n"
                    "🧠 **Neural Networks**: Advanced deep learning models\n\n"
                    "Which category fits your task?",
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            else:
                # OLD WORKFLOW: Show numbered list (text input)
                target_column = session.selections.get("target_column")
                feature_count = len(session.selections.get("feature_columns", []))

                message = (
                    f"🤖 **Select Model Type**\n\n"
                    f"Target: `{target_column}`\n"
                    f"Features: {feature_count} columns\n\n"
                    f"**Available models:**\n"
                    f"1. Linear Regression\n"
                    f"2. Random Forest\n"
                    f"3. Neural Network (sklearn MLP)\n"
                    f"4. Auto (best model selected)\n"
                    f"5. Keras Binary Classification\n"
                    f"6. Keras Multiclass Classification\n"
                    f"7. Keras Regression\n\n"
                    f"**Reply with number or name:**"
                )
                await query.edit_message_text(message, parse_mode="Markdown")

        elif current_state == MLTrainingState.SPECIFYING_ARCHITECTURE.value:
            # Architecture specification prompt
            message = (
                f"🏗️ **Architecture Configuration**\n\n"
                f"Choose architecture:\n"
                f"1. Default template (recommended)\n"
                f"2. Custom JSON (advanced)\n\n"
                f"Enter choice:"
            )
            await query.edit_message_text(message, parse_mode="Markdown")

        elif current_state == MLTrainingState.COLLECTING_HYPERPARAMETERS.value:
            # Hyperparameter collection prompt
            hyperparam_step = session.selections.get('hyperparam_step', 'epochs')

            if hyperparam_step == 'epochs':
                message = (
                    f"⚙️ **Training Parameters**\n\n"
                    f"How many epochs? (Recommended: 100-500)"
                )
            else:  # batch_size
                epochs = session.selections.get('hyperparameters', {}).get('epochs', 'N/A')
                message = (
                    f"✅ Epochs: **{epochs}**\n\n"
                    f"Batch size? (Recommended: 32-128, default: **32**)"
                )

            await query.edit_message_text(message, parse_mode="Markdown")

        # Local Path Workflow States (Phase 2: Workflow Back Button Fix)
        elif current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value:
            # Data source selection (Upload vs Local Path)
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            from src.bot.messages.local_path_messages import LocalPathMessages

            keyboard = [
                [InlineKeyboardButton("📤 Upload File", callback_data="data_source:telegram")],
                [InlineKeyboardButton("📂 Use Local Path", callback_data="data_source:local_path")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                LocalPathMessages.data_source_selection_prompt(),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif current_state == MLTrainingState.AWAITING_FILE_PATH.value:
            # File path input prompt
            from src.bot.messages.local_path_messages import LocalPathMessages

            # Get data_loader from bot_data (already initialized at startup)
            data_loader = context.bot_data['data_loader']

            await query.edit_message_text(
                LocalPathMessages.file_path_input_prompt(data_loader.allowed_directories),
                parse_mode="Markdown"
            )

        elif current_state == MLTrainingState.CHOOSING_LOAD_OPTION.value:
            # Load option selection (Load Now vs Defer) - THE KEY STATE FOR USER'S BUG
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            from src.bot.messages.local_path_messages import LocalPathMessages, add_back_button
            from src.utils.path_validator import get_file_size_mb
            from pathlib import Path

            # Get file path from session
            file_path = session.file_path
            if not file_path:
                self.logger.error("File path missing in session for CHOOSING_LOAD_OPTION state")
                await query.edit_message_text(
                    "❌ **Error**: File path not found. Please restart with /train",
                    parse_mode="Markdown"
                )
                return

            # Get file size
            size_mb = get_file_size_mb(Path(file_path))

            # Recreate keyboard with back button
            keyboard = [
                [InlineKeyboardButton("🔄 Load Now", callback_data="load_option:immediate")],
                [InlineKeyboardButton("⏳ Defer Loading", callback_data="load_option:defer")]
            ]
            add_back_button(keyboard)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                LocalPathMessages.load_option_prompt(file_path, size_mb),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif current_state == MLTrainingState.CONFIRMING_SCHEMA.value:
            # Schema confirmation after data load
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            from src.bot.messages.local_path_messages import LocalPathMessages, add_back_button

            # Reconstruct schema confirmation UI from detected_schema
            detected_schema = session.detected_schema
            if not detected_schema:
                self.logger.error("detected_schema missing in session for CONFIRMING_SCHEMA state")
                await query.edit_message_text(
                    "❌ **Error**: Schema data not found. Please restart with /train",
                    parse_mode="Markdown"
                )
                return

            # Build summary message
            summary = (
                f"📊 **Dataset Summary**\n\n"
                f"• Rows: {detected_schema.get('n_rows', 'N/A')}\n"
                f"• Columns: {detected_schema.get('n_columns', 'N/A')}\n"
                f"• Quality Score: {detected_schema.get('quality_score', 'N/A'):.2f}"
            )

            suggested_target = detected_schema.get('target')
            suggested_features = detected_schema.get('features', [])
            task_type = detected_schema.get('task_type')

            keyboard = [
                [InlineKeyboardButton("✅ Accept Schema", callback_data="schema:accept")],
                [InlineKeyboardButton("❌ Try Different File", callback_data="schema:reject")]
            ]
            add_back_button(keyboard)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                LocalPathMessages.schema_confirmation_prompt(
                    summary, suggested_target, suggested_features, task_type
                ),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif current_state == MLTrainingState.AWAITING_SCHEMA_INPUT.value:
            # Manual schema input (deferred loading)
            from src.bot.messages.local_path_messages import LocalPathMessages

            await query.edit_message_text(
                LocalPathMessages.schema_input_prompt(),
                parse_mode="Markdown"
            )

        else:
            # Unknown state or non-interactive state
            self.logger.warning(f"⚠️ Cannot render state: {current_state}")
            await query.edit_message_text(
                f"⚠️ **Navigation Error**\n\n"
                f"Current state: {current_state}\n\n"
                f"Use /cancel to exit workflow.",
                parse_mode="Markdown"
            )

    async def handle_hyperparameter_collection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Handle hyperparameter collection for Keras models.

        Multi-turn conversation:
        1. epochs (required)
        2. batch_size (optional, default 32)

        States:
        - Input: User-provided hyperparameter values
        - Output: Store in session.selections['hyperparameters']
        - Next: TRAINING
        """
        user_input = update.message.text.strip()
        current_step = session.selections.get('hyperparam_step', 'epochs')
        self.logger.info(
            f"⚙️ handle_hyperparameter_collection() - user_id={session.user_id}, "
            f"step='{current_step}', input='{user_input}'"
        )

        if 'hyperparameters' not in session.selections:
            session.selections['hyperparameters'] = {
                'verbose': 1,  # Always 1 for Telegram
                'validation_split': 0.0  # Default: no validation split
            }
            self.logger.debug("📝 Initialized hyperparameters dict in session")

        try:
            if current_step == 'epochs':
                epochs = int(user_input)

                # Validate epochs
                if epochs < 1 or epochs > 10000:
                    await update.message.reply_text(
                        "⚠️ **Invalid Epochs**\n\n"
                        "Epochs must be between **1** and **10000**.\n"
                        "Recommended: 100-500 for most datasets.\n\n"
                        "Please enter epochs:",
                        parse_mode="Markdown"
                    )
                    return

                session.selections['hyperparameters']['epochs'] = epochs
                self.logger.info(f"✅ Epochs stored: {epochs}")

                # Move to batch_size
                await update.message.reply_text(
                    f"✅ Epochs: **{epochs}**\n\n"
                    f"Batch size? (Recommended: 32-128, default: **32**)",
                    parse_mode="Markdown"
                )
                session.selections['hyperparam_step'] = 'batch_size'
                self.logger.debug("📝 Advanced to batch_size step")

            elif current_step == 'batch_size':
                batch_size = int(user_input)

                # Validate batch_size
                if batch_size < 1:
                    self.logger.warning(f"⚠️ Invalid batch_size: {batch_size}")
                    await update.message.reply_text(
                        "⚠️ **Invalid Batch Size**\n\n"
                        "Batch size must be at least **1**.\n"
                        "Recommended: 32-128.\n\n"
                        "Please enter batch size:",
                        parse_mode="Markdown"
                    )
                    return

                session.selections['hyperparameters']['batch_size'] = batch_size
                self.logger.info(f"✅ Batch size stored: {batch_size}")

                # Show summary and start training
                architecture = session.selections['architecture']
                hyperparams = session.selections['hyperparameters']

                self.logger.info(
                    f"📊 Training config ready: model={session.selections['model_type']}, "
                    f"epochs={hyperparams['epochs']}, batch_size={hyperparams['batch_size']}, "
                    f"layers={len(architecture['layers'])}"
                )

                await update.message.reply_text(
                    f"✅ **Training Configuration**\n\n"
                    f"• Model: {session.selections['model_type']}\n"
                    f"• Epochs: **{hyperparams['epochs']}**\n"
                    f"• Batch size: **{hyperparams['batch_size']}**\n"
                    f"• Layers: {len(architecture['layers'])}\n\n"
                    f"🚀 Starting training...",
                    parse_mode="Markdown"
                )

                # Transition to training
                self.logger.info(f"🔄 Transitioning: COLLECTING_HYPERPARAMETERS → TRAINING")
                session.current_state = MLTrainingState.TRAINING.value
                await self.state_manager.update_session(session)
                self.logger.info(f"✅ Session updated: now in TRAINING state")

                # Trigger actual training
                self.logger.info(f"🚀 Triggering Keras training execution")
                await self.execute_training(update, context, session)

        except ValueError:
            await update.message.reply_text(
                f"❌ Please enter a valid number for **{current_step}**.",
                parse_mode="Markdown"
            )
