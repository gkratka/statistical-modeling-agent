"""Score workflow handler - Combined train and predict in single prompt."""

import asyncio
import time
from typing import Optional

from telegram import Update, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, UserSession, WorkflowType
from src.core.parsers.score_template_parser import (
    parse_score_template,
    validate_score_config,
    format_config_summary,
    ScoreConfig,
    SUPPORTED_MODELS
)
from src.bot.messages.score_messages import ScoreMessages
from src.utils.logger import get_logger
from src.utils.exceptions import ValidationError


logger = get_logger(__name__)


class ScoreWorkflowHandler:
    """Handle score workflow (combined train + predict)."""

    def __init__(self, state_manager: StateManager):
        """Initialize handler.

        Args:
            state_manager: State manager instance
        """
        self.state_manager = state_manager
        self.messages = ScoreMessages()

    async def handle_score_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /score command initiation.

        Args:
            update: Telegram update
            context: Bot context
        """
        user_id = update.effective_user.id
        conversation_id = f"chat_{update.effective_chat.id}"

        logger.info(f"🎯 /score command initiated by user {user_id}")

        # Get or create session
        session = await self.state_manager.get_or_create_session(
            user_id, conversation_id
        )

        # Check for active workflow
        if session.workflow_type is not None:
            logger.warning(
                f"User {user_id} has active workflow: {session.workflow_type.value}"
            )
            await update.message.reply_text(
                f"⚠️ You have an active {session.workflow_type.value} workflow.\n"
                f"Use /cancel to cancel it before starting /score.",
                parse_mode="Markdown"
            )
            return

        # Check for help request
        if update.message.text.lower() in ['/score help', '/score --help']:
            await update.message.reply_text(
                self.messages.help_message(),
                parse_mode="Markdown"
            )
            return

        # Check for models list request
        if update.message.text.lower() in ['/score models', '/score list']:
            await update.message.reply_text(
                self.messages.supported_models_info(),
                parse_mode="Markdown"
            )
            return

        # Start score workflow
        try:
            await self.state_manager.start_workflow(session, WorkflowType.SCORE_WORKFLOW)

            # Set initial state (will be added to state_manager.py)
            from src.core.state_manager import ScoreWorkflowState
            session.current_state = ScoreWorkflowState.AWAITING_TEMPLATE.value
            await self.state_manager.update_session(session)

            logger.info(f"✅ Score workflow started for user {user_id}")

            # Send template prompt
            await update.message.reply_text(
                self.messages.template_prompt(),
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"Failed to start score workflow: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Failed to start workflow: {str(e)}",
                parse_mode="Markdown"
            )

    async def handle_template_submission(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Handle template text submission.

        Args:
            update: Telegram update
            context: Bot context
            session: User session
        """
        user_input = update.message.text
        logger.info(f"📝 Template submission from user {session.user_id}")

        # Show validating message
        validating_msg = await update.message.reply_text(
            self.messages.validating_template_message(),
            parse_mode="Markdown"
        )

        try:
            # Phase 1: Parse template
            logger.debug("Phase 1: Parsing template")
            config = parse_score_template(user_input)
            logger.info(
                f"✅ Template parsed successfully: model={config.model_type}, "
                f"features={len(config.feature_columns)}"
            )

            # Phase 2: Validate paths
            logger.debug("Phase 2: Validating file paths")
            await self._validate_paths(config, context, session)

            # Phase 3: Validate schemas
            logger.debug("Phase 3: Validating data schemas")
            await self._validate_schemas(config, session)

            # Phase 4: Check for warnings
            warnings = validate_score_config(config)
            if warnings:
                logger.info(f"⚠️ Validation warnings: {len(warnings)}")

            # Store config in session
            session.selections["score_config"] = config.to_dict()
            await self.state_manager.update_session(session)

            logger.info("✅ All validations passed")

            # Update validating message to show success
            await validating_msg.edit_text(
                self.messages.validation_complete_message(),
                parse_mode="Markdown"
            )

            # Transition to confirmation state
            from src.core.state_manager import ScoreWorkflowState
            session.current_state = ScoreWorkflowState.CONFIRMING_EXECUTION.value
            await self.state_manager.update_session(session)

            # Show configuration summary and confirmation
            summary = format_config_summary(config)
            confirmation_text = self.messages.confirmation_prompt(summary, warnings)
            keyboard = self.messages.create_confirmation_keyboard()
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                confirmation_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        except ValidationError as e:
            logger.warning(f"Template validation failed: {e}")
            await validating_msg.delete()

            error_msg = self.messages.format_parse_error(str(e))
            await update.message.reply_text(error_msg, parse_mode="Markdown")

            # Stay in AWAITING_TEMPLATE state for retry

        except Exception as e:
            logger.error(f"Unexpected error during template validation: {e}", exc_info=True)
            await validating_msg.delete()

            error_msg = self.messages.unexpected_error_message(str(e))
            await update.message.reply_text(error_msg, parse_mode="Markdown")

            # Cancel workflow on unexpected errors
            await self.state_manager.cancel_workflow(session)

    async def _validate_paths(
        self,
        config: ScoreConfig,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Validate training and prediction data paths.

        Args:
            config: Score configuration
            context: Bot context
            session: User session

        Raises:
            ValidationError: If path validation fails
        """
        from src.utils.path_validator import PathValidator

        # Get path validator from context
        path_validator = context.bot_data.get('path_validator')
        if not path_validator:
            logger.warning("PathValidator not in bot_data, creating new instance")
            from src.utils.config_loader import load_config
            app_config = load_config()
            path_validator = PathValidator(app_config)

        # Validate training data path
        try:
            train_result = path_validator.validate_path(config.train_data_path)
            if not train_result.is_valid:
                raise ValidationError(
                    self.messages.format_path_error(
                        "TRAIN_DATA",
                        config.train_data_path,
                        train_result.error_type,
                        path_validator.allowed_directories,
                        path_validator.allowed_extensions
                    )
                )
            logger.debug(f"✅ Training data path valid: {config.train_data_path}")

        except Exception as e:
            raise ValidationError(f"TRAIN_DATA path error: {str(e)}")

        # Validate prediction data path
        try:
            predict_result = path_validator.validate_path(config.predict_data_path)
            if not predict_result.is_valid:
                raise ValidationError(
                    self.messages.format_path_error(
                        "PREDICT_DATA",
                        config.predict_data_path,
                        predict_result.error_type,
                        path_validator.allowed_directories,
                        path_validator.allowed_extensions
                    )
                )
            logger.debug(f"✅ Prediction data path valid: {config.predict_data_path}")

        except Exception as e:
            raise ValidationError(f"PREDICT_DATA path error: {str(e)}")

    async def _validate_schemas(
        self,
        config: ScoreConfig,
        session: UserSession
    ) -> None:
        """Validate data schemas match configuration.

        Args:
            config: Score configuration
            session: User session

        Raises:
            ValidationError: If schema validation fails
        """
        from src.processors.data_loader import DataLoader

        # Get data loader
        data_loader = DataLoader()

        # Load and validate training data schema
        try:
            logger.debug(f"Loading training data for schema validation: {config.train_data_path}")
            train_df = await asyncio.to_thread(
                data_loader.load_from_local_path,
                config.train_data_path
            )

            # Store training data shape
            config.train_data_shape = train_df.shape
            logger.info(f"✅ Training data loaded: {train_df.shape}")

            # Validate target column exists
            if config.target_column not in train_df.columns:
                available_cols = train_df.columns.tolist()
                raise ValidationError(
                    self.messages.format_schema_error(
                        "Training Data",
                        f"TARGET column '{config.target_column}' not found",
                        available_cols
                    )
                )

            # Validate feature columns exist
            missing_features = [
                f for f in config.feature_columns
                if f not in train_df.columns
            ]
            if missing_features:
                available_cols = train_df.columns.tolist()
                raise ValidationError(
                    self.messages.format_schema_error(
                        "Training Data",
                        f"FEATURES not found: {', '.join(missing_features)}",
                        available_cols
                    )
                )

            logger.debug(
                f"✅ Training schema valid: target={config.target_column}, "
                f"features={len(config.feature_columns)}"
            )

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Training data loading failed: {str(e)}")

        # Load and validate prediction data schema
        try:
            logger.debug(f"Loading prediction data for schema validation: {config.predict_data_path}")
            predict_df = await asyncio.to_thread(
                data_loader.load_from_local_path,
                config.predict_data_path
            )

            # Store prediction data shape
            config.predict_data_shape = predict_df.shape
            logger.info(f"✅ Prediction data loaded: {predict_df.shape}")

            # Validate feature columns exist (target not required)
            missing_features = [
                f for f in config.feature_columns
                if f not in predict_df.columns
            ]
            if missing_features:
                available_cols = predict_df.columns.tolist()
                raise ValidationError(
                    self.messages.format_schema_error(
                        "Prediction Data",
                        f"FEATURES not found: {', '.join(missing_features)}",
                        available_cols
                    )
                )

            logger.debug(f"✅ Prediction schema valid: features={len(config.feature_columns)}")

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Prediction data loading failed: {str(e)}")

    async def handle_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession,
        confirmed: bool
    ) -> None:
        """Handle user confirmation or cancellation.

        Args:
            update: Telegram update (callback query)
            context: Bot context
            session: User session
            confirmed: True if user confirmed, False if cancelled
        """
        query = update.callback_query
        await query.answer()

        if not confirmed:
            logger.info(f"User {session.user_id} cancelled score workflow")
            await query.edit_message_text(
                self.messages.cancel_message(),
                parse_mode="Markdown"
            )
            await self.state_manager.cancel_workflow(session)
            return

        logger.info(f"User {session.user_id} confirmed execution")

        # Transition to executing state
        from src.core.state_manager import ScoreWorkflowState
        session.current_state = ScoreWorkflowState.TRAINING_MODEL.value
        await self.state_manager.update_session(session)

        # Show execution starting message
        await query.edit_message_text(
            self.messages.execution_starting_message(),
            parse_mode="Markdown"
        )

        # Execute workflow
        await self._execute_score_workflow(update, context, session)

    async def _execute_score_workflow(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """Execute complete score workflow (train + predict).

        Args:
            update: Telegram update
            context: Bot context
            session: User session
        """
        start_time = time.time()

        try:
            # Get configuration from session
            config_dict = session.selections.get("score_config")
            config = ScoreConfig.from_dict(config_dict)

            # Get effective user from update (handles both message and callback_query)
            if update.message:
                chat = update.message.chat
            elif update.callback_query:
                chat = update.callback_query.message.chat
            else:
                raise ValueError("Update has neither message nor callback_query")

            # Phase 1: Load Training Data
            await self._send_phase_update(chat.id, context, 1, "Loading Training Data")
            train_df = await self._load_data(config.train_data_path)
            logger.info(f"✅ Phase 1 complete: loaded {train_df.shape[0]:,} training rows")

            # Phase 2: Train Model
            await self._send_phase_update(chat.id, context, 2, "Training Model")
            model_result = await self._train_model(config, train_df, session)
            model_id = model_result['model_id']
            training_metrics = model_result['metrics']
            config.model_id = model_id
            logger.info(f"✅ Phase 2 complete: model {model_id} trained")

            # Phase 3: Save Model (already done by MLEngine)
            await self._send_phase_update(chat.id, context, 3, "Saving Model")
            logger.info(f"✅ Phase 3 complete: model saved")

            # Phase 4: Load Prediction Data
            await self._send_phase_update(chat.id, context, 4, "Loading Prediction Data")
            predict_df = await self._load_data(config.predict_data_path)
            logger.info(f"✅ Phase 4 complete: loaded {predict_df.shape[0]:,} prediction rows")

            # Phase 5: Generate Predictions
            await self._send_phase_update(chat.id, context, 5, "Generating Predictions")
            prediction_result = await self._generate_predictions(
                config, predict_df, session
            )
            logger.info(f"✅ Phase 5 complete: {prediction_result['n_predictions']:,} predictions generated")

            # Phase 6: Format Results
            await self._send_phase_update(chat.id, context, 6, "Formatting Results")
            total_time = time.time() - start_time

            # Send success message
            success_msg = self.messages.success_message(
                model_id=model_id,
                training_metrics=training_metrics,
                prediction_summary=prediction_result,
                total_time=total_time
            )

            await context.bot.send_message(
                chat_id=chat.id,
                text=success_msg,
                parse_mode="Markdown"
            )

            logger.info(
                f"✅ Score workflow complete for user {session.user_id} "
                f"in {total_time:.1f}s"
            )

            # Transition to complete state
            from src.core.state_manager import ScoreWorkflowState
            session.current_state = ScoreWorkflowState.COMPLETE.value
            await self.state_manager.update_session(session)

            # Clean up workflow
            await self.state_manager.cancel_workflow(session)

        except Exception as e:
            logger.error(f"Score workflow execution failed: {e}", exc_info=True)

            # Determine which phase failed
            current_state = session.current_state
            error_phase = self._get_phase_name(current_state)
            last_completed_phase = self._get_previous_phase(current_state)

            error_msg = self.messages.partial_success_message(
                phase_completed=last_completed_phase,
                error_phase=error_phase,
                error_message=str(e)
            )

            # Get chat from update
            if update.message:
                chat_id = update.message.chat.id
            elif update.callback_query:
                chat_id = update.callback_query.message.chat.id
            else:
                chat_id = session.user_id

            await context.bot.send_message(
                chat_id=chat_id,
                text=error_msg,
                parse_mode="Markdown"
            )

            # Clean up workflow
            await self.state_manager.cancel_workflow(session)

    async def _send_phase_update(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        phase: int,
        phase_name: str
    ) -> None:
        """Send phase update message.

        Args:
            chat_id: Chat ID to send message to
            context: Bot context
            phase: Phase number (1-6)
            phase_name: Phase description
        """
        msg = self.messages.phase_update_message(phase, phase_name)
        await context.bot.send_message(
            chat_id=chat_id,
            text=msg,
            parse_mode="Markdown"
        )

    async def _load_data(self, file_path: str):
        """Load data from file path.

        Args:
            file_path: Path to data file

        Returns:
            Loaded DataFrame
        """
        from src.processors.data_loader import DataLoader

        data_loader = DataLoader()
        return await asyncio.to_thread(
            data_loader.load_from_local_path,
            file_path
        )

    async def _train_model(self, config: ScoreConfig, train_df, session: UserSession) -> dict:
        """Train ML model.

        Args:
            config: Score configuration
            train_df: Training DataFrame
            session: User session

        Returns:
            Dictionary with model_id and metrics
        """
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig

        ml_engine = MLEngine(MLEngineConfig.get_default())

        # Train model using MLEngine
        result = await asyncio.to_thread(
            ml_engine.train_model,
            data=train_df,
            task_type=config.task_type,
            model_type=config.model_type,
            target_column=config.target_column,
            feature_columns=config.feature_columns,
            user_id=session.user_id,
            hyperparameters=config.hyperparameters or {},
            test_size=0.2
        )

        return result

    async def _generate_predictions(
        self,
        config: ScoreConfig,
        predict_df,
        session: UserSession
    ) -> dict:
        """Generate predictions using trained model.

        Args:
            config: Score configuration
            predict_df: Prediction DataFrame
            session: User session

        Returns:
            Dictionary with predictions and summary statistics
        """
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig
        import numpy as np

        ml_engine = MLEngine(MLEngineConfig.get_default())

        # Generate predictions
        predictions = await asyncio.to_thread(
            ml_engine.predict,
            user_id=session.user_id,
            model_id=config.model_id,
            data=predict_df
        )

        # Calculate summary statistics
        pred_array = np.array(predictions)
        summary = {
            'n_predictions': len(predictions),
            'mean_prediction': float(np.mean(pred_array)),
            'std_prediction': float(np.std(pred_array)),
            'min_prediction': float(np.min(pred_array)),
            'max_prediction': float(np.max(pred_array))
        }

        return summary

    def _get_phase_name(self, state: str) -> str:
        """Get human-readable phase name from state.

        Args:
            state: Workflow state

        Returns:
            Phase name
        """
        from src.core.state_manager import ScoreWorkflowState

        phase_map = {
            ScoreWorkflowState.TRAINING_MODEL.value: "Training Model",
            ScoreWorkflowState.RUNNING_PREDICTION.value: "Generating Predictions",
        }

        return phase_map.get(state, "Unknown Phase")

    def _get_previous_phase(self, state: str) -> str:
        """Get previous phase name from current state.

        Args:
            state: Current workflow state

        Returns:
            Previous phase name
        """
        from src.core.state_manager import ScoreWorkflowState

        if state == ScoreWorkflowState.TRAINING_MODEL.value:
            return "Data Validation"
        elif state == ScoreWorkflowState.RUNNING_PREDICTION.value:
            return "Model Training"
        else:
            return "Configuration"
