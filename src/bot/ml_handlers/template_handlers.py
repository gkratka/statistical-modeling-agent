"""
Template workflow handlers for ML training.

This module provides handlers for saving and loading ML training templates.
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackContext, ContextTypes

from src.core.state_manager import MLTrainingState, StateManager
from src.core.template_manager import TemplateManager
from src.core.training_template import TemplateConfig
from src.bot.messages import template_messages
from src.bot.messages.training_messages import format_training_metrics
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
from src.utils.i18n_manager import I18nManager
from src.worker.job_queue import JobType, JobStatus

logger = logging.getLogger(__name__)


class TemplateHandlers:
    """Handlers for ML training template operations."""

    def __init__(
        self,
        state_manager: StateManager,
        template_manager: TemplateManager,
        data_loader: DataLoader,
        path_validator: PathValidator,
        training_executor: Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], None]] = None
    ):
        """
        Initialize template handlers.

        Args:
            state_manager: StateManager instance
            template_manager: TemplateManager instance
            data_loader: DataLoader instance
            path_validator: PathValidator instance
            training_executor: Optional callback to auto-execute training after template load
        """
        self.state_manager = state_manager
        self.template_manager = template_manager
        self.data_loader = data_loader
        self.path_validator = path_validator
        self.training_executor = training_executor

    # =========================================================================
    # Save Template Workflow
    # =========================================================================

    async def handle_template_save_request(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle 'Save as Template' button click."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        # Save state snapshot before transition
        session.save_state_snapshot()

        # Transition to SAVING_TEMPLATE state
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.SAVING_TEMPLATE.value
        )

        if not success:
            await query.edit_message_text(f"❌ {error_msg}")
            return

        # Prompt for template name with i18n
        locale = session.language if session.language else None
        keyboard = [[InlineKeyboardButton(I18nManager.t('workflow_state.buttons.cancel', locale=locale), callback_data="cancel_template")]]
        await query.edit_message_text(
            template_messages.TEMPLATE_SAVE_PROMPT,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        logger.info(f"User {user_id} initiated template save")

    async def handle_template_name_input(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template name text input."""
        user_id = update.effective_user.id
        chat_id = update.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session or session.current_state != MLTrainingState.SAVING_TEMPLATE.value:
            return

        template_name = update.message.text.strip()

        # Validate name
        is_valid, error_msg = self.template_manager.validate_template_name(template_name)
        if not is_valid:
            await update.message.reply_text(
                template_messages.TEMPLATE_INVALID_NAME.format(error=error_msg),
                parse_mode="Markdown"
            )
            return

        # Check if exists
        if self.template_manager.template_exists(user_id, template_name):
            await update.message.reply_text(
                template_messages.TEMPLATE_EXISTS.format(name=template_name),
                parse_mode="Markdown"
            )
            return

        # Build template config from session
        template_config = {
            "file_path": session.file_path or "",
            "defer_loading": getattr(session, "load_deferred", False),
            "target_column": session.selections.get("target_column", ""),
            "feature_columns": session.selections.get("feature_columns", []),
            "model_category": session.selections.get("model_category", ""),
            "model_type": session.selections.get("model_type", ""),
            "hyperparameters": session.selections.get("hyperparameters", {})
        }

        # Validate required fields
        if not all([template_config["file_path"], template_config["target_column"],
                   template_config["feature_columns"], template_config["model_type"]]):
            await update.message.reply_text(
                "❌ Cannot save template: Missing required configuration. "
                "Please complete the training setup first."
            )
            return

        # Save template
        success, message = self.template_manager.save_template(
            user_id=user_id,
            template_name=template_name,
            config=template_config
        )

        if success:
            await update.message.reply_text(
                template_messages.TEMPLATE_SAVED_SUCCESS.format(name=template_name),
                parse_mode="Markdown"
            )

            # Offer to continue training or finish with i18n
            locale = session.language if session.language else None
            keyboard = [
                [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.start_training', locale=locale), callback_data="start_training")],
                [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.done_exit', locale=locale), callback_data="complete")]
            ]
            await update.message.reply_text(
                template_messages.TEMPLATE_CONTINUE_TRAINING,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

            logger.info(f"Template '{template_name}' saved for user {user_id}")
        else:
            await update.message.reply_text(
                template_messages.TEMPLATE_SAVE_FAILED.format(error=message),
                parse_mode="Markdown"
            )

    # =========================================================================
    # Load Template Workflow
    # =========================================================================

    async def handle_template_source_selection(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle 'Use Template' data source selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        # Save state snapshot before transition
        session.save_state_snapshot()

        # Transition to LOADING_TEMPLATE state
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.LOADING_TEMPLATE.value
        )

        if not success:
            await query.edit_message_text(f"❌ {error_msg}")
            return

        # Get user's templates
        templates = self.template_manager.list_templates(user_id)

        if not templates:
            await query.edit_message_text(
                template_messages.TEMPLATE_NO_TEMPLATES,
                parse_mode="Markdown"
            )
            return

        # Display templates as buttons with i18n
        locale = session.language if session.language else None
        keyboard = []
        for template in templates:
            button_text = f"📄 {template.template_name}"
            callback_data = f"load_template:{template.template_name}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

        keyboard.append([InlineKeyboardButton(I18nManager.t('workflow_state.buttons.back', locale=locale), callback_data="workflow_back")])

        await query.edit_message_text(
            template_messages.TEMPLATE_LOAD_PROMPT.format(count=len(templates)),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        logger.info(f"User {user_id} browsing {len(templates)} templates")

    async def handle_template_selection(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle specific template selection."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        # Extract template name from callback_data
        template_name = query.data.split(":", 1)[1]

        # Load template
        template = self.template_manager.load_template(user_id, template_name)
        if not template:
            await query.edit_message_text(
                template_messages.TEMPLATE_NOT_FOUND.format(name=template_name),
                parse_mode="Markdown"
            )
            return

        # Update last_used timestamp
        template.last_used = datetime.now(timezone.utc).isoformat()
        config = {
            "file_path": template.file_path,
            "target_column": template.target_column,
            "feature_columns": template.feature_columns,
            "model_category": template.model_category,
            "model_type": template.model_type,
            "hyperparameters": template.hyperparameters,
            "defer_loading": template.defer_loading,
            "last_used": template.last_used,
            "description": template.description
        }
        self.template_manager.save_template(user_id, template_name, config)

        # Populate session with template data
        session.file_path = template.file_path
        session.load_deferred = template.defer_loading
        session.selections["target_column"] = template.target_column
        session.selections["feature_columns"] = template.feature_columns
        session.selections["model_category"] = template.model_category
        session.selections["model_type"] = template.model_type
        session.selections["hyperparameters"] = template.hyperparameters

        # Save state snapshot before transition
        session.save_state_snapshot()

        # Transition to CONFIRMING_TEMPLATE
        success, error_msg, _ = await self.state_manager.transition_state(
            session,
            MLTrainingState.CONFIRMING_TEMPLATE.value
        )

        if not success:
            await query.edit_message_text(f"❌ {error_msg}")
            return

        # Display configuration summary
        summary = template_messages.format_template_summary(
            template_name=template.template_name,
            file_path=template.file_path,
            target=template.target_column,
            features=template.feature_columns,
            model_category=template.model_category,
            model_type=template.model_type,
            created_at=template.created_at
        )

        locale = session.language if session.language else None

        # DEBUG: Log defer_loading value to diagnose issue
        logger.info(f"DEBUG defer_loading: template={template_name}, value={template.defer_loading}, type={type(template.defer_loading)}")
        print(f"🔍 DEBUG defer_loading: template={template_name}, value={template.defer_loading}, type={type(template.defer_loading)}")

        # Respect template's defer_loading setting
        if template.defer_loading:
            # Template was saved with defer_loading=True, show button to load & train
            keyboard = [
                [InlineKeyboardButton(
                    I18nManager.t('workflow_state.buttons.load_and_train', locale=locale, default="🚀 Load Data & Train"),
                    callback_data="template_load_and_train"
                )]
            ]
            await query.edit_message_text(
                summary + "\n\n⏳ *Data loading deferred*\n\nClick button when ready to load data and start training:",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            # Stay in CONFIRMING_TEMPLATE state - don't transition to COMPLETE
            logger.info(f"User {user_id} selected deferred template '{template_name}', showing load & train button")
        else:
            # defer_loading=False: Dispatch training job to worker immediately
            file_path = session.file_path
            validation_result = self.path_validator.validate_path(file_path)

            if not validation_result["is_valid"]:
                await query.edit_message_text(
                    template_messages.TEMPLATE_FILE_PATH_INVALID.format(
                        path=file_path,
                        error=validation_result['error']
                    ),
                    parse_mode="Markdown"
                )
                return

            try:
                # Show starting message
                await query.edit_message_text(
                    summary + "\n\n🚀 *Sending training job to worker...*\n\nThe worker will load and process the data locally.",
                    parse_mode="Markdown"
                )

                # Get job queue from context
                websocket_server = context.bot_data.get('websocket_server')
                job_queue = websocket_server.job_queue if websocket_server else None

                if not job_queue:
                    await query.edit_message_text(
                        "❌ *Worker not connected*\n\nPlease connect a local worker first using `/start`.",
                        parse_mode="Markdown"
                    )
                    return

                # Check if worker is connected for this user
                websocket_server = context.bot_data.get('websocket_server')
                worker_manager = websocket_server.worker_manager if websocket_server else None
                if not worker_manager or not worker_manager.is_user_connected(user_id):
                    await query.edit_message_text(
                        "❌ *No worker connected*\n\nPlease connect a local worker first using `/start`.",
                        parse_mode="Markdown"
                    )
                    return

                # Create job params from template config (worker will load data locally)
                job_params = {
                    'file_path': file_path,
                    'task_type': self._infer_task_type(template.model_type),
                    'model_type': template.model_type,
                    'target_column': template.target_column,
                    'feature_columns': template.feature_columns,
                    'hyperparameters': template.hyperparameters or {},
                    'test_size': 0.2
                }

                logger.info(f"Dispatching template training job to worker: {job_params}")

                # Create job - worker will load data locally
                job_id = await job_queue.create_job(
                    user_id=user_id,
                    job_type=JobType.TRAIN,
                    params=job_params,
                    timeout=600.0  # 10 minutes
                )

                # Save state snapshot before transition
                session.save_state_snapshot()

                # Transition to TRAINING state
                success, error_msg, _ = await self.state_manager.transition_state(
                    session,
                    MLTrainingState.TRAINING.value
                )

                if not success:
                    await query.message.reply_text(f"❌ {error_msg}")
                    return

                logger.info(f"User {user_id} dispatched template '{template_name}' training job {job_id} to worker")

                # Poll for job completion
                import asyncio
                max_wait = 600  # 10 minutes
                poll_interval = 2
                elapsed = 0

                await query.edit_message_text(
                    summary + "\n\n⏳ *Training in progress...*\n\nWorker is loading data and training the model locally.",
                    parse_mode="Markdown"
                )

                while elapsed < max_wait:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                    job = job_queue.get_job(job_id)
                    if not job:
                        await query.edit_message_text(
                            "❌ *Training job not found*\n\nPlease try again.",
                            parse_mode="Markdown"
                        )
                        return

                    if job.status == JobStatus.COMPLETED:
                        # Training succeeded
                        result = job.result or {}
                        model_id = result.get('model_id', 'Unknown')
                        model_type = result.get('model_info', {}).get('model_type', 'Unknown')
                        metrics = result.get('metrics', {})
                        training_time = result.get('training_time', 0)
                        task_type = result.get('model_info', {}).get('task_type', 'classification')

                        # Format metrics with priority ordering
                        metrics_text = format_training_metrics(metrics, task_type)

                        # Format dataset stats
                        dataset_stats = result.get('dataset_stats', {})
                        stats_text = ""
                        if dataset_stats:
                            stats_lines = [f"• Rows: {dataset_stats.get('n_rows', 'N/A')}"]
                            if 'class_distribution' in dataset_stats:
                                dist = dataset_stats['class_distribution']
                                dist_str = ", ".join(f"Class {k}: {v['count']} ({v['pct']}%)" for k, v in sorted(dist.items()))
                                stats_lines.append(f"• Classes: {dist_str}")
                            elif 'quartiles' in dataset_stats:
                                q = dataset_stats['quartiles']
                                stats_lines.append(f"• Target: Q1={q['q1']:.2f}, Median={q['median']:.2f}, Q3={q['q3']:.2f}")
                            stats_text = "📊 *Dataset:*\n" + "\n".join(stats_lines) + "\n\n"

                        await query.edit_message_text(
                            f"✅ *Training Complete!*\n\n"
                            f"🎯 Model: {model_type}\n"
                            f"🆔 Model ID: `{model_id}`\n\n"
                            f"{stats_text}"
                            f"📈 *Performance Metrics:*\n{metrics_text}\n\n"
                            f"⏱ Training Time: {training_time:.2f}s",
                            parse_mode="Markdown"
                        )

                        # Transition to COMPLETE
                        await self.state_manager.transition_state(session, MLTrainingState.COMPLETE.value)
                        return

                    elif job.status == JobStatus.FAILED:
                        await query.edit_message_text(
                            f"❌ *Training Failed*\n\n{job.error or 'Unknown error'}",
                            parse_mode="Markdown"
                        )
                        return

                    elif job.status == JobStatus.TIMEOUT:
                        await query.edit_message_text(
                            "❌ *Training timed out*\n\nPlease try again with a smaller dataset.",
                            parse_mode="Markdown"
                        )
                        return

                # Max wait exceeded
                await query.edit_message_text(
                    "❌ *Training exceeded maximum wait time*\n\nPlease check your worker logs.",
                    parse_mode="Markdown"
                )

            except Exception as e:
                logger.error(f"Error dispatching template training job: {e}", exc_info=True)
                await query.edit_message_text(
                    template_messages.TEMPLATE_LOAD_FAILED.format(error=str(e)),
                    parse_mode="Markdown"
                )

    async def handle_template_load_option(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template data loading option (now vs defer)."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        if query.data == "template_load_now":
            session.load_deferred = False

            # Validate and load file
            file_path = session.file_path
            validation_result = self.path_validator.validate_path(file_path)

            if not validation_result["is_valid"]:
                await query.edit_message_text(
                    template_messages.TEMPLATE_FILE_PATH_INVALID.format(
                        path=file_path,
                        error=validation_result['error']
                    ),
                    parse_mode="Markdown"
                )
                return

            try:
                # Load data (returns tuple: df, metadata, schema)
                df, _, _ = await self.data_loader.load_from_local_path(file_path)

                # Store in bot_data
                data_key = f"user_{user_id}_conv_chat_{chat_id}_data"
                context.bot_data[data_key] = df

                await query.edit_message_text(
                    template_messages.TEMPLATE_DATA_LOADED.format(
                        rows=len(df),
                        columns=len(df.columns)
                    ),
                    parse_mode="Markdown"
                )

                # Offer training action (Bug #10 fix) with i18n
                locale = session.language if session.language else None
                keyboard = [
                    [InlineKeyboardButton(I18nManager.t('workflow_state.buttons.start_training', locale=locale), callback_data="start_training")]
                ]
                await query.message.reply_text(
                    I18nManager.t('workflows.ml_training.what_next', locale=locale),
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )

                # Save state snapshot before transition
                session.save_state_snapshot()

                # Transition to TRAINING
                success, error_msg, _ = await self.state_manager.transition_state(
                    session,
                    MLTrainingState.TRAINING.value
                )

                if not success:
                    await query.message.reply_text(f"❌ {error_msg}")
                    return

                # Start training (will be handled by main handler)
                logger.info(f"User {user_id} loading template data immediately")

            except Exception as e:
                logger.error(f"Error loading template data: {e}", exc_info=True)
                await query.edit_message_text(
                    template_messages.TEMPLATE_LOAD_FAILED.format(error=str(e)),
                    parse_mode="Markdown"
                )

        elif query.data == "template_defer":
            session.load_deferred = True

            await query.edit_message_text(
                template_messages.TEMPLATE_DATA_DEFERRED,
                parse_mode="Markdown"
            )

            # Save state snapshot before transition
            session.save_state_snapshot()

            # Transition to COMPLETE
            success, error_msg, _ = await self.state_manager.transition_state(
                session,
                MLTrainingState.COMPLETE.value
            )

            if not success:
                await query.message.reply_text(f"❌ {error_msg}")

            logger.info(f"User {user_id} deferred template data loading")

    async def handle_template_load_and_train(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle 'Load & Train' button click for deferred templates."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found. Please start a new training session with /train")
            return

        locale = session.language if session.language else None
        file_path = session.file_path

        # Check if worker is connected FIRST (for prod where file is on user's machine)
        websocket_server = context.bot_data.get('websocket_server')
        worker_manager = websocket_server.worker_manager if websocket_server else None
        worker_connected = worker_manager and worker_manager.is_user_connected(user_id)

        if not worker_connected:
            # No worker - validate path locally (dev scenario)
            validation_result = self.path_validator.validate_path(file_path)
            if not validation_result["is_valid"]:
                await query.edit_message_text(
                    template_messages.TEMPLATE_FILE_PATH_INVALID.format(
                        path=file_path,
                        error=validation_result['error']
                    ),
                    parse_mode="Markdown"
                )
                return

        # Worker connected or local validation passed - proceed with training
        try:
            # Show starting message
            await query.edit_message_text(
                "🚀 *Sending training job to worker...*\n\nThe worker will load and process the data locally.",
                parse_mode="Markdown"
            )

            job_queue = websocket_server.job_queue if websocket_server else None
            if not job_queue or not worker_connected:
                await query.edit_message_text(
                    "❌ *Worker not connected*\n\nPlease connect a local worker first using `/start`.",
                    parse_mode="Markdown"
                )
                return

            # Create job params from session config (worker will load data locally)
            job_params = {
                'file_path': file_path,
                'task_type': self._infer_task_type(session.selections.get('model_type', '')),
                'model_type': session.selections.get('model_type'),
                'target_column': session.selections.get('target_column'),
                'feature_columns': session.selections.get('feature_columns', []),
                'hyperparameters': session.selections.get('hyperparameters', {}),
                'test_size': 0.2
            }

            logger.info(f"Dispatching template training job to worker: {job_params}")

            # Create job - worker will load data locally
            job_id = await job_queue.create_job(
                user_id=user_id,
                job_type=JobType.TRAIN,
                params=job_params,
                timeout=600.0  # 10 minutes
            )

            # Transition to TRAINING state
            session.save_state_snapshot()
            success, error_msg, _ = await self.state_manager.transition_state(
                session,
                MLTrainingState.TRAINING.value
            )

            if not success:
                await query.message.reply_text(f"❌ {error_msg}")
                return

            logger.info(f"User {user_id} dispatched template training job {job_id} to worker")

            # Poll for job completion
            import asyncio
            max_wait = 600  # 10 minutes
            poll_interval = 2
            elapsed = 0

            await query.edit_message_text(
                "⏳ *Training in progress...*\n\nWorker is loading data and training the model locally.",
                parse_mode="Markdown"
            )

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                job = job_queue.get_job(job_id)
                if not job:
                    await query.edit_message_text(
                        "❌ *Training job not found*\n\nPlease try again.",
                        parse_mode="Markdown"
                    )
                    return

                if job.status == JobStatus.COMPLETED:
                    # Training succeeded
                    result = job.result or {}
                    model_id = result.get('model_id', 'Unknown')
                    model_type = result.get('model_info', {}).get('model_type', 'Unknown')
                    metrics = result.get('metrics', {})
                    training_time = result.get('training_time', 0)
                    task_type = result.get('model_info', {}).get('task_type', 'classification')

                    # Format metrics with priority ordering
                    metrics_text = format_training_metrics(metrics, task_type)

                    # Format dataset stats
                    dataset_stats = result.get('dataset_stats', {})
                    stats_text = ""
                    if dataset_stats:
                        stats_lines = [f"• Rows: {dataset_stats.get('n_rows', 'N/A')}"]
                        if 'class_distribution' in dataset_stats:
                            dist = dataset_stats['class_distribution']
                            dist_str = ", ".join(f"Class {k}: {v['count']} ({v['pct']}%)" for k, v in sorted(dist.items()))
                            stats_lines.append(f"• Classes: {dist_str}")
                        elif 'quartiles' in dataset_stats:
                            q = dataset_stats['quartiles']
                            stats_lines.append(f"• Target: Q1={q['q1']:.2f}, Median={q['median']:.2f}, Q3={q['q3']:.2f}")
                        stats_text = "📊 *Dataset:*\n" + "\n".join(stats_lines) + "\n\n"

                    await query.edit_message_text(
                        f"✅ *Training Complete!*\n\n"
                        f"🎯 Model: {model_type}\n"
                        f"🆔 Model ID: `{model_id}`\n\n"
                        f"{stats_text}"
                        f"📈 *Performance Metrics:*\n{metrics_text}\n\n"
                        f"⏱ Training Time: {training_time:.2f}s",
                        parse_mode="Markdown"
                    )

                    # Transition to COMPLETE
                    await self.state_manager.transition_state(session, MLTrainingState.COMPLETE.value)
                    return

                elif job.status == JobStatus.FAILED:
                    await query.edit_message_text(
                        f"❌ *Training Failed*\n\n{job.error or 'Unknown error'}",
                        parse_mode="Markdown"
                    )
                    return

                elif job.status == JobStatus.TIMEOUT:
                    await query.edit_message_text(
                        "❌ *Training timed out*\n\nPlease try again with a smaller dataset.",
                        parse_mode="Markdown"
                    )
                    return

            # Max wait exceeded
            await query.edit_message_text(
                "❌ *Training exceeded maximum wait time*\n\nPlease check your worker logs.",
                parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"Error dispatching template training job: {e}", exc_info=True)
            await query.edit_message_text(
                template_messages.TEMPLATE_LOAD_FAILED.format(error=str(e)),
                parse_mode="Markdown"
            )

    # =========================================================================
    # Cancel Template Workflow
    # =========================================================================

    async def handle_cancel_template(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Handle template workflow cancellation."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        if not session:
            await query.edit_message_text("❌ Session not found.")
            return

        # Restore previous state using back navigation
        if session.restore_previous_state():
            await query.edit_message_text("❌ Template operation cancelled.")
            logger.info(f"User {user_id} cancelled template operation")
        else:
            await query.edit_message_text("❌ Cannot cancel: No previous state available.")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _infer_task_type(self, model_type: str) -> str:
        """Infer task type from model type string.

        Args:
            model_type: Model type string (e.g., 'xgboost_binary_classification')

        Returns:
            Task type: 'classification', 'regression', or 'neural_network'
        """
        model_type_lower = model_type.lower()

        if 'classification' in model_type_lower:
            return 'classification'
        elif 'regression' in model_type_lower:
            return 'regression'
        elif 'neural' in model_type_lower or 'keras' in model_type_lower or 'mlp' in model_type_lower:
            return 'neural_network'
        else:
            # Default to classification
            return 'classification'
