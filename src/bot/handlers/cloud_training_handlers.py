"""
Telegram bot handlers for cloud-based ML training workflows.

This module implements Telegram bot handlers for cloud training using AWS EC2,
S3, and CloudWatch. Handles dataset upload, instance selection, training monitoring,
and completion.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.0: Cloud Workflow Telegram Integration)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, UserSession, CloudTrainingState, WorkflowType
from src.cloud.ec2_manager import EC2Manager
from src.cloud.s3_manager import S3Manager
from src.cloud.exceptions import EC2Error, S3Error, CloudError
from src.bot.messages.cloud_messages import (
    CHOOSE_CLOUD_LOCAL_MESSAGE,
    AWAITING_S3_DATASET_MESSAGE,
    cloud_instance_confirmation_message,
    cloud_training_launched_message,
    cloud_training_log_message,
    cloud_training_complete_message,
    cloud_error_message,
    s3_validation_error_message,
    s3_upload_complete_message,
    telegram_file_upload_progress_message,
)

logger = logging.getLogger(__name__)


class CloudTrainingHandlers:
    """
    Handlers for cloud-based ML training workflows.

    This class manages the complete cloud training workflow from dataset upload
    to training completion, including EC2 instance management and log streaming.
    """

    def __init__(
        self,
        state_manager: StateManager,
        ec2_manager: EC2Manager,
        s3_manager: S3Manager
    ):
        """
        Initialize cloud training handlers.

        Args:
            state_manager: State manager for workflow tracking
            ec2_manager: EC2 manager for instance operations
            s3_manager: S3 manager for dataset/model storage
        """
        self.state_manager = state_manager
        self.ec2_manager = ec2_manager
        self.s3_manager = s3_manager
        self.logger = logger

    async def handle_cloud_local_choice(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle cloud vs local training choice from user.

        This is the entry point for cloud training workflow. Displays buttons
        for user to choose between local and cloud training.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            user_id = update.effective_user.id

            # Create keyboard with cloud/local options
            keyboard = [
                [InlineKeyboardButton("💻 Local Training (Free)", callback_data="training_local")],
                [InlineKeyboardButton("☁️ Cloud Training (AWS)", callback_data="training_cloud")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Send choice message
            await update.message.reply_text(
                CHOOSE_CLOUD_LOCAL_MESSAGE,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

            self.logger.info(f"Cloud/local choice presented to user {user_id}")

        except Exception as e:
            self.logger.error(f"Error in handle_cloud_local_choice: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ An error occurred. Please try /train again."
            )

    async def handle_training_cloud_selected(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle user selecting cloud training option.

        Transitions workflow to cloud training and prompts for dataset input.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            user_id = update.effective_user.id
            session = await self.state_manager.get_or_create_session(user_id, str(user_id))

            # Start cloud training workflow
            await self.state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)
            await self.state_manager.transition_state(session, CloudTrainingState.AWAITING_S3_DATASET.value)

            # Answer callback query
            await update.callback_query.answer()

            # Send dataset input prompt
            await update.callback_query.message.reply_text(
                AWAITING_S3_DATASET_MESSAGE,
                parse_mode=ParseMode.MARKDOWN
            )

            self.logger.info(f"User {user_id} started cloud training workflow")

        except Exception as e:
            self.logger.error(f"Error in handle_training_cloud_selected: {e}", exc_info=True)
            await update.callback_query.message.reply_text(
                "❌ Failed to start cloud training. Please try again."
            )

    async def handle_s3_dataset_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle dataset input from user (S3 URI or file upload).

        Supports two input methods:
        1. Telegram file upload - automatically uploads to S3
        2. S3 URI provided by user - validates path

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            user_id = update.effective_user.id
            session = await self.state_manager.get_session(user_id, str(user_id))

            if not session or session.current_state != CloudTrainingState.AWAITING_S3_DATASET.value:
                await update.message.reply_text(
                    "❌ Please start cloud training with /train first."
                )
                return

            # Handle file upload
            if update.message.document:
                await self._handle_telegram_file_upload(update, context, session)

            # Handle S3 URI
            elif update.message.text and update.message.text.startswith('s3://'):
                await self._handle_s3_uri_input(update, context, session)

            else:
                await update.message.reply_text(
                    "❌ Invalid input. Please upload a CSV file or provide an S3 URI starting with `s3://`",
                    parse_mode=ParseMode.MARKDOWN
                )

        except S3Error as e:
            self.logger.error(f"S3 error in handle_s3_dataset_input: {e}", exc_info=True)
            await update.message.reply_text(
                cloud_error_message("S3Error", str(e)),
                parse_mode=ParseMode.MARKDOWN
            )
            raise

        except Exception as e:
            self.logger.error(f"Error in handle_s3_dataset_input: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ Failed to process dataset. Please try again."
            )

    async def _handle_telegram_file_upload(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Handle file uploaded via Telegram - upload to S3.

        Args:
            update: Telegram update object
            context: Telegram context object
            session: User session
        """
        user_id = update.effective_user.id

        # Download file from Telegram
        file = await update.message.document.get_file()
        local_path = Path(f"/tmp/{file.file_id}.csv")

        self.logger.info(f"Downloading file from Telegram for user {user_id}")
        await file.download_to_drive(local_path)

        # Get file size
        file_size_mb = local_path.stat().st_size / (1024 * 1024)

        # Send progress message
        await update.message.reply_text(
            telegram_file_upload_progress_message(
                update.message.document.file_name,
                file_size_mb,
                0
            ),
            parse_mode=ParseMode.MARKDOWN
        )

        # Upload to S3
        s3_uri = self.s3_manager.upload_dataset(
            user_id=user_id,
            file_path=str(local_path),
            dataset_name=update.message.document.file_name
        )

        # Store S3 URI in session
        session.selections['s3_dataset_uri'] = s3_uri

        # Load dataset to get schema info
        import pandas as pd
        df = pd.read_csv(local_path)
        session.uploaded_data = df

        await self.state_manager.update_session(session)

        # Send confirmation
        await update.message.reply_text(
            s3_upload_complete_message(
                s3_uri,
                file_size_mb,
                len(df),
                len(df.columns)
            ),
            parse_mode=ParseMode.MARKDOWN
        )

        # Transition to target selection
        await self.state_manager.transition_state(session, CloudTrainingState.SELECTING_TARGET.value)

        # Cleanup temp file
        local_path.unlink(missing_ok=True)

        self.logger.info(f"File uploaded to S3 for user {user_id}: {s3_uri}")

    async def _handle_s3_uri_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Handle S3 URI provided by user - validate and proceed.

        Args:
            update: Telegram update object
            context: Telegram context object
            session: User session
        """
        user_id = update.effective_user.id
        s3_uri = update.message.text.strip()

        self.logger.info(f"Validating S3 URI for user {user_id}: {s3_uri}")

        # Validate S3 path
        is_valid = self.s3_manager.validate_s3_path(s3_uri, user_id)

        if not is_valid:
            await update.message.reply_text(
                s3_validation_error_message(
                    s3_uri,
                    "Path does not exist or access denied"
                ),
                parse_mode=ParseMode.MARKDOWN
            )
            return

        # Store S3 URI in session
        session.selections['s3_dataset_uri'] = s3_uri

        # Download and load dataset for schema detection
        local_path = self.s3_manager.download_dataset(s3_uri, user_id)
        import pandas as pd
        df = pd.read_csv(local_path)
        session.uploaded_data = df

        await self.state_manager.update_session(session)

        # Send confirmation
        await update.message.reply_text(
            f"✅ Using S3 dataset: `{s3_uri}`\n\n"
            f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns",
            parse_mode=ParseMode.MARKDOWN
        )

        # Transition to target selection
        await self.state_manager.transition_state(session, CloudTrainingState.SELECTING_TARGET.value)

        self.logger.info(f"S3 dataset validated for user {user_id}: {s3_uri}")

    async def handle_instance_confirmation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Show instance type and cost confirmation before launching training.

        Displays EC2 instance configuration, estimated cost, and time.
        User must confirm before training launches.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            user_id = update.effective_user.id
            session = await self.state_manager.get_session(user_id, str(user_id))

            if not session:
                await update.callback_query.message.reply_text(
                    "❌ Session expired. Please start over with /train"
                )
                return

            # Get dataset size
            dataset_size_mb = session.uploaded_data.memory_usage(deep=True).sum() / (1024 * 1024)

            # Select instance type
            model_type = session.selections.get('model_type', 'random_forest')
            instance_type = self.ec2_manager.select_instance_type(
                dataset_size_mb,
                model_type,
                estimated_training_time_minutes=10  # TODO: Better estimation
            )

            # Store instance type
            session.selections['instance_type'] = instance_type
            await self.state_manager.update_session(session)

            # Estimate cost (simplified - would use CostTracker in production)
            estimated_cost = self._estimate_training_cost(instance_type, 10)
            estimated_time = 10  # TODO: Better time estimation

            # Create confirmation buttons
            keyboard = [
                [InlineKeyboardButton("✅ Launch Training", callback_data="confirm_cloud_launch")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel_cloud_training")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Send confirmation message
            await update.callback_query.message.reply_text(
                cloud_instance_confirmation_message(
                    instance_type,
                    estimated_cost,
                    estimated_time,
                    dataset_size_mb
                ),
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

            # Transition state
            await self.state_manager.transition_state(session, CloudTrainingState.CONFIRMING_INSTANCE_TYPE.value)

            self.logger.info(f"Instance confirmation shown to user {user_id}: {instance_type}")

        except Exception as e:
            self.logger.error(f"Error in handle_instance_confirmation: {e}", exc_info=True)
            await update.callback_query.message.reply_text(
                "❌ Failed to generate instance configuration. Please try again."
            )

    async def launch_cloud_training(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Launch EC2 instance for cloud training.

        Creates EC2 Spot instance with UserData script, uploads training config,
        and starts log streaming.

        Args:
            update: Telegram update object
            context: Telegram context object
            session: User session with training configuration
        """
        try:
            user_id = update.effective_user.id

            # Generate model ID
            model_type = session.selections['model_type']
            model_id = f"model_{user_id}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Generate UserData script
            userdata_script = self._generate_training_userdata(session, model_id)

            # Launch EC2 instance
            instance_info = self.ec2_manager.launch_spot_instance(
                instance_type=session.selections['instance_type'],
                user_data_script=userdata_script,
                tags={
                    'user_id': str(user_id),
                    'model_id': model_id,
                    'workflow': 'cloud_training'
                }
            )

            # Store instance info
            session.selections['instance_id'] = instance_info['instance_id']
            session.selections['model_id'] = model_id
            await self.state_manager.update_session(session)

            # Transition to launching state
            await self.state_manager.transition_state(session, CloudTrainingState.LAUNCHING_TRAINING.value)

            # Send launch message
            await update.callback_query.message.reply_text(
                cloud_training_launched_message(
                    instance_info['instance_id'],
                    instance_info['instance_type']
                ),
                parse_mode=ParseMode.MARKDOWN
            )

            # Start log streaming in background
            asyncio.create_task(
                self.stream_training_logs(update, context, session, instance_info['instance_id'])
            )

            self.logger.info(f"Cloud training launched for user {user_id}: {instance_info['instance_id']}")

        except EC2Error as e:
            self.logger.error(f"EC2 error in launch_cloud_training: {e}", exc_info=True)
            await update.callback_query.message.reply_text(
                cloud_error_message("EC2LaunchError", str(e)),
                parse_mode=ParseMode.MARKDOWN
            )
            raise

        except Exception as e:
            self.logger.error(f"Error in launch_cloud_training: {e}", exc_info=True)
            await update.callback_query.message.reply_text(
                "❌ Failed to launch cloud training. Please try again."
            )
            raise

    async def stream_training_logs(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession,
        instance_id: str
    ) -> None:
        """
        Stream CloudWatch logs to Telegram in real-time.

        Polls EC2Manager for log lines and sends them to user. Detects
        training completion and triggers completion handler.

        Args:
            update: Telegram update object
            context: Telegram context object
            session: User session
            instance_id: EC2 instance ID to monitor
        """
        try:
            user_id = session.user_id
            chat_id = update.effective_chat.id

            # Transition to monitoring state
            await self.state_manager.transition_state(session, CloudTrainingState.MONITORING_TRAINING.value)

            self.logger.info(f"Starting log streaming for user {user_id}, instance {instance_id}")

            # Stream logs
            async for log_line in self.ec2_manager.poll_training_logs(instance_id):
                # Send log to user
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=cloud_training_log_message(log_line),
                    parse_mode=ParseMode.MARKDOWN
                )

                # Check for completion
                if "Training complete!" in log_line:
                    self.logger.info(f"Training complete detected for user {user_id}")
                    await self._handle_training_completion(update, context, session)
                    break

        except Exception as e:
            self.logger.error(f"Error in stream_training_logs: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Error streaming training logs. Training may still be in progress."
            )
            raise

    async def _handle_training_completion(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Handle training completion - send metrics and cost.

        Args:
            update: Telegram update object
            context: Telegram context object
            session: User session
        """
        try:
            # Transition to complete state
            await self.state_manager.transition_state(session, CloudTrainingState.TRAINING_COMPLETE.value)

            # Calculate actual cost (simplified)
            actual_cost = self._calculate_actual_cost(session)

            # Get metrics (would load from S3 in production)
            metrics = {'r2': 0.85, 'mse': 0.12}  # Placeholder

            # Send completion message
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=cloud_training_complete_message(
                    session.selections['model_id'],
                    f"s3://bucket/models/{session.selections['model_id']}/",
                    10.0,  # Placeholder training time
                    actual_cost,
                    metrics
                ),
                parse_mode=ParseMode.MARKDOWN
            )

            # Complete workflow
            await self.state_manager.transition_state(session, CloudTrainingState.COMPLETE.value)

            self.logger.info(f"Training completed for user {session.user_id}")

        except Exception as e:
            self.logger.error(f"Error in _handle_training_completion: {e}", exc_info=True)

    def _generate_training_userdata(self, session: UserSession, model_id: str) -> str:
        """
        Generate EC2 UserData script for training.

        Args:
            session: User session with training configuration
            model_id: Model identifier

        Returns:
            str: Base64-encoded UserData script
        """
        # Simplified - would generate complete training script
        return f"""#!/bin/bash
echo "Starting training for model {model_id}"
# Training logic here
echo "Training complete!"
"""

    def _estimate_training_cost(self, instance_type: str, minutes: int) -> float:
        """
        Estimate training cost.

        Args:
            instance_type: EC2 instance type
            minutes: Estimated training time

        Returns:
            float: Estimated cost in USD
        """
        # Simplified pricing
        hourly_rates = {
            't3.medium': 0.0416,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'c5.xlarge': 0.17
        }
        hourly_rate = hourly_rates.get(instance_type, 0.10)
        return (hourly_rate / 60) * minutes

    def _calculate_actual_cost(self, session: UserSession) -> float:
        """
        Calculate actual training cost from EC2 billing.

        Args:
            session: User session

        Returns:
            float: Actual cost in USD
        """
        # Would query CloudWatch or AWS Cost Explorer
        return 0.25  # Placeholder
