"""
Telegram bot handlers for cloud-based prediction workflows.

This module implements Telegram bot handlers for cloud predictions using AWS Lambda,
S3, and trained models. Handles dataset upload, model selection, Lambda invocation,
and result retrieval.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.0: Cloud Workflow Telegram Integration)
"""

import logging
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, UserSession, CloudPredictionState, WorkflowType
from src.cloud.lambda_manager import LambdaManager
from src.cloud.s3_manager import S3Manager
from src.cloud.exceptions import LambdaError, S3Error
from src.bot.messages.cloud_messages import (
    CHOOSE_CLOUD_LOCAL_MESSAGE,
    AWAITING_S3_DATASET_MESSAGE,
    cloud_prediction_launched_message,
    cloud_prediction_complete_message,
    cloud_error_message,
    s3_validation_error_message,
)

logger = logging.getLogger(__name__)


class CloudPredictionHandlers:
    """
    Handlers for cloud-based prediction workflows.

    This class manages the complete cloud prediction workflow from dataset upload
    to result retrieval, using AWS Lambda for serverless predictions.
    """

    def __init__(
        self,
        state_manager: StateManager,
        lambda_manager: LambdaManager,
        s3_manager: S3Manager
    ):
        """
        Initialize cloud prediction handlers.

        Args:
            state_manager: State manager for workflow tracking
            lambda_manager: Lambda manager for prediction invocation
            s3_manager: S3 manager for dataset/result storage
        """
        self.state_manager = state_manager
        self.lambda_manager = lambda_manager
        self.s3_manager = s3_manager
        self.logger = logger

    async def handle_cloud_prediction_choice(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle user selecting cloud prediction option.

        Transitions workflow to cloud prediction and prompts for dataset input.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            user_id = update.effective_user.id
            session = await self.state_manager.get_or_create_session(user_id, str(user_id))

            # Start cloud prediction workflow
            await self.state_manager.start_workflow(session, WorkflowType.CLOUD_PREDICTION)
            await self.state_manager.transition_state(session, CloudPredictionState.AWAITING_S3_DATASET.value)

            # Answer callback query
            await update.callback_query.answer()

            # Send dataset input prompt
            await update.callback_query.message.reply_text(
                AWAITING_S3_DATASET_MESSAGE,
                parse_mode=ParseMode.MARKDOWN
            )

            self.logger.info(f"User {user_id} started cloud prediction workflow")

        except Exception as e:
            self.logger.error(f"Error in handle_cloud_prediction_choice: {e}", exc_info=True)
            await update.callback_query.message.reply_text(
                "❌ Failed to start cloud prediction. Please try again."
            )

    async def handle_s3_prediction_dataset(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle prediction dataset input from user (S3 URI or file upload).

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

            if not session or session.current_state != CloudPredictionState.AWAITING_S3_DATASET.value:
                await update.message.reply_text(
                    "❌ Please start cloud prediction with /predict first."
                )
                return

            # Handle file upload
            if update.message.document:
                await self._handle_file_upload(update, context, session)

            # Handle S3 URI
            elif update.message.text and update.message.text.startswith('s3://'):
                await self._handle_s3_uri(update, context, session)

            else:
                await update.message.reply_text(
                    "❌ Invalid input. Please upload a CSV file or provide an S3 URI starting with `s3://`",
                    parse_mode=ParseMode.MARKDOWN
                )

        except S3Error as e:
            self.logger.error(f"S3 error in handle_s3_prediction_dataset: {e}", exc_info=True)
            await update.message.reply_text(
                cloud_error_message("S3Error", str(e)),
                parse_mode=ParseMode.MARKDOWN
            )
            raise

        except Exception as e:
            self.logger.error(f"Error in handle_s3_prediction_dataset: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ Failed to process prediction dataset. Please try again."
            )

    async def _handle_file_upload(
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
        local_path = f"/tmp/{file.file_id}.csv"

        self.logger.info(f"Downloading prediction file from Telegram for user {user_id}")
        await file.download_to_drive(local_path)

        # Upload to S3
        s3_uri = self.s3_manager.upload_dataset(
            user_id=user_id,
            file_path=local_path,
            dataset_name=update.message.document.file_name
        )

        # Store S3 URI in session
        session.selections['s3_dataset_uri'] = s3_uri
        await self.state_manager.update_session(session)

        # Send confirmation
        await update.message.reply_text(
            f"✅ Prediction dataset uploaded to S3: `{s3_uri}`",
            parse_mode=ParseMode.MARKDOWN
        )

        # Transition to model selection
        await self.state_manager.transition_state(session, CloudPredictionState.SELECTING_MODEL.value)

        self.logger.info(f"Prediction dataset uploaded to S3 for user {user_id}: {s3_uri}")

    async def _handle_s3_uri(
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

        self.logger.info(f"Validating prediction S3 URI for user {user_id}: {s3_uri}")

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
        await self.state_manager.update_session(session)

        # Send confirmation
        await update.message.reply_text(
            f"✅ Using S3 dataset for prediction: `{s3_uri}`",
            parse_mode=ParseMode.MARKDOWN
        )

        # Transition to model selection
        await self.state_manager.transition_state(session, CloudPredictionState.SELECTING_MODEL.value)

        self.logger.info(f"Prediction S3 dataset validated for user {user_id}: {s3_uri}")

    async def launch_cloud_prediction(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: UserSession
    ) -> None:
        """
        Launch Lambda function for cloud prediction.

        Invokes AWS Lambda with model and dataset, waits for completion,
        and retrieves results from S3.

        Args:
            update: Telegram update object
            context: Telegram context object
            session: User session with prediction configuration
        """
        try:
            user_id = session.user_id

            # Get configuration from session
            model_id = session.selections.get('selected_model_id')
            s3_dataset_uri = session.selections.get('s3_dataset_uri')
            prediction_column_name = session.selections.get('prediction_column_name', 'prediction')

            if not model_id or not s3_dataset_uri:
                await update.message.reply_text(
                    "❌ Missing model or dataset. Please start over with /predict"
                )
                return

            # Transition to launching state
            await self.state_manager.transition_state(session, CloudPredictionState.LAUNCHING_PREDICTION.value)

            # Send launch message
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=cloud_prediction_launched_message(
                    request_id="lambda-" + str(hash(s3_dataset_uri))[:8],
                    num_rows=1000  # Placeholder - would get from dataset metadata
                ),
                parse_mode=ParseMode.MARKDOWN
            )

            # Invoke Lambda function
            self.logger.info(f"Invoking Lambda for user {user_id}, model {model_id}")

            response = self.lambda_manager.invoke_prediction(
                user_id=user_id,
                model_id=model_id,
                s3_dataset_uri=s3_dataset_uri,
                prediction_column_name=prediction_column_name
            )

            # Extract results from response
            s3_output_uri = response['body']['s3_output_uri']
            num_predictions = response['body']['num_predictions']
            execution_time_ms = response['body']['execution_time_ms']

            # Generate presigned URL for download
            presigned_url = self.s3_manager.generate_presigned_url(s3_output_uri, expiration=3600)

            # Estimate cost
            cost = self.lambda_manager.estimate_prediction_cost(num_predictions)

            # Transition to complete state
            await self.state_manager.transition_state(session, CloudPredictionState.PREDICTION_COMPLETE.value)

            # Send completion message
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=cloud_prediction_complete_message(
                    s3_output_uri,
                    num_predictions,
                    execution_time_ms,
                    cost,
                    presigned_url
                ),
                parse_mode=ParseMode.MARKDOWN
            )

            # Complete workflow
            await self.state_manager.transition_state(session, CloudPredictionState.COMPLETE.value)

            self.logger.info(f"Cloud prediction completed for user {user_id}: {s3_output_uri}")

        except LambdaError as e:
            self.logger.error(f"Lambda error in launch_cloud_prediction: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=cloud_error_message("LambdaError", str(e)),
                parse_mode=ParseMode.MARKDOWN
            )
            raise

        except Exception as e:
            self.logger.error(f"Error in launch_cloud_prediction: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Failed to complete cloud prediction. Please try again."
            )
            raise
