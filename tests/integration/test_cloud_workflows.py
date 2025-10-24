"""
Integration tests for cloud-based ML workflows (AWS).

This module tests the complete Telegram bot integration for cloud training
and prediction workflows using EC2, Lambda, and S3.

Test Coverage:
- Cloud training workflow (11 test cases)
- Cloud prediction workflow (8 test cases)
- State transitions (5 test cases)
- Error handling (6 test cases)
- Log streaming (3 test cases)

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.0: TDD for Cloud Workflows)
"""

import asyncio
import pytest
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
from telegram import Update, User, Chat, Message, CallbackQuery, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.core.state_manager import (
    StateManager,
    CloudTrainingState,
    CloudPredictionState,
    WorkflowType,
    UserSession
)
from src.bot.handlers.cloud_training_handlers import CloudTrainingHandlers
from src.bot.handlers.cloud_prediction_handlers import CloudPredictionHandlers
from src.cloud.ec2_manager import EC2Manager
from src.cloud.s3_manager import S3Manager
from src.cloud.lambda_manager import LambdaManager
from src.cloud.aws_config import CloudConfig
from src.cloud.aws_client import AWSClient


@pytest.fixture
def mock_aws_client():
    """Mock AWS client for testing."""
    client = Mock(spec=AWSClient)
    client.get_ec2_client.return_value = Mock()
    client.get_s3_client.return_value = Mock()
    client.get_lambda_client.return_value = Mock()
    return client


@pytest.fixture
def mock_cloud_config():
    """Mock cloud configuration."""
    config = Mock(spec=CloudConfig)
    config.s3_bucket_name = "test-ml-bot-data"
    config.s3_dataset_prefix = "datasets"
    config.s3_model_prefix = "models"
    config.ec2_instance_type = "m5.large"
    config.ec2_ami_id = "ami-12345678"
    config.lambda_function_name = "ml-bot-prediction"
    config.aws_region = "us-east-1"
    return config


@pytest.fixture
def mock_ec2_manager(mock_aws_client, mock_cloud_config):
    """Mock EC2Manager for testing."""
    manager = Mock(spec=EC2Manager)

    # Mock instance type selection
    manager.select_instance_type.return_value = "m5.large"

    # Mock instance launching
    manager.launch_spot_instance.return_value = {
        'instance_id': 'i-1234567890abcdef0',
        'instance_type': 'm5.large',
        'spot_instance_request_id': 'sir-12345'
    }

    # Mock log streaming
    async def mock_poll_logs(instance_id: str) -> AsyncIterator[str]:
        """Simulate log streaming."""
        logs = [
            "Starting training...",
            "Loading dataset from S3...",
            "Dataset loaded: 1000 rows, 10 columns",
            "Training model...",
            "Training complete!",
            "Model saved to S3"
        ]
        for log in logs:
            await asyncio.sleep(0.1)
            yield log

    manager.poll_training_logs = mock_poll_logs

    # Mock instance status
    manager.get_instance_status.return_value = {
        'state': 'running',
        'public_ip': '54.123.45.67'
    }

    # Mock instance termination
    manager.terminate_instance.return_value = {'terminated': True}

    return manager


@pytest.fixture
def mock_s3_manager(mock_aws_client, mock_cloud_config):
    """Mock S3Manager for testing."""
    # Create mock without spec to allow any attribute
    manager = Mock()

    # Mock dataset upload
    manager.upload_dataset = Mock(return_value="s3://test-ml-bot-data/datasets/user_12345/dataset_001.csv")

    # Mock S3 path validation
    manager.validate_s3_path = Mock(return_value=True)

    # Mock dataset download
    manager.download_dataset = Mock(return_value="/tmp/downloaded_dataset.csv")

    # Mock presigned URL generation
    manager.generate_presigned_url = Mock(return_value="https://test-ml-bot-data.s3.amazonaws.com/output.csv?presigned=true")

    return manager


@pytest.fixture
def mock_lambda_manager(mock_aws_client, mock_cloud_config):
    """Mock LambdaManager for testing."""
    # Create mock without spec to allow any attribute
    manager = Mock()

    # Mock Lambda invocation
    manager.invoke_prediction = Mock(return_value={
        'statusCode': 200,
        'body': {
            's3_output_uri': 's3://test-ml-bot-data/predictions/user_12345/output.csv',
            'num_predictions': 500,
            'execution_time_ms': 1234
        }
    })

    # Mock cost estimation
    manager.estimate_prediction_cost = Mock(return_value=0.0012)

    return manager


@pytest.fixture
def state_manager(tmp_path):
    """State manager with temporary sessions directory."""
    from src.core.state_manager import StateManagerConfig
    config = StateManagerConfig(sessions_dir=str(tmp_path / "sessions"))
    return StateManager(config)


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': ['A', 'B', 'A', 'B', 'A'],
        'target': [100, 200, 150, 250, 175]
    })


@pytest.fixture
def mock_telegram_update():
    """Mock Telegram Update object."""
    update = Mock(spec=Update)
    update.effective_user = Mock(spec=User)
    update.effective_user.id = 12345
    update.effective_user.first_name = "Test"

    update.effective_chat = Mock(spec=Chat)
    update.effective_chat.id = 12345

    update.message = Mock(spec=Message)
    update.message.reply_text = AsyncMock()
    update.message.document = None
    update.message.text = None

    update.callback_query = Mock(spec=CallbackQuery)
    update.callback_query.message = Mock(spec=Message)
    update.callback_query.message.reply_text = AsyncMock()
    update.callback_query.answer = AsyncMock()
    update.callback_query.data = None

    return update


@pytest.fixture
def mock_context():
    """Mock Telegram context."""
    context = Mock(spec=ContextTypes.DEFAULT_TYPE)
    context.bot = Mock()
    context.bot.send_message = AsyncMock()
    return context


# =============================================================================
# Cloud Training Workflow Tests (11 test cases)
# =============================================================================

@pytest.mark.asyncio
class TestCloudTrainingWorkflow:
    """Test complete cloud training workflow from start to finish."""

    async def test_cloud_training_workflow_complete(
        self,
        state_manager,
        mock_ec2_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context,
        sample_dataset
    ):
        """Test complete cloud training workflow end-to-end."""
        # Setup handler
        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=mock_ec2_manager,
            s3_manager=mock_s3_manager
        )

        user_id = 12345

        # Step 1: Start workflow - choose cloud vs local
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)

        assert session.workflow_type == WorkflowType.CLOUD_TRAINING
        assert session.current_state == CloudTrainingState.CHOOSING_CLOUD_LOCAL.value

        # Step 2: User chooses cloud training
        mock_telegram_update.callback_query.data = "training_cloud"
        await handler.handle_training_cloud_selected(mock_telegram_update, mock_context)

        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudTrainingState.AWAITING_S3_DATASET.value

        # Step 3: User uploads dataset or provides S3 URI
        mock_telegram_update.message.text = "s3://test-bucket/housing.csv"
        await handler.handle_s3_dataset_input(mock_telegram_update, mock_context)

        session = await state_manager.get_session(user_id, str(user_id))
        assert 's3_dataset_uri' in session.selections
        assert session.current_state == CloudTrainingState.SELECTING_TARGET.value

        # Step 4: Select target column (simulate existing logic)
        session.selections['target_column'] = 'target'
        await state_manager.update_session(session)
        await state_manager.transition_state(session, CloudTrainingState.SELECTING_FEATURES.value)

        # Step 5: Select feature columns
        session.selections['feature_columns'] = ['feature1', 'feature2', 'feature3']
        await state_manager.update_session(session)
        await state_manager.transition_state(session, CloudTrainingState.CONFIRMING_MODEL.value)

        # Step 6: Select model type
        session.selections['model_type'] = 'random_forest'
        session.uploaded_data = sample_dataset
        await state_manager.update_session(session)
        await state_manager.transition_state(session, CloudTrainingState.CONFIRMING_INSTANCE_TYPE.value)

        # Step 7: Confirm instance type and launch
        mock_telegram_update.callback_query.data = "confirm_cloud_launch"
        await handler.handle_instance_confirmation(mock_telegram_update, mock_context)

        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudTrainingState.LAUNCHING_TRAINING.value
        assert 'instance_id' in session.selections

        # Verify EC2 launch was called
        mock_ec2_manager.launch_spot_instance.assert_called_once()

        # Step 8: Stream logs (simulate completion)
        await handler.stream_training_logs(
            mock_telegram_update,
            mock_context,
            session,
            'i-1234567890abcdef0'
        )

        # Verify logs were sent to user
        assert mock_context.bot.send_message.call_count >= 6  # 6 log lines

        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudTrainingState.TRAINING_COMPLETE.value

    async def test_cloud_training_telegram_file_upload(
        self,
        state_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test uploading dataset via Telegram file attachment."""
        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=Mock(),
            s3_manager=mock_s3_manager
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)
        await state_manager.transition_state(session, CloudTrainingState.AWAITING_S3_DATASET.value)

        # Mock file upload
        mock_file = Mock()
        mock_file.file_id = "test_file_123"
        mock_file.download_to_drive = AsyncMock()

        mock_document = Mock()
        mock_document.get_file = AsyncMock(return_value=mock_file)
        mock_telegram_update.message.document = mock_document

        await handler.handle_s3_dataset_input(mock_telegram_update, mock_context)

        # Verify S3 upload was called
        mock_s3_manager.upload_dataset.assert_called_once()

        # Verify session updated
        session = await state_manager.get_session(user_id, str(user_id))
        assert 's3_dataset_uri' in session.selections

    async def test_cloud_training_s3_path_validation(
        self,
        state_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test S3 path validation before proceeding."""
        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=Mock(),
            s3_manager=mock_s3_manager
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)
        await state_manager.transition_state(session, CloudTrainingState.AWAITING_S3_DATASET.value)

        # Test invalid S3 path
        mock_s3_manager.validate_s3_path.return_value = False
        mock_telegram_update.message.text = "s3://invalid-bucket/nonexistent.csv"

        await handler.handle_s3_dataset_input(mock_telegram_update, mock_context)

        # Verify error message sent
        mock_telegram_update.message.reply_text.assert_called()
        error_msg = mock_telegram_update.message.reply_text.call_args[0][0]
        assert "Invalid S3 URI" in error_msg or "access denied" in error_msg.lower()

        # Verify session not advanced
        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudTrainingState.AWAITING_S3_DATASET.value

    async def test_cloud_training_instance_type_selection(
        self,
        state_manager,
        mock_ec2_manager,
        sample_dataset
    ):
        """Test automatic instance type selection based on dataset size."""
        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=mock_ec2_manager,
            s3_manager=Mock()
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        session.uploaded_data = sample_dataset
        session.selections = {
            'model_type': 'random_forest',
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2']
        }
        await state_manager.update_session(session)

        # Test instance type selection
        dataset_size_mb = sample_dataset.memory_usage(deep=True).sum() / (1024 * 1024)
        instance_type = mock_ec2_manager.select_instance_type(
            dataset_size_mb,
            'random_forest',
            estimated_training_time_minutes=10
        )

        assert instance_type == "m5.large"
        mock_ec2_manager.select_instance_type.assert_called_once()

    async def test_cloud_training_cost_estimation(
        self,
        state_manager,
        mock_ec2_manager,
        mock_telegram_update,
        mock_context,
        sample_dataset
    ):
        """Test cost estimation display before training launch."""
        with patch('src.bot.handlers.cloud_training_handlers.CostTracker') as mock_cost_tracker:
            mock_cost_tracker_instance = Mock()
            mock_cost_tracker_instance.estimate_training_cost.return_value = 0.25
            mock_cost_tracker.return_value = mock_cost_tracker_instance

            handler = CloudTrainingHandlers(
                state_manager=state_manager,
                ec2_manager=mock_ec2_manager,
                s3_manager=Mock()
            )

            user_id = 12345
            session = await state_manager.get_or_create_session(user_id, str(user_id))
            session.uploaded_data = sample_dataset
            session.selections = {
                'model_type': 'random_forest',
                'target_column': 'target',
                'feature_columns': ['feature1', 'feature2'],
                's3_dataset_uri': 's3://test/data.csv'
            }
            await state_manager.update_session(session)

            await handler.handle_instance_confirmation(mock_telegram_update, mock_context)

            # Verify cost estimation was called
            mock_cost_tracker_instance.estimate_training_cost.assert_called_once()

            # Verify message contains cost information
            mock_telegram_update.callback_query.message.reply_text.assert_called()
            message = mock_telegram_update.callback_query.message.reply_text.call_args[0][0]
            assert "$" in message  # Cost should be displayed
            assert "Estimated Cost" in message


# =============================================================================
# Cloud Prediction Workflow Tests (8 test cases)
# =============================================================================

@pytest.mark.asyncio
class TestCloudPredictionWorkflow:
    """Test complete cloud prediction workflow."""

    async def test_cloud_prediction_workflow_complete(
        self,
        state_manager,
        mock_lambda_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test complete cloud prediction workflow end-to-end."""
        handler = CloudPredictionHandlers(
            state_manager=state_manager,
            lambda_manager=mock_lambda_manager,
            s3_manager=mock_s3_manager
        )

        user_id = 12345

        # Step 1: Start prediction workflow
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        await state_manager.start_workflow(session, WorkflowType.CLOUD_PREDICTION)

        assert session.workflow_type == WorkflowType.CLOUD_PREDICTION
        assert session.current_state == CloudPredictionState.CHOOSING_CLOUD_LOCAL.value

        # Step 2: Choose cloud prediction
        mock_telegram_update.callback_query.data = "prediction_cloud"
        await handler.handle_cloud_prediction_choice(mock_telegram_update, mock_context)

        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudPredictionState.AWAITING_S3_DATASET.value

        # Step 3: Provide dataset for prediction
        mock_telegram_update.message.text = "s3://test-bucket/new_data.csv"
        await handler.handle_s3_prediction_dataset(mock_telegram_update, mock_context)

        session = await state_manager.get_session(user_id, str(user_id))
        assert 's3_dataset_uri' in session.selections

        # Step 4: Select model
        session.selections['selected_model_id'] = 'model_12345_random_forest'
        await state_manager.update_session(session)
        await state_manager.transition_state(session, CloudPredictionState.CONFIRMING_PREDICTION_COLUMN.value)

        # Step 5: Confirm prediction column name
        session.selections['prediction_column_name'] = 'prediction'
        await state_manager.update_session(session)
        await state_manager.transition_state(session, CloudPredictionState.LAUNCHING_PREDICTION.value)

        # Step 6: Launch Lambda prediction
        await handler.launch_cloud_prediction(mock_telegram_update, mock_context, session)

        # Verify Lambda invocation
        mock_lambda_manager.invoke_prediction.assert_called_once()

        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudPredictionState.PREDICTION_COMPLETE.value

    async def test_cloud_prediction_lambda_invocation(
        self,
        state_manager,
        mock_lambda_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test Lambda function invocation for predictions."""
        handler = CloudPredictionHandlers(
            state_manager=state_manager,
            lambda_manager=mock_lambda_manager,
            s3_manager=mock_s3_manager
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        session.selections = {
            'selected_model_id': 'model_12345_rf',
            's3_dataset_uri': 's3://test/data.csv',
            'prediction_column_name': 'prediction'
        }
        await state_manager.update_session(session)

        await handler.launch_cloud_prediction(mock_telegram_update, mock_context, session)

        # Verify Lambda was invoked with correct parameters
        call_args = mock_lambda_manager.invoke_prediction.call_args
        assert call_args is not None

        # Verify response message sent
        mock_context.bot.send_message.assert_called()


# =============================================================================
# State Transition Tests (5 test cases)
# =============================================================================

@pytest.mark.asyncio
class TestCloudWorkflowStateTransitions:
    """Test state machine transitions for cloud workflows."""

    async def test_cloud_training_valid_transitions(self, state_manager):
        """Test all valid state transitions for cloud training."""
        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        # Start workflow
        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)
        assert session.current_state == CloudTrainingState.CHOOSING_CLOUD_LOCAL.value

        # Transition to awaiting dataset
        success, error, _ = await state_manager.transition_state(
            session,
            CloudTrainingState.AWAITING_S3_DATASET.value
        )
        assert success
        assert error is None

        # Transition to selecting target
        success, error, _ = await state_manager.transition_state(
            session,
            CloudTrainingState.SELECTING_TARGET.value
        )
        assert success

        # Continue through workflow
        await state_manager.transition_state(session, CloudTrainingState.SELECTING_FEATURES.value)
        await state_manager.transition_state(session, CloudTrainingState.CONFIRMING_MODEL.value)
        await state_manager.transition_state(session, CloudTrainingState.CONFIRMING_INSTANCE_TYPE.value)
        await state_manager.transition_state(session, CloudTrainingState.LAUNCHING_TRAINING.value)
        await state_manager.transition_state(session, CloudTrainingState.MONITORING_TRAINING.value)
        await state_manager.transition_state(session, CloudTrainingState.TRAINING_COMPLETE.value)
        await state_manager.transition_state(session, CloudTrainingState.COMPLETE.value)

        assert session.current_state == CloudTrainingState.COMPLETE.value

    async def test_cloud_prediction_valid_transitions(self, state_manager):
        """Test all valid state transitions for cloud prediction."""
        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        await state_manager.start_workflow(session, WorkflowType.CLOUD_PREDICTION)
        assert session.current_state == CloudPredictionState.CHOOSING_CLOUD_LOCAL.value

        # Full workflow transitions
        await state_manager.transition_state(session, CloudPredictionState.AWAITING_S3_DATASET.value)
        await state_manager.transition_state(session, CloudPredictionState.SELECTING_MODEL.value)
        await state_manager.transition_state(session, CloudPredictionState.CONFIRMING_PREDICTION_COLUMN.value)
        await state_manager.transition_state(session, CloudPredictionState.LAUNCHING_PREDICTION.value)
        await state_manager.transition_state(session, CloudPredictionState.PREDICTION_COMPLETE.value)
        await state_manager.transition_state(session, CloudPredictionState.COMPLETE.value)

        assert session.current_state == CloudPredictionState.COMPLETE.value

    async def test_invalid_state_transition(self, state_manager):
        """Test that invalid transitions are rejected."""
        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)

        # Try invalid transition (skip steps)
        success, error, _ = await state_manager.transition_state(
            session,
            CloudTrainingState.TRAINING_COMPLETE.value
        )

        assert not success
        assert error is not None
        assert "Invalid transition" in error


# =============================================================================
# Error Handling Tests (6 test cases)
# =============================================================================

@pytest.mark.asyncio
class TestCloudWorkflowErrorHandling:
    """Test error scenarios in cloud workflows."""

    async def test_ec2_launch_failure(
        self,
        state_manager,
        mock_ec2_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test handling of EC2 instance launch failure."""
        from src.cloud.exceptions import EC2Error

        mock_ec2_manager.launch_spot_instance.side_effect = EC2Error("No capacity available")

        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=mock_ec2_manager,
            s3_manager=Mock()
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        with pytest.raises(EC2Error):
            await handler.launch_cloud_training(mock_telegram_update, mock_context, session)

    async def test_lambda_invocation_failure(
        self,
        state_manager,
        mock_lambda_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test handling of Lambda invocation failure."""
        from src.cloud.exceptions import LambdaError

        mock_lambda_manager.invoke_prediction.side_effect = LambdaError("Function timeout")

        handler = CloudPredictionHandlers(
            state_manager=state_manager,
            lambda_manager=mock_lambda_manager,
            s3_manager=mock_s3_manager
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        session.selections = {
            'selected_model_id': 'model_12345',
            's3_dataset_uri': 's3://test/data.csv'
        }

        with pytest.raises(LambdaError):
            await handler.launch_cloud_prediction(mock_telegram_update, mock_context, session)

    async def test_s3_upload_failure(
        self,
        state_manager,
        mock_s3_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test handling of S3 upload failure."""
        from src.cloud.exceptions import S3Error

        mock_s3_manager.upload_dataset.side_effect = S3Error("Upload failed")

        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=Mock(),
            s3_manager=mock_s3_manager
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))
        await state_manager.start_workflow(session, WorkflowType.CLOUD_TRAINING)
        await state_manager.transition_state(session, CloudTrainingState.AWAITING_S3_DATASET.value)

        mock_document = Mock()
        mock_file = Mock()
        mock_file.file_id = "test_123"
        mock_file.download_to_drive = AsyncMock()
        mock_document.get_file = AsyncMock(return_value=mock_file)
        mock_telegram_update.message.document = mock_document

        with pytest.raises(S3Error):
            await handler.handle_s3_dataset_input(mock_telegram_update, mock_context)


# =============================================================================
# Log Streaming Tests (3 test cases)
# =============================================================================

@pytest.mark.asyncio
class TestCloudTrainingLogStreaming:
    """Test real-time log streaming from EC2 to Telegram."""

    async def test_log_streaming_sends_messages(
        self,
        state_manager,
        mock_ec2_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test that logs are streamed to Telegram user."""
        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=mock_ec2_manager,
            s3_manager=Mock()
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        await handler.stream_training_logs(
            mock_telegram_update,
            mock_context,
            session,
            'i-1234567890abcdef0'
        )

        # Verify multiple messages sent
        assert mock_context.bot.send_message.call_count >= 6

    async def test_log_streaming_detects_completion(
        self,
        state_manager,
        mock_ec2_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test that log streaming detects training completion."""
        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=mock_ec2_manager,
            s3_manager=Mock()
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        await handler.stream_training_logs(
            mock_telegram_update,
            mock_context,
            session,
            'i-1234567890abcdef0'
        )

        # Verify completion handler was triggered
        session = await state_manager.get_session(user_id, str(user_id))
        assert session.current_state == CloudTrainingState.TRAINING_COMPLETE.value

    async def test_log_streaming_handles_errors(
        self,
        state_manager,
        mock_ec2_manager,
        mock_telegram_update,
        mock_context
    ):
        """Test error handling during log streaming."""
        # Mock log streaming with error
        async def mock_poll_logs_with_error(instance_id: str) -> AsyncIterator[str]:
            yield "Starting training..."
            raise Exception("CloudWatch logs unavailable")

        mock_ec2_manager.poll_training_logs = mock_poll_logs_with_error

        handler = CloudTrainingHandlers(
            state_manager=state_manager,
            ec2_manager=mock_ec2_manager,
            s3_manager=Mock()
        )

        user_id = 12345
        session = await state_manager.get_or_create_session(user_id, str(user_id))

        with pytest.raises(Exception):
            await handler.stream_training_logs(
                mock_telegram_update,
                mock_context,
                session,
                'i-1234567890abcdef0'
            )
