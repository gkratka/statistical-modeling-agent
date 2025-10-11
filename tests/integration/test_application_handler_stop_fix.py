"""Integration tests for ApplicationHandlerStop exception handling fix.

This test suite validates that ApplicationHandlerStop exceptions are properly
re-raised and not caught by generic exception handlers, preventing false error
messages from being displayed to users.

Test coverage:
1. File path input with ApplicationHandlerStop
2. Schema input with ApplicationHandlerStop
3. Generic exceptions still caught correctly
4. Control flow vs error exception distinction

Related: dev/implemented/file-path-error-message-1.md
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes, ApplicationHandlerStop

from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import StateManager, MLTrainingState
from src.processors.data_loader import DataLoader
from src.utils.exceptions import ValidationError


@pytest.fixture
def mock_state_manager():
    """Create mock StateManager."""
    manager = AsyncMock(spec=StateManager)

    # Create mock session
    mock_session = Mock()
    mock_session.user_id = 12345
    mock_session.conversation_id = "chat_123"
    mock_session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
    mock_session.file_path = None
    mock_session.selections = {}

    manager.get_session = AsyncMock(return_value=mock_session)
    manager.update_session = AsyncMock()
    manager.transition_state = AsyncMock(return_value=(True, None, None))

    return manager


@pytest.fixture
def mock_data_loader():
    """Create mock DataLoader."""
    loader = Mock(spec=DataLoader)
    loader.allowed_directories = ["/tmp", "/home/user/data"]
    loader.local_max_size_mb = 100
    loader.local_extensions = [".csv", ".xlsx"]
    return loader


@pytest.fixture
def mock_update():
    """Create mock Telegram Update object."""
    update = Mock(spec=Update)

    # Mock user
    user = Mock(spec=User)
    user.id = 12345
    update.effective_user = user

    # Mock chat
    chat = Mock(spec=Chat)
    chat.id = 67890
    update.effective_chat = chat

    # Mock message
    message = Mock(spec=Message)
    message.text = "/tmp/test.csv"
    message.reply_text = AsyncMock()
    update.message = message
    update.effective_message = message

    return update


@pytest.fixture
def mock_context():
    """Create mock ContextTypes object."""
    return Mock(spec=ContextTypes.DEFAULT_TYPE)


@pytest.fixture
def handler(mock_state_manager, mock_data_loader):
    """Create LocalPathMLTrainingHandler instance."""
    return LocalPathMLTrainingHandler(mock_state_manager, mock_data_loader)


class TestApplicationHandlerStopFix:
    """Test suite for ApplicationHandlerStop exception handling."""

    @pytest.mark.asyncio
    async def test_file_path_input_raises_application_handler_stop(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that _process_file_path_input raises ApplicationHandlerStop
        after successful path validation without catching it as an error.

        Expected behavior:
        - Path validation succeeds
        - ApplicationHandlerStop is raised
        - No error message displayed to user
        - Load options are shown
        """
        # Setup: valid path that passes validation
        mock_update.message.text = "/tmp/test.csv"

        # Mock successful path validation
        with patch('src.bot.ml_handlers.ml_training_local_path.validate_local_path') as mock_validate:
            mock_validate.return_value = (True, None, "/tmp/test.csv")

            with patch('src.bot.ml_handlers.ml_training_local_path.get_file_size_mb') as mock_size:
                mock_size.return_value = 1.5  # 1.5 MB file

                # Mock session
                session = await mock_state_manager.get_session(12345, "chat_123")
                session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

                # Execute: should raise ApplicationHandlerStop
                with pytest.raises(ApplicationHandlerStop):
                    await handler._process_file_path_input(
                        mock_update,
                        mock_context,
                        session,
                        "/tmp/test.csv"
                    )

                # Verify: load options were shown (not error message)
                assert mock_update.message.reply_text.call_count >= 1

                # Check that NO error message was sent
                for call in mock_update.message.reply_text.call_args_list:
                    message_text = call[0][0] if call[0] else ""
                    assert "Unexpected Error" not in message_text
                    assert "❌" not in message_text or "Load Now" in message_text  # Load options contain ❌ for reject

    @pytest.mark.asyncio
    async def test_schema_input_raises_application_handler_stop(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that _process_schema_input raises ApplicationHandlerStop
        after successful schema parsing without catching it as an error.

        Expected behavior:
        - Schema parsing succeeds
        - ApplicationHandlerStop is raised
        - No error message displayed to user
        - Model selection is shown
        """
        # Setup: valid schema input
        schema_text = "price, sqft, bedrooms"
        mock_update.message.text = schema_text

        # Mock session
        session = await mock_state_manager.get_session(12345, "chat_123")
        session.current_state = MLTrainingState.AWAITING_SCHEMA_INPUT.value
        session.file_path = "/tmp/test.csv"
        session.selections = {}

        # Mock successful schema parsing
        with patch('src.bot.ml_handlers.ml_training_local_path.SchemaParser') as mock_parser:
            mock_schema = Mock()
            mock_schema.target = "price"
            mock_schema.features = ["sqft", "bedrooms"]
            mock_schema.format_detected = "comma_separated"
            mock_schema.raw_input = schema_text

            mock_parser.parse = Mock(return_value=mock_schema)

            # Execute: should raise ApplicationHandlerStop
            with pytest.raises(ApplicationHandlerStop):
                await handler._process_schema_input(
                    mock_update,
                    mock_context,
                    session,
                    schema_text
                )

            # Verify: schema accepted message and model selection shown
            assert mock_update.message.reply_text.call_count >= 1

            # Check that NO error message was sent
            for call in mock_update.message.reply_text.call_args_list:
                message_text = call[0][0] if call[0] else ""
                assert "Unexpected Error" not in message_text

    @pytest.mark.asyncio
    async def test_file_path_generic_exceptions_still_caught(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that generic exceptions (NOT ApplicationHandlerStop) are still
        properly caught and result in error messages to users.

        Expected behavior:
        - Generic exception occurs
        - Exception is caught
        - User-friendly error message displayed
        - ApplicationHandlerStop is NOT raised
        """
        # Setup: path that will cause an exception
        mock_update.message.text = "/tmp/test.csv"

        # Mock validation to raise a generic exception
        with patch('src.bot.ml_handlers.ml_training_local_path.validate_local_path') as mock_validate:
            mock_validate.side_effect = RuntimeError("Unexpected validation error")

            # Mock session
            session = await mock_state_manager.get_session(12345, "chat_123")
            session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

            # Execute: should NOT raise ApplicationHandlerStop (exception handled internally)
            await handler._process_file_path_input(
                mock_update,
                mock_context,
                session,
                "/tmp/test.csv"
            )

            # Verify: error message was sent to user
            assert mock_update.message.reply_text.called

            # Find the error message call
            error_message_sent = False
            for call in mock_update.message.reply_text.call_args_list:
                message_text = call[0][0] if call[0] else ""
                if "Unexpected Error" in message_text or "❌" in message_text:
                    error_message_sent = True
                    break

            assert error_message_sent, "Expected error message to be sent to user"

    @pytest.mark.asyncio
    async def test_schema_input_validation_error_caught(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that ValidationError from schema parsing is properly caught
        and results in a specific error message to users.

        Expected behavior:
        - ValidationError is raised by schema parser
        - Exception is caught by specific handler
        - Schema parse error message displayed
        - ApplicationHandlerStop is NOT raised
        """
        # Setup: invalid schema input
        schema_text = "invalid_schema_format"
        mock_update.message.text = schema_text

        # Mock session
        session = await mock_state_manager.get_session(12345, "chat_123")
        session.current_state = MLTrainingState.AWAITING_SCHEMA_INPUT.value
        session.file_path = "/tmp/test.csv"
        session.selections = {}

        # Mock schema parser to raise ValidationError
        with patch('src.bot.ml_handlers.ml_training_local_path.SchemaParser') as mock_parser:
            mock_parser.parse.side_effect = ValidationError("Invalid schema format")

            # Execute: should NOT raise ApplicationHandlerStop (ValidationError handled)
            await handler._process_schema_input(
                mock_update,
                mock_context,
                session,
                schema_text
            )

            # Verify: schema parse error message was sent
            assert mock_update.message.reply_text.called

            # Verify specific error handling (not generic "Unexpected Error")
            message_text = mock_update.message.reply_text.call_args[0][0]
            # Should be schema-specific error, not generic error
            assert "Invalid schema format" in message_text or "Schema" in message_text

    @pytest.mark.asyncio
    async def test_exception_handler_ordering(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that exception handlers are in correct order:
        1. ApplicationHandlerStop (re-raise immediately)
        2. Specific exceptions (ValidationError, etc.)
        3. Generic Exception (catch-all)

        This ensures ApplicationHandlerStop is never caught as an error.
        """
        # Setup: valid path
        mock_update.message.text = "/tmp/test.csv"

        # Mock successful validation
        with patch('src.bot.ml_handlers.ml_training_local_path.validate_local_path') as mock_validate:
            mock_validate.return_value = (True, None, "/tmp/test.csv")

            with patch('src.bot.ml_handlers.ml_training_local_path.get_file_size_mb') as mock_size:
                mock_size.return_value = 1.5

                session = await mock_state_manager.get_session(12345, "chat_123")
                session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

                # Verify that ApplicationHandlerStop is raised (not caught)
                with pytest.raises(ApplicationHandlerStop):
                    await handler._process_file_path_input(
                        mock_update,
                        mock_context,
                        session,
                        "/tmp/test.csv"
                    )

                # If we get here without ApplicationHandlerStop being raised,
                # it means it was caught by a generic handler (BUG)
                # The pytest.raises above ensures this doesn't happen


class TestControlFlowVsErrorExceptions:
    """Test suite to verify distinction between control flow and error exceptions."""

    @pytest.mark.asyncio
    async def test_application_handler_stop_is_control_flow(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that ApplicationHandlerStop is treated as control flow, not an error.

        Control flow characteristics:
        - Raised intentionally to stop handler propagation
        - Should NOT trigger error messages
        - Should NOT be logged as errors
        - Should be re-raised immediately
        """
        mock_update.message.text = "/tmp/test.csv"

        with patch('src.bot.ml_handlers.ml_training_local_path.validate_local_path') as mock_validate:
            mock_validate.return_value = (True, None, "/tmp/test.csv")

            with patch('src.bot.ml_handlers.ml_training_local_path.get_file_size_mb') as mock_size:
                mock_size.return_value = 1.5

                session = await mock_state_manager.get_session(12345, "chat_123")
                session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

                # Verify control flow: raises without error handling
                with pytest.raises(ApplicationHandlerStop):
                    await handler._process_file_path_input(
                        mock_update,
                        mock_context,
                        session,
                        "/tmp/test.csv"
                    )

                # Verify no error messages sent (only success/info messages)
                for call in mock_update.message.reply_text.call_args_list:
                    message_text = call[0][0] if call[0] else ""
                    # Should not contain generic error indicators
                    if "❌" in message_text:
                        # If ❌ is present, it should be in load options context
                        assert "Load Now" in message_text or "Defer" in message_text

    @pytest.mark.asyncio
    async def test_validation_error_is_actual_error(
        self,
        handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Test that ValidationError is treated as an actual error, not control flow.

        Error characteristics:
        - Indicates something went wrong
        - Should trigger error messages
        - Should be logged as errors
        - Should NOT be re-raised (handled internally)
        """
        schema_text = "invalid"
        mock_update.message.text = schema_text

        session = await mock_state_manager.get_session(12345, "chat_123")
        session.current_state = MLTrainingState.AWAITING_SCHEMA_INPUT.value
        session.file_path = "/tmp/test.csv"
        session.selections = {}

        with patch('src.bot.ml_handlers.ml_training_local_path.SchemaParser') as mock_parser:
            mock_parser.parse.side_effect = ValidationError("Invalid format")

            # Execute: should NOT raise (error handled)
            await handler._process_schema_input(
                mock_update,
                mock_context,
                session,
                schema_text
            )

            # Verify error message sent to user
            assert mock_update.message.reply_text.called

            # Verify it's an actual error message
            message_text = mock_update.message.reply_text.call_args[0][0]
            assert "Invalid" in message_text or "Error" in message_text or "❌" in message_text


@pytest.mark.asyncio
async def test_integration_no_false_errors_on_success(
    handler,
    mock_update,
    mock_context,
    mock_state_manager
):
    """
    Integration test: Verify complete workflow without false error messages.

    User workflow:
    1. User enters valid file path
    2. Path validation succeeds
    3. Load options shown (NO error message)
    4. User selects defer loading
    5. User enters schema
    6. Schema parsing succeeds
    7. Model selection shown (NO error message)

    This is the bug that was fixed: false "Unexpected Error" messages.
    """
    # Step 1: Valid file path input
    mock_update.message.text = "/tmp/test.csv"

    with patch('src.bot.ml_handlers.ml_training_local_path.validate_local_path') as mock_validate:
        mock_validate.return_value = (True, None, "/tmp/test.csv")

        with patch('src.bot.ml_handlers.ml_training_local_path.get_file_size_mb') as mock_size:
            mock_size.return_value = 1.5

            session = await mock_state_manager.get_session(12345, "chat_123")
            session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

            # Path input: should raise ApplicationHandlerStop
            with pytest.raises(ApplicationHandlerStop):
                await handler._process_file_path_input(
                    mock_update,
                    mock_context,
                    session,
                    "/tmp/test.csv"
                )

            # Verify no false error message after path input
            path_error_found = False
            for call in mock_update.message.reply_text.call_args_list:
                message_text = call[0][0] if call[0] else ""
                if "Unexpected Error" in message_text and "/tmp/test.csv" in message_text:
                    path_error_found = True

            assert not path_error_found, "False error message after valid path input (BUG)"

    # Step 2: Schema input
    mock_update.message.reply_text.reset_mock()  # Clear previous calls
    schema_text = "price, sqft, bedrooms"

    session.current_state = MLTrainingState.AWAITING_SCHEMA_INPUT.value
    session.file_path = "/tmp/test.csv"

    with patch('src.bot.ml_handlers.ml_training_local_path.SchemaParser') as mock_parser:
        mock_schema = Mock()
        mock_schema.target = "price"
        mock_schema.features = ["sqft", "bedrooms"]
        mock_schema.format_detected = "comma_separated"
        mock_schema.raw_input = schema_text

        mock_parser.parse = Mock(return_value=mock_schema)

        # Schema input: should raise ApplicationHandlerStop
        with pytest.raises(ApplicationHandlerStop):
            await handler._process_schema_input(
                mock_update,
                mock_context,
                session,
                schema_text
            )

        # Verify no false error message after schema input
        schema_error_found = False
        for call in mock_update.message.reply_text.call_args_list:
            message_text = call[0][0] if call[0] else ""
            if "Unexpected Error" in message_text:
                schema_error_found = True

        assert not schema_error_found, "False error message after valid schema input (BUG)"
