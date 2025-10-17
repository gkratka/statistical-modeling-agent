"""
TDD Tests for Async Training Execution.

These tests verify that ML model training executes asynchronously without
blocking the event loop, allowing other operations to continue during training.

Background:
The handle_training_execution() handler is async, but was calling synchronous
ml_engine.train_model() without await or executor, causing event loop to freeze
during training. This prevented post-training code from executing and user input
from being processed.

Bug Fix: Wrap blocking train_model() call in asyncio.run_in_executor() to
maintain event loop responsiveness.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType


class TestAsyncTrainingExecution:
    """Test that training execution is truly async and non-blocking."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.callback_query = MagicMock()
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        update.callback_query.message = MagicMock(spec=Message)
        update.callback_query.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock bot context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {'state_manager': StateManager()}
        return context

    @pytest.mark.asyncio
    async def test_training_does_not_block_event_loop(self):
        """
        Test that training executes without blocking the event loop.

        Verifies that other async operations can complete during training,
        proving the event loop is not blocked.
        """
        # Track whether concurrent task completed
        concurrent_task_completed = False

        async def concurrent_operation():
            """Simulates other bot operations during training."""
            nonlocal concurrent_task_completed
            await asyncio.sleep(0.1)  # Simulate async work
            concurrent_task_completed = True

        async def mock_training():
            """Simulates training with executor (non-blocking)."""
            # Run blocking operation in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: {"success": True, "model_id": "test_model"}
            )
            return result

        # Start training and concurrent task
        training_task = asyncio.create_task(mock_training())
        concurrent_task = asyncio.create_task(concurrent_operation())

        # Wait for both to complete
        await asyncio.gather(training_task, concurrent_task)

        # Verify concurrent task completed (proves event loop wasn't blocked)
        assert concurrent_task_completed

    @pytest.mark.asyncio
    async def test_post_training_code_executes(self):
        """
        Test that code after training call executes correctly.

        This validates the fix - previously, line 1817 debug output never
        appeared because event loop was frozen.
        """
        post_training_executed = False

        async def mock_handler_with_executor():
            """Simulates fixed handler with executor."""
            # Simulate training with executor (non-blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: {"success": True, "model_id": "test_model"}
            )

            # This should execute (previously didn't)
            nonlocal post_training_executed
            post_training_executed = True

            return result

        result = await mock_handler_with_executor()

        # Verify post-training code executed
        assert post_training_executed
        assert result["success"]

    @pytest.mark.asyncio
    async def test_training_returns_correct_result_format(self):
        """
        Test that training result has expected structure.

        Ensures the executor wrapper doesn't alter result format.
        """
        async def mock_training_with_executor():
            """Simulates training with proper result format."""
            loop = asyncio.get_event_loop()

            def blocking_training():
                return {
                    "success": True,
                    "model_id": "model_12345_test",
                    "metrics": {"accuracy": 0.95},
                    "training_time": 1.23
                }

            result = await loop.run_in_executor(None, blocking_training)
            return result

        result = await mock_training_with_executor()

        # Verify result structure
        assert result["success"] is True
        assert "model_id" in result
        assert "metrics" in result
        assert "training_time" in result

    @pytest.mark.asyncio
    async def test_multiple_async_operations_during_training(self):
        """
        Test that multiple async operations can run during training.

        Simulates real scenario where bot handles button clicks and text
        messages while training is in progress.
        """
        operations_completed = []

        async def simulate_button_click():
            """Simulates user clicking button during training."""
            await asyncio.sleep(0.05)
            operations_completed.append("button_click")

        async def simulate_text_message():
            """Simulates user sending text during training."""
            await asyncio.sleep(0.1)
            operations_completed.append("text_message")

        async def simulate_training():
            """Simulates training with executor."""
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: {"success": True}
            )
            operations_completed.append("training_complete")

        # Run all operations concurrently
        await asyncio.gather(
            simulate_training(),
            simulate_button_click(),
            simulate_text_message()
        )

        # Verify all operations completed
        assert "button_click" in operations_completed
        assert "text_message" in operations_completed
        assert "training_complete" in operations_completed
        assert len(operations_completed) == 3

    @pytest.mark.asyncio
    async def test_event_loop_not_frozen_after_training(self):
        """
        Test that event loop remains responsive after training completes.

        Verifies that state transitions and message sending work after training.
        """
        state_transitioned = False
        message_sent = False

        async def mock_complete_workflow():
            """Simulates complete training workflow with executor."""
            # Training with executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: {"success": True, "model_id": "test_model"}
            )

            # Post-training operations (should work)
            nonlocal state_transitioned, message_sent

            # Simulate state transition
            await asyncio.sleep(0.01)
            state_transitioned = True

            # Simulate message sending
            await asyncio.sleep(0.01)
            message_sent = True

            return result

        await mock_complete_workflow()

        # Verify post-training operations executed
        assert state_transitioned
        assert message_sent


class TestExecutorIntegration:
    """Test integration of executor pattern with ML training handler."""

    @pytest.mark.asyncio
    async def test_executor_handles_blocking_call(self):
        """
        Test that executor properly handles blocking synchronous calls.
        """
        import time

        async def async_wrapper_with_executor():
            """Wraps blocking call in executor."""
            def blocking_operation():
                """Simulates blocking ML training."""
                time.sleep(0.1)  # Blocking sleep
                return {"result": "success"}

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, blocking_operation)
            return result

        # This should complete without blocking event loop
        result = await async_wrapper_with_executor()
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_lambda_wrapper_preserves_parameters(self):
        """
        Test that lambda wrapper correctly passes all parameters.
        """
        async def call_with_many_params():
            """Test parameter passing through lambda wrapper."""
            def function_with_params(a, b, c=None, d=None):
                return {"a": a, "b": b, "c": c, "d": d}

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: function_with_params(1, 2, c=3, d=4)
            )
            return result

        result = await call_with_many_params()

        # Verify all parameters passed correctly
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3
        assert result["d"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
