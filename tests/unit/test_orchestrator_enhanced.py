"""
Unit tests for Enhanced Orchestrator components.

This module tests the enhanced orchestrator functionality including
workflow engine, error recovery, data manager integration, and
enhanced execute_task method with comprehensive edge case coverage.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.orchestrator import (
    TaskOrchestrator,
    WorkflowEngine,
    DataManager,
    ErrorRecoverySystem,
    FeedbackLoop,
    WorkflowState
)
from src.core.parser import TaskDefinition
from src.processors.data_loader import DataLoader
from src.utils.exceptions import ValidationError, DataError, AgentError


class TestTaskOrchestratorEnhanced:
    """Test enhanced TaskOrchestrator functionality."""

    @pytest.fixture
    def mock_data_loader(self):
        """Provide mock DataLoader for testing."""
        loader = MagicMock()
        loader.load_data = AsyncMock(return_value=(
            pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]}),
            {"file_type": "csv", "rows": 3, "columns": 2}
        ))
        return loader

    @pytest.fixture
    def orchestrator(self, mock_data_loader):
        """Provide TaskOrchestrator instance for testing."""
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_data_loader,
            state_ttl_minutes=5
        )

    @pytest.fixture
    def sample_task(self):
        """Provide sample TaskDefinition for testing."""
        return TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["age", "income"], "statistics": ["mean", "median"]},
            data_source=None,
            user_id=12345,
            conversation_id="test_conv_123",
            confidence_score=0.95
        )

    @pytest.fixture
    def sample_data(self):
        """Provide sample DataFrame for testing."""
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "score": [85, 90, 95, 88, 92]
        })

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test enhanced orchestrator initialization."""
        assert orchestrator.state_manager is not None
        assert orchestrator.data_manager is not None
        assert orchestrator.workflow_engine is not None
        assert orchestrator.error_recovery is not None
        assert orchestrator.feedback_loop is not None

    @pytest.mark.asyncio
    async def test_execute_task_with_state_management(self, orchestrator, sample_task, sample_data):
        """Test execute_task with state management integration."""
        # Mock the stats engine to avoid dependency issues
        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_stats.return_value = {
                "age": {"mean": 35.0, "median": 35.0},
                "income": {"mean": 70000.0, "median": 70000.0}
            }

            result = await orchestrator.execute_task(sample_task, sample_data)

            assert result["success"] is True
            assert "workflow_state" in result
            assert "workflow_active" in result
            assert result["workflow_state"] == "idle"

    @pytest.mark.asyncio
    async def test_execute_task_with_cached_data(self, orchestrator, sample_task):
        """Test execute_task loading data from cache."""
        # First, simulate saving data to cache
        state = await orchestrator.state_manager.get_state(sample_task.user_id, sample_task.conversation_id)
        state.data_sources = ["data_12345_test"]

        # Mock cached data
        cached_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        orchestrator.data_manager.data_cache["data_12345_test"] = (cached_data, {"cached": True})

        await orchestrator.state_manager.save_state(state)

        # Mock the stats engine
        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_stats.return_value = {"col1": {"mean": 2.0}}

            result = await orchestrator.execute_task(sample_task)

            assert result["success"] is True
            mock_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_no_data_error(self, orchestrator, sample_task):
        """Test execute_task when no data is available."""
        result = await orchestrator.execute_task(sample_task)

        assert result["success"] is False
        assert result["error_code"] == "NO_DATA"
        assert "No data available" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_task_with_progress_callback(self, orchestrator, sample_task, sample_data):
        """Test execute_task with progress callback."""
        progress_calls = []

        def progress_callback(message):
            progress_calls.append(message)

        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_stats.return_value = {"result": "test"}

            await orchestrator.execute_task(sample_task, sample_data, progress_callback=progress_callback)

            assert len(progress_calls) > 0
            assert "Initializing task execution" in progress_calls
            assert "Validating inputs" in progress_calls

    @pytest.mark.asyncio
    async def test_execute_task_with_timeout(self, orchestrator, sample_task, sample_data):
        """Test execute_task with timeout handling."""
        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            # Simulate a slow operation
            async def slow_operation(*args, **kwargs):
                await asyncio.sleep(2)
                return {"result": "test"}

            mock_stats.side_effect = slow_operation

            result = await orchestrator.execute_task(sample_task, sample_data, timeout=0.1)

            # The error recovery should handle the timeout
            assert "action" in result  # Error recovery response

    @pytest.mark.asyncio
    async def test_execute_with_recovery_retry_logic(self, orchestrator, sample_task, sample_data):
        """Test retry logic in _execute_with_recovery."""
        call_count = 0

        def failing_function(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Temporary network error")
            return {"result": "success"}

        with patch.object(orchestrator, 'ENGINE_ROUTES') as mock_routes:
            mock_routes.__getitem__.return_value = lambda self, task, data: failing_function()

            result = await orchestrator._execute_with_recovery(sample_task, sample_data, 30.0, None)

            assert call_count == 3  # Initial attempt + 2 retries
            assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_is_recoverable_error(self, orchestrator):
        """Test error classification for retry logic."""
        assert orchestrator._is_recoverable_error(ConnectionError("Network error"))
        assert orchestrator._is_recoverable_error(asyncio.TimeoutError())
        assert orchestrator._is_recoverable_error(Exception("connection failed"))
        assert orchestrator._is_recoverable_error(Exception("temporary issue"))

        assert not orchestrator._is_recoverable_error(ValidationError("Invalid input", "field", "value"))
        assert not orchestrator._is_recoverable_error(ValueError("Bad value"))


class TestWorkflowEngineIntegration:
    """Test WorkflowEngine integration with TaskOrchestrator."""

    @pytest.fixture
    def workflow_engine(self):
        """Provide WorkflowEngine for testing."""
        from src.core.orchestrator import StateManager
        state_manager = StateManager(ttl_minutes=5)
        return WorkflowEngine(state_manager)

    @pytest.mark.asyncio
    async def test_start_ml_training_workflow(self, workflow_engine):
        """Test starting ML training workflow."""
        pytest.skip("Workflow engine integration not yet fully implemented")

    @pytest.mark.asyncio
    async def test_advance_workflow_state_transitions(self, workflow_engine):
        """Test workflow state transitions."""
        pytest.skip("Workflow engine integration not yet fully implemented")

    @pytest.mark.asyncio
    async def test_validate_workflow_transitions(self, workflow_engine):
        """Test workflow transition validation."""
        pytest.skip("Workflow engine integration not yet fully implemented")


class TestDataManagerIntegration:
    """Test DataManager integration with TaskOrchestrator."""

    @pytest.fixture
    def data_manager(self):
        """Provide DataManager for testing."""
        mock_loader = MagicMock()
        return DataManager(mock_loader)

    @pytest.mark.asyncio
    async def test_load_and_cache_data(self, data_manager):
        """Test data loading and caching."""
        mock_file = MagicMock()
        mock_file.file_id = "test_file_123"

        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        data_manager.data_loader.load_data.return_value = (test_data, {"rows": 3})

        data_id = await data_manager.load_data(mock_file, "test.csv", 12345)

        assert data_id.startswith("data_12345_")
        assert data_id in data_manager.data_cache

        # Test retrieval
        retrieved_data, metadata = await data_manager.get_data(data_id)
        pd.testing.assert_frame_equal(retrieved_data, test_data)

    @pytest.mark.asyncio
    async def test_validate_data_requirements(self, data_manager):
        """Test data validation against requirements."""
        test_data = pd.DataFrame({"age": [25, 30], "income": [50000, 60000]})
        data_id = "test_data_123"
        data_manager.data_cache[data_id] = (test_data, {"columns": ["age", "income"]})

        # Valid requirements
        valid_requirements = {"required_columns": ["age", "income"]}
        is_valid = await data_manager.validate_data(data_id, valid_requirements)
        assert is_valid is True

        # Invalid requirements
        invalid_requirements = {"required_columns": ["age", "income", "missing_col"]}
        is_valid = await data_manager.validate_data(data_id, invalid_requirements)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_clear_user_cache(self, data_manager):
        """Test clearing cached data for specific user."""
        # Add test data for multiple users
        data_manager.data_cache["data_123_file1"] = (pd.DataFrame(), {})
        data_manager.data_cache["data_123_file2"] = (pd.DataFrame(), {})
        data_manager.data_cache["data_456_file1"] = (pd.DataFrame(), {})

        await data_manager.clear_cache(123)

        # User 123 data should be cleared
        assert "data_123_file1" not in data_manager.data_cache
        assert "data_123_file2" not in data_manager.data_cache

        # User 456 data should remain
        assert "data_456_file1" in data_manager.data_cache


class TestErrorRecoveryIntegration:
    """Test ErrorRecoverySystem integration with TaskOrchestrator."""

    @pytest.fixture
    def error_recovery(self):
        """Provide ErrorRecoverySystem for testing."""
        return ErrorRecoverySystem()

    @pytest.mark.asyncio
    async def test_handle_validation_error(self, error_recovery):
        """Test handling validation errors."""
        error = ValidationError("Invalid column", "column", "invalid_col")
        context = {"task": "test_task", "user_id": 12345}

        result = await error_recovery.handle_error(error, context)

        assert result["action"] == "escalate"
        assert "suggestions" in result
        assert result["error_type"] == "validation"

    @pytest.mark.asyncio
    async def test_handle_network_error_with_retry(self, error_recovery):
        """Test handling network errors with retry logic."""
        error = ConnectionError("Network timeout")
        context = {"retry_attempt": 0}

        result = await error_recovery.handle_error(error, context)

        assert result["action"] == "retry"
        assert result["attempt"] == 1
        assert result["error_type"] == "network"

    @pytest.mark.asyncio
    async def test_error_classification(self, error_recovery):
        """Test error type classification."""
        assert error_recovery._classify_error(asyncio.TimeoutError()) == "timeout"
        assert error_recovery._classify_error(ValidationError("test", "field", "value")) == "validation"
        assert error_recovery._classify_error(DataError("test")) == "data_error"
        assert error_recovery._classify_error(ConnectionError("network error")) == "network"
        assert error_recovery._classify_error(ValueError("unknown")) == "unknown"

    @pytest.mark.asyncio
    async def test_recovery_suggestions(self, error_recovery):
        """Test recovery suggestion generation."""
        data_suggestions = await error_recovery.suggest_recovery("data_error", {})
        assert len(data_suggestions) > 0
        assert any("data format" in suggestion.lower() for suggestion in data_suggestions)

        validation_suggestions = await error_recovery.suggest_recovery("validation", {})
        assert len(validation_suggestions) > 0
        assert any("column names" in suggestion.lower() for suggestion in validation_suggestions)


class TestFeedbackLoopIntegration:
    """Test FeedbackLoop integration with TaskOrchestrator."""

    @pytest.fixture
    def feedback_loop(self):
        """Provide FeedbackLoop for testing."""
        return FeedbackLoop()

    @pytest.mark.asyncio
    async def test_request_clarification(self, feedback_loop):
        """Test clarification request formatting."""
        message = await feedback_loop.request_clarification(
            "unclear column selection",
            ["age", "income", "score"]
        )

        assert "unclear column selection" in message
        assert "age" in message
        assert "income" in message
        assert "score" in message

    @pytest.mark.asyncio
    async def test_show_progress(self, feedback_loop):
        """Test progress message formatting."""
        message = await feedback_loop.show_progress("Training model", 0.75)

        assert "Training model" in message
        assert "75%" in message or "0.75" in message

    @pytest.mark.asyncio
    async def test_confirm_action(self, feedback_loop):
        """Test action confirmation formatting."""
        message = await feedback_loop.confirm_action(
            "Delete all data",
            ["All uploaded data will be removed", "This action cannot be undone"]
        )

        assert "Delete all data" in message
        assert "All uploaded data will be removed" in message
        assert "This action cannot be undone" in message

    @pytest.mark.asyncio
    async def test_suggest_alternatives(self, feedback_loop):
        """Test alternative suggestion formatting."""
        message = await feedback_loop.suggest_alternatives(
            "Complex analysis failed",
            ["Try simpler statistics", "Upload cleaner data", "Use different columns"]
        )

        assert "Complex analysis failed" in message
        assert "Try simpler statistics" in message
        assert "Upload cleaner data" in message
        assert "Use different columns" in message


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def full_orchestrator(self):
        """Provide fully configured orchestrator for integration tests."""
        mock_loader = MagicMock()
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_loader,
            state_ttl_minutes=5
        )

    @pytest.mark.asyncio
    async def test_complete_stats_workflow(self, full_orchestrator):
        """Test complete statistics workflow from start to finish."""
        pytest.skip("Complete workflow integration test not yet implemented")

    @pytest.mark.asyncio
    async def test_ml_training_multi_step_workflow(self, full_orchestrator):
        """Test ML training multi-step workflow."""
        pytest.skip("ML workflow integration test not yet implemented")

    @pytest.mark.asyncio
    async def test_error_recovery_and_retry_scenario(self, full_orchestrator):
        """Test error recovery and retry in real scenario."""
        pytest.skip("Error recovery integration test not yet implemented")

    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, full_orchestrator):
        """Test multiple users with separate workflow states."""
        pytest.skip("Concurrent sessions test not yet implemented")


# Performance and stress tests
class TestPerformanceMetrics:
    """Test performance characteristics of enhanced orchestrator."""

    @pytest.mark.asyncio
    async def test_state_management_performance(self):
        """Test state management performance with large number of states."""
        pytest.skip("Performance testing not yet implemented")

    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test concurrent task execution performance."""
        pytest.skip("Concurrent execution testing not yet implemented")

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large datasets."""
        pytest.skip("Memory usage testing not yet implemented")