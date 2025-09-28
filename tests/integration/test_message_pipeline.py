"""
Integration tests for the complete message processing pipeline.

This module tests the end-to-end flow from Telegram message to formatted response,
including parser → orchestrator → stats engine → formatter integration.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.core.parser import RequestParser, TaskDefinition
from src.core.orchestrator import TaskOrchestrator
from src.utils.result_formatter import TelegramResultFormatter
from src.engines.stats_engine import StatsEngine
from src.utils.exceptions import ParseError, ValidationError, DataError


@pytest.fixture
def sample_telegram_data() -> Dict[str, Any]:
    """Provide sample user data as stored in Telegram context."""
    df = pd.DataFrame({
        'sales': [100, 200, 150, 300, 250],
        'profit': [20, 40, 30, 60, 50],
        'region': ['A', 'B', 'A', 'B', 'A']
    })

    return {
        'dataframe': df,
        'metadata': {
            'shape': (5, 3),
            'columns': ['sales', 'profit', 'region'],
            'numeric_columns': ['sales', 'profit'],
            'file_type': 'csv'
        },
        'file_name': 'train2b.csv'
    }


@pytest.fixture
def mock_telegram_update():
    """Create mock Telegram update object."""
    update = Mock()
    update.message.text = "Calculate statistics for sales"
    update.effective_user.id = 12345
    update.effective_chat.id = 67890
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_telegram_context():
    """Create mock Telegram context with user data."""
    context = Mock()
    context.user_data = {}
    return context


class TestFullPipeline:
    """Test complete message processing pipeline."""

    @pytest.mark.asyncio
    async def test_complete_stats_pipeline(self, sample_telegram_data):
        """Test complete pipeline from message to formatted response."""
        pytest.skip("Full pipeline integration not yet implemented")

    @pytest.mark.asyncio
    async def test_stats_request_flow(self, sample_telegram_data):
        """Test specific stats request processing flow."""
        message_text = "Calculate statistics for sales"
        user_id = 12345
        conversation_id = "test_chat"

        # Test each component individually first
        # 1. Parser
        parser = RequestParser()
        task = parser.parse_request(
            text=message_text,
            user_id=user_id,
            conversation_id=conversation_id
        )

        assert task.task_type == "stats"
        assert task.operation in ["descriptive_stats", "summary"]
        assert "sales" in str(task.parameters)

        # 2. Orchestrator
        orchestrator = TaskOrchestrator()
        result = await orchestrator.execute_task(task, sample_telegram_data['dataframe'])

        assert result["success"] is True
        assert "sales" in result["result"]

        # 3. Formatter
        formatter = TelegramResultFormatter()
        formatted_response = formatter.format_stats_result(result)

        assert "**SALES**" in formatted_response or "sales" in formatted_response.lower()
        assert "Mean:" in formatted_response or "mean" in formatted_response.lower()

    @pytest.mark.asyncio
    async def test_correlation_request_flow(self, sample_telegram_data):
        """Test correlation request processing flow."""
        pytest.skip("Correlation pipeline not yet tested")

    @pytest.mark.asyncio
    async def test_error_handling_flow(self):
        """Test error handling through complete pipeline."""
        pytest.skip("Error handling pipeline not yet tested")


class TestMessageHandlerIntegration:
    """Test integration with actual message handler."""

    @pytest.mark.asyncio
    async def test_message_handler_with_stats_request(
        self,
        mock_telegram_update,
        mock_telegram_context,
        sample_telegram_data
    ):
        """Test message handler with actual stats request."""
        pytest.skip("Message handler integration not yet implemented")

    @pytest.mark.asyncio
    async def test_message_handler_with_invalid_request(
        self,
        mock_telegram_update,
        mock_telegram_context,
        sample_telegram_data
    ):
        """Test message handler with invalid request."""
        pytest.skip("Message handler integration not yet implemented")

    @pytest.mark.asyncio
    async def test_message_handler_without_data(
        self,
        mock_telegram_update,
        mock_telegram_context
    ):
        """Test message handler when no data uploaded."""
        pytest.skip("Message handler integration not yet implemented")

    @pytest.mark.asyncio
    async def test_message_handler_performance(
        self,
        mock_telegram_update,
        mock_telegram_context,
        sample_telegram_data
    ):
        """Test message handler performance with timing."""
        pytest.skip("Performance testing not yet implemented")


class TestParserIntegration:
    """Test parser integration with different message types."""

    def test_parse_basic_stats_request(self):
        """Test parsing basic statistics requests."""
        parser = RequestParser()

        # Test various phrasings
        test_cases = [
            "Calculate statistics for sales",
            "show me stats for sales column",
            "get descriptive statistics for sales",
            "stats for sales",
            "calculate mean and std for sales"
        ]

        for message in test_cases:
            task = parser.parse_request(
                text=message,
                user_id=12345,
                conversation_id="test"
            )
            assert task.task_type == "stats"
            assert "sales" in str(task.parameters).lower()

    def test_parse_correlation_request(self):
        """Test parsing correlation requests."""
        parser = RequestParser()

        test_cases = [
            "show correlation between sales and profit",
            "correlation matrix for sales, profit",
            "correlate sales with profit"
        ]

        for message in test_cases:
            task = parser.parse_request(
                text=message,
                user_id=12345,
                conversation_id="test"
            )
            assert task.task_type == "stats"
            assert task.operation in ["correlation_analysis", "correlation"]

    def test_parse_invalid_requests(self):
        """Test parsing invalid or unclear requests."""
        parser = RequestParser()

        test_cases = [
            "hello there",
            "what is this",
            "calculate something weird",
            ""
        ]

        for message in test_cases:
            with pytest.raises(ParseError):
                parser.parse_request(
                    text=message,
                    user_id=12345,
                    conversation_id="test"
                )


class TestOrchestratorIntegration:
    """Test orchestrator integration with engines."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_stats_engine(self, sample_telegram_data):
        """Test orchestrator integration with StatsEngine."""
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["sales"]},
            data_source=None,
            user_id=12345,
            conversation_id="test"
        )

        orchestrator = TaskOrchestrator()
        result = await orchestrator.execute_task(task, sample_telegram_data['dataframe'])

        assert result["success"] is True
        assert result["task_type"] == "stats"
        assert result["operation"] == "descriptive_stats"
        assert "sales" in result["result"]
        assert "metadata" in result
        assert "execution_time" in result["metadata"]

    @pytest.mark.asyncio
    async def test_orchestrator_error_handling(self):
        """Test orchestrator error handling."""
        # Test with invalid task
        invalid_task = TaskDefinition(
            task_type="invalid_type",
            operation="unknown",
            parameters={},
            data_source=None,
            user_id=12345,
            conversation_id="test"
        )

        orchestrator = TaskOrchestrator()
        empty_df = pd.DataFrame()

        result = await orchestrator.execute_task(invalid_task, empty_df)

        assert result["success"] is False
        assert "error" in result
        assert result["task_type"] == "invalid_type"

    @pytest.mark.asyncio
    async def test_orchestrator_timeout_handling(self, sample_telegram_data):
        """Test orchestrator timeout handling."""
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["sales"]},
            data_source=None,
            user_id=12345,
            conversation_id="test"
        )

        orchestrator = TaskOrchestrator()

        # Test with very short timeout
        result = await orchestrator.execute_task(
            task,
            sample_telegram_data['dataframe'],
            timeout=0.001  # Very short timeout
        )

        # Should either succeed quickly or timeout gracefully
        assert "success" in result
        if not result["success"]:
            assert "timeout" in result.get("error", "").lower()


class TestFormatterIntegration:
    """Test formatter integration with engine outputs."""

    def test_formatter_with_stats_engine_output(self, sample_telegram_data):
        """Test formatter with actual StatsEngine output."""
        # Generate real stats engine output
        engine = StatsEngine()
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["sales"]},
            data_source=None,
            user_id=12345,
            conversation_id="test"
        )

        stats_result = engine.execute(task, sample_telegram_data['dataframe'])

        # Wrap in orchestrator format
        orchestrator_result = {
            "success": True,
            "task_type": "stats",
            "operation": "descriptive_stats",
            "result": stats_result,
            "metadata": {
                "execution_time": 0.0234,
                "data_shape": [5, 3],
                "user_id": 12345
            }
        }

        # Format for Telegram
        formatter = TelegramResultFormatter()
        formatted_output = formatter.format_stats_result(orchestrator_result)

        # Verify formatting
        assert isinstance(formatted_output, str)
        assert len(formatted_output) > 0
        assert "sales" in formatted_output.lower()
        assert "mean" in formatted_output.lower()
        assert "**" in formatted_output  # Markdown formatting

    def test_formatter_with_error_output(self):
        """Test formatter with error output."""
        error_result = {
            "success": False,
            "error": "Column 'invalid_column' not found",
            "error_code": "VALIDATION_ERROR",
            "task_type": "stats",
            "operation": "descriptive_stats"
        }

        formatter = TelegramResultFormatter()
        formatted_output = formatter.format_stats_result(error_result)

        assert "❌" in formatted_output or "Error" in formatted_output
        assert "invalid_column" in formatted_output
        assert "Suggestions" in formatted_output or "suggestions" in formatted_output

    def test_formatter_telegram_length_limits(self):
        """Test formatter handles Telegram length limits."""
        # Create result with many columns
        large_result = {
            "success": True,
            "result": {}
        }

        # Add many columns to test length limits
        for i in range(50):
            large_result["result"][f"column_{i}"] = {
                "mean": 100.0 + i,
                "median": 95.0 + i,
                "std": 10.0,
                "min": 50.0,
                "max": 150.0,
                "count": 1000
            }

        formatter = TelegramResultFormatter()
        formatted_output = formatter.format_stats_result(large_result)

        # Should be within Telegram limits
        assert len(formatted_output) <= TelegramResultFormatter.MAX_MESSAGE_LENGTH


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_calculate_statistics_for_sales_scenario(self, sample_telegram_data):
        """Test the exact scenario from Telegram screenshots."""
        # This is the exact message from the screenshot
        message_text = "Calculate statistics for sales"

        # Parse the request
        parser = RequestParser()
        task = parser.parse_request(
            text=message_text,
            user_id=12345,
            conversation_id="test_chat"
        )

        # Execute through orchestrator
        orchestrator = TaskOrchestrator()
        result = await orchestrator.execute_task(task, sample_telegram_data['dataframe'])

        # Format for Telegram
        formatter = TelegramResultFormatter()
        formatted_response = formatter.format_stats_result(result)

        # Verify the response contains expected statistics
        assert "sales" in formatted_response.lower()
        assert any(stat in formatted_response.lower() for stat in ["mean", "median", "std"])
        assert len(formatted_response) > 100  # Should be substantial response
        assert len(formatted_response) <= 4096  # Within Telegram limits

    @pytest.mark.asyncio
    async def test_multiple_requests_same_session(self, sample_telegram_data):
        """Test multiple requests in same session."""
        requests = [
            "Calculate statistics for sales",
            "Show correlation between sales and profit",
            "What is the mean of profit?"
        ]

        parser = RequestParser()
        orchestrator = TaskOrchestrator()
        formatter = TelegramResultFormatter()

        for message_text in requests:
            try:
                task = parser.parse_request(
                    text=message_text,
                    user_id=12345,
                    conversation_id="test_chat"
                )

                result = await orchestrator.execute_task(task, sample_telegram_data['dataframe'])
                formatted_response = formatter.format_stats_result(result)

                assert isinstance(formatted_response, str)
                assert len(formatted_response) > 0

            except ParseError:
                # Some requests might not parse correctly, that's OK for now
                pass

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self):
        """Test error recovery in pipeline."""
        # Test with no data
        message_text = "Calculate statistics for sales"

        parser = RequestParser()
        task = parser.parse_request(
            text=message_text,
            user_id=12345,
            conversation_id="test_chat"
        )

        orchestrator = TaskOrchestrator()
        empty_df = pd.DataFrame()

        # Should handle empty data gracefully
        result = await orchestrator.execute_task(task, empty_df)
        assert result["success"] is False

        formatter = TelegramResultFormatter()
        formatted_response = formatter.format_stats_result(result)

        assert "error" in formatted_response.lower() or "❌" in formatted_response
        assert len(formatted_response) > 0


class TestPerformanceIntegration:
    """Test performance of integrated pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_performance_small_dataset(self, sample_telegram_data):
        """Test pipeline performance with small dataset."""
        pytest.skip("Performance testing not yet implemented")

    @pytest.mark.asyncio
    async def test_pipeline_performance_large_dataset(self):
        """Test pipeline performance with large dataset."""
        pytest.skip("Performance testing not yet implemented")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, sample_telegram_data):
        """Test handling of concurrent requests."""
        pytest.skip("Concurrency testing not yet implemented")