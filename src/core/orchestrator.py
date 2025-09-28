"""
Task Orchestrator for the Statistical Modeling Agent.

This module coordinates task execution by routing TaskDefinition objects
to appropriate engines and managing the complete execution lifecycle.
"""

import time
import logging
from typing import Dict, Any, Optional
import pandas as pd
import asyncio

from src.core.parser import TaskDefinition
from src.engines.stats_engine import StatsEngine
from src.utils.exceptions import ValidationError, DataError, AgentError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskOrchestrator:
    """
    Central orchestrator for task execution and engine coordination.

    Routes TaskDefinition objects to appropriate engines, manages execution
    lifecycle, and standardizes results across different engine types.
    """

    # Engine routing mapping
    ENGINE_ROUTES = {
        "stats": lambda self, task, data: self._execute_stats_task(task, data)
    }

    # Engine name mapping
    ENGINE_NAMES = {
        "stats": "StatsEngine",
        "ml_train": "MLTrainingEngine",
        "ml_score": "MLScoringEngine",
        "data_info": "DataInfoEngine"
    }

    def __init__(self, enable_logging: bool = True) -> None:
        """Initialize the task orchestrator."""
        self.stats_engine = StatsEngine()
        self.enable_logging = enable_logging
        self.logger = logger
        if enable_logging:
            logger.info("TaskOrchestrator initialized with StatsEngine")

    async def execute_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame,
        timeout: Optional[float] = 30.0
    ) -> Dict[str, Any]:
        """
        Main execution method for task routing and processing.

        Args:
            task: TaskDefinition object with task details
            data: pandas DataFrame for analysis
            timeout: Maximum execution time in seconds

        Returns:
            Standardized result dictionary with execution metadata

        Raises:
            ValidationError: If task validation fails
            DataError: If data processing fails
            AgentError: If execution fails unexpectedly
        """
        start_time = time.time()

        try:
            # Validate inputs
            self._validate_task(task)
            self._validate_data(data)

            if self.enable_logging:
                self.logger.info(f"Executing task: {task.task_type}/{task.operation} for user {task.user_id}")

            # Route to appropriate engine
            if task.task_type in self.ENGINE_ROUTES:
                result = await asyncio.wait_for(
                    self.ENGINE_ROUTES[task.task_type](self, task, data),
                    timeout=timeout
                )
            elif task.task_type in ["ml_train", "ml_score"]:
                raise ValidationError(
                    f"{task.task_type.replace('_', ' ').title()} tasks not yet implemented",
                    field="task_type", value=task.task_type
                )
            else:
                raise ValidationError(
                    f"Unsupported task type: {task.task_type}",
                    field="task_type", value=task.task_type
                )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Build standardized result
            standardized_result = {
                "success": True,
                "task_type": task.task_type,
                "operation": task.operation,
                "result": result,
                "metadata": self._build_metadata(task, data, execution_time)
            }

            if self.enable_logging:
                self.logger.info(f"Task completed successfully in {execution_time:.4f}s")

            return standardized_result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Task execution timed out after {timeout}s"
            self.logger.error(error_msg)

            return self._build_error_result(
                task, error_msg, "TIMEOUT_ERROR", execution_time
            )

        except (ValidationError, DataError) as e:
            execution_time = time.time() - start_time
            self.logger.warning(f"Task validation/data error: {e.message}")

            return self._build_error_result(
                task, e.message, e.error_code, execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error in task execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return self._build_error_result(
                task, error_msg, "EXECUTION_ERROR", execution_time
            )

    async def _execute_stats_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute statistical analysis tasks using StatsEngine.

        Args:
            task: TaskDefinition for statistics operation
            data: DataFrame for analysis

        Returns:
            Statistics results from engine
        """
        try:
            # Execute through stats engine synchronously
            # Note: StatsEngine.execute is not async, so we run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.stats_engine.execute,
                task,
                data
            )

            return result

        except Exception as e:
            # Re-raise with context
            if isinstance(e, (ValidationError, DataError)):
                raise
            else:
                raise AgentError(f"Stats engine execution failed: {str(e)}")

    def _validate_task(self, task: TaskDefinition) -> None:
        """Validate TaskDefinition object."""
        validations = [
            (not isinstance(task, TaskDefinition), "Invalid task type, expected TaskDefinition", "task", str(type(task))),
            (not task.task_type, "Task type cannot be empty", "task_type", ""),
            (not task.operation, "Task operation cannot be empty", "operation", "")
        ]

        for condition, message, field, value in validations:
            if condition:
                raise ValidationError(message, field=field, value=value)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame", field="data", value=str(type(data)))
        if data.empty or len(data) == 0:
            raise DataError("DataFrame cannot be empty", data_shape=data.shape if not data.empty else (0, 0))

    def _build_metadata(self, task: TaskDefinition, data: pd.DataFrame, execution_time: float) -> Dict[str, Any]:
        """Build execution metadata."""
        return {
            "execution_time": round(execution_time, 4),
            "data_shape": data.shape,
            "user_id": task.user_id,
            "conversation_id": task.conversation_id,
            "confidence_score": task.confidence_score,
            "timestamp": time.time(),
            "engine_used": self.ENGINE_NAMES.get(task.task_type, "UnknownEngine")
        }

    def _get_engine_name(self, task_type: str) -> str:
        """Get engine name for task type."""
        return self.ENGINE_NAMES.get(task_type, "UnknownEngine")

    def _build_error_result(
        self,
        task: TaskDefinition,
        error_message: str,
        error_code: str,
        execution_time: float
    ) -> Dict[str, Any]:
        """Build standardized error result."""
        return {
            "success": False,
            "error": error_message,
            "error_code": error_code,
            "task_type": task.task_type,
            "operation": task.operation,
            "metadata": {
                "execution_time": round(execution_time, 4),
                "user_id": task.user_id,
                "conversation_id": task.conversation_id,
                "timestamp": time.time(),
                "engine_attempted": self._get_engine_name(task.task_type)
            }
        }

    def get_supported_task_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported task types and operations.

        Returns:
            Dictionary mapping task types to their capabilities
        """
        return {
            "stats": {
                "engine": "StatsEngine",
                "operations": [
                    "descriptive_stats",
                    "correlation_analysis",
                    "mean_analysis",
                    "median_analysis",
                    "std_analysis"
                ],
                "status": "available",
                "description": "Statistical analysis and descriptive statistics"
            },
            "ml_train": {
                "engine": "MLTrainingEngine",
                "operations": ["train_model"],
                "status": "planned",
                "description": "Machine learning model training"
            },
            "ml_score": {
                "engine": "MLScoringEngine",
                "operations": ["predict", "score"],
                "status": "planned",
                "description": "Machine learning predictions and scoring"
            },
            "data_info": {
                "engine": "DataInfoEngine",
                "operations": ["describe_data", "summarize"],
                "status": "planned",
                "description": "Data exploration and information"
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on orchestrator and engines.

        Returns:
            Health status information
        """
        health_status = {
            "orchestrator": "healthy",
            "engines": {},
            "timestamp": time.time()
        }

        # Check StatsEngine health
        try:
            # Simple test with minimal data
            test_data = pd.DataFrame({"test": [1, 2, 3]})
            test_task = TaskDefinition(
                task_type="stats",
                operation="descriptive_stats",
                parameters={"columns": ["test"]},
                data_source=None,
                user_id=0,
                conversation_id="health_check"
            )

            start_time = time.time()
            result = await self.execute_task(test_task, test_data, timeout=5.0)
            health_time = time.time() - start_time

            health_status["engines"]["stats"] = {
                "status": "healthy" if result["success"] else "error",
                "response_time": round(health_time, 4),
                "last_check": time.time()
            }

        except Exception as e:
            health_status["engines"]["stats"] = {
                "status": "error",
                "error": str(e),
                "last_check": time.time()
            }

        return health_status


# Convenience functions for backward compatibility
async def execute_stats_task(task: TaskDefinition, data: pd.DataFrame) -> Dict[str, Any]:
    """Execute a statistics task using the orchestrator."""
    orchestrator = TaskOrchestrator()
    return await orchestrator.execute_task(task, data)


async def get_orchestrator_health() -> Dict[str, Any]:
    """Get orchestrator health status."""
    orchestrator = TaskOrchestrator()
    return await orchestrator.health_check()