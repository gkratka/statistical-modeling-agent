"""
Enhanced Task Orchestrator for the Statistical Modeling Agent.

This module coordinates task execution by routing TaskDefinition objects
to appropriate engines, managing conversation state, multi-step workflows,
and providing intelligent error handling with user feedback.
"""

import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from src.core.parser import TaskDefinition
from src.engines.stats_engine import StatsEngine
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.processors.data_loader import DataLoader
from src.generators.script_generator import ScriptGenerator
from src.execution.executor import ScriptExecutor, SandboxConfig
from src.utils.exceptions import ValidationError, DataError, AgentError, ScriptGenerationError, ExecutionError
from src.utils.logger import get_logger
from src.core.state_manager import (
    StateManager,
    StateManagerConfig,
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)

logger = get_logger(__name__)


class DataManager:
    """Manages data lifecycle and coordination with DataLoader."""

    def __init__(self, data_loader: DataLoader) -> None:
        """Initialize data manager with DataLoader instance."""
        self.data_loader = data_loader
        self.data_cache: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}
        self.logger = logger

    async def load_data(
        self,
        telegram_file,
        file_name: str,
        file_size: int,
        user_id: int,
        context
    ) -> str:
        """Load data through DataLoader and cache it."""
        try:
            # Use DataLoader to process file
            dataframe, metadata = await self.data_loader.load_from_telegram(
                telegram_file, file_name, file_size, context
            )

            # Generate data ID
            data_id = f"data_{user_id}_{int(time.time())}"

            # Cache the data
            await self.cache_data(data_id, dataframe, metadata)

            self.logger.info(f"Loaded and cached data: {data_id} ({dataframe.shape})")
            return data_id

        except Exception as e:
            self.logger.error(f"Failed to load data for user {user_id}: {e}")
            raise DataError(f"Data loading failed: {str(e)}")

    async def get_data(self, data_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Retrieve cached data by ID."""
        if data_id not in self.data_cache:
            raise DataError(f"Data not found: {data_id}")

        return self.data_cache[data_id]

    async def validate_data(self, data_id: str, requirements: Dict[str, Any]) -> bool:
        """Validate data against requirements."""
        if data_id not in self.data_cache:
            return False

        dataframe, metadata = self.data_cache[data_id]

        # Check minimum rows
        min_rows = requirements.get("min_rows", 1)
        if len(dataframe) < min_rows:
            return False

        # Check required columns
        required_columns = requirements.get("required_columns", [])
        if not all(col in dataframe.columns for col in required_columns):
            return False

        # Check numeric columns for ML tasks
        if requirements.get("require_numeric"):
            numeric_columns = dataframe.select_dtypes(include=['number']).columns
            if len(numeric_columns) < requirements.get("min_numeric_columns", 1):
                return False

        return True

    async def cache_data(
        self,
        data_id: str,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> None:
        """Cache dataframe with metadata."""
        self.data_cache[data_id] = (dataframe.copy(), metadata.copy())

    async def clear_cache(self, user_id: int) -> None:
        """Clear cached data for specific user."""
        user_data_ids = [
            data_id for data_id in self.data_cache.keys()
            if data_id.startswith(f"data_{user_id}_")
        ]

        for data_id in user_data_ids:
            del self.data_cache[data_id]

        if user_data_ids:
            self.logger.info(f"Cleared {len(user_data_ids)} cached datasets for user {user_id}")


class WorkflowEngine:
    """
    Simplified workflow engine that delegates to StateManager.

    Provides compatibility layer for existing orchestrator code while
    using the new StateManager for actual state management.
    """

    def __init__(self, state_manager: StateManager) -> None:
        """Initialize workflow engine with new StateManager."""
        self.state_manager = state_manager
        self.logger = logger

    async def start_ml_training(self, session: UserSession) -> str:
        """Start ML training workflow."""
        await self.state_manager.start_workflow(
            session,
            WorkflowType.ML_TRAINING
        )
        return "ML training workflow started. Please upload your training data."

    async def start_ml_prediction(self, session: UserSession) -> str:
        """Start ML prediction workflow."""
        await self.state_manager.start_workflow(
            session,
            WorkflowType.ML_PREDICTION
        )
        return "ML prediction workflow started. Please select a trained model."

    async def start_stats_analysis(self, session: UserSession) -> str:
        """Start stats analysis workflow."""
        await self.state_manager.start_workflow(
            session,
            WorkflowType.STATS_ANALYSIS
        )
        return "Stats analysis workflow started. Please upload your data."

    async def advance_training_workflow(
        self,
        session: UserSession,
        new_state: str
    ) -> Tuple[bool, Optional[str], List[str]]:
        """Advance ML training workflow to next state."""
        return await self.state_manager.transition_state(session, new_state)

    async def cancel_workflow(self, session: UserSession) -> None:
        """Cancel active workflow."""
        await self.state_manager.cancel_workflow(session)
        self.logger.info(f"Cancelled workflow for user {session.user_id}")


class ErrorRecoverySystem:
    """Intelligent error handling with retry strategies."""

    def __init__(self, max_retries: int = 3) -> None:
        """Initialize error recovery system."""
        self.max_retries = max_retries
        self.retry_strategies = self._init_strategies()
        self.logger = logger

    def _init_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize retry strategies for different error types."""
        return {
            "network": {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "base_delay": 1.0
            },
            "timeout": {
                "max_retries": 2,
                "backoff_factor": 1.5,
                "base_delay": 2.0
            },
            "data_error": {
                "max_retries": 1,
                "backoff_factor": 1.0,
                "base_delay": 0.5
            },
            "validation": {
                "max_retries": 0,  # No retry for validation errors
                "backoff_factor": 1.0,
                "base_delay": 0.0
            }
        }

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle error with appropriate recovery strategy."""
        error_type = self._classify_error(error)
        strategy = self.retry_strategies.get(error_type, self.retry_strategies["data_error"])

        # Get retry attempt from context
        retry_attempt = context.get("retry_attempt", 0)

        if retry_attempt < strategy["max_retries"]:
            # Attempt retry
            delay = strategy["base_delay"] * (strategy["backoff_factor"] ** retry_attempt)

            self.logger.warning(f"Error {error_type}, retrying in {delay}s (attempt {retry_attempt + 1})")
            await asyncio.sleep(delay)

            return {
                "action": "retry",
                "delay": delay,
                "attempt": retry_attempt + 1,
                "error_type": error_type
            }
        else:
            # Escalate to user
            suggestions = await self.suggest_recovery(error_type, context)
            user_message = await self.escalate_to_user(error, suggestions)

            return {
                "success": False,
                "error": str(error),
                "error_code": error_type.upper(),
                "action": "escalate",
                "message": user_message,
                "suggestions": suggestions,
                "error_type": error_type
            }

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling."""
        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        elif isinstance(error, ValidationError):
            return "validation"
        elif isinstance(error, DataError):
            return "data_error"
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            return "network"
        else:
            return "unknown"

    async def suggest_recovery(
        self,
        error_type: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate recovery suggestions."""
        if error_type == "data_error":
            return [
                "Check your data format (CSV, Excel)",
                "Ensure data has proper column headers",
                "Verify data contains numeric values for analysis"
            ]
        elif error_type == "validation":
            return [
                "Check column names match your data",
                "Verify data types are appropriate",
                "Try simpler analysis first"
            ]
        elif error_type == "timeout":
            return [
                "Try with smaller dataset",
                "Use simpler analysis options",
                "Check internet connection"
            ]
        else:
            return [
                "Try uploading data again",
                "Contact support if problem persists"
            ]

    async def escalate_to_user(
        self,
        error: Exception,
        suggestions: List[str]
    ) -> str:
        """Create user-friendly error message."""
        message = f"❌ **Operation Failed**\n\n"
        message += f"**Error**: {str(error)}\n\n"
        message += "**Suggestions**:\n"

        for i, suggestion in enumerate(suggestions, 1):
            message += f"{i}. {suggestion}\n"

        return message

    async def retry_with_backoff(
        self,
        operation: Callable,
        max_attempts: int,
        error_type: str = "unknown"
    ) -> Any:
        """Execute operation with exponential backoff."""
        strategy = self.retry_strategies.get(error_type, self.retry_strategies["data_error"])

        for attempt in range(max_attempts):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e

                delay = strategy["base_delay"] * (strategy["backoff_factor"] ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s")
                await asyncio.sleep(delay)


class FeedbackLoop:
    """Manages user feedback and clarification requests."""

    def __init__(self, formatter=None) -> None:
        """Initialize feedback loop."""
        self.formatter = formatter
        self.logger = logger

    async def request_clarification(
        self,
        ambiguous_input: str,
        options: List[str]
    ) -> str:
        """Request clarification from user."""
        message = f"🤔 **Need Clarification**\n\n"
        message += f"Your request: \"{ambiguous_input}\"\n\n"
        message += "**Please choose:**\n"

        for i, option in enumerate(options, 1):
            message += f"{i}. {option}\n"

        return message

    async def show_progress(
        self,
        operation: str,
        progress: float
    ) -> str:
        """Show operation progress."""
        percentage = int(progress * 100)
        progress_bar = "█" * (percentage // 10) + "░" * (10 - percentage // 10)

        return f"⏳ **{operation}**\n`{progress_bar}` {percentage}%"

    async def confirm_action(
        self,
        action: str,
        consequences: List[str]
    ) -> str:
        """Request action confirmation."""
        message = f"⚠️ **Confirm Action**\n\n"
        message += f"**Action**: {action}\n\n"
        message += "**This will**:\n"

        for consequence in consequences:
            message += f"• {consequence}\n"

        message += "\nReply 'yes' to confirm or 'no' to cancel."
        return message

    async def suggest_alternatives(
        self,
        failed_action: str,
        alternatives: List[str]
    ) -> str:
        """Suggest alternative actions."""
        message = f"💡 **Alternative Suggestions**\n\n"
        message += f"Since \"{failed_action}\" didn't work, try:\n\n"

        for i, alternative in enumerate(alternatives, 1):
            message += f"{i}. {alternative}\n"

        return message


class TaskOrchestrator:
    """
    Enhanced central orchestrator for task execution, state management, and workflow coordination.

    Manages TaskDefinition objects routing to appropriate engines while maintaining
    conversation state, coordinating multi-step workflows, and providing intelligent
    error handling with user feedback.
    """

    # Engine routing mapping
    ENGINE_ROUTES = {
        "stats": lambda self, task, data: self._execute_stats_task(task, data),
        "script": lambda self, task, data: self._execute_script_task(task, data),
        "ml_train": lambda self, task, data: self._execute_ml_train_task(task, data),
        "ml_score": lambda self, task, data: self._execute_ml_score_task(task, data)
    }

    # Engine name mapping
    ENGINE_NAMES = {
        "stats": "StatsEngine",
        "script": "ScriptEngine",
        "ml_train": "MLEngine",
        "ml_score": "MLEngine",
        "data_info": "DataInfoEngine"
    }

    def __init__(
        self,
        enable_logging: bool = True,
        data_loader: Optional[DataLoader] = None,
        state_config: Optional[StateManagerConfig] = None,
        ml_config: Optional[MLEngineConfig] = None
    ) -> None:
        """Initialize the enhanced task orchestrator."""
        # Core engines
        self.stats_engine = StatsEngine()
        self.ml_engine = MLEngine(ml_config or MLEngineConfig.get_default())
        self.enable_logging = enable_logging
        self.logger = logger

        # Enhanced components - use new StateManager
        self.state_manager = StateManager(config=state_config or StateManagerConfig())
        self.data_manager = DataManager(data_loader or DataLoader())
        self.workflow_engine = WorkflowEngine(self.state_manager)
        self.error_recovery = ErrorRecoverySystem()
        self.feedback_loop = FeedbackLoop()

        # Background cleanup task
        self._cleanup_task = None
        if enable_logging:
            logger.info("Enhanced TaskOrchestrator initialized with state management, workflow support, and ML Engine")

    async def execute_task(
        self,
        task: TaskDefinition,
        data: Optional[pd.DataFrame] = None,
        timeout: Optional[float] = 30.0,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Enhanced task execution with state management and workflow support.

        Args:
            task: TaskDefinition object with task details
            data: Optional DataFrame (can be loaded from state if not provided)
            timeout: Maximum execution time
            progress_callback: Optional callback for progress updates

        Returns:
            Enhanced result dictionary with workflow state and next steps
        """
        start_time = time.time()

        try:
            # Get or create session (new StateManager API)
            session = await self.state_manager.get_or_create_session(
                task.user_id,
                task.conversation_id
            )

            if progress_callback:
                progress_callback("Initializing task execution")

            # Check if this is part of a multi-step workflow (new API)
            if session.workflow_type is not None:
                if self.enable_logging:
                    self.logger.info(f"Continuing workflow {session.workflow_type.value} for user {task.user_id}")

                # For now, just log - actual workflow handling will be in WorkflowHandler
                # This maintains compatibility while transitioning to new architecture

            # Handle data loading if needed (use session's uploaded_data)
            if data is None and session.uploaded_data is not None:
                if progress_callback:
                    progress_callback("Loading cached data")

                # Use data from session
                data = session.uploaded_data
                if self.enable_logging:
                    self.logger.info(f"Using session data: {data.shape}")

            # Validate inputs
            if progress_callback:
                progress_callback("Validating inputs")

            self._validate_task(task)
            if data is not None:
                self._validate_data(data)

            if self.enable_logging:
                self.logger.info(f"Executing enhanced task: {task.task_type}/{task.operation} for user {task.user_id}")

            # Execute with enhanced error recovery
            result = await self._execute_with_recovery(task, data, timeout, progress_callback)

            # Return result directly (state management moved to workflow handlers)
            return result

        except Exception as e:
            # Enhanced error handling with recovery suggestions
            error_context = {
                "task": task,
                "data_shape": data.shape if data is not None else None,
                "user_id": task.user_id,
                "conversation_id": task.conversation_id,
                "execution_time": time.time() - start_time
            }

            return await self.error_recovery.handle_error(e, error_context)

    async def _execute_with_recovery(
        self,
        task: TaskDefinition,
        data: Optional[pd.DataFrame],
        timeout: Optional[float],
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute task with retry logic and error recovery."""
        retry_attempt = 0
        max_retries = 3

        while retry_attempt <= max_retries:
            try:
                # Route to appropriate engine
                if task.task_type in self.ENGINE_ROUTES:
                    if progress_callback:
                        progress_callback(f"Executing {task.operation}")

                    result = await asyncio.wait_for(
                        self.ENGINE_ROUTES[task.task_type](self, task, data),
                        timeout=timeout
                    )

                    return result

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

            except Exception as e:
                # Handle recoverable errors with retry logic
                if retry_attempt < max_retries and self._is_recoverable_error(e):
                    retry_attempt += 1
                    delay = 2.0 ** (retry_attempt - 1)  # Exponential backoff

                    if self.enable_logging:
                        self.logger.warning(f"Retrying task after error (attempt {retry_attempt}): {e}")

                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-recoverable error or max retries reached
                    if self.enable_logging:
                        self.logger.error(f"Task execution failed after {retry_attempt} retries: {str(e)}", exc_info=True)
                    raise e

        # This should never be reached due to the loop structure
        raise AgentError("Unexpected exit from retry loop")

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable and worth retrying."""
        recoverable_types = [
            ConnectionError,
            asyncio.TimeoutError,
            # Add other transient error types as needed
        ]

        for error_type in recoverable_types:
            if isinstance(error, error_type):
                return True

        # Check for network-related errors in string
        error_str = str(error).lower()
        network_indicators = ["connection", "timeout", "network", "temporary"]

        return any(indicator in error_str for indicator in network_indicators)

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

    async def _execute_script_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute script generation and execution pipeline.

        Args:
            task: TaskDefinition for script operation
            data: DataFrame for analysis

        Returns:
            Script execution results with metadata
        """
        try:
            # Generate secure script
            from pathlib import Path
            template_dir = Path(__file__).parent.parent.parent / 'templates'
            generator = ScriptGenerator(template_dir)
            script = generator.generate(task)

            # Execute in sandbox
            executor = ScriptExecutor()
            config = SandboxConfig(
                timeout=30,
                memory_limit=2048,
                allow_network=False
            )

            # Prepare input data
            input_data = {
                'dataframe': data.to_dict(),
                'parameters': task.parameters
            }

            # Execute script
            result = await executor.run_sandboxed(script, input_data, config)

            return {
                'success': result.success,
                'output': result.output,
                'script_hash': result.script_hash,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'metadata': {
                    'operation': task.operation,
                    'template_used': f"{task.operation}.j2",
                    'security_validated': True,
                    'resource_limits': config.__dict__
                }
            }

        except Exception as e:
            # Re-raise with context
            if isinstance(e, (ScriptGenerationError, ExecutionError, ValidationError)):
                raise
            else:
                raise AgentError(f"Script execution pipeline failed: {str(e)}")

    async def _execute_ml_train_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute ML model training task.

        Args:
            task: TaskDefinition for ML training
            data: Training data DataFrame

        Returns:
            Training results with model_id and metrics
        """
        try:
            # Extract parameters
            params = task.parameters
            model_type = params.get('model_type', 'linear')
            target_column = params.get('target_column')
            feature_columns = params.get('feature_columns', [])
            task_type = params.get('task_type', 'regression')
            user_id = params.get('user_id') or task.user_id  # Use task.user_id as fallback
            hyperparameters = params.get('hyperparameters', {})
            preprocessing_config = params.get('preprocessing', {})

            # Validate required parameters
            if not target_column:
                raise ValidationError(
                    "target_column is required for ML training",
                    field="target_column"
                )

            if not feature_columns:
                # Use all columns except target as features
                feature_columns = [col for col in data.columns if col != target_column]

            # Train model using ML Engine
            result = self.ml_engine.train_model(
                data=data,
                task_type=task_type,
                model_type=model_type,
                target_column=target_column,
                feature_columns=feature_columns,
                user_id=user_id,
                hyperparameters=hyperparameters,
                preprocessing_config=preprocessing_config
            )

            return {
                'success': True,
                'model_id': result['model_id'],
                'metrics': result['metrics'],
                'training_time': result['training_time'],
                'model_info': result['model_info'],
                'task_type': 'ml_train'
            }

        except Exception as e:
            # Re-raise with context
            if isinstance(e, (ValidationError, DataError)):
                raise
            else:
                raise AgentError(f"ML training failed: {str(e)}")

    async def _execute_ml_score_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute ML model prediction/scoring task.

        Args:
            task: TaskDefinition for ML prediction
            data: Input data DataFrame for prediction

        Returns:
            Prediction results with predictions and probabilities
        """
        try:
            # Extract parameters
            params = task.parameters
            model_id = params.get('model_id')
            user_id = params.get('user_id', 0)

            # Validate required parameters
            if not model_id:
                raise ValidationError(
                    "model_id is required for ML prediction",
                    field="model_id"
                )

            # Make predictions using ML Engine
            result = self.ml_engine.predict(
                user_id=user_id,
                model_id=model_id,
                data=data
            )

            return {
                'success': True,
                'predictions': result['predictions'],
                'probabilities': result.get('probabilities'),
                'classes': result.get('classes'),
                'model_id': result['model_id'],
                'n_predictions': result['n_predictions'],
                'task_type': 'ml_score'
            }

        except Exception as e:
            # Re-raise with context
            if isinstance(e, (ValidationError, DataError)):
                raise
            else:
                raise AgentError(f"ML prediction failed: {str(e)}")

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