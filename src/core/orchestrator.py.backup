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

logger = get_logger(__name__)


class WorkflowState(Enum):
    """Enumeration of workflow states for multi-step processes."""
    IDLE = "idle"
    AWAITING_DATA = "awaiting_data"
    DATA_LOADED = "data_loaded"
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIGURING_MODEL = "configuring_model"
    TRAINING = "training"
    TRAINED = "trained"
    PREDICTING = "predicting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ConversationState:
    """State tracking for user conversations and workflows."""
    user_id: int
    conversation_id: str
    workflow_state: WorkflowState
    current_step: Optional[str]
    context: Dict[str, Any]
    partial_results: Dict[str, Any]
    data_sources: List[str]
    created_at: datetime
    last_activity: datetime

    def get_key(self) -> str:
        """Generate unique key for state storage."""
        return f"{self.user_id}:{self.conversation_id}"


class StateManager:
    """Manages conversation states with TTL-based cleanup."""

    def __init__(self, ttl_minutes: int = 60) -> None:
        """Initialize state manager with TTL configuration."""
        self.states: Dict[str, ConversationState] = {}
        self.ttl = ttl_minutes
        self.logger = logger

    async def get_state(self, user_id: int, conversation_id: str) -> ConversationState:
        """Get or create conversation state."""
        key = f"{user_id}:{conversation_id}"

        if key in self.states:
            state = self.states[key]
            # Update last activity
            state.last_activity = datetime.now()
            return state

        # Create new state
        new_state = ConversationState(
            user_id=user_id,
            conversation_id=conversation_id,
            workflow_state=WorkflowState.IDLE,
            current_step=None,
            context={},
            partial_results={},
            data_sources=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        self.states[key] = new_state
        self.logger.info(f"Created new conversation state for user {user_id}")
        return new_state

    async def save_state(self, state: ConversationState) -> None:
        """Save conversation state."""
        key = state.get_key()
        state.last_activity = datetime.now()
        self.states[key] = state
        self.logger.debug(f"Saved state for user {state.user_id}, workflow: {state.workflow_state.value}")

    async def clear_state(self, user_id: int, conversation_id: str) -> None:
        """Clear specific conversation state."""
        key = f"{user_id}:{conversation_id}"
        if key in self.states:
            del self.states[key]
            self.logger.info(f"Cleared state for user {user_id}")

    async def cleanup_expired(self) -> None:
        """Remove expired conversation states."""
        cutoff_time = datetime.now() - timedelta(minutes=self.ttl)
        expired_keys = [
            key for key, state in self.states.items()
            if state.last_activity < cutoff_time
        ]

        for key in expired_keys:
            del self.states[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired conversation states")


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
    """Manages multi-step workflows and state transitions."""

    def __init__(self, state_manager: StateManager) -> None:
        """Initialize workflow engine."""
        self.state_manager = state_manager
        self.workflow_templates = self._init_templates()
        self.logger = logger

    def _init_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize workflow templates."""
        return {
            "ml_training": {
                "steps": [
                    {"state": WorkflowState.AWAITING_DATA, "prompt": "Please upload your training data"},
                    {"state": WorkflowState.SELECTING_TARGET, "prompt": "Select target variable from: {columns}"},
                    {"state": WorkflowState.SELECTING_FEATURES, "prompt": "Select feature columns from: {features}"},
                    {"state": WorkflowState.CONFIGURING_MODEL, "prompt": "Choose model type: neural_network, random_forest, linear_regression"},
                    {"state": WorkflowState.TRAINING, "prompt": "Training model... Please wait"},
                    {"state": WorkflowState.COMPLETED, "prompt": "Training complete! Results: {results}"}
                ],
                "transitions": {
                    WorkflowState.IDLE: [WorkflowState.AWAITING_DATA],
                    WorkflowState.AWAITING_DATA: [WorkflowState.DATA_LOADED],
                    WorkflowState.DATA_LOADED: [WorkflowState.SELECTING_TARGET],
                    WorkflowState.SELECTING_TARGET: [WorkflowState.SELECTING_FEATURES],
                    WorkflowState.SELECTING_FEATURES: [WorkflowState.CONFIGURING_MODEL],
                    WorkflowState.CONFIGURING_MODEL: [WorkflowState.TRAINING],
                    WorkflowState.TRAINING: [WorkflowState.TRAINED],
                    WorkflowState.TRAINED: [WorkflowState.PREDICTING, WorkflowState.COMPLETED],
                    WorkflowState.PREDICTING: [WorkflowState.COMPLETED]
                }
            },
            "stats_analysis": {
                "steps": [
                    {"state": WorkflowState.AWAITING_DATA, "prompt": "Please upload your data for analysis"},
                    {"state": WorkflowState.DATA_LOADED, "prompt": "Data loaded. What analysis would you like?"},
                    {"state": WorkflowState.COMPLETED, "prompt": "Analysis complete: {results}"}
                ],
                "transitions": {
                    WorkflowState.IDLE: [WorkflowState.AWAITING_DATA],
                    WorkflowState.AWAITING_DATA: [WorkflowState.DATA_LOADED],
                    WorkflowState.DATA_LOADED: [WorkflowState.COMPLETED]
                }
            }
        }

    async def start_workflow(
        self,
        workflow_type: str,
        state: ConversationState
    ) -> Dict[str, Any]:
        """Start a new workflow."""
        if workflow_type not in self.workflow_templates:
            raise ValidationError(f"Unknown workflow type: {workflow_type}")

        template = self.workflow_templates[workflow_type]
        first_step = template["steps"][0]

        state.workflow_state = first_step["state"]
        state.current_step = workflow_type
        state.context["workflow_type"] = workflow_type
        state.context["step_index"] = 0

        await self.state_manager.save_state(state)

        return {
            "message": first_step["prompt"],
            "workflow_state": state.workflow_state.value,
            "next_steps": ["Upload data file"]
        }

    async def advance_workflow(
        self,
        state: ConversationState,
        user_input: str
    ) -> Dict[str, Any]:
        """Advance workflow to next step."""
        workflow_type = state.context.get("workflow_type")
        if not workflow_type or workflow_type not in self.workflow_templates:
            raise ValidationError("Invalid workflow state")

        template = self.workflow_templates[workflow_type]
        current_index = state.context.get("step_index", 0)

        # Validate transition
        if not await self.validate_transition(state.workflow_state, user_input):
            return {
                "error": "Invalid input for current step",
                "current_state": state.workflow_state.value,
                "expected_input": self._get_expected_input(state)
            }

        # Move to next step
        next_index = current_index + 1
        if next_index < len(template["steps"]):
            next_step = template["steps"][next_index]
            state.workflow_state = next_step["state"]
            state.context["step_index"] = next_index

            await self.state_manager.save_state(state)

            return {
                "message": next_step["prompt"],
                "workflow_state": state.workflow_state.value,
                "progress": f"{next_index + 1}/{len(template['steps'])}"
            }
        else:
            # Workflow complete
            state.workflow_state = WorkflowState.COMPLETED
            await self.state_manager.save_state(state)

            return {
                "message": "Workflow completed successfully",
                "workflow_state": state.workflow_state.value,
                "results": state.partial_results
            }

    async def validate_transition(
        self,
        from_state: WorkflowState,
        user_input: str
    ) -> bool:
        """Validate if transition is allowed."""
        # Basic validation - can be enhanced with specific rules
        if from_state == WorkflowState.AWAITING_DATA:
            return "upload" in user_input.lower() or "file" in user_input.lower()
        elif from_state == WorkflowState.SELECTING_TARGET:
            return len(user_input.strip()) > 0  # Non-empty input
        elif from_state == WorkflowState.SELECTING_FEATURES:
            return len(user_input.strip()) > 0  # Non-empty input
        elif from_state == WorkflowState.CONFIGURING_MODEL:
            valid_models = ["neural_network", "random_forest", "linear_regression"]
            return any(model in user_input.lower() for model in valid_models)

        return True  # Allow other transitions

    async def get_next_prompt(self, state: ConversationState) -> str:
        """Get next prompt for current workflow state."""
        workflow_type = state.context.get("workflow_type")
        if not workflow_type:
            return "What would you like to do?"

        template = self.workflow_templates.get(workflow_type, {})
        steps = template.get("steps", [])
        step_index = state.context.get("step_index", 0)

        if step_index < len(steps):
            return steps[step_index]["prompt"]

        return "Workflow completed"

    def _get_expected_input(self, state: ConversationState) -> str:
        """Get expected input description for current state."""
        if state.workflow_state == WorkflowState.AWAITING_DATA:
            return "Please upload a data file"
        elif state.workflow_state == WorkflowState.SELECTING_TARGET:
            return "Specify target column name"
        elif state.workflow_state == WorkflowState.SELECTING_FEATURES:
            return "List feature column names (comma-separated)"
        elif state.workflow_state == WorkflowState.CONFIGURING_MODEL:
            return "Choose: neural_network, random_forest, or linear_regression"

        return "Continue with workflow"

    async def cancel_workflow(self, state: ConversationState) -> None:
        """Cancel current workflow and reset to idle."""
        state.workflow_state = WorkflowState.IDLE
        state.current_step = None
        state.context.clear()
        state.partial_results.clear()

        await self.state_manager.save_state(state)
        self.logger.info(f"Cancelled workflow for user {state.user_id}")


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
        state_ttl_minutes: int = 60,
        ml_config: Optional[MLEngineConfig] = None
    ) -> None:
        """Initialize the enhanced task orchestrator."""
        # Core engines
        self.stats_engine = StatsEngine()
        self.ml_engine = MLEngine(ml_config or MLEngineConfig.get_default())
        self.enable_logging = enable_logging
        self.logger = logger

        # Enhanced components
        self.state_manager = StateManager(ttl_minutes=state_ttl_minutes)
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
            # Get or create conversation state
            state = await self.state_manager.get_state(task.user_id, task.conversation_id)

            if progress_callback:
                progress_callback("Initializing task execution")

            # Check if this is part of a multi-step workflow
            if state.workflow_state != WorkflowState.IDLE:
                if self.enable_logging:
                    self.logger.info(f"Continuing workflow {state.workflow_state} for user {task.user_id}")

                # Advance workflow
                workflow_result = await self.workflow_engine.advance_workflow(state, task.raw_input if hasattr(task, 'raw_input') else "")

                if workflow_result.get("completed", False):
                    # Workflow completed, proceed with normal execution
                    pass
                else:
                    # Return workflow continuation prompt
                    return {
                        "success": True,
                        "workflow_state": state.workflow_state.value,
                        "next_step": workflow_result.get("next_prompt", "Please continue..."),
                        "workflow_active": True,
                        "metadata": {
                            "execution_time": time.time() - start_time,
                            "user_id": task.user_id,
                            "conversation_id": task.conversation_id
                        }
                    }

            # Handle data loading if needed
            if data is None and state.data_sources:
                if progress_callback:
                    progress_callback("Loading cached data")

                try:
                    data, metadata = await self.data_manager.get_data(state.data_sources[-1])
                    if self.enable_logging:
                        self.logger.info(f"Loaded cached data: {data.shape}")
                except Exception as e:
                    self.logger.warning(f"Failed to load cached data: {e}")
                    return {
                        "success": False,
                        "error": "No data available. Please upload data first.",
                        "error_code": "NO_DATA",
                        "task_type": task.task_type,
                        "metadata": {"execution_time": time.time() - start_time, "user_id": task.user_id}
                    }

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

            # Update state with results
            state.partial_results[task.operation] = result
            state.last_activity = datetime.now()
            await self.state_manager.save_state(state)

            # Add workflow context to result
            result["workflow_state"] = state.workflow_state.value
            result["workflow_active"] = state.workflow_state != WorkflowState.IDLE

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
            user_id = params.get('user_id', 0)
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